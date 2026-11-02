# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch Covo-Audio-Chat (tencent/Covo-Audio-Chat) for the 310P NPU path.

One orchestrator class serves both stages; the dispatch layer hands every
stage engine ``model_arch == "CovoAudioForConditionalGeneration"`` and this
module applies both patch groups (each guarded by its own idempotency flag):

Stage 0 (fused_thinker_talker, LLM_AR):
  * Whisper encoder attention dispatch: force the *eager* attention
    interface (explicit matmul/softmax/matmul) on 310P.  transformers 5.x
    resolves the interface per attention layer via
    ``config._attn_implementation``; the default "sdpa" routes to
    ``F.scaled_dot_product_attention``, which the 310P torch_npu
    dispatcher rejects globally (research/012, 7th+ confirmations on this
    image).  The Qwen2 LLM part runs on vLLM's native AR engine
    (vllm-ascend backend) and is untouched.
  * dtype: the overlay (``deploy/covo_audio_310p.yaml``) pins the engine
    dtype to float16 at construction time (310P is FP16-only; the upstream
    checkpoint is bf16 storage).  The loader materializes disk-bf16 weights
    as fp16 (loader-ok already-fp16 evidence pattern, ENV-PRECHECK P1-13).

Stage 1 (code2wav, LLM_GENERATION):
  * token2wav DiT ``MultiHeadAttention`` calls ``F.scaled_dot_product_attention``
    directly -> replaced with an explicit math decomposition via a module-level
    ``F`` shim (fp32-accumulated scores, softmax in the model dtype; chunked
    when the L x S score matrix would exceed the budget).
  * ``common.alias_free_activation.kaiser_sinc_filter1d`` NPU branch moves the
    Kaiser window to the NPU while leaving ``time_indices``/``sinc_filter`` on
    the default device (CPU in the stage-1 worker), so ``window * sinc``
    crashes with a device mismatch while BigVGAN is still being built (the
    aten::kaiser_window no-NPU-kernel family, research/012; indextts
    precedent).  The filter math is pinned to CPU (the GPU/CPU lines'
    portable behavior); the registered buffers then ride the module move.
  * ``Token2WavDecoder.inference`` is wrapped to reseed the default RNG
    before the flow-matching initial noise draw (the only free randomness in
    the do_sample=False production path) so same-seed runs are
    bit-reproducible (POLICY section 1, level 3 gate).
  * dtype: the upstream deploy yaml already pins stage 1 to float32
    (BigVGAN/token2wav weights are fp32 — official deploy contract, do not
    convert); the 310P overlay keeps it as-is.

Registration is dispatch-driven (``apply_model_patches`` in this package):
both patch groups are idempotent and cheap; each stage engine core applies
them before its model is constructed (npu_model_runner.load_model hook).

Kill switch: VLLM_OMNI_310P_COVO_PATCHES=0 disables all patches above
(used for the POLICY section 1 level-2 unpatched-reference run).
"""

from __future__ import annotations

import math
import os

import torch

from vllm_omni.platforms.npu._310p import is_310p

_COVO_VOCODER_SEED = 42

# fp32 score-matrix budget in elements (4B each). The score tensor is
# (B, H, Lq, Lk) — the head/batch dims must count toward the budget. Above
# this, the query axis is chunked so the fp32 intermediate stays ~<=256MB on
# the 3GB stage-1 pool.
_FP32_SCORES_BUDGET = 1 << 26

_STAGE0_PATCHED = False
_STAGE1_PATCHED = False


def _patches_disabled() -> bool:
    return os.environ.get("VLLM_OMNI_310P_COVO_PATCHES", "1") == "0"


def _log(msg: str) -> None:
    """Scalar-only telemetry (ENV-PRECHECK P1-13: never pass tensors)."""
    try:
        from vllm.logger import init_logger

        init_logger(__name__).info(f"[310P][covo] {msg}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Math SDPA (explicit matmul / softmax / matmul decomposition)
# ---------------------------------------------------------------------------


def _math_sdpa_single(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None, scale: float
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.transpose(-2, -1))
    scores = scores * scale
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def _math_sdpa_chunked(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None, scale: float
) -> torch.Tensor:
    rows = q.shape[-2]
    head_dim = max(q.shape[0] * q.shape[1], 1)  # batch * heads
    block = max(_FP32_SCORES_BUDGET // max(k.shape[-2] * head_dim, 1), 64)
    out = torch.empty_like(q)
    for start in range(0, rows, block):
        end = min(start + block, rows)
        q_b = q[..., start:end, :]
        m_b = None
        if attn_mask is not None:
            m_b = attn_mask[..., start:end, :]
        out[..., start:end, :] = _math_sdpa_single(q_b, k, v, m_b, scale)
    return out


def math_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0
) -> torch.Tensor:
    """Portable SDPA replacement for 310P (F.sdpa dead zone, research/012)."""
    del dropout_p  # inference path only; no dropout at eval
    if q.shape[-1] == 0:
        return q
    scale = q.shape[-1] ** -0.5
    score_elems = q.shape[0] * q.shape[1] * q.shape[-2] * k.shape[-2]
    if attn_mask is not None and score_elems > _FP32_SCORES_BUDGET:
        return _math_sdpa_chunked(q, k, v, attn_mask, scale)
    return _math_sdpa_single(q, k, v, attn_mask, scale)


class _FShim:
    """Module-level stand-in for ``torch.nn.functional`` in token2wav.

    Only ``scaled_dot_product_attention`` is redirected (to the math
    decomposition); every other attribute delegates to the real ``F``.
    """

    scaled_dot_product_attention = staticmethod(math_sdpa)

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Stage 0: fused_thinker_talker
# ---------------------------------------------------------------------------


def apply_stage0_patches(model_config) -> None:
    """Force the Whisper encoder onto the eager (math) attention path."""
    global _STAGE0_PATCHED
    if _STAGE0_PATCHED or _patches_disabled():
        return
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return
    encoder_config = getattr(hf_config, "encoder_config", None)
    if encoder_config is not None:
        encoder_config._attn_implementation = "eager"
    _STAGE0_PATCHED = True
    _log("stage0: whisper encoder _attn_implementation -> eager")


# ---------------------------------------------------------------------------
# Stage 1: code2wav (token2wav)
# ---------------------------------------------------------------------------


def _patch_bigvgan_kaiser_filter_for_310p() -> None:
    """CPU-safe Kaiser-sinc filter for BigVGAN LowPassFilter construction.

    The NPU branch of ``common.alias_free_activation.kaiser_sinc_filter1d``
    moves the Kaiser window to the NPU but leaves ``time_indices`` and
    ``sinc_filter`` on the default device (CPU in the stage-1 worker), so
    ``kaiser_window * sinc_filter`` crashes with an NPU/CPU device mismatch
    while BigVGAN is still being built on CPU.  The GPU/CPU lines compute the
    whole filter on the default device, so a CPU-only computation is exactly
    the portable behavior; the registered buffer is moved with the module
    afterwards.
    """

    from vllm_omni.model_executor.models.common import alias_free_activation

    original = alias_free_activation.kaiser_sinc_filter1d
    if getattr(original, "_covo_310p_cpu_safe", False):
        return

    def _cpu_kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
        # Same math as the shared implementation, everything on CPU.
        is_even = kernel_size % 2 == 0
        half_size = kernel_size // 2
        delta_f = 4 * half_width
        attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
        if attenuation > 50.0:
            beta = 0.1102 * (attenuation - 8.7)
        elif attenuation >= 21.0:
            beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
        else:
            beta = 0.0
        kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32, device="cpu")
        if is_even:
            time_indices = torch.arange(-half_size, half_size, device="cpu") + 0.5
        else:
            time_indices = torch.arange(kernel_size, device="cpu") - half_size
        if cutoff == 0:
            return torch.zeros((1, 1, kernel_size), dtype=torch.float32)
        sinc_filter = torch.sinc(2 * cutoff * time_indices)
        normalized_filter = 2 * cutoff * kaiser_window * sinc_filter
        normalized_filter /= normalized_filter.sum()
        return normalized_filter.view(1, 1, kernel_size)

    _cpu_kaiser_sinc_filter1d._covo_310p_cpu_safe = True  # type: ignore[attr-defined]
    alias_free_activation.kaiser_sinc_filter1d = _cpu_kaiser_sinc_filter1d
    _log("stage1: BigVGAN kaiser_sinc_filter1d pinned to CPU (device mismatch avoided)")


def apply_stage1_patches(model_config) -> None:
    """Replace F.sdpa in the token2wav DiT + seed the vocoder RNG + kaiser CPU pin."""
    global _STAGE1_PATCHED
    if _STAGE1_PATCHED or _patches_disabled():
        return
    from vllm_omni.model_executor.models.covo_audio import token2wav

    # Kaiser filter math must be CPU-pinned BEFORE the first Token2WavDecoder
    # construction (BigVGAN LowPassFilter buffers are built in __init__, on
    # the default device).
    _patch_bigvgan_kaiser_filter_for_310p()

    if getattr(token2wav, "_omni_f", None) is None:
        token2wav.F = _FShim(token2wav.F)
        token2wav._omni_f = token2wav.F

    orig_inference = token2wav.Token2WavDecoder.inference
    if not getattr(orig_inference, "_covo_310p_seeded", False):

        def seeded_inference(self, data, **kwargs):
            # do_sample=False path has exactly one free randomness: the
            # flow-matching initial noise (token2wav.sample). Reseed the
            # default generator per inference call for bit-parity runs.
            torch.manual_seed(_COVO_VOCODER_SEED)
            return orig_inference(self, data, **kwargs)

        seeded_inference._covo_310p_seeded = True  # type: ignore[attr-defined]
        token2wav.Token2WavDecoder.inference = seeded_inference

    _STAGE1_PATCHED = True
    _log("stage1: token2wav math-SDPA shim + vocoder seed(42) + kaiser CPU pin installed")


def apply_covo_audio_patches(model_config) -> None:
    """Entry point from the _310p patch dispatcher (both stages).

    ``model_arch`` is "CovoAudioForConditionalGeneration" for both stage
    engines (one orchestrator class discriminates on model_stage), so both
    groups are applied per engine; each is idempotent and cheap.
    """
    if not is_310p():
        return
    apply_stage0_patches(model_config)
    apply_stage1_patches(model_config)
