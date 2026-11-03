# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P patches for the GLM-TTS two-stage pipeline (stage-0 AR + stage-1 DiT).

GLM-TTS (zai-org/GLM-TTS, MIT) topology on vLLM-Omni:

  stage 0  LLM_AR          GQA Llama 1.55B (num_attention_heads=16 /
                           num_key_value_heads=4) speech-token AR,
                           RAS sampling, dtype float16 on 310P
                           (upstream deploy default bfloat16 is rejected
                           by the 310P AclNN layer, EZ1001).
  stage 1  LLM_GENERATION  Flow-matching DiT (trans_dim 768, depth 18,
                           heads 12, dim_head 64) running its Euler ODE in
                           float32 + HiFT/Vocos2D vocoder.

Both stages run on the LLM runtime (NPUGenerationWorker / NPUARWorker),
so the engine-level 310P diffusion dispatch (commit 3959306) and the
DiffusionEngine 512x512 dummy-warmup skip (6e4e4e3b6) are NOT on this
pipeline's code path — verified against tag v0.32.0 (5917730a4b):
``init_diffusion_model_runner_runtime`` is only called from
``vllm_omni/diffusion/worker/diffusion_model_runner.py`` and
``LLM_GENERATION`` maps to ``(StageType.LLM, "generation")``.

Walls patched here (all is_310p-gated, kill-switch
``VLLM_OMNI_310P_GLM_TTS_PATCHES=0`` skips installation):

1. DiT attention -> explicit torch math.  The DiT stage decodes in
   float32 and every attention branch of ``DiTAttention`` lands on a
   dead NPU SDPA path on 310P: the masked branch and the auto-fp32
   branch call ``F.scaled_dot_product_attention`` (global F.sdpa dead
   zone, 012/015 evidence), and the default branch routes through the
   diffusion DiffusionAttention backend (same kernel).  Replaced with
   matmul + fp32-softmax + matmul, numerically equivalent (mask is an
   additive float [1,1,T,T]).
2. Vocoder (HiFT 24kHz / Vocos2D-JIT 32kHz) pinned to CPU.  Community
   910B precedent (GitCode gcw_coj3XaOd/GLM-TTS, 2026-05): HiFT's
   ``torch.istft`` maps to aclnnUnfoldGrad and fails on NPU
   ("Op UnfoldGrad has 1 uncontiguous input"); the TorchScript Vocos2D
   checkpoint is likewise not NPU-runnable.  Both wrappers are
   self-referential on device (``mel.to(self.device)``), so pinning the
   load device to CPU is sufficient; mel is D2H per chunk.  P1-11:
   HiFT already runs float32; the serve script must set
   ``OMP_NUM_THREADS>=16`` (310P SoC CPU FP16 GEMM pathology).
3. WhisperVQ speech tokenizer pinned to CPU + float32.  Same community
   precedent: VQ quantization on NPU emits corrupt token ids
   (argmin/index numeric anomaly on NPU); CPU FP32 per P1-11.
   CampPlus ONNX already falls back to CPUExecutionProvider on
   non-cuda devices.
4. AR-stage logits NaN guard.  Community 910B pit 5: NPU LLM decode
   occasionally yields inf/NaN logits (multinomial then returns out-of-
   vocab ids).  ``nan_to_num`` guard after the fp32 cast in
   ``compute_logits`` (AR stage only; the DiT stage branches first).

Stage-0 dtype (bfloat16 -> float16) is set by the deploy overlay
``vllm_omni/deploy/glm_tts_310p.yaml``, not here.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

_KILL_SWITCH_ENV = "VLLM_OMNI_310P_GLM_TTS_PATCHES"
_PATCHED = False


def _kill_switch_off() -> bool:
    return os.environ.get(_KILL_SWITCH_ENV, "1") == "0"


def _math_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit torch math SDPA for [B, T, H, D] inputs.

    310P NPU has no usable fused SDPA kernel for the fp32 DiT path
    (F.sdpa global dead zone); matmul + fp32 softmax + matmul is the
    only validated path (012/015 precedent, maxdiff vs CPU fp32 <= 1.8e-03
    at fp16 precision).
    """
    # [B, T, H, D] -> [B, H, T, D]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask.to(scores.device, dtype=torch.float32)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype).permute(0, 2, 1, 3)  # -> [B, T, H, D]


def _patch_dit_attention() -> None:
    """Replace DiTAttention attention core with explicit math (310P)."""
    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit as _dit_mod

    if getattr(_dit_mod.DiTAttention, "_310p_math_sdpa", False):
        return

    orig_forward = _dit_mod.DiTAttention.forward

    def _forward_310p(self, x, padding_mask=None, rope=None, attn_mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = _dit_mod.apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = _dit_mod.apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        query = query.view(batch_size, seq_len, self.heads, self.dim_head)
        key = key.view(batch_size, seq_len, self.heads, self.dim_head)
        value = value.view(batch_size, seq_len, self.heads, self.dim_head)

        # All branches (masked / auto-fp32 / diffusion-backend) route
        # through the explicit math path on 310P.
        out = _math_sdpa(query, key, value, self.scale, attn_mask)
        # reshape (not view): the permuted [B, T, H, D] output of the math
        # kernel is not contiguous, and the upstream .view would reject it
        # (same class of bug upstream branch (a) carries on every backend).
        out = out.reshape(batch_size, seq_len, self.inner_dim)
        out = out.to(query.dtype)
        out = self.to_out(out)

        if padding_mask is not None:
            from einops import rearrange

            padding_mask = rearrange(padding_mask, "b 1 n -> b n 1").bool()
            out = out.masked_fill(~padding_mask, 0.0)
        return out

    _forward_310p._310p_math_sdpa = True  # type: ignore[attr-defined]
    _dit_mod.DiTAttention.forward = _forward_310p  # type: ignore[assignment]
    _dit_mod.DiTAttention._310p_math_sdpa = True  # type: ignore[attr-defined]
    # Keep the original for the kill-switch / A-B comparison.
    _dit_mod.DiTAttention._310p_orig_forward = orig_forward  # type: ignore[attr-defined]


def _patch_vocoder_cpu_pin() -> None:
    """Pin the stage-1 vocoder load device to CPU on 310P."""
    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit_wrapper as _wrp

    cls = _wrp.GLMTTSDiTForGeneration
    if getattr(cls, "_310p_vocoder_cpu_pinned", False):
        return

    orig = cls._ensure_vocoder_loaded

    def _ensure_vocoder_loaded_310p(self):
        if self._vocoder is not None:
            return
        device = torch.device("cpu")
        self._vocoder, self._vocoder_sample_rate = _wrp.load_vocoder(
            self.model_dir, device, self.sample_rate
        )
        if self._vocoder is None:
            raise RuntimeError(
                "GLM-TTS stage 1 could not load a vocoder. "
                f"Expected either {self._hift_weights_path} or {self._vocos_jit_path}."
            )

    _ensure_vocoder_loaded_310p._310p_vocoder_cpu_pinned = True  # type: ignore[attr-defined]
    cls._ensure_vocoder_loaded = _ensure_vocoder_loaded_310p  # type: ignore[assignment]
    cls._310p_vocoder_cpu_pinned = True  # type: ignore[attr-defined]
    cls._310p_orig_ensure_vocoder = orig  # type: ignore[attr-defined]


def _patch_speech_tokenizer_cpu_pin() -> None:
    """Pin the WhisperVQ speech tokenizer + force float32 on 310P."""
    from vllm_omni.model_executor.models.glm_tts import voice_clone as _vc
    import vllm_omni.model_executor.models.glm_tts.glm_tts as _gm

    if getattr(_vc.load_voice_clone_frontend, "_310p_cpu_pinned", False):
        # Already wrapped — still make sure the glm_tts namespace (import
        # binding) points at the wrapper.
        _gm.load_voice_clone_frontend = _vc.load_voice_clone_frontend  # type: ignore[assignment]
        return

    orig = _vc.load_voice_clone_frontend

    def _load_voice_clone_frontend_310p(
        model_root,
        model_device,
        *,
        speech_tokenizer_cache=None,
        campplus_cache=None,
        campplus_path=None,
    ):
        result = orig(
            model_root,
            torch.device("cpu"),
            speech_tokenizer_cache=speech_tokenizer_cache,
            campplus_cache=campplus_cache,
            campplus_path=campplus_path,
        )
        speech_tokenizer, campplus_session = result
        if speech_tokenizer is not None:
            model, _fe = speech_tokenizer
            tensors = list(model.parameters()) + list(model.buffers())
            if any(t.device.type != "cpu" or t.dtype == torch.float16 for t in tensors):
                # P1-11: CPU preprocessing component runs float32 (310P SoC
                # CPU FP16 GEMM pathology), not on any NPU.
                model.to(device=torch.device("cpu"), dtype=torch.float32)
        return speech_tokenizer, campplus_session

    _load_voice_clone_frontend_310p._310p_cpu_pinned = True  # type: ignore[attr-defined]
    _vc.load_voice_clone_frontend = _load_voice_clone_frontend_310p  # type: ignore[assignment]
    _vc.load_voice_clone_frontend._310p_orig = orig  # type: ignore[attr-defined]
    # glm_tts.py binds the name at import time (``from .voice_clone import
    # load_voice_clone_frontend``) — the mm-processor path (main process)
    # resolves through that namespace, so both must be rebound.
    import vllm_omni.model_executor.models.glm_tts.glm_tts as _gm

    if not getattr(
        getattr(_gm, "load_voice_clone_frontend", None), "_310p_cpu_pinned", False
    ):
        _gm.load_voice_clone_frontend = _load_voice_clone_frontend_310p  # type: ignore[assignment]


def _patch_logits_nan_guard() -> None:
    """Guard AR-stage logits against NPU inf/NaN (community 910B pit 5)."""
    from vllm_omni.model_executor.models.glm_tts import glm_tts as _m

    cls = _m.GLMTTSForConditionalGeneration
    if getattr(cls, "_310p_nan_guarded", False):
        return

    orig = cls.compute_logits

    def _compute_logits_310p(self, hidden_states, sampling_metadata=None):
        out = orig(self, hidden_states, sampling_metadata)
        if self.model_stage != "glm_tts_dit" and isinstance(out, torch.Tensor):
            out = torch.nan_to_num(out, nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)
        return out

    _compute_logits_310p._310p_nan_guarded = True  # type: ignore[attr-defined]
    cls.compute_logits = _compute_logits_310p  # type: ignore[assignment]
    cls._310p_nan_guarded = True  # type: ignore[attr-defined]
    cls._310p_orig_compute_logits = orig  # type: ignore[attr-defined]


def _patch_torch_nan_to_num_device_safe() -> None:
    """Make ``torch.nan_to_num`` AclNN-safe on NPU tensors.

    310P does not implement ``aclnnNanToNum`` (EZ1001 ``NanToNum is not
    supported on this device``).  Upstream GLM-TTS decode paths call
    ``torch.nan_to_num`` on-device -- RAS nucleus sampling
    (``nucleus_ras_sampling.py``) and the AR logits guard below.

    The replacement composes ``isfinite``/``where``/sign comparisons, all
    AclNN-supported on 310P3 (probe-verified).  Note: ``== +/-inf`` is
    *broken* on 310P AclNN (always False), so inf classification is done
    via the IEEE sign of non-finite values instead (NaN compares False to
    both 0 tests, +inf is > 0, -inf is < 0).  CPU tensors keep the
    original behavior.
    """
    orig = torch.nan_to_num
    if getattr(orig, "_310p_nan_to_num_wrapped", False):
        return

    def _safe(
        input: Any,
        nan: float = 0.0,
        posinf: float | None = None,
        neginf: float | None = None,
        *,
        copy: bool = True,
        inplace: bool = False,
    ) -> Any:
        if isinstance(input, torch.Tensor) and input.device.type == "npu" and not inplace:
            finite = torch.isfinite(input)
            if posinf is None:
                posinf = float(torch.finfo(input.dtype).max)
            if neginf is None:
                neginf = float(torch.finfo(input.dtype).min)
            nan_t = torch.tensor(float(nan), dtype=input.dtype, device=input.device)
            pos_t = torch.tensor(float(posinf), dtype=input.dtype, device=input.device)
            neg_t = torch.tensor(float(neginf), dtype=input.dtype, device=input.device)
            return torch.where(
                finite,
                input,
                torch.where(input > 0, pos_t, torch.where(input < 0, neg_t, nan_t)),
            )
        # Default copy/inplace semantics; forwarded kwargs only where the
        # installed torch build accepts them (older torch has no `copy`).
        kwargs = dict(nan=nan, posinf=posinf, neginf=neginf)
        if inplace:
            kwargs["inplace"] = True
        try:
            return orig(input, **kwargs)
        except TypeError:
            return orig(input, nan=nan, posinf=posinf, neginf=neginf)

    _safe._310p_nan_to_num_wrapped = True  # type: ignore[attr-defined]
    torch.nan_to_num = _safe  # type: ignore[assignment]


def _patch_mel_extraction_cpu_pin() -> None:
    """Run the mm-processor's mel extraction on CPU on 310P.

    ``extract_prompt_feat`` feeds ``current_omni_platform.get_torch_device()``
    (NPU) to the STFT-based mel path; ``torch.stft`` produces complex
    tensors, which 310P AclNN cannot represent (ERR01007 ``Current settings
    do not support Complex dtype``).  The mel feature is a preprocessing
    artifact — the worker moves it to its own device at handoff.
    """
    from vllm_omni.model_executor.models.glm_tts import voice_clone as _vc
    import vllm_omni.model_executor.models.glm_tts.glm_tts as _gm

    if getattr(_vc.extract_prompt_feat, "_310p_cpu_pinned", False):
        _gm.extract_prompt_feat = _vc.extract_prompt_feat  # type: ignore[assignment]
        return

    orig = _vc.extract_prompt_feat

    def _extract_prompt_feat_310p(ref_audio_wav, ref_audio_sr, model_device, *args, **kwargs):
        return orig(ref_audio_wav, ref_audio_sr, torch.device("cpu"), *args, **kwargs)

    _extract_prompt_feat_310p._310p_cpu_pinned = True  # type: ignore[attr-defined]
    _vc.extract_prompt_feat = _extract_prompt_feat_310p  # type: ignore[assignment]
    _gm.extract_prompt_feat = _extract_prompt_feat_310p  # type: ignore[assignment]


_PROCESS_SHIM_PATCHED = False


def apply_glm_tts_process_shims() -> None:
    """Install the process-wide GLM-TTS shims (310P only, every process).

    The GLM-TTS mm-processor loads the WhisperVQ voice-clone frontend in
    the orchestrator/main process — a place the worker-gated
    ``apply_model_patches`` dispatch never reaches — where upstream pins
    it to ``current_omni_platform.get_torch_device()`` (NPU).  The frontend
    CPU pin and the device-safe ``torch.nan_to_num`` shim are
    model-independent, so install them from the platform-level hook
    (``NPUOmniPlatform.__init__`` runs in every vllm_omni process).
    """
    global _PROCESS_SHIM_PATCHED
    if _PROCESS_SHIM_PATCHED:
        return

    from vllm_omni.platforms.npu._310p import is_310p

    if not is_310p():
        return
    if _kill_switch_off():
        import vllm

        vllm.logger.init_logger(__name__).info(
            "[310P][GLM-TTS] kill switch %s=0: process shims skipped",
            _KILL_SWITCH_ENV,
        )
        _PROCESS_SHIM_PATCHED = True
        return

    _patch_torch_nan_to_num_device_safe()
    _patch_speech_tokenizer_cpu_pin()
    _patch_mel_extraction_cpu_pin()
    _PROCESS_SHIM_PATCHED = True


def apply_glm_tts_310p_patches() -> None:
    """Install the GLM-TTS 310P patch set (idempotent).

    Called from ``_310p/patch/__init__.py::apply_model_patches`` for both
    stage workers (model_arch GLMTTSForConditionalGeneration).  No-op off
    310P and under the kill switch.
    """
    global _PATCHED
    if _PATCHED:
        return

    from vllm_omni.platforms.npu._310p import is_310p

    if not is_310p():
        return
    if _kill_switch_off():
        import vllm

        vllm.logger.init_logger(__name__).info(
            "[310P][GLM-TTS] kill switch %s=0: patch installation skipped",
            _KILL_SWITCH_ENV,
        )
        _PATCHED = True
        return

    _patch_dit_attention()
    _patch_vocoder_cpu_pin()
    _patch_speech_tokenizer_cpu_pin()
    _patch_torch_nan_to_num_device_safe()
    _patch_logits_nan_guard()
    _PATCHED = True

    import vllm

    vllm.logger.init_logger(__name__).info(
        "[310P][GLM-TTS] patch set installed: DiT math-SDPA, vocoder CPU pin, "
        "WhisperVQ CPU+fp32 pin, NPU-safe torch.nan_to_num, AR logits NaN guard"
    )
