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

    if getattr(_vc.load_voice_clone_frontend, "_310p_cpu_pinned", False):
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
    _patch_logits_nan_guard()
    _PATCHED = True

    import vllm

    vllm.logger.init_logger(__name__).info(
        "[310P][GLM-TTS] patch set installed: DiT math-SDPA, vocoder CPU pin, "
        "WhisperVQ CPU+fp32 pin, AR logits NaN guard"
    )
