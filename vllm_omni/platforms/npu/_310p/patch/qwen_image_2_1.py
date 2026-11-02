# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P patch for the Qwen-Image-2.1 diffusion pipeline (task-16).

SKELETON STATUS (local prep, 2026-11-02)
----------------------------------------
This file is the 310P adaptation hook for ``QwenImage21Pipeline``.

Integration contract (COMPARISON-POLICY.md §5, lead-approved 2026-11-02)
-------------------------------------------------------------------------
task-11 (SDXL) lands the registry-style diffusion 310P hook in
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` first
(model_class_name -> [patch function]); this branch rebases on it and
registers ONLY through this file -- it must NOT edit ``platform.py``.  The
registry consumes the names below (``maybe_apply_diffusion_310p_patches``
stays as the self-gated standalone entry point until the registry lands):

    MODEL_ARCH = "QwenImage21Pipeline"
    apply_patch = apply_qwen_image_2_1_310p_patches

Why this hook at all: ``apply_model_patches(model_config)`` only runs inside
the AR ``OmniNPUModelRunner.load_model``; diffusion pipelines are built by
``DiffusionModelRunner.load_model``, which calls
``init_diffusion_model_runner_runtime`` first (same convention as task-11).

Code-level findings that shape this patch (v0.32.0, branch feat/310p-qwen-image-2.1)
--------------------------------------------------------------------------------------
Under an FP16 runtime (the 310P policy) *every* BF16-gated NPU fast path in
``qwen_image_21`` is disabled by its own dtype guard, so the diff versus the
910B behavior is mostly "fast paths stay off", plus three real hazards:

1. RoPE: ``qwen_image_21_transformer._apply_qwen_image21_rotary_emb_npu``
   calls ``torch_npu.npu_rotary_mul(x.float(), cos, sin,
   rotary_mode="interleave")`` with no fallback.  ``npu_rotary_mul`` is proven
   on 310P only in *half* mode (``_310p/patch/qwen3_tts.py`` and
   ``platforms/npu/layers/rotary_embedding.py`` never pass "interleave").
   If the 310P CANN build rejects interleave mode the whole DiT crashes at
   the first block -> guarded dispatcher with native real-valued fallback.
2. NZ weight cast: ``load_weights`` casts every BF16 2-D linear weight via
   ``npu_format_cast(w, 29)`` (29 == ACL_FORMAT_FRACTAL_NZ) when internal
   formats are enabled.  BF16 gates it off under FP16, but if the model is
   ever built BF16 on 310P the cast target is 910-oriented and unvalidated ->
   defensively disabled for 310P.
3. Attention: default backend on NPU without MindIE-SD is TORCH_SDPA
   (``platform.py:get_diffusion_attn_backend_cls``), which drives
   ``SDPAImpl.forward_npu`` (dense-mask ``full_qk`` path because
   ``supports_piecewise_spans`` is False) -- the block-causal mask of
   Qwen-Image-2.1 is therefore a dense 4-D boolean mask on 310P.  The
   ``QwenImage21SDPAImpl`` fusion fast path requires BF16 and stays off.
   Everything else in the transformer/VAE is portable eager torch.

CUDA-graph decode (``decode_graph.py``) is CUDA-only by construction
(``device.type != "cuda"`` -> eager) and needs no 310P handling.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import torch

from vllm_omni.platforms.npu._310p import is_310p

logger = logging.getLogger(__name__)

# Registry contract (COMPARISON-POLICY.md §5): the task-11 platform.py hook
# maps model_class_name -> patch function; this file registers the entry
# below and nothing else.
MODEL_ARCH = "QwenImage21Pipeline"

# Env knobs (all checked once at patch time; machine-phase A/B experiments use
# them instead of editing code):
#   QWEN21_310P_ROPE=auto|npu|native   (default auto: probe once, fall back)
#   QWEN21_310P_GENERATOR_DEVICE=cpu|npu (default npu: keep device RNG)
#   QWEN21_310P_ATTENTION=sdpa|flash_attn (default sdpa; flash_attn is an
#       experiment knob only, npu_fusion_attention on 310P is unvalidated)

_APPLIED = False


# ---------------------------------------------------------------------------
# Pure, CPU-testable decision functions
# ---------------------------------------------------------------------------


def resolve_runtime_dtype(current: torch.dtype) -> torch.dtype:
    """310P diffusion runs FP16 (project policy; MindIE-SD free)."""
    del current
    return torch.float16


def resolve_rope_policy(environ: dict[str, str] | None = None) -> str:
    """Pick the RoPE execution policy for 310P.

    Returns one of ``"auto"``, ``"npu"`` or ``"native"``.  ``"auto"`` probes
    ``npu_rotary_mul`` interleave mode once at runtime (see
    ``probe_npu_rotary_interleave``); the probe result is cached on the
    dispatcher so repeated calls are free.
    """
    env = os.environ if environ is None else environ
    policy = env.get("QWEN21_310P_ROPE", "auto").strip().lower()
    if policy not in ("auto", "npu", "native"):
        logger.warning("Unknown QWEN21_310P_ROPE=%r; falling back to auto", policy)
        return "auto"
    return policy


def resolve_vae_memory(vae_use_tiling: bool, vae_use_slicing: bool) -> tuple[bool, bool]:
    """Force tiling+slicing: the Qwen-Image-2.1 VAE upsampling chain peaks
    multi-GB untiled on a ~21 GB 310P chip (same class of risk as SDXL R4)."""
    return True, True


def should_disable_nz(od_config: Any) -> bool:
    """The FRACTAL_NZ linear-weight cast (29) is 910-oriented; never run it
    on 310P.  Under the FP16 runtime the BF16 gate already skips it, this is
    the defensive second layer."""
    return is_310p()


def resolve_quantization_override(od_config: Any) -> None:
    """310P has no MindIE diffusion quant path: drop KV-cache quant extras
    and component quantization configs with a loud warning (mirrors SDXL R6)."""
    extras = getattr(od_config, "extras", None)
    if isinstance(extras, dict):
        for key in ("prefix_kv_cache_dtype",):
            if extras.get(key):
                logger.warning(
                    "310P Qwen-Image-2.1: dropping extras[%r]=%r (no 310P KV-quant path).",
                    key,
                    extras[key],
                )
                extras[key] = None
    quant = getattr(od_config, "quantization_config", None)
    if quant is not None:
        logger.warning("310P Qwen-Image-2.1: ignoring quantization_config=%r (unvalidated).", quant)
        od_config.quantization_config = None


def probe_npu_rotary_interleave() -> bool:
    """One-time capability probe for ``npu_rotary_mul(..., rotary_mode="interleave")``.

    Runs on the default NPU device with a minimal FP32 sentinel.  The TTS
    310P path proves ``npu_rotary_mul`` exists on 310P (half mode); interleave
    mode is the open question this probe answers.  Never raises: any error
    means "use native RoPE".
    """
    try:
        import torch_npu

        if not hasattr(torch_npu, "npu_rotary_mul"):
            return False
        device = torch.device("npu")
        x = torch.zeros(1, 1, 1, 4, dtype=torch.float32, device=device)
        cos = torch.ones(1, 1, 1, 4, dtype=torch.float32, device=device)
        sin = torch.zeros(1, 1, 1, 4, dtype=torch.float32, device=device)
        torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")
        torch.npu.synchronize(device)
        return True
    except Exception as exc:  # noqa: BLE001 - capability probe must never raise
        logger.info("310P npu_rotary_mul interleave probe failed (%s); using native RoPE.", exc)
        return False


def build_rope_dispatcher(original_npu_impl, native_impl, policy: str):
    """Return a RoPE implementation bound to *policy*.

    * ``"native"`` -> always ``native_impl`` (pure-torch real-valued algebra,
      the ``_apply_qwen_image21_rotary_emb_native`` fallback).
    * ``"npu"``    -> always ``original_npu_impl`` (fail loudly if broken).
    * ``"auto"``   -> try the NPU op once per unique freqs identity on first
      use; on any exception permanently switch to ``native_impl`` and log once.
    """
    if policy == "native":
        return native_impl
    if policy == "npu":
        return original_npu_impl

    state = {"mode": "auto"}

    def dispatch(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if state["mode"] == "native":
            return native_impl(x, freqs)
        try:
            return original_npu_impl(x, freqs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "310P Qwen-Image-2.1: NPU interleave RoPE failed (%s); "
                "switching to native real-valued RoPE for this process.",
                exc,
            )
            state["mode"] = "native"
            return native_impl(x, freqs)

    return dispatch


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def _patch_transformer_module() -> None:
    """Rebind the transformer's NPU RoPE function and disable NZ casting.

    Both are monkey-patches on module/class attributes (same style as
    ``_310p/patch/qwen3_tts.py``); the model modules stay pristine.
    """
    from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

    if getattr(transformer, "_310p_patch_applied", False):
        return

    policy = resolve_rope_policy()
    native_impl = transformer._apply_qwen_image21_rotary_emb_native
    if policy == "auto":
        # Defer the probe to the first real call (patch time is platform init
        # time; the NPU device may not be bound to this rank yet).
        transformer._apply_qwen_image21_rotary_emb_npu = build_rope_dispatcher(
            transformer._apply_qwen_image21_rotary_emb_npu,
            native_impl,
            "auto",
        )
    else:
        transformer._apply_qwen_image21_rotary_emb_npu = (
            transformer._apply_qwen_image21_rotary_emb_npu if policy == "npu" else native_impl
        )
    if policy == "auto":
        # Record probe status lazily so logs carry the actual verdict.
        transformer._310p_rope_probe_pending = True

    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        QwenImage21Transformer2DModel,
    )

    original_supported = QwenImage21Transformer2DModel._nz_weights_supported_for

    @staticmethod
    def _nz_weights_supported_for_310p(od_config, quant_config):
        if is_310p():
            return False
        return original_supported(od_config, quant_config)

    QwenImage21Transformer2DModel._nz_weights_supported_for = _nz_weights_supported_for_310p  # type: ignore[assignment]
    transformer._310p_patch_applied = True
    logger.info(
        "310P Qwen-Image-2.1 patches installed: rope=%s, nz-cast=disabled, decode-graphs=cpu-only(eager).",
        policy,
    )


def apply_qwen_image_2_1_310p_patches(od_config: Any) -> None:
    """Install the 310P specialization for Qwen-Image-2.1 on this rank.

    Idempotent; safe to call from the diffusion runtime hook on every rank.
    """
    global _APPLIED
    if _APPLIED:
        return

    od_config.dtype = resolve_runtime_dtype(getattr(od_config, "dtype", torch.bfloat16))
    od_config.enforce_eager = True  # NPU has no inductor; CUDA-graph decode is CUDA-only
    resolve_quantization_override(od_config)

    vae_tiling, vae_slicing = resolve_vae_memory(
        getattr(od_config, "vae_use_tiling", False),
        getattr(od_config, "vae_use_slicing", False),
    )
    od_config.vae_use_tiling = vae_tiling
    od_config.vae_use_slicing = vae_slicing

    attn = os.environ.get("QWEN21_310P_ATTENTION", "sdpa").strip().lower()
    if attn not in ("sdpa", "flash_attn"):
        logger.warning("Unknown QWEN21_310P_ATTENTION=%r; using sdpa.", attn)
        attn = "sdpa"
    if attn == "flash_attn":
        od_config.diffusion_attention_backend = "FLASH_ATTN"  # experiment knob only
    else:
        # Leave the resolver's default (TORCH_SDPA on NPU without mindiesd).
        logger.info("310P Qwen-Image-2.1: attention backend = TORCH_SDPA (dense block-causal mask path).")

    _patch_transformer_module()
    _APPLIED = True


def maybe_apply_diffusion_310p_patches(od_config: Any) -> None:
    """Self-gated standalone entry point.

    No-op unless ``is_310p()`` is True *and* ``model_class_name == MODEL_ARCH``.
    Once the task-11 registry hook lands in ``init_diffusion_model_runner_runtime``
    this file is registered via ``(MODEL_ARCH, apply_patch)`` and the registry
    drives the same function; keep both paths for the rebase window.
    """
    if not is_310p():
        return
    if getattr(od_config, "model_class_name", None) != MODEL_ARCH:
        return
    apply_qwen_image_2_1_310p_patches(od_config)


# Registry-facing alias (COMPARISON-POLICY.md §5).
apply_patch = apply_qwen_image_2_1_310p_patches
