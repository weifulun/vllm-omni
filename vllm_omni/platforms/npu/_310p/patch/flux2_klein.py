# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch FLUX.2-klein for the 310P NPU diffusion path.

Registered in ``vllm_omni.platforms.npu._310p.patch.__init__`` (the 310P
diffusion patch registry, COMPARISON-POLICY.md section 5); ``platform.py``
stays untouched by this branch. Dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
pipeline is constructed.

310P diffusion decision: pure ``torch_npu`` + FP16, no MindIE-SD kernels.
Invariants enforced here:

* ``od_config.dtype`` must be ``float16``.  310P is FP16-only (no BF16); a
  non-FP16 config would sail silently onto unqualified operators.
* The diffusion attention backend resolves to ``TORCH_SDPA`` whenever the
  user did not select one: MindIE-SD's ``FLASH_ATTN`` / ``RAINFUSION_ATTN``
  kernels do not exist on 310P, and the platform's ``find_spec("mindiesd")``
  fallback cannot be trusted on this card.  An explicit request for a
  MindIE-SD-only backend fails at patch time with a clear error instead of
  deep inside a kernel call.  (A native ``torch_npu`` fusion-attention
  acceleration is a later A/B candidate, tracked with task-11's report.)
* Numerics-changing cache backends are disabled while the 310P path is
  unvalidated, so the same-seed reference comparison (COMPARISON-POLICY.md
  section 1) is not broken by step-skipping caches.
* ``quantization_config`` must be unset: FP8/INT8 310P operator coverage is
  out of scope for the t2i baseline.
* LoRA is accepted with a loud warning (out of scope for the parity
  baseline; weights differ from the reference run by design).
* Chain-1 OOM downgrade (lead-fixed order): when the chip's 21589 MB budget
  is exceeded, ``FLUX2KLEIN_310P_ENCODER_CPU=1`` pins the Qwen3 text
  encoder on host memory (same playbook as model-qwenimg21 M8/R12); the
  wrapper moves it back after pipeline construction.  Chain-2 (SP=2) stays
  gated on the task-5/task-7 HCCL validation.
* VAE decode is forced onto the diffusers tiling/slicing path before the
  pipeline is loaded: the untiled 1024^2 decode peak is 8 GB+ of activations,
  which does not fit the 24 GB per-chip budget with weights resident
  (same finding as task-11's SDXL risk list).

Component placement (logged at startup, matches the pipeline's own
``get_local_device()`` behavior):

* ``transformer``: sharded across the configured TP/SP group (Ulysses strict
  mode requires ``seq_len % sp == 0`` and ``heads % ulysses_degree == 0``).
* ``text_encoder`` (Qwen3ForCausalLM) and ``vae`` (AutoencoderKLFlux2):
  replicated on every rank's local chip, forward-per-rank.  The 4B t2i
  baseline therefore fits on a single 310P chip (24 GB); SP=2 scale-out is
  the memory headroom path and is gated on the task-5/task-7 HCCL
  validation (COMPARISON-POLICY.md section 4), not assumed here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: ``od_config.model_class_name`` value for the FLUX.2-klein pipeline.
FLUX2_KLEIN_ARCH = "Flux2KleinPipeline"

#: Backends whose NPU kernels ship only with MindIE-SD (absent on 310P).
_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False
_VAE_TILING_WRAPPED = False

#: Chain-1 OOM downgrade: pin the Qwen3 text encoder on host memory.
_ENCODER_CPU_ENV = "FLUX2KLEIN_310P_ENCODER_CPU"


def _force_fp16(od_config: Any) -> None:
    dtype = getattr(od_config, "dtype", None)
    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}); "
        "310P is FP16-only and the FLUX.2-klein line has no bf16 decision. "
        "Pass float16 as the diffusion dtype."
    )


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections."""
    from vllm_omni.diffusion.data import AttentionSpec

    attn_config = getattr(od_config, "diffusion_attention_config", None)
    if attn_config is None:
        return "TORCH_SDPA"

    default_spec = getattr(attn_config, "default", None)
    if default_spec is not None and str(getattr(default_spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
        raise RuntimeError(
            f"diffusion attention backend {default_spec.backend!r} requires MindIE-SD kernels, "
            "which are unavailable on 310P; use TORCH_SDPA."
        )

    per_role = getattr(attn_config, "per_role", None) or {}
    for role, spec in per_role.items():
        if str(getattr(spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention per-role backend for {role!r} is {spec.backend!r}, "
                "which requires MindIE-SD kernels unavailable on 310P; use TORCH_SDPA."
            )

    if default_spec is None:
        attn_config.default = AttentionSpec(backend="TORCH_SDPA")
    return "TORCH_SDPA"


def _disable_numerical_cache_backends(od_config: Any) -> None:
    backend = getattr(od_config, "cache_backend", None)
    if backend is None or backend == "none":
        return
    if backend not in _NUMERICAL_CACHE_BACKENDS:
        logger.warning(
            "[310P][flux2-klein] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][flux2-klein] disabling numerics-changing cache backend %r on 310P "
        "(same-seed reference comparison per COMPARISON-POLICY.md section 1); "
        "re-enable after 310P validation.",
        backend,
    )
    od_config.cache_backend = "none"


def _check_heads_divisibility(od_config: Any) -> None:
    """Strict-Ulysses preflight: head count must divide the ulysses degree."""
    parallel = getattr(od_config, "parallel_config", None)
    ulysses = getattr(parallel, "ulysses_degree", 1) if parallel is not None else 1
    if not ulysses or ulysses <= 1:
        return
    tf_config = getattr(od_config, "tf_model_config", None)
    if tf_config is None:
        return
    heads = tf_config.get("num_attention_heads", None)
    if heads is None:
        return
    if heads % ulysses != 0:
        raise RuntimeError(
            f"strict Ulysses on 310P needs num_attention_heads % ulysses_degree == 0, "
            f"got heads={heads}, ulysses_degree={ulysses}. "
            "Single-chip TP=1 baseline is unaffected."
        )


def _maybe_enable_vae_tiling(vae: Any) -> bool:
    """Put the VAE onto diffusers' tiling/slicing decode path.

    Returns True when both switches engaged.  Guarded with ``getattr`` so a
    diffusers revision without the switches still boots (the warning keeps
    the untiled-decode memory risk on the record).
    """
    enabled = True
    for attr in ("enable_tiling", "enable_slicing"):
        fn = getattr(vae, attr, None)
        if not callable(fn):
            logger.warning(
                "[310P][flux2-klein] VAE %s has no %r; untiled decode activation peak "
                "(8 GB+ at 1024^2) is a memory risk on 24 GB chips.",
                type(vae).__name__,
                attr,
            )
            enabled = False
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - boot must not die on a tiling flag
            logger.warning("[310P][flux2-klein] VAE %s() failed: %s", attr, exc)
            enabled = False
    return enabled


def _install_vae_tiling_wrap() -> None:
    """Wrap ``Flux2KleinPipeline.__init__`` so the VAE switches on before
    the first request (the runner loads the pipeline right after the patch
    runs)."""
    global _VAE_TILING_WRAPPED
    if _VAE_TILING_WRAPPED:
        return

    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline

    original_init = Flux2KleinPipeline.__init__

    def _init_with_vae_tiling(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._310p_vae_tiling = _maybe_enable_vae_tiling(self.vae)
        if os.environ.get(_ENCODER_CPU_ENV, "0") == "1":
            self.text_encoder = self.text_encoder.to("cpu")
            self._310p_encoder_cpu = True
            logger.warning(
                "[310P][flux2-klein] text encoder pinned to CPU via %s "
                "(chain-1 OOM downgrade, chip budget 21589 MB); per-request "
                "H2D/D2H expected, encode stage timed separately.",
                _ENCODER_CPU_ENV,
            )

    Flux2KleinPipeline.__init__ = _init_with_vae_tiling
    _VAE_TILING_WRAPPED = True


def _log_placement_plan(od_config: Any) -> None:
    parallel = getattr(od_config, "parallel_config", None)
    tp = getattr(parallel, "tensor_parallel_size", 1) if parallel is not None else 1
    sp = getattr(parallel, "sequence_parallel_size", None) if parallel is not None else None
    ulysses = getattr(parallel, "ulysses_degree", 1) if parallel is not None else 1
    ring = getattr(parallel, "ring_degree", 1) if parallel is not None else 1
    allgather = getattr(parallel, "allgather_degree", 1) if parallel is not None else 1
    logger.info(
        "[310P][flux2-klein] dtype=fp16 attention=TORCH_SDPA vae_tiling=forced | parallel: tp=%s "
        "sp=%s (ulysses=%s ring=%s allgather=%s) | placement: transformer sharded across the "
        "TP/SP group; text_encoder (Qwen3) and vae (AutoencoderKLFlux2) replicated per rank "
        "on each rank's local chip; SP>1 requires task-5/task-7 HCCL validation first.",
        tp,
        sp,
        ulysses,
        ring,
        allgather,
    )


def _install_attention_guard() -> None:
    """Wrap the NPU platform backend selector so 310P can never resolve to
    MindIE-SD-only kernels, even for per-role or per-request selections that
    bypass the od_config-level default."""
    global _ATTENTION_GUARD_INSTALLED
    if _ATTENTION_GUARD_INSTALLED:
        return

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    # Unwrap the classmethod descriptor so ``cls`` is passed exactly once.
    original = NPUOmniPlatform.__dict__["get_diffusion_attn_backend_cls"]
    original = getattr(original, "__func__", original)

    @classmethod
    def _guarded(cls, selected_backend, head_size, allow_trtllm_default: bool = True):  # noqa: ANN001
        if selected_backend is None:
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
        if str(selected_backend).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention backend {selected_backend!r} requires MindIE-SD "
                "kernels, which are unavailable on 310P; use TORCH_SDPA."
            )
        return original(cls, selected_backend, head_size, allow_trtllm_default=allow_trtllm_default)

    NPUOmniPlatform.get_diffusion_attn_backend_cls = _guarded
    _ATTENTION_GUARD_INSTALLED = True


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all FLUX.2-klein 310P invariants. Idempotent per process.

    Contract name required by the 310P diffusion patch registry
    (``_DIFFUSION_310P_PATCH_MODULES`` dispatch in
    ``vllm_omni.platforms.npu._310p.patch``).
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _force_fp16(od_config)
    _validate_attention_selection(od_config)
    _disable_numerical_cache_backends(od_config)
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is not None:
        raise RuntimeError(
            f"310P FLUX.2-klein baseline is unquantized; got quantization_config={quant_config!r}. "
            "FP8/INT8 operator coverage on 310P is out of scope for the t2i reference baseline."
        )
    if getattr(od_config, "lora_path", None) is not None:
        logger.warning(
            "[310P][flux2-klein] lora_path=%r is out of scope for the 310P reference "
            "baseline; outputs will not match the reference run weights.",
            od_config.lora_path,
        )
    _check_heads_divisibility(od_config)
    _install_vae_tiling_wrap()
    _install_attention_guard()
    _log_placement_plan(od_config)
