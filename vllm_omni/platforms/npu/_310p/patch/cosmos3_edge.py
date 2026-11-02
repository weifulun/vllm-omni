# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cosmos3 family 310P runtime patch (task-25: Cosmos3-Edge, v0.32.0 line).

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
diffusion pipeline is constructed.

Cosmos3 facts driving the invariants:

- The 3.9B-parameter unified AR+diffusion transformer (7.7 GB FP16) plus the
  Wan 3D VAE must fit ONE 21589 MB 310P3 chip: FP16-only, no over-budget
  force-fitting.  INT8 W8A8 stays the sanctioned POLICY section-2 fallback
  tier for the family (not needed on Edge); FP8/MXFP* give no hardware
  benefit on 310P.
- Fused ``F.scaled_dot_product_attention`` is broken on 310P (research/012
  dtprobe: ERR01007 on every dtype/mask/length; torch MATH backend via SDPA
  equally broken).  The explicit math decomposition lives in
  ``patch/attention.py`` and is installed by the registry dispatcher before
  this module runs - this module only gates the backend SELECTION.
- ``FlowUniPCMultistepScheduler`` computes sigmas in numpy on CPU and is
  parked back on CPU after loading (``_cpu_scheduler_state``): no pin needed.
- Guardrails are ON by default and need the gated
  ``nvidia/Cosmos-1.0-Guardrail`` weights; offline 310P nodes have no egress,
  and the upstream Edge recipe runs its official examples with
  ``"guardrails": false`` ("you own license compliance").  The patch forces
  the server-level gate off with a loud warning.
- Single-chip GO (task-25): TP1 only.  Sequence/CFG parallel degrees need
  multi-rank HCCL which the GATE-TP2 status keeps closed; the pipeline's
  sequential CFG (two forwards with cache swap at cfg_size=1) is the
  supported single-worker path.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Pipeline class names serving the Cosmos3 family (Edge / Nano / Super share
#: ``Cosmos3OmniDiffusersPipeline``; the HF model_index carries the shorter
#: alias ``Cosmos3OmniPipeline`` which resolves to the same class).
COSMOS3_ARCHS = ("Cosmos3OmniPipeline", "Cosmos3OmniDiffusersPipeline")

_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False


def _force_fp16(od_config: Any) -> None:
    """310P is FP16-only; the Cosmos3 line has no bf16 decision."""
    dtype = getattr(od_config, "dtype", None)
    import torch

    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}); "
        "310P is FP16-only and the Cosmos3 line has no bf16 decision "
        "(BF16-only NPU is absent; FP16 per COMPARISON-POLICY.md section 2). "
        "Pass float16 as the diffusion dtype."
    )


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections.

    The math decomposition in ``patch/attention.py`` then replaces both the
    ``SDPAImpl.forward_npu`` entry and ``torch.nn.functional``.
    ``scaled_dot_product_attention`` for anything the pipeline built itself.
    """
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
            "[310P][cosmos3] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][cosmos3] disabling numerics-changing cache backend %r on 310P "
        "(same-seed reference comparison per COMPARISON-POLICY.md section 1); "
        "re-enable after 310P validation.",
        backend,
    )
    od_config.cache_backend = "none"


def _quantization_method(quant_config: Any) -> str | None:
    """Resolve the quantization method name from a QuantizationConfig object."""
    getter = getattr(quant_config, "get_name", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # pragma: no cover - defensive, contract-level
            pass
    for attr in ("method", "name"):
        value = getattr(quant_config, attr, None)
        if isinstance(value, str):
            return value
    return None


def _validate_quantization(od_config: Any) -> None:
    """Precision-tier gate: FP16 primary, INT8 W8A8 sanctioned fallback only.

    POLICY section 2 (revised 2026-11-02): FP16 is the first choice; when
    weights do not fit the chip budget, INT8 is the fallback tier (no BF16,
    no over-budget force-fitting, the quantization configuration must be
    recorded in the report).  research/009: on 310P only INT8 W8A8 has a
    hardware benefit (``npu_quant_matmul``); FP8/MXFP* fall back to FP16
    matmul with no gain, ConvRot NPU applicability is unvalidated,
    bitsandbytes is CUDA-only.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][cosmos3] quantization_config method=%r accepted as the "
            "POLICY section-2 second tier (research/009): online dynamic INT8 "
            "via npu_quant_matmul, layers with output dim > 65535 fall back "
            "to unquantized (mixed-precision semantics). The per-layer "
            "fallback log is MANDATORY in the real-device report, together "
            "with precision evidence vs the FP16 reference. On Cosmos3-Edge "
            "(3.9B, 7.7 GB FP16) INT8 is unnecessary - the FP16 tier fits "
            "the single-chip budget.",
            method,
        )
        return
    raise RuntimeError(
        f"310P Cosmos3 rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): FP16 is the primary tier "
        "(quantization_config unset; Cosmos3-Edge fits the single-chip "
        "21589 MB budget at FP16); only the INT8 W8A8 fallback "
        "(method='int8') is sanctioned on 310P (POLICY section 2, "
        "research/009). FP8/MXFP* give no hardware benefit on 310P (served "
        "with FP16 matmul); ConvRot-INT4 is unvalidated; bitsandbytes is "
        "CUDA-only."
    )


def _force_vae_tiling(od_config: Any) -> None:
    """Engage the diffusers tiling/slicing decode path for the Wan 3D VAE.

    Untiled 3D decode activation peaks for the video geometry (480x832, up to
    48 latent frames) sit close to the 21589 MB per-chip budget with
    transformer weights resident; tiling is the task-11 SDXL precedent and the
    reference-run choice.  ``registry.py`` applies ``vae.use_tiling/
    use_slicing`` from ``od_config`` after pipeline construction, so setting
    the config flags at patch time is sufficient.
    """
    forced = False
    for attr in ("vae_use_tiling", "vae_use_slicing"):
        if not getattr(od_config, attr, False):
            setattr(od_config, attr, True)
            forced = True
    if forced:
        logger.warning(
            "[310P][cosmos3] forcing vae_use_tiling=true vae_use_slicing=true "
            "(untiled 3D VAE decode activation peaks do not fit the 21589 MB "
            "chip budget with weights resident; task-11 SDXL precedent)."
        )


def _force_guardrails_off(od_config: Any) -> None:
    """Disable the server-level Cosmos3 guardrail gate on offline 310P.

    Guardrails are enabled by default and load the gated
    ``nvidia/Cosmos-1.0-Guardrail`` checkpoints at pipeline build time; the
    offline 310P nodes have no egress.  The upstream Edge recipe runs its
    official examples with ``"guardrails": false`` ("you own license
    compliance"), which is the same choice this patch makes - loudly.
    """
    model_config = getattr(od_config, "model_config", None)
    if not isinstance(model_config, dict):
        return
    if not bool(model_config.get("guardrails", True)):
        return
    model_config["guardrails"] = False
    logger.warning(
        "[310P][cosmos3] forcing model_config['guardrails']=false: the "
        "guardrail models (nvidia/Cosmos-1.0-Guardrail, gated) are "
        "unreachable on the offline 310P nodes. This matches the upstream "
        "Edge recipe's official examples, which run with guardrails off "
        "(license compliance is the operator's responsibility per the "
        "NVIDIA Open Model License)."
    )


def _validate_single_chip_parallelism(od_config: Any) -> None:
    """GO task-25 is single-chip TP1: reject multi-rank parallel degrees.

    Sequence parallel (Ulysses/Ring/AllGather-KV) and CFG parallel each need
    more than one diffusion worker rank; on one Duo card that routes into the
    multi-worker HCCL path whose gate is CLOSED (GATE-TP2-STATUS.md).  The
    single-worker path handles CFG sequentially (two forwards with a cached
    KV swap), which is the intended 310P topology.
    """
    for attr in ("ulysses_degree", "ring_degree", "allgather_degree"):
        value = getattr(od_config, attr, 1) or 1
        if value != 1:
            raise RuntimeError(
                f"310P single-chip Cosmos3 run rejects {attr}={value}: "
                "sequence parallel needs multi-rank HCCL, whose gate is closed "
                "(GATE-TP2-STATUS.md). Run TP1; CFG is handled sequentially "
                "by the pipeline at cfg_parallel_size=1."
            )
    if (getattr(od_config, "cfg_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            f"310P single-chip Cosmos3 run rejects cfg_parallel_size="
            f"{getattr(od_config, 'cfg_parallel_size', None)}: CFG parallel "
            "needs two worker ranks. Run cfg_parallel_size=1 (sequential CFG)."
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


def _log_placement_plan(od_config: Any) -> None:
    logger.info(
        "[310P][cosmos3] placement plan: dtype=fp16 attention=TORCH_SDPA "
        "(math decomposition) vae_tiling=forced guardrails=off "
        "parallel=TP1(sequential CFG) scheduler=cpu-flow-unipc "
        "model_class=%s",
        getattr(od_config, "model_class_name", None),
    )


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all Cosmos3 310P invariants. Idempotent per process.

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
    _validate_quantization(od_config)
    _force_vae_tiling(od_config)
    _force_guardrails_off(od_config)
    _validate_single_chip_parallelism(od_config)
    _install_attention_guard()
    _log_placement_plan(od_config)
