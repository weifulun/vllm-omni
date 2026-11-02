# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Boogu-Image-0.1 family 310P runtime patch (task-33: Base, v0.32.0 line).

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
diffusion pipeline is constructed.

Boogu-Image-0.1-Base capacity ledger (single 310P3 chip, 21589 MB measured):

- transformer ``BooguImageTransformer2DModel`` 10.3B = **20.59 GB FP16**;
  INT8 W8A8 tier ~= 10.3 GB (all linear output dims <= ~27k, inside the
  ``npu_quant_matmul`` 65535 cap -> zero fallback layers expected, still
  log-verified per POLICY section 2);
- FLUX-family ``AutoencoderKL`` (16ch, ``force_upcast=true`` -> VAE runs
  FP32) 0.34 GB FP16 / ~0.67 GB resident;
- FP16 tier: 20.59 + 0.67 = 21.26 GB -> ~0.3 GB left of the 21589 MB budget,
  i.e. **activations do not fit** (1024x1024 denoise/decode peaks ~1-4 GB).
  The INT8 W8A8 tier is therefore the GO-sanctioned primary tier for this
  model (task-33 GO ladder: INT8 first; FP16 + encoder/VAE component CPU
  pin second; mmu retune last).  POLICY section 2: "weights do not fit the
  chip budget -> INT8 fallback tier".
- Qwen3-VL understanding encoder (``mllm``) 17.5 GB: cannot co-reside with
  the transformer on one chip in ANY tier; it is pinned to CPU at the
  container level (capacity-feasibility patch, see
  ``_310p/patch/boogu_mllm_pin.py`` installed in the container FS - same
  standing as the is_310p import fix; NOT gated by
  ``VLLM_OMNI_310P_DIFFUSION_PATCHES`` so the POLICY section-1 level-2
  same-seed reference stays comparable).
- Scheduler (ported ``FlowMatchEulerDiscreteScheduler`` with time-shift v1,
  static seq_len=4096) computes the shift in numpy on CPU; the denoise-loop
  step is ``sample + (t_next - t) * model_output`` (Sub/Mul only, no NPU
  division) - no RealDiv2-on-NPU exposure (research/012 family).
- RoPE: real-valued cos/sin tables built on CPU (float32 on NPU via the
  upstream ``is_npu()`` branch), gathered per forward - already the
  research/012 CPU-table standing; no patch needed.

Numerics invariants enforced here (gated by the registry kill-switch,
engine-level math-SDPA decomposition apart): FP16-only, TORCH_SDPA with the
platform-selector guard, numerics-changing cache backends rejected (the
pipeline itself also refuses any cache backend), quantization tier gate
(FP16 primary / INT8 W8A8 sanctioned, FP8/MXFP*/bitsandbytes rejected), VAE
tiling+slicing forced, single-chip TP1 parallelism gate.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Pipeline class names serving the Boogu-Image-0.1 family (Base and Turbo
#: share the module; Base is the task-33 GO checkpoint).
BOOGU_ARCHS = ("BooguImagePipeline", "BooguImageTurboPipeline")

_MINDIE_SD_ONLY_BACKENDS = frozenset({"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"})

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False


def _force_fp16(od_config: Any) -> None:
    """310P is FP16-only; reject non-float16 dtypes (no BF16 decision)."""
    dtype = getattr(od_config, "dtype", None)
    import torch

    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}); "
        "310P is FP16-only (EZ1001 hardware rejection of BF16; "
        "COMPARISON-POLICY.md section 2). Pass float16 as the diffusion dtype."
    )


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections.

    The math decomposition in ``patch/attention.py`` (engine level, applied
    before the kill-switch gate) then replaces both the ``SDPAImpl``
    ``forward_npu`` entry and ``torch.nn.functional``.
    ``scaled_dot_product_attention`` (the diffusers VAE mid-block attention
    is covered by the same F-level wrap).
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
            "[310P][boogu] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][boogu] disabling numerics-changing cache backend %r on 310P "
        "(same-seed reference comparison per COMPARISON-POLICY.md section 1); "
        "re-enable after 310P validation. (The Boogu pipeline itself also "
        "raises NotImplementedError for any cache backend.)",
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
    """Precision-tier gate for the Boogu family on 310P.

    Tier ladder (task-33, real-device evidence):
    - FP16 with layerwise offload is the feasible tier: full-resident FP16
      does not fit the single-chip 21589 MB budget (20.59 GB transformer +
      ~0.67 GB FP32 VAE), so the DiT blocks stream from host RAM
      (``--enable-layerwise-offload``; the Boogu transformer declares
      ``_layerwise_offload_blocks_attrs``).
    - INT8 W8A8 is REJECTED: the POLICY section-2 fallback tier depends on
      ``torch_npu.npu_quant_matmul`` (aclnnQuantMatmulV5), which fails with
      error 161002 on every shape tested on 310P3 (task-33 probe matrix,
      CANN 9.1.0 + torch_npu 2.10.0.post4) - the kernel is not functional on
      this chip generation (research/009's 310P INT8 premise, falsified in
      the field; the H3-NPU INT8 recipe targets 910B/950-class 800I cards).
    - FP8/MXFP* give no hardware benefit on 310P, ConvRot-INT4 is
      unvalidated, bitsandbytes is CUDA-only.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        offload_on = bool(getattr(od_config, "enable_layerwise_offload", False))
        logger.warning(
            "[310P][boogu] quantization_config unset (FP16 tier). Full-resident "
            "FP16 does NOT fit the single-chip 21589 MB budget for "
            "Boogu-Image-0.1-Base (20.59 GB transformer + ~0.67 GB FP32 VAE); "
            "run with --enable-layerwise-offload so the DiT blocks stream from "
            "host RAM (task-33 measured-fits path; 1024x1024 activations need "
            "several GB). enable_layerwise_offload=%s.%s The INT8 W8A8 tier "
            "(POLICY section 2) is hardware-infeasible on 310P3: "
            "torch_npu.npu_quant_matmul (aclnnQuantMatmulV5) fails with error "
            "161002 on every shape tested (task-33 probe matrix, CANN 9.1.0 + "
            "torch_npu 2.10.0.post4).",
            offload_on,
            "" if offload_on else " (OFF: expect OOM)",
        )
        return
    method = _quantization_method(quant_config)
    if method == "component":
        # Per-component routing (ComponentQuantizationConfig), e.g. the
        # official NPU INT8 invocation '{"transformer": {"method": "int8"},
        # "mllm": null}'.  Every routed component must be the sanctioned tier
        # (null); INT8 is rejected outright on 310P3 (see the int8 branch:
        # the QuantMatmulV5 kernel is not functional on this chip generation).
        components = getattr(quant_config, "_components", None) or {}
        default_cfg = getattr(quant_config, "_default", None)
        routed = {prefix: _quantization_method(sub) if sub is not None else None for prefix, sub in components.items()}
        routed["<default>"] = _quantization_method(default_cfg) if default_cfg is not None else None
        if all(m is None for m in routed.values()):
            logger.warning(
                "[310P][boogu] per-component quantization accepted: %r (all components null = FP16 tier).",
                routed,
            )
            return
        raise RuntimeError(
            f"310P Boogu-Image rejects per-component quantization methods {routed!r}: "
            "the INT8 W8A8 tier (method='int8') is hardware-infeasible on 310P3 "
            "(npu_quant_matmul / aclnnQuantMatmulV5 fails with error 161002 on "
            "every shape tested, CANN 9.1.0 + torch_npu 2.10.0.post4, task-33 "
            "probe matrix); run the FP16 tier with --enable-layerwise-offload "
            "instead (POLICY section 2 tier ladder, task-33 GO)."
        )
    if method == "int8":
        raise RuntimeError(
            f"310P Boogu-Image rejects quantization method {method!r}: the INT8 "
            "W8A8 tier (POLICY section 2 fallback) is hardware-infeasible on "
            "310P3 - torch_npu.npu_quant_matmul (aclnnQuantMatmulV5) fails with "
            "error 161002 on every shape tested, including canonical "
            "4096x4096 (task-33 probe matrix, CANN 9.1.0 + torch_npu 2.10.0.post4; "
            "npu_dynamic_quant additionally fails 561103 on unaligned K). Run "
            "the FP16 tier with --enable-layerwise-offload (DiT blocks stream "
            "from host RAM; full-resident FP16 needs 21.26 GB vs 21589 MB budget)."
        )
    raise RuntimeError(
        f"310P Boogu-Image rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): only the FP16 tier (unset or "
        "per-component all-null) is feasible on 310P - INT8 W8A8's kernel "
        "(npu_quant_matmul) is not functional on 310P3 (task-33 probe matrix); "
        "FP8/MXFP* give no hardware benefit on 310P, ConvRot-INT4 is "
        "unvalidated, bitsandbytes is CUDA-only."
    )


def _force_vae_tiling(od_config: Any) -> None:
    """Engage the diffusers tiling/slicing decode path for the FLUX VAE.

    The FLUX-family ``AutoencoderKL`` (16ch) runs FP32 (``force_upcast``) on
    310P; the untiled 1024x1024 decode activation peak does not fit the
    residual chip budget with the transformer resident (FP16 tier) and is
    kept tiled in the INT8 tier as well so the two tiers stay comparable.
    ``registry.py`` applies ``vae.use_tiling/use_slicing`` from
    ``od_config`` after pipeline construction, so setting the config flags
    at patch time is sufficient.  task-11 SDXL precedent.
    """
    forced = False
    for attr in ("vae_use_tiling", "vae_use_slicing"):
        if not getattr(od_config, attr, False):
            setattr(od_config, attr, True)
            forced = True
    if forced:
        logger.warning(
            "[310P][boogu] forcing vae_use_tiling=true vae_use_slicing=true "
            "(untiled FLUX-VAE FP32 decode peaks do not fit the 21589 MB "
            "chip budget with the transformer resident; task-11 SDXL "
            "precedent, kept in both FP16/INT8 tiers for comparability)."
        )


def _validate_single_chip_parallelism(od_config: Any) -> None:
    """GO task-33 is single-chip TP1: reject multi-rank parallel degrees.

    Sequence parallel (Ulysses/Ring/AllGather-KV) and CFG parallel each need
    more than one diffusion worker rank; on one Duo card that routes into
    the multi-worker HCCL path (GATE-TP2-STATUS.md).  The single-worker path
    handles CFG sequentially (two forwards, cached text-embed swap), which
    is the intended 310P topology.  The pipeline itself also rejects
    ulysses/ring/allgather > 1; this gate fails fast at config time.
    """
    for attr in ("ulysses_degree", "ring_degree", "allgather_degree"):
        value = getattr(od_config, attr, 1) or 1
        if value != 1:
            raise RuntimeError(
                f"310P single-chip Boogu run rejects {attr}={value}: "
                "sequence parallel needs multi-rank HCCL (GATE-TP2-STATUS.md). "
                "Run TP1; CFG is handled sequentially by the pipeline at "
                "cfg_parallel_size=1."
            )
    if (getattr(od_config, "cfg_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            f"310P single-chip Boogu run rejects cfg_parallel_size="
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
        "[310P][boogu] placement plan: dtype=fp16 attention=TORCH_SDPA "
        "(math decomposition) vae_tiling=forced mllm=CPU-pin(container "
        "feasibility patch, kill-switch independent) quantization=%s "
        "parallel=TP1(sequential CFG) scheduler=cpu-numpy-flow-match "
        "(static time-shift seq_len=4096, no NPU division) rope=CPU real "
        "tables model_class=%s",
        _quantization_method(getattr(od_config, "quantization_config", None)) or "FP16",
        getattr(od_config, "model_class_name", None),
    )


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all Boogu-Image 310P invariants. Idempotent per process.

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
    _validate_single_chip_parallelism(od_config)
    _install_attention_guard()
    _log_placement_plan(od_config)
