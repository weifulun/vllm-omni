# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ming-Image Design / Design-Layer 310P runtime patch (task-36).

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
diffusion pipeline is constructed.  Model branches never touch platform.py.

Ming-Image facts driving the invariants (checkpoint metadata, v0.32.0 code):

- Two-stage topology: stage 0 is the AR design agent (``mllm/``,
  BailingMoeV2 MoE 17.00B + Qwen2.5-ViT, ``mllm/model.safetensors.index.json``
  total_parameters=17,001,125,376 / total_size=34,002,250,752 B BF16) and
  stage 1 is this diffusion pipeline.  Stage 0 runs as its own TP1 engine and
  is OUT of scope for this module; its placement (needs >= 34 GB FP16, i.e.
  not a single 21589 MB chip) is a lead-level decision.
- Stage 1 residency is the 12.3 GB DiT headline of the 014 survey:
  ``transformer/`` (MingImageTransformer2DModel, a ZImageTransformer2DModel
  adapter, 12,309,802,112 B on disk BF16) + ``MingImageConditioning``
  (Qwen2-1.5B connector, 1,543,714,304 params F32 on disk -> 3.09 GB FP16)
  + RGBA ``AutoencoderKLQwenImage`` VAE (253,817,336 B) + ``mlp/`` projectors
  (124,837,576 B) ~= 15.8 GB FP16 resident, under the 21589 MB budget with
  headroom for the head-chunked attention peaks.
- The DiT attention/RoPE/time-embedding machinery is the shared Z-Image family
  (same node task-14 line ran it green 4x at 1024^2): the hardware-level
  shims (FP16 default pin, MATH-SDPA head-chunked decomposition, warmup
  skip, encoder dtype pin) are installed by the registry dispatcher BEFORE
  this module runs; nothing model-specific to redo here.
- No HF text encoder in this stage: text conditioning arrives as hidden
  states from stage 0 plus the learned query/direct projectors
  (``MingImageConditioning``), so there is no HF-encoder CPU-offload target
  and no ``_encode_prompt`` device-wrap needed (unlike the Z-Image line).
- ``FlowMatchEulerDiscreteScheduler`` (shift 6.0, no dynamic shifting) is the
  same scheduler family the Z-Image line ran green on this node; sigmas are
  computed on the CPU side of the ZImagePipeline loop, so no RealDiv-on-NPU
  pin is required (research/012 class; watch item for the first real run).
- The upstream deploy yaml ships stage 1 with ``enforce_eager: false`` and
  ``diffusion_compile_granularity: regional`` (torch.compile reduce-overhead
  buckets / CUDA-graph decode capture).  CUDA-graph semantics are unvalidated
  on 310P and every diffusion line on this cluster runs eager (task-11/14/19
  /25 precedent); this module forces eager decode.

Same-seed comparability (COMPARISON-POLICY.md section 1): numerics-changing
cache backends are disabled, so patched and level-2 reference runs share the
same denoising path modulo the explicitly gated model invariants.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Pipeline class names serving the Ming-Image family.  The diffusion
#: registry knows both the primary name and the layered alias
#: (``_DIFFUSION_MODELS["MingImageLayeredDiffusionPipeline"] = ...[
#: "MingImageDiffusionPipeline"]``).
MING_IMAGE_ARCHS = ("MingImageDiffusionPipeline", "MingImageLayeredDiffusionPipeline")

_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_APPLIED = False


def _force_fp16(od_config: Any) -> None:
    """310P is FP16-only; force the runtime dtype before pipeline build.

    The engine default is BF16, which 310P refuses at the hardware level
    (EZ1001 / aclnnInplaceOne DT_BFLOAT16).  Forcing mirrors the Z-Image
    line (task-14) instead of hard-rejecting an explicit bf16, so an
    unedited upstream deploy config still lands on a runnable FP16 path;
    the POLICY section-2 ladder (FP16 first, no over-budget force-fitting)
    is satisfied because the stage-1 residency (~15.8 GB) fits the chip.
    """
    import torch

    if getattr(od_config, "dtype", None) is torch.float16:
        return
    logger.warning(
        "[310P][ming-image] forcing diffusion runtime dtype %s -> float16 "
        "(310P is FP16-only; no BF16 elementwise kernels).",
        getattr(od_config, "dtype", None),
    )
    od_config.dtype = torch.float16


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections.

    The math decomposition in ``patch/attention.py`` (head-chunked, 256 MiB
    score-matrix budget) is installed by the registry dispatcher before this
    module runs; this gate only pins the backend SELECTION.
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
    """Disable numerics-changing cache backends (same-seed comparability).

    The Ming-Image pipeline carries a cache_dit adapter
    (``_cache_dit_adapter_config``); it only activates when
    ``cache_backend=cache_dit`` is selected (the shipped deploy yaml leaves
    it empty).  POLICY section 1 pins denoising numerics for the same-seed
    double run, so any numerics-changing backend is neutralized.
    """
    backend = getattr(od_config, "cache_backend", None)
    if backend is None or backend == "none":
        return
    if backend not in _NUMERICAL_CACHE_BACKENDS:
        logger.warning(
            "[310P][ming-image] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][ming-image] disabling numerics-changing cache backend %r on 310P "
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

    POLICY section 2: FP16 is the first choice; INT8 is the fallback tier
    only when weights do not fit the chip budget (no BF16, no over-budget
    force-fitting).  The Ming-Image stage-1 DiT (6.15B, 12.3 GB FP16) fits
    the single-chip budget comfortably, so INT8 is NOT expected on this line;
    research/012 generational conclusion: 310P diffusion-side W8A8 has no
    hardware path on this cluster (AR-side W8A8 is a separate, mature path
    and out of scope here).  FP8/MXFP* give no hardware benefit on 310P.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][ming-image] quantization_config method='int8' is the POLICY "
            "section-2 fallback tier but UNNECESSARY on this line: the stage-1 DiT "
            "(12.3 GB FP16) fits the 21589 MB single-chip budget.  If enabled, the "
            "per-layer fallback log and precision evidence vs the FP16 reference are "
            "MANDATORY in the real-device report.",
        )
        return
    raise RuntimeError(
        f"310P Ming-Image rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): FP16 is the primary tier "
        "(quantization_config unset; the DiT fits the single-chip budget); "
        "only the INT8 W8A8 fallback (method='int8') is sanctioned on 310P "
        "(POLICY section 2). FP8/MXFP* give no hardware benefit on 310P; "
        "ConvRot-INT4 is unvalidated; bitsandbytes is CUDA-only."
    )


def _force_eager_decode(od_config: Any) -> None:
    """Force eager decode: no torch.compile buckets, no CUDA-graph capture.

    The shipped deploy yaml (``vllm_omni/deploy/ming_image.yaml``) configures
    stage 1 with ``enforce_eager: false`` and ``diffusion_compile_granularity:
    regional``; the model runner then calls the pipeline's ``setup_compile()``
    (reduce-overhead buckets over the DiT) and would engage the CUDA-graph
    decode path.  CUDA-graph / inductor semantics are unvalidated on 310P
    (CANN 9.1.0 + torch_npu 2.10.0.post4), and every diffusion line on this
    cluster (task-11 SDXL, task-14 Z-Image, task-19 MOSS-Image, task-25
    Cosmos3) validates with enforce-eager as the production path.
    """
    if getattr(od_config, "enforce_eager", True):
        return
    logger.warning(
        "[310P][ming-image] forcing enforce_eager=true: CUDA-graph decode capture "
        "and regional torch.compile (reduce-overhead) are unvalidated on 310P; "
        "eager is the fleet-validated diffusion path (task-11/14/19/25 precedent). "
        "diffusion_compile_granularity remains %r but is not engaged while "
        "enforce_eager is set.",
        getattr(od_config, "diffusion_compile_granularity", None),
    )
    od_config.enforce_eager = True


def _force_vae_tiling(od_config: Any) -> None:
    """Engage the diffusers tiling/slicing decode path for the RGBA VAE.

    ``AutoencoderKLQwenImage`` (z_dim 16, input_channels 4, no attention
    scales) decodes 1024^2/2048^2 RGBA frames; the untiled activation peaks
    sit close to / over the 21589 MB budget with the ~15.8 GB stage-1
    weights resident (Z-Image level-2 probe measured an untiled-VAE
    GroupNorm GE EL0019).  Tiling is the task-11 SDXL precedent.
    ``registry.py`` applies ``vae.use_tiling/use_slicing`` from
    ``od_config`` after pipeline construction, so setting the flags at patch
    time is sufficient.
    """
    forced = False
    for attr in ("vae_use_tiling", "vae_use_slicing"):
        if not getattr(od_config, attr, False):
            setattr(od_config, attr, True)
            forced = True
    if forced:
        logger.warning(
            "[310P][ming-image] forcing vae_use_tiling=true vae_use_slicing=true "
            "(untiled RGBA VAE decode activation peaks do not fit the 21589 MB "
            "chip budget with stage-1 weights resident; task-11 SDXL precedent)."
        )


def _neutralize_attention_kv_quant(od_config: Any) -> None:
    """NPU attention-KV quantization is a MindIE-only path; refuse it loudly."""
    kv_cache_dtype = getattr(od_config, "diffusion_kv_cache_dtype", None)
    if kv_cache_dtype not in (None, "auto", "float"):
        logger.warning(
            "[310P][ming-image] diffusion_kv_cache_dtype=%r is not supported on 310P "
            "(MindIE-only); neutralizing to 'auto'.",
            kv_cache_dtype,
        )
        od_config.diffusion_kv_cache_dtype = "auto"


def _validate_single_chip_parallelism(od_config: Any) -> None:
    """TP1-only preflight for the diffusion stage.

    Sequence/CFG parallel degrees need multi-rank HCCL; GATE-TP2-STATUS
    keeps the multi-worker path gated on this cluster.  The diffusion stage
    runs as its own single-worker engine (stage 0, the AR design agent, is a
    SEPARATE engine and may use the gated TP2 patch-stack path independently
    - that is not this module's concern).
    """
    for attr, label in (
        ("ulysses_degree", "Ulysses sequence parallelism"),
        ("ring_degree", "ring sequence parallelism"),
        ("allgather_degree", "allgather sequence parallelism"),
        ("cfg_parallel_size", "CFG parallelism"),
    ):
        value = getattr(od_config, attr, None)
        if value and value > 1:
            raise RuntimeError(
                f"310P Ming-Image single-chip GO rejects {label} "
                f"({attr}={value}): multi-rank HCCL is gated on this cluster "
                "(GATE-TP2-STATUS.md); the supported topology is TP1 with "
                "sequential CFG / sequential stage flow."
            )


def _log_placement_plan(od_config: Any) -> None:
    logger.info(
        "[310P][ming-image] placement plan: dtype=fp16 attention=TORCH_SDPA "
        "(math decomposition, head-chunked) vae_tiling=forced eager_decode=forced "
        "cache=none parallel=TP1 scheduler=cpu-flowmatch-euler(shift 6.0) "
        "conditioning=connector-Qwen2-1.5B+mlp-on-chip model_class=%s",
        getattr(od_config, "model_class_name", None),
    )


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all Ming-Image stage-1 310P invariants. Idempotent per process.

    Contract name required by the 310P diffusion patch registry
    (``_DIFFUSION_310P_PATCH_MODULES`` dispatch in
    ``vllm_omni.platforms.npu._310p.patch``).  Hardware-level invariants
    (MATH-SDPA, warmup skip, FP16 default pin, HF encoder dtype pin / CPU
    offload) are installed by the dispatcher before this module runs and are
    NOT part of the ``VLLM_OMNI_310P_DIFFUSION_PATCHES=0`` kill switch.
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _force_fp16(od_config)
    _validate_attention_selection(od_config)
    _disable_numerical_cache_backends(od_config)
    _validate_quantization(od_config)
    _force_eager_decode(od_config)
    _force_vae_tiling(od_config)
    _neutralize_attention_kv_quant(od_config)
    _validate_single_chip_parallelism(od_config)
    _log_placement_plan(od_config)
