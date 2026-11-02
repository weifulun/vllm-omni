# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch the LTX-2.5 (LTX2 family) diffusion pipelines for the 310P NPU path.

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); ``platform.py`` stays
untouched.  Dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
pipeline is constructed.

310P decision (task-21, LTX-2.5): pure ``torch_npu`` + FP16, no MindIE-SD
kernels.  Capacity facts driving the invariants below (chip budget 21589 MB
measured per the mount recipe):

* The Full/SFT one-stage transformer is a 19B DiT: FP16 weights ~= 38 GB,
  i.e. 19 GB per chip under SP=2 - over budget once activations, the ConvVAE
  and latents are added.  FP16 SP=2 is therefore infeasible on 310P and the
  sanctioned tier is INT8 W8A8 (POLICY section 2, research/009):
  ``quantization=int8`` (DiffusionInt8Config online dynamic quantization via
  ``torch_npu.npu_quant_matmul``, output-dim > 65535 layers fall back to
  unquantized - mixed-precision semantics, no upstream 310P diffusion INT8
  precedent exists, this is the first run).
* The default LTX-2.5 video decoder is the Native Diffusion VAE, which loads
  its NATTEN attention kernels from the HuggingFace Hub at startup.  There is
  no NATTEN kernel build for NPU, so the canonical DiffVAE cannot boot on
  310P; the patch forces the legacy convolutional VAE
  (``extras["ltx2_use_conv_vae"]=true``, same choice as the XPU 32 GB
  reference run).
* The Gemma-4 12B text encoder (23.9 GB FP16) is replicated per rank by the
  generic placement path.  Replicating it next to the INT8 transformer shard
  (~9.5 GB per chip under SP=2) does not fit the chip budget, so the patch
  pins the encoder on host memory with a device-aware forward wrapper
  (chain-1 OOM downgrade, same playbook as flux2_klein and qwenimg21).  With
  ``--enable-layerwise-offload`` the OffloadPlan streams the encoder blocks
  instead, and the pin is skipped automatically.

Invariants enforced here:

* ``od_config.dtype`` must be ``float16`` (310P is FP16-only, no BF16);
* the diffusion attention backend resolves to ``TORCH_SDPA`` (MindIE-SD-only
  backends rejected at config time and at the platform selector level);
* numerics-changing cache backends are disabled (same-seed reference
  comparison, COMPARISON-POLICY.md section 1);
* precision tiers: FP16 primary; INT8 W8A8 (method ``int8``) accepted with a
  loud warning (per-layer fallback evidence is mandatory in the report);
  FP8/MXFP* (no 310P hardware benefit), ConvRot (unvalidated) and
  bitsandbytes (CUDA-only) are rejected;
* ``ltx2_use_conv_vae`` is forced True (NPU has no NATTEN kernels);
* VAE decode is put on the diffusers tiling/slicing path (untiled decode
  activation peaks do not fit the 21589 MB chip budget);
* the Gemma-4 text encoder is pinned to CPU (device-aware forward wrapper)
  unless a layerwise offload strategy is enabled;
* strict-Ulysses preflight: attention heads must divide the ulysses degree.

Component placement (logged at startup):

* ``transformer`` (LTX2VideoTransformer3DModel, 19B): sharded across the
  configured SP group; SP=2 (one Duo card, intra-card HCCS chip pair) is the
  standard scale-out path - feasibility judged against the task-10 HCCL
  topology table's intra-card pair bandwidth (~13.4 GB/s measured).
* ``text_encoder`` (Gemma-4 12B): CPU (pinned by this patch) or streamed by
  the layerwise offload plan; prompt embeds are moved back to the NPU before
  the connectors.
* ``connectors`` / ``vae`` (ConvVAE) / ``audio_vae`` / ``vocoder`` /
  ``duration_head``: resident on each rank's local chip.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: ``od_config.model_class_name`` values patched by this module.
LTX25_ARCHS = (
    "LTX2Pipeline",
    "LTX2DistilledOneStagePipeline",
    "LTX2TwoStagePipeline",
    "LTX2DistilledTwoStagePipeline",
)

#: Backends whose NPU kernels ship only with MindIE-SD (absent on 310P).
_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

#: ``ltx2_use_conv_vae`` extra key (diffusion_decoder vs ConvVAE selection).
_CONV_VAE_EXTRA = "ltx2_use_conv_vae"

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False
_PIPELINE_INIT_WRAPPED = False

#: Chain-1 OOM downgrade: pin the Gemma-4 text encoder on host memory.
_ENCODER_CPU_ENV = "LTX25_310P_ENCODER_CPU"


def _force_fp16(od_config: Any) -> None:
    dtype = getattr(od_config, "dtype", None)
    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}); "
        "310P is FP16-only and the LTX-2.5 line has no bf16 decision "
        "(research/009, POLICY section 2). Pass float16 as the diffusion dtype."
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
            "[310P][ltx2.5] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][ltx2.5] disabling numerics-changing cache backend %r on 310P "
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
    hardware benefit (``npu_quant_matmul``, 280 TOPS INT8 per card);
    FP8/MXFP* fall back to FP16 matmul with no gain, ConvRot NPU
    applicability is unvalidated, bitsandbytes is CUDA-only.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][ltx2.5] quantization_config method=%r accepted as the "
            "POLICY section-2 second tier (research/009): online dynamic INT8 "
            "via npu_quant_matmul, layers with output dim > 65535 fall back "
            "to unquantized (mixed-precision semantics). The per-layer "
            "fallback log (count + shapes) is MANDATORY in the real-device "
            "report, together with precision evidence vs the FP16 reference; "
            "no upstream 310P diffusion INT8 precedent exists (first run).",
            method,
        )
        return
    raise RuntimeError(
        f"310P LTX-2.5 rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): FP16 is the primary tier "
        "(quantization_config unset; infeasible here - 19B transformer "
        "~= 19 GB/chip at SP=2 vs 21589 MB budget); only the INT8 W8A8 "
        "fallback (method='int8') is sanctioned on 310P (POLICY section 2, "
        "research/009). FP8/MXFP* give no hardware benefit on 310P (served "
        "with FP16 matmul); ConvRot-INT4 is unvalidated; bitsandbytes is "
        "CUDA-only."
    )


def _force_conv_vae(od_config: Any) -> None:
    """Force the legacy convolutional VAE: the canonical DiffVAE needs NATTEN.

    ``LTX25_NATIVE`` video decoding loads its attention kernels from the
    HuggingFace Hub (NATTEN); there is no NATTEN kernel build for NPU, so the
    canonical decoder fails early at pipeline construction on 310P.  The
    XPU 32 GB reference run made the same choice
    (``ltx2_use_conv_vae=true``).  If the user explicitly opted OUT of the
    ConvVAE, that selection cannot work on this card - fail at patch time
    with a clear message instead of inside the NATTEN kernel load.
    """
    extras = getattr(od_config, "extras", None)
    if extras is None:
        od_config.extras = extras = {}
    if not isinstance(extras, dict):
        raise TypeError(f"od_config.extras must be a dict, got {type(extras)!r}")

    current = extras.get(_CONV_VAE_EXTRA)
    if current is False:
        raise RuntimeError(
            "ltx2_use_conv_vae=false (canonical DiffVAE) is unavailable on 310P: "
            "the DiffVAE decoder loads NATTEN kernels from the HuggingFace Hub "
            "at startup and no NATTEN kernel build exists for NPU. Remove the "
            "explicit opt-out; the 310P patch forces the convolutional VAE."
        )
    if current is True:
        return
    extras[_CONV_VAE_EXTRA] = True
    logger.warning(
        "[310P][ltx2.5] forcing %s=true: the canonical LTX-2.5 DiffVAE decoder "
        "requires NATTEN Hub kernels that do not exist for NPU "
        "(XPU 32 GB reference run made the same choice); the legacy "
        "convolutional VAE is used instead.",
        _CONV_VAE_EXTRA,
    )


def _force_vae_tiling(od_config: Any) -> None:
    """Engage the diffusers tiling/slicing decode path for the VAE.

    The untiled decode activation peak (8 GB+ class) does not fit the 21589 MB
    per-chip budget with weights resident.  ``registry.py`` applies
    ``vae.use_tiling/use_slicing`` from ``od_config`` after pipeline
    construction, so setting the config flags at patch time is sufficient.
    """
    forced = False
    for attr in ("vae_use_tiling", "vae_use_slicing"):
        if not getattr(od_config, attr, False):
            setattr(od_config, attr, True)
            forced = True
    if forced:
        logger.warning(
            "[310P][ltx2.5] forcing vae_use_tiling=true vae_use_slicing=true "
            "(untiled VAE decode activation peaks do not fit the 21589 MB "
            "chip budget; task-11 SDXL risk-list precedent)."
        )


def _layerwise_offload_enabled(od_config: Any) -> bool:
    """True when an offload strategy that streams components is configured.

    With LAYER_WISE / DISTRIBUTED_LAYER_WISE the OffloadPlan streams the
    encoder blocks (and optionally the DiT) between host and device, so the
    CPU encoder pin below is neither needed nor correct (it would fight the
    residency manager).
    """
    if getattr(od_config, "enable_layerwise_offload", False) or getattr(
        od_config, "enable_distributed_layerwise_offload", False
    ):
        return True
    offload_config = getattr(od_config, "diffusion_offload_config", None)
    if offload_config:
        return True
    return False


def _map_tensors_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: _map_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_map_tensors_to_device(item, device) for item in value)
    return value


def _install_encoder_cpu_pin(self: Any, encoder_cpu_env_enabled: bool) -> None:
    """Pin the Gemma-4 text encoder on CPU with a device-aware forward wrapper.

    The generic placement path replicates the encoder on every rank's local
    chip; 12B FP16 (23.9 GB) next to the INT8 transformer shard does not fit
    the chip budget.  The wrapper moves tensor inputs to the encoder's device
    for the forward and returns the result on the pipeline device, so the
    downstream connector/transformer path sees NPU tensors as usual.
    """
    encoder = getattr(self, "text_encoder", None)
    if encoder is None or getattr(encoder, "_310p_cpu_pinned", False):
        return
    if not encoder_cpu_env_enabled:
        return

    device = self.device

    encoder.to("cpu")
    encoder._310p_cpu_pinned = True  # type: ignore[attr-defined]

    original_forward = encoder.forward

    def _forward(*args, **kwargs):
        args = _map_tensors_to_device(args, torch.device("cpu"))
        kwargs = _map_tensors_to_device(kwargs, torch.device("cpu"))
        output = original_forward(*args, **kwargs)
        return _map_tensors_to_device(output, device)

    _forward._310p_wrapped = True  # type: ignore[attr-defined]
    encoder.forward = _forward  # type: ignore[method-assign]

    logger.warning(
        "[310P][ltx2.5] Gemma-4 text encoder pinned to CPU via %s "
        "(chain-1 OOM downgrade, chip budget 21589 MB; the encoder is "
        "replicated per rank and 23.9 GB FP16 does not fit next to the "
        "INT8 transformer shard). Per-request encode runs on CPU; the "
        "encode stage is timed separately in the report. Set %s=0 only "
        "together with --enable-layerwise-offload (OffloadPlan streams the "
        "encoder blocks) after validating capacity.",
        _ENCODER_CPU_ENV,
        _ENCODER_CPU_ENV,
    )


def _install_pipeline_init_wrap(encoder_cpu_env_enabled: bool) -> None:
    """Wrap ``LTX2Pipeline.__init__`` so the post-construction pins engage
    before the first request (the runner loads the pipeline right after the
    patch runs).  Idempotent per process."""
    global _PIPELINE_INIT_WRAPPED
    if _PIPELINE_INIT_WRAPPED:
        return

    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

    original_init = LTX2Pipeline.__init__

    def _init_with_310p_pins(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _install_encoder_cpu_pin(self, encoder_cpu_env_enabled)

    LTX2Pipeline.__init__ = _init_with_310p_pins  # type: ignore[assignment]
    _PIPELINE_INIT_WRAPPED = True


def _check_heads_divisibility(od_config: Any) -> None:
    """Strict-Ulysses preflight: head count must divide the ulysses degree."""
    parallel = getattr(od_config, "parallel_config", None)
    ulysses = getattr(parallel, "ulysses_degree", 1) if parallel is not None else 1
    if not ulysses or ulysses <= 1:
        return
    tf_config = getattr(od_config, "tf_model_config", None)
    getter = getattr(tf_config, "get", None)
    if not callable(getter):
        return
    heads = getter("num_attention_heads", None)
    if heads is None:
        return
    if heads % ulysses != 0:
        raise RuntimeError(
            f"strict Ulysses on 310P needs num_attention_heads % ulysses_degree == 0, "
            f"got heads={heads}, ulysses_degree={ulysses}. "
            "Single-chip TP=1 baseline is unaffected."
        )


def _log_placement_plan(od_config: Any) -> None:
    parallel = getattr(od_config, "parallel_config", None)
    tp = getattr(parallel, "tensor_parallel_size", 1) if parallel is not None else 1
    ulysses = getattr(parallel, "ulysses_degree", 1) if parallel is not None else 1
    ring = getattr(parallel, "ring_degree", 1) if parallel is not None else 1
    allgather = getattr(parallel, "allgather_degree", 1) if parallel is not None else 1
    quant = getattr(od_config, "quantization_config", None)
    method = _quantization_method(quant) if quant is not None else "fp16 (no quantization)"
    logger.info(
        "[310P][ltx2.5] dtype=fp16 attention=TORCH_SDPA conv_vae=forced vae_tiling=forced "
        "encoder=%s quantization=%s | parallel: tp=%s ulysses=%s ring=%s allgather=%s "
        "| placement: transformer (19B DiT) sharded across the SP group; "
        "text_encoder (Gemma-4 12B) CPU-pinned or streamed by the offload plan; "
        "connectors/vae/audio_vae/vocoder/duration_head resident per rank; "
        "intra-card SP=2 rides HCCS (~13.4 GB/s per the task-10 topology "
        "table).",
        "cpu" if os.environ.get(_ENCODER_CPU_ENV, "1") == "1" and not _layerwise_offload_enabled(od_config) else "streamed/cpu",
        method,
        tp,
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

    NPUOmniPlatform.get_diffusion_attn_backend_cls = _guarded  # type: ignore[assignment]
    _ATTENTION_GUARD_INSTALLED = True


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all LTX-2.5 310P invariants. Idempotent per process.

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
    _force_conv_vae(od_config)
    _force_vae_tiling(od_config)

    if getattr(od_config, "lora_path", None) is not None:
        logger.warning(
            "[310P][ltx2.5] lora_path=%r is out of scope for the 310P reference "
            "baseline; outputs will not match the reference run weights.",
            od_config.lora_path,
        )

    encoder_cpu_env_enabled = os.environ.get(_ENCODER_CPU_ENV, "1") == "1"
    if not encoder_cpu_env_enabled and not _layerwise_offload_enabled(od_config):
        logger.warning(
            "[310P][ltx2.5] %s=0 without --enable-layerwise-offload: the "
            "replicated 12B encoder will be placed on the NPU and will likely "
            "exceed the 21589 MB chip budget alongside the transformer shard. "
            "Proceeding as configured.",
            _ENCODER_CPU_ENV,
        )
    _install_pipeline_init_wrap(encoder_cpu_env_enabled and not _layerwise_offload_enabled(od_config))

    _check_heads_divisibility(od_config)
    _install_attention_guard()
    _log_placement_plan(od_config)
