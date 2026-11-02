# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch SenseNova-U1.5-8B-MoT for the 310P NPU diffusion path.

Registered in ``vllm_omni.platforms.npu._310p.patch.__init__`` (the 310P
diffusion patch registry, COMPARISON-POLICY.md section 5); ``platform.py``
stays untouched by this branch. Dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
pipeline is constructed.

Model reality (diverges from the queue's "8B fits single chip" assumption;
see report): the checkpoint is 17.53B params / 35.07 GB in bf16 (MoT dual
dense towers - "8B" is per-token active).  FP16 weights alone exceed the
21589 MB single-chip budget, so per POLICY section 2 the sanctioned
primary tier for this model on 310P is **INT8 W8A8** (``DiffusionInt8Config``
online dynamic quantization: ``npu_dynamic_quant`` per-channel weight +
per-token activation, ``npu_quant_matmul`` INT8xINT8->FP16; layers with
output dim > 65535 fall back to unquantized - research/009).

310P invariants enforced here (design converges with the SDXL task-11 and
FLUX.2-klein branches):

* ``od_config.dtype`` must be ``float16`` (310P is FP16-only; the engine
  default bf16 is rejected by hardware with EZ1001).
* Quantization gate: ``method=int8`` accepted as the POLICY section-2 tier
  for this model (with a loud warning); FP8/MXFP* (no 310P benefit),
  ConvRot-INT4 (unvalidated), bitsandbytes (CUDA-only) rejected.  FP16
  (no quant config) on a single chip is rejected at patch time with a
  clear over-budget message instead of dying mid-load.
* The diffusion attention backend resolves to ``TORCH_SDPA`` whenever the
  user did not select one; MindIE-SD-only backends fail at patch time
  (kernels absent on 310P).  The SDPA entry points themselves are swapped
  for the explicit math decomposition by the shared
  ``patch/attention.py`` (fused NPU SDPA is broken on 310P, all dtypes).
* Numerics-changing cache backends are disabled while the 310P path is
  unvalidated (same-seed reference comparison, COMPARISON-POLICY.md
  section 1).
* Pipeline construction runs on CPU (loader ``initialize_model`` wrapped):
  every device-less construction-time op - LM RoPE ``inv_freq``
  (arange/power/division), NEOVisionModel axial RoPE precompute
  (arange/power/division) - would otherwise land on the NPU under
  ``with target_device:`` and hit the 310P CANN 9.1.0 dynamic-TBE compile
  gap (RealDiv family - research/012; upstream 910B CI on the identical
  stack never trips it).  The loader's post-loading path
  (``_process_weights_after_loading``) moves the pipeline to the target
  device afterwards, so final placement is unchanged.
* The flow-matching timestep schedule is computed on CPU
  (``_init_noise_and_schedule`` / ``_apply_time_schedule`` overrides):
  the shift arithmetic ``shift*sigma/(1+(shift-1)*sigma)`` is a tensor
  RealDiv that upstream executes on the NPU; the CPU rewrite keeps the
  exact fp32 op sequence (bit-identical values).
* The per-step denoising division ``(x_pred - z) / (1 - t).clamp_min(t_eps)``
  and the Euler update ``z + (t_next - t) * v_pred`` are rewritten so the
  schedule constants are computed on the CPU.  Upstream, the denominator
  ``(1 - t).clamp_min(t_eps)`` and the delta ``t_next - t`` are fp32 0-dim
  tensors materialized on the NPU (RealDiv / ClipByValue / Sub on-device).
  The rewrite precomputes them as Python floats from the CPU schedule and
  materializes them as fp32 **scalar** tensors.  This is value-preserving
  and bit-exact against upstream (verified by the CPU contract test with
  ``torch.equal``): ATen keeps the result dtype FP16 in both forms because a
  0-dim/scalar denominator does not promote the fp16 numerator, and the
  denominator value is identical.  The NPU op list shrinks from
  ``Sub + Clamp + RealDiv(fp16, fp32_0dim)`` to a single static-shape
  ``RealDiv(fp16_tensor, fp32_scalar)`` - the dynamic-shape RealDiv/ClipByValue
  ops that trip the 310P CANN 9.1.0 TBE gap (research/012) are never
  dispatched, while the arithmetic values are unchanged.

Component placement (logged at startup):

* ``language_model`` (SenseNovaU1ForCausalLM, 42 layers, dual MoT towers)
  and the fm modules: resident on the single chip.  Under INT8 the dense
  linears are quantized online; ``lm_head`` (151936 > 65535) and the token
  embeddings stay FP16.
* ``vision_model`` (NEOVisionModel, understanding side) and
  ``fm_modules["vision_model_mot_gen"]``: FP16, replicated on the chip.
  OOM downgrade chain (lead-fixed order):
  1. ``VLLM_OMNI_310P_SENSENOVA_MOTGEN_CPU=1`` - pin the gen-side ViT to
     host memory after weight load (t2i does not forward it at all; saves
     ~1.1 GiB resident).  Breaks it2i if used.
  2. ``VLLM_OMNI_310P_SENSENOVA_VISION_CPU=1`` - pin the understanding
     side ViT to host memory (img2text pays one H2D per request).
  3. Single-chip memory tier when resident fp16 weights exceed the
     ~21.6 GiB budget AND INT8 is unavailable: launch with the framework's
     ``--enable-layerwise-offload``.  The pipeline declares
     ``_dit_modules = ["language_model"]`` (the MoT LM streams block-by-
     block from host to the chip; ViT/fm_modules stay resident), so only
     one LM block plus the resident encoders occupy device memory.  INT8
     itself is infeasible on this 310P build: ``npu_quant_matmul``
     (aclnnQuantMatmulV5) rejects the INT8 W8A8 recipe (per-token
     activation scale unsupported on inference-series cards; even the
     minimal no-pertoken form fails ACLNN_ERR_PARAM_INVALID 161002 -
     task-17 report, section 量化配置).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: ``od_config.model_class_name`` value for the SenseNova-U1.5 pipeline.
SENSENOVA_U15_ARCH = "SenseNovaU1Pipeline"

#: Backends whose NPU kernels ship only with MindIE-SD (absent on 310P).
_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

#: OOM downgrade chain env vars (see module docstring).
_VISION_CPU_ENV = "VLLM_OMNI_310P_SENSENOVA_VISION_CPU"
_MOTGEN_CPU_ENV = "VLLM_OMNI_310P_SENSENOVA_MOTGEN_CPU"
_LM_CPU_ENV = "VLLM_OMNI_310P_SENSENOVA_LM_CPU"

_APPLIED = False
_LOADER_CPU_BUILD_INSTALLED = False
_VISION_PIN_INSTALLED = False
_ATTENTION_GUARD_INSTALLED = False


def _force_fp16(od_config: Any) -> None:
    dtype = getattr(od_config, "dtype", None)
    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}); "
        "310P is FP16-only (no BF16; EZ1001 on DT_BFLOAT16). "
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
            "[310P][sensenova-u1.5] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][sensenova-u1.5] disabling numerics-changing cache backend %r on 310P "
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


def _offload_active(od_config: Any) -> bool:
    """True when the framework's diffusion CPU offload owns weight residency.

    With ``--enable-layerwise-offload`` the loader loads to host and the
    offload backend streams the declared dit component (here:
    ``language_model``) block-by-block, so the single-chip FP16 residency
    budget no longer applies to the LM weights.
    """
    try:
        from vllm_omni.diffusion.offloader.config import offload_enabled
    except Exception:  # pragma: no cover - import guard
        return False
    try:
        return bool(offload_enabled(od_config))
    except Exception:  # pragma: no cover - defensive
        return False


def _validate_quantization(od_config: Any) -> None:
    """Precision-tier gate for SenseNova-U1.5 on 310P (POLICY section 2).

    FP16 is the primary tier in general, but for THIS model the FP16
    weights (35.07 GB) exceed the 21589 MB single-chip budget.  Two
    sanctioned single-chip paths exist: INT8 W8A8 (research/009: online
    dynamic INT8 via npu_quant_matmul; layers wider than 65535 fall back
    to unquantized) or FP16 under the framework's layerwise offload
    (``--enable-layerwise-offload``: the MoT LM streams from host).  NOTE:
    on the 127 C4 310P (CANN 9.1.0) the INT8 op turned out to be infeasible
    (npu_quant_matmul / aclnnQuantMatmulV5 rejects the W8A8 recipe with
    ACLNN_ERR_PARAM_INVALID 161002 - task-17 report, section 量化配置), so
    offload is the operative single-chip tier there.  Single-chip FP16
    (no quant config, no offload) is rejected at patch time rather than
    dying mid-load with an allocator OOM.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        parallel = getattr(od_config, "parallel_config", None)
        tp = getattr(parallel, "tensor_parallel_size", 1) if parallel is not None else 1
        sp = getattr(parallel, "sequence_parallel_size", 1) if parallel is not None else 1
        if tp * max(sp or 1, 1) <= 1 and not _offload_active(od_config):
            raise RuntimeError(
                "SenseNova-U1.5-8B-MoT on 310P single chip cannot run the plain FP16 "
                "tier: the checkpoint is 17.53B params / 35.07 GB (bf16 shards), above "
                "the 21589 MB chip budget before activations.  Use either "
                "quantization_config method='int8' (DiffusionInt8Config, online dynamic "
                "quantization) or the framework's --enable-layerwise-offload (FP16 "
                "weights stream from host; required on 127 C4 where the INT8 op is "
                "infeasible - see report).  A two-chip FP16 split is out of scope "
                "for this task (TP1-only GO)."
            )
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][sensenova-u1.5] quantization_config method=%r accepted as the "
            "POLICY section-2 tier for this model (FP16 single-chip is over budget, "
            "35.07 GB > 21589 MB): online dynamic INT8 via npu_quant_matmul, layers "
            "with output dim > 65535 (lm_head 151936) fall back to unquantized "
            "(mixed-precision semantics). Precision evidence vs the FP16 baseline is "
            "MANDATORY in the real-device report.",
            method,
        )
        return
    raise RuntimeError(
        f"310P SenseNova-U1.5 rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): INT8 W8A8 (method='int8') is the "
        "sanctioned tier for this model on 310P (POLICY section 2, research/009). "
        "FP8/MXFP* give no hardware benefit on 310P (served with FP16 matmul); "
        "ConvRot-INT4 is unvalidated; bitsandbytes is CUDA-only."
    )


def _flip_int8_load_device(model: Any, target_device: torch.device) -> int:
    """Point online-INT8 layers' ``_load_device`` at the target device.

    Companion to ``_install_loader_cpu_build``: construction runs under
    ``torch.device("cpu")`` (310P construction-time op safety), but online
    INT8 quantization (``npu_dynamic_quant``) has no CPU kernel, so the
    layers it owns must still materialize their weights on the accelerator,
    exactly as in the upstream non-310P flow.  Returns the flipped count.
    """
    from vllm_omni.quantization.int8_config import UnquantizedHostLinearMethod

    flipped = 0
    for module in model.modules():
        load_device = getattr(module, "_load_device", None)
        if load_device is None:
            continue
        if str(load_device) != "cpu":
            continue
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        if isinstance(quant_method, UnquantizedHostLinearMethod):
            # Over-wide host fallback: stays in host memory on purpose.
            continue
        module._load_device = torch.device(str(target_device))
        flipped += 1
    if flipped:
        logger.info(
            "[310P][sensenova-u1.5] flipped _load_device of %d online-INT8 "
            "layers to %s (npu_dynamic_quant has no CPU kernel; over-wide "
            "host fallback layers untouched)",
            flipped,
            target_device,
        )
    return flipped


def _install_loader_cpu_build() -> None:
    """Run pipeline construction on CPU (310P dynamic-TBE gap, research/012).

    The ``"default"`` load-format branch of the diffusers loader builds the
    pipeline inside ``with target_device:`` (npu:0).  Every device-less op
    at construction time (LM RoPE ``inv_freq`` arange/power/division,
    NEOVisionModel axial-RoPE precompute, TimestepEmbedder ``freqs``) would
    land on the NPU and hit the 310P CANN 9.1.0 dynamic-variant compile
    gap.  Mirrors the SDXL/Qwen-Image-2.1 branches: ``initialize_model``
    runs under ``torch.device("cpu")``; the loader's post-loading path
    (``_process_weights_after_loading``) moves the pipeline to the target
    device afterwards, so final placement is unchanged.
    """
    global _LOADER_CPU_BUILD_INSTALLED
    if _LOADER_CPU_BUILD_INSTALLED:
        return

    loader_mod = importlib.import_module("vllm_omni.diffusion.model_loader.diffusers_loader")
    if getattr(loader_mod, "_310p_cpu_build_patched", False):
        _LOADER_CPU_BUILD_INSTALLED = True
        return

    orig_initialize_model = loader_mod.initialize_model

    def _initialize_model_on_cpu(od_config: Any) -> Any:
        with torch.device("cpu"):
            return orig_initialize_model(od_config)

    loader_mod.initialize_model = _initialize_model_on_cpu
    loader_mod._310p_cpu_build_patched = True  # type: ignore[attr-defined]
    logger.info(
        "[310P][sensenova-u1.5] pipeline construction pinned to CPU "
        "(RoPE/vision-RoPE build-time arange/power/division ops would hit the "
        "310P dynamic-TBE RealDiv gap; loader post-load path restores NPU placement)"
    )

    # INT8 online-quantization companion fix: the CPU pin above must NOT leak
    # into the weight-load device.
    #
    # The online INT8 linear method materializes each layer's weight at first
    # load on ``layer._load_device``, which ``create_weights`` captures from
    # ``torch.get_default_device()`` - so under the pin above it comes out as
    # CPU.  ``npu_dynamic_quant`` (online W8A8) has no CPU kernel (torch_npu
    # 2.10: NotImplementedError 'Could not run npu::npu_dynamic_quant with
    # arguments from the CPU backend'), so ``load_weights`` dies before the
    # first shard completes.  The upstream non-310P flow builds the model
    # under ``with target_device:`` and ``_load_device`` is the NPU - restore
    # exactly that for the INT8-target layers right after construction.  The
    # over-wide host-fallback layers (UnquantizedHostLinearMethod, output dim
    # > 65535, e.g. lm_head 151936) deliberately keep their host placement.
    loader_cls = getattr(loader_mod, "DiffusersPipelineLoader", None)
    if loader_cls is not None and not getattr(loader_cls, "_310p_int8_load_device_patched", False):
        orig_init_from_load_format = loader_cls._init_from_load_format

        def _init_from_load_format_with_int8_fix(
            self: Any,
            load_format: Any,
            target_device: Any,
            custom_pipeline_name: Any = None,
            is_hsdp: bool = False,
        ) -> Any:
            model = orig_init_from_load_format(
                self, load_format, target_device, custom_pipeline_name, is_hsdp
            )
            if getattr(target_device, "type", None) == "npu":
                _flip_int8_load_device(model, target_device)
            return model

        loader_cls._init_from_load_format = _init_from_load_format_with_int8_fix  # type: ignore[method-assign]
        loader_cls._310p_int8_load_device_patched = True  # type: ignore[attr-defined]

    if loader_cls is not None:
        # Post-load placement fix (companion to the CPU construction pin):
        # the non-HSDP "default" load path performs NO whole-model
        # ``model.to(target_device)`` - upstream it is unnecessary because
        # construction happens under ``with target_device:`` (every param
        # starts on the accelerator).  Under the CPU pin, every NON-INT8
        # param (token/lm_head embeddings, the whole NEOVision tower, VAE,
        # over-wide fp16 fallback linears, build-time buffers) stays host
        # resident and the first forward dies with "indices is on npu:0,
        # different from other tensors on cpu" (vocab_parallel_embedding).
        # Restore the invariant: move what is still on host right before the
        # quantization sweep (which then runs exactly as upstream).  INT8
        # layers are already NPU-resident via the flipped _load_device.
        orig_process_after = loader_cls._process_weights_after_loading

        def _process_after_with_placement(self: Any, model: Any, target_device: Any) -> None:
            if getattr(target_device, "type", None) == "npu":
                n_cpu = sum(1 for param in model.parameters() if param.device.type == "cpu")
                if n_cpu:
                    model.to(target_device)
                    logger.info(
                        "[310P][sensenova-u1.5] moved %d host-resident params to %s "
                        "post-load (CPU construction pin; INT8 layers already "
                        "accelerator-resident)",
                        n_cpu,
                        target_device,
                    )
            orig_process_after(self, model, target_device)

        if not getattr(loader_cls, "_310p_placement_patched", False):
            loader_cls._process_weights_after_loading = _process_after_with_placement  # type: ignore[method-assign]
            loader_cls._310p_placement_patched = True  # type: ignore[attr-defined]

    _LOADER_CPU_BUILD_INSTALLED = True


def _install_vision_cpu_pin() -> None:
    """Env-gated OOM downgrade: pin NEOVision modules and/or the MoT LM to
    host memory.

    Runs AFTER the loader's post-loading placement (which lands everything
    on the chip), so the pin is a post-hoc D2H move of already-loaded
    weights.  Default off: both ViTs and the LM stay on-chip, matching the
    generic 310P path (COMPARISON-POLICY.md section 1 level 2 parity).
    """
    global _VISION_PIN_INSTALLED
    if _VISION_PIN_INSTALLED:
        return
    if (
        os.environ.get(_VISION_CPU_ENV, "0") != "1"
        and os.environ.get(_MOTGEN_CPU_ENV, "0") != "1"
        and os.environ.get(_LM_CPU_ENV, "0") != "1"
    ):
        return

    loader_mod = importlib.import_module("vllm_omni.diffusion.model_loader.diffusers_loader")
    loader_cls = getattr(loader_mod, "DiffusersPipelineLoader", None)
    if loader_cls is None:
        logger.warning(
            "[310P][sensenova-u1.5] %s requested but DiffusersPipelineLoader not found; pin skipped.",
            (_VISION_CPU_ENV, _MOTGEN_CPU_ENV, _LM_CPU_ENV),
        )
        return

    original = loader_cls._process_weights_after_loading

    def _post_load_with_pin(self: Any, model: Any, target_device: Any) -> None:
        original(self, model, target_device)
        if os.environ.get(_MOTGEN_CPU_ENV, "0") == "1" and "vision_model_mot_gen" in getattr(model, "fm_modules", {}):
            model.fm_modules["vision_model_mot_gen"] = model.fm_modules["vision_model_mot_gen"].to("cpu")
            logger.warning(
                "[310P][sensenova-u1.5] fm_modules['vision_model_mot_gen'] pinned to CPU via %s "
                "(t2i does not forward it; it2i is broken in this mode).",
                _MOTGEN_CPU_ENV,
            )
        if os.environ.get(_VISION_CPU_ENV, "0") == "1" and getattr(model, "vision_model", None) is not None:
            model.vision_model = model.vision_model.to("cpu")
            logger.warning(
                "[310P][sensenova-u1.5] vision_model pinned to CPU via %s "
                "(img2text pays one H2D transfer per request).",
                _VISION_CPU_ENV,
            )
        if os.environ.get(_LM_CPU_ENV, "0") == "1" and getattr(model, "language_model", None) is not None:
            model.language_model = model.language_model.to("cpu")
            logger.warning(
                "[310P][sensenova-u1.5] language_model pinned to CPU via %s "
                "(MoT LM resident fp16 exceeds the single-chip budget; "
                "denoising and AR decode run at CPU speed).",
                _LM_CPU_ENV,
            )

    loader_cls._process_weights_after_loading = _post_load_with_pin  # type: ignore[method-assign]
    _VISION_PIN_INSTALLED = True


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
    def _guarded(cls: Any, selected_backend: Any, head_size: Any, allow_trtllm_default: bool = True):  # noqa: ANN001
        if selected_backend is None:
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
        if str(selected_backend).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention backend {selected_backend!r} requires MindIE-SD "
                "kernels, which are unavailable on 310P; use TORCH_SDPA."
            )
        return original(cls, selected_backend, head_size, allow_trtllm_default=allow_trtllm_default)

    NPUOmniPlatform.get_diffusion_attn_backend_cls = _guarded  # type: ignore[method-assign]
    _ATTENTION_GUARD_INSTALLED = True


def _log_placement_plan(od_config: Any) -> None:
    parallel = getattr(od_config, "parallel_config", None)
    tp = getattr(parallel, "tensor_parallel_size", 1) if parallel is not None else 1
    sp = getattr(parallel, "sequence_parallel_size", None) if parallel is not None else None
    quant = getattr(od_config, "quantization_config", None)
    logger.info(
        "[310P][sensenova-u1.5] dtype=fp16 attention=TORCH_SDPA quant=%s | parallel: tp=%s sp=%s | "
        "placement: language_model + fm_modules resident on the single chip; "
        "vision_model + fm_modules['vision_model_mot_gen'] FP16 on-chip "
        "(OOM downgrade chain: %s then %s). "
        "TP=1 only (task GO); two-chip FP16 split is out of scope.",
        _quantization_method(quant) if quant is not None else "none",
        tp,
        sp,
        _MOTGEN_CPU_ENV,
        _VISION_CPU_ENV,
    )


# ---------------------------------------------------------------------------
# 310P pipeline subclass: CPU-computed schedule + static-shape denoising ops
# ---------------------------------------------------------------------------


class _SenseNovaU15Overrides:
    """Mixin holding the 310P arithmetic rewrites (MRO first, so its methods
    win over the upstream ``SenseNovaU1Pipeline``).

    Every overridden method is a verbatim copy of the tag 5917730 upstream
    implementation with the marked lines changed; the changes are
    value-preserving (see module docstring): fp32 schedule arithmetic moves
    to CPU bit-identically, and the per-step division/Euler-update
    denominators are materialized as fp32 ``(1,)`` tensors so the NPU never
    sees a dynamic-shape RealDiv/ClipByValue or a 0-dim-tensor Mul.
    """

    def _apply_time_schedule(self, t: torch.Tensor, image_seq_len: int, timestep_shift: float) -> torch.Tensor:
        # Upstream computes the shift division on the NPU (tensor RealDiv);
        # 310P CANN 9.1.0 fails dynamic-shape RealDiv (research/012).  Same
        # fp32 op sequence on CPU - bit-identical values.
        t_cpu = t.detach().cpu().float()
        sigma = 1.0 - t_cpu
        shift = timestep_shift
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        return (1.0 - sigma).to(device=t.device, dtype=t.dtype)

    def _init_noise_and_schedule(self, p: Any) -> Any:
        ns = super()._init_noise_and_schedule(p)
        # Rebuild the timestep schedule on CPU (upstream builds it on-device
        # via linspace + RealDiv shift); values are bit-identical fp32.
        n = ns.timesteps.shape[0]
        ns.timesteps = self._apply_time_schedule(
            torch.linspace(0.0, 1.0, n, device="cpu"),
            ns.token_h * ns.token_w,
            p.timestep_shift,
        ).to(device=self.device)
        # CPU floats of the (device-resident) schedule, used to materialize
        # the per-step denominators/deltas below.  One D2H copy per request.
        ns.t_floats = [float(x) for x in ns.timesteps.detach().cpu()]
        return ns

    def _prepare_input_images(self, images: list) -> Any:
        """310P: pixel_values take the serving dtype, not hardcoded bf16.

        Upstream (pipeline_sensenova_u1.py ``_prepare_input_images``) ends
        with ``pv.to(self.device, dtype=torch.bfloat16)`` - fine on A800s,
        fatal on 310P where every op rejects DT_BFLOAT16 (first observed as
        ``AclNN_Parameter_Error(EZ1001): tensor 0 not implemented for
        DT_BFLOAT16`` from the cat kernel inside the startup dummy run, and
        it would equally kill the real img2img/img2text paths).  The model
        loads in the configured dtype (fp16 on 310P), so pixel values must
        match it.  Falls back to bf16 only when the serving dtype is
        unavailable (upstream behavior).
        """
        from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import _load_image_native

        pixel_values_list = []
        grid_hw_list = []
        num_images = len(images)
        max_pixels_per_image = min(2048 * 2048, (4096 * 4096) // num_images)
        pixel_dtype = getattr(getattr(self, "od_config", None), "dtype", None)
        if pixel_dtype is None:
            vision_model = getattr(self, "vision_model", None)
            try:
                pixel_dtype = next(vision_model.parameters()).dtype if vision_model is not None else None
            except StopIteration:
                pixel_dtype = None
        if pixel_dtype is None:
            pixel_dtype = torch.bfloat16
        for img in images:
            pv, ghw = _load_image_native(
                img,
                patch_size=self.vis_cfg.patch_size,
                downsample_ratio=self.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=max_pixels_per_image,
            )
            pixel_values_list.append(pv.to(self.device, dtype=pixel_dtype))
            grid_hw_list.append(ghw.to(self.device))
        return torch.cat(pixel_values_list), torch.cat(grid_hw_list)

    def _t2i_predict_v(
        self,
        input_embeds: torch.Tensor,
        indexes_image: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_values: Any,
        t: torch.Tensor,
        z: torch.Tensor,
        image_token_num: int,
        t_eps: float,
        image_size: Any = None,
        cache_dit_skip: bool = False,
        **_kw: Any,
    ) -> torch.Tensor:
        B, L = z.shape[0], z.shape[1]
        denoising_model = self.language_model if cache_dit_skip else self.denoising_transformer
        outputs = denoising_model(
            inputs_embeds=input_embeds,
            exist_und=False,
            exist_gen=True,
            indexes=indexes_image,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            cache_dit_skip=cache_dit_skip,
            use_cache=True,
            compute_logits=False,
        )
        last_hidden_state = outputs.hidden_states

        if self.model_cfg.use_pixel_head:
            merge_size = self.merge_size
            token_h = image_size[1] // (self.patch_size * merge_size)
            token_w = image_size[0] // (self.patch_size * merge_size)
            img_reshaped = last_hidden_state[:, -image_token_num:].view(B, token_h, token_w, -1)
            img_2d = torch.einsum("b h w c -> b c h w", img_reshaped).contiguous()
            smoothed_img_2d = self.fm_modules["fm_head"](img_2d)
            smoothed_reshaped = smoothed_img_2d.view(
                B, 3, token_h, self.patch_size * merge_size, token_w, self.patch_size * merge_size
            )
            smoothed_reshaped = torch.einsum("b c h p w q -> b h w p q c", smoothed_reshaped)
            x_pred = smoothed_reshaped.contiguous().view(
                B, L, self.patch_size * merge_size * self.patch_size * merge_size * 3
            )
        else:
            x_pred = self.fm_modules["fm_head"](last_hidden_state[:, -image_token_num:].view(B, L, -1)).view(B, L, -1)

        # Upstream:  v_pred = (x_pred - z) / (1 - t).clamp_min(t_eps)
        # 310P rewrite: the denominator (fp32 0-dim upstream, built on-device
        # by Sub + Clamp) is computed on the CPU from the schedule float and
        # materialized as an fp32 scalar tensor.  Bit-exact vs upstream
        # (CPU contract test): 1.0 - t and max(., t_eps) are exact in f64
        # for fp32 inputs, and ATen keeps the division result FP16 in both
        # forms (a 0-dim scalar does not promote the fp16 numerator).
        t_f = float(t)
        denom = max(1.0 - t_f, float(t_eps))
        denom_t = torch.tensor(denom, device=z.device, dtype=torch.float32)
        v_pred = (x_pred - z) / denom_t
        return v_pred

    def _run_denoising_loop(self, ns: Any, caches: dict, p: Any, think_text: str = "", is_it2i: bool = False) -> Any:
        # Upstream verbatim (tag 5917730), two lines changed: the Euler
        # update delta ``(t_next - t)`` (fp32 0-dim Sub upstream) is taken
        # from the CPU schedule floats and materialized as an fp32 (1,)
        # tensor.  The update's type promotion (fp16 z + fp32 delta * fp32
        # v_pred -> fp32) is unchanged.
        from vllm_omni.diffusion.data import DiffusionOutput
        from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
            VisionGrid,
            _patchify,
            _to_pil,
            _unpatchify,
        )
        from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import clear_flash_kv_cache

        merge_size = self.merge_size
        image_prediction = ns.image_prediction
        grid = VisionGrid.from_tensor(ns.grid_hw)
        times = ns.timesteps[:-1]
        if is_it2i:
            cfg_steps = ((times > p.cfg_interval[0]) & (times < p.cfg_interval[1])) | (p.cfg_interval[0] == 0)
            cfg_steps &= p.cfg_scale != 1 or p.img_cfg_scale != 1
        else:
            cfg_steps = (times >= p.cfg_interval[0]) & (times <= p.cfg_interval[1])
            cfg_steps &= p.cfg_scale > 1
        cfg_steps = cfg_steps.tolist()

        noise_scale_emb = None
        if self.model_cfg.add_noise_scale_embedding:
            ns_tensor = torch.full(
                (p.batch_size * ns.token_h * ns.token_w,),
                ns.noise_scale / self.model_cfg.noise_scale_max_value,
                dtype=ns.timesteps.dtype,
                device=ns.timesteps.device,
            )
            noise_scale_emb = self.fm_modules["noise_scale_embedder"](ns_tensor).view(
                p.batch_size, ns.token_h * ns.token_w, -1
            )

        for step_i in range(p.num_steps):
            t = ns.timesteps[step_i]
            t_next = ns.timesteps[step_i + 1]

            z = _patchify(image_prediction, self.patch_size * merge_size)
            image_input = _patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self._extract_feature(
                image_input.view(p.batch_size * ns.grid_h * ns.grid_w, -1),
                gen_model=True,
                grid_hw=ns.grid_hw,
                grid=grid,
            ).view(p.batch_size, ns.token_h * ns.token_w, -1)

            t_expanded = t.expand(p.batch_size * ns.token_h * ns.token_w)
            timestep_embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(
                p.batch_size,
                ns.token_h * ns.token_w,
                -1,
            )
            if noise_scale_emb is not None:
                timestep_embeddings = timestep_embeddings + noise_scale_emb
            image_embeds = image_embeds + timestep_embeddings

            v_pred = self._denoise(
                image_prediction, ns, t, z, image_embeds, caches, p, step_i, is_it2i, cfg_steps[step_i]
            )
            # Upstream:  z = z + (t_next - t) * v_pred
            # 310P rewrite: the delta (fp32 0-dim Sub upstream, on-device)
            # comes from the CPU schedule floats as an fp32 scalar tensor.
            # Bit-exact (CPU contract test): same fp32 value, same fp16
            # result dtype (scalar does not promote).
            dt_f = ns.t_floats[step_i + 1] - ns.t_floats[step_i]
            dt_t = torch.tensor(dt_f, device=z.device, dtype=torch.float32)
            z = z + dt_t * v_pred
            image_prediction = _unpatchify(z, self.patch_size * merge_size, p.image_size[1], p.image_size[0])

        # Cleanup KV caches
        for key in ("cond", "uncond", "img_cond"):
            if key in caches and not isinstance(caches[key], dict):
                clear_flash_kv_cache(caches[key])

        images = _to_pil(image_prediction)
        img = images[0] if images else None
        metadata = {}
        if think_text:
            metadata["text"] = {"think_text": think_text}
        return DiffusionOutput(
            output={
                "payload": {"image": img},
                "metadata": metadata,
            }
        )


def _gate_stub_triton_kernel() -> None:
    """Gate the fused qk-norm+RoPE Triton entry when the container's triton
    package is a stub (127 C4 reality check).

    The 127 C4 container ships no real triton package: vllm.triton_utils
    synthesizes a stub module whose `jit` is an identity decorator, so
    qk_norm_rope_kernel stays a plain Python function and `kernel[grid]`
    raises TypeError at RUNTIME (127 C4: dummy it2i warmup died in
    _project_and_rope -> fused_rmsnorm_rope triton_qk_norm_rope L200).
    The upstream availability check only catches ImportError, so gate the
    module-level entry here: None makes the transformer fall through to its
    PyTorch qk-norm+RoPE path (the author's intended "unavailable" branch).
    """
    name = "vllm_omni.diffusion.models.sensenova_u1.fused_rmsnorm_rope"
    module = sys.modules.get(name)
    if module is None:
        try:
            module = importlib.import_module(name)
        except ImportError:  # contract-test environment (module faked/absent)
            return
    if inspect.isfunction(getattr(module, "qk_norm_rope_kernel", None)):
        module.triton_qk_norm_rope = None


def _build_310p_pipeline_cls():
    """Materialize the 310P subclass at apply time (the heavy diffusion
    imports stay out of module import so contract tests run CPU-only)."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline
    from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
        SenseNovaU1ForCausalLM,
    )

    # The MoT LM declares no _layerwise_offload_blocks_attrs upstream, so the
    # layerwise offload backend finds no streamable blocks and falls back to
    # component.module.to(device) on the WHOLE 17.53B LM (35.07 GB) - an
    # immediate single-chip OOM ("Tried to allocate 194 MiB; 19.51 GiB in
    # use", 127 C4).  Declare the 42-layer MoT stack so the backend streams
    # one layer at a time (weights resident on host).  Discovery resolves the
    # name via a single-level getattr (upstream precedent: minimax_h3 "blocks",
    # cosmos3 "layers"), so expose the nested self.model.layers as a class
    # property named "layers" (the wrapper carries no own "layers" attr).
    if not getattr(SenseNovaU1ForCausalLM, "_layerwise_offload_blocks_attrs", None):
        if "layers" not in vars(SenseNovaU1ForCausalLM):
            SenseNovaU1ForCausalLM.layers = property(  # type: ignore[prop-deletion]
                lambda self: self.model.layers
            )
        SenseNovaU1ForCausalLM._layerwise_offload_blocks_attrs = ["layers"]

    _gate_stub_triton_kernel()

    cls = type(
        "_SenseNovaU15Pipeline310P",
        (_SenseNovaU15Overrides, SenseNovaU1Pipeline),
        {"__module__": __name__},
    )
    globals()["_SenseNovaU15Pipeline310P"] = cls  # registry name lookup
    return cls


def get_sensenova_u1_post_process_func(od_config: Any):  # noqa: D401 - re-export
    """Re-exported from the pipeline module for the registry override."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        get_sensenova_u1_post_process_func as _upstream,
    )

    return _upstream(od_config)


__all__ = ["apply_diffusion_310p_patches", "get_sensenova_u1_post_process_func", "_SenseNovaU15Pipeline310P"]


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all SenseNova-U1.5 310P invariants.  Idempotent per process.

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
    _install_loader_cpu_build()
    _install_vision_cpu_pin()
    _install_attention_guard()
    _build_310p_pipeline_cls()
    from vllm_omni.diffusion.registry import register_diffusion_model

    register_diffusion_model(
        SENSENOVA_U15_ARCH,
        module_name=__name__,
        class_name="_SenseNovaU15Pipeline310P",
        post_process_func_name="get_sensenova_u1_post_process_func",
    )
    _log_placement_plan(od_config)
    logger.info("[310P][sensenova-u1.5] 310P patches installed (CPU build pin, CPU schedule, static-shape denoising ops)")
