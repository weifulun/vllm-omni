# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"

# Diffusion 310P patch registry (COMPARISON-POLICY.md §5):
# od_config.model_class_name -> patch modules applied (in order) before the
# diffusion pipeline is constructed.  Each model branch registers itself here
# with a single line and keeps its logic in its own patch module, which must
# expose apply_diffusion_310p_patches(od_config).  platform.py keeps calling
# apply_diffusion_310p_patches() unchanged.
_DIFFUSION_310P_PATCH_MODULES: dict[str, list[str]] = {
    "StableDiffusionXLPipeline": ["vllm_omni.platforms.npu._310p.patch.sdxl"],
    "ZImagePipeline": ["vllm_omni.platforms.npu._310p.patch.z_image"],
}


def apply_patches() -> None:
    global _WORKER_PATCHED

    if _WORKER_PATCHED:
        return

    from vllm_omni.platforms.npu._310p.patch.worker import apply_patch

    apply_patch()
    _WORKER_PATCHED = True


def _maybe_skip_startup_warmup() -> None:
    """Skip the engine's 512x512 dummy warmup on 310P (research/012 class).

    The generic ``DiffusionEngine._dummy_run`` warms up at a fixed 512x512
    geometry.  On 310P (CANN 9.1.0), the conv sub-graphs that geometry
    generates fail TBE dynamic-variant pre-compilation (MemSet via
    ops_legacy/dynamic/mem_set.py) even though the real request geometry
    compiles and runs.  ``VLLM_OMNI_310P_SKIP_WARMUP=1`` no-ops the dummy
    run; kernels are then pre-compiled on the first real request.
    """
    if os.environ.get("VLLM_OMNI_310P_SKIP_WARMUP", "0") != "1":
        return
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    if getattr(DiffusionEngine, "_310p_warmup_skipped", False):
        return

    def _dummy_run_skipped(self):
        logger.info(
            "310P: skipping startup dummy warmup (VLLM_OMNI_310P_SKIP_WARMUP=1); first request pre-compiles kernels"
        )

    DiffusionEngine._dummy_run = _dummy_run_skipped
    DiffusionEngine._310p_warmup_skipped = True  # type: ignore[attr-defined]


def apply_diffusion_310p_patches(od_config) -> None:
    """Apply the 310P diffusion runtime patches for ``od_config``'s pipeline.

    Dispatches through ``_DIFFUSION_310P_PATCH_MODULES`` so model branches
    never touch platform.py.  Pipelines without a registered 310P patch are a
    no-op (they run on the generic 310P diffusion path).

    ``VLLM_OMNI_310P_DIFFUSION_PATCHES=0`` disables the 310P model patches
    (dtype forcing, VAE tiling, CPU encoder pinning) only: it is the
    comparison hook for COMPARISON-POLICY.md §1 level 2 (run the generic
    310P path with an explicit FP16 config as the same-seed reference).
    The MATH-SDPA hardware shim, the HF encoder dtype pin and the startup
    warmup skip are NOT part of that gate — the fused NPU SDPA kernel, BF16
    elementwise kernels and the 512x512 dummy warmup are all broken on 310P
    at the hardware level, so those shims apply to every 310P diffusion run
    (including the generic level-2 reference) and are toggled only by their
    own knobs (``VLLM_OMNI_310P_FORCE_MATH_SDPA``,
    ``VLLM_OMNI_310P_SKIP_WARMUP``).
    """
    from vllm_omni.platforms.npu._310p.patch.attention import maybe_patch_sdpa_math

    maybe_patch_sdpa_math()
    _pin_hf_encoder_dtype()
    _offload_hf_text_encoder()
    # Hardware-level too: the engine's 512x512 dummy warmup generates conv
    # sub-graphs that fail 310P TBE dynamic-variant pre-compilation (MemSet,
    # research/012) or OOM on the chip (aclnnDivs, Z-Image level-2 probe),
    # while the real request geometry compiles and runs.  Must be outside the
    # model-patch gate so generic level-2 runs can start up at all.
    _maybe_skip_startup_warmup()

    if os.environ.get("VLLM_OMNI_310P_DIFFUSION_PATCHES", "1") == "0":
        return

    model_arch = getattr(od_config, "model_class_name", None)
    module_names = _DIFFUSION_310P_PATCH_MODULES.get(model_arch)
    if not module_names:
        return

    import importlib

    for module_name in module_names:
        module = importlib.import_module(module_name)
        module.apply_diffusion_310p_patches(od_config)


def _pin_hf_encoder_dtype() -> None:
    """310P hardware invariant: HF text encoders must not be built in BF16.

    Official checkpoints commonly ship ``config.json`` with
    ``torch_dtype: bfloat16`` (e.g. the Z-Image Qwen3 text encoder).
    transformers ``_from_config`` honours the config dtype over the global
    default, and the loader's device context places the resulting parameter
    initializers (RMSNorm ``torch.ones``) on the NPU, where aclnn has no BF16
    elementwise support (``aclnnInplaceOne`` -> "Tensor self not implemented
    for DT_BFLOAT16").  Wrap the diffusion loader's
    ``create_transformers_model`` so the encoder is always constructed and
    materialised in FP16, the only 310P runtime dtype.  Decoupled from the
    model patch gate by design (hardware-level, like the MATH-SDPA shim).
    """
    import torch

    import vllm_omni.diffusion.models.utils as _dmu

    if getattr(_dmu.create_transformers_model, "_310p_dtype_pinned", False):
        return

    original = _dmu.create_transformers_model

    def _pinned(auto_cls, od_config, hf_config, dtype=None, device=None):
        try:
            hf_config.torch_dtype = torch.float16
        except Exception:  # noqa: BLE001 - config objects vary
            pass
        return original(auto_cls, od_config, hf_config, dtype=torch.float16, device=device)

    _pinned._310p_dtype_pinned = True  # type: ignore[attr-defined]
    _dmu.create_transformers_model = _pinned  # type: ignore[assignment]

    # Pipeline modules that bind ``create_transformers_model`` into their own
    # namespace at import time (before this shim runs) keep the original
    # binding; rebind them explicitly.  Extend as model branches land.
    import importlib

    for module_name in (
        "vllm_omni.diffusion.models.z_image.pipeline_z_image",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if getattr(module.create_transformers_model, "_310p_dtype_pinned", False):
            continue
        module.create_transformers_model = _pinned  # type: ignore[assignment]

    logger.info("[310P] HF text-encoder construction pinned to float16 (no BF16 on aclnn)")


def _offload_hf_text_encoder() -> None:
    """310P single-chip memory invariant: HF text encoders do not live on the NPU.

    At 1024x1024 the Z-Image Qwen3-4B encoder (8GiB FP16) plus the 6.15B
    transformer (11.43GiB) plus the VAE already leave no headroom for DiT
    activations on the 21.08GiB chip (measured: generic level-2 run OOMs with
    19.47GiB allocated).  Moving the text encoder to the CPU host is therefore
    a memory invariant of 310P diffusion, not a model optimization: it applies
    to every 310P diffusion pipeline after construction (the model patch suite
    does the same for its pipelines; this hook covers the generic path too).
    forwards are once-per-image and run in FP32 on the CPU host (fast CPU
    path, exact numerics with the model patch suite's encoder pin).
    ``VLLM_OMNI_310P_ENCODER_CPU=0`` opts out (experiments with headroom).
    """
    if os.environ.get("VLLM_OMNI_310P_ENCODER_CPU", "1") != "1":
        return

    import torch

    import vllm_omni.diffusion.registry as _registry

    if getattr(_registry.initialize_model, "_310p_encoder_offloaded", False):
        return

    original = _registry.initialize_model

    def _offloaded(od_config, *args, **kwargs):
        model = original(od_config, *args, **kwargs)
        for attr in ("text_encoder", "text_encoder_2"):
            module = getattr(model, attr, None)
            if module is not None:
                # FP32 on the CPU host: fp16 CPU matmul is far slower (no fast
                # ARM fp16 path) and the model patch suite pins FP32 for the
                # same reason — numerics match L1 exactly.
                module.to("cpu", dtype=torch.float32)
        # Base pipelines assume the encoder sits on the execution device: they
        # push input_ids to it and never move the resulting embeds back.  With
        # the CPU offload, wrap _encode_prompt so the forward runs on the
        # encoder's device and the embeds return to the execution device in
        # the DiT's runtime dtype.  310P model-patch subclasses (e.g.
        # _ZImage310P) handle this in their own override and are left
        # untouched.
        if hasattr(model, "text_encoder") and not hasattr(model, "_encoder_device"):
            exec_dtype = next(iter(model.transformer.parameters())).dtype

            original_encode = model._encode_prompt

            def _cpu_encode(prompt, device=None, prompt_embeds=None, max_sequence_length=512):
                embeds = original_encode(
                    prompt, device="cpu", prompt_embeds=prompt_embeds, max_sequence_length=max_sequence_length
                )
                return [e.to(model._execution_device, dtype=exec_dtype) for e in embeds]

            model._encode_prompt = _cpu_encode
        return model

    _offloaded._310p_encoder_offloaded = True  # type: ignore[attr-defined]
    _registry.initialize_model = _offloaded  # type: ignore[assignment]
    # The diffusers loader binds ``initialize_model`` into its own namespace at
    # import time; rebind it explicitly (same early-binding hazard as
    # ``create_transformers_model``).
    try:
        import vllm_omni.diffusion.model_loader.diffusers_loader as _loader

        if not getattr(_loader.initialize_model, "_310p_encoder_offloaded", False):
            _loader.initialize_model = _offloaded  # type: ignore[assignment]
    except ImportError:
        pass
    logger.info("[310P] HF text encoders offloaded to CPU after construction (single-chip memory invariant)")


def apply_model_patches(model_config) -> None:
    model_arch = getattr(model_config, "model_arch", None)
    if model_arch == _QWEN3_TTS_TALKER_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_talker_patches

        apply_talker_patches()
        return
    elif model_arch == _QWEN3_TTS_CODE2WAV_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_code2wav_patches

        apply_code2wav_patches()
