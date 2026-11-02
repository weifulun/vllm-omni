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
# expose apply_diffusion_310p_patches(od_config).  The platform call site
# (NPUOmniPlatform.init_diffusion_model_runner_runtime) invokes
# apply_diffusion_310p_patches() unchanged, verbatim-convergent with the
# task-11 (SDXL) design.
#
# Engine-level 310P invariants (math-SDPA decomposition, startup-warmup skip,
# POLICY §1 level-2 kill switch) live in this dispatcher and apply to every
# registered model; per-model invariants live in the registered modules.
_DIFFUSION_310P_PATCH_MODULES: dict[str, list[str]] = {
    # Cosmos3 family (task-25): the shared Cosmos3OmniDiffusersPipeline serves
    # Edge / Nano / Super; the 310P invariants (FP16, attention-backend gate,
    # cache disabling, quantization tier gate, VAE tiling, guardrails gate)
    # are model-agnostic within the family, so one module covers all three
    # checkpoints.  Both class-name aliases are registered: the HF model_index
    # carries ``_class_name: Cosmos3OmniPipeline`` while the pipeline registry
    # also knows the longer ``Cosmos3OmniDiffusersPipeline`` name.
    "Cosmos3OmniDiffusersPipeline": ["vllm_omni.platforms.npu._310p.patch.cosmos3_edge"],
    "Cosmos3OmniPipeline": ["vllm_omni.platforms.npu._310p.patch.cosmos3_edge"],
    # Boogu-Image-0.1 family (task-33): Base (task-33 GO checkpoint) and the
    # Turbo DMD variant share the pipeline module; both ride the same 310P
    # invariants (FP16, attention-backend gate, cache rejection, quantization
    # tier gate - INT8 W8A8 is the GO-sanctioned primary tier for the family
    # because the FP16 tier does not fit one 21589 MB chip - VAE tiling,
    # single-chip TP1 preflight).  The Qwen3-VL understanding encoder is
    # pinned to CPU at the container level (capacity feasibility, kill-switch
    # independent - see patch/boogu_image.py header).
    "BooguImagePipeline": ["vllm_omni.platforms.npu._310p.patch.boogu_image"],
    "BooguImageTurboPipeline": ["vllm_omni.platforms.npu._310p.patch.boogu_image"],
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
    ops_legacy/dynamic/mem_set.py, and the RealDiv/OnesLike/Mul/ClipByValue
    family in ops_legacy/dynamic/*) even though the real request geometry
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

    ``VLLM_OMNI_310P_DIFFUSION_PATCHES=0`` disables the whole dispatch: it is
    the comparison hook for COMPARISON-POLICY.md §1 level 2 (run the generic
    310P path with an explicit FP16 config as the same-seed reference).
    """
    # MATH-SDPA decoupling (sdxl line, 939fc24 design): the math decomposition
    # is a hardware-level invariant on 310P (fused SDPA kernel is dead there),
    # NOT a per-model patch.  Install it before the kill-switch gate so the
    # COMPARISON-POLICY.md sec.1 level-2 same-seed reference runs keep it
    # active too - the generic path differs from the patched path only in the
    # per-model invariants below, which is exactly the delta under test.
    from vllm_omni.platforms.npu._310p.patch.attention import maybe_patch_sdpa_math

    maybe_patch_sdpa_math()

    # Warmup-skip decoupling (task-33, mirrors the MATH-SDPA decoupling
    # above): the 512x512 dummy warmup is a hardware-level hazard on 310P
    # (research/012: its conv sub-graphs fail TBE dynamic-variant
    # pre-compilation) and, with a CPU-pinned MLLM, the warmup's image-input
    # vision encode burns ~10 CPU minutes - longer than the engine's 600 s
    # orchestrator init gate, so a kill-switch (level-2) server could never
    # become ready while the dummy run still ran.  COMPARISON-POLICY.md §1
    # level 2 keeps the env-driven warmup skip active, so the measured
    # level-2-vs-level-1 delta stays the per-model invariants only.
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


def apply_model_patches(model_config) -> None:
    model_arch = getattr(model_config, "model_arch", None)
    if model_arch == _QWEN3_TTS_TALKER_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_talker_patches

        apply_talker_patches()
        return
    elif model_arch == _QWEN3_TTS_CODE2WAV_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_code2wav_patches

        apply_code2wav_patches()
