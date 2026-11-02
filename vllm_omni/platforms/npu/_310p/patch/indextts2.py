# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS 2(.5) 310P (Atlas 300I Duo) patches.

IndexTTS 2.5 is a two-stage TTS pipeline:

- Stage 0 (GPT-2 AR talker) is vLLM-native: PagedAttention blocks, a plain
  LM head, and SDPA-based conformer conditioning. It carries no GPU-only
  acceleration (no NPUGraph, no CUDA graph, no torch.compile path inside the
  model code), so on 310P the only action is traceability: the dispatch
  branch logs that the stage was seen by the 310P patch surface.
- Stage 1 (S2Mel CFM/DiT + BigVGAN) is the guarded surface. The 910B deploy
  config pins ``s2mel_dit_bf16: true`` (bf16 estimator pre-cast + bf16
  autocast), and the DiT/vocoder torch.compile and CUDA-graph runtimes are
  CUDA-gated in model code but still advertised by the config switches.
  310P is FP16-only (no BF16), so this patch forces every bf16 switch and
  every GPU-only runtime off at the model level — belt-and-braces with the
  ``deploy/indextts2_5_310p.yaml`` overlay.

The wrap is installed on the shared decoder class (both ``indextts2`` and
``indextts2_5`` pipelines resolve to ``IndexTTS2S2MelDecoder``), so the same
guard covers IndexTTS-2 on 310P.

The stage-1 external-model preload also trips a device bug in the shared
BigVGAN path: ``common.alias_free_activation.kaiser_sinc_filter1d`` moves the
Kaiser window to the NPU while leaving the sinc side on the default device
(CPU in the stage-1 worker), so ``window * sinc`` fails during
``_load_bigvgan`` before the module is ever moved on-device. The 310P line
pins that filter math to CPU (the GPU/CPU lines' portable behavior); the
registered buffers then ride the module move.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

# Config switches that are meaningless or unsafe on 310P (FP16-only chip,
# no CUDA graph capture, no inductor). Forcing them off keeps the FP32 CFM
# solver + FP32 eager BigVGAN path, which is the correctness path on 310P.
_FORCED_OFF_FIELDS = (
    "s2mel_dit_bf16",
    "s2mel_vocoder_bf16",
    "s2mel_dit_torch_compile",
    "s2mel_dit_cuda_graph",
    "s2mel_vocoder_torch_compile",
    "s2mel_vocoder_cuda_graph",
)

_applied = False


def _force_310p_fp16_only_runtime(model: object) -> None:
    """Force the 310P-safe runtime switches off on a constructed decoder."""
    changed = []
    for field in _FORCED_OFF_FIELDS:
        if getattr(model, field, None):
            setattr(model, field, False)
            changed.append(field)
    if changed:
        logger.info(
            "[310P] IndexTTS S2Mel FP16-only runtime enforced: %s "
            "(bf16 disabled — 310P has no BF16; GPU runtimes disabled)",
            ", ".join(changed),
        )


def _patch_bigvgan_kaiser_filter_for_310p() -> None:
    """CPU-safe Kaiser-sinc filter for BigVGAN LowPassFilter construction.

    The NPU branch of ``common.alias_free_activation.kaiser_sinc_filter1d``
    moves the Kaiser window to the NPU but leaves ``time_indices`` and
    ``sinc_filter`` on the default device (CPU in the stage-1 worker), so
    ``kaiser_window * sinc_filter`` crashes with an NPU/CPU device mismatch
    while BigVGAN is still being built on CPU. The GPU/CPU lines compute the
    whole filter on the default device, so a CPU-only computation is exactly
    the portable behavior; the registered buffer is moved with the module
    afterwards.
    """
    import math

    import torch

    from vllm_omni.model_executor.models.common import alias_free_activation

    original = alias_free_activation.kaiser_sinc_filter1d
    if getattr(original, "_indextts2_310p_cpu_safe", False):
        return

    def _cpu_kaiser_sinc_filter1d(
        cutoff: float, half_width: float, kernel_size: int
    ) -> torch.Tensor:
        # Same math as the shared implementation, everything on CPU.
        is_even = kernel_size % 2 == 0
        half_size = kernel_size // 2
        delta_f = 4 * half_width
        attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
        if attenuation > 50.0:
            beta = 0.1102 * (attenuation - 8.7)
        elif attenuation >= 21.0:
            beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
        else:
            beta = 0.0
        kaiser_window = torch.kaiser_window(
            kernel_size, beta=beta, periodic=False, dtype=torch.float32, device="cpu"
        )
        if is_even:
            time_indices = torch.arange(-half_size, half_size, device="cpu") + 0.5
        else:
            time_indices = torch.arange(kernel_size, device="cpu") - half_size
        if cutoff == 0:
            return torch.zeros((1, 1, kernel_size), dtype=torch.float32)
        sinc_filter = torch.sinc(2 * cutoff * time_indices)
        normalized_filter = 2 * cutoff * kaiser_window * sinc_filter
        normalized_filter /= normalized_filter.sum()
        return normalized_filter.view(1, 1, kernel_size)

    _cpu_kaiser_sinc_filter1d._indextts2_310p_cpu_safe = True  # type: ignore[attr-defined]
    alias_free_activation.kaiser_sinc_filter1d = _cpu_kaiser_sinc_filter1d
    logger.info(
        "[310P] BigVGAN kaiser_sinc_filter1d pinned to CPU "
        "(stage-1 preload device mismatch avoided)"
    )


def apply_indextts2_talker_310p_patch() -> None:
    """Stage 0 (GPT-2 AR talker): no 310P code patch required.

    The talker is vLLM-native AR (PagedAttention) with SDPA-based conformer
    conditioning; nothing GPU-specific is installed by model code, so the
    310P line runs it unmodified under the NPU AR worker.
    """
    logger.info(
        "[310P] IndexTTS talker (GPT-2 AR): no code patch required; "
        "vLLM-native AR runs unmodified on the NPU AR worker"
    )


def apply_indextts2_310p_patches() -> None:
    """Idempotent: wrap the S2Mel decoder ``__init__`` for 310P forcing."""
    global _applied
    if _applied:
        return
    _applied = True

    from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
        IndexTTS2S2MelDecoder,
    )

    original_init = IndexTTS2S2MelDecoder.__init__
    if getattr(original_init, "_indextts2_310p_patched", False):
        return

    # Mirror the decoder's exact signature: vLLM's old-style model loader
    # introspects ``__init__`` to guess constructor arguments, so a
    # ``*args/**kwargs`` wrapper would hide the required keyword-only
    # ``vllm_config`` and break model construction.
    def _patched_init(self: object, *, vllm_config: object, prefix: str = "") -> None:
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        from vllm_omni.platforms.npu._310p import is_310p

        if is_310p():
            _force_310p_fp16_only_runtime(self)

    _patched_init._indextts2_310p_patched = True  # type: ignore[attr-defined]
    IndexTTS2S2MelDecoder.__init__ = _patched_init  # type: ignore[method-assign]
    logger.info("[310P] IndexTTS S2Mel decoder __init__ wrapped (FP16-only enforcement)")

    # External-model preload (BigVGAN) builds LowPassFilter Kaiser-sinc
    # buffers before the module is moved on-device; the shared NPU branch
    # mixes NPU and default-device tensors there, so pin the filter math to
    # CPU for the 310P line.
    _patch_bigvgan_kaiser_filter_for_310p()
