# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
_MING3_TTS_TINY_ARCHS = (
    "Ming3TTSForConditionalGeneration",
    "LingV3TTSForConditionalGeneration",
    "Ming3TTSLLMModel",
)


def apply_patches() -> None:
    global _WORKER_PATCHED

    if _WORKER_PATCHED:
        return

    from vllm_omni.platforms.npu._310p.patch.worker import apply_patch

    apply_patch()

    # Worker-level: rebind the BailingMoeV3 KDA/conv ops to the pure-torch
    # 310P implementations before any model is constructed.  Harmless for
    # other lines (idempotent; the rebind targets only the bailing_moe_v3
    # module, which non-Ming models never import in this process).
    from vllm_omni.platforms.npu._310p.patch.ming3_tts import install_kda_ops

    install_kda_ops()
    _WORKER_PATCHED = True


def apply_model_patches(model_config) -> None:
    model_arch = getattr(model_config, "model_arch", None)
    if model_arch == _QWEN3_TTS_TALKER_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_talker_patches

        apply_talker_patches()
        return
    elif model_arch == _QWEN3_TTS_CODE2WAV_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_code2wav_patches

        apply_code2wav_patches()
    elif model_arch in _MING3_TTS_TINY_ARCHS:
        from vllm_omni.platforms.npu._310p.patch.ming3_tts import (
            apply_ming3_tts_tiny_patches,
        )

        apply_ming3_tts_tiny_patches()
