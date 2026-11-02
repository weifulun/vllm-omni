# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
# Qwen2.5-Omni: all three stages share one entry arch; the stage is carried
# on model_config.model_stage (thinker / talker / code2wav).
_QWEN2_5_OMNI_ARCH = "Qwen2_5OmniForConditionalGeneration"


def apply_patches() -> None:
    global _WORKER_PATCHED

    if _WORKER_PATCHED:
        return

    from vllm_omni.platforms.npu._310p.patch.worker import apply_patch

    apply_patch()
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
    elif model_arch == _QWEN2_5_OMNI_ARCH:
        model_stage = getattr(model_config, "model_stage", None)
        from vllm_omni.platforms.npu._310p.patch import qwen2_5_omni as _qwen25_patch

        if model_stage == "thinker":
            _qwen25_patch.apply_thinker_patches()
        elif model_stage == "talker":
            _qwen25_patch.apply_talker_patches()
        elif model_stage == "code2wav":
            _qwen25_patch.apply_code2wav_patches()
