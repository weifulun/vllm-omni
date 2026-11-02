# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
# Covo-Audio-Chat (tencent/Covo-Audio-Chat): one orchestrator class for both
# stages; arch spelling varies with stage/config resolution, so match all of
# them and let the covo patch module discriminate on model_config.model_stage.
_COVO_ARCHES = frozenset(
    {
        "CovoAudioForConditionalGeneration",
        "CovoAudioForCausalLM",
        "CovoAudioModel",
        "CovoAudioLLMForConditionalGeneration",
        "CovoAudioCode2WavForConditionalGeneration",
    }
)


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
    elif model_arch in _COVO_ARCHES:
        # One orchestrator class for both stages (model_stage discriminates
        # inside the model); both patch groups are idempotent, so apply the
        # combined entry point per engine.
        from vllm_omni.platforms.npu._310p.patch.covo_audio import apply_covo_audio_patches

        apply_covo_audio_patches(model_config)
