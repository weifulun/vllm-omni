# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
_MINICPMO_ARCHS = (
    "MiniCPMO45OmniForConditionalGeneration",
    "MiniCPMO45OmniLLMForConditionalGeneration",
    "MiniCPMO45OmniTTSForConditionalGeneration",
    "MiniCPMO45Code2Wav",
)
_MINICPMO_CODE2WAV_ARCH = "MiniCPMO45Code2Wav"


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
    elif model_arch in _MINICPMO_ARCHS:
        # Stages 0 (thinker) and 1 (talker) share the pipeline-level arch;
        # ``model_stage`` ("llm" / "tts" / "code2wav") disambiguates them.
        from vllm_omni.platforms.npu._310p.patch import minicpmo

        model_stage = getattr(model_config, "model_stage", None)
        if model_arch == _MINICPMO_CODE2WAV_ARCH or model_stage == "code2wav":
            minicpmo.apply_code2wav_310p_patches()
        elif model_stage == "tts":
            minicpmo.apply_talker_310p_patches()
        else:
            minicpmo.apply_thinker_310p_patches()
