# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
_COSYVOICE3_MODEL_ARCH = "CosyVoice3Model"
_COSYVOICE3_TALKER_STAGE = "cosyvoice3_talker"
_COSYVOICE3_CODE2WAV_STAGE = "cosyvoice3_code2wav"


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
    elif model_arch == _COSYVOICE3_MODEL_ARCH:
        # Both CosyVoice3 stages resolve the same model class; the stage
        # discriminator is ``model_config.model_stage`` (set per stage by the
        # stage runtime).
        model_stage = getattr(model_config, "model_stage", None)
        if model_stage == _COSYVOICE3_TALKER_STAGE:
            from vllm_omni.platforms.npu._310p.patch.cosyvoice3 import apply_talker_310p_patches

            apply_talker_310p_patches()
        elif model_stage == _COSYVOICE3_CODE2WAV_STAGE:
            from vllm_omni.platforms.npu._310p.patch.cosyvoice3 import apply_code2wav_310p_patches

            apply_code2wav_310p_patches()
