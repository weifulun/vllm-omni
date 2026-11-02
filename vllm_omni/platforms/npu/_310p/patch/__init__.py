# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
_MING_TTS_ARCH = "MingTTSForConditionalGeneration"


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
    elif model_arch == _MING_TTS_ARCH:
        # Both Ming stages (stage-0 LLM+flow, stage-1 audio VAE) report the
        # wrapper arch; the patch swaps only the stage-0 LLM class.
        from vllm_omni.platforms.npu._310p.patch.ming_tts import apply_ming_tts_patches

        apply_ming_tts_patches()
