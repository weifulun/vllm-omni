# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
# MiniMax Music 3 is a two-stage line: the AR talker and the acoustic
# decoder register under different model_arch spellings, and the repo-root
# HF config spells the composite model one way more (dual-spelling gate,
# the same 128 trap the OmniVoice line hit).
_MINIMAX_MUSIC3_TALKER_ARCHS = frozenset(
    {
        "MiniMaxMusic3TalkerForConditionalGeneration",
        "MiniMaxMusic3ForConditionalGeneration",
    }
)
_MINIMAX_MUSIC3_ACOUSTIC_ARCHS = frozenset({"MiniMaxMusic3AcousticForConditionalGeneration"})


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
        return
    elif model_arch in _MINIMAX_MUSIC3_TALKER_ARCHS:
        from vllm_omni.platforms.npu._310p.patch.minimax_music3 import apply_talker_patches

        apply_talker_patches()
        return
    elif model_arch in _MINIMAX_MUSIC3_ACOUSTIC_ARCHS:
        from vllm_omni.platforms.npu._310p.patch.minimax_music3 import apply_acoustic_patches

        apply_acoustic_patches()
