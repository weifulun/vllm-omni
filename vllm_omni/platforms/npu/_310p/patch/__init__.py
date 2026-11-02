# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"
_INDEXTTS2_TALKER_ARCH = "IndexTTS2TalkerForConditionalGeneration"
_INDEXTTS2_S2MEL_ARCH = "IndexTTS2S2MelDecoder"
_INDEXTTS25_TALKER_ARCH = "IndexTTS25TalkerForConditionalGeneration"
_INDEXTTS25_S2MEL_ARCH = "IndexTTS25S2MelDecoder"


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
    elif model_arch in (_INDEXTTS25_TALKER_ARCH, _INDEXTTS2_TALKER_ARCH):
        # Stage 0 (GPT-2 AR talker) is vLLM-native AR; nothing to patch.
        from vllm_omni.platforms.npu._310p.patch.indextts2 import (
            apply_indextts2_talker_310p_patch,
        )

        apply_indextts2_talker_310p_patch()
    elif model_arch in (_INDEXTTS25_S2MEL_ARCH, _INDEXTTS2_S2MEL_ARCH):
        # Stage 1 (S2Mel CFM/DiT + BigVGAN): force the 310P-safe FP16-only
        # runtime (bf16 switches and GPU-only runtimes off) before the first
        # request can observe the config.
        from vllm_omni.platforms.npu._310p.patch.indextts2 import (
            apply_indextts2_310p_patches,
        )

        apply_indextts2_310p_patches()
