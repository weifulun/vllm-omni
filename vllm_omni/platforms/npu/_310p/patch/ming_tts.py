# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Ming-3.0-TTS (dense 0.5B) for the 310P NPU path.

Ming-3.0-tts runs as a two-stage omni pipeline (stage-0 LLM+flow head,
stage-1 audio VAE).  Unlike the Qwen3-TTS line, the Ming model code carries
no NPU graph capture of its own: the CFM CUDAGraph fast path is guarded by
``is_cuda`` and therefore never engages on an NPU host (eager flow head is
the NPU contract), and the audio-VAE decode path already coerces latents to
the decoder parameter dtype/device.  The 310P-specific surgery is therefore
limited to the dtype/runtime pins shared by every 310P TTS line:

1. Stage-0 flow-matching head, aggregator and the stop/speaker heads are
   pinned to FP16 (310P has no BF16: AclNN EZ1001 exposes no
   DT_BFLOAT16).  The pin is enforced in the 310P subclass so that a
   BF16/FP32 upstream config can never leak BF16 onto the flow head even
   though the generic NPU path would honor it.
2. The CFM CUDAGraph executor is pinned off for 310P explicitly (the
   ``is_cuda`` guard already keeps it off; the pin documents and enforces
   the contract so the eager flow head cannot be silently bypassed by a
   future ``torch.cuda``/``torch.npu`` aliasing change).

The stage-1 audio VAE needs no class surgery: ``MingAudioVAEModel.forward``
and ``AudioVAE.low_level_reconstruct`` already re-cast latent chunks to
``next(self.decoder.parameters()).dtype/device``, which equals the engine
load dtype (FP16 under the 310P deploy overlay).
"""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.ming_tts import ming_tts_llm

_RUNTIME_DTYPE = torch.float16
_PATCHED = False


class _MingLLMModel310P(ming_tts_llm.MingLLMModel):
    """Stage-0 Ming LLM+flow head specialized for the 310P NPU path."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # super() resolved fm_dtype from the (possibly bf16) upstream config
        # and already cast the flow submodules to it; re-pin to FP16 and
        # re-cast so the 310P contract holds regardless of deploy dtype.
        self.fm_dtype = _RUNTIME_DTYPE
        self.flowloss.to(dtype=_RUNTIME_DTYPE)
        self.linear_proj_audio.to(dtype=_RUNTIME_DTYPE)
        self.stop_head.to(dtype=_RUNTIME_DTYPE)
        self.spk_head.to(dtype=_RUNTIME_DTYPE)
        # CFM CUDAGraph is a CUDA-only fast path (``z_diff_cond.is_cuda``
        # guard). Pin it off explicitly on 310P: the eager flow head is the
        # contract, and torch.compile of the DiT must never be attempted.
        self._cfm_graph = None
        self._cfm_graph_enabled = False

    def _maybe_build_cfm_graph(self, z_diff_cond: torch.Tensor) -> None:
        # Hard off on 310P even though the base guard (is_cuda) already
        # declines on NPU tensors.
        del z_diff_cond
        return None


def apply_ming_tts_patches() -> None:
    """Install the Ming-3.0-TTS 310P patch.

    Only the stage-0 LLM class is swapped; stage-1 (audio VAE) keeps the
    generic implementation.  Registration is keyed on the wrapper arch
    ``MingTTSForConditionalGeneration`` which both stages report, so this
    hook fires in both stage processes — the swap is a no-op for stage-1,
    which never instantiates the LLM class.
    """

    global _PATCHED

    if _PATCHED:
        return

    ming_tts_llm.MingLLMModel = _MingLLMModel310P

    _PATCHED = True
