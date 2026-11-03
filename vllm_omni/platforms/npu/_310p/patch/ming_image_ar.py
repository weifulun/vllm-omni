# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""310P AR-side patch for the Ming-Image stage-0 MLLM (task-36b, recipe A).

Port of the ming3 line's pre-add RMSNorm convention fix (task-27,
agents/model-ming3 report §3.1 #4) to the ``BailingMoeV2`` backbone that
Ming-Image's 17.0B MoE AR design agent runs on.

Why the backbone norms need pinning
------------------------------------
``BailingMoeV2DecoderLayer`` calls vLLM's fused two-arg ``RMSNorm``::

    hidden_states, residual = self.input_layernorm(hidden_states, residual)

i.e. it normalizes the **sum** (``rms(x + residual)``).  The
HF-transformers BailingMoE implementation this checkpoint was trained and
served with instead normalizes the **pre-add** input (``rms(x)``) *before*
adding to the residual stream.  The two conventions converge under BF16
(the official 1x80GiB CUDA config), but on the 310P's FP16-only stack the
sum-normalized trajectory runs ~100x hotter (task-27 measured a residual
stream absmax ~1.6e5 by layer 4, above the FP16 ceiling of 65504, with 171
non-finite prefill values) and corrupts the decode hidden states.  The 310P
path therefore pins the backbone norms to the training convention: FP32
math, weight shared with the original module (no copies).

Swap mechanism
--------------
Mirrors the ming3 line: ``NPUModelRunner.load_model`` dispatches
``apply_model_patches(self.model_config)`` on 310P *before* the stage-0
model is constructed, and the Ming-Image stage-0 arch resolves to
``MingImageForConditionalGeneration``, whose ``__init__`` instantiates the
module-level ``BailingMoeV2ForCausalLM`` name in the ``ming_image.model``
namespace.  Replacing that one name with the 310P subclass is sufficient —
no other module constructs the backbone for this pipeline.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from vllm_omni.model_executor.models.ming_flash_omni.modeling_bailing_moe_v2 import (
    BailingMoeV2ForCausalLM,
)

logger = logging.getLogger(__name__)

_PATCHED = False

#: Backbone decoder-layer norm attribute names (BailingMoeV2DecoderLayer).
_LAYER_NORM_NAMES = ("input_layernorm", "post_attention_layernorm")


class _MingPreAddRMSNorm310P(nn.Module):
    """HF-transformers pre-add-convention RMSNorm for the 310P path.

    ``forward(x, residual=None)`` returns ``rms(x)`` (normalized pre-add
    input) plus the updated residual stream ``x + residual`` when a
    residual is supplied; returns the normalized tensor alone otherwise.
    All math is done in FP32 and cast back to the input dtype; the weight
    is shared with the original module's parameter (no copies).
    """

    def __init__(self, weight: nn.Parameter, variance_epsilon: float) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = variance_epsilon

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        normed = xf * torch.rsqrt(
            xf.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        normed = (normed * self.weight.float()).to(x.dtype)
        if residual is None:
            return normed
        residual_new = (xf + residual.float()).to(x.dtype)
        return normed, residual_new


class _BailingMoeV2ForCausalLM310P(BailingMoeV2ForCausalLM):
    """Stage-0 BailingMoeV2 backbone specialized for the 310P NPU path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._install_preadd_norms()

    def _install_preadd_norms(self) -> None:
        """Swap the backbone decoder-layer norms to the HF pre-add convention.

        Guarded so the contract-test fake backbone (no decoder layers)
        passes untouched; the final outer norm is left as-is (it is a plain
        normalization without residual-add semantics).
        """
        backbone = getattr(self, "model", None)
        layers = getattr(backbone, "layers", None) if backbone is not None else None
        if layers is None:
            return
        swapped = 0
        for lyr in layers:
            for name in _LAYER_NORM_NAMES:
                mod = getattr(lyr, name, None)
                if mod is None or not hasattr(mod, "weight"):
                    continue
                setattr(
                    lyr,
                    name,
                    _MingPreAddRMSNorm310P(mod.weight, mod.variance_epsilon),
                )
                swapped += 1
        logger.info(
            "310P: Ming-Image AR backbone pre-add norms installed (layers=%d swapped=%d)",
            len(layers),
            swapped,
        )


def apply_ming_image_ar_patches() -> None:
    """Install the Ming-Image AR-side 310P patch.

    Only the stage-0 backbone class is swapped; the stage-1 diffusion
    pipeline keeps the generic 310P diffusion patch (``ming_image.py``).
    Idempotent: the class reference is swapped once per process.
    """
    global _PATCHED

    if _PATCHED:
        return

    from vllm_omni.model_executor.models.ming_image import model as ming_image_model

    ming_image_model.BailingMoeV2ForCausalLM = _BailingMoeV2ForCausalLM310P  # type: ignore[attr-defined]

    _PATCHED = True
    logger.info("310P: Ming-Image AR pre-add norm patch installed (BailingMoeV2 backbone)")
