# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P attention patch: explicit math SDPA decomposition.

On 310P (CANN 9.1.0 + torch 2.10.0+cpu / torch_npu 2.10.0.post4) every
torch SDPA path is unusable:

* the fused NPU SDPA kernel fails for fp16 and fp32, all head geometries and
  sequence lengths (s=64..16384, heads=5/d=64) with
  ``can not cast format when output is input``; at pipeline level it surfaces
  as a GE graph-build failure in the attention sub-graph (dynamic MemSet
  pre-compile, ops_legacy/dynamic/mem_set.py);
* routing through torch's MATH backend (``sdpa_kernel(SDPBackend.MATH)``)
  fails with the same format-cast error, while the identical op sequence
  executed as plain eager ops (scale-mul -> matmul -> [bias add] ->
  softmax -> matmul) is verified working at every SDXL attention shape.

So this patch replaces ``sdpa._sdpa_attention`` with a direct math
decomposition on the vllm-omni side (image stays read-only), preserving the
upstream GQA head-expansion, float-bias, and causal semantics.
Knob: ``VLLM_OMNI_310P_FORCE_MATH_SDPA`` (default 1).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _math_sdpa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask, causal: bool, scale
) -> torch.Tensor:
    # GQA: expand KV heads to match query heads (mirrors upstream logic).
    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError(
                "GQA requires query heads to be a multiple of KV heads, "
                f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}."
            )
        repeat_num = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat_num, dim=1)
        value = value.repeat_interleave(repeat_num, dim=1)

    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    attn = torch.matmul(query * scale, key.transpose(-2, -1))
    if causal:
        seq = attn.shape[-1]
        causal_mask = torch.ones(seq, seq, dtype=torch.bool, device=attn.device).tril()
        attn = attn.masked_fill(~causal_mask, float("-inf"))
    if attention_mask is not None:
        attn = attn + attention_mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, value)


def _patch_f_sdpa() -> None:
    """Wrap F.scaled_dot_product_attention (diffusers VAE mid-block attention).

    The vllm-omni SDPA backend is covered by the ``_sdpa_attention`` swap,
    but the diffusers VAE decoder's mid-block attention calls
    ``F.scaled_dot_product_attention`` directly; on 310P that fused kernel
    is broken, so the torch-level call gets the same math decomposition.
    """
    import torch.nn.functional as F

    if getattr(F.scaled_dot_product_attention, "_310p_math_patched", False):
        return

    _orig_sdpa = F.scaled_dot_product_attention

    def _math_f_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        if dropout_p > 0.0:
            raise NotImplementedError("dropout is not supported by the 310P math SDPA path")
        if kwargs:
            raise NotImplementedError(f"unexpected kwargs for 310P math SDPA: {sorted(kwargs)}")
        if enable_gqa and query.shape[1] != key.shape[1] and key.shape[1] > 0:
            repeat_num = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_num, dim=1)
            value = value.repeat_interleave(repeat_num, dim=1)
        return _math_sdpa(query, key, value, attn_mask, is_causal, scale)

    _math_f_sdpa._310p_math_patched = True  # type: ignore[attr-defined]
    _math_f_sdpa.__wrapped__ = _orig_sdpa  # type: ignore[attr-defined]
    F.scaled_dot_product_attention = _math_f_sdpa


def maybe_patch_sdpa_math() -> None:
    """Replace the SDPA entry points with the explicit math decomposition."""
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("VLLM_OMNI_310P_FORCE_MATH_SDPA", "1") != "1":
        return

    import vllm_omni.diffusion.attention.backends.sdpa as _sdpa

    if not getattr(_sdpa, "_310p_math_patched", False):
        _sdpa._sdpa_attention = _math_sdpa
        _sdpa._310p_math_patched = True  # type: ignore[attr-defined]

    _patch_f_sdpa()
    _PATCHED = True
    logger.info("310P: SDPA entry points replaced with explicit math decomposition")
