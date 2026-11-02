# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P AR-Diffusion paged-attention hardware shim: explicit math decomposition.

The AR-Diffusion paged-KV path
(``vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention``, shared by
ABot-World and the LingBot-World family) routes NPU queries through
``torch_npu.npu_fusion_attention`` (the FlashAttentionScore fused kernel
family).  On 310P (CANN 9.1.0) that fused kernel family is dead —
``npu_fusion_attention`` fails ``AclOpKernelInit`` (research/012 §F.sdpa,
five independent on-card confirmations), so every AR-Diffusion paged
self-attention call on NPU would fail at graph build.

This shim replaces the NPU branch with the explicit math decomposition that
is verified on-card at every diffusion attention shape (fp16 matmul, fp32
softmax, head-chunked score matrix on the same 256 MiB budget as
``attention.py``).  It mirrors the NPU branch's gather semantics exactly
(block_table window gather, packed TND layout, per-sequence varlen
bounds), so semantics are unchanged — only the fused kernel is swapped for
its verified decomposition.

Hardware-level by design (same convention as the MATH-SDPA shim in
``attention.py``): installed for every 310P diffusion run — including the
``VLLM_OMNI_310P_DIFFUSION_PATCHES=0`` kill-switch level-2 reference — and
toggled only by its own knob
``VLLM_OMNI_310P_PAGED_MATH_ATTENTION`` (default 1).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False
_KNOB = "VLLM_OMNI_310P_PAGED_MATH_ATTENTION"
_HEAD_CHUNK_BYTES_ENV = "VLLM_OMNI_310P_ATTN_HEAD_CHUNK_BYTES"
_DEFAULT_HEAD_CHUNK_BYTES = 256 * 1024 * 1024


def _head_chunk_budget_bytes() -> int:
    try:
        return int(os.environ.get(_HEAD_CHUNK_BYTES_ENV, str(_DEFAULT_HEAD_CHUNK_BYTES)))
    except ValueError:
        return _DEFAULT_HEAD_CHUNK_BYTES


def _math_varlen_paged_attention(
    query_flat,
    packed_key,
    packed_value,
    query_start_loc,
    seq_lens,
    softmax_scale: float,
):
    """Explicit math decomposition of the varlen paged window attention.

    ``query_flat``: (T, H, D) flattened TND queries over the batch.
    ``packed_key``/``packed_value``: (max_seq_len, H, D) gathered window.
    Sequence ``i`` uses queries ``[query_start_loc[i], query_start_loc[i+1])``
    and keys ``[: seq_lens[i]]``.  Returns (T, H, D) in the query dtype.
    """
    total = query_flat.shape[0]
    heads = query_flat.shape[1]
    q_lens = (query_start_loc[1:] - query_start_loc[:-1]).cpu().tolist()
    kv_lens = seq_lens.cpu().tolist()
    kv_lens = kv_lens if isinstance(kv_lens, list) else [kv_lens]

    bytes_per_head = total * max(kv_lens) * 4  # fp32 scores per head, single seq
    budget = _head_chunk_budget_bytes()
    head_chunk = max(1, min(heads, budget // max(1, bytes_per_head)))

    out = query_flat.new_empty(query_flat.shape)
    q0 = 0
    for seq_idx in range(len(q_lens)):
        q_len = int(q_lens[seq_idx])
        k_len = int(kv_lens[seq_idx])
        q = query_flat[q0 : q0 + q_len]
        k = packed_key[:k_len]
        v = packed_value[:k_len]
        for h0 in range(0, heads, head_chunk):
            h1 = min(h0 + head_chunk, heads)
            # Output layout (h, query, key): softmax over the key axis (dim=-1).
            scores = torch_einsum("lhd,khd->hlk", q[:, h0:h1].float(), k[:, h0:h1].float())
            scores = scores * float(softmax_scale)
            probs = scores.softmax(dim=-1)
            out[q0 : q0 + q_len, h0:h1] = torch_einsum("hlk,khd->lhd", probs.to(v.dtype), v[:, h0:h1])
        q0 += q_len
    return out


# torch imported lazily to keep module import cheap in CPU test harnesses.
def torch_einsum(equation, *tensors):
    import torch

    return torch.einsum(equation, *tensors)


def maybe_patch_ar_paged_math() -> None:
    """Wrap ``ar_diffusion_paged_attention`` with the 310P math fallback.

    Safe to call more than once.  Only NPU-device queries are diverted; CPU
    tensors (and non-310P hosts, where this module is never installed) fall
    through to the original implementation unchanged.
    """
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get(_KNOB, "1") != "1":
        return
    try:
        import vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention as _pa
    except ImportError:
        # AR-Diffusion infrastructure absent from this build; nothing to shim.
        return

    if getattr(_pa.ar_diffusion_paged_attention, "_310p_math_patched", False):
        _PATCHED = True
        return

    original = _pa.ar_diffusion_paged_attention

    def _patched(query, key_cache, value_cache, *args, **kwargs):
        # Default "npu" (production).  Contract tests set "cpu" to exercise
        # the diversion path on host tensors; the gather + math composition is
        # device-agnostic.
        divert_device = os.environ.get("VLLM_OMNI_310P_PAGED_MATH_DEVICE", "npu")
        if getattr(query.device, "type", None) == divert_device and not kwargs.get("causal", False):
            block_table = kwargs.get("block_table")
            query_start_loc = kwargs.get("query_start_loc")
            seq_lens = kwargs.get("seq_lens")
            max_seq_len = kwargs.get("max_seq_len")
            softmax_scale = kwargs.get("softmax_scale")
            if None in (block_table, query_start_loc, seq_lens, max_seq_len, softmax_scale):
                raise ValueError("310P paged math fallback requires the varlen paged window arguments")
            batched = query.dim() == 4
            if batched:
                query_flat = query.reshape(-1, *query.shape[2:])
            else:
                query_flat = query
            block_size = key_cache.shape[1]
            num_blocks = (int(max_seq_len) + block_size - 1) // block_size
            block_ids = block_table[0, :num_blocks].long()
            gathered_k = key_cache.index_select(0, block_ids).flatten(0, 1)[: int(max_seq_len)]
            gathered_v = value_cache.index_select(0, block_ids).flatten(0, 1)[: int(max_seq_len)]
            out = _math_varlen_paged_attention(
                query_flat,
                gathered_k,
                gathered_v,
                query_start_loc,
                seq_lens,
                softmax_scale,
            )
            return out.reshape(query.shape) if batched else out
        return original(query, key_cache, value_cache, *args, **kwargs)

    _patched._310p_math_patched = True  # type: ignore[attr-defined]
    _pa.ar_diffusion_paged_attention = _patched
    _PATCHED = True
    logger.info("310P: AR-Diffusion paged attention NPU branch replaced with explicit math decomposition")
