# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit math SDPA for the Wan-Dancer 310P port (task-40).

Ascend 310P (CANN 9.1.0) cannot execute ``torch.nn.functional.
scaled_dot_product_attention`` ("can not cast format when output is input",
research/012 dead-zone, 5 independent confirmations).  Every attention in the
Wan-Dancer stack (DiT self/cross, music injector, VAE mid-block) is therefore
computed with explicit matmul + fp32-softmax + matmul, chunked over query
positions so the per-head score matrix stays under a configurable byte budget
(256 MiB default — zimage/fluxklein precedent).

``math_sdpa`` is device-agnostic: on CPU it is the contract-test reference;
on NPU it is the production path (CANN GE lowers the three primitives).
"""

from __future__ import annotations

import math
import os

import torch

__all__ = ["math_sdpa", "resolve_head_chunk_budget_bytes"]

_DEFAULT_HEAD_CHUNK_BYTES = 256 * 1024 * 1024


def resolve_head_chunk_budget_bytes() -> int:
    """Per-head score-matrix byte budget for ``math_sdpa`` chunking."""
    raw = os.environ.get("VLLM_OMNI_310P_ATTN_HEAD_CHUNK_BYTES")
    if raw is None:
        return _DEFAULT_HEAD_CHUNK_BYTES
    return max(1024 * 1024, int(raw))


def _causal_band(row_start: int, row_end: int, sk: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Additive causal mask for global rows [row_start, row_end) x cols [0, sk).

    A column is kept iff ``col <= global_row`` (standard causal semantics).
    """
    rows = torch.arange(row_start, row_end, device=device)
    cols = torch.arange(sk, device=device)
    keep = cols[None, :] <= rows[:, None]
    return torch.where(keep, torch.zeros((), dtype=dtype, device=device), torch.full((), float("-inf"), dtype=dtype, device=device))


def _chunk_rows_for_scores(num_rows: int, num_cols: int, head_dim: int, budget_bytes: int) -> int:
    """Largest row chunk such that rows*cols*head_dim*sizeof(fp32 scores) <= budget."""
    elems_per_row = num_cols * head_dim
    rows = max(1, budget_bytes // (elems_per_row * 4))
    return min(num_rows, rows)


def math_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """fp16-in / fp32-softmax / dtype-out attention with row chunking.

    Layout: ``(batch, heads, seq, head_dim)`` for q/k/v (BSHD).  ``attn_mask``
    is an additive float mask broadcastable to ``(batch, heads, sq, sk)``.
    KV-head broadcasting is the caller's responsibility (repeat_interleave
    before the call — matches the 310P "expanded-KV GQA" rule from 012).
    """
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"math_sdpa: unsupported dtype {q.dtype}")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    out_dtype = q.dtype
    q32 = q.to(compute_dtype)
    k32 = k.to(compute_dtype)
    v32 = v.to(compute_dtype)

    if attn_mask is not None and attn_mask.dtype != compute_dtype:
        attn_mask = attn_mask.to(compute_dtype)

    batch, heads, sq, hd = q.shape
    sk = k.shape[2]
    budget = resolve_head_chunk_budget_bytes()

    # Fast path: whole score matrix fits the budget -> one matmul.
    if sq * sk * hd * 4 <= budget:
        scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
        if causal:
            scores = scores + _causal_band(0, sq, sk, compute_dtype, q.device).unsqueeze(0).unsqueeze(0)
        if attn_mask is not None:
            scores = scores + attn_mask
        scores = scores.softmax(dim=-1)
        return torch.matmul(scores, v32).to(out_dtype)

    # Chunked path (large sequences on 310P).
    rows = _chunk_rows_for_scores(sq, sk, hd, budget)
    out = torch.empty(batch, heads, sq, hd, dtype=out_dtype, device=q.device)
    for start in range(0, sq, rows):
        end = min(start + rows, sq)
        q_c = q32[:, :, start:end]  # (b, h, rc, hd)
        scores = torch.matmul(q_c, k32.transpose(-2, -1)) * scale  # (b, h, rc, sk)
        if causal:
            scores = scores + _causal_band(start, end, sk, compute_dtype, q.device).unsqueeze(0).unsqueeze(0)
        if attn_mask is not None and attn_mask.dim() == 4:
            scores = scores + attn_mask[:, :, start:end]
        scores = scores.softmax(dim=-1)
        out[:, :, start:end] = torch.matmul(scores, v32).to(out_dtype)
    return out
