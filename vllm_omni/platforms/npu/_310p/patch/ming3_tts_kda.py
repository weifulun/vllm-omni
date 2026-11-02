# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P-safe KDA (gated delta rule) ops for the BailingMoeV3 backbone.

The Ming-3.0-tts-tiny LLM backbone (``bailing_moe_v3``, 18 KDA / 6 MLA
layers) runs its linear-attention layers through FLA Triton kernels
(``fused_kda_gate``, ``chunk_kda``, ``fused_recurrent_kda``,
``fused_recurrent_kda_fwd``) and the vLLM mamba-ops ``causal_conv1d_fn`` /
``causal_conv1d_update``.  On the 310P image none of those can execute:

* the image ships no Triton (``HAS_TRITON=False``), so the upstream FLA
  kernels are placeholders that raise on call;
* ``vllm_ascend`` only installs its NPU KDA/causal-conv swaps through
  ``patch/worker/patch_triton.py`` when ``HAS_TRITON`` is true, so no swap
  reaches the 310P worker at all;
* the image's own ``vllm_ascend/ops/triton/kda`` tree is Triton as well.

This module provides pure-torch replacements of exactly the functions
``vllm.model_executor.models.bailing_moe_v3`` calls, transcribed from the
FLA Triton kernel sources (per-token gated-delta-rule recurrence), and
rebinds them onto the ``bailing_moe_v3`` module namespace — the same
rebind the 910-series ``patch_triton`` performs, gated here on
``is_310p()`` so the CUDA/910 paths are untouched.

Semantics preserved (checked against ``fused_recurrent_gated_delta_rule_
fwd_kernel`` with ``IS_KDA=True``):

* ``q``/``k`` are L2-normalized (eps 1e-6) in fp32 before the scale;
* the state decays elementwise by ``exp(g_t)`` where ``g_t`` is the raw
  per-token gate (the ``fused_kda_gate`` output);
* the delta rule runs ``u = S @ k_t``; ``v_t = (v_t - u) * beta_t``;
  ``S += v_t outer k_t``; and the output is read from the *updated* state
  (``o_t = S_{t+1} @ q_t``);
* the final state is written back to the mamba state pool slot selected by
  ``ssm_state_indices``.

The gate itself (``fused_kda_gate``) is elementwise:
``gate = lower_bound * sigmoid(exp(A_log_h) * (x + dt_bias))`` under the
safe-gate branch, else ``-exp(A_log_h) * softplus(x + dt_bias, thr=20)``.

Kill switch: ``VLLM_OMNI_310P_MING3TTS_PATCH=0`` skips installation
(matches the 310P TTS-line convention).
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_PATCH_ENV = "VLLM_OMNI_310P_MING3TTS_PATCH"
_INSTALLED = False

_RCP_LN2 = 1.4426950408889634  # noqa: F841 (parity constant, unused)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def kda_gate_torch(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Pure-torch ``fused_kda_gate``.

    ``g: [..., H*D]`` -> ``[..., H, D]`` fp32, per-token raw gate (no
    cumsum; the recurrence consumes per-token gates directly).
    """
    orig_shape = g.shape[:-1]
    g = g.reshape(-1, g.shape[-1]).float()
    HD = g.shape[-1]
    H = A.numel()
    if H * head_k_dim != HD:
        raise ValueError(f"KDA gate shape mismatch: H*D={HD}, H={H}, D={head_k_dim}")
    g4 = g.view(-1, H, head_k_dim)
    if g_bias is not None:
        g4 = g4 + g_bias.float().reshape(H, head_k_dim)
    a = A.float().reshape(H).exp()
    if lower_bound is not None:
        y = lower_bound * torch.sigmoid(a.view(1, H, 1) * g4)
    else:
        y = -a.view(1, H, 1) * nn.functional.softplus(g4, beta=beta, threshold=threshold)
    return y.view(*orig_shape, H, head_k_dim)


# ---------------------------------------------------------------------------
# Recurrent gated delta rule (shared by prefill and decode paths)
# ---------------------------------------------------------------------------


def _l2norm_last(x: torch.Tensor) -> torch.Tensor:
    # Kernel: b_x / sqrt(sum(b_x * b_x) + 1e-6)  (fp32)
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _expand_heads(t: torch.Tensor, h: int, hv: int) -> torch.Tensor:
    if hv == h:
        return t
    rep = hv // h
    return t.unsqueeze(1).expand(-1, rep, -1, -1).reshape(-1, hv, *t.shape[2:])


def _kda_recurrent_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    bos: int,
    eos: int,
    out: torch.Tensor,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Run the per-token recurrence over tokens [bos, eos); returns final S.

    q/k: [T, H, D]; v: [T, HV, V]; g: [T, HV, D]; beta: [T, HV] — all
    already sliced to the sequence and fp32-cast by the caller.
    state: [HV, V, D] fp32, in place.
    """
    T = eos - bos
    H = q.shape[1]
    HV = v.shape[1]
    V = v.shape[2]
    D = q.shape[2]
    if HV != H:
        q = _expand_heads(q, H, HV)
        k = _expand_heads(k, H, HV)
    if use_qk_l2norm:
        q = _l2norm_last(q)
        k = _l2norm_last(k)
    q = q * scale
    for t in range(T):
        # b_h *= exp(b_gk)   [kernel: decay before the delta correction]
        state.mul_(torch.exp(g[t]).unsqueeze(1))  # [HV, 1, D] over [HV, V, D]
        # b_v -= sum(b_h * b_k, 1); b_v *= b_beta; b_h += b_v[:,None]*b_k[None,:]
        u = torch.einsum("hvd,hd->hv", state, k[t])
        vt = (v[t] - u) * beta[t].unsqueeze(-1)
        state.add_(vt.unsqueeze(1) * k[t].unsqueeze(2))
        # b_o = sum(b_h * b_q, 1)  from the UPDATED state
        out[t] = torch.einsum("hvd,hd->hv", state, q[t])
    return state


# ---------------------------------------------------------------------------
# Replacements for the FLA kda entry points
# ---------------------------------------------------------------------------


def chunk_kda_310p(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,  # safe_gate / lower_bound / chunk_size: gate already applied
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure-torch replacement for ``chunk_kda`` (prefill recurrence)."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)
    B, T, H, D = q.shape
    HV, V = v.shape[2], v.shape[3]
    out = q.new_empty(1, T, HV, V)
    final_states = None
    if output_final_state:
        ref = initial_state if initial_state is not None else q
        final_states = torch.empty(
            (len(cu_seqlens) - 1, HV, V, D),
            dtype=ref.dtype if initial_state is not None else torch.float32,
            device=q.device,
        )
    assert B == 1, "varlen prefill is flattened to batch=1 by the caller"
    out32 = torch.empty_like(out, dtype=torch.float32)
    for n in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        if eos <= bos:
            continue
        if initial_state is not None:
            # clone(): .float() aliases an already-fp32 tensor (torch returns
            # self for identity conversions), which would corrupt the caller's
            # state pool through the in-place recurrence.
            state = initial_state[n].clone()
        else:
            state = torch.zeros((HV, V, D), dtype=torch.float32, device=q.device)
        _kda_recurrent_sequence(
            q[0, bos:eos].float(),
            k[0, bos:eos].float(),
            v[0, bos:eos].float(),
            g[0, bos:eos].float(),
            beta[0, bos:eos].float(),
            scale,
            state,
            bos=bos,
            eos=eos,
            out=out32[0, bos:eos],
            use_qk_l2norm=use_qk_l2norm_in_kernel,
        )
        out[0, bos:eos].copy_(out32[0, bos:eos])
        if output_final_state:
            final_states[n].copy_(state)
    return out, final_states


def fused_recurrent_kda_310p(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor = None,
    ssm_state_indices: torch.LongTensor = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch replacement for ``fused_recurrent_kda`` (decode rows)."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)
    B, T, H, D = q.shape
    HV, V = v.shape[2], v.shape[-1]
    out = q.new_empty(1, T, HV, V)
    out32 = torch.empty_like(out, dtype=torch.float32)
    for n in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        if eos <= bos:
            continue
        slot = int(ssm_state_indices[n]) if ssm_state_indices is not None else n
        # clone(): .float() aliases an already-fp32 pool slot; the recurrence
        # runs in place and must not touch the pool until the final copy_.
        state = initial_state[slot].clone() if initial_state is not None else None
        if state is None:
            state = torch.zeros((HV, V, D), dtype=torch.float32, device=q.device)
        final = _kda_recurrent_sequence(
            q[0, bos:eos].float(),
            k[0, bos:eos].float(),
            v[0, bos:eos].float(),
            g[0, bos:eos].float(),
            beta[0, bos:eos].float(),
            scale,
            state,
            bos=bos,
            eos=eos,
            out=out32[0, bos:eos],
            use_qk_l2norm=use_qk_l2norm_in_kernel,
        )
        out[0, bos:eos].copy_(out32[0, bos:eos])
        if inplace_final_state and initial_state is not None:
            initial_state[slot].copy_(final)
    return out, initial_state


def fused_recurrent_kda_fwd_310p(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.LongTensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spec-decode entry; the 310P line runs eager single-stream (no spec
    decoding), so this delegates to the row-wise recurrence.  ``num_accepted_
    tokens`` beyond one accepted token per row is not supported here and
    raises rather than silently producing wrong states."""
    if num_accepted_tokens is not None and bool(
        (num_accepted_tokens > 1).any()
    ):
        raise NotImplementedError(
            "310P KDA: multi-token spec acceptance is not supported "
            "(eager single-stream serving only)"
        )
    return fused_recurrent_kda_310p(
        q,
        k,
        v,
        g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )


# ---------------------------------------------------------------------------
# causal_conv1d swap (reuse the image's pure-torch NPU implementations)
# ---------------------------------------------------------------------------


def _filter_kwargs(fn, kwargs: dict) -> dict:
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {key: val for key, val in kwargs.items() if key in params}


def make_conv_swaps() -> tuple[Any, Any]:
    from vllm_ascend.ops.causal_conv1d import (
        causal_conv1d_fn as _conv_fn_impl,
        causal_conv1d_update as _conv_update_impl,
    )

    def _conv_fn(*args, **kwargs):
        return _conv_fn_impl(*args, **_filter_kwargs(_conv_fn_impl, kwargs))

    def _conv_update(*args, **kwargs):
        return _conv_update_impl(*args, **_filter_kwargs(_conv_update_impl, kwargs))

    return _conv_fn, _conv_update


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install_kda_310p_ops() -> None:
    """Rebind the KDA + causal-conv ops on ``bailing_moe_v3`` for 310P.

    Idempotent.  No-op when the kill switch is set or when the upstream
    module layout changed unexpectedly (logs and leaves the upstream path).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get(_PATCH_ENV, "1") == "0":
        logger.info("Ming3 310P KDA patch disabled by %s=0", _PATCH_ENV)
        return

    import vllm.model_executor.models.bailing_moe_v3 as bm3

    expected = (
        "chunk_kda",
        "fused_recurrent_kda",
        "fused_recurrent_kda_fwd",
        "fused_kda_gate",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    if any(not hasattr(bm3, name) for name in expected):
        logger.warning(
            "Ming3 310P KDA patch: upstream bailing_moe_v3 module layout "
            "changed; skipping install (upstream path kept)"
        )
        return

    conv_fn, conv_update = make_conv_swaps()
    setattr(bm3, "fused_kda_gate", kda_gate_torch)
    setattr(bm3, "chunk_kda", chunk_kda_310p)
    setattr(bm3, "fused_recurrent_kda", fused_recurrent_kda_310p)
    setattr(bm3, "fused_recurrent_kda_fwd", fused_recurrent_kda_fwd_310p)
    setattr(bm3, "causal_conv1d_fn", conv_fn)
    setattr(bm3, "causal_conv1d_update", conv_update)

    # Keep the FLA module in sync for any other import site.
    try:
        import vllm.third_party.flash_linear_attention.ops.kda as fla_kda

        for name, fn in (
            ("fused_kda_gate", kda_gate_torch),
            ("chunk_kda", chunk_kda_310p),
            ("fused_recurrent_kda", fused_recurrent_kda_310p),
        ):
            if hasattr(fla_kda, name):
                setattr(fla_kda, name, fn)
    except Exception as exc:  # noqa: BLE001 - best effort sync
        logger.debug("Ming3 310P KDA patch: FLA module sync skipped (%s)", exc)

    _INSTALLED = True
    logger.info("Ming3 310P KDA torch ops installed on bailing_moe_v3")
