# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P attention backend patch: force torch's MATH SDPA decomposition.

On 310P (CANN 9.1.0 + torch 2.10.0+cpu / torch_npu 2.10.0.post4), the fused
NPU implementation of ``F.scaled_dot_product_attention`` is unusable: every
launch fails with ``can not cast format when output is input`` for fp16 and
fp32, all head geometries, all sequence lengths (repro: single conv-free
process, s=64..16384, heads=5/d=64).  The pipeline-level symptom is a GE
graph-build failure inside the attention sub-graph (dynamic MemSet
pre-compilation error, ops_legacy/dynamic/mem_set.py).

The math decomposition (scale -> bmm -> softmax -> bmm) is verified working
at the full SDXL attention shapes (16384 tokens, fp16) on the same card.

Upstream ``sdpa._sdpa_attention`` already knows about this class of issue:
it forces ``sdpa_kernel(SDPBackend.MATH)`` when a float bias must be
preserved.  This patch widens that to *all* calls on 310P by wrapping the
module-level function (vllm-omni side only; image stays read-only).
Knob: ``VLLM_OMNI_310P_FORCE_MATH_SDPA`` (default 1).
"""

from __future__ import annotations

import os

_PATCHED = False


def maybe_patch_sdpa_math() -> None:
    """Wrap ``sdpa._sdpa_attention`` so every SDPA call uses the MATH kernel."""
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("VLLM_OMNI_310P_FORCE_MATH_SDPA", "1") != "1":
        return

    from torch.nn.attention import SDPBackend, sdpa_kernel

    import vllm_omni.diffusion.attention.backends.sdpa as _sdpa

    if getattr(_sdpa, "_310p_math_patched", False):
        _PATCHED = True
        return

    _orig = _sdpa._sdpa_attention

    def _math_forced(query, key, value, attention_mask, causal, scale):
        # ``_orig`` may re-enter the MATH context for float-bias inputs; the
        # context manager is nestable, and otherwise MATH simply wins.
        with sdpa_kernel(SDPBackend.MATH):
            return _orig(query, key, value, attention_mask, causal, scale)

    _sdpa._sdpa_attention = _math_forced
    _sdpa._310p_math_patched = True  # type: ignore[attr-defined]
    _PATCHED = True
