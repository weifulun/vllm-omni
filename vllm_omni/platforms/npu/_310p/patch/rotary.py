# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P AR-worker rotary fix: route MRoPE to the torch-native implementation.

The Qwen2.5-Omni thinker stage of AuK uses multimodal rotary positions
(``MRotaryEmbedding``).  On this CANN 9.1.0 / torch_npu build the
``vllm_ascend`` M-RoPE fast path dispatches to the aclnn kernel
``RopeWithSinCosCacheV2`` (``torch_npu.npu_mrope``), which has no 310P
implementation at all:

    AclNN_Parameter_Error(EZ1001): support for Ascend310P is not implemented
    Tensor queryIn not implemented for DT_FLOAT16, should be in dtype
    support list [].

Observed on node 128 (task-24): the stage-0 engine core dies during the
profiling dummy run, before any request is served.  The dtype support list is
empty for every dtype, so no cast around the call can help.

The ``vllm_ascend`` wrapper already delegates to vLLM's torch-native
``MRotaryEmbedding.forward_oot`` (plain indexing / trig / mul ops, all
implemented on 310P) whenever the section is not the fast-path default, so
the 310P-correct fix is to make the fast-path branch take that same
delegation.  The patch is installed from the AuK thinker model class
(``AuKForConditionalGeneration.__init__``) — which runs in the AR engine
core process — before the thinker layers (and their rotary instances) are
constructed.  It is idempotent and a no-op off-NPU / off-310P.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["maybe_patch_310p_mrope"]

_PATCHED = False


def maybe_patch_310p_mrope() -> None:
    """Force the torch-native MRoPE path for AR workers on 310P (no-op elsewhere)."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import torch_npu  # noqa: F401
    except Exception:
        return  # not an NPU process

    from vllm_omni.platforms.npu._310p import is_310p

    if not is_310p():
        return

    try:
        from vllm_ascend.ops import rotary_embedding as _ascend_rotary
    except ImportError:
        return

    cls = _ascend_rotary.AscendMRotaryEmbedding
    if getattr(cls, "_auk_310p_mrope_patched", False):
        _PATCHED = True
        return

    # IMPORTANT: resolve the torch-native implementation through the MRO, not
    # through ``from vllm... import MRotaryEmbedding`` — both vllm-ascend and
    # vllm-omni swap that module-level symbol (the name points at their own
    # subclasses), so a symbol import silently re-fetches the aclnn-based
    # implementation and the assignment becomes a no-op (observed node 128:
    # patch log fires, identical npu_mrope crash).  The MRO is fixed at class
    # definition time: the first non-ascend base IS vLLM's own MRotaryEmbedding,
    # whose forward_oot delegates to forward_native (pure torch).
    base = None
    for cand in cls.__mro__[1:]:
        if cand.__name__ == "MRotaryEmbedding" and cand.__module__.startswith(
            "vllm.model_executor"
        ):
            base = cand
            break
    if base is None:
        logger.warning(
            "[310P] MRoPE patch: no vLLM MRotaryEmbedding base found in MRO; "
            "leaving the aclnn fast path in place."
        )
        return
    cls.forward_oot = base.forward_oot
    cls._auk_310p_mrope_patched = True
    _PATCHED = True
    logger.info(
        "[310P] MRoPE routed to torch-native forward_oot via %s.forward_oot "
        "(aclnn RopeWithSinCosCacheV2 has no 310P kernel, EZ1001).",
        base.__module__,
    )
