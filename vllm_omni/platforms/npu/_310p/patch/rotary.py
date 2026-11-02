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
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
        from vllm_ascend.ops import rotary_embedding as _ascend_rotary
    except ImportError:
        return

    cls = _ascend_rotary.AscendMRotaryEmbedding
    if getattr(cls, "_auk_310p_mrope_patched", False):
        _PATCHED = True
        return
    cls.forward_oot = MRotaryEmbedding.forward_oot
    cls._auk_310p_mrope_patched = True
    _PATCHED = True
    logger.info(
        "[310P] MRoPE routed to torch-native forward_oot "
        "(aclnn RopeWithSinCosCacheV2 has no 310P kernel, EZ1001)."
    )
