# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Omni NPU worker device initialization for 310P.

This patches:
    vllm_omni.platforms.npu.worker.base.OmniNPUWorkerBase

Omni AR/generation workers share this base initialization path, so the 310P
compile-mode setup is applied here after device initialization.
"""

from __future__ import annotations

from vllm.v1.sample.sampler import Sampler
from vllm_ascend._310p.sample.sampler import AscendSampler310

from vllm_omni.platforms.npu._310p import disable_jit_compile
from vllm_omni.platforms.npu.worker import base as worker_base


class _OmniNPUWorkerBase310P(worker_base.OmniNPUWorkerBase):
    def _init_device(self):
        device = super()._init_device()
        disable_jit_compile()
        return device


def _install_empty_cache_noop() -> None:
    """310P EH0012 workaround for the standard image (model-mossfull, 2026-11-02).

    On the standard 310P image, the control graph submitted by
    ``torch_npu``'s ``empty_cache()`` poisons later ``aclrt`` allocator
    lookups on the multi-worker (TP>1) path:
    ``Invalid_Argument(EH0012): aclrtAllocatorGetByStream failed ... The
    stream is not registered with any allocator`` is raised around the
    KV-cache reshape op (``ReshapeCacheOperation``) during worker
    initialization. Skipping the flush is semantically lossless
    (``empty_cache`` only releases uncached allocator blocks) and avoids
    the defective control-graph submission. Installed on
    ``torch.accelerator.empty_cache`` (vLLM's model-loader call site) and
    ``torch.npu.empty_cache`` (the underlying implementation).
    """
    import logging

    import torch

    logger = logging.getLogger(__name__)

    def _wrap(name: str, module) -> None:
        fn = getattr(module, "empty_cache", None)
        if fn is None or getattr(fn, "_moss_310p_ec_noop", False):
            return

        def noop(*args, **kwargs):
            logger.info(
                "[310P] skipping empty_cache (%s) - EH0012 workaround", name
            )

        noop._moss_310p_ec_noop = True
        setattr(module, "empty_cache", noop)
        logger.info("[310P] empty_cache no-op installed for %s", name)

    _wrap("torch.accelerator", torch.accelerator)
    try:
        import torch.npu as _npu

        _wrap("torch.npu", _npu)
    except ImportError:  # torch_npu optional at import time
        pass


def apply_patch() -> None:
    # Triton-Ascend does not target 310P; use vLLM's native penalty path.
    AscendSampler310.apply_penalties = staticmethod(Sampler.apply_penalties)
    worker_base.OmniNPUWorkerBase = _OmniNPUWorkerBase310P
    _install_empty_cache_noop()
