# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Omni NPU worker device initialization for 310P.

This patches:
    vllm_omni.platforms.npu.worker.base.OmniNPUWorkerBase

Omni AR/generation workers share this base initialization path, so the 310P
compile-mode setup is applied here after device initialization.
"""

from __future__ import annotations

import logging
import os

from vllm.v1.sample.sampler import Sampler
from vllm_ascend._310p.sample.sampler import AscendSampler310

from vllm_omni.platforms.npu._310p import disable_jit_compile
from vllm_omni.platforms.npu.worker import base as worker_base

logger = logging.getLogger(__name__)


class _OmniNPUWorkerBase310P(worker_base.OmniNPUWorkerBase):
    def _init_device(self):
        device = super()._init_device()
        disable_jit_compile()
        return device


def apply_patch() -> None:
    # Triton-Ascend does not target 310P; use vLLM's native penalty path.
    AscendSampler310.apply_penalties = staticmethod(Sampler.apply_penalties)
    worker_base.OmniNPUWorkerBase = _OmniNPUWorkerBase310P
    _pin_fp16_default_dtype()


def _pin_fp16_default_dtype() -> None:
    """310P hardware invariant: FP16-only, no BF16 elementwise kernels.

    aclnn has no BF16 support for initializer kernels (``aclnnInplaceOne``
    fails with "Tensor self not implemented for DT_BFLOAT16").  Component
    construction under the engine's default (BF16) dtype leaks NPU-side
    parameter inits (e.g. Qwen3 RMSNorm ``torch.ones`` in the Z-Image text
    encoder) onto the chip and crashes the pipeline build.  Pinning the
    process default dtype to FP16 at platform-construction time (every
    spawned worker re-runs platform construction, so this is per-process)
    makes stray parameter/tensor creation FP16 everywhere on 310P; models
    that need FP32 (schedulers, norms) set their dtype explicitly.
    ``VLLM_OMNI_310P_DEFAULT_FP16=0`` opts out (A/B experiments only).
    """
    if os.environ.get("VLLM_OMNI_310P_DEFAULT_FP16", "1") != "1":
        return
    import torch

    if torch.get_default_dtype() is not torch.float16:
        logger.info("[310P] pinning torch default dtype to float16 (FP16-only hardware)")
        torch.set_default_dtype(torch.float16)
