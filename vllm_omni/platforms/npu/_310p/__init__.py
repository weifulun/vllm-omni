# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P platform helpers."""

from __future__ import annotations

import torch


def is_310p(device: torch.device | None = None) -> bool:
    """Return True on Ascend 310P devices, False otherwise.

    Dual import path (mossfull 5a5a49692 precedent / ENV-PRECHECK P1-4):
    this vllm_ascend build exposes ``is_310p`` under
    ``vllm_ascend.device.device_config`` while older/newer builds put it in
    ``vllm_ascend.utils``.  A single-path import silently degrades to False
    and the entire 310P patch stack never installs — assert in-container
    (fresh interpreter) after container setup.
    """
    if device is not None and device.type != "npu":
        return False
    try:
        from vllm_ascend.utils import is_310p as ascend_is_310p

        return bool(ascend_is_310p())
    except (ImportError, AttributeError):
        pass
    try:
        from vllm_ascend.device.device_config import is_310p as ascend_is_310p

        return bool(ascend_is_310p())
    except (ImportError, AttributeError):
        return False


def disable_jit_compile() -> None:
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)


def apply_patches() -> None:
    if not is_310p():
        return

    from vllm_omni.platforms.npu._310p.patch import apply_patches as apply_310p_patches

    apply_310p_patches()
