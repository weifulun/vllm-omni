# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P platform helpers."""

from __future__ import annotations

import torch


def is_310p(device: torch.device | None = None) -> bool:
    """Return True on Ascend 310P devices, False otherwise."""
    if device is not None and device.type != "npu":
        return False
    try:
        from vllm_ascend.utils import is_310p as ascend_is_310p

        return bool(ascend_is_310p())
    except (ImportError, AttributeError):
        # API drift: some vllm_ascend builds expose the build-info based
        # detector under vllm_ascend.device.device_config instead of
        # vllm_ascend.utils; without the fallback the whole 310P patch
        # stack no-ops silently.
        try:
            from vllm_ascend.device.device_config import (
                is_310p as ascend_is_310p_fallback,
            )

            return bool(ascend_is_310p_fallback())
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
