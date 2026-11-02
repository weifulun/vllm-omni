# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P platform helpers."""

from __future__ import annotations

import torch


def is_310p(device: torch.device | None = None) -> bool:
    """Return True on Ascend 310P devices, False otherwise.

    Older vllm-ascend builds expose ``vllm_ascend.utils.is_310p``; the
    v0.32.0-310p image builds it in ``vllm_ascend.device.device_config``
    only (machine-verified 2026-11-03 on 126 C3: ``utils`` import raises
    ImportError, ``device_config.is_310p()`` returns True). Without the
    fallback this helper silently reports False and the whole 310P patch
    surface is skipped.
    """
    if device is not None and device.type != "npu":
        return False
    try:
        from vllm_ascend.utils import is_310p as ascend_is_310p

        return bool(ascend_is_310p())
    except ImportError:
        pass
    except AttributeError:
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
