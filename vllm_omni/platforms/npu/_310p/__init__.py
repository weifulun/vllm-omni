# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P platform helpers."""

from __future__ import annotations

import torch


def is_310p(device: torch.device | None = None) -> bool:
    """Return True on Ascend 310P devices, False otherwise.

    Detection is version-tolerant across vllm-ascend builds:
    1. torch_npu SoC name (``Ascend310P*``) — base-level, always present;
    2. ``vllm_ascend.utils.is_310p`` — older builds;
    3. ``vllm_ascend.utils.get_ascend_device_type() == AscendDeviceType._310P``
       — newer builds (device-config API, 0.19.1rc2 line).
    """
    if device is not None and device.type != "npu":
        return False
    try:
        import torch_npu

        name = torch_npu.npu.get_device_name(0)
        if isinstance(name, str) and name.startswith("Ascend310P"):
            return True
    except Exception:
        pass
    try:
        from vllm_ascend.utils import is_310p as ascend_is_310p

        return bool(ascend_is_310p())
    except (ImportError, AttributeError):
        pass
    try:
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        return get_ascend_device_type() == AscendDeviceType._310P
    except Exception:
        pass
    # API drift (GATE-TP2 stack item, task-12 aee740b99/5a5a49692 lineage):
    # some vllm-ascend builds expose the detector under
    # vllm_ascend.device.device_config instead of vllm_ascend.utils; without
    # this fallback the whole 310P patch stack no-ops silently.
    try:
        from vllm_ascend.device.device_config import is_310p as ascend_is_310p_cfg

        return bool(ascend_is_310p_cfg())
    except Exception:
        return False


def disable_jit_compile() -> None:
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)


def apply_patches() -> None:
    if not is_310p():
        return

    from vllm_omni.platforms.npu._310p.patch import apply_patches as apply_310p_patches

    apply_310p_patches()
