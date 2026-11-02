# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-device 310P wiring tests for the OmniVoice 310P patch (task-23).

These run on a real 310P card inside the omni-310p-126-omnivoice-omnivoice
container and assert the device-level prerequisites the model phase depends
on:

* the runtime reports 310P and ``is_310p()`` agrees;
* the worker/model-runner 310P branches are selected;
* the NPU device RNG probe result is sane and (when usable) device rand
  works; the manual-SDPA decomposition runs on the NPU in fp16 with a dense
  boolean mask (fused SDPA is unsupported on 310P3, CANN 9.1.0);
* the CPU RoPE table moves to the NPU cleanly.

Full-pipeline numerics are qualified by the first-run benchmark
(scripts/omnivoice_first_run.py).
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm_ascend")

import torch
import torch_npu
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "310P"}, num_cards=1)]


@pytest.fixture(scope="module", autouse=True)
def require_real_310p() -> None:
    """Skip rather than fail if the runner is not actually a 310P card."""
    runtime = device_type_from_runtime_soc(torch_npu.npu.get_soc_version())
    if runtime is not AscendDeviceType._310P:
        pytest.skip(f"not a 310P host (runtime device type: {runtime})")


def test_runtime_and_package_agree_on_310p() -> None:
    runtime = device_type_from_runtime_soc(torch_npu.npu.get_soc_version())
    assert runtime is AscendDeviceType._310P
    assert get_ascend_device_type() is AscendDeviceType._310P


def test_omni_detects_310p() -> None:
    from vllm_omni.platforms.npu._310p import is_310p

    assert is_310p() is True


def test_worker_patch_is_applied() -> None:
    from vllm_omni.platforms.npu._310p import apply_patches
    from vllm_omni.platforms.npu._310p import patch as patch_registry

    apply_patches()
    assert patch_registry._WORKER_PATCHED is True


def test_npu_generator_probe() -> None:
    from vllm_omni.platforms.npu._310p.patch.omnivoice import probe_npu_rng

    ok = probe_npu_rng()
    if ok:
        device = torch.device("npu", torch.npu.current_device())
        gen = torch.Generator(device=device).manual_seed(42)
        x = torch.rand((8,), generator=gen, device=device, dtype=torch.float32)
        torch.npu.synchronize()
        assert x.shape == (8,) and torch.isfinite(x).all()
    # Either outcome is valid; the patch branches on it.  What must never
    # happen is a silent mismatch between probe and runtime behavior.
    again = probe_npu_rng()
    assert again is ok


def test_manual_sdpa_on_npu_fp16_masked() -> None:
    from vllm_omni.platforms.npu._310p.patch.omnivoice import manual_sdpa_forward

    device = torch.device("npu", torch.npu.current_device())
    torch.manual_seed(0)
    q = torch.randn(1, 4, 96, 32, dtype=torch.float16, device=device)
    k = torch.randn(1, 4, 144, 32, dtype=torch.float16, device=device)
    v = torch.randn(1, 4, 144, 32, dtype=torch.float16, device=device)
    mask = torch.ones(1, 1, 96, 144, dtype=torch.bool, device=device)
    mask[:, :, 4:, 120:] = False
    out = manual_sdpa_forward(q, k, v, mask, False, 32**-0.5)
    torch.npu.synchronize()
    assert out.dtype == torch.float16
    assert out.shape == (1, 4, 96, 32)
    assert torch.isfinite(out.float()).all()


def test_rope_table_cpu_to_npu() -> None:
    from vllm_omni.platforms.npu._310p.patch.omnivoice import _precompute_rope_table_cpu

    device = torch.device("npu", torch.npu.current_device())
    table = _precompute_rope_table_cpu(128, 4096, theta=1_000_000.0, device=device)
    assert table.device.type == "cpu"
    on_npu = table[:1024].to(device=device, dtype=torch.float16).contiguous()
    torch.npu.synchronize()
    assert on_npu.device == device
    assert on_npu.dtype == torch.float16
    assert torch.isfinite(on_npu.float()).all()
