# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-runnable contract tests for the Boogu MLLM CPU pin (task-33).

Exercises ``_CpuPinnedMLLM`` / ``install_mllm_cpu_pin`` semantics with a fake
encoder module - no NPU, no vllm-ascend, no real Qwen3-VL weights:

- tensor inputs are moved to CPU on entry and the wrapped encoder is called
  on CPU;
- ``_apply`` / ``to`` pinning: a pipeline-wide device move cannot drag the
  encoder off CPU;
- attribute delegation (``dtype``, ``config``) reaches the inner module;
- the installer wraps ``BooguImagePipeline._load_mllm`` idempotently.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_pin_module():
    spec = importlib.util.spec_from_file_location(
        "boogu_mllm_pin_under_test",
        _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "boogu_mllm_pin.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEncoder(nn.Module):
    """Stands in for Qwen3VLModel: records the device its forward ran on."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.forward_devices: list[str] = []

    @property
    def dtype(self):
        return self.lin.weight.dtype

    @property
    def arch_tag(self):
        return "fake-encoder"

    def forward(self, x):
        self.forward_devices.append(str(x.device))
        return self.lin(x)


@pytest.fixture()
def pin():
    return _load_pin_module()


class TestCpuPinnedMLLM:
    def test_forward_moved_to_cpu(self, pin):
        wrapper = pin._CpuPinnedMLLM(_FakeEncoder())
        x_npu = torch.randn(2, 4)  # "on-device" activation (CPU tensor here)
        out = wrapper(x_npu)
        assert wrapper._mllm.forward_devices == ["cpu"]
        assert out.shape == (2, 4)

    def test_apply_pin_blocks_pipeline_wide_move(self, pin):
        wrapper = pin._CpuPinnedMLLM(_FakeEncoder())
        # Simulate a pipeline-wide .to(): params must stay put (data_ptr and
        # device are the identity - ``.data`` is not stable across accesses
        # on newer torch).
        before_ptr = wrapper._mllm.lin.weight.data_ptr()
        before_dev = wrapper._mllm.lin.weight.device
        wrapper.to("cpu")
        assert wrapper._mllm.lin.weight.data_ptr() == before_ptr
        # _apply recursion must stop at the wrapper.
        wrapper._apply(lambda t: t)
        assert wrapper._mllm.lin.weight.data_ptr() == before_ptr
        assert wrapper._mllm.lin.weight.device == before_dev

    def test_attribute_delegation(self, pin):
        inner = _FakeEncoder()
        wrapper = pin._CpuPinnedMLLM(inner)
        assert wrapper.dtype == inner.dtype
        assert wrapper.arch_tag is inner.arch_tag  # non-colliding attr name

    def test_wrap_moves_encoder_to_cpu(self, pin):
        wrapper = pin._CpuPinnedMLLM(_FakeEncoder())
        assert wrapper._mllm.lin.weight.device.type == "cpu"
        assert wrapper._mllm.training is False


def _fake_pipeline_module(monkeypatch: pytest.MonkeyPatch):
    """Fakes-first stand-in for the boogu_image pipeline package so the
    installer can be exercised without the real (NPU-bound) import chain."""

    class FakeBooguImagePipeline:
        def _load_mllm(self, model, local_files_only, mllm_quant_config, prefetch_list):
            return _FakeEncoder()

    pkg = types.ModuleType("vllm_omni.diffusion.models.boogu_image")
    pipeline_mod = types.ModuleType("vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image")
    pipeline_mod.BooguImagePipeline = FakeBooguImagePipeline
    pkg.pipeline_boogu_image = pipeline_mod
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.models.boogu_image", pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image", pipeline_mod)
    return FakeBooguImagePipeline


class TestInstallMLLMCpuPin:
    def test_install_wraps_load_mllm_idempotently(self, pin, monkeypatch):
        FakeBooguImagePipeline = _fake_pipeline_module(monkeypatch)
        installed = pin.install_mllm_cpu_pin()
        assert installed is True
        assert pin.install_mllm_cpu_pin() is True  # idempotent
        assert getattr(FakeBooguImagePipeline._load_mllm, "_310p_cpu_pinned", False) is True

    def test_wrapped_load_returns_pinned_module(self, pin, monkeypatch):
        FakeBooguImagePipeline = _fake_pipeline_module(monkeypatch)
        pin.install_mllm_cpu_pin()
        self_obj = types.SimpleNamespace()
        result = FakeBooguImagePipeline._load_mllm(self_obj, "/fake", True, None, ["mllm"])
        assert isinstance(result, pin._CpuPinnedMLLM)
        assert result._mllm.forward_devices == []
        out = result(torch.randn(1, 4))
        assert result._mllm.forward_devices == ["cpu"]
        assert out.shape == (1, 4)
