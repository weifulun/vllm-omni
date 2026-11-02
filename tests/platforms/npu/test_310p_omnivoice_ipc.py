# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract tests for the 310P NPU-aware output-transport fix in
``vllm_omni/diffusion/ipc.py`` (task-23).

Background (machine-verified on 126/C3, 2026-11-03): the diffusion result
MessageQueue pickles whatever SHM packing leaves behind.  A sub-threshold NPU
tensor (e.g. a 24kHz TTS waveform, ~350KB) rides the payload raw; pickling an
NPU tensor forces the legacy storage serialization
(``torch/storage.py __reduce__`` -> ``torch.save(_use_new_zipfile_
serialization=False)`` -> ``_legacy_save`` -> ``storage._write_file``), which
torch_npu cannot write for NPU storages
(``RuntimeError: unknown format type:1869375090``).  CUDA storages pickle
fine, which is why upstream never hit this.

The fix: ``_tensor_storage_unpicklable`` routes every NPU tensor through
D2H+SHM regardless of the size threshold.  These tests run on host CPU by
loading ``ipc.py`` as a standalone module with its heavy ``vllm_omni``
dependencies faked.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import typing

import pytest
import torch

REPO_ROOT = str(__file__.rsplit("/tests/", 1)[0])

# The production host CI box may run Python 3.9, where typing.TypeGuard
# (imported at module scope by ipc.py) does not exist yet.  Backfill it so the
# standalone load works; the container (py3.12) keeps the real one.
if not hasattr(typing, "TypeGuard"):
    typing.TypeGuard = typing.Any  # type: ignore[attr-defined]


def _load_ipc_module(monkeypatch: pytest.MonkeyPatch):
    """Load ``vllm_omni/diffusion/ipc.py`` standalone with faked deps."""
    logger_mod = types.ModuleType("vllm.logger")

    def _init_logger(name: str):
        import logging

        return logging.getLogger(name)

    logger_mod.init_logger = _init_logger
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.logger = logger_mod

    data_mod = types.ModuleType("vllm_omni.diffusion.data")

    class DiffusionOutput:  # minimal stand-in (unused by the packing core)
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    data_mod.DiffusionOutput = DiffusionOutput

    media_mod = types.ModuleType("vllm_omni.diffusion.media")
    for name in (
        "DiffusionMediaOutput",
        "FloatVideoConsumer",
        "VideoColorModel",
        "VideoMediaOutput",
        "VideoTensorEncoding",
        "VideoTensorLayout",
        "VideoTensorSpec",
        "VideoTransportConstraints",
        "VideoValueRange",
    ):
        setattr(media_mod, name, type(name, (), {}))

    omni_mod = types.ModuleType("vllm_omni")
    diffusion_mod = types.ModuleType("vllm_omni.diffusion")
    for name, mod in {
        "vllm": vllm_mod,
        "vllm.logger": logger_mod,
        "vllm_omni": omni_mod,
        "vllm_omni.diffusion": diffusion_mod,
        "vllm_omni.diffusion.data": data_mod,
        "vllm_omni.diffusion.media": media_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    path = f"{REPO_ROOT}/vllm_omni/diffusion/ipc.py"
    spec = importlib.util.spec_from_file_location("ipc_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _tensor_storage_unpicklable predicate
# ---------------------------------------------------------------------------


def test_unpicklable_predicate_devices(monkeypatch):
    ipc = _load_ipc_module(monkeypatch)
    assert ipc._tensor_storage_unpicklable(torch.zeros(4)) is False

    class _FakeDevice:
        def __init__(self, dev_type: str):
            self.type = dev_type

    class _FakeTensor:
        def __init__(self, dev_type: str):
            self.device = _FakeDevice(dev_type)

    assert ipc._tensor_storage_unpicklable(_FakeTensor("npu")) is True
    assert ipc._tensor_storage_unpicklable(_FakeTensor("cuda")) is False
    assert ipc._tensor_storage_unpicklable(_FakeTensor("cpu")) is False


# ---------------------------------------------------------------------------
# _pack_tensor_if_large routing
# ---------------------------------------------------------------------------


def test_small_cpu_tensor_passes_through(monkeypatch):
    ipc = _load_ipc_module(monkeypatch)
    small = torch.zeros(256)  # 2KB fp32, far below the 1MB threshold
    assert ipc._pack_tensor_if_large(small) is small


def test_small_npu_tensor_routed_to_shm(monkeypatch):
    ipc = _load_ipc_module(monkeypatch)
    small = torch.zeros(256)
    monkeypatch.setattr(ipc, "_tensor_storage_unpicklable", lambda t: True)
    handle = ipc._pack_tensor_if_large(small)
    assert isinstance(handle, dict)
    assert handle.get("__tensor_shm__") is True
    # Round-trip must restore identical values/dtype/device-on-host.
    restored = ipc._tensor_from_shm(handle)
    assert torch.equal(restored, small)
    assert restored.dtype == small.dtype


def test_small_cpu_tensor_unaffected_by_predicate_false(monkeypatch):
    ipc = _load_ipc_module(monkeypatch)
    small = torch.zeros(256)
    monkeypatch.setattr(ipc, "_tensor_storage_unpicklable", lambda t: False)
    assert ipc._pack_tensor_if_large(small) is small


def test_large_tensor_still_shm_routed(monkeypatch):
    """Regression: the size-threshold path predates the NPU fix and must
    keep working (a >1MB CPU tensor is SHM-packed even with the predicate
    disabled)."""
    ipc = _load_ipc_module(monkeypatch)
    big = torch.zeros(300_000)  # 1.2MB fp32 > 1MB threshold
    monkeypatch.setattr(ipc, "_tensor_storage_unpicklable", lambda t: False)
    handle = ipc._pack_tensor_if_large(big)
    assert isinstance(handle, dict)
    assert handle.get("__tensor_shm__") is True
    restored = ipc._tensor_from_shm(handle)
    assert torch.equal(restored, big)
