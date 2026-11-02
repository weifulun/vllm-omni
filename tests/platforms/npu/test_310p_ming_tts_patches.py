# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the 310P Ming-3.0-TTS patch.

Same approach as ``test_310p_patches.py``: the patch module is loaded from
source with a fake ``ming_tts_llm`` dependency, so the wiring (registry
dispatch, FP16 pin, CFM-graph contract, deploy overlay) is validated
without an NPU, model checkpoint or the real vLLM runtime.

Real-device behavior (eager flow head numerics, audio-VAE decode dtype
coercion, first-chunk latency) is qualified in the 310P real-device phase
of task-27.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _FakeMingLLMBase(torch.nn.Module):
    """Stand-in for ``ming_tts_llm.MingLLMModel`` (same __init__ contract)."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        resolved = _resolve_fake_runtime_dtype(vllm_config)
        self.fm_dtype = resolved
        self.flowloss = torch.nn.Module()
        self.flowloss.fc = torch.nn.Linear(8, 8)
        self.linear_proj_audio = torch.nn.Linear(8, 4)
        self.stop_head = torch.nn.Linear(4, 2)
        self.spk_head = torch.nn.Linear(16, 4)
        self.flowloss.to(dtype=self.fm_dtype)
        self.linear_proj_audio.to(dtype=self.fm_dtype)
        self.stop_head.to(dtype=self.fm_dtype)
        self.spk_head.to(dtype=self.fm_dtype)
        self._cfm_graph = None
        self._cfm_graph_enabled = True  # base default: enabled

    def _maybe_build_cfm_graph(self, z_diff_cond):
        if not self._cfm_graph_enabled or not z_diff_cond.is_cuda:
            return None
        return SimpleNamespace(executed=True)


def _resolve_fake_runtime_dtype(vllm_config) -> torch.dtype:
    """Mirror of ming_tts_llm._resolve_ming_runtime_dtype for the fake config."""
    dtype = getattr(vllm_config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.strip().lower()
        if normalized in ("float16", "half", "torch.float16"):
            return torch.float16
        if normalized in ("bfloat16", "bf16", "torch.bfloat16"):
            return torch.bfloat16
        if normalized in ("float32", "fp32", "torch.float32"):
            return torch.float32
    return torch.float32


def _install_package_chain(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Install empty parent packages so lazy dotted imports resolve."""
    for name in names:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def _load_ming_patch(monkeypatch: pytest.MonkeyPatch):
    _install_package_chain(
        monkeypatch,
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.model_executor.models.ming_tts",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
        "vllm_omni.platforms.npu._310p.patch",
    )
    fake_llm = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.ming_tts.ming_tts_llm",
        MingLLMModel=_FakeMingLLMBase,
    )
    root = _repo_root()
    patch = _load_source_module(
        "ming_tts_patch_under_test", root / "vllm_omni/platforms/npu/_310p/patch/ming_tts.py"
    )
    return patch, fake_llm


def _fake_vllm_config(dtype: object) -> SimpleNamespace:
    return SimpleNamespace(dtype=dtype)


class TestMingTTS310PClass:
    def test_subclass_and_pin(self, monkeypatch: pytest.MonkeyPatch):
        patch, _ = _load_ming_patch(monkeypatch)
        assert issubclass(patch._MingLLMModel310P, _FakeMingLLMBase)

        # BF16 upstream config: the 310P pin must still land on FP16.
        model = patch._MingLLMModel310P(vllm_config=_fake_vllm_config("bfloat16"))
        assert model.fm_dtype is torch.float16
        for submodule in (model.flowloss, model.linear_proj_audio, model.stop_head, model.spk_head):
            params = [p.dtype for p in submodule.parameters()]
            assert params and all(d is torch.float16 for d in params)

        # CFM graph contract: hard off on 310P, even for a "cuda" tensor.
        assert model._cfm_graph_enabled is False
        assert model._cfm_graph is None
        fake_cuda = torch.ones(1, 8).cuda() if torch.cuda.is_available() else torch.ones(1, 8)
        assert model._maybe_build_cfm_graph(fake_cuda) is None

    def test_fp16_deploy_config_is_idempotent(self, monkeypatch: pytest.MonkeyPatch):
        patch, _ = _load_ming_patch(monkeypatch)
        model = patch._MingLLMModel310P(vllm_config=_fake_vllm_config("float16"))
        assert model.fm_dtype is torch.float16
        assert model._cfm_graph_enabled is False


class TestApplyMingTTSPatches:
    def test_registry_swap_and_idempotency(self, monkeypatch: pytest.MonkeyPatch):
        patch, fake_llm = _load_ming_patch(monkeypatch)
        assert fake_llm.MingLLMModel is _FakeMingLLMBase
        patch.apply_ming_tts_patches()
        assert fake_llm.MingLLMModel is patch._MingLLMModel310P
        # Second call is a no-op (same object, no re-import).
        patch.apply_ming_tts_patches()
        assert fake_llm.MingLLMModel is patch._MingLLMModel310P


class TestDispatch:
    def test_arch_dispatch_routes_to_ming_patch(self, monkeypatch: pytest.MonkeyPatch):
        patch, fake_llm = _load_ming_patch(monkeypatch)
        root = _repo_root()
        init_mod = _load_source_module(
            "patch_init_under_test", root / "vllm_omni/platforms/npu/_310p/patch/__init__.py"
        )
        # The dispatch lazy-imports the real package path; point it at the
        # source-loaded module instead.
        monkeypatch.setitem(
            sys.modules,
            "vllm_omni.platforms.npu._310p.patch.ming_tts",
            patch,
        )
        init_mod.apply_model_patches(SimpleNamespace(model_arch="MingTTSForConditionalGeneration"))
        assert fake_llm.MingLLMModel is patch._MingLLMModel310P

    def test_unknown_arch_is_a_noop(self, monkeypatch: pytest.MonkeyPatch):
        _load_ming_patch(monkeypatch)
        root = _repo_root()
        init_mod = _load_source_module(
            "patch_init_under_test2", root / "vllm_omni/platforms/npu/_310p/patch/__init__.py"
        )
        init_mod.apply_model_patches(SimpleNamespace(model_arch="SomeOtherArch"))
        # No ming patch installed (module absent from sys.modules import chain).
        assert "vllm_omni.platforms.npu._310p.patch.ming_tts" not in sys.modules


class TestDeployOverlay:
    def test_310p_overlay_pins_fp16_topology(self):
        root = _repo_root()
        overlay = root / "vllm_omni" / "deploy" / "ming_tts_310p.yaml"
        base = root / "vllm_omni" / "deploy" / "ming_tts.yaml"
        assert overlay.is_file(), "missing 310P deploy overlay"

        import yaml

        with overlay.open() as f:
            cfg = yaml.safe_load(f)
        with base.open() as f:
            base_cfg = yaml.safe_load(f)

        assert cfg["dtype"] == "float16", "310P is FP16-only; overlay must pin float16"
        assert base_cfg["dtype"] == "bfloat16", "base deploy stays BF16 for GPU lines"
        assert cfg["pipeline"] == base_cfg["pipeline"] == "ming_tts"
        assert cfg["async_chunk"] is True
        assert [s["stage_id"] for s in cfg["stages"]] == [0, 1]
        for stage in cfg["stages"]:
            assert stage["devices"] == "0", "single-chip 310P: both stages on logical device 0"
            assert stage["enforce_eager"] is True
        # Connector geometry must match the base pipeline (latent chunk sizes).
        extra = cfg["connectors"]["connector_of_shared_memory"]["extra"]
        base_extra = base_cfg["connectors"]["connector_of_shared_memory"]["extra"]
        assert extra == base_extra
