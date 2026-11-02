# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the 310P IndexTTS 2(.5) patch.

Same approach as ``test_310p_voxcpm2_patches.py``: the patch module is loaded
from source with fake dependencies (``vllm.logger``, the S2Mel decoder class,
and the ``is_310p`` probe), so the wiring (registry dispatch, FP16-only
runtime forcing, idempotency) is validated without an NPU, a model
checkpoint or the native external models.

Real-device behavior (FP32 CFM numerics, BigVGAN FP32 eager latency,
TBE warmup cost on dynamic shapes) is qualified in the 310P real-device
phase of task-26.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PATCH_PACKAGE_MARKER = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / PATCH_PACKAGE_MARKER).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {PATCH_PACKAGE_MARKER}")


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


class FakeS2MelDecoder(nn.Module):
    """Stands in for IndexTTS2S2MelDecoder with the 910B-line defaults.

    Mirrors the real ``__init__`` contract: config switches default to the
    GPU/bf16 line (bf16 on, torch.compile on, CUDA graph off) exactly as
    ``deploy/indextts2_5.yaml`` pins them.
    """

    constructed: list["FakeS2MelDecoder"] = []

    def __init__(self, *, vllm_config=None, prefix: str = "") -> None:
        del vllm_config, prefix
        super().__init__()
        self.s2mel_dit_bf16 = True
        self.s2mel_vocoder_bf16 = True
        self.s2mel_vocoder_torch_compile = True
        self.s2mel_dit_cuda_graph = False
        self.s2mel_dit_torch_compile = True
        self.s2mel_vocoder_cuda_graph = True
        FakeS2MelDecoder.constructed.append(self)


_SWITCHES = (
    "s2mel_dit_bf16",
    "s2mel_vocoder_bf16",
    "s2mel_dit_torch_compile",
    "s2mel_dit_cuda_graph",
    "s2mel_vocoder_torch_compile",
    "s2mel_vocoder_cuda_graph",
)


# Captured before any test runs: the pristine (unwrapped) decoder init.
_PRISTINE_INIT = FakeS2MelDecoder.__init__


@pytest.fixture()
def patch_env(monkeypatch: pytest.MonkeyPatch):
    """Install the fake dependency surface and load the patch modules from source."""
    monkeypatch.setattr(FakeS2MelDecoder, "constructed", [])
    # The 310P patch wraps ``__init__`` on the shared class object; restore the
    # pristine version between tests so each dispatch starts clean.
    monkeypatch.setattr(FakeS2MelDecoder, "__init__", _PRISTINE_INIT)

    # vllm.logger (the patch module's only vllm import)
    import logging

    _install_fake_module(
        monkeypatch,
        "vllm",
    )
    _install_fake_module(
        monkeypatch,
        "vllm.logger",
        init_logger=lambda name: logging.getLogger(name),
    )

    # Fake parent packages so the import machinery never executes the real
    # package __init__ chain (no vllm / vllm_ascend on this host)
    for parent in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.model_executor.models.indextts2",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
    ):
        if parent not in sys.modules:
            _install_fake_module(monkeypatch, parent)

    # Controllable is_310p probe under the fake _310p package
    probe = {"is_310p": True}
    sys.modules["vllm_omni.platforms.npu._310p"].is_310p = (
        lambda device=None: probe["is_310p"]
    )

    root = _repo_root()
    patch_dir = root / PATCH_PACKAGE_MARKER

    # The shared decoder module the patch imports lazily at apply time
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder",
        IndexTTS2S2MelDecoder=FakeS2MelDecoder,
    )

    # Sibling patch modules referenced by the dispatch (only worker/qwen3_tts
    # branches are ever reached by the qwen3_tts arches; provide inert fakes so
    # any dispatch can resolve)
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.worker",
        apply_patch=lambda: None,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=lambda: None,
        apply_code2wav_patches=lambda: None,
    )

    # Source-load the IndexTTS patch module and the dispatch package
    indextts2_mod = _load_source_module(
        "vllm_omni.platforms.npu._310p.patch.indextts2",
        patch_dir / "indextts2.py",
    )
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.indextts2", **{
        k: v for k, v in vars(indextts2_mod).items() if not k.startswith("__")
    })
    dispatch_mod = _load_source_module(
        "vllm_omni.platforms.npu._310p.patch",
        patch_dir / "__init__.py",
    )
    return SimpleNamespace(probe=probe, dispatch=dispatch_mod, indextts2=indextts2_mod)


def test_dispatch_wraps_s2mel_decoder_for_2_5_arch(patch_env):
    dispatch = patch_env.dispatch
    init_before = FakeS2MelDecoder.__init__
    dispatch.apply_model_patches(SimpleNamespace(model_arch="IndexTTS25S2MelDecoder"))
    assert FakeS2MelDecoder.__init__ is not init_before, "decoder __init__ must be wrapped"
    assert getattr(FakeS2MelDecoder.__init__, "_indextts2_310p_patched", False)


def test_dispatch_also_covers_indextts2_arch(patch_env):
    dispatch = patch_env.dispatch
    init_before = FakeS2MelDecoder.__init__
    dispatch.apply_model_patches(SimpleNamespace(model_arch="IndexTTS2S2MelDecoder"))
    assert FakeS2MelDecoder.__init__ is not init_before


def test_talker_arch_is_traceability_only(patch_env):
    """Stage 0 (GPT-2 AR) needs no code patch: dispatch must not wrap the decoder."""
    dispatch = patch_env.dispatch
    init_before = FakeS2MelDecoder.__init__
    dispatch.apply_model_patches(SimpleNamespace(model_arch="IndexTTS25TalkerForConditionalGeneration"))
    assert FakeS2MelDecoder.__init__ is init_before, "talker arch must not touch the decoder"


def test_unknown_arch_is_noop(patch_env):
    patch_env.dispatch.apply_model_patches(SimpleNamespace(model_arch=None))
    patch_env.dispatch.apply_model_patches(SimpleNamespace())
    assert FakeS2MelDecoder.constructed == []


def test_310p_forces_fp16_only_runtime(patch_env):
    probe = patch_env.probe
    probe["is_310p"] = True
    patch_env.dispatch.apply_model_patches(SimpleNamespace(model_arch="IndexTTS25S2MelDecoder"))
    model = FakeS2MelDecoder()
    assert len(FakeS2MelDecoder.constructed) == 1
    for field in _SWITCHES:
        assert getattr(model, field) is False, f"{field} must be forced off on 310P"


def test_non_310p_keeps_configured_runtime(patch_env):
    probe = patch_env.probe
    probe["is_310p"] = False
    patch_env.dispatch.apply_model_patches(SimpleNamespace(model_arch="IndexTTS25S2MelDecoder"))
    model = FakeS2MelDecoder()
    # 910 line: the wrapped init must leave the configured switches untouched
    assert model.s2mel_dit_bf16 is True
    assert model.s2mel_dit_torch_compile is True
    assert model.s2mel_vocoder_cuda_graph is True
    assert model.s2mel_dit_cuda_graph is False


def test_apply_is_idempotent(patch_env):
    probe = patch_env.probe
    probe["is_310p"] = True
    dispatch = patch_env.dispatch
    dispatch.apply_model_patches(SimpleNamespace(model_arch="IndexTTS25S2MelDecoder"))
    wrapped_once = FakeS2MelDecoder.__init__
    # Second dispatch round (e.g. a sibling stage re-entering the runner) must
    # not re-wrap over the already-wrapped init.
    patch_env.indextts2.apply_indextts2_310p_patches()
    assert FakeS2MelDecoder.__init__ is wrapped_once, "double-wrap detected"
    # Constructing through the once-wrapped init still forces the runtime
    FakeS2MelDecoder()
    assert FakeS2MelDecoder.constructed[0].s2mel_dit_bf16 is False


def test_deploy_overlay_contract():
    import yaml

    root = _repo_root()
    overlay_path = root / "vllm_omni" / "deploy" / "indextts2_5_310p.yaml"
    overlay = yaml.safe_load(overlay_path.read_text())
    assert overlay["base_config"] == "indextts2_5.yaml"
    assert overlay["dtype"] == "float16"
    stages = {s["stage_id"]: s for s in overlay.get("stages", [])}
    hf = stages[1]["hf_overrides"]
    for field in _SWITCHES:
        assert hf[field] is False, f"overlay must pin {field}: false"
    # The base config is the one that needs the override (document the
    # relationship so a future base edit cannot silently re-enable bf16).
    base = yaml.safe_load((root / "vllm_omni" / "deploy" / "indextts2_5.yaml").read_text())
    base_stages = {s["stage_id"]: s for s in base["stages"]}
    assert base_stages[1]["hf_overrides"]["s2mel_dit_bf16"] is True
