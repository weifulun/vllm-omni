# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the 310P MOSS-TTS-Nano patch (skeleton).

Loads ``_310p/patch/moss_tts_nano.py`` and the patch registry from source
with fake model/vllm-ascend dependencies, mirroring the 8B-family MOSS 310P
contract tests: the patch *wiring* and *semantics* are validated without an
NPU.

Nano-specific surface under test:
* registry routing for ``MossTTSNanoForCausalLM``;
* FP16 recast of the not-vLLM-managed remote LM + audio tokenizer;
* the scoped ``torch.multinomial`` guard: the remote LM samples text and
  audio tokens with ``torch.multinomial(probs, num_samples=1)`` on device
  tensors, which the 310P patch rewrites to
  ``vllm_ascend.sample.sampler.random_sample`` (device-global RNG).

Real-device assertions (SDPA additive-mask backend on 310P, fp16 RoPE
numerics, audio quality) live in the hardware-gated test added in the
real-device phase.
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


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSubModel(nn.Module):
    """Stand-in for the remote LM / audio tokenizer sub-models.

    Built fp32 (mirroring the 910/CUDA non-fp16 path); carries a Linear plus
    a floating-point buffer (``inv_freq``-style) so the fp16 recast is
    observable on both params and buffers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.register_buffer("inv_freq", torch.linspace(1.0, 0.1, 4))


class FakeNanoForGeneration(nn.Module):
    def __init__(self, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self._lm = _FakeSubModel()
        self._audio_tokenizer = _FakeSubModel()


def _install_nano_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    sampling_calls: list = []

    def random_sample(probs, generators):
        sampling_calls.append({"generators": generators, "probs_shape": tuple(probs.shape)})
        return torch.full((probs.shape[0],), 7, dtype=torch.long)

    # -- vllm-ascend 310P dependency (only random_sample is needed) --------
    _install_fake_module(monkeypatch, "vllm_ascend", sample=None)
    _install_fake_module(monkeypatch, "vllm_ascend.sample", sampler=None)
    _install_fake_module(
        monkeypatch,
        "vllm_ascend.sample.sampler",
        random_sample=random_sample,
    )

    # -- remote model package ------------------------------------------------
    modeling_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.moss_tts_nano.modeling_moss_tts_nano",
        MossTTSNanoForGeneration=FakeNanoForGeneration,
    )
    package = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models",
        moss_tts_nano=None,
    )
    nano_pkg = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.moss_tts_nano",
        modeling_moss_tts_nano=modeling_module,
    )
    setattr(package, "moss_tts_nano", nano_pkg)
    _install_fake_module(monkeypatch, "vllm_omni", model_executor=None)
    _install_fake_module(monkeypatch, "vllm_omni.model_executor", models=package)

    return SimpleNamespace(modeling=modeling_module, sampling_calls=sampling_calls)


def _load_nano_patch(monkeypatch: pytest.MonkeyPatch, handles):
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "moss_tts_nano.py"
    return _load_source_module("moss_tts_nano_310p_patch_under_test", path)


def _load_patch_registry(monkeypatch: pytest.MonkeyPatch, handles):
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    return _load_source_module("moss_tts_nano_310p_patch_registry_under_test", path)


def _isolated_multinomial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore torch.multinomial to whatever it is at test start (teardown)."""
    monkeypatch.setattr(torch, "multinomial", torch.multinomial)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_nano_registry_routes_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)

    calls = {"nano": 0, "qwen3": 0}

    def apply_nano_patches():
        calls["nano"] += 1

    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.worker", apply_patch=lambda: None)
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=lambda: None,
        apply_code2wav_patches=lambda: None,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.moss_tts_nano",
        apply_nano_patches=apply_nano_patches,
    )

    registry = _load_patch_registry(monkeypatch, handles)
    cfg = lambda arch: SimpleNamespace(model_arch=arch)  # noqa: E731

    registry.apply_model_patches(cfg("MossTTSNanoForCausalLM"))
    registry.apply_model_patches(cfg("MossTTSNanoForCausalLM"))
    assert calls["nano"] == 2

    # Pre-existing routes are untouched.
    registry.apply_model_patches(cfg("Qwen3TTSTalkerForConditionalGeneration"))
    registry.apply_model_patches(cfg("SomeOtherArch"))
    assert calls == {"nano": 2, "qwen3": 0}


# ---------------------------------------------------------------------------
# Class swap + idempotency
# ---------------------------------------------------------------------------


def test_apply_nano_patches_swaps_class_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)

    original_cls = handles.modeling.MossTTSNanoForGeneration
    patch.apply_nano_patches()
    swapped_cls = handles.modeling.MossTTSNanoForGeneration
    assert swapped_cls is not original_cls
    assert issubclass(swapped_cls, original_cls)
    assert swapped_cls.__name__ == "_MossTTSNano310P"

    wrapped_first = torch.multinomial
    patch.apply_nano_patches()  # second call must not re-wrap
    assert torch.multinomial is wrapped_first
    assert handles.modeling.MossTTSNanoForGeneration is swapped_cls


def test_guard_installed_exactly_once_across_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)

    patch._install_multinomial_guard()
    original = patch._ORIGINAL_MULTINOMIAL
    assert original is not None
    patch._install_multinomial_guard()
    assert patch._ORIGINAL_MULTINOMIAL is original  # not re-captured (no double wrap)


# ---------------------------------------------------------------------------
# FP16 recast of not-vLLM-managed sub-models
# ---------------------------------------------------------------------------


def test_310p_nano_init_recasts_submodels_to_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)
    patch.apply_nano_patches()

    model = handles.modeling.MossTTSNanoForGeneration(vllm_config=object(), prefix="")

    for sub in (model._lm, model._audio_tokenizer):
        assert sub.linear.weight.dtype == torch.float16
        assert sub.inv_freq.dtype == torch.float16
    # The model itself keeps the base-class state (no attribute dropped).
    assert model.prefix == ""
    assert model.vllm_config is not None


# ---------------------------------------------------------------------------
# torch.multinomial guard semantics
# ---------------------------------------------------------------------------


def test_guard_predicate_rejects_non_matching_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)

    probs = torch.softmax(torch.randn(3, 11), dim=-1)
    predicate = patch._multinomial_310p_path
    # CPU device (no NPU in this env) -> never the 310P path.
    assert predicate(probs, 1, True, None) is False
    # Even forced to the "right" device semantics, wrong call shapes delegate:
    assert patch._multinomial_310p_path(probs, 2, True, None) is False
    assert patch._multinomial_310p_path(probs, 1, False, None) is False
    assert patch._multinomial_310p_path(probs, 1, True, torch.empty(0)) is False
    assert patch._multinomial_310p_path(probs.new_zeros(3, 11, dtype=torch.long), 1, True, None) is False


def test_guard_uses_random_sample_on_310p_path(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)
    patch.apply_nano_patches()

    # Force the 310P/NPU decision branch without a real NPU tensor.
    monkeypatch.setattr(patch, "_multinomial_310p_path", lambda *a: True)

    probs = torch.softmax(torch.randn(3, 11), dim=-1)
    out = torch.multinomial(probs, 1)

    assert out.shape == (3, 1)  # torch.multinomial contract: (..., num_samples)
    assert out.dtype == torch.long
    assert (out == 7).all()  # fake random_sample constant
    assert len(handles.sampling_calls) == 1
    assert handles.sampling_calls[0]["generators"] == {}  # device-global RNG
    assert handles.sampling_calls[0]["probs_shape"] == (3, 11)  # row-major flatten


def test_guard_delegates_non_matching_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)
    patch.apply_nano_patches()

    # Real predicate on CPU tensors -> delegation branch (no NPU here).
    probs = torch.softmax(torch.randn(3, 11), dim=-1)
    out = torch.multinomial(probs, 1)
    assert out.shape == (3, 1)
    assert handles.sampling_calls == []  # random_sample never touched


def test_guard_deterministic_sampling_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)
    patch.apply_nano_patches()

    # Force the 310P/NPU decision branch and flip the parity-smoke gate.
    monkeypatch.setattr(patch, "_multinomial_310p_path", lambda *a: True)
    monkeypatch.setattr(patch, "_DETERMINISTIC_SAMPLING", True)

    logits = torch.tensor([[0.1, 0.9, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    out = torch.multinomial(probs, 1)

    assert out.shape == (2, 1)  # torch.multinomial contract: (..., num_samples)
    assert out.dtype == torch.long
    # Greedy argmax (pure function of the probs): random_sample never touched.
    assert handles.sampling_calls == []
    assert torch.equal(out, probs.argmax(dim=-1, keepdim=True).long())


def test_guard_preserves_original_semantics_on_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_nano_fakes(monkeypatch)
    _isolated_multinomial(monkeypatch)
    patch = _load_nano_patch(monkeypatch, handles)
    patch.apply_nano_patches()

    original = patch._ORIGINAL_MULTINOMIAL
    probs = torch.softmax(torch.randn(4, 5), dim=-1)
    torch.manual_seed(0)
    delegated = torch.multinomial(probs, 1, replacement=True)
    torch.manual_seed(0)
    expected = original(probs, 1, replacement=True)
    assert torch.equal(delegated, expected)
