# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the CosyVoice3 310P patch wiring.

The tests load the patch module and the ``_310p.patch`` registry from source
with fake CosyVoice3 / vLLM-Ascend dependencies, so they validate the patch
contract (stage dispatch, sampler rebind, FP16 forcing, NZ/FRACTAL_Z packing,
F0-predictor exclusion, conditioning-dtype boundary) without loading a real
model or NPU kernels.
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


def _fake_ascend_random_sample(probs, generators):
    del generators
    return (probs.float() * 0 + 7.0).long()


def _fake_maybe_trans_nz(weight):
    # Sentinel packing: mark Linear weights so the test can prove the pass ran.
    return torch.full_like(weight, -12345.0)


def _fake_npu_format_cast(weight, _fmt):
    # Sentinel packing: mark group-1 conv weights with a distinct value.
    return torch.full_like(weight, -4321.0)


def _install_cosyvoice3_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    class FakeCosyVoice3Model(nn.Module):
        def __init__(self, *, vllm_config, prefix: str = ""):
            del prefix
            super().__init__()
            self.vllm_config = vllm_config
            self.model_stage = "cosyvoice3_talker"
            self.code2wav = None

        def load_weights(self, weights):
            del weights
            return None

    class FakeSpeechEmbedding(nn.Embedding):
        def __init__(self):
            super().__init__(8, 4, dtype=torch.float16)

    class FakeCode2Wav(nn.Module):
        def __init__(self, *, flow_model, hift):
            super().__init__()
            self.flow_model = flow_model
            self.hift = hift

    class FakeFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, dtype=torch.float32)
            self.conv = nn.Conv1d(4, 4, 3, dtype=torch.float32)
            self.grouped_conv = nn.Conv1d(4, 4, 3, groups=4, dtype=torch.float32)
            self.register_buffer("freqs_cis", torch.view_as_complex(torch.zeros(2, 2, 4, 2)), persistent=False)
            self.inference_calls: list[dict] = []

        def inference(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len,
                      embedding, streaming=True, finalize=False, n_timesteps=10):
            del token, token_len, prompt_token, prompt_token_len, prompt_feat_len, streaming, finalize, n_timesteps
            self.inference_calls.append(
                {
                    "prompt_feat_dtype": prompt_feat.dtype,
                    "embedding_dtype": embedding.dtype,
                }
            )
            return prompt_feat.float(), None

    hift_conv = nn.Conv1d(4, 8, 3, dtype=torch.float32)
    hift = nn.Module()
    hift.conv = hift_conv
    f0_predictor = nn.Module()
    f0_predictor.conv = nn.Conv1d(4, 1, 3, dtype=torch.float32)
    f0_predictor.to(torch.device("cpu"))
    hift.f0_predictor = f0_predictor

    code2wav = FakeCode2Wav(flow_model=FakeFlowModel(), hift=hift)

    class FakeMultiModalProcessor:
        load_calls: list[str] = []

        @classmethod
        def _load_s3_model(cls):
            cls.load_calls.append("loaded")
            model = nn.Linear(4, 4)
            return model, SimpleNamespace(), "npu"

    _install_fake_module(monkeypatch, "vllm", __version__="0.32.0")
    _install_fake_module(monkeypatch, "vllm.logger", init_logger=lambda _name: SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None))

    _install_fake_module(monkeypatch, "vllm_ascend", __version__="0.32.0")
    _install_fake_module(monkeypatch, "vllm_ascend.sample")
    _install_fake_module(monkeypatch, "vllm_ascend.sample.sampler", random_sample=_fake_ascend_random_sample)
    _install_fake_module(monkeypatch, "vllm_ascend.utils", maybe_trans_nz=_fake_maybe_trans_nz)
    _install_fake_module(monkeypatch, "torch_npu", npu_format_cast=_fake_npu_format_cast)

    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models.cosyvoice3")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.cosyvoice3.cosyvoice3",
        CosyVoice3Model=FakeCosyVoice3Model,
        CosyVoice3MultiModalProcessor=FakeMultiModalProcessor,
        random_sample=lambda _probs, _gens: torch.zeros(1, dtype=torch.long),
    )
    return SimpleNamespace(
        FakeCosyVoice3Model=FakeCosyVoice3Model,
        FakeFlowModel=FakeFlowModel,
        FakeCode2Wav=FakeCode2Wav,
        FakeMultiModalProcessor=FakeMultiModalProcessor,
        code2wav=code2wav,
        flow_model=code2wav.flow_model,
        hift=hift,
        f0_predictor=f0_predictor,
    )


def _load_cosyvoice3_patch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fakes = _install_cosyvoice3_patch_fakes(monkeypatch)
    root = _repo_root()
    patch_path = root / "vllm_omni/platforms/npu/_310p/patch/cosyvoice3.py"
    module = _load_source_module("vllm_omni_test_310p_cosyvoice3_patch", patch_path)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.cosyvoice3", module)
    return module, fakes


def _load_registry(monkeypatch: pytest.MonkeyPatch):
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.worker")
    root = _repo_root()
    registry_path = root / "vllm_omni/platforms/npu/_310p/patch/__init__.py"
    return _load_source_module("vllm_omni_test_310p_cosyvoice3_registry", registry_path)


def test_registry_dispatches_cosyvoice3_by_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_mod, fakes = _load_cosyvoice3_patch(monkeypatch)
    registry = _load_registry(monkeypatch)
    cv3_module = sys.modules["vllm_omni.model_executor.models.cosyvoice3.cosyvoice3"]
    base_cls = fakes.FakeCosyVoice3Model

    monkeypatch.setattr(patch_mod, "_TALKER_PATCHED", False, raising=False)
    monkeypatch.setattr(patch_mod, "_CODE2WAV_PATCHED", False, raising=False)

    # Wrong arch: untouched.
    registry.apply_model_patches(SimpleNamespace(model_arch="SomethingElse", model_stage="cosyvoice3_talker"))
    assert cv3_module.CosyVoice3Model is base_cls

    # Talker stage installs the talker subclass.
    registry.apply_model_patches(
        SimpleNamespace(model_arch="CosyVoice3Model", model_stage="cosyvoice3_talker")
    )
    assert cv3_module.CosyVoice3Model is not base_cls
    assert cv3_module.random_sample is _fake_ascend_random_sample

    # Second dispatch with the code2wav stage installs the code2wav subclass
    # (separate worker process in production; here both land on the same fake).
    monkeypatch.setattr(patch_mod, "_TALKER_PATCHED", False, raising=False)
    monkeypatch.setattr(patch_mod, "_CODE2WAV_PATCHED", False, raising=False)
    cv3_module.CosyVoice3Model = base_cls
    registry.apply_model_patches(
        SimpleNamespace(model_arch="CosyVoice3Model", model_stage="cosyvoice3_code2wav")
    )
    assert cv3_module.CosyVoice3Model is not base_cls
    assert issubclass(cv3_module.CosyVoice3Model, base_cls)

    # Unknown stage: no patch.
    monkeypatch.setattr(patch_mod, "_TALKER_PATCHED", False, raising=False)
    monkeypatch.setattr(patch_mod, "_CODE2WAV_PATCHED", False, raising=False)
    cv3_module.CosyVoice3Model = base_cls
    registry.apply_model_patches(SimpleNamespace(model_arch="CosyVoice3Model", model_stage="unknown"))
    assert cv3_module.CosyVoice3Model is base_cls


def test_talker_patch_verifies_fp16_and_pins_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_mod, fakes = _load_cosyvoice3_patch(monkeypatch)
    cv3_module = sys.modules["vllm_omni.model_executor.models.cosyvoice3.cosyvoice3"]

    patch_mod.apply_talker_310p_patches()
    patched_cls = cv3_module.CosyVoice3Model
    assert patched_cls is not fakes.FakeCosyVoice3Model
    assert issubclass(patched_cls, fakes.FakeCosyVoice3Model)
    assert cv3_module.random_sample is _fake_ascend_random_sample

    # All-FP16 instance: no stray cast, clean construction.
    model = patched_cls(vllm_config=SimpleNamespace())
    for param in model.parameters():
        assert param.dtype == torch.float16

    # Instance with a stray FP32 parameter: defensive cast engages.
    stray = patched_cls(vllm_config=SimpleNamespace())
    layer = nn.Linear(4, 4, dtype=torch.float32)
    stray.add_module("stray", layer)
    assert layer.weight.dtype == torch.float32
    assert patch_mod._verify_talker_fp16(stray) == 2
    assert stray.stray.weight.dtype == torch.float16

    # S3 tokenizer pinned to CPU: the wrapped classmethod returns "cpu".
    processor_cls = cv3_module.CosyVoice3MultiModalProcessor
    model_s3, _s3, device = processor_cls._load_s3_model()
    assert device == "cpu"

    # Idempotent: a second call does not re-wrap.
    patch_mod.apply_talker_310p_patches()
    assert cv3_module.CosyVoice3Model is patched_cls


def test_code2wav_patch_forces_fp16_packs_and_casts_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_mod, fakes = _load_cosyvoice3_patch(monkeypatch)
    cv3_module = sys.modules["vllm_omni.model_executor.models.cosyvoice3.cosyvoice3"]

    patch_mod.apply_code2wav_310p_patches()
    patched_cls = cv3_module.CosyVoice3Model
    assert issubclass(patched_cls, fakes.FakeCosyVoice3Model)

    model = patched_cls(vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16)))
    model.model_stage = "cosyvoice3_code2wav"
    model.code2wav = fakes.code2wav

    flow_model = fakes.flow_model
    hift = fakes.hift
    f0 = fakes.f0_predictor

    model.load_weights(None)

    # Flow model: FP16, linear packed (sentinel), group-1 conv packed,
    # grouped conv skipped, complex buffer intact.
    assert flow_model.linear.weight.dtype == torch.float16
    assert torch.all(flow_model.linear.weight == -12345.0)
    assert torch.all(flow_model.conv.weight == -4321.0)
    assert flow_model.grouped_conv.weight.dtype == torch.float16
    assert not torch.all(flow_model.grouped_conv.weight == -4321.0)
    assert flow_model.freqs_cis.is_complex()

    # HiFT body: FP16 + packed.  F0 predictor: left on CPU in FP32, unpacked.
    assert hift.conv.weight.dtype == torch.float16
    assert torch.all(hift.conv.weight == -4321.0)
    assert next(f0.parameters()).device.type == "cpu"
    assert f0.conv.weight.dtype == torch.float32
    assert not torch.all(f0.conv.weight == -4321.0)

    # Conditioning boundary: FP32 embedding / prompt_feat arrive at the
    # original inference as FP16.
    embedding = torch.randn(1, 192, dtype=torch.float32)
    prompt_feat = torch.randn(1, 80, 16, dtype=torch.float32)
    flow_model.inference(
        torch.zeros(1, 8, dtype=torch.long),
        torch.tensor([8]),
        torch.zeros(1, 4, dtype=torch.long),
        torch.tensor([4]),
        prompt_feat,
        torch.tensor([16]),
        embedding,
    )
    assert flow_model.inference_calls[0]["embedding_dtype"] == torch.float16
    assert flow_model.inference_calls[0]["prompt_feat_dtype"] == torch.float16
