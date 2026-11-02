# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the 310P VoxCPM2 patch.

Same approach as ``test_310p_patches.py``: the patch module is loaded from
source with fake dependencies, so the wiring (registry dispatch, NPUGraph
neutralization, FP16 casting, weight packing, CPU resample) is validated
without an NPU, model checkpoint or the native voxcpm package.

Real-device behavior (NPUGraph-off LocDiT eager numerics, FRACTAL_Z conv
performance, first-chunk latency) is qualified in the 310P real-device phase
of task-13.
"""

from __future__ import annotations

import importlib.util
import logging
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


class _RecordingResample(nn.Module):
    """torchaudio.transforms.Resample stand-in recording input device."""

    calls: list[torch.device] = []

    def __init__(self, orig_sr: int, target_sr: int) -> None:
        super().__init__()
        self.orig_sr = orig_sr
        self.target_sr = target_sr

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        _RecordingResample.calls.append(waveform.device)
        return waveform


def _make_fake_tts() -> nn.Module:
    tts = nn.Module()
    tts.audio_vae = nn.Sequential(
        nn.Conv1d(4, 4, 3),  # group-1 conv: FRACTAL_Z packing candidate
        nn.ConvTranspose1d(4, 4, 3),
        nn.Conv1d(4, 4, 3, groups=2),  # grouped: must stay ND
    )
    # Real tts module registers feat_decoder as a submodule; the fake must
    # too, so .modules() traversal (the cast/pack helpers) reaches it.
    tts.feat_decoder = nn.Module()
    tts.feat_decoder.estimator = nn.Sequential(nn.Linear(8, 8))
    tts.fusion_concat_proj = nn.Linear(8, 8)
    tts._encode_sample_rate = 16000
    tts.patch_size = 2
    tts.chunk_size = 2
    tts.audio_vae.latent_dim = 4
    # Mixed buffer dtypes that a careless module.to(dtype) would corrupt
    # (register directly: newer torch rejects register_buffer over an
    # existing plain attribute).
    tts.register_buffer("freqs_cis", torch.randn(8, 8, dtype=torch.complex64), persistent=False)
    tts.register_buffer("token_ids", torch.tensor([1, 2, 3]), persistent=False)

    def _encode(audio, sample_rate, *args, **kwargs):
        del sample_rate, args, kwargs
        frames = audio.numel() // 4
        return torch.randn(1, 4, frames)

    tts.audio_vae.encode = _encode
    return tts


class FakeVoxCPM2TalkerBase(nn.Module):
    """Stands in for VoxCPM2TalkerForConditionalGeneration.__init__ side effects.

    Mirrors the v0.32.0 talker: ``_tts`` keeps the *native checkpoint* dtype
    (no engine-side cast), ``_side_dtype`` derives from the fusion projection.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        del prefix
        super().__init__()
        self.vllm_config = vllm_config
        self._tts = _make_fake_tts()
        self._side_dtype = torch.bfloat16  # native checkpoint dtype
        self._setup_calls = 0
        self._vae_decode_dtypes: list[torch.dtype] = []

    @property
    def tts(self) -> nn.Module:
        # Mirrors the v0.32.0 talker (``tts`` is a property over ``_tts``).
        return self._tts

    def _setup_execution_optimizations(self) -> None:
        self._setup_calls += 1

    def _run_vae_decode(self, feat: torch.Tensor) -> torch.Tensor:
        # Records the dtype the latent arrives in; the 310P subclass must
        # re-cast FP32 latents to the VAE parameter dtype before this sees
        # them (FP16 convs reject mixed input/bias dtypes on 310P).
        self._vae_decode_dtypes.append(feat.dtype)
        return feat

    def load_weights(self, weights):
        del weights
        return {"loaded"}


def _install_voxcpm2_patch_fakes(monkeypatch: pytest.MonkeyPatch):
    fake_vox_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker",
        VoxCPM2TalkerForConditionalGeneration=FakeVoxCPM2TalkerBase,
        _encode_raw_audio=lambda *args, **kwargs: torch.empty(0),
    )
    fake_adapter = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu.models.voxcpm2_talker",
        setup_voxcpm2_loc_dit_npu_graph=lambda model: setattr(
            model.tts.feat_decoder.estimator, "_voxcpm2_npu_graph_runner", object()
        ),
    )
    maybe_trans_nz_calls: list[torch.Tensor] = []
    format_cast_calls: list[tuple[torch.Tensor, int]] = []

    def _maybe_trans_nz(weight: torch.Tensor) -> torch.Tensor:
        maybe_trans_nz_calls.append(weight)
        return weight

    def _npu_format_cast(weight: torch.Tensor, fmt: int) -> torch.Tensor:
        format_cast_calls.append((weight, fmt))
        return weight

    _install_fake_module(monkeypatch, "vllm")

    def _init_logger(name: str) -> logging.Logger:
        logger_ = logging.getLogger(f"voxcpm2_310p_test.{name}")
        logger_.setLevel(logging.INFO)
        return logger_

    _install_fake_module(monkeypatch, "vllm.logger", init_logger=_init_logger)
    _install_fake_module(monkeypatch, "torch_npu", npu_format_cast=_npu_format_cast)
    _install_fake_module(monkeypatch, "vllm_ascend")
    _install_fake_module(monkeypatch, "vllm_ascend.utils", maybe_trans_nz=_maybe_trans_nz)
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.voxcpm2",
        voxcpm2_talker=fake_vox_module,
    )
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu.models",
        voxcpm2_talker=fake_adapter,
    )
    fake_transforms = _install_fake_module(
        monkeypatch,
        "torchaudio.transforms",
        Resample=_RecordingResample,
    )
    _install_fake_module(monkeypatch, "torchaudio", transforms=fake_transforms)
    return fake_vox_module, fake_adapter, maybe_trans_nz_calls, format_cast_calls


def _load_voxcpm2_patch(monkeypatch: pytest.MonkeyPatch):
    (
        fake_vox_module,
        fake_adapter,
        maybe_trans_nz_calls,
        format_cast_calls,
    ) = _install_voxcpm2_patch_fakes(monkeypatch)
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "voxcpm2.py"
    module = _load_source_module("vllm_omni_test_310p_voxcpm2_patch", path)
    return (
        module,
        SimpleNamespace(
            vox=fake_vox_module,
            adapter=fake_adapter,
            nz_calls=maybe_trans_nz_calls,
            fmt_calls=format_cast_calls,
        ),
    )


def _fake_vllm_config(dtype: torch.dtype = torch.float16):
    return SimpleNamespace(model_config=SimpleNamespace(dtype=dtype))


def test_registry_dispatches_voxcpm2_talker_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = (
        _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    )
    registry = _load_source_module("vllm_omni_test_310p_voxcpm2_registry", registry_path)
    calls = {"worker": 0, "voxcpm2": 0, "qwen3_tts": 0}

    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.worker",
        apply_patch=lambda: calls.__setitem__("worker", calls["worker"] + 1),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=lambda: calls.__setitem__("qwen3_tts", calls["qwen3_tts"] + 1),
        apply_code2wav_patches=lambda: None,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.voxcpm2",
        apply_voxcpm2_310p_patches=lambda: calls.__setitem__("voxcpm2", calls["voxcpm2"] + 1),
    )

    registry.apply_patches()
    registry.apply_model_patches(SimpleNamespace(model_arch="OtherModel"))
    registry.apply_model_patches(SimpleNamespace(model_arch="VoxCPM2TalkerForConditionalGeneration"))

    # Idempotence of the patch itself is covered by
    # test_patch_neutralizes_npugraph_and_swaps_talker (double apply call);
    # the registry dispatches on every matching load_model.
    assert calls == {"worker": 1, "voxcpm2": 1, "qwen3_tts": 0}


def test_patch_neutralizes_npugraph_and_swaps_talker(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fakes = _load_voxcpm2_patch(monkeypatch)
    original_setup = fakes.adapter.setup_voxcpm2_loc_dit_npu_graph
    original_base = fakes.vox.VoxCPM2TalkerForConditionalGeneration
    original_encode = fakes.vox._encode_raw_audio

    module.apply_voxcpm2_310p_patches()
    module.apply_voxcpm2_310p_patches()  # idempotent

    # LocDiT NPUGraph wrapper must be the logged no-op, not the 910 setup.
    assert fakes.adapter.setup_voxcpm2_loc_dit_npu_graph is not original_setup
    estimator = _make_fake_tts().feat_decoder.estimator
    fakes.adapter.setup_voxcpm2_loc_dit_npu_graph(SimpleNamespace(tts=estimator))
    assert not hasattr(estimator, "_voxcpm2_npu_graph_runner")

    # The registry-resolved talker class is the 310P subclass, not the base.
    swapped = fakes.vox.VoxCPM2TalkerForConditionalGeneration
    assert swapped is not original_base
    assert issubclass(swapped, original_base)

    # The device-sensitive resample path is replaced at module scope.
    assert fakes.vox._encode_raw_audio is module._encode_raw_audio_310p
    assert original_encode is not module._encode_raw_audio_310p


def test_talker310p_forces_fp16_and_keeps_complex_buffers(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    module, fakes = _load_voxcpm2_patch(monkeypatch)
    module.apply_voxcpm2_310p_patches()

    swapped = fakes.vox.VoxCPM2TalkerForConditionalGeneration
    model = swapped(vllm_config=_fake_vllm_config(torch.float16))

    # tts side-path parameters are FP16 now.
    assert all(p.dtype is torch.float16 for p in model._tts.feat_decoder.estimator.parameters())
    assert model._tts.fusion_concat_proj.weight.dtype is torch.float16
    assert next(model._tts.audio_vae.parameters()).dtype is torch.float16
    assert model._side_dtype is torch.float16

    # Complex/int buffers survived the cast (RoPE cache intact).
    assert model._tts.freqs_cis.dtype is torch.complex64
    assert model._tts.token_ids.dtype is torch.int64

    # Linear weights were NZ-packed, group-1 convs FRACTAL_Z-packed,
    # grouped convs left alone.
    assert len(fakes.nz_calls) >= 2
    formats = {fmt for _, fmt in fakes.fmt_calls}
    assert formats == {module._ACL_FORMAT_FRACTAL_Z}
    assert caplog.text.count("VoxCPM2 310P: packed tts weights") == 1


def test_talker310p_warns_on_non_fp16_deployment(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    module, fakes = _load_voxcpm2_patch(monkeypatch)
    module.apply_voxcpm2_310p_patches()
    swapped = fakes.vox.VoxCPM2TalkerForConditionalGeneration
    with caplog.at_level(logging.WARNING):
        swapped(vllm_config=_fake_vllm_config(torch.bfloat16))
    assert "voxcpm2_310p.yaml" in caplog.text


def test_cast_helper_skips_complex_and_int(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fakes = _load_voxcpm2_patch(monkeypatch)

    holder = nn.Module()
    holder.linear = nn.Linear(4, 4)
    holder.register_buffer("complex_buf", torch.randn(4, dtype=torch.complex64))
    holder.register_buffer("int_buf", torch.arange(4))
    holder.register_buffer("float_buf", torch.ones(4, dtype=torch.float32))

    converted = module._cast_module_to_dtype(holder, torch.float16)

    assert holder.linear.weight.dtype is torch.float16
    assert holder.float_buf.dtype is torch.float16
    assert holder.complex_buf.dtype is torch.complex64
    assert holder.int_buf.dtype is torch.int64
    assert converted == 3  # linear weight+bias, float_buf


def test_pack_weights_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fakes = _load_voxcpm2_patch(monkeypatch)
    fakes.nz_calls.clear()
    fakes.fmt_calls.clear()

    holder = nn.Module()
    holder.lin = nn.Linear(8, 8)
    holder.conv = nn.Conv1d(4, 4, 3)
    holder.tconv = nn.ConvTranspose1d(4, 4, 3)
    holder.grouped = nn.Conv1d(4, 4, 3, groups=2)

    linear_count, conv_count = module._pack_npu_weights(holder)

    assert (linear_count, conv_count) == (1, 2)
    assert any(
        w.data_ptr() == holder.lin.weight.data_ptr() for w in fakes.nz_calls
    )
    packed_weights = {w.data_ptr() for w, _ in fakes.fmt_calls}
    assert packed_weights == {holder.conv.weight.data_ptr(), holder.tconv.weight.data_ptr()}
    assert all(fmt == module._ACL_FORMAT_FRACTAL_Z for _, fmt in fakes.fmt_calls)


def test_fused_qkv_buffer_is_packed_after_lazy_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fakes = _load_voxcpm2_patch(monkeypatch)
    module.apply_voxcpm2_310p_patches()
    swapped = fakes.vox.VoxCPM2TalkerForConditionalGeneration
    fakes.nz_calls.clear()
    model = swapped(vllm_config=_fake_vllm_config())

    # The LocDiT fusion rewrite registers its fused buffer lazily.
    fused = torch.randn(24, 8)
    model._tts.feat_decoder.estimator[0].register_buffer(
        "_voxcpm2_fused_qkv_weight", fused, persistent=False
    )

    model._setup_execution_optimizations()

    assert any(w.data_ptr() == fused.data_ptr() for w in fakes.nz_calls)
    assert model._setup_calls == 1


def test_encode_raw_audio_resamples_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fakes = _load_voxcpm2_patch(monkeypatch)
    _RecordingResample.calls.clear()

    tts = _make_fake_tts()
    samples = [0.0] * 3200  # 0.2s @ 16k
    feat = module._encode_raw_audio_310p(tts, samples, 16000, keep_on_device=True)

    # Resample path not exercised (sr == encode_sr): no resample call.
    assert _RecordingResample.calls == []
    assert feat.shape == (400, 2, 4)  # (frames, patch_size, latent_dim)

    _RecordingResample.calls.clear()
    tts2 = _make_fake_tts()
    tts2._encode_sample_rate = 8000
    feat = module._encode_raw_audio_310p(tts2, [0.0] * 1600, 16000, keep_on_device=True)
    assert _RecordingResample.calls == [torch.device("cpu")]
    assert feat.dim() == 3


def test_talker310p_casts_vae_decode_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP32 AR latents must reach the FP16 AudioVAE as FP16.

    Real-device first run (126, C2) died in ``F.conv1d`` with
    ``Input type (float) and bias type (c10::Half) should be the same``:
    the VAE decode convs were FP16 (cast by this patch) while the talker's
    AR-side latent projection stays FP32.  The 310P talker must cast at the
    ``_run_vae_decode`` boundary.
    """
    module, fakes = _load_voxcpm2_patch(monkeypatch)
    module.apply_voxcpm2_310p_patches()
    swapped = fakes.vox.VoxCPM2TalkerForConditionalGeneration
    model = swapped(vllm_config=_fake_vllm_config(torch.float16))

    # The VAE runs FP16 after the patch, so the decode boundary must see
    # FP16 regardless of the incoming latent dtype.
    assert next(model._tts.audio_vae.parameters()).dtype is torch.float16

    model._run_vae_decode(torch.randn(1, 4, 8))  # FP32 latent (AR projection)
    model._run_vae_decode(torch.randn(1, 4, 8, dtype=torch.float16))  # already FP16

    assert model._vae_decode_dtypes == [torch.float16, torch.float16]
