# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the 310P MOSS-TTS patches (skeleton).

Loads ``_310p/patch/moss_tts.py`` and the patch registry from source with
fake model/vllm-ascend dependencies, mirroring the Qwen3-TTS 310P contract
tests: the patch *wiring* and *semantics* are validated without an NPU.

Real-device assertions (kernel numerics, FRACTAL_NZ mask capture, ring-state
invariants) live in the hardware-gated test added in the real-device phase.
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


def _install_moss_fakes(monkeypatch: pytest.MonkeyPatch, sampling_calls: list) -> tuple:
    """Install fake MOSS model modules + vllm-ascend 310P deps; return handles."""
    hidden = 6
    n_vq = 4
    vocab = 9  # audio_vocab + 1 sentinel

    # -- vllm-ascend 310P dependencies -------------------------------------

    def random_sample(probs, generators):
        sampling_calls.append(generators)
        return torch.full((probs.shape[0],), 7, dtype=torch.long)

    def _apply_top_k_top_p_pytorch(logits, k, p):
        return logits

    def aligned_16(tensor: torch.Tensor) -> torch.Tensor:
        pad = (-tensor.shape[0]) % 16
        return torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad)) if pad else tensor

    format_cast_calls: list = []

    def npu_format_cast(tensor, acl_format):
        format_cast_calls.append((acl_format, tuple(tensor.shape)))
        return tensor

    flash_calls: list = []

    def _npu_flash_attention(query, key, value, mask, seq_len, scale_value, num_heads, num_kv_heads, out):
        flash_calls.append(
            {
                "query": tuple(query.shape),
                "dtype": query.dtype,
                "mask": None if mask is None else tuple(mask.shape),
                "seq_len": tuple(seq_len.shape),
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
            }
        )
        out.copy_(query + key + value)

    _install_fake_module(monkeypatch, "vllm_ascend", sample=None, utils=None)
    _install_fake_module(monkeypatch, "vllm_ascend.sample", sampler=None)
    _install_fake_module(
        monkeypatch,
        "vllm_ascend.sample.sampler",
        random_sample=random_sample,
        apply_top_k_top_p=_apply_top_k_top_p_pytorch,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_ascend.utils",
        ACL_FORMAT_FRACTAL_NZ=29,
        aligned_16=aligned_16,
        nd_to_nz_2d=lambda t: t,
    )
    _install_fake_module(
        monkeypatch,
        "torch_npu",
        npu_format_cast=npu_format_cast,
        _npu_flash_attention=_npu_flash_attention,
        npu_apply_rotary_pos_emb=None,  # unused by the portable RoPE path
    )

    # -- MOSS talker fakes ---------------------------------------------------

    class FakeDelayTalker(nn.Module):
        have_multimodal_outputs = True

        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__()
            self.n_vq = n_vq
            self.audio_pad_code = 0
            self.audio_vocab_size = vocab - 1
            self.multinomial_calls = 0
            self.audio_heads = nn.ModuleList([nn.Linear(hidden, vocab, bias=False) for _ in range(n_vq)])
            self.audio_embeddings = nn.ModuleList([nn.Embedding(vocab, hidden) for _ in range(n_vq)])
            with torch.no_grad():
                self._stacked_audio_head_w = torch.stack(
                    [h.weight for h in self.audio_heads]
                ).to(torch.bfloat16)
                self._stacked_audio_emb_w = torch.stack(
                    [e.weight for e in self.audio_embeddings]
                ).to(torch.bfloat16)

        def load_weights(self, weights):
            return {"loaded"}

    class FakeRealtimeTalker(FakeDelayTalker):
        pass

    class FakeLocalTalker(FakeDelayTalker):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.talker_mtp_graph_safe = True  # NPU override must force False

    talker_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker",
        MossTTSDelayTalkerForGeneration=FakeDelayTalker,
        MossTTSRealtimeTalkerForGeneration=FakeRealtimeTalker,
        MossTTSLocalTalkerForGeneration=FakeLocalTalker,
    )

    # -- MOSS codec fakes ------------------------------------------------------

    lut_builds: list = []

    class FakeCodecV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(hidden, hidden)
            self.quantizer = SimpleNamespace(
                num_quantizers=n_vq,
                _decode_lut=None,
                build_decode_lut=lambda num_quantizers, dtype: lut_builds.append((num_quantizers, dtype)),
            )

    class FakeCodecDecoder(nn.Module):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__()
            self.vllm_config = vllm_config
            self._codec = None

        def load_weights(self, weights):
            self._codec = FakeCodecV2()
            return {"codec"}

    class FakeCodecV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(hidden, hidden)
            self.quantizer = SimpleNamespace(num_quantizers=n_vq, _decode_lut=None)

    codec_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec",
        MossTTSCodecDecoder=FakeCodecDecoder,
    )

    original_get_torch_npu = lambda: None  # noqa: E731
    v2_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2",
        MossAudioTokenizerModel=FakeCodecV2,
        _get_torch_npu=original_get_torch_npu,
    )

    # -- package parents ---------------------------------------------------------

    _install_fake_module(monkeypatch, "vllm_omni", model_executor=None)
    _install_fake_module(monkeypatch, "vllm_omni.model_executor", models=None)
    package = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models",
        moss_tts=None,
    )
    moss_tts_pkg = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.moss_tts",
        modeling_moss_tts_talker=talker_module,
        modeling_moss_tts_codec=codec_module,
        audio_tokenizer_v2=v2_module,
    )
    setattr(package, "moss_tts", moss_tts_pkg)

    return SimpleNamespace(
        talker_module=talker_module,
        codec_module=codec_module,
        v2_module=v2_module,
        codec_v1_cls=FakeCodecV1,
        sampling_calls=sampling_calls,
        format_cast_calls=format_cast_calls,
        flash_calls=flash_calls,
        lut_builds=lut_builds,
    )


def _load_moss_patch(monkeypatch: pytest.MonkeyPatch, handles):
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "moss_tts.py"
    return _load_source_module("moss_tts_310p_patch_under_test", path)


def _load_patch_registry(monkeypatch: pytest.MonkeyPatch, handles):
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    return _load_source_module("moss_310p_patch_registry_under_test", path)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_moss_registry_routes_arches_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])

    calls = {"talker": 0, "codec": 0}

    def apply_talker_patches():
        calls["talker"] += 1

    def apply_codec_patches():
        calls["codec"] += 1

    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.worker", apply_patch=lambda: None)
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=lambda: None,
        apply_code2wav_patches=lambda: None,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.moss_tts",
        apply_talker_patches=apply_talker_patches,
        apply_codec_patches=apply_codec_patches,
    )

    registry = _load_patch_registry(monkeypatch, handles)
    cfg = lambda arch: SimpleNamespace(model_arch=arch)  # noqa: E731

    for arch in ("MossTTSDelayModel", "MossTTSRealtime", "MossTTSLocalModel"):
        registry.apply_model_patches(cfg(arch))
    assert calls["talker"] == 3

    for _ in range(2):
        registry.apply_model_patches(cfg("MossTTSCodecDecoder"))
    assert calls["codec"] == 2

    # Unknown arches are a no-op.
    registry.apply_model_patches(cfg("SomeOtherArch"))
    assert calls == {"talker": 3, "codec": 2}


def test_apply_talker_patches_swaps_module_classes_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    original_delay = handles.talker_module.MossTTSDelayTalkerForGeneration
    patch.apply_talker_patches()
    patch.apply_talker_patches()  # idempotent
    assert handles.talker_module.MossTTSDelayTalkerForGeneration is not original_delay
    assert issubclass(handles.talker_module.MossTTSDelayTalkerForGeneration, original_delay)
    assert handles.talker_module.MossTTSLocalTalkerForGeneration is not original_delay


def test_apply_codec_patches_swaps_decoder_and_installs_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    original_decoder = handles.codec_module.MossTTSCodecDecoder
    original_get_torch_npu = handles.v2_module._get_torch_npu

    patch.apply_codec_patches()
    patch.apply_codec_patches()  # idempotent
    assert handles.codec_module.MossTTSCodecDecoder is not original_decoder
    assert issubclass(handles.codec_module.MossTTSCodecDecoder, original_decoder)

    shim = handles.v2_module._get_torch_npu()
    assert shim is not None and shim is not handles.v2_module._get_torch_npu
    assert hasattr(shim, "npu_rotary_mul") and hasattr(shim, "npu_fusion_attention")


# ---------------------------------------------------------------------------
# Talker semantics
# ---------------------------------------------------------------------------


def test_delay_talker_sampling_uses_random_sample_not_multinomial(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    talker = patch._MossTTSDelayTalker310P(vllm_config=None)
    last_h = torch.randn(1, 6, dtype=torch.float32)

    # State: 2 heads pre-delay-active (idx < audio_lengths), no post-delay
    # restriction (delayed_lengths == -1).
    state = {"audio_lengths": 2, "delayed_lengths": -1, "is_audio": True, "step": 3}
    codes = talker._sample_audio_codes(last_h, state)

    assert handles.sampling_calls and all(g == {} for g in handles.sampling_calls), (
        "audio sampling must run through random_sample with the global generator"
    )
    assert codes.shape == (4,) and codes.dtype == torch.long
    # Heads 0/1 sampled (fake random_sample -> 7), heads 2/3 stay at pad code.
    assert codes[0].item() == 7 and codes[1].item() == 7
    assert codes[2].item() == talker.audio_pad_code and codes[3].item() == talker.audio_pad_code


def test_delay_talker_sampling_respects_delayed_post_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    talker = patch._MossTTSDelayTalker310P(vllm_config=None)
    last_h = torch.randn(1, 6, dtype=torch.float32)

    # delayed_lengths=1 -> post_mask = idx > 0: head 0 still lags (pad),
    # heads 1..3 sampled; audio_lengths=4 keeps every head pre-delay-active.
    state = {"audio_lengths": 4, "delayed_lengths": 1, "is_audio": True, "step": 9}
    codes = talker._sample_audio_codes(last_h, state)
    assert codes[0].item() == talker.audio_pad_code
    assert codes[1].item() == 7 and codes[2].item() == 7 and codes[3].item() == 7


def test_delay_talker_load_weights_casts_caches_to_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    talker = patch._MossTTSDelayTalker310P(vllm_config=None)
    assert talker._stacked_audio_head_w.dtype == torch.bfloat16
    loaded = talker.load_weights(iter(()))
    assert loaded == {"loaded"}
    assert talker._stacked_audio_head_w.dtype == torch.float16
    assert talker._stacked_audio_emb_w.dtype == torch.float16


def test_local_talker_graph_safety_flag_forced_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    talker = patch._MossTTSLocalTalker310P(vllm_config=None)
    assert talker.talker_mtp_graph_safe is False


# ---------------------------------------------------------------------------
# Codec semantics
# ---------------------------------------------------------------------------


def test_codec_310p_disables_streaming_npugraph(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    codec = patch._MossTTSCodecDecoder310P(vllm_config=SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False)))
    # Even with graph capture enabled, the 310P override must return no batch
    # sizes -> the streaming session never builds the NPUGraph wrapper.
    assert codec._streaming_graph_batch_sizes_from_compilation_config() == []


def test_codec_310p_load_weights_casts_v2_decoder_to_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    codec = patch._MossTTSCodecDecoder310P(vllm_config=None)
    loaded = codec.load_weights(iter(()))
    assert loaded == {"codec"}
    decoder_params = list(codec._codec.decoder.parameters())
    assert decoder_params and all(p.dtype == torch.float16 for p in decoder_params)
    assert handles.lut_builds == [(4, torch.float16)], "decode LUT must be rebuilt in FP16"


def test_codec_310p_leaves_v1_decoder_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    codec = patch._MossTTSCodecDecoder310P(vllm_config=None)
    codec._codec = handles.codec_v1_cls()  # v1 stand-in: FP32 decoder, no LUT hook
    patch._cast_codec_decoder_to_310p_dtype(codec._codec)
    decoder_params = list(codec._codec.decoder.parameters())
    assert decoder_params and all(p.dtype == torch.float32 for p in decoder_params)
    assert handles.lut_builds == [], "v1 codec must not build an FP16 LUT"


# ---------------------------------------------------------------------------
# torch_npu shim semantics
# ---------------------------------------------------------------------------


def test_shim_rope_matches_portable_neox_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    shim = patch._TorchNpu310PShim()
    x = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)  # (B, H, T, D)
    cos = torch.randn(2, 1, 4, 8)
    sin = torch.randn(2, 1, 4, 8)

    out = shim.npu_rotary_mul(x, cos, sin)

    x32 = x.float()
    x1, x2 = x32.chunk(2, dim=-1)
    expected = (x32 * cos + torch.cat((-x2, x1), dim=-1) * sin).to(x.dtype)
    assert out.shape == x.shape and out.dtype == x.dtype
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


def test_shim_attention_converts_mask_and_calls_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    handles = _install_moss_fakes(monkeypatch, sampling_calls=[])
    patch = _load_moss_patch(monkeypatch, handles)

    shim = patch._TorchNpu310PShim()
    bsz, heads, seq, dim = 1, 2, 3, 4
    q = torch.randn(bsz, heads, seq, dim, dtype=torch.float16)
    k = torch.randn(bsz, heads, seq, dim, dtype=torch.float16)
    v = torch.randn(bsz, heads, seq, dim, dtype=torch.float16)
    mask = torch.zeros(bsz, 1, seq, seq, dtype=torch.bool)
    mask[:, :, 0, 1:] = True  # row 0: masked columns

    (attn_out,) = shim.npu_fusion_attention(
        q, k, v, head_num=heads, input_layout="BNSD", atten_mask=mask, scale=0.5
    )

    assert attn_out.shape == q.shape and attn_out.dtype == q.dtype
    assert len(handles.flash_calls) == 1
    call = handles.flash_calls[0]
    # Token-major: B*T=3 tokens aligned up to 16, one row extended in seq_len.
    assert call["query"] == (16, heads, dim)
    assert call["dtype"] == torch.float16
    # (B, 1, q_tokens_padded, kv_tokens_padded)
    assert call["mask"] == (1, 1, 16, 16)
    assert call["seq_len"] == (1,)
    assert handles.format_cast_calls, "mask must be format-cast to FRACTAL_NZ"
    assert handles.format_cast_calls[-1][0] == 29
