# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Qwen2.5-Omni 310P patch wiring.

The tests load the patch module and the shared registry from source with
fake Qwen2.5-Omni / vllm-ascend dependencies, so they validate the patch
contract without loading real model or NPU kernels.  The fake
``_npu_flash_attention`` implements dense softmax attention from the additive
mask, so the 310P flash path is checked numerically against the portable
block-windowed SDPA reference.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEFAULT_SPEAKER = "Chelsie"


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


# ===================================================================
#  Fake Qwen2.5-Omni model surface (portable reference implementations)
# ===================================================================


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Verbatim copy of the upstream DiT RoPE (interleaved-pair convention)."""

    def rotate_half_codec(x):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


def _block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    look_backward_block: int = 0,
    look_ahead_block: int = 0,
) -> torch.Tensor:
    """Verbatim copy of the upstream block-windowed SDPA reference."""
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_blocks = -(-seq_len // block_size)
    pad = num_blocks * block_size - seq_len
    if pad:
        query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    blocked_shape = (batch_size, num_heads, num_blocks, block_size, head_dim)
    query = query.view(blocked_shape)
    key = key.view(blocked_shape)
    value = value.view(blocked_shape)

    has_neighbours = look_backward_block > 0 or look_ahead_block > 0
    key_valid = None
    if pad or has_neighbours:
        key_valid = torch.ones(num_blocks, block_size, dtype=torch.bool, device=query.device)
        if pad:
            key_valid[-1, block_size - pad :] = False

    if not has_neighbours:
        keys, values, valid = key, value, key_valid
    else:
        keys, values, valids = [], [], []
        for offset in range(-look_backward_block, look_ahead_block + 1):
            if offset == 0:
                keys.append(key)
                values.append(value)
                valids.append(key_valid)
                continue
            shifted_key = torch.zeros_like(key)
            shifted_value = torch.zeros_like(value)
            valid = torch.zeros_like(key_valid)
            if offset > 0:
                shifted_key[:, :, : num_blocks - offset] = key[:, :, offset:]
                shifted_value[:, :, : num_blocks - offset] = value[:, :, offset:]
                valid[: num_blocks - offset] = key_valid[offset:]
            else:
                shifted_key[:, :, -offset:] = key[:, :, : num_blocks + offset]
                shifted_value[:, :, -offset:] = value[:, :, : num_blocks + offset]
                valid[-offset:] = key_valid[: num_blocks + offset]
            keys.append(shifted_key)
            values.append(shifted_value)
            valids.append(valid)
        keys = torch.cat(keys, dim=3)
        values = torch.cat(values, dim=3)
        valid = torch.cat(valids, dim=1)
    mask = None if valid is None else valid.view(1, 1, num_blocks, 1, -1)

    out = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
    return out.reshape(batch_size, num_heads, num_blocks * block_size, head_dim)[:, :, :seq_len]


class _FakeDiTAttention(torch.nn.Module):
    """Portable stand-in for the upstream DiTAttention (CPU testable)."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.qkv_proj = torch.nn.Linear(self.dim, 3 * self.inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [torch.nn.Linear(self.inner_dim, self.dim), torch.nn.Dropout(0.0)]
        )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        block_size=None,
        look_backward_block=0,
        look_ahead_block=0,
    ):
        batch_size = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.inner_dim, self.inner_dim, self.inner_dim], dim=-1)
        head_dim = self.inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query[:, :1], key[:, :1] = _apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)
        attention_weights = _block_sparse_attention(
            query,
            key,
            value,
            block_size=block_size,
            look_backward_block=look_backward_block,
            look_ahead_block=look_ahead_block,
        )
        attention_weights = attention_weights.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        attention_weights = attention_weights.to(query.dtype)
        return self.to_out[0](attention_weights)


class _FakeQwen25OmniEntry(torch.nn.Module):
    """Entry-class stand-in with the upstream dummy/fp32 semantics."""

    DEFAULT_SPEAKER = DEFAULT_SPEAKER

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.model_stage = getattr(vllm_config.model_config, "model_stage", None) or "thinker"
        self.talker = None
        self.token2wav = None
        self.talker_config = SimpleNamespace(
            dit_config=SimpleNamespace(enc_emb_dim=8, mel_dim=4),
            tts_text_start_token_id=1,
            tts_text_end_token_id=2,
        )
        self.token2wav_config = self.talker_config
        self._token2wav_conds: dict[str, torch.Tensor] = {}
        self._token2wav_ref_mels: dict[str, torch.Tensor] = {}
        self.thinker_embedding = torch.nn.Embedding(16, 4)

    def _module_device(self, module):
        if module is None or not list(module.parameters()):
            return torch.device("cpu")
        return next(module.parameters()).device

    def _init_special_tokens_embeddings(self):
        dev = self._module_device(self.talker)
        self.talker_embedding = torch.nn.Embedding(16, 4)
        self.tts_text_spk_token_ids = {"m02": 5, "Chelsie": 6}
        self.default_tts_text_spk_type = "m02"
        self.tts_text_spk_token_ids["prefix_caching"] = 5
        self.embed_text_bos_token = self.thinker_embedding(torch.tensor([1], dtype=torch.long, device=dev))
        self.embed_text_spk_tokens = {
            key: self.thinker_embedding(torch.tensor([value], dtype=torch.long, device=dev))
            for key, value in self.tts_text_spk_token_ids.items()
        }
        self.embed_text_eos_token = self.thinker_embedding(torch.tensor([2], dtype=torch.long, device=dev))
        self.embed_text_pad_token = self.thinker_embedding(torch.tensor([3], dtype=torch.long, device=dev))
        self.embed_codec_bos_token = self.talker_embedding(torch.tensor([8], dtype=torch.long, device=dev))
        self.embed_codec_eos_token = self.talker_embedding(torch.tensor([9], dtype=torch.long, device=dev))
        self.embed_codec_pad_token = self.talker_embedding(torch.tensor([7], dtype=torch.long, device=dev))
        return {"special_tokens"}

    def load_weights(self, weights):
        return {"entry"}

    def _codec_to_audio(self, codec_tokens, speaker=DEFAULT_SPEAKER):
        if self.token2wav is None:
            return None
        voice = next((key for key in self._token2wav_conds if key.lower() == speaker.lower()), speaker)
        cond = None
        ref_mel = None
        if voice in self._token2wav_conds and voice in self._token2wav_ref_mels:
            cond = self._token2wav_conds[voice]
            ref_mel = self._token2wav_ref_mels[voice]
        token2wav_dev = self._module_device(self.token2wav)
        if cond is None:
            cond = torch.zeros(
                (1, self.token2wav_config.dit_config.enc_emb_dim),
                device=token2wav_dev,
                dtype=torch.float32,
            )
        if ref_mel is None:
            ref_mel = torch.zeros(
                (1, 300, self.token2wav_config.dit_config.mel_dim),
                device=token2wav_dev,
                dtype=torch.float32,
            )
        self.last_cond, self.last_ref_mel = cond, ref_mel
        return torch.zeros(1, dtype=cond.dtype)


class _FakeThinker(torch.nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(8, 8)
        self.visual = torch.nn.Linear(8, 8)
        self.language_model = torch.nn.Linear(8, 8)

    def load_weights(self, weights):
        return {"thinker"}


class _FakeTalker(torch.nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.thinker_to_talker_proj = torch.nn.Linear(8, 4)
        self.language_model = torch.nn.Linear(4, 4)

    def load_weights(self, weights):
        return {"talker"}


def _make_fake_model_modules(monkeypatch: pytest.MonkeyPatch):
    package = _install_fake_module(monkeypatch, "vllm_omni.model_executor.models.qwen2_5_omni")
    entry = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni",
        Qwen2_5OmniForConditionalGeneration=_FakeQwen25OmniEntry,
        DEFAULT_SPEAKER=_FakeQwen25OmniEntry.DEFAULT_SPEAKER,
    )
    thinker = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker",
        Qwen2_5OmniThinkerForConditionalGeneration=_FakeThinker,
    )
    talker = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_talker",
        Qwen2_5OmniTalkerForConditionalGeneration=_FakeTalker,
    )
    token2wav = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav",
        DiTAttention=_FakeDiTAttention,
        block_sparse_attention=_block_sparse_attention,
        apply_rotary_pos_emb=_apply_rotary_pos_emb,
    )
    package.qwen2_5_omni = entry
    package.qwen2_5_omni_thinker = thinker
    package.qwen2_5_omni_talker = talker
    package.qwen2_5_omni_token2wav = token2wav
    return entry, thinker, talker, token2wav


def _make_fake_torch_npu(monkeypatch: pytest.MonkeyPatch):
    state = {"flash_calls": [], "nz_casts": 0}

    def aligned_16(tensor: torch.Tensor) -> torch.Tensor:
        pad = (-tensor.shape[0]) % 16
        if pad == 0:
            return tensor
        padding = (0,) * 2 * (tensor.dim() - 1) + (0, pad)
        return F.pad(tensor, padding)

    def _npu_flash_attention(query, key, value, mask, seq_len, scale_value, num_heads, num_kv_heads, out):
        state["flash_calls"].append(
            {
                "query": query.clone(),
                "mask": mask.clone(),
                "seq_len": seq_len.clone(),
                "scale_value": scale_value,
                "num_heads": num_heads,
            }
        )
        # Dense fp32 softmax attention from the additive mask (fake NZ = identity).
        total = query.shape[0]
        batch = seq_len.shape[0]
        dim = query.shape[-1]
        starts = torch.cumsum(torch.tensor([0] + list(seq_len.tolist()), dtype=torch.long), 0)
        attn = torch.empty((total, num_heads, dim), dtype=query.dtype)
        for r in range(batch):
            s0, s1 = int(starts[r]), int(starts[r + 1])
            q = query[s0:s1].permute(1, 0, 2).float()
            k = key[s0:s1].permute(1, 0, 2).float()
            v = value[s0:s1].permute(1, 0, 2).float()
            scores = torch.einsum("hqd,hkd->hqk", q, k) * scale_value
            scores = scores + mask[0, 0, s0:s1, s0:s1].float()
            weights = torch.softmax(scores, dim=-1)
            # ``attn`` is token-major (total, H, D); the einsum is head-major.
            attn[s0:s1] = torch.einsum("hqk,hkd->hqd", weights, v).permute(1, 0, 2).to(query.dtype)
        out.copy_(attn)
        return out

    def npu_format_cast(tensor, fmt):
        state["nz_casts"] += 1
        return tensor

    utils = _install_fake_module(
        monkeypatch,
        "vllm_ascend.utils",
        ACL_FORMAT_FRACTAL_NZ="FRACTAL_NZ",
        aligned_16=aligned_16,
        nd_to_nz_2d=lambda tensor: tensor,
    )
    _install_fake_module(monkeypatch, "vllm_ascend")
    npu = _install_fake_module(
        monkeypatch,
        "torch_npu",
        _npu_flash_attention=_npu_flash_attention,
        npu_format_cast=npu_format_cast,
    )
    return utils, npu, state


def _install_fake_parent_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    for dotted in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
    ):
        if dotted not in sys.modules:
            stub = types.ModuleType(dotted)
            monkeypatch.setitem(sys.modules, dotted, stub)


def _load_qwen25omni_310p_patch(monkeypatch: pytest.MonkeyPatch):
    root = _repo_root()
    patch_dir = root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    _install_fake_parent_chain(monkeypatch)
    _utils, _npu, state = _make_fake_torch_npu(monkeypatch)
    _make_fake_model_modules(monkeypatch)
    module = _load_source_module("qwen25omni_310p_patch_under_test", patch_dir / "qwen2_5_omni.py")
    return module, state


def _load_qwen25omni_registry(monkeypatch: pytest.MonkeyPatch, patch_module):
    root = _repo_root()
    patch_dir = root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    calls = {"qwen3_tts_talker": 0, "qwen3_tts_code2wav": 0, "worker": 0}

    def _count(name):
        def _fn():
            calls[name] += 1

        return _fn

    qwen3_tts = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=_count("qwen3_tts_talker"),
        apply_code2wav_patches=_count("qwen3_tts_code2wav"),
    )
    worker = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.worker",
        apply_patch=_count("worker"),
    )
    registry = _load_source_module("qwen25omni_310p_registry_under_test", patch_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch", registry)
    setattr(sys.modules["vllm_omni.platforms.npu._310p"], "patch", registry)
    registry.qwen3_tts = qwen3_tts
    registry.worker = worker
    registry.qwen2_5_omni = patch_module
    return registry, calls


def _stage_config(stage: str, arch: str = "Qwen2_5OmniForConditionalGeneration"):
    return SimpleNamespace(model_arch=arch, model_stage=stage)


# ===================================================================
#  Registry routing
# ===================================================================


@pytest.mark.parametrize("stage", ["thinker", "talker", "code2wav"])
def test_registry_routes_qwen25omni_stages(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    registry, calls = _load_qwen25omni_registry(monkeypatch, patch_module)

    registry.apply_model_patches(_stage_config(stage))

    assert patch_module._PATCHED_ENTRY
    if stage == "thinker":
        assert patch_module._PATCHED_THINKER and not patch_module._PATCHED_TALKER and not patch_module._PATCHED_CODE2WAV
    elif stage == "talker":
        assert patch_module._PATCHED_TALKER and not patch_module._PATCHED_THINKER and not patch_module._PATCHED_CODE2WAV
    else:
        assert patch_module._PATCHED_CODE2WAV and not patch_module._PATCHED_THINKER and not patch_module._PATCHED_TALKER
    # Qwen3-TTS routing must stay untouched by the Qwen2.5-Omni branch.
    assert calls["qwen3_tts_talker"] == 0
    assert calls["qwen3_tts_code2wav"] == 0


def test_registry_qwen3_tts_routing_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    registry, calls = _load_qwen25omni_registry(monkeypatch, patch_module)

    registry.apply_model_patches(_stage_config(None, arch="Qwen3TTSTalkerForConditionalGeneration"))
    registry.apply_model_patches(_stage_config(None, arch="Qwen3TTSCode2Wav"))
    registry.apply_model_patches(_stage_config(None, arch="SomeOtherArch"))

    assert calls["qwen3_tts_talker"] == 1
    assert calls["qwen3_tts_code2wav"] == 1
    assert not patch_module._PATCHED_ENTRY


def test_registry_missing_model_stage_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    registry, _ = _load_qwen25omni_registry(monkeypatch, patch_module)

    registry.apply_model_patches(SimpleNamespace(model_arch="Qwen2_5OmniForConditionalGeneration"))

    assert not patch_module._PATCHED_ENTRY


# ===================================================================
#  Entry class (all stages)
# ===================================================================


def test_entry_special_embeddings_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    model = patch_module._Qwen2_5OmniEntry310P(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model_stage="talker")))
    model.talker = torch.nn.Linear(4, 4)

    model._init_special_tokens_embeddings()

    assert model._npu_runtime_dtype is torch.float16
    for name in (
        "embed_text_bos_token",
        "embed_text_eos_token",
        "embed_text_pad_token",
        "embed_codec_bos_token",
        "embed_codec_eos_token",
        "embed_codec_pad_token",
    ):
        assert getattr(model, name).dtype is torch.float16, name
    assert all(t.dtype is torch.float16 for t in model.embed_text_spk_tokens.values())


def test_entry_code2wav_recast(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    model = patch_module._Qwen2_5OmniEntry310P(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model_stage="code2wav")))
    inner = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model.token2wav = inner
    model.token2wav.spk_dict = {"Chelsie": {"cond": torch.zeros(1, 8), "ref_mel": torch.zeros(1, 300, 4)}}
    model._token2wav_conds["Chelsie"] = torch.zeros(1, 8)
    model._token2wav_ref_mels["Chelsie"] = torch.zeros(1, 300, 4)

    model.load_weights(iter(()))

    assert all(p.dtype is torch.float16 for p in model.token2wav.parameters())
    assert model.token2wav.spk_dict["Chelsie"]["cond"].dtype is torch.float16
    assert model.token2wav.spk_dict["Chelsie"]["ref_mel"].dtype is torch.float16
    assert model._token2wav_conds["Chelsie"].dtype is torch.float16
    assert model._token2wav_ref_mels["Chelsie"].dtype is torch.float16


def test_entry_codec_to_audio_dummy_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    model = patch_module._Qwen2_5OmniEntry310P(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model_stage="code2wav")))
    model.token2wav = torch.nn.Linear(4, 4)
    model._token2wav_conds = {}
    model._token2wav_ref_mels = {}

    model._codec_to_audio(torch.tensor([[1, 2, 3]]), speaker="unknown-voice")

    assert model.last_cond.dtype is torch.float16
    assert model.last_ref_mel.dtype is torch.float16
    assert model.last_cond.shape == (1, 8)
    assert model.last_ref_mel.shape == (1, 300, 4)


def test_entry_talker_stage_skips_token2wav_recast(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    model = patch_module._Qwen2_5OmniEntry310P(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model_stage="talker")))
    model.token2wav = torch.nn.Linear(4, 4)

    model.load_weights(iter(()))

    # Talker stage must not touch the (absent) token2wav sub-model dtype.
    assert next(model.token2wav.parameters()).dtype is torch.float32


# ===================================================================
#  Thinker / talker sub-models
# ===================================================================


def test_thinker_tower_recast(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    model = patch_module._Qwen2_5OmniThinker310P(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model_stage="thinker")))

    model.load_weights(iter(()))

    assert all(p.dtype is torch.float16 for p in model.audio_tower.parameters())
    assert all(p.dtype is torch.float16 for p in model.visual.parameters())
    # The language model stays engine-managed (deploy dtype), not recast here.
    assert all(p.dtype is torch.float32 for p in model.language_model.parameters())


def test_talker_proj_recast(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    model = patch_module._Qwen2_5OmniTalker310P(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model_stage="talker")))

    model.load_weights(iter(()))

    assert all(p.dtype is torch.float16 for p in model.thinker_to_talker_proj.parameters())


# ===================================================================
#  DiT attention: 310P flash path vs portable SDPA reference
# ===================================================================


def _dit_config():
    return SimpleNamespace(dim=32, num_attention_heads=2, head_dim=16, dropout=0.0)


def _rope_for(seq_len: int, head_dim: int, device: torch.device = torch.device("cpu")):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)
    freqs = torch.stack((freqs, freqs), dim=-1).reshape(seq_len, head_dim)
    return freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)


def _make_dit_310p(patch_module, seed: int = 0):
    torch.manual_seed(seed)
    attn = patch_module._DiTAttention310P(_dit_config())
    return attn.to(torch.float16)


def _reference_forward(attn, hidden_states, cos, sin, block_size, lb, la):
    """Upstream DiTAttention semantics via the fake reference implementation."""
    batch_size = hidden_states.shape[0]
    qkv = attn.qkv_proj(hidden_states)
    query, key, value = qkv.split([attn.inner_dim, attn.inner_dim, attn.inner_dim], dim=-1)
    head_dim = attn.inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    query[:, :1], key[:, :1] = _apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)
    weights = _block_sparse_attention(query, key, value, block_size=block_size, look_backward_block=lb, look_ahead_block=la)
    weights = weights.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
    return attn.to_out[0](weights)


@pytest.mark.parametrize("block_size,lb,la", [(8, 0, 0), (6, 1, 1), (4, 1, 0)])
def test_dit_attention_310p_flash_matches_sdpa_reference(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    lb: int,
    la: int,
) -> None:
    patch_module, state = _load_qwen25omni_310p_patch(monkeypatch)
    patch_module._USE_SDPA_FALLBACK = False
    attn = _make_dit_310p(patch_module)

    hidden_states = torch.randn(2, 10, 32, dtype=torch.float16)
    cos, sin = _rope_for(10, 16)

    # DiT runs under inference mode in production (the RoPE head-slice is an
    # in-place view op that only legal there).
    with torch.inference_mode():
        reference = _reference_forward(attn, hidden_states, cos, sin, block_size, lb, la)
        actual = attn(
            hidden_states,
            (cos, sin),
            block_size=block_size,
            look_backward_block=lb,
            look_ahead_block=la,
        )

    assert len(state["flash_calls"]) == 1  # one flash call for the single forward
    assert state["nz_casts"] >= 1
    torch.testing.assert_close(actual.float(), reference.float(), atol=2e-2, rtol=1e-2)
    assert actual.dtype is torch.float16


def test_dit_attention_310p_alignment_and_seq_lens(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, state = _load_qwen25omni_310p_patch(monkeypatch)
    patch_module._USE_SDPA_FALLBACK = False
    attn = _make_dit_310p(patch_module)

    # B=2, T=10, block_size=6 -> stream=12/row, real=24, aligned=32.
    hidden_states = torch.randn(2, 10, 32, dtype=torch.float16)
    cos, sin = _rope_for(10, 16)
    with torch.inference_mode():
        attn(hidden_states, (cos, sin), block_size=6, look_backward_block=0, look_ahead_block=0)

    call = state["flash_calls"][-1]
    assert call["query"].shape[0] == 32  # 16-token aligned stream
    assert call["seq_len"].tolist() == [12, 20]  # block-padded rows; last row carries the align pad
    mask = call["mask"]
    assert mask.shape == (1, 1, 32, 32)
    additive = mask.squeeze().float()
    # Block-padded keys (positions 10:12 of each 12-token row) must be masked.
    for row in range(2):
        for pos in (10, 11):
            assert (additive[:, row * 12 + pos] == -torch.finfo(torch.float16).max).all()
    # Alignment-pad keys (24:32) must be masked.
    assert (additive[:, 24:] == -torch.finfo(torch.float16).max).all()
    # No query row is fully masked (own-block window always keeps valid keys).
    assert (additive != -torch.finfo(torch.float16).max).any(dim=1).all()


def test_dit_attention_310p_rope_first_head_only(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, state = _load_qwen25omni_310p_patch(monkeypatch)
    patch_module._USE_SDPA_FALLBACK = False
    attn = _make_dit_310p(patch_module)

    hidden_states = torch.randn(1, 8, 32, dtype=torch.float16)
    cos, sin = _rope_for(8, 16)

    # Reference with RoPE on head 0 only; a wrong all-heads rotation would
    # fail the equivalence against the upstream first-head-only semantics.
    with torch.inference_mode():
        reference = _reference_forward(attn, hidden_states, cos, sin, 4, 0, 0)
        actual = attn(hidden_states, (cos, sin), block_size=4)

    torch.testing.assert_close(actual.float(), reference.float(), atol=2e-2, rtol=1e-2)


def test_dit_attention_sdpa_fallback_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, state = _load_qwen25omni_310p_patch(monkeypatch)
    patch_module._USE_SDPA_FALLBACK = True
    attn = _make_dit_310p(patch_module)

    hidden_states = torch.randn(2, 10, 32, dtype=torch.float16)
    cos, sin = _rope_for(10, 16)
    with torch.inference_mode():
        actual = attn(hidden_states, (cos, sin), block_size=8, look_backward_block=1, look_ahead_block=1)
        reference = _reference_forward(attn, hidden_states, cos, sin, 8, 1, 1)
    assert len(state["flash_calls"]) == 0
    torch.testing.assert_close(actual.float(), reference.float(), atol=1e-3, rtol=1e-3)


def test_dit_attention_flash_max_tokens_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, state = _load_qwen25omni_310p_patch(monkeypatch)
    patch_module._USE_SDPA_FALLBACK = False
    patch_module._FLASH_MAX_ALIGNED_TOKENS = 16  # force the SDPA path for any test shape
    attn = _make_dit_310p(patch_module)

    hidden_states = torch.randn(2, 10, 32, dtype=torch.float16)
    cos, sin = _rope_for(10, 16)
    with torch.inference_mode():
        actual = attn(hidden_states, (cos, sin), block_size=8)
        reference = _reference_forward(attn, hidden_states, cos, sin, 8, 0, 0)
    assert len(state["flash_calls"]) == 0
    torch.testing.assert_close(actual.float(), reference.float(), atol=1e-3, rtol=1e-3)


# ===================================================================
#  Registration mechanics
# ===================================================================


def test_apply_code2wav_patches_swaps_dit_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    token2wav_module = sys.modules["vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav"]

    patch_module.apply_code2wav_patches()
    patch_module.apply_code2wav_patches()  # idempotent

    assert token2wav_module.DiTAttention is patch_module._DiTAttention310P
    assert sys.modules["vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni"].Qwen2_5OmniForConditionalGeneration is patch_module._Qwen2_5OmniEntry310P
    assert patch_module._PATCHED_CODE2WAV
    assert not patch_module._PATCHED_THINKER
    assert not patch_module._PATCHED_TALKER


def test_apply_thinker_patches_swaps_thinker_class(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_module, _ = _load_qwen25omni_310p_patch(monkeypatch)
    thinker_module = sys.modules["vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker"]
    talker_module = sys.modules["vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_talker"]

    patch_module.apply_thinker_patches()

    assert thinker_module.Qwen2_5OmniThinkerForConditionalGeneration is patch_module._Qwen2_5OmniThinker310P
    assert talker_module.Qwen2_5OmniTalkerForConditionalGeneration is _FakeTalker
