# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the 310P OmniVoice patch.

Same approach as the sibling 310P patch tests: the patch module is loaded from
source with fake dependency modules, so the wiring (gate, attention runtime
dtype cast, manual SDPA swap, CPU RoPE table, CPU-RNG fallback) is validated
without an NPU, model checkpoint or the native package.

Real-device behavior (fused-SDPA-free attention path, NPU RNG, first-packet
latency) is qualified in the 310P real-device phase of task-23.
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
import torch.nn.functional as F

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
# Fakes mirroring the upstream classes the patch rebinds
# ---------------------------------------------------------------------------


class _FakeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps


def _fake_fused_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, eps, num_heads, num_kv_heads):
    # Return [B, N, S, D] per tensor with k/v broadcast to the query-head
    # count (upstream contract: the prologue emits [B, N, S, D]).
    nq, nkv = num_heads, num_kv_heads
    q = qkv[:, :, :nq]
    k = qkv[:, :, nq : nq + nkv]
    v = qkv[:, :, nq + nkv :]
    k = k.repeat_interleave(nq // nkv, dim=2)
    v = v.repeat_interleave(nq // nkv, dim=2)
    return q, k, v


class _FakeAttentionOp:
    """Records the dtypes it received; returns a zero output."""

    class _Fallback:
        def __init__(self, parent):
            self._parent = parent

        def forward(self, q, k, v, metadata):
            self._parent.seen_dtypes.append(q.dtype)
            assert k.dtype == q.dtype and v.dtype == q.dtype
            # (B, H, S, D) like the real attention op output.
            return torch.zeros((1, q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=q.dtype)

    def __init__(self):
        self.seen_dtypes: list[torch.dtype] = []
        self.sdpa_fallback = self._Fallback(self)
        self.use_packed_varlen = False


class _FakeOmniVoiceAttention(nn.Module):
    """Minimal parent mirroring the attributes the 310P subclass uses."""

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_qkv_heads = self.num_heads + 2 * self.num_kv_heads
        self.qkv_proj = nn.Linear(hidden, self.num_qkv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = _FakeRMSNorm(self.head_dim)
        self.k_norm = _FakeRMSNorm(self.head_dim)
        self.attention_op = _FakeAttentionOp()
        self.use_packed_varlen = False

    def forward(self, hidden_states, rope_table, attn_metadata):  # pragma: no cover
        raise NotImplementedError("parent forward replaced by the 310P subclass")


def _install_omnivoice_fakes(monkeypatch: pytest.MonkeyPatch, is_310p_value: bool):
    """Install the fake dependency chain the patch module imports."""
    gen_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.omnivoice.omnivoice_generator",
        OmniVoiceAttention=_FakeOmniVoiceAttention,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.omnivoice.fused_qkv_rope",
        fused_qkv_norm_rope=_fake_fused_qkv_norm_rope,
    )
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models.omnivoice")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.attention.backends.sdpa",
        _sdpa_attention=lambda *a, **k: F.scaled_dot_product_attention(a[0], a[1], a[2]),
    )
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.attention.backends")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.attention")

    class _FakeDiffusionEngine:
        def run_startup_warmup(self):
            raise AssertionError("warmup should have been skipped")

    _install_fake_module(
        monkeypatch, "vllm_omni.diffusion.diffusion_engine", DiffusionEngine=_FakeDiffusionEngine
    )
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p", is_310p=lambda *a, **k: is_310p_value)
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni")
    return gen_mod, sys.modules["vllm_omni.diffusion.attention.backends.sdpa"]


def _load_patch(monkeypatch: pytest.MonkeyPatch, is_310p_value: bool):
    gen_mod, sdpa_mod = _install_omnivoice_fakes(monkeypatch, is_310p_value)
    patch = _load_source_module(
        "_310p_patch_omnivoice_under_test",
        _repo_root() / "vllm_omni/platforms/npu/_310p/patch/omnivoice.py",
    )
    return patch, gen_mod, sdpa_mod


# The *resolved* od_config.model_class_name is the architecture name from
# config.json ("OmniVoice"); the registry also exposes the class-name spelling
# "OmniVoicePipeline".  Both must pass the gate (machine-verified on 126/C3:
# the resolved value is "OmniVoice").
def _od_config(model_class_name: str = "OmniVoice"):
    return SimpleNamespace(model_class_name=model_class_name)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_should_patch_gating(monkeypatch):
    patch, _, _ = _load_patch(monkeypatch, True)
    assert patch.should_patch(_od_config("OmniVoice")) is True
    assert patch.should_patch(_od_config("OmniVoicePipeline")) is True
    assert patch.should_patch(_od_config("QwenImage21Pipeline")) is False
    assert patch.should_patch(SimpleNamespace(model_class_name=None)) is False
    # Off-310P the gate must refuse even for the right model.  The patch
    # module bound ``is_310p`` into its own namespace at load time.
    monkeypatch.setattr(patch, "is_310p", lambda *a, **k: False)
    assert patch.should_patch(_od_config("OmniVoice")) is False


def test_entry_noop_off_310p(monkeypatch):
    patch, gen_mod, sdpa_mod = _load_patch(monkeypatch, False)
    original_attn = gen_mod.OmniVoiceAttention
    original_sdpa = sdpa_mod._sdpa_attention
    patch.maybe_apply_omnivoice_310p_patches(_od_config())
    assert gen_mod.OmniVoiceAttention is original_attn
    assert sdpa_mod._sdpa_attention is original_sdpa
    assert not getattr(gen_mod, "_310p_rope_cpu_patched", False)


def test_entry_applies_all_patches_on_310p(monkeypatch):
    patch, gen_mod, sdpa_mod = _load_patch(monkeypatch, True)
    patch.maybe_apply_omnivoice_310p_patches(_od_config())
    # attention rebind (runtime-dtype cast)
    assert getattr(gen_mod, "_310p_attention_dtype_patched", False) is True
    assert gen_mod.OmniVoiceAttention is not _FakeOmniVoiceAttention
    # manual SDPA swap
    assert getattr(sdpa_mod, "_310p_manual_sdpa_patched", False) is True
    assert sdpa_mod._sdpa_attention is patch.manual_sdpa_forward
    # CPU rope table
    assert getattr(gen_mod, "_310p_rope_cpu_patched", False) is True
    assert gen_mod._precompute_rope_table is patch._precompute_rope_table_cpu
    # RNG: the fallback must be installed exactly when the probe reports the
    # NPU RNG unusable (probe is host-dependent: CPU test host -> False;
    # real 310P container with a working torch_npu RNG -> True).
    rng_ok = patch._NPU_RNG_OK
    assert rng_ok in (True, False)
    compat = getattr(torch.Generator, "_310p_cpu_compat", False)
    assert compat is (not rng_ok)
    if not rng_ok:
        assert gen_mod._gumbel_sample is patch.gumbel_sample_cpu
    # idempotency: a second application is a silent no-op
    patch.maybe_apply_omnivoice_310p_patches(_od_config())


# ---------------------------------------------------------------------------
# Attention runtime-dtype cast
# ---------------------------------------------------------------------------


def _cast_to_runtime(attn, dtype: torch.dtype) -> None:
    """Mirror the real loader: modules built under the fp16 default dtype."""
    attn.qkv_proj.to(dtype)
    attn.o_proj.to(dtype)
    attn.q_norm.weight.data = attn.q_norm.weight.data.to(dtype)
    attn.k_norm.weight.data = attn.k_norm.weight.data.to(dtype)


def test_attention_310p_casts_runtime_dtype_not_bf16(monkeypatch):
    patch, gen_mod, _ = _load_patch(monkeypatch, True)
    patch.maybe_apply_omnivoice_310p_patches(_od_config())
    cfg = SimpleNamespace(hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8)
    attn = gen_mod.OmniVoiceAttention(cfg)
    _cast_to_runtime(attn, torch.float16)
    hidden = torch.randn(16, 32, dtype=torch.float16)
    rope_table = torch.randn(16, 8, dtype=torch.float16)
    out = attn.forward(hidden, rope_table, None)
    assert out.shape == (16, 32)
    assert out.dtype == torch.float16
    # The prologue output must have been cast to the hidden dtype — never bf16.
    dtypes = attn.attention_op.seen_dtypes
    assert dtypes and all(dtype == torch.float16 for dtype in dtypes)
    # bf16 runtime: cast follows the hidden dtype (910 parity).
    attn_bf = gen_mod.OmniVoiceAttention(cfg)
    _cast_to_runtime(attn_bf, torch.bfloat16)
    out_bf = attn_bf.forward(torch.randn(16, 32, dtype=torch.bfloat16), rope_table.bfloat16(), None)
    assert out_bf.dtype == torch.bfloat16
    assert all(dtype == torch.bfloat16 for dtype in attn_bf.attention_op.seen_dtypes)


# ---------------------------------------------------------------------------
# Manual SDPA
# ---------------------------------------------------------------------------


def test_manual_sdpa_matches_fused_sdpa(monkeypatch):
    # bf16/fp32 only: this test host's CPU build lacks fp16 SDPA (the
    # NPU-side fp16 path is qualified by test_310p_omnivoice_real_device.py).
    patch, _, _ = _load_patch(monkeypatch, True)
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        q = torch.randn(1, 3, 17, 16, dtype=dtype)
        k = torch.randn(1, 3, 23, 16, dtype=dtype)
        v = torch.randn(1, 3, 23, 16, dtype=dtype)
        mask = torch.ones(1, 1, 17, 23, dtype=torch.bool)
        mask[0, 0, 3:, 19:] = False
        scale = 16**-0.5
        got = patch.manual_sdpa_forward(q, k, v, mask, False, scale)
        want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        assert got.dtype == dtype
        assert torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2)
        # Causal parity on a square shape (the only alignment both torch
        # backends agree on; OmniVoice itself never runs causal).
        qq = torch.randn(1, 3, 23, 16, dtype=dtype)
        got_c = patch.manual_sdpa_forward(qq, k, v, None, True, scale)
        want_c = F.scaled_dot_product_attention(qq, k, v, is_causal=True, scale=scale)
        assert torch.allclose(got_c.float(), want_c.float(), atol=2e-2, rtol=2e-2)


def test_manual_sdpa_chunked_path(monkeypatch):
    patch, _, _ = _load_patch(monkeypatch, True)
    monkeypatch.setattr(patch, "_SDPA_DENSE_LIMIT_BYTES", 1)  # force chunked path
    torch.manual_seed(1)
    q = torch.randn(1, 2, 600, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 600, 16, dtype=torch.float16)
    v = torch.randn(1, 2, 600, 16, dtype=torch.float16)
    got = patch.manual_sdpa_forward(q, k, v, None, False, 16**-0.5)
    want = F.scaled_dot_product_attention(q, k, v, scale=16**-0.5)
    assert torch.allclose(got.float(), want.float(), atol=3e-2, rtol=3e-2)


# ---------------------------------------------------------------------------
# CPU RoPE table
# ---------------------------------------------------------------------------


def test_rope_table_cpu_matches_reference():
    patch, _, _ = _load_patch(pytest.MonkeyPatch(), True)
    head_dim, max_len, theta = 64, 4096, 1_000_000.0
    table = patch._precompute_rope_table_cpu(head_dim, max_len, theta=theta, device="npu:0")
    assert table.device.type == "cpu"
    assert table.shape == (max_len, head_dim)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    ref = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    assert torch.equal(table, ref)


# ---------------------------------------------------------------------------
# CPU-RNG Gumbel fallback
# ---------------------------------------------------------------------------


def test_gumbel_sample_cpu_fallback_deterministic():
    patch, _, _ = _load_patch(pytest.MonkeyPatch(), True)
    logits = torch.randn(4, 64) * 5.0
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    gen3 = torch.Generator().manual_seed(7)
    out1 = patch.gumbel_sample_cpu(logits, 5.0, gen1)
    out2 = patch.gumbel_sample_cpu(logits, 5.0, gen2)
    out3 = patch.gumbel_sample_cpu(logits, 5.0, gen3)
    assert torch.equal(out1, out2)  # same seed -> same stream
    assert not torch.equal(out1, out3)  # different seed -> different stream
    assert torch.isfinite(out1).all()
    base = logits / 5.0
    assert (out1 - base).abs().min() > 0  # gumbel offset present


def test_generator_compat_materializes_cpu_generator_on_npu_request():
    if not getattr(torch.Generator, "_310p_cpu_compat", False):
        # Install the fallback machinery without an NPU present.
        mp = pytest.MonkeyPatch()
        patch, _, _ = _load_patch(mp, True)
        patch._patch_rng_fallback()
    g = torch.Generator(device="npu:0")
    assert g.device.type == "cpu"
    assert torch.Generator().manual_seed(1).device.type == "cpu"
    assert torch.Generator(device="cpu").device.type == "cpu"


# ---------------------------------------------------------------------------
# Startup dummy-run skip (main-process side)
# ---------------------------------------------------------------------------


def test_warmup_skip_gated(monkeypatch):
    patch, _, _ = _load_patch(monkeypatch, True)
    engine = sys.modules["vllm_omni.diffusion.diffusion_engine"].DiffusionEngine
    original = engine.run_startup_warmup
    # Wrong model -> no-op.
    patch.maybe_skip_startup_warmup_310p(_od_config("QwenImage21Pipeline"))
    assert engine.run_startup_warmup is original
    # Right model on 310P -> replaced; calling it must not run the dummy graph.
    patch.maybe_skip_startup_warmup_310p(_od_config("OmniVoicePipeline"))
    assert engine.run_startup_warmup is not original
    inst = engine.__new__(engine)
    engine.run_startup_warmup(inst)  # should not raise
    assert getattr(engine, "_310p_warmup_skipped", False) is True
    # Idempotent: second application is a silent no-op.
    patch.maybe_skip_startup_warmup_310p(_od_config("OmniVoicePipeline"))


def test_warmup_skip_noop_off_310p(monkeypatch):
    patch, _, _ = _load_patch(monkeypatch, False)
    engine = sys.modules["vllm_omni.diffusion.diffusion_engine"].DiffusionEngine
    original = engine.run_startup_warmup
    patch.maybe_skip_startup_warmup_310p(_od_config("OmniVoicePipeline"))
    assert engine.run_startup_warmup is original


# ---------------------------------------------------------------------------
# Deploy overlay
# ---------------------------------------------------------------------------


def test_deploy_overlay_is_fp16_and_minimal():
    import yaml

    root = _repo_root()
    cfg = yaml.safe_load((root / "vllm_omni/deploy/omnivoice_310p.yaml").read_text())
    assert cfg["pipeline"] == "omnivoice"
    stage = cfg["stages"][0]
    assert stage["dtype"] == "float16"
    assert stage["enforce_eager"] is True
    assert stage["devices"] == "0"
    assert stage["additional_config"]["omnivoice_asr"]["load_asr_on_startup"] is False
    # The upstream base file must stay untouched (float32 reference).
    base = yaml.safe_load((root / "vllm_omni/deploy/omnivoice.yaml").read_text())
    assert base["stages"][0]["dtype"] == "float32"
