# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the 310P Ming-3.0-TTS-tiny (MoE) patch.

Same approach as ``test_310p_ming_tts_patches.py`` (dense line): the patch
modules are loaded from source with fake dependencies, so the wiring
(KDA op rebind, registry dispatch, FP16/CFM pins, deploy overlay) is
validated without an NPU, model checkpoint or the real vLLM runtime.

The KDA op tests additionally pin the *numerics*: the pure-torch gate /
recurrence / decode-row implementations are checked against an independent
naive per-token reference (fp64), including L2 normalization, safe-gate
clamping, GQA head expansion, state carry-over and slot selection.

Real-device behavior (eager flow head numerics, KDA state pool dtypes,
first-chunk latency, audio QA) is qualified in the 310P real-device phase
of task-29.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

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


def _install_package_chain(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def _load_kda_module(monkeypatch: pytest.MonkeyPatch):
    """Load the KDA ops module from source (imports torch only)."""
    root = _repo_root()
    return _load_source_module(
        "ming3_tts_kda_under_test",
        root / "vllm_omni/platforms/npu/_310p/patch/ming3_tts_kda.py",
    )


# ---------------------------------------------------------------------------
# Naive references (independent of the op under test)
# ---------------------------------------------------------------------------


def _ref_gate(g, A, head_dim, g_bias=None, lower_bound=None):
    """Naive fused_kda_gate: per-token gate, no cumsum."""
    T, HD = g.shape
    H = A.numel()
    x = g.view(T, H, head_dim).double()
    if g_bias is not None:
        x = x + g_bias.view(H, head_dim).double()
    a = A.view(H).double().exp()
    if lower_bound is not None:
        y = lower_bound * torch.sigmoid(a.view(1, H, 1) * x)
    else:
        sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
        y = -a.view(1, H, 1) * sp
    return y.view(T, H, head_dim)


def _ref_recurrent(q, k, v, g, beta, scale, state, use_l2norm=True, tol=1e-6):
    """Naive per-token gated delta rule; fp64.  Returns (out, final_state)."""
    T, H, D = q.shape
    HV, V = v.shape[1], v.shape[2]
    if HV != H:
        rep = HV // H
        q = q.unsqueeze(1).expand(-1, rep, -1, -1).reshape(T, HV, D)
        k = k.unsqueeze(1).expand(-1, rep, -1, -1).reshape(T, HV, D)
    if use_l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * scale
    S = state.clone()
    out = torch.empty(T, HV, V, dtype=torch.float64)
    for t in range(T):
        S = S * torch.exp(g[t]).unsqueeze(1)  # [HV, V, D] * [HV, 1, D]
        u = torch.einsum("hvd,hd->hv", S, k[t])
        vt = (v[t] - u) * beta[t].unsqueeze(-1)
        S = S + vt.unsqueeze(1) * k[t].unsqueeze(2)
        out[t] = torch.einsum("hvd,hd->hv", S, q[t])
    return out, S


# ---------------------------------------------------------------------------
# KDA gate
# ---------------------------------------------------------------------------


class TestKdaGate:
    def test_safe_gate_branch(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        torch.manual_seed(0)
        H, D = 3, 8
        g = torch.randn(1, 7, H * D, dtype=torch.float32)
        A_log = torch.nn.Parameter(torch.randn(1, 1, H, 1))
        bias = torch.randn(H * D)
        got = kda.kda_gate_torch(g, A_log, D, g_bias=bias, lower_bound=-5.0)
        want = _ref_gate(
            g[0].double(), A_log.double(), D, g_bias=bias, lower_bound=-5.0
        ).float().unsqueeze(0)
        assert got.shape == (1, 7, H, D)
        assert torch.allclose(got, want, atol=1e-5)
        # safe gate range: (lower_bound, 0]
        assert (got < 0).all() and (got > -5.0).all()

    def test_softplus_branch(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        torch.manual_seed(1)
        H, D = 2, 4
        g = torch.randn(1, 5, H * D)
        A_log = torch.nn.Parameter(torch.randn(1, 1, H, 1))
        got = kda.kda_gate_torch(g, A_log, D)
        want = _ref_gate(g[0].double(), A_log.double(), D).float().unsqueeze(0)
        assert torch.allclose(got, want, atol=1e-5)
        assert (got < 0).all()

    def test_shape_passthrough(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        g = torch.randn(2, 9, 24)
        A_log = torch.nn.Parameter(torch.ones(1, 1, 3, 1))
        got = kda.kda_gate_torch(g, A_log, 8)
        assert got.shape == (2, 9, 3, 8)


# ---------------------------------------------------------------------------
# Chunked (prefill) recurrence
# ---------------------------------------------------------------------------


class TestChunkKda:
    @staticmethod
    def _case(seed: int, H: int, HV: int, D: int, V: int, T: int, use_l2: bool, init: bool):
        torch.manual_seed(seed)
        q = torch.randn(1, T, H, D)
        k = torch.randn(1, T, H, D)
        v = torch.randn(1, T, HV, V)
        g = -0.2 * torch.rand(1, T, HV, D)  # stable negative gates
        beta = torch.rand(1, T, HV) * 0.9 + 0.05
        cu = torch.tensor([0, T], dtype=torch.int32)
        init_state = None
        if init:
            init_state = torch.randn(1, HV, V, D)
        return q, k, v, g, beta, cu, init_state

    def test_matches_naive_reference(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        for seed, use_l2, init in ((0, True, False), (1, True, True), (2, False, False)):
            q, k, v, g, beta, cu, init_state = self._case(seed, 3, 3, 8, 8, 6, use_l2, init)
            scale = 8 ** -0.5
            out, final = kda.chunk_kda_310p(
                q, k, v, g, beta,
                initial_state=init_state,
                output_final_state=True,
                scale=scale,
                use_qk_l2norm_in_kernel=use_l2,
                cu_seqlens=cu,
            )
            ref_out, ref_state = _ref_recurrent(
                q[0].double(), k[0].double(), v[0].double(), g[0].double(),
                beta[0].double(), scale,
                init_state[0].double() if init_state is not None
                else torch.zeros(3, 8, 8),
                use_l2norm=use_l2,
            )
            assert torch.allclose(out[0].double(), ref_out, atol=2e-4)
            assert torch.allclose(final[0].double(), ref_state, atol=2e-4)
            assert out.dtype == q.dtype

    def test_gqa_head_expansion(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        torch.manual_seed(3)
        H, HV, D, V, T = 2, 4, 6, 6, 5
        q = torch.randn(1, T, H, D)
        k = torch.randn(1, T, H, D)
        v = torch.randn(1, T, HV, V)
        g = -0.1 * torch.rand(1, T, HV, D)
        beta = torch.rand(1, T, HV)
        cu = torch.tensor([0, T], dtype=torch.int32)
        out, final = kda.chunk_kda_310p(
            q, k, v, g, beta, scale=D ** -0.5,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu, output_final_state=True,
        )
        ref_out, ref_state = _ref_recurrent(
            q[0].double(), k[0].double(), v[0].double(), g[0].double(),
            beta[0].double(), D ** -0.5, torch.zeros(HV, V, D),
        )
        assert torch.allclose(out[0].double(), ref_out, atol=2e-4)
        assert torch.allclose(final[0].double(), ref_state, atol=2e-4)

    def test_multi_sequence_varlen(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        torch.manual_seed(4)
        H, D, V = 2, 6, 6
        T1, T2 = 4, 3
        q = torch.randn(1, T1 + T2, H, D)
        k = torch.randn(1, T1 + T2, H, D)
        v = torch.randn(1, T1 + T2, H, V)
        g = -0.1 * torch.rand(1, T1 + T2, H, D)
        beta = torch.rand(1, T1 + T2, H)
        cu = torch.tensor([0, T1, T1 + T2], dtype=torch.int32)
        init = torch.randn(2, H, V, D)
        out, final = kda.chunk_kda_310p(
            q, k, v, g, beta, initial_state=init, output_final_state=True,
            scale=D ** -0.5, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
        )
        for n, (bos, eos) in enumerate(((0, T1), (T1, T1 + T2))):
            ref_out, ref_state = _ref_recurrent(
                q[0, bos:eos].double(), k[0, bos:eos].double(), v[0, bos:eos].double(),
                g[0, bos:eos].double(), beta[0, bos:eos].double(), D ** -0.5,
                init[n].double(),
            )
            assert torch.allclose(out[0, bos:eos].double(), ref_out, atol=2e-4)
            assert torch.allclose(final[n].double(), ref_state, atol=2e-4)


# ---------------------------------------------------------------------------
# Decode row (fused_recurrent_kda)
# ---------------------------------------------------------------------------


class TestFusedRecurrentKda:
    def test_decode_row_state_pool(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        torch.manual_seed(5)
        H, D, V, slots = 3, 8, 8, 4
        T = 1  # one decode row
        q = torch.randn(1, T, H, D)
        k = torch.randn(1, T, H, D)
        v = torch.randn(1, T, H, V)
        g = -0.1 * torch.rand(1, T, H, D)
        beta = torch.rand(1, T, H)
        pool = torch.randn(slots, H, V, D)
        pool0 = pool.clone()  # capture BEFORE the op (it overwrites slot 2)
        cu = torch.tensor([0, T], dtype=torch.int32)
        idx = torch.tensor([2], dtype=torch.long)
        out, returned = kda.fused_recurrent_kda_310p(
            q, k, v, g, beta=beta, initial_state=pool, inplace_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu, ssm_state_indices=idx,
        )
        assert returned is pool
        ref_out, ref_state = _ref_recurrent(
            q[0].double(), k[0].double(), v[0].double(), g[0].double(),
            beta[0].double(), D ** -0.5, pool0[2].double(),
        )
        assert torch.allclose(out[0].double(), ref_out, atol=2e-4)
        assert torch.allclose(pool[2].double(), ref_state, atol=2e-4)

    def test_multi_token_row(self, monkeypatch: pytest.MonkeyPatch):
        """Prefill-length row through the decode entry (chunked prefill is
        the hot path, but the recurrence must agree for L>1 as well)."""
        kda = _load_kda_module(monkeypatch)
        torch.manual_seed(6)
        H, D, V = 2, 6, 6
        T = 7
        q = torch.randn(1, T, H, D)
        k = torch.randn(1, T, H, D)
        v = torch.randn(1, T, H, V)
        g = -0.1 * torch.rand(1, T, H, D)
        beta = torch.rand(1, T, H)
        pool = torch.zeros(1, H, V, D)
        cu = torch.tensor([0, T], dtype=torch.int32)
        idx = torch.tensor([0], dtype=torch.long)
        out, _ = kda.fused_recurrent_kda_310p(
            q, k, v, g, beta=beta, initial_state=pool, inplace_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu, ssm_state_indices=idx,
        )
        ref_out, ref_state = _ref_recurrent(
            q[0].double(), k[0].double(), v[0].double(), g[0].double(),
            beta[0].double(), D ** -0.5, torch.zeros(H, V, D),
        )
        assert torch.allclose(out[0].double(), ref_out, atol=2e-4)
        assert torch.allclose(pool[0].double(), ref_state, atol=2e-4)

    def test_spec_multi_accept_raises(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        torch.manual_seed(7)
        q = torch.randn(1, 1, 2, 4)
        with pytest.raises(NotImplementedError):
            kda.fused_recurrent_kda_fwd_310p(
                q, q.clone(), torch.randn(1, 1, 2, 4),
                -0.1 * torch.rand(1, 1, 2, 4), torch.rand(1, 1, 2),
                cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
                ssm_state_indices=torch.tensor([0], dtype=torch.long),
                initial_state=torch.zeros(1, 2, 4, 4),
                num_accepted_tokens=torch.tensor([2], dtype=torch.long),
            )


# ---------------------------------------------------------------------------
# Conv swap kwargs filter
# ---------------------------------------------------------------------------


class TestConvSwap:
    def test_filter_kwargs_drops_unknown(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)

        def _strict(x, weight, metadata=None):
            return x

        got = kda._filter_kwargs(_strict, {"weight": 1, "metadata": "m", "junk": 3})
        assert got == {"weight": 1, "metadata": "m"}

        def _permissive(x, weight, **kw):
            return x

        assert kda._filter_kwargs(
            _permissive, {"weight": 1, "junk": 3}
        ) == {"weight": 1, "junk": 3}

    def test_conv_swap_factory_returns_wrappers(self, monkeypatch: pytest.MonkeyPatch):
        kda = _load_kda_module(monkeypatch)
        calls = []

        class _FakeConv:
            def causal_conv1d_fn(x, weight, bias=None, activation=None, **kw):
                calls.append(("fn", activation))
                return x

            def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, **kw):
                calls.append(("update", activation))
                return x

        monkeypatch.setitem(sys.modules, "vllm_ascend", types.ModuleType("vllm_ascend"))
        ops_mod = types.ModuleType("vllm_ascend.ops")
        conv_mod = types.ModuleType("vllm_ascend.ops.causal_conv1d")
        conv_mod.causal_conv1d_fn = _FakeConv.causal_conv1d_fn
        conv_mod.causal_conv1d_update = _FakeConv.causal_conv1d_update
        monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops_mod)
        monkeypatch.setitem(sys.modules, "vllm_ascend.ops.causal_conv1d", conv_mod)

        fn, update = kda.make_conv_swaps()
        x = torch.randn(2, 3)
        assert fn(x, torch.randn(3, 4), activation="silu", metadata=object()) is x
        assert update(x, x, torch.randn(3, 4), validate_data=True) is x
        assert calls == [("fn", "silu"), ("update", None)]


# ---------------------------------------------------------------------------
# Install / rebind wiring
# ---------------------------------------------------------------------------


class TestInstallKdaOps:
    @staticmethod
    def _fake_bm3(monkeypatch: pytest.MonkeyPatch, missing: tuple = ()):
        bm3 = types.ModuleType("vllm.model_executor.models.bailing_moe_v3")
        for name in (
            "chunk_kda",
            "fused_recurrent_kda",
            "fused_recurrent_kda_fwd",
            "fused_kda_gate",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        ):
            if name not in missing:
                setattr(bm3, name, lambda *a, **k: None)
        _install_package_chain(
            monkeypatch,
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.models",
        )
        monkeypatch.setitem(sys.modules, "vllm.model_executor.models.bailing_moe_v3", bm3)
        return bm3

    def _load_install_module(self, monkeypatch: pytest.MonkeyPatch):
        # Conv swap factory imports vllm_ascend.ops.causal_conv1d at call time;
        # pre-seed a fake so install succeeds without the image package.
        conv_mod = types.ModuleType("vllm_ascend.ops.causal_conv1d")
        conv_mod.causal_conv1d_fn = lambda *a, **k: None
        conv_mod.causal_conv1d_update = lambda *a, **k: None
        _install_package_chain(
            monkeypatch,
            "vllm_ascend",
            "vllm_ascend.ops",
            "vllm_ascend.ops.causal_conv1d",
        )
        monkeypatch.setitem(sys.modules, "vllm_ascend.ops.causal_conv1d", conv_mod)
        # The gated RMS-norm rebind touches these two namespaces at install.
        la_mod = types.ModuleType("vllm_ascend.ops.layernorm")
        la_mod.rms_norm_gated = lambda *a, **k: None
        _install_package_chain(
            monkeypatch, "vllm_ascend.ops.triton", "vllm_ascend.ops.triton.kda"
        )
        kda_triton_mod = types.ModuleType("vllm_ascend.ops.triton.kda.kda")
        kda_triton_mod.rms_norm_gated = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "vllm_ascend.ops.layernorm", la_mod)
        monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.kda.kda", kda_triton_mod)
        return _load_kda_module(monkeypatch)

    def test_rebind_all_six_ops(self, monkeypatch: pytest.MonkeyPatch):
        kda = self._load_install_module(monkeypatch)
        bm3 = self._fake_bm3(monkeypatch)
        before = {n: getattr(bm3, n) for n in (
            "chunk_kda", "fused_recurrent_kda", "fused_recurrent_kda_fwd",
            "fused_kda_gate", "causal_conv1d_fn", "causal_conv1d_update",
        )}
        kda.install_kda_310p_ops()
        assert getattr(bm3, "fused_kda_gate") is kda.kda_gate_torch
        assert getattr(bm3, "chunk_kda") is kda.chunk_kda_310p
        assert getattr(bm3, "fused_recurrent_kda") is kda.fused_recurrent_kda_310p
        assert getattr(bm3, "fused_recurrent_kda_fwd") is kda.fused_recurrent_kda_fwd_310p
        assert getattr(bm3, "causal_conv1d_fn") is not before["causal_conv1d_fn"]
        assert getattr(bm3, "causal_conv1d_update") is not before["causal_conv1d_update"]
        # The gated RMS-norm (TRITON on upstream) is rebound in both import
        # sites the KDA block can resolve it from.
        la_mod = sys.modules["vllm_ascend.ops.layernorm"]
        kda_triton_mod = sys.modules["vllm_ascend.ops.triton.kda.kda"]
        assert la_mod.rms_norm_gated is kda.rms_norm_gated_310p
        assert kda_triton_mod.rms_norm_gated is kda.rms_norm_gated_310p

    def test_rms_norm_gated_matches_reference(self, monkeypatch: pytest.MonkeyPatch):
        # Kernel semantics (IS_RMS_NORM path, fp32 compute, gate after the
        # weight/bias): y = x*rsqrt(mean(x^2)+eps)*w + b; y *= silu(g)/sigmoid(g).
        # Inputs are fp32; the fp64 reference is computed from the same
        # fp32 values (the implementation downcasts to fp32 like upstream).
        kda = _load_kda_module(monkeypatch)
        x32 = torch.randn(4, 3, 8, dtype=torch.float32)
        g32 = torch.randn(4, 3, 8, dtype=torch.float32)
        w32 = torch.randn(8, dtype=torch.float32) + 2.0
        b32 = torch.randn(8, dtype=torch.float32)
        x, g, w, b = (t.double() for t in (x32, g32, w32, b32))
        eps = 1e-5
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w + b
        out = kda.rms_norm_gated_310p(x32, g32, w32, b32, "swish", eps=eps)
        assert torch.allclose(out.double(), normed * g * torch.sigmoid(g), atol=1e-6)
        out2 = kda.rms_norm_gated_310p(x32, g32, w32, b32, "sigmoid", eps=eps)
        assert torch.allclose(out2.double(), normed * torch.sigmoid(g), atol=1e-6)
        # Unsupported variants must fail loud, not approximate.
        with pytest.raises(RuntimeError):
            kda.rms_norm_gated_310p(x, g, w, b, "swish", residual=torch.zeros_like(x))
        with pytest.raises(RuntimeError):
            kda.rms_norm_gated_310p(x, g, w, b, "swish", prenorm=True)
        # Input dtype is preserved (fp32 compute, cast back).
        out16 = kda.rms_norm_gated_310p(x.half(), g.half(), w.half(), b.half(), "swish")
        assert out16.dtype is torch.float16
        # The out= buffer path writes in place.
        buf = torch.empty_like(x)
        ret = kda.rms_norm_gated_310p(x, g, w, b, "swish", eps=eps, out=buf)
        assert ret is buf

    def test_kill_switch_is_noop(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_OMNI_310P_MING3TTS_PATCH", "0")
        kda = self._load_install_module(monkeypatch)
        bm3 = self._fake_bm3(monkeypatch)
        before = getattr(bm3, "chunk_kda")
        kda.install_kda_310p_ops()
        assert getattr(bm3, "chunk_kda") is before

    def test_upstream_layout_change_skips_install(self, monkeypatch: pytest.MonkeyPatch):
        kda = self._load_install_module(monkeypatch)
        bm3 = self._fake_bm3(monkeypatch, missing=("fused_kda_gate",))
        kda.install_kda_310p_ops()
        # Layout check failed: nothing rebound, no exception.
        assert getattr(bm3, "chunk_kda") is not kda.chunk_kda_310p


# ---------------------------------------------------------------------------
# Model patch wiring (fake Ming3TTSLLMModel dependency)
# ---------------------------------------------------------------------------


class _FakeMing3LLMBase(nn.Module):
    """Stand-in for ``ming3_tts_llm.Ming3TTSLLMModel`` (same __init__ contract)."""

    latent_patch_key = "ming3_latent_patch"
    has_patch_key = "ming3_has_patch"

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        dtype = getattr(vllm_config, "dtype", None)
        self.fm_dtype = (
            torch.float16 if str(dtype).startswith("float16") or dtype is torch.float16
            else torch.bfloat16
        )
        self.ming_config = SimpleNamespace(sample_rate=24000)
        self.model = nn.Module()
        self._pending_postprocess_updates: dict[str, dict] = {}
        self.flowloss = nn.Sequential(nn.Linear(8, 8))
        self.linear_proj_audio = nn.Linear(8, 4)
        self.stop_head = nn.Linear(4, 2)
        self.spk_head = nn.Linear(16, 4)
        for m in (self.flowloss, self.linear_proj_audio, self.stop_head, self.spk_head):
            m.to(dtype=self.fm_dtype)

    def _maybe_build_cfm_graph(self, z_diff_cond):
        return SimpleNamespace(executed=True)  # base default: engaged

    def make_omni_output(
        self,
        model_outputs,
        *,
        model_intermediate_buffer=None,
        seq_token_counts=None,
        request_token_spans=None,
        request_sample_eligible=None,
        **kwargs,
    ):
        # Test fixture: the 310P override wraps this passthrough.
        return self._npu_test_output

    def on_requests_finished(self, finished_req_ids) -> None:
        pass


class _FakeAudioVAE(nn.Module):
    """Stand-in for ``common.ming.audio_vae.AudioVAE`` (decoder-only build)."""

    def __init__(self, config, *, build_encoder: bool = True) -> None:
        super().__init__()
        self.decoder = SimpleNamespace(upsampling=SimpleNamespace())
        self.proj = nn.Linear(4, 4)


class _FakeMing3WrapperBase(nn.Module):
    """Stand-in for ``ming3_tts.Ming3TTSForConditionalGeneration``."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model = SimpleNamespace(_npu310_attach_audio_vae=None)
        self.ming_config = SimpleNamespace(
            audio_tokenizer_config=SimpleNamespace(tag="fake"),
            sample_rate=24000,
        )
        self.step_vae = None
        self.audio_vae = None
        self._probe = nn.Linear(2, 2)


# Matches ``vllm_omni.model_executor.models.output_templates.OmniOutput``
# (a NamedTuple; the 310P patch reads attributes and uses ``_replace``).
_FakeOmniOutput = namedtuple(
    "OmniOutput",
    ["text_hidden_states", "multimodal_outputs", "intermediate_tensors"],
    defaults=[None, None, None],
)


def _fake_ming3_llm(monkeypatch: pytest.MonkeyPatch):
    _install_package_chain(
        monkeypatch,
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.model_executor.models.ming3_tts",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
        "vllm_omni.platforms.npu._310p.patch",
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.output_templates",
        OmniOutput=_FakeOmniOutput,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.ming3_tts.ming3_tts",
        Ming3TTSForConditionalGeneration=_FakeMing3WrapperBase,
        AudioVAE=_FakeAudioVAE,
    )
    fake_llm = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.ming3_tts.ming3_tts_llm",
        Ming3TTSLLMModel=_FakeMing3LLMBase,
        KEY_REQUEST_ID="ming3_request_id",
        _normalize_request_infos=lambda buf: list(buf or []),
        _get_request_token_counts=lambda hidden, infos, a=None, b=None: [1] * len(infos),
    )
    # The kda ops module is imported by the model patch module at top level.
    kda_mod = _load_source_module(
        "vllm_omni.platforms.npu._310p.patch.ming3_tts_kda",
        _repo_root() / "vllm_omni/platforms/npu/_310p/patch/ming3_tts_kda.py",
    )
    monkeypatch.setitem(
        sys.modules, "vllm_omni.platforms.npu._310p.patch.ming3_tts_kda", kda_mod
    )
    return fake_llm


def _load_model_patch(monkeypatch: pytest.MonkeyPatch):
    fake_llm = _fake_ming3_llm(monkeypatch)
    patch = _load_source_module(
        "ming3_tts_patch_under_test",
        _repo_root() / "vllm_omni/platforms/npu/_310p/patch/ming3_tts.py",
    )
    return patch, fake_llm


class TestMing3TTS310PClass:
    def test_subclass_and_fp16_pin(self, monkeypatch: pytest.MonkeyPatch):
        patch, _ = _load_model_patch(monkeypatch)
        assert issubclass(patch._Ming3TTSLLMModel310P, _FakeMing3LLMBase)
        # BF16 upstream config: the pin must still land on FP16.
        model = patch._Ming3TTSLLMModel310P(vllm_config=SimpleNamespace(dtype="bfloat16"))
        assert model.fm_dtype is torch.float16
        for sub in (model.flowloss, model.linear_proj_audio, model.stop_head, model.spk_head):
            params = [p.dtype for p in sub.parameters()]
            assert params and all(d is torch.float16 for d in params)

    def test_cfm_graph_hard_off(self, monkeypatch: pytest.MonkeyPatch):
        patch, _ = _load_model_patch(monkeypatch)
        model = patch._Ming3TTSLLMModel310P(vllm_config=SimpleNamespace(dtype="float16"))
        assert model._maybe_build_cfm_graph(torch.ones(1, 8)) is None

    def test_cfm_api_shims(self, monkeypatch: pytest.MonkeyPatch):
        # main's Ming3TTSLLMModel calls the CFM-graph-cache API that only
        # main's MingLLMModel provides; the tag tree carries the older
        # dense class.  The 310P subclass must carry the missing methods.
        patch, _ = _load_model_patch(monkeypatch)
        for method in (
            "_init_cfm_graph_state",
            "_warmup_cfm_graphs",
            "_draw_request_cfm_noises",
        ):
            assert callable(getattr(patch._Ming3TTSLLMModel310P, method))
        vcfg = SimpleNamespace(
            dtype="float16",
            compilation_config=SimpleNamespace(
                cudagraph_mode="NONE",
                cudagraph_capture_sizes=[1],
                cudagraph_num_of_warmups=1,
            ),
            model_config=SimpleNamespace(enforce_eager=True),
            scheduler_config=SimpleNamespace(max_num_seqs=1),
        )
        model = patch._Ming3TTSLLMModel310P(vllm_config=vcfg)
        model._init_cfm_graph_state()
        assert model._cfm_graph_enabled is False  # enforce_eager pins the gate off
        assert model._cfm_graphs == {}
        assert model._cfm_graph_failed_batch_sizes == set()
        assert model._cfm_graph_batch_sizes == [1]
        model._warmup_cfm_graphs()
        assert model._cfm_graph_warmed_up is True
        # Non-eager + non-NONE mode: the flag must come back on (it stays
        # inert because _maybe_build_cfm_graph hard-returns None here).
        vcfg.compilation_config.cudagraph_mode = "FULL_AND_PIECEWISE"
        vcfg.model_config.enforce_eager = False
        model2 = patch._Ming3TTSLLMModel310P(vllm_config=vcfg)
        model2._init_cfm_graph_state()
        assert model2._cfm_graph_enabled is True
        assert model2._maybe_build_cfm_graph(torch.ones(1, 8)) is None

    def test_kill_switch_skips_swap(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_OMNI_310P_MING3TTS_PATCH", "0")
        patch, fake_llm = _load_model_patch(monkeypatch)
        patch.apply_ming3_tts_tiny_patches()
        assert fake_llm.Ming3TTSLLMModel is _FakeMing3LLMBase
        wrapper_mod = sys.modules["vllm_omni.model_executor.models.ming3_tts.ming3_tts"]
        assert wrapper_mod.Ming3TTSForConditionalGeneration is _FakeMing3WrapperBase

    def test_swap_patches_both_classes(self, monkeypatch: pytest.MonkeyPatch):
        patch, fake_llm = _load_model_patch(monkeypatch)
        patch.apply_ming3_tts_tiny_patches()
        assert fake_llm.Ming3TTSLLMModel is patch._Ming3TTSLLMModel310P
        wrapper_mod = sys.modules["vllm_omni.model_executor.models.ming3_tts.ming3_tts"]
        assert wrapper_mod.Ming3TTSForConditionalGeneration is patch._Ming3TTSWrapper310P

    def test_wrapper_loads_audio_vae_only(self, monkeypatch: pytest.MonkeyPatch):
        # 310P: _load_step_vae must build the plain AudioVAE decoder and
        # skip StepVAEDecoder (PackedQwen2 needs vllm.vllm_flash_attn,
        # CUDA-only), attaching the VAE to the LLM model for the V1
        # streaming decode.
        patch, _ = _load_model_patch(monkeypatch)
        wrapper = patch._Ming3TTSWrapper310P(
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16))
        )
        attached = []
        wrapper.model = SimpleNamespace(_npu310_attach_audio_vae=attached.append)
        weights = [
            ("audio.proj.weight", torch.ones(4, 4)),
            ("audio.proj.bias", torch.zeros(4)),
        ]
        loaded = wrapper._load_step_vae(weights)
        assert wrapper.audio_vae is not None and wrapper.step_vae is None
        assert attached and attached[0] is wrapper.audio_vae
        assert loaded == {
            "audio_vae.proj.weight",
            "audio_vae.proj.bias",
        }
        assert wrapper.audio_vae.decoder.upsampling.frame_holdback is True
        # Unknown tensor name must fail loud.
        with pytest.raises(RuntimeError, match="unknown tensor"):
            wrapper._load_step_vae([("audio.missing", torch.ones(4, 4))])

    def _run_v1_decode_step(
        self, monkeypatch: pytest.MonkeyPatch, finished: bool
    ) -> tuple[object, object, list]:
        patch, _ = _load_model_patch(monkeypatch)
        model = patch._Ming3TTSLLMModel310P(vllm_config=SimpleNamespace(dtype="float16"))

        vae_calls: list[dict] = []

        class _FakeVAE:
            def decode(
                self,
                latent,
                past_key_values=None,
                use_cache=False,
                stream_state=None,
                last_chunk=False,
            ):
                vae_calls.append(
                    dict(
                        use_cache=use_cache,
                        last_chunk=last_chunk,
                        pkv=past_key_values,
                        shape=tuple(latent.shape),
                    )
                )
                return (
                    torch.ones((1, 8), dtype=torch.float16),
                    ("ss",),
                    "pkv",
                )

        vae = _FakeVAE()
        model._npu310_attach_audio_vae(vae)
        if finished:
            model._pending_postprocess_updates = {"r1": {"stop_reason": "stop_head"}}
        KEY_REQUEST_ID = sys.modules[
            "vllm_omni.model_executor.models.ming3_tts.ming3_tts_llm"
        ].KEY_REQUEST_ID
        latent_rows = torch.randn(2, 4, 64)
        has_patch = torch.tensor([True, False])
        model._npu_test_output = _FakeOmniOutput(
            text_hidden_states=torch.zeros(2, 8),
            multimodal_outputs={
                model.latent_patch_key: latent_rows,
                model.has_patch_key: has_patch,
            },
        )
        out = model.make_omni_output(
            None,
            model_intermediate_buffer=[
                {KEY_REQUEST_ID: "r1"},
                {KEY_REQUEST_ID: "r2"},
            ],
        )
        return model, out, vae_calls

    def test_v1_streaming_decode_payload(self, monkeypatch: pytest.MonkeyPatch):
        # Non-final step: one emitting request (r1) decodes one latent
        # patch incrementally; r2 (no patch row) gets an empty payload.
        model, out, calls = self._run_v1_decode_step(monkeypatch, finished=False)
        assert len(calls) == 1
        assert calls[0]["use_cache"] is True
        assert calls[0]["last_chunk"] is False
        assert calls[0]["shape"] == (1, 4, 64)
        mm = out.multimodal_outputs
        assert mm["model_outputs"][0].shape == (8,)
        assert mm["model_outputs"][0].dtype is torch.float32
        assert mm["model_outputs"][1].numel() == 0
        assert len(mm["sr"]) == 2 and int(mm["sr"][0]) == 24000
        # In-flight state is kept for the next step.
        assert model._npu310_stream_state.get("r1") == ("ss",)
        assert model._npu310_past_kv.get("r1") == "pkv"

    def test_v1_streaming_decode_final_chunk(self, monkeypatch: pytest.MonkeyPatch):
        # Stop decision this step: the decode must flush (last_chunk=True)
        # and the per-request state must be dropped.
        model, out, calls = self._run_v1_decode_step(monkeypatch, finished=True)
        assert len(calls) == 1
        assert calls[0]["last_chunk"] is True
        assert "r1" not in model._npu310_stream_state
        assert "r1" not in model._npu310_past_kv
        # Second decode of the same request re-enters from a clean state.
        model._pending_postprocess_updates = {}
        out2 = model.make_omni_output(
            None,
            model_intermediate_buffer=[
                {
                    sys.modules[
                        "vllm_omni.model_executor.models.ming3_tts.ming3_tts_llm"
                    ].KEY_REQUEST_ID: "r1"
                },
            ],
        )
        assert out2.multimodal_outputs["model_outputs"][0].numel() == 8
        assert calls[-1]["last_chunk"] is False
        assert calls[-1]["shape"] == (1, 4, 64)

    def test_v1_decode_skipped_without_vae(self, monkeypatch: pytest.MonkeyPatch):
        # No AudioVAE attached (not the single-stage path): the payload
        # must pass through untouched.
        patch, _ = _load_model_patch(monkeypatch)
        model = patch._Ming3TTSLLMModel310P(vllm_config=SimpleNamespace(dtype="float16"))
        model._npu_test_output = _FakeOmniOutput(
            text_hidden_states=torch.zeros(2, 8),
            multimodal_outputs={"opaque": 1},
        )
        out = model.make_omni_output(None, model_intermediate_buffer=[{}, {}])
        assert out.multimodal_outputs == {"opaque": 1}

    def test_on_requests_finished_cleans_decode_state(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        patch, _ = _load_model_patch(monkeypatch)
        model = patch._Ming3TTSLLMModel310P(vllm_config=SimpleNamespace(dtype="float16"))
        model._npu310_stream_state["r1"] = ("ss",)
        model._npu310_past_kv["r1"] = "pkv"
        model.on_requests_finished({"r1"})
        assert model._npu310_stream_state == {}
        assert model._npu310_past_kv == {}


class TestApplyMing3TTSKdaOps:
    def test_install_idempotent_and_kill_switched(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("VLLM_OMNI_310P_MING3TTS_PATCH", raising=False)
        patch, fake_llm = _load_model_patch(monkeypatch)
        bm3 = types.ModuleType("vllm.model_executor.models.bailing_moe_v3")
        conv_mod = types.ModuleType("vllm_ascend.ops.causal_conv1d")
        conv_mod.causal_conv1d_fn = lambda *a, **k: None
        conv_mod.causal_conv1d_update = lambda *a, **k: None
        _install_package_chain(
            monkeypatch,
            "vllm", "vllm.model_executor", "vllm.model_executor.models",
            "vllm_ascend", "vllm_ascend.ops",
        )
        for name in (
            "chunk_kda", "fused_recurrent_kda", "fused_recurrent_kda_fwd",
            "fused_kda_gate", "causal_conv1d_fn", "causal_conv1d_update",
        ):
            setattr(bm3, name, lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, "vllm.model_executor.models.bailing_moe_v3", bm3)
        monkeypatch.setitem(sys.modules, "vllm_ascend.ops.causal_conv1d", conv_mod)
        kda_mod = sys.modules["vllm_omni.platforms.npu._310p.patch.ming3_tts_kda"]

        patch.install_kda_ops()
        assert getattr(bm3, "chunk_kda") is kda_mod.chunk_kda_310p
        patch.install_kda_ops()  # idempotent
        assert getattr(bm3, "chunk_kda") is kda_mod.chunk_kda_310p


class TestArchDispatch:
    def test_all_three_arch_keys_route(self, monkeypatch: pytest.MonkeyPatch):
        patch, fake_llm = _load_model_patch(monkeypatch)
        init_mod = _load_source_module(
            "patch_init_under_test",
            _repo_root() / "vllm_omni/platforms/npu/_310p/patch/__init__.py",
        )
        monkeypatch.setitem(
            sys.modules, "vllm_omni.platforms.npu._310p.patch.ming3_tts", patch
        )
        for arch in (
            "Ming3TTSForConditionalGeneration",
            "LingV3TTSForConditionalGeneration",
            "Ming3TTSLLMModel",
        ):
            init_mod.apply_model_patches(SimpleNamespace(model_arch=arch))
            assert fake_llm.Ming3TTSLLMModel is patch._Ming3TTSLLMModel310P

    def test_unknown_arch_is_a_noop(self, monkeypatch: pytest.MonkeyPatch):
        _load_model_patch(monkeypatch)
        init_mod = _load_source_module(
            "patch_init_under_test2",
            _repo_root() / "vllm_omni/platforms/npu/_310p/patch/__init__.py",
        )
        init_mod.apply_model_patches(SimpleNamespace(model_arch="SomeOtherArch"))
        assert "vllm_omni.platforms.npu._310p.patch.ming3_tts" not in sys.modules


# ---------------------------------------------------------------------------
# Deploy overlay
# ---------------------------------------------------------------------------


def _merge_stage_lists(base_stages, overlay_stages):
    """Mirror of stage_config._merge_stage_lists (by stage_id, overlay wins)."""
    by_id = {s["stage_id"]: dict(s) for s in (base_stages or [])}
    for stage in overlay_stages or []:
        sid = stage["stage_id"]
        if sid in by_id:
            merged = dict(by_id[sid])
            for k, v in stage.items():
                if k == "compilation_config" and isinstance(v, dict) and isinstance(
                    merged.get(k), dict
                ):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            by_id[sid] = merged
        else:
            by_id[sid] = dict(stage)
    return list(by_id.values())


class TestDeployOverlay:
    def test_310p_overlay_pins_fp16_eager_single_chip(self):
        root = _repo_root()
        overlay_path = root / "vllm_omni" / "deploy" / "ming3_tts_310p.yaml"
        base_path = root / "vllm_omni" / "deploy" / "ming3_tts.yaml"
        assert overlay_path.is_file(), "missing 310P deploy overlay"
        assert base_path.is_file(), "missing base deploy"

        import yaml

        with overlay_path.open() as f:
            overlay = yaml.safe_load(f)
        with base_path.open() as f:
            base = yaml.safe_load(f)

        assert overlay["base_config"] == "ming3_tts.yaml"
        assert overlay["dtype"] == "float16", "310P is FP16-only"
        assert base["dtype"] == "bfloat16", "base deploy stays BF16 for GPU lines"
        # Local checkpoints ship HF custom config code; the HF-ID recipe
        # stays trust_remote_code=false (base), the local-dir line enables it.
        assert overlay["trust_remote_code"] is True
        # MRv2 is CUDA recipe: the NPU platform block must run the legacy
        # V1 async-chunk data plane (tag tree raises on v2 + npu).
        npu = overlay["platforms"]["npu"]
        assert npu["model_runner"] == "v1"
        base_runner = base.get("model_runner")
        assert base_runner in ("v2", None), "base recipe should stay MRv2"

        # Resolved stage-0 contract: base stages + platform overlay pins
        # (loader semantics: platform stages deep-merge onto base stages).
        base_stage = dict(base["stages"][0])
        for k, v in npu["stages"][0].items():
            if k == "compilation_config" and isinstance(v, dict):
                base_stage["compilation_config"] = {
                    **base_stage.get("compilation_config", {}),
                    **v,
                }
            else:
                base_stage[k] = v
        assert base_stage["devices"] == "0", "single-chip 310P: logical device 0"
        assert base_stage["enforce_eager"] is True
        assert base_stage["max_num_seqs"] == 1
        assert base_stage["gpu_memory_utilization"] == 0.95
        assert base_stage["compilation_config"]["cudagraph_mode"] == "NONE"
        # Inherited runtime contracts survive the overlay.
        assert base_stage["max_model_len"] == 4096
        assert base_stage["max_num_batched_tokens"] == 4096
        assert base_stage["async_scheduling"] is True
        assert base_stage["additional_config"]["ar_prefill_admission_max_wait_ms"] == 150
        assert base["trust_remote_code"] is False
        assert base["pipeline"] == "ming3_tts"
        assert base["async_chunk"] is True
