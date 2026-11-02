# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Covo-Audio-Chat 310P patch surface.

The tests load the patch modules from source with fake token2wav / alias-free
dependencies, so they validate the patch contract (dispatch wiring, math-SDPA
decomposition, F-shim delegation, Kaiser CPU pin, vocoder seeding, idempotency,
kill switch) without loading real model weights or NPU kernels.

Intended to run inside the 310P container (vllm_omni installed).  On a bare
host without vllm_omni, the dispatch/entry-point tests skip.
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


def _load_covo_patch(name: str = "covo_audio_patch_under_test"):
    return _load_source_module(
        name,
        _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "covo_audio.py",
    )


def _load_dispatcher(name: str = "covo_dispatch_under_test"):
    return _load_source_module(
        name,
        _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py",
    )


def _install_fake_alias_free(monkeypatch: pytest.MonkeyPatch):
    """Fake common.alias_free_activation with a recording kaiser function."""

    def fake_kaiser(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
        return torch.ones(1, 1, kernel_size, dtype=torch.float32)

    fake_kaiser._original_marker = True
    pkg_common = types.ModuleType("vllm_omni.model_executor.models.common")
    mod = types.ModuleType("vllm_omni.model_executor.models.common.alias_free_activation")
    mod.kaiser_sinc_filter1d = fake_kaiser
    monkeypatch.setitem(sys.modules, "vllm_omni.model_executor.models.common", pkg_common)
    monkeypatch.setitem(sys.modules, "vllm_omni.model_executor.models.common.alias_free_activation", mod)
    return mod


def _install_fake_token2wav(monkeypatch: pytest.MonkeyPatch):
    """Fake vllm_omni.model_executor.models.covo_audio.token2wav module."""
    calls = {"inference": 0}

    class Token2WavDecoder:
        def __init__(self):
            self.state = {"initial_seed_at_entry": None}

        def inference(self, data, **kwargs):
            self.state["initial_seed_at_entry"] = torch.initial_seed()
            calls["inference"] += 1
            return "audio"

    fake_F = types.SimpleNamespace(silu=torch.nn.functional.silu)
    token2wav = types.ModuleType("vllm_omni.model_executor.models.covo_audio.token2wav")
    token2wav.F = fake_F
    token2wav.Token2WavDecoder = Token2WavDecoder
    pkg = types.ModuleType("vllm_omni.model_executor.models.covo_audio")
    pkg.token2wav = token2wav
    monkeypatch.setitem(sys.modules, "vllm_omni.model_executor.models.covo_audio", pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.model_executor.models.covo_audio.token2wav", token2wav)
    return token2wav, calls


# ---------------------------------------------------------------------------
# math SDPA
# ---------------------------------------------------------------------------


def test_math_sdpa_matches_explicit_reference():
    mod = _load_covo_patch()
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.float32):
        q = torch.randn(2, 4, 37, 16, dtype=dtype)
        k = torch.randn(2, 4, 53, 16, dtype=dtype)
        v = torch.randn(2, 4, 53, 16, dtype=dtype)
        mask = torch.zeros(2, 1, 37, 53, dtype=dtype)
        mask[..., :10] = torch.finfo(dtype).min  # prefix block

        out = mod.math_sdpa(q, k, v, attn_mask=mask)

        scale = 16**-0.5
        ref_scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
        ref_scores = ref_scores + mask.float()
        ref = torch.matmul(torch.softmax(ref_scores, dim=-1).to(dtype), v)
        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), f"dtype={dtype}"
        assert out.dtype == dtype


def test_math_sdpa_chunked_matches_single_path():
    mod = _load_covo_patch()
    torch.manual_seed(1)
    # Exercise the chunked path directly (block size derived from the budget)
    # and verify it reproduces the single-pass math decomposition.
    L = 1200
    q = torch.randn(1, 2, L, 8, dtype=torch.float16)
    k = torch.randn(1, 2, L, 8, dtype=torch.float16)
    v = torch.randn(1, 2, L, 8, dtype=torch.float16)
    mask = torch.zeros(1, 1, L, L, dtype=torch.float16)

    out_chunked = mod._math_sdpa_chunked(q, k, v, mask, 8**-0.5)

    single = mod._math_sdpa_single(q, k, v, mask, 8**-0.5)
    assert torch.allclose(out_chunked, single, atol=1e-2, rtol=1e-2)
    # Multi-row block: force a small block via the budget to cover the loop.
    small = torch.randn(1, 2, 300, 8, dtype=torch.float16)
    kk = torch.randn(1, 2, 300, 8, dtype=torch.float16)
    vv = torch.randn(1, 2, 300, 8, dtype=torch.float16)
    m2 = torch.zeros(1, 1, 300, 300, dtype=torch.float16)
    assert torch.allclose(
        mod._math_sdpa_chunked(small, kk, vv, m2, 8**-0.5),
        mod._math_sdpa_single(small, kk, vv, m2, 8**-0.5),
        atol=1e-2,
        rtol=1e-2,
    )


def test_fshim_delegates_and_redirects_only_sdpa():
    mod = _load_covo_patch()
    shim = mod._FShim(torch.nn.functional)
    assert shim.silu is torch.nn.functional.silu
    assert shim.avg_pool1d is torch.nn.functional.avg_pool1d
    assert shim.scaled_dot_product_attention is mod.math_sdpa
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    out = shim.scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 1, 4, 8)


# ---------------------------------------------------------------------------
# Stage 0: whisper encoder eager pin
# ---------------------------------------------------------------------------


def test_stage0_pins_whisper_encoder_to_eager():
    mod = _load_covo_patch()
    encoder_config = SimpleNamespace(_attn_implementation="sdpa")
    model_config = SimpleNamespace(hf_config=SimpleNamespace(encoder_config=encoder_config))

    mod.apply_stage0_patches(model_config)

    assert encoder_config._attn_implementation == "eager"
    assert mod._STAGE0_PATCHED is True
    # Idempotent second call keeps the pin.
    mod.apply_stage0_patches(model_config)
    assert encoder_config._attn_implementation == "eager"


def test_stage0_noop_without_encoder_config():
    mod = _load_covo_patch(name="covo_stage0_noop")
    mod.apply_stage0_patches(SimpleNamespace(hf_config=SimpleNamespace()))  # no raise
    assert mod._STAGE0_PATCHED is False


# ---------------------------------------------------------------------------
# Stage 1: F shim + kaiser pin + vocoder seed
# ---------------------------------------------------------------------------


def test_stage1_installs_shim_kaiser_pin_and_seed(monkeypatch):
    mod = _load_covo_patch()
    token2wav, calls = _install_fake_token2wav(monkeypatch)
    alias_free = _install_fake_alias_free(monkeypatch)

    mod.apply_stage1_patches(SimpleNamespace())

    # F shim installed once, sdpa redirected, other attributes delegate.
    assert token2wav._omni_f is token2wav.F
    assert token2wav.F.silu is torch.nn.functional.silu
    assert token2wav.F.scaled_dot_product_attention is mod.math_sdpa

    # Kaiser filter math pinned to CPU with the same signature.
    pinned = alias_free.kaiser_sinc_filter1d
    assert getattr(pinned, "_covo_310p_cpu_safe", False) is True
    out = pinned(cutoff=0.25, half_width=0.1, kernel_size=9)
    assert out.device.type == "cpu"
    assert out.shape == (1, 1, 9)
    assert torch.isfinite(out).all()

    # Vocoder seeded: inner call observes initial_seed == 42.
    decoder = token2wav.Token2WavDecoder()
    torch.manual_seed(1234)
    result = decoder.inference({"dummy": 1})
    assert result == "audio"
    assert decoder.state["initial_seed_at_entry"] == 42
    assert calls["inference"] == 1

    # Idempotency: second application must not re-wrap or re-pin.
    mod.apply_stage1_patches(SimpleNamespace())
    decoder2 = token2wav.Token2WavDecoder()
    decoder2.inference({"dummy": 1})
    assert decoder2.state["initial_seed_at_entry"] == 42
    assert calls["inference"] == 2  # inner called once per invocation, no extra wraps


def test_stage1_disabled_by_kill_switch(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_310P_COVO_PATCHES", "0")
    mod = _load_covo_patch()
    token2wav, _ = _install_fake_token2wav(monkeypatch)
    alias_free = _install_fake_alias_free(monkeypatch)

    mod.apply_stage1_patches(SimpleNamespace())

    assert getattr(token2wav, "_omni_f", None) is None
    assert alias_free.kaiser_sinc_filter1d._original_marker is True
    assert mod._STAGE1_PATCHED is False


# ---------------------------------------------------------------------------
# Dispatch wiring + entry-point gating
# ---------------------------------------------------------------------------


def test_dispatch_routes_covo_arches(monkeypatch):
    pytest.importorskip("vllm_omni")
    from vllm_omni.platforms.npu._310p.patch import covo_audio as real_covo

    seen = {}

    def fake_apply(model_config):
        seen["config"] = model_config

    monkeypatch.setattr(real_covo, "apply_covo_audio_patches", fake_apply)
    dispatcher = _load_dispatcher()

    for arch in sorted(dispatcher._COVO_ARCHES):
        cfg = SimpleNamespace(model_arch=arch)
        dispatcher.apply_model_patches(cfg)
        assert seen.get("config") is cfg, arch

    # Non-covo arches must not route to the covo patch.
    seen.clear()
    dispatcher.apply_model_patches(SimpleNamespace(model_arch="SomeOtherArch"))
    assert seen == {}


def test_apply_entry_point_gated_by_is_310p(monkeypatch):
    """apply_covo_audio_patches must be a no-op when is_310p() is False."""
    pytest.importorskip("vllm_omni")
    from vllm_omni.platforms.npu._310p import is_310p

    if is_310p():
        pytest.skip("running on a 310P box; False-gate not exercisable")

    mod = _load_covo_patch(name="covo_gate_test")
    token2wav, _ = _install_fake_token2wav(monkeypatch)
    mod.apply_covo_audio_patches(SimpleNamespace())
    assert getattr(token2wav, "_omni_f", None) is None
    assert mod._STAGE1_PATCHED is False
