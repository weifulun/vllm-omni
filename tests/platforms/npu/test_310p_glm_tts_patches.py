# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the 310P GLM-TTS patch set.

Same approach as ``test_310p_indextts2_patches.py``: the patch module is
loaded from source with fake dependencies (``vllm.logger``, the GLM-TTS model
modules, the ``is_310p`` probe), so the wiring (registry dispatch, math-SDPA
numerics, CPU pins, NaN guard, idempotency, kill switch, deploy overlay) is
validated without an NPU or a model checkpoint.

Real-device behavior (NPU SDPA dead zone, HiFT istft, WhisperVQ VQ
numerics, TBE warmup cost) is qualified in the 310P real-device phase of
task-39.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _BaseDiTAttention(nn.Module):
    """Minimal DiTAttention stand-in with the real projection layout.

    Base only — ``_make_fake_stack`` derives a fresh subclass per test so
    class-level patch markers never leak between tests.
    """

    def __init__(self, dim: int = 16, heads: int = 2, dim_head: int = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.scale = 1.0 / (dim_head**0.5)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x, padding_mask=None, rope=None, attn_mask=None):
        # 910B-line shape: [B, T, H, D] projections then attention core.
        query = self.to_q(x).view(*x.shape[:2], self.heads, self.dim_head)
        key = self.to_k(x).view(*x.shape[:2], self.heads, self.dim_head)
        value = self.to_v(x).view(*x.shape[:2], self.heads, self.dim_head)
        out = torch.nn.functional.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)
        out = out.view(*x.shape[:2], self.inner_dim)
        return self.to_out(out.to(query.dtype))


class _BaseDiTWrapper(nn.Module):
    """Minimal GLMTTSDiTForGeneration stand-in."""

    def __init__(self):
        super().__init__()
        self.model_dir = "/fake"
        self.sample_rate = 24000
        self._vocoder = "SENTINEL"
        self._vocoder_sample_rate = 24000
        self._hift_weights_path = "/fake/hift/hift.pt"
        self._vocos_jit_path = "/fake/vocos2d/generator_jit.ckpt"

    def _ensure_vocoder_loaded(self):
        if self._vocoder is not None:
            return
        device = next(self.parameters()).device if len(self.parameters()) else torch.device("meta")
        self._vocoder = load_vocoder(self.model_dir, device, self.sample_rate)


class _BaseGLMTTS(nn.Module):
    model_stage = "glm_tts"

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return hidden_states + 1.0


def _make_fake_stack(monkeypatch: pytest.MonkeyPatch, is_310p_value: bool):
    """Install the fake vllm_omni.model_executor.models.glm_tts package tree."""
    for parent in (
        "vllm",
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
    ):
        if parent not in sys.modules:
            _install_fake_module(monkeypatch, parent)

    # Fresh subclasses per test: the patch set stamps class-level markers
    # (_310p_math_sdpa / _310p_vocoder_cpu_pinned / _310p_nan_guarded), which
    # must not survive across tests.
    dit_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.glm_tts.glm_tts_dit",
        DiTAttention=type("DiTAttention", (_BaseDiTAttention,), {}),
        apply_rotary_pos_emb=lambda *a, **k: None,
    )

    calls = {"load_vocoder": []}

    def load_vocoder(model_root, device, sample_rate=24000):
        calls["load_vocoder"].append((model_root, device, sample_rate))
        return ("VOCODER", sample_rate)

    wrapper_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.glm_tts.glm_tts_dit_wrapper",
        GLMTTSDiTForGeneration=type("GLMTTSDiTForGeneration", (_BaseDiTWrapper,), {}),
        load_vocoder=load_vocoder,
    )
    wrapper_mod._calls = calls

    vc_calls = {"devices": []}

    def load_voice_clone_frontend(
        model_root,
        model_device,
        *,
        speech_tokenizer_cache=None,
        campplus_cache=None,
        campplus_path=None,
    ):
        vc_calls["devices"].append(model_device)
        model = nn.Linear(4, 4, dtype=torch.float16)  # simulate fp16 checkpoint
        return (model, "FE"), "CAMPPLUS"

    voice_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.glm_tts.voice_clone",
        load_voice_clone_frontend=load_voice_clone_frontend,
    )
    voice_mod._calls = vc_calls

    model_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.glm_tts.glm_tts",
        GLMTTSForConditionalGeneration=type(
            "GLMTTSForConditionalGeneration", (_BaseGLMTTS,), {}
        ),
    )

    pkg = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.glm_tts",
        glm_tts_dit=dit_mod,
        glm_tts_dit_wrapper=wrapper_mod,
        voice_clone=voice_mod,
        glm_tts=model_mod,
    )

    _310p_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p",
        is_310p=lambda *a, **k: is_310p_value,
    )
    # Real __path__ so submodule imports (patch dispatch) resolve against the
    # repo sources.
    _310p_mod.__path__ = [str(_repo_root() / "vllm_omni/platforms/npu/_310p")]

    import vllm

    _install_fake_module(monkeypatch, "vllm.logger", init_logger=lambda name: _QuietLogger())
    vllm.logger = sys.modules["vllm.logger"]

    return pkg


class _QuietLogger:
    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def error(self, *a, **k):
        return None


@pytest.fixture
def glmtts_patch(monkeypatch):
    root = _repo_root()
    monkeypatch.chdir(root)
    return monkeypatch


def _fresh_module(monkeypatch, name: str, path: Path):
    monkeypatch.delitem(sys.modules, name, raising=False)
    return _load_source_module(name, path)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_math_sdpa_matches_reference(monkeypatch):
    root = _repo_root()
    monkeypatch.chdir(root)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src", PATCH_PACKAGE_MARKER / "glm_tts.py")

    torch.manual_seed(0)
    B, T, H, D = 2, 17, 3, 8
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    scale = 1.0 / (D**0.5)

    # no-mask reference (fp64)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double().permute(0, 2, 1, 3),
        k.double().permute(0, 2, 1, 3),
        v.double().permute(0, 2, 1, 3),
        scale=scale,
    ).permute(0, 2, 1, 3)
    got = mod._math_sdpa(q, k, v, scale, None)
    assert got.dtype == q.dtype
    assert (got.double() - ref).abs().max().item() < 2e-5

    # additive float mask reference
    mask = torch.randn(1, 1, T, T)
    ref_m = torch.nn.functional.scaled_dot_product_attention(
        q.double().permute(0, 2, 1, 3),
        k.double().permute(0, 2, 1, 3),
        v.double().permute(0, 2, 1, 3),
        attn_mask=mask.double(),
        scale=scale,
    ).permute(0, 2, 1, 3)
    got_m = mod._math_sdpa(q, k, v, scale, mask)
    assert (got_m.double() - ref_m).abs().max().item() < 2e-5

    # -inf block mask must produce finite output (no NaN leak)
    neginf = torch.full((1, 1, T, T), float("-inf"))
    neginf[:, :, :, 0] = 0.0
    got_i = mod._math_sdpa(q, k, v, scale, neginf)
    assert torch.isfinite(got_i).all()


def test_dit_attention_patch_installs_math_forward(monkeypatch):
    root = _repo_root()
    monkeypatch.chdir(root)
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "1")
    _make_fake_stack(monkeypatch, is_310p_value=True)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src2", PATCH_PACKAGE_MARKER / "glm_tts.py")
    mod.apply_glm_tts_310p_patches()

    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit as dit_mod

    assert getattr(dit_mod.DiTAttention, "_310p_math_sdpa", False) is True

    attn = dit_mod.DiTAttention(dim=16, heads=2, dim_head=8).eval()
    x = torch.randn(1, 11, 16)
    with torch.no_grad():
        out = attn(x)
    # reference: explicit math with the same projections
    q = attn.to_q(x).view(1, 11, 2, 8)
    k = attn.to_k(x).view(1, 11, 2, 8)
    v = attn.to_v(x).view(1, 11, 2, 8)
    ref = mod._math_sdpa(q, k, v, attn.scale, None).reshape(1, 11, 16)
    ref = attn.to_out(ref)
    assert (out - ref).abs().max().item() < 1e-6


def test_apply_is_noop_off_310p(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "1")
    _make_fake_stack(monkeypatch, is_310p_value=False)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src3", PATCH_PACKAGE_MARKER / "glm_tts.py")
    mod.apply_glm_tts_310p_patches()
    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit as dit_mod

    assert not getattr(dit_mod.DiTAttention, "_310p_math_sdpa", False)
    assert mod._PATCHED is False  # not latched when off-310P


def test_kill_switch_skips_install(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "0")
    _make_fake_stack(monkeypatch, is_310p_value=True)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src4", PATCH_PACKAGE_MARKER / "glm_tts.py")
    mod.apply_glm_tts_310p_patches()
    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit as dit_mod

    assert not getattr(dit_mod.DiTAttention, "_310p_math_sdpa", False)
    assert mod._PATCHED is True  # latched so we don't retry per worker


def test_idempotent_double_apply(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "1")
    _make_fake_stack(monkeypatch, is_310p_value=True)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src5", PATCH_PACKAGE_MARKER / "glm_tts.py")
    mod.apply_glm_tts_310p_patches()
    first_forward = mod._patch_dit_attention.__globals__  # sanity: module loaded
    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit as dit_mod

    fwd_before = dit_mod.DiTAttention.forward
    mod.apply_glm_tts_310p_patches()
    assert dit_mod.DiTAttention.forward is fwd_before  # no double wrap


def test_vocoder_cpu_pin(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "1")
    _make_fake_stack(monkeypatch, is_310p_value=True)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src6", PATCH_PACKAGE_MARKER / "glm_tts.py")
    mod.apply_glm_tts_310p_patches()

    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit_wrapper as wrp_mod

    instance = wrp_mod.GLMTTSDiTForGeneration()
    instance._vocoder = None
    instance._ensure_vocoder_loaded()
    calls = wrp_mod._calls["load_vocoder"]
    assert len(calls) == 1
    root, device, sr = calls[0]
    assert device == torch.device("cpu")
    assert sr == 24000


def test_speech_tokenizer_cpu_fp32_pin(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "1")
    _make_fake_stack(monkeypatch, is_310p_value=True)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src7", PATCH_PACKAGE_MARKER / "glm_tts.py")
    mod.apply_glm_tts_310p_patches()

    from vllm_omni.model_executor.models.glm_tts import voice_clone as vc_mod

    # "meta" stands in for the NPU device type on machines without torch_npu:
    # the contract is that whatever device is handed in gets forced to CPU.
    st, cp = vc_mod.load_voice_clone_frontend("/fake", torch.device("meta"))
    assert cp == "CAMPPLUS"
    model, fe = st
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # original was called with the CPU device
    assert vc_mod._calls["devices"] == [torch.device("cpu")]


def test_logits_nan_guard_ar_only(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "1")
    _make_fake_stack(monkeypatch, is_310p_value=True)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src8", PATCH_PACKAGE_MARKER / "glm_tts.py")
    mod.apply_glm_tts_310p_patches()

    from vllm_omni.model_executor.models.glm_tts import glm_tts as m_mod

    cls = m_mod.GLMTTSForConditionalGeneration
    inst = cls()
    inst.model_stage = "glm_tts"
    x = torch.tensor([1.0, float("inf"), float("nan"), -float("inf")])
    out = inst.compute_logits(x, None)
    assert torch.isfinite(out).all()

    inst.model_stage = "glm_tts_dit"  # DiT stage passes through unguarded
    out2 = inst.compute_logits(x, None)
    assert torch.isinf(out2[1]) and torch.isnan(out2[2])


def test_registry_dispatch_for_glm_tts_arch(monkeypatch):
    """apply_model_patches must route the GLM-TTS arch to the patch set."""
    monkeypatch.setenv("VLLM_OMNI_310P_GLM_TTS_PATCHES", "1")
    _make_fake_stack(monkeypatch, is_310p_value=True)
    mod = _fresh_module(monkeypatch, "glm_tts_310p_patch_src9", PATCH_PACKAGE_MARKER / "glm_tts.py")

    # Load the real dispatch __init__ from source with the fake glm_tts module
    # importable under its package name.
    pkg_init = PATCH_PACKAGE_MARKER / "__init__.py"
    dispatch = _load_source_module("_310p_patch_pkg_src", pkg_init)
    cfg = types.SimpleNamespace(model_arch="GLMTTSForConditionalGeneration")
    dispatch.apply_model_patches(cfg)
    from vllm_omni.model_executor.models.glm_tts import glm_tts_dit as dit_mod

    assert getattr(dit_mod.DiTAttention, "_310p_math_sdpa", False) is True


def test_deploy_overlay_shape():
    import yaml

    root = _repo_root()
    path = root / "vllm_omni" / "deploy" / "glm_tts_310p.yaml"
    assert path.is_file(), "glm_tts_310p.yaml overlay missing"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["base_config"] == "glm_tts.yaml"
    by_id = {s["stage_id"]: s for s in cfg["stages"]}
    assert by_id[0]["dtype"] == "float16"
    assert by_id[0]["enforce_eager"] is True
    assert by_id[1]["dtype"] == "float32"
    assert by_id[1]["enforce_eager"] is True
    assert by_id[1]["hf_overrides"]["use_dit_cuda_graphs"] is False
    # base untouched: seed 42 default survives through the base config
    base = root / "vllm_omni" / "deploy" / "glm_tts.yaml"
    with open(base) as f:
        bcfg = yaml.safe_load(f)
    s0 = next(s for s in bcfg["stages"] if s["stage_id"] == 0)
    assert s0["default_sampling_params"]["seed"] == 42
    assert s0["dtype"] == "bfloat16"  # 910B-line default the overlay replaces
