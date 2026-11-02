# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-runnable contract tests for the 310P Qwen-Image-2.1 patch (task-16).

Skeleton status: validates the *patch contract* (dispatch gating, dtype
enforcement, RoPE fallback policy, NZ-cast disable, VAE memory policy) with
fake modules, mirroring ``test_310p_patches.py``.  No NPU is required; on a
310P box the real-device smoke test (see risk list section 3) extends this
file with a ``@hardware_test``-style marker once the branch lands.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Source loading with fakes (same convention as test_310p_patches.py)
# ---------------------------------------------------------------------------


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


def _native_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Reference real-valued RoPE used as the native fallback oracle."""
    cos = torch.real(freqs).float().unsqueeze(1)
    sin = torch.imag(freqs).float().unsqueeze(1)
    pairs = x.float().unflatten(-1, (-1, 2))
    x1, x2 = pairs.unbind(-1)
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).to(x.dtype)



class FakeQwenImage21Rope(torch.nn.Module):
    """Minimal stand-in mirroring the stock complex-table semantics."""

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        with torch.device("cpu"):
            pos_index = torch.arange(64)
            neg_index = torch.arange(8).flip(0) * -1 - 1
            self.freqs = [
                torch.cat([self.rope_params(pos_index, dim), self.rope_params(neg_index, dim)], dim=0)
                for dim in axes_dim
            ]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, img_shapes, image_pad_mask, device):
        # Stock behaviour: move the complex tables to the operand device.
        self.freqs = [freq.to(device) for freq in self.freqs]
        return self.freqs[0][torch.zeros(1, dtype=torch.long)]


def _load_qwen_image_2_1_patch(monkeypatch: pytest.MonkeyPatch, is_310p_value: bool = True):
    """Load the patch source with a controllable ``is_310p`` and a fake
    transformer module (so nothing real is imported from the model tree)."""
    rope_calls: list[str] = []

    def npu_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rope_calls.append("npu")
        raise RuntimeError("simulated: 310P interleave mode unsupported")

    class FakeQwenImage21Transformer2DModel(torch.nn.Module):
        @staticmethod
        def _nz_weights_supported_for(od_config, quant_config):  # noqa: ANN001
            return True

    fake_transformer = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer",
        _apply_qwen_image21_rotary_emb_npu=npu_rope,
        _apply_qwen_image21_rotary_emb_native=_native_rope,
        QwenImage21Transformer2DModel=FakeQwenImage21Transformer2DModel,
        QwenImage21Rope=FakeQwenImage21Rope,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.qwen_image_21",
        qwen_image_21_transformer=fake_transformer,
    )
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models")

    # Fake pipeline + loader modules for the placement/CPU-build hooks that
    # ``apply_qwen_image_2_1_310p_patches`` imports at runtime.
    pipeline_mod = types.ModuleType("vllm_omni_fake_pipeline_qwen_image_21")

    def _fpwp_passthrough(factory, *args, **kwargs):
        return factory(*args, **kwargs)

    class _FakeEnc:
        pass

    class _FakeProc:
        pass

    class _FakePipeline:
        def __init__(self):
            self.transformer = SimpleNamespace(device=torch.device("cpu"))

        def encode_prompt(self, *args, **kwargs):
            return (
                torch.zeros(2, 3),
                torch.ones(2, 3, dtype=torch.long),
                torch.zeros(2, 3, dtype=torch.bool),
            )

    pipeline_mod.from_pretrained_with_prefetch = _fpwp_passthrough
    pipeline_mod.Qwen3VLForConditionalGeneration = _FakeEnc
    pipeline_mod.Qwen3VLProcessor = _FakeProc
    pipeline_mod.QwenImage21Pipeline = _FakePipeline
    qwen_pkg = types.ModuleType("vllm_omni.diffusion.models.qwen_image_21")
    qwen_pkg.qwen_image_21_transformer = fake_transformer
    qwen_pkg.pipeline_qwen_image_21 = pipeline_mod
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.models.qwen_image_21", qwen_pkg)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21",
        pipeline_mod,
    )
    setattr(sys.modules["vllm_omni.diffusion.models"], "qwen_image_21",
            sys.modules["vllm_omni.diffusion.models.qwen_image_21"])
    class _FakeLoadedPipeline:
        def __init__(self):
            self.transformer = torch.nn.Linear(4, 4)
            self.vae = torch.nn.Linear(4, 4)

    class _FakeDiffusersLoader:
        def __init__(self):
            self.loaded = _FakeLoadedPipeline()

        def load_model(self, model, *args, **kwargs):
            return self.loaded

    loader_mod = _install_fake_module(
        monkeypatch, "vllm_omni.diffusion.model_loader.diffusers_loader",
        initialize_model=lambda od_config: od_config,
        DiffusersPipelineLoader=_FakeDiffusersLoader,
    )
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.model_loader",
                        types.ModuleType("vllm_omni_fake_model_loader_pkg"))
    setattr(sys.modules["vllm_omni.diffusion"], "model_loader",
            sys.modules["vllm_omni.diffusion.model_loader"])

    # Fake TORCH_SDPA backend module for the manual-SDPA swap hook.
    def _orig_sdpa_sentinel(*args, **kwargs):
        raise AssertionError("original _sdpa_attention must be rebound on 310P")

    sdpa_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.attention.backends.sdpa",
        _sdpa_attention=_orig_sdpa_sentinel,
    )
    attention_pkg = types.ModuleType("vllm_omni.diffusion.attention")
    backends_pkg = types.ModuleType("vllm_omni.diffusion.attention.backends")
    backends_pkg.sdpa = sdpa_mod
    attention_pkg.backends = backends_pkg
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.attention", attention_pkg)
    monkeypatch.setitem(
        sys.modules, "vllm_omni.diffusion.attention.backends", backends_pkg
    )
    setattr(sys.modules["vllm_omni.diffusion"], "attention", attention_pkg)
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p", is_310p=lambda: is_310p_value)

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "qwen_image_2_1.py"
    module = _load_source_module("vllm_omni_test_310p_qwen_image_2_1_patch", path)
    return module, fake_transformer, rope_calls


def _make_od_config(model_class_name: str = "QwenImage21Pipeline", **overrides) -> SimpleNamespace:
    defaults = dict(
        model_class_name=model_class_name,
        dtype=torch.bfloat16,
        enforce_eager=False,
        vae_use_tiling=False,
        vae_use_slicing=False,
        quantization_config=None,
        extras={},
        diffusion_attention_backend=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_dispatch_is_gated_on_310p_and_model_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)

    module._APPLIED = False
    module.maybe_apply_diffusion_310p_patches(_make_od_config())
    assert module._APPLIED is True

    # Non-Qwen-Image-2.1 diffusion models are untouched by this patch file.
    other = _make_od_config(model_class_name="StableDiffusionXLPipeline")
    module.maybe_apply_diffusion_310p_patches(other)
    assert other.dtype is torch.bfloat16  # untouched


def test_dispatch_is_noop_off_310p(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch, is_310p_value=False)
    config = _make_od_config()
    module.maybe_apply_diffusion_310p_patches(config)
    assert module._APPLIED is False
    assert config.dtype is torch.bfloat16


def test_runtime_dtype_forced_fp16_and_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)

    assert module.resolve_runtime_dtype(torch.bfloat16) is torch.float16
    assert module.resolve_runtime_dtype(torch.float16) is torch.float16

    module._APPLIED = False
    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    # Re-run must be idempotent (single-shot guard).
    assert module._APPLIED is True


def test_apply_enforces_runtime_configuration(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    module._APPLIED = False
    config = _make_od_config(extras={"prefix_kv_cache_dtype": "fp8"}, quantization_config=object())
    with caplog.at_level("WARNING"):
        module.apply_qwen_image_2_1_310p_patches(config)

    assert config.dtype is torch.float16
    assert config.enforce_eager is True
    assert config.vae_use_tiling is True
    assert config.vae_use_slicing is True
    # 310P has no MindIE diffusion quant path: extras and quant config dropped.
    assert config.extras["prefix_kv_cache_dtype"] is None
    assert config.quantization_config is None
    assert any("prefix_kv_cache_dtype" in record.message for record in caplog.records)


def test_attention_default_is_sdpa_and_env_experiment_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    monkeypatch.setenv("QWEN21_310P_ATTENTION", "flash_attn")
    module._APPLIED = False
    config = _make_od_config()
    module.apply_qwen_image_2_1_310p_patches(config)
    assert config.diffusion_attention_backend == "FLASH_ATTN"

    monkeypatch.setenv("QWEN21_310P_ATTENTION", "bogus")
    module._APPLIED = False
    config2 = _make_od_config()
    module.apply_qwen_image_2_1_310p_patches(config2)
    assert config2.diffusion_attention_backend is None  # stays on the resolver default


@pytest.mark.parametrize("policy", ["auto", "npu", "native", "bogus"])
def test_rope_policy_resolution(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    if policy == "bogus":
        assert module.resolve_rope_policy({"QWEN21_310P_ROPE": policy}) == "auto"
        return
    monkeypatch.setenv("QWEN21_310P_ROPE", policy)
    assert module.resolve_rope_policy() == policy
    monkeypatch.delenv("QWEN21_310P_ROPE")


def test_rope_dispatcher_cpu_extracted_tables_and_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """310P AICore has no complex-dtype compute: the dispatcher must extract
    real cos/sin tables on CPU (never run ``real/imag`` on the operand
    device) and do the rotate itself.  auto: one ``npu_rotary_mul`` attempt,
    sticky fallback to the algebra on failure; the stock complex-on-device
    implementations must never be invoked."""
    import torch_npu

    module, fake_transformer, rope_calls = _load_qwen_image_2_1_patch(monkeypatch)

    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, dtype=torch.float16)
    freqs = torch.polar(torch.ones(8, 32), torch.randn(8, 32))
    expected = _native_rope(x, freqs)

    npu_attempts = {"n": 0}

    def _boom(*args: object, **kwargs: object) -> None:
        npu_attempts["n"] += 1
        raise RuntimeError("interleave unsupported (simulated)")

    monkeypatch.setattr(torch_npu, "npu_rotary_mul", _boom)

    # auto: first call attempts npu_rotary_mul, then the algebra (sticky).
    dispatch = module.build_rope_dispatcher(
        fake_transformer._apply_qwen_image21_rotary_emb_npu, _native_rope, "auto"
    )
    with caplog.at_level("WARNING"):
        first = dispatch(x, freqs)
    assert npu_attempts["n"] == 1  # exactly one NPU op attempt
    assert rope_calls == []  # stock complex-on-device impl never invoked
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert "algebraic" in " ".join(record.message for record in caplog.records)

    # Subsequent calls must not retry the NPU op (sticky fallback).
    second = dispatch(x, freqs)
    assert npu_attempts["n"] == 1
    assert rope_calls == []
    torch.testing.assert_close(second, expected, rtol=0, atol=0)

    # native policy: the NPU op is never attempted.
    native = module.build_rope_dispatcher(
        fake_transformer._apply_qwen_image21_rotary_emb_npu, _native_rope, "native"
    )
    out = native(x, freqs)
    assert npu_attempts["n"] == 1
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_patch_rebinds_transformer_rope_and_disables_nz(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fake_transformer, _ = _load_qwen_image_2_1_patch(monkeypatch)
    monkeypatch.setenv("QWEN21_310P_ROPE", "native")

    module.apply_qwen_image_2_1_310p_patches(_make_od_config())

    # RoPE rebound to the pure-torch native implementation.
    assert fake_transformer._apply_qwen_image21_rotary_emb_npu is _native_rope
    # NZ weight cast is disabled for 310P, enabled elsewhere (fake base returns True).
    model_cls = fake_transformer.QwenImage21Transformer2DModel
    assert model_cls._nz_weights_supported_for(object(), None) is False
    # Idempotent: a second application must not re-patch.
    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    assert fake_transformer._310p_patch_applied is True


def test_nz_gate_passes_through_off_310p(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-310P NPU boxes the original NZ eligibility is preserved."""
    module, fake_transformer, _ = _load_qwen_image_2_1_patch(monkeypatch, is_310p_value=False)
    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    # _APPLIED guard: apply_* unconditionally applies dtype/eager on this rank;
    # the contract under test is the NZ gate delegation.
    model_cls = fake_transformer.QwenImage21Transformer2DModel
    assert model_cls._nz_weights_supported_for(object(), None) is True


# ---------------------------------------------------------------------------
# Encoder placement / CPU-build contract (machine findings, 2026-11-02)
# ---------------------------------------------------------------------------


def test_encoder_placement_resolver_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    assert module.resolve_encoder_placement() == "cpu"


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(2, 2))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN002, ANN003
        return cls()


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN002, ANN003
        return object()


def _make_fake_pipeline_module(monkeypatch: pytest.MonkeyPatch, calls: list):
    pipeline_mod = types.ModuleType("fake_pipeline_qwen_image_21")

    def from_pretrained_with_prefetch(factory, *args, **kwargs):
        calls.append((factory, torch.tensor(1).device.type))
        return factory(*args, **kwargs)

    pipeline_mod.from_pretrained_with_prefetch = from_pretrained_with_prefetch
    pipeline_mod.Qwen3VLForConditionalGeneration = _FakeEncoder
    pipeline_mod.Qwen3VLProcessor = _FakeProcessor
    monkeypatch.setitem(sys.modules, "fake_pipeline_qwen_image_21", pipeline_mod)
    return pipeline_mod


def test_pipeline_placement_pins_encoder_processor_and_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    calls: list = []
    pipeline_mod = _make_fake_pipeline_module(monkeypatch, calls)
    module._patch_pipeline_encoder_placement(pipeline_mod)
    assert pipeline_mod._310p_encoder_patched is True

    # Encoder factory: run under the CPU default-device context (even though
    # the ambient default here is meta, so the redirect is observable) and
    # pin the returned model to CPU.
    with torch.device("meta"):
        model = pipeline_mod.from_pretrained_with_prefetch(_FakeEncoder.from_pretrained, "model")
    assert calls[-1][1] == "cpu"
    assert model._310p_cpu_pinned is True
    model.to("npu:0")  # must be redirected to CPU
    assert model.w.device.type == "cpu"

    # Processor factory: wrapped in the CPU feature proxy.
    proc = pipeline_mod.from_pretrained_with_prefetch(_FakeProcessor.from_pretrained, "model")
    assert isinstance(proc, module._CPUProcessorProxy)

    # Other factories (VAE, scheduler) pass through untouched.
    sentinel = object()

    def other_factory(*args, **kwargs):
        return sentinel

    out = pipeline_mod.from_pretrained_with_prefetch(other_factory, "model")
    assert out is sentinel

    # Idempotent.
    module._patch_pipeline_encoder_placement(pipeline_mod)


def test_encoder_keeps_native_dtype_with_shadow_and_vae_casts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encoder: CPU-friendly native weights untouched, ``.dtype`` shadowed
    to the 310P runtime dtype so the pipeline casts ``prompt_embeds`` at the
    DiT boundary.  VAE: BF16 checkpoint weights cast to the runtime dtype
    before NPU placement (310P has no BF16 compute path)."""
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    calls: list = []
    pipeline_mod = _make_fake_pipeline_module(monkeypatch, calls)

    class _FakeVAEBf16(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(2, 2).to(torch.bfloat16))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # noqa: ANN002, ANN003
            return cls()

    pipeline_mod.DistributedAutoencoderKLQwenImage21 = _FakeVAEBf16
    module._patch_pipeline_encoder_placement(pipeline_mod)

    # Encoder: native (CPU) parameter dtype must survive; .dtype reports the
    # 310P runtime dtype (the pipeline reads it for prompt_embeds).
    with torch.device("meta"):
        enc = pipeline_mod.from_pretrained_with_prefetch(
            _FakeEncoder.from_pretrained, "model"
        )
    assert enc.w.dtype == torch.float32
    assert enc.dtype == torch.float16

    # VAE: BF16 weights are cast to the runtime dtype on CPU.
    vae = pipeline_mod.from_pretrained_with_prefetch(
        _FakeVAEBf16.from_pretrained, "model"
    )
    assert vae.w.dtype == torch.float16


def test_cpu_feature_proxy_redirects_npu_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    seen: dict = {}

    class FakeFeature:
        def to(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return self

    feat = FakeFeature()
    proxy = module._CPUFeatureProxy(feat)
    proxy.to("npu:0")
    assert seen["args"] == ("cpu",)
    proxy.to(torch.device("npu", 0))
    assert seen["args"] == ("cpu",)
    proxy.to("cpu")
    assert seen["args"] == ("cpu",)
    proxy.to(dtype=torch.float32)
    assert seen["kwargs"] == {"dtype": torch.float32}


def test_loader_cpu_build_patch_wraps_initialize_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)

    env: dict = {}

    def _probe_init(od_config):
        # Probe the ambient *default* device with a device-less tensor.
        env["device"] = torch.tensor(1).device.type
        return ("model", od_config)

    loader_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.model_loader.diffusers_loader",
        initialize_model=_probe_init,
    )
    module._patch_loader_cpu_build()
    assert loader_mod._310p_cpu_build_patched is True

    with torch.device("meta"):
        model, od_config = loader_mod.initialize_model("od")
    assert model == "model"
    assert od_config == "od"
    # The construction ran under the CPU default-device context.
    assert env["device"] == "cpu"

    # Idempotent: the second call must not wrap the already-wrapped entry.
    module._patch_loader_cpu_build()
    loader_mod.initialize_model("od")
    assert env["device"] == "cpu"


def test_manual_sdpa_swap_rebinds_and_is_equivalent(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    sdpa_mod = sys.modules["vllm_omni.diffusion.attention.backends.sdpa"]
    original = sdpa_mod._sdpa_attention

    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    swapped = sdpa_mod._sdpa_attention
    assert swapped is not original
    assert swapped is module._manual_sdpa_forward
    assert sdpa_mod._310p_manual_sdpa_patched is True

    # Idempotent: re-applying must not re-bind over the swapped entry.
    module._patch_attention_manual_sdpa()
    assert sdpa_mod._sdpa_attention is swapped

    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    v = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    scale = 0.25

    # No mask, no causal.
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    assert torch.allclose(swapped(q, k, v, None, False, scale), ref, atol=1e-5)

    # Bool keep-mask (True = keep, torch SDPA semantics).
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[0, 1:] = False
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale
    )
    assert torch.allclose(swapped(q, k, v, mask, False, scale), ref, atol=1e-5)

    # Float additive bias.
    bias = torch.randn(1, 2, 4, 4, dtype=torch.float32) * 0.1
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, scale=scale
    )
    assert torch.allclose(swapped(q, k, v, bias, False, scale), ref, atol=1e-5)

    # Causal.
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=scale
    )
    assert torch.allclose(swapped(q, k, v, None, True, scale), ref, atol=1e-5)

    # GQA parity: 4 query heads against 2 KV heads.
    qg = torch.randn(1, 4, 4, 8, dtype=torch.float32)
    kg = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    vg = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(
        qg,
        kg.repeat_interleave(2, dim=1),
        vg.repeat_interleave(2, dim=1),
        scale=scale,
    )
    assert torch.allclose(swapped(qg, kg, vg, None, False, scale), ref, atol=1e-5)


def test_manual_sdpa_chunked_path_matches_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    swapped = sys.modules["vllm_omni.diffusion.attention.backends.sdpa"]._sdpa_attention

    # Force the chunked (online-softmax) path regardless of shape.
    monkeypatch.setattr(module, "_SDPA_DENSE_LIMIT_BYTES", 0)
    monkeypatch.setattr(module, "_SDPA_CHUNK_SIZE", 4)

    torch.manual_seed(1)
    q = torch.randn(1, 2, 9, 6, dtype=torch.float32)
    k = torch.randn(1, 2, 9, 6, dtype=torch.float32)
    v = torch.randn(1, 2, 9, 6, dtype=torch.float32)
    scale = 0.5

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    assert torch.allclose(swapped(q, k, v, None, False, scale), ref, atol=1e-4)

    # Causal (chunked tril offset across chunk boundaries).
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=scale
    )
    assert torch.allclose(swapped(q, k, v, None, True, scale), ref, atol=1e-4)

    # Float additive bias.
    bias = torch.randn(1, 2, 9, 9, dtype=torch.float32) * 0.1
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, scale=scale
    )
    assert torch.allclose(swapped(q, k, v, bias, False, scale), ref, atol=1e-4)

    # Bool keep-mask.
    mask = torch.ones(9, 9, dtype=torch.bool)
    mask[:, 4:] = False
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale
    )
    assert torch.allclose(swapped(q, k, v, mask, False, scale), ref, atol=1e-4)

    # GQA inside the chunked path.
    qg = torch.randn(1, 4, 9, 6, dtype=torch.float32)
    kg = torch.randn(1, 2, 9, 6, dtype=torch.float32)
    vg = torch.randn(1, 2, 9, 6, dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(
        qg,
        kg.repeat_interleave(2, dim=1),
        vg.repeat_interleave(2, dim=1),
        scale=scale,
    )
    assert torch.allclose(swapped(qg, kg, vg, None, False, scale), ref, atol=1e-4)




def test_rope_forward_keeps_complex_tables_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 310P forward wrapper must never move the complex table.

    Runs the stock token-index arithmetic on the CPU and returns the
    selected complex rows on the CPU, so the NPU never sees a complex
    tensor (no transfer, no complex compute -- both unsupported on the
    310P AICore).  The operand device in the signature is irrelevant.
    """
    module, fake_transformer, _ = _load_qwen_image_2_1_patch(monkeypatch)
    od_config = _make_od_config()
    module.apply_qwen_image_2_1_310p_patches(od_config)

    rope_cls = fake_transformer.QwenImage21Rope
    assert getattr(rope_cls, "_310p_forward_patched", False) is True
    rope = rope_cls(theta=10000, axes_dim=[4, 2, 2])
    assert all(f.device.type == "cpu" and f.is_complex() for f in rope.freqs)

    # seq: 3 text tokens + 2x2 image tokens
    mask = torch.tensor([False, False, False, True, True, True, True])
    out = rope([(1, 2, 2)], mask, device=torch.device("npu"))

    # Contract 1: the complex table never leaves the host.
    assert out.device.type == "cpu" and out.is_complex()
    assert all(f.device.type == "cpu" for f in rope.freqs)

    # Contract 2: values match the stock index arithmetic (CPU reference).
    fi = torch.tensor([0, 1, 2, 3, 3, 3, 3], dtype=torch.long)
    hi, wi = fi.clone(), fi.clone()
    hi[3:] = torch.tensor([-1, -1, 0, 0])
    wi[3:] = torch.tensor([-1, 0, -1, 0])
    hi = torch.where(hi < 0, hi + rope.freqs[1].shape[0], hi)
    wi = torch.where(wi < 0, wi + rope.freqs[2].shape[0], wi)
    expected = torch.cat(
        [rope.freqs[0].index_select(0, fi), rope.freqs[1].index_select(0, hi),
         rope.freqs[2].index_select(0, wi)],
        dim=-1,
    )
    assert out.shape == (7, sum(d // 2 for d in rope.axes_dim))
    torch.testing.assert_close(out, expected)


def test_encode_prompt_moves_outputs_to_dit_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boundary wrapper relocates encode_prompt outputs onto the DiT.

    The CPU-pinned encoder produces host tensors; the stock dtype-only
    cast would leave them there and the NPU DiT would meet mixed-device
    operands.  The wrapper moves every returned tensor to
    ``transformer.device`` (real dtype, no complex).
    """
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    od_config = _make_od_config()
    module.apply_qwen_image_2_1_310p_patches(od_config)

    pipeline_cls = sys.modules["vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21"].QwenImage21Pipeline
    assert getattr(pipeline_cls, "_310p_encode_boundary_patched", False) is True

    pipe = pipeline_cls()
    if torch.npu.is_available():
        pipe.transformer.device = torch.device("npu")
        embeds, mask, pad = pipeline_cls.encode_prompt(pipe)
        assert all(t.device.type == "npu" for t in (embeds, mask, pad))
        # dtype is preserved (fp16 cast already happened inside the original)
        assert embeds.dtype == torch.float32
    else:  # no NPU in this environment: the move is a no-op onto CPU
        embeds, mask, pad = pipeline_cls.encode_prompt(pipe)
        assert all(t.device.type == "cpu" for t in (embeds, mask, pad))


def test_loader_moves_components_to_npu_after_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The post-load placement wrapper relocates off-NPU transformer/vae.

    The CPU-build patch leaves every param born on the host; the stock
    non-quant post-load path moves nothing, so the wrapper must perform
    the explicit CPU->NPU move (upstream custom_pipeline-branch pattern).
    """
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    if not torch.npu.is_available():
        pytest.skip("NPU required for the placement move")
    module._patch_loader_final_placement()
    loader_mod = sys.modules["vllm_omni.diffusion.model_loader.diffusers_loader"]
    loader_cls = loader_mod.DiffusersPipelineLoader
    assert getattr(loader_cls, "_310p_final_placement_patched", False) is True

    loader = loader_cls()
    loaded = loader_cls.load_model(loader, object())
    assert next(loaded.transformer.parameters()).device.type == "npu"
    assert next(loaded.vae.parameters()).device.type == "npu"


def test_torch_sdpa_gate_passes_through_non_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The F.scaled_dot_product_attention gate keeps stock behaviour off-NPU.

    The VAE mid-block attention calls torch's F.sdpa directly; on 310P the
    fused CANN attention rejects its fp16 path.  The gate must route NPU
    calls to the manual eager decomposition while leaving every non-NPU
    call (CPU text encoder) on the stock function untouched.
    """
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    module._patch_torch_sdpa_npu_gate()
    F = torch.nn.functional
    assert getattr(F, "_310p_sdpa_npu_gated", False) is True

    q = torch.randn(1, 2, 8, 4, requires_grad=False)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    # CPU passthrough: must be the stock function's result, numerically
    # equal to the manual math path.
    out = F.scaled_dot_product_attention(q, k, v)
    scale = 1.0 / (4 ** 0.5)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    ref = torch.softmax(scores, dim=-1) @ v
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


def test_loader_vae_device_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """1024² degradation ①: VAE_DEVICE env pins the VAE to another device.

    With ``VLLM_OMNI_310P_VAE_DEVICE`` set the post-load wrapper moves the
    VAE to that device (second die of the 310I Duo) instead of the default
    local NPU; the transformer keeps the default placement and the
    unconfigured default behaviour is unchanged (sibling test).
    """
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    if not torch.npu.is_available():
        pytest.skip("NPU required for the placement move")
    monkeypatch.setenv("VLLM_OMNI_310P_VAE_DEVICE", "cpu")
    module._patch_loader_final_placement()
    loader_mod = sys.modules["vllm_omni.diffusion.model_loader.diffusers_loader"]
    loader_cls = loader_mod.DiffusersPipelineLoader
    assert getattr(loader_cls, "_310p_final_placement_patched", False) is True

    loader = loader_cls()
    loaded = loader_cls.load_model(loader, object())
    assert next(loaded.transformer.parameters()).device.type == "npu"
    assert next(loaded.vae.parameters()).device.type == "cpu"
