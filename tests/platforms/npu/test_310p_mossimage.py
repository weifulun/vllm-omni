# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-runnable contract tests for the 310P MOSS-Image patch (task-19).

Validates the *patch contract* (dispatch gating + kill switch, dtype
enforcement, attention selection, quantization tiers, cache-backend guard,
heads preflight, manual-SDPA equivalence incl. chunked path, loader CPU
construction, CPU-pinned encoder, explicit post-load placement) with fake
modules.  No NPU is required; on a 310P box the real-device smoke test
(see risk list section 5) extends this file once the branch lands.

Run from the repo root (py3.9-compatible)::

    python3 -m pytest tests/platforms/npu/test_310p_mossimage.py \
        --noconftest -o addopts="" -v
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.nn import functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Source loading with fakes (same convention as test_310p_qwen_image_21_patches)
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


def _load_mossimage_patch(monkeypatch: pytest.MonkeyPatch, is_310p_value: bool = True):
    """Load the patch source with a controllable ``is_310p`` and fake
    loader/attention/pipeline modules (nothing real is imported)."""
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models")

    # Fake loader module for the CPU-construction hook.
    loader_mod = _install_fake_module(
        monkeypatch, "vllm_omni.diffusion.model_loader.diffusers_loader",
        initialize_model=lambda od_config: od_config,
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

    # Fake MOSSImage pipeline module for the encoder-placement + post-load
    # placement hooks.
    pipeline_mod = types.ModuleType("vllm_omni_fake_pipeline_mossimage")

    def _fpwp_passthrough(factory, *args, **kwargs):
        return factory(*args, **kwargs)

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return types.SimpleNamespace(**{"_built": True})

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return types.SimpleNamespace(**{"_tokenizer": True})

    class _FakeMOSSImagePipeline:
        def __init__(self):
            self.transformer = torch.nn.Linear(4, 4)
            self.vae = torch.nn.Linear(4, 4)
            self.text_encoder = torch.nn.Linear(4, 4)

        def load_weights(self, weights):
            return "loaded"

        def encode_prompt(self, prompt, max_sequence_length=512):
            # CPU outputs: mirrors the CPU-pinned encoder (invariant 4) --
            # the 310P wrap under test must move these to the execution device.
            return torch.zeros(1, 8, 4), torch.ones(1, 8, dtype=torch.bool)

    class _FakeMOSSImageScheduler:
        """Mimics diffusers: materializes sigmas/timesteps as float64 tensors
        on the execution device (the fp64 grid the 310P wrap must downcast)."""

        def set_timesteps(self, num_inference_steps=50, device=None, shift=None):
            dev = device if device is not None else torch.device("cpu")
            self.sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64, device=dev)
            self.timesteps = torch.tensor([999.0, 499.0], dtype=torch.float64, device=dev)
            return None

    class _FakeMOSSImageTransformer:
        freq_dim = 8

        def _time_modulation(self, timestep, batch, device):
            # Faithful replica of the upstream fp64 body (what the 310P wrap
            # must replace -- float64 arange + times.double()).
            times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
            times = times.expand(batch)
            half = self.freq_dim // 2
            freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
            phase = times.double()[:, None] * freq
            embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
            time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
            mod = self.shared_block_modulation(
                time.to(self.shared_block_modulation.weight.dtype)
            ).view(batch, 6, -1)
            head = self.video_head_modulation(
                time.to(self.video_head_modulation.weight.dtype)
            ).view(batch, 2, -1)
            return mod, head

    pipeline_mod.from_pretrained_with_prefetch = _fpwp_passthrough
    pipeline_mod.AutoModel = _FakeAutoModel
    pipeline_mod.AutoTokenizer = _FakeTokenizer
    pipeline_mod.MOSSImagePipeline = _FakeMOSSImagePipeline
    pipeline_mod.MOSSImageScheduler = _FakeMOSSImageScheduler
    monkeypatch.setitem(
        sys.modules, "vllm_omni.diffusion.models.mossimage",
        types.ModuleType("vllm_omni_fake_mossimage_pkg"),
    )
    # Fake transformer module (the _time_modulation fp32-replacement target);
    # registered after the fake package above so the attribute binds.
    transformer_mod = types.ModuleType("vllm_omni_fake_mossimage_transformer")
    transformer_mod.MOSSImageTransformer = _FakeMOSSImageTransformer
    monkeypatch.setitem(
        sys.modules, "vllm_omni.diffusion.models.mossimage.mossimage_transformer",
        transformer_mod,
    )
    setattr(
        sys.modules["vllm_omni.diffusion.models.mossimage"],
        "mossimage_transformer",
        transformer_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.models.mossimage.pipeline_mossimage",
        pipeline_mod,
    )
    # `from ...mossimage import pipeline_mossimage` resolves via getattr on the
    # fake package; bind the submodule attribute explicitly.
    setattr(
        sys.modules["vllm_omni.diffusion.models.mossimage"],
        "pipeline_mossimage",
        pipeline_mod,
    )
    setattr(sys.modules["vllm_omni.diffusion.models"], "mossimage",
            sys.modules["vllm_omni.diffusion.models.mossimage"])

    # Fake get_local_device for the post-load placement hook.
    distributed_pkg = types.ModuleType("vllm_omni.diffusion.distributed")
    distributed_utils = types.ModuleType("vllm_omni.diffusion.distributed.utils")
    distributed_utils.get_local_device = lambda: torch.device("cpu")
    distributed_pkg.utils = distributed_utils
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.distributed", distributed_pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.distributed.utils",
                        distributed_utils)

    # Fake diffusion.data for AttentionSpec.
    diffusion_data = _install_fake_module(
        monkeypatch, "vllm_omni.diffusion.data",
        AttentionSpec=lambda **kw: SimpleNamespace(**kw),
    )
    setattr(sys.modules["vllm_omni.diffusion"], "data", diffusion_data)

    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p",
                         is_310p=lambda: is_310p_value)

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "mossimage.py"
    module = _load_source_module("vllm_omni_test_310p_mossimage_patch", path)
    module._APPLIED = False
    return module, loader_mod, sdpa_mod, pipeline_mod


def _make_od_config(model_class_name: str = "MOSSImagePipeline", **overrides) -> SimpleNamespace:
    defaults = dict(
        model_class_name=model_class_name,
        dtype=torch.bfloat16,
        enforce_eager=False,
        vae_use_tiling=False,
        vae_use_slicing=False,
        quantization_config=None,
        cache_backend=None,
        diffusion_attention_config=None,
        tensor_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Dispatch gating / kill switch
# ---------------------------------------------------------------------------


def test_dispatch_is_gated_on_310p_and_model_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    module._APPLIED = False
    module.maybe_apply_diffusion_310p_patches(_make_od_config())
    assert module._APPLIED is True

    # Other diffusion models are untouched by this patch file.
    other = _make_od_config(model_class_name="StableDiffusionXLPipeline")
    module.maybe_apply_diffusion_310p_patches(other)
    assert other.dtype is torch.bfloat16  # untouched


def test_dispatch_kill_switch_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")

    module._APPLIED = False
    module.maybe_apply_diffusion_310p_patches(_make_od_config())
    assert module._APPLIED is False


def test_dispatch_is_noop_off_310p(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch, is_310p_value=False)
    config = _make_od_config()
    module.maybe_apply_diffusion_310p_patches(config)
    assert module._APPLIED is False
    assert config.dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# Runtime configuration enforcement
# ---------------------------------------------------------------------------


def test_runtime_dtype_forced_fp16_and_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    assert module.resolve_runtime_dtype(torch.bfloat16) is torch.float16
    assert module.resolve_runtime_dtype(torch.float16) is torch.float16

    module._APPLIED = False
    module.apply_diffusion_310p_patches(_make_od_config())
    assert module._APPLIED is True


def test_apply_forces_bf16_config_to_fp16_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)
    module._APPLIED = False
    config = _make_od_config(dtype=torch.bfloat16)
    with caplog.at_level("WARNING"):
        module.apply_diffusion_310p_patches(config)

    assert config.dtype is torch.float16
    assert config.enforce_eager is True
    assert config.vae_use_tiling is True
    assert config.vae_use_slicing is True
    assert any("forcing runtime dtype" in record.message for record in caplog.records)


def test_apply_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)
    module._APPLIED = False
    config = _make_od_config(dtype=torch.float16)
    module.apply_diffusion_310p_patches(config)
    module.apply_diffusion_310p_patches(config)  # second call must be a no-op
    assert module._APPLIED is True
    assert config.dtype is torch.float16


# ---------------------------------------------------------------------------
# Attention selection
# ---------------------------------------------------------------------------


def _attention_config(default: Any = None, per_role: Any = None):
    class _Spec:
        def __init__(self, backend):
            self.backend = backend

    cfg = SimpleNamespace(default=_Spec(default) if default is not None else None,
                          per_role=per_role or {})
    return cfg


def test_attention_default_is_torch_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    # No explicit config: resolver reports TORCH_SDPA.
    assert module.validate_attention_selection(_make_od_config()) == "TORCH_SDPA"

    # Empty config object: default spec is set to TORCH_SDPA.
    config = _make_od_config(diffusion_attention_config=_attention_config())
    assert module.validate_attention_selection(config) == "TORCH_SDPA"
    assert str(config.diffusion_attention_config.default.backend).upper() == "TORCH_SDPA"


def test_attention_rejects_mindie_sd_only_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    for backend in ("FLASH_ATTN", "RAINFUSION_ATTN"):
        config = _make_od_config(diffusion_attention_config=_attention_config(default=backend))
        with pytest.raises(RuntimeError, match="MindIE-SD"):
            module.validate_attention_selection(config)

    config = _make_od_config(
        diffusion_attention_config=_attention_config(
            per_role={"self": SimpleNamespace(backend="FLASH_ATTN")}
        )
    )
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.validate_attention_selection(config)


# ---------------------------------------------------------------------------
# Quantization tiers (POLICY section 2)
# ---------------------------------------------------------------------------


def _quant_config(method: str) -> SimpleNamespace:
    return SimpleNamespace(get_name=lambda: method, method=method)


def test_quantization_gates(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    # No quantization: pass-through.
    module.validate_quantization(_make_od_config())

    # INT8 W8A8: sanctioned fallback tier, accepted with a loud warning.
    config = _make_od_config(quantization_config=_quant_config("int8"))
    with caplog.at_level("WARNING"):
        module.validate_quantization(config)
    assert config.quantization_config is not None
    assert any("INT8 W8A8" in record.message for record in caplog.records)

    # Everything else is rejected on 310P.
    for method in ("fp8", "bitsandbytes", "mxfp4"):
        bad = _make_od_config(quantization_config=_quant_config(method))
        with pytest.raises(RuntimeError, match="rejects quantization method"):
            module.validate_quantization(bad)


# ---------------------------------------------------------------------------
# Cache backend guard + heads preflight
# ---------------------------------------------------------------------------


def test_cache_backend_guard(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    config = _make_od_config(cache_backend="tea_cache")
    with caplog.at_level("WARNING"):
        module.disable_numerical_cache_backends(config)
    assert config.cache_backend == "none"
    assert any("rejecting numerics-changing cache backend" in record.message
               for record in caplog.records)

    untouched = _make_od_config(cache_backend="none")
    module.disable_numerical_cache_backends(untouched)
    assert untouched.cache_backend == "none"


def test_heads_divisibility_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    # TP1 baseline: SP disabled, always fine.
    module.check_heads_divisibility(_make_od_config())

    # tp > 2 is not sanctioned (research/013 environment-level failure).
    with pytest.raises(RuntimeError, match="not sanctioned"):
        module.check_heads_divisibility(_make_od_config(tensor_parallel_size=3))

    # TP2 + ulysses 2: heads 24 % 2 == 0 and tp 2 % 2 == 0 -> valid SP.
    module.check_heads_divisibility(
        _make_od_config(tensor_parallel_size=2, ulysses_degree=2)
    )

    # ulysses 7 does not divide the 24 transformer heads.
    with pytest.raises(RuntimeError, match="does not divide"):
        module.check_heads_divisibility(
            _make_od_config(tensor_parallel_size=2, ulysses_degree=7)
        )

    # Under TP1 no SP rank exists, so any degree > 1 is unsatisfiable.
    with pytest.raises(RuntimeError, match="does not divide"):
        module.check_heads_divisibility(_make_od_config(ulysses_degree=4))


# ---------------------------------------------------------------------------
# Manual SDPA (fused kernel unsupported on 310P3)
# ---------------------------------------------------------------------------


def _rand_bhsd(b, h, s, d, device="cpu", dtype=torch.float32):
    return torch.randn(b, h, s, d, device=device, dtype=dtype)


def test_manual_sdpa_swap_rebinds_and_is_equivalent(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, sdpa_mod, _ = _load_mossimage_patch(monkeypatch)
    module._APPLIED = False
    module.apply_diffusion_310p_patches(_make_od_config(dtype=torch.float16))

    # The module-global rebind happened.
    assert sdpa_mod._sdpa_attention is module._manual_sdpa_forward
    assert getattr(sdpa_mod, "_310p_manual_sdpa_patched", False) is True

    torch.manual_seed(0)
    b, h, s, d = 1, 8, 64, 64
    q, k, v = _rand_bhsd(b, h, s, d), _rand_bhsd(b, h, s, d), _rand_bhsd(b, h, s, d)
    scale = 1.0 / d**0.5

    # No mask.
    got = module._manual_sdpa_forward(q, k, v, None, False, scale)
    want = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, want, atol=2e-4), "unmasked dense path diverges"

    # Bool keep-mask.
    mask = torch.ones(b, 1, s, s, dtype=torch.bool)
    mask[:, :, :, -8:] = False
    got = module._manual_sdpa_forward(q, k, v, mask, False, scale)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(got, want, atol=2e-4), "bool-mask path diverges"

    # Float additive bias.
    bias = torch.randn(b, 1, s, s)
    got = module._manual_sdpa_forward(q, k, v, bias, False, scale)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    assert torch.allclose(got, want, atol=2e-4), "float-bias path diverges"

    # Causal.
    got = module._manual_sdpa_forward(q, k, v, None, True, scale)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(got, want, atol=2e-4), "causal path diverges"

    # GQA expansion parity (q heads 8, kv heads 4, head dim 32).
    q_g, k_g, v_g = _rand_bhsd(1, 8, 32, 32), _rand_bhsd(1, 4, 32, 32), _rand_bhsd(1, 4, 32, 32)
    scale_g = 1.0 / 32**0.5
    got = module._manual_sdpa_forward(q_g, k_g, v_g, None, False, scale_g)
    want = F.scaled_dot_product_attention(q_g, k_g, v_g, enable_gqa=True)
    assert torch.allclose(got, want, atol=2e-4), "GQA expansion path diverges"


def test_manual_sdpa_chunked_path_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    # Force the chunked path on a small shape by shrinking the byte budget.
    monkeypatch.setattr(module, "_SDPA_DENSE_LIMIT_BYTES", 1)
    torch.manual_seed(1)
    b, h, s, d = 1, 4, 200, 32  # 200 > chunk 512? no: 200 < 512 -> single chunk
    q, k, v = _rand_bhsd(b, h, s, d), _rand_bhsd(b, h, s, d), _rand_bhsd(b, h, s, d)
    scale = 1.0 / d**0.5

    got = module._manual_sdpa_forward(q, k, v, None, False, scale)
    want = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, want, atol=2e-4), "chunked path diverges (single chunk)"

    # Multi-chunk shape (s > chunk) with a causal mask.
    s = 700
    q, k, v = _rand_bhsd(b, h, s, d), _rand_bhsd(b, h, s, d), _rand_bhsd(b, h, s, d)
    got = module._manual_sdpa_forward(q, k, v, None, True, scale)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(got, want, atol=2e-4), "chunked causal path diverges"


# ---------------------------------------------------------------------------
# Loader CPU construction (dynamic-TBE RealDiv/OnesLike guard)
# ---------------------------------------------------------------------------


def test_loader_cpu_build_patch_wraps_initialize_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module, loader_mod, _, _ = _load_mossimage_patch(monkeypatch)
    module._APPLIED = False

    seen: dict[str, Any] = {}

    def _fake_initialize_model(od_config):
        seen["ambient_device"] = torch.get_default_device()
        seen["config"] = od_config
        return "built"

    loader_mod.initialize_model = _fake_initialize_model
    module._patch_loader_cpu_build()

    assert getattr(loader_mod, "_310p_cpu_build_patched", False) is True
    wrapped = loader_mod.initialize_model
    config = _make_od_config()
    assert wrapped(config) == "built"
    assert seen["ambient_device"] == torch.device("cpu"), (
        f"initialize_model must run under a CPU device context, "
        f"got {seen['ambient_device']}"
    )
    assert seen["config"] is config

    # Idempotent: second install must not re-wrap.
    module._patch_loader_cpu_build()
    assert loader_mod.initialize_model is wrapped


# ---------------------------------------------------------------------------
# Encoder CPU pin (Qwen3.5, 8.39B)
# ---------------------------------------------------------------------------


def test_pipeline_encoder_placement_pins_auto_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, pipeline_mod = _load_mossimage_patch(monkeypatch)
    module._APPLIED = False

    built_devices: list[Any] = []

    class _RecordingEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

        def record_build(self):
            built_devices.append(torch.get_default_device())
            return self

    # Rebuild the fake AutoModel with a real Module-returning factory.
    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _RecordingEncoder().record_build()

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return "tokenizer"

    pipeline_mod.AutoModel = _FakeAutoModel
    pipeline_mod.AutoTokenizer = _FakeTokenizer

    module._patch_pipeline_encoder_placement(pipeline_mod)
    assert getattr(pipeline_mod, "_310p_encoder_patched", False) is True

    # Encoder factory: built under CPU context + pinned.
    encoder = pipeline_mod.from_pretrained_with_prefetch(_FakeAutoModel.from_pretrained, "ckpt")
    assert built_devices == [torch.device("cpu")]
    assert getattr(encoder, "_310p_cpu_pinned", False) is True
    # Any later NPU-directed .to is redirected to CPU.
    encoder.to("npu")
    assert encoder.weight.device.type == "cpu"

    # Non-encoder factories pass through untouched.
    tokenizer = pipeline_mod.from_pretrained_with_prefetch(_FakeTokenizer.from_pretrained, "ckpt")
    assert tokenizer == "tokenizer"


def test_cpu_pinned_encoder_forward_coerces_input_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pipeline_moss_image.encode_prompt moves tokenizer output to
    ``_execution_device`` (NPU) before the encoder call; a CPU-pinned encoder
    must still see its inputs on the encoder's own device (verified on-device
    as the first-run ``embed_tokens`` npu:0/cpu crash)."""
    module, _, _, _ = _load_mossimage_patch(monkeypatch)
    module._APPLIED = False

    seen: dict[str, Any] = {}

    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

        def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kw):
            del kw
            seen["input_device"] = input_ids.device if input_ids is not None else None
            seen["mask_device"] = attention_mask.device if attention_mask is not None else None
            seen["use_cache"] = use_cache
            return "embeds"

    encoder = module._make_cpu_pinned(_Encoder())
    assert getattr(encoder, "_310p_forward_wrapped", False) is True

    res = encoder(
        input_ids=torch.zeros(3, 4, dtype=torch.long),
        attention_mask=torch.ones(3, 4, dtype=torch.bool),
        use_cache=False,
    )
    assert res == "embeds"
    # Inputs coerced onto the encoder's (CPU) parameter device; non-tensors pass through.
    assert seen["input_device"].type == "cpu"
    assert seen["mask_device"].type == "cpu"
    assert seen["use_cache"] is False

    # Re-pinning is a no-op: forward is not wrapped twice.
    wrapped = encoder.forward
    module._make_cpu_pinned(encoder)
    assert encoder.forward is wrapped


# ---------------------------------------------------------------------------
# Explicit post-load placement (transformer+VAE -> local device)
# ---------------------------------------------------------------------------


def test_post_load_placement_moves_transformer_and_vae(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, pipeline_mod = _load_mossimage_patch(monkeypatch)
    module._APPLIED = False

    moves: dict[str, list[Any]] = {"transformer": [], "vae": [], "text_encoder": []}

    class _SpyLinear(torch.nn.Linear):
        _spy_name = "unknown"

        def to(self, *args, **kwargs):  # noqa: D102
            moves[self._spy_name].append(args[0] if args else kwargs.get("device"))
            return super().to(*args, **kwargs)

    class _SpyTransformer(_SpyLinear):
        _spy_name = "transformer"

    class _SpyVAE(_SpyLinear):
        _spy_name = "vae"

    class _SpyEncoder(_SpyLinear):
        _spy_name = "text_encoder"

    def _wrapped_init(self):
        self.transformer = _SpyTransformer(4, 4)
        self.vae = _SpyVAE(4, 4)
        self.text_encoder = _SpyEncoder(4, 4)
        self.text_encoder._310p_cpu_pinned = True  # pretend encoder pin is active

    original_load_weights = pipeline_mod.MOSSImagePipeline.load_weights

    def _fake_load_weights(self, weights):
        return "loaded"

    pipeline_mod.MOSSImagePipeline.load_weights = _fake_load_weights
    pipeline_mod.MOSSImagePipeline.__init__ = _wrapped_init

    module._install_post_load_placement()
    assert getattr(pipeline_mod.MOSSImagePipeline, "_310p_placement_patched", False) is True

    instance = pipeline_mod.MOSSImagePipeline()
    result = instance.load_weights(None)
    assert result == "loaded"
    # get_local_device() fake returns cpu on this machine; the wrap must call
    # .to(target) on transformer AND vae, and must not touch the encoder.
    assert moves["transformer"] == [torch.device("cpu")]
    assert moves["vae"] == [torch.device("cpu")]
    assert moves["text_encoder"] == []

    # Idempotent: second install is a no-op (wrap not stacked twice).
    first_wrap = pipeline_mod.MOSSImagePipeline.load_weights
    module._install_post_load_placement()
    assert pipeline_mod.MOSSImagePipeline.load_weights is first_wrap


# ---------------------------------------------------------------------------
# Registry dispatch (COMPARISON-POLICY.md section 5 scaffold)
# ---------------------------------------------------------------------------


def test_registry_dispatch_and_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _load_source_module(
        "vllm_omni_test_310p_registry",
        _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py",
    )

    calls: list[Any] = []

    class _FakeApply:
        def apply_diffusion_310p_patches(self, od_config):
            calls.append(od_config)

    fake_mossimage = _install_fake_module(
        monkeypatch, "vllm_omni.platforms.npu._310p.patch.mossimage",
        apply_diffusion_310p_patches=lambda od_config: calls.append(od_config),
    )

    config = _make_od_config()
    registry.apply_diffusion_310p_patches(config)
    assert len(calls) == 1 and calls[0] is config

    # Unknown arch: no-op.
    calls.clear()
    registry.apply_diffusion_310p_patches(_make_od_config(model_class_name="OtherPipeline"))
    assert calls == []

    # Kill switch: no-op even for the registered arch.
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    registry.apply_diffusion_310p_patches(config)
    assert calls == []


# ---------------------------------------------------------------------------
# Engine-path device handoff + fp64 avoidance (first-run fixes, 2026-11-03)
# ---------------------------------------------------------------------------


def test_encode_prompt_output_moved_to_execution_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, pipeline_mod = _load_mossimage_patch(monkeypatch)

    module._APPLIED = False
    module.apply_diffusion_310p_patches(_make_od_config())

    cls = pipeline_mod.MOSSImagePipeline
    assert cls._310p_encode_device_patched is True

    # Same-device passthrough: plain-cpu embeds + plain-cpu transformer ->
    # tensors pass through without a move (identity preserved).
    pipe = cls()
    embeds, mask = pipe.encode_prompt("dummy run")
    assert embeds.device == torch.device("cpu")
    assert mask.device == torch.device("cpu")

    # Cross-device: a param-less transformer makes the wrap fall back to
    # ``pipe.device`` (spoofed as npu:0); both embeds and mask must go through
    # ``.to()`` (recorded via a stub -- no NPU exists on the test host).
    class _NoParams(torch.nn.Module):
        pass

    calls: list[Any] = []

    def _recording_to(self, *args: Any, **kwargs: Any) -> Any:
        calls.append((self.shape, args, kwargs))
        return self

    pipe.transformer = _NoParams()
    pipe.device = torch.device("meta")  # any non-cpu type; no hardware needed
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "to", _recording_to)
        pipe.encode_prompt("dummy run")
    assert len(calls) == 2  # embeds + mask both moved
    assert all(args and args[0] == torch.device("meta") for _, args, _ in calls)


def test_time_modulation_fp32_wrap_replaces_fp64_body(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, _ = _load_mossimage_patch(monkeypatch)

    transformer_mod = sys.modules["vllm_omni.diffusion.models.mossimage.mossimage_transformer"]
    cls = transformer_mod.MOSSImageTransformer
    original = cls._time_modulation  # fake fp64 replica from the helper

    module._APPLIED = False
    module.apply_diffusion_310p_patches(_make_od_config())

    assert cls._310p_time_mod_fp32 is True
    assert cls._time_modulation is not original

    # Idempotent: the installer must not re-wrap an already-patched class.
    patched = cls._time_modulation
    module._patch_time_modulation_fp32()
    assert cls._time_modulation is patched

    # Functional: the patched body must compute fp32-only (fp64 promotion
    # trapped) and return the (mod, head) contract with the right shapes.
    inst = cls.__new__(cls)
    inst.time_embedding = torch.nn.Sequential(
        torch.nn.Linear(8, 4), torch.nn.SiLU(), torch.nn.Linear(4, 4),
    )
    inst.shared_block_modulation = torch.nn.Linear(4, 24)
    inst.video_head_modulation = torch.nn.Linear(4, 8)

    def _no_double(self):
        raise AssertionError("fp64 promotion must not occur on 310P")

    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "double", _no_double)
        mod, head = inst._time_modulation(torch.tensor([0.5]), 1, torch.device("cpu"))
    assert mod.shape == (1, 6, 4)
    assert head.shape == (1, 2, 4)
    assert mod.dtype == torch.float32 and head.dtype == torch.float32


def test_scheduler_sigmas_downcast_to_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _, pipeline_mod = _load_mossimage_patch(monkeypatch)

    module._APPLIED = False
    module.apply_diffusion_310p_patches(_make_od_config())

    cls = pipeline_mod.MOSSImageScheduler
    assert cls._310p_sigmas_fp32 is True

    sched = cls()
    sched.set_timesteps(2, device=torch.device("cpu"))
    assert sched.sigmas.dtype == torch.float32
    assert sched.timesteps.dtype == torch.float32

    # Already-fp32 grids pass through untouched.
    sched.sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)
    sched.timesteps = torch.tensor([999.0], dtype=torch.float32)
    sched.set_timesteps(2, device=torch.device("cpu"))
    assert sched.sigmas.dtype == torch.float32
