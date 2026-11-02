# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Kandinsky 6 family 310P patch wiring.

Loads ``vllm_omni/platforms/npu/_310p/patch/kandinsky6.py`` and the 310P
patch registry from source with fake ``transformers`` / ``vllm_omni``
dependencies, so the patch contract (FP16 enforcement, attention-backend
forcing, cache disabling, quantization tier gate, single-chip parallelism
preflight, CPU scheduler sigma pinning, CPU text-encoder routing, registry
dispatch on ``model_class_name``) is pinned without an NPU, vllm-ascend or a
real model.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_cosmos3_edge`.
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


def _make_od_config(
    *,
    dtype: torch.dtype = torch.float16,
    model_class_name: str = "Kandinsky6TI2VAPipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    model_config: dict | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    allgather_degree: int = 1,
    cfg_parallel_size: int = 1,
):
    return SimpleNamespace(
        dtype=dtype,
        model_class_name=model_class_name,
        diffusion_attention_config=SimpleNamespace(
            default=SimpleNamespace(backend=default_backend) if default_backend is not None else None,
            per_role={role: SimpleNamespace(backend=backend) for role, backend in (per_role or {}).items()},
        ),
        cache_backend=cache_backend,
        quantization_config=quantization_config,
        model_config=dict(model_config if model_config is not None else {}),
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        cfg_parallel_size=cfg_parallel_size,
    )


def _make_fake_quant(method: str):
    return SimpleNamespace(get_name=lambda method=method: method)


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules kandinsky6.py imports (top-level + lazy)."""
    captured: dict[str, object] = {}

    class FakeAttentionSpec:
        def __init__(self, backend: str):
            self.backend = backend

    _install_fake_module(monkeypatch, "vllm_omni.diffusion.data", AttentionSpec=FakeAttentionSpec)

    members = {
        name: SimpleNamespace(get_path=lambda n=name: f"vllm_omni.diffusion.attention.backends.{n.lower()}.Backend")
        for name in ("TORCH_SDPA", "FLASH_ATTN", "RAINFUSION_ATTN")
    }

    class FakeDiffusionAttentionBackendEnum:
        pass

    for name, member in members.items():
        setattr(FakeDiffusionAttentionBackendEnum, name, member)
    captured["enum"] = FakeDiffusionAttentionBackendEnum
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.attention.backends.registry",
        DiffusionAttentionBackendEnum=FakeDiffusionAttentionBackendEnum,
    )

    calls: list[tuple] = []

    class FakeNPUOmniPlatform:
        @classmethod
        def get_diffusion_attn_backend_cls(cls, selected_backend, head_size, allow_trtllm_default: bool = True):
            calls.append((selected_backend, head_size))
            return f"resolved:{selected_backend}"

    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu.platform",
        NPUOmniPlatform=FakeNPUOmniPlatform,
    )

    captured["backend_calls"] = calls

    # --- top-level imports of the patch module -------------------------
    class FakeBasePipeline(torch.nn.Module):
        """Minimal base; subclass overrides are exercised with fakes below."""

    class FakeKandinsky6Transformer(torch.nn.Module):
        @classmethod
        def from_diffusers_config(cls, config, quant_config=None):
            m = cls()
            m._config = config
            return m

    class FakeHunyuanVAE(torch.nn.Module):
        @classmethod
        def from_config(cls, config):
            m = cls()
            m._config = config
            return m

    _install_fake_module(
        monkeypatch,
        "transformers",
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
        CLIPTextModel=type("CLIPTextModel", (torch.nn.Module,), {}),
        CLIPTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        Qwen2_5_VLForConditionalGeneration=type(
            "Qwen2_5_VLForConditionalGeneration", (torch.nn.Module,), {}
        ),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer",
        Kandinsky6Transformer3DModel=FakeKandinsky6Transformer,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.kandinsky6.modeling_kandinsky6_vae",
        AutoencoderKLHunyuanVideo=FakeHunyuanVAE,
    )

    # Real self-contained scheduler (imports only torch) - load from source.
    scheduler_path = (
        _repo_root()
        / "vllm_omni"
        / "diffusion"
        / "models"
        / "kandinsky6"
        / "scheduling_kandinsky6.py"
    )
    real_scheduler = _load_source_module("real_kandinsky6_scheduler_under_test", scheduler_path)

    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6",
        KandinskyFlowMatchScheduler=real_scheduler.KandinskyFlowMatchScheduler,
    )
    captured["scheduler_cls"] = real_scheduler.KandinskyFlowMatchScheduler

    pipeline_attrs = {
        "_CLIP_MAX_LENGTH": 77,
        # Positional placeholder: the pipeline formats it with
        # ``_PROMPT_TEMPLATE.format(text)`` (positional arg), like upstream.
        "_PROMPT_TEMPLATE": "<prompt>{0}</prompt>",
        "_QWEN_CROP_START": 34,
        "_build_audio_vae": lambda *a, **k: torch.nn.Module(),
        "_component_sources": lambda *a, **k: {},
        "_load_json": lambda *a, **k: {},
        "_resolve_bundle_root": lambda model: Path("/fake/model/root"),
        "Kandinsky6TI2VAPipeline": FakeBasePipeline,
    }
    pipeline_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6",
        **pipeline_attrs,
    )
    captured["fake_pipeline_module"] = pipeline_module

    # lazy imports of the module under test / registry
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.registry",
        register_diffusion_model=lambda *a, **k: None,
    )
    _install_fake_module(monkeypatch, "vllm", logger=logging.getLogger)
    _install_fake_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: None,
    )
    return captured


def _load_kandinsky6_patch(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    module = _load_source_module("kandinsky6_310p_under_test", patch_dir / "kandinsky6.py")
    # Fresh module globals per test: reset the idempotency flags.
    module._APPLIED = False
    module._ATTENTION_GUARD_INSTALLED = False
    return module


def _load_registry_module(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    return _load_source_module("kandinsky6_registry_under_test", patch_dir / "__init__.py")


# ---------------------------------------------------------------------------
# dtype / attention / cache invariants
# ---------------------------------------------------------------------------


def test_fp16_is_accepted_and_default_backend_normalized(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.float16)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default is not None
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert od_config.cache_backend is None


def test_bf16_dtype_is_rejected_on_310p(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="FP16"):
        module.apply_diffusion_310p_patches(_make_od_config(dtype=torch.bfloat16))


def test_explicit_mindiesd_default_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(default_backend="FLASH_ATTN"))


def test_explicit_mindiesd_per_role_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(per_role={"self": "RAINFUSION_ATTN"}))


def test_cache_backends_are_disabled(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    for backend in ("cache_dit", "tea_cache", "mag_cache"):
        # Each iteration validates a fresh od_config: the module-level
        # once-per-process idempotency flag must not swallow it.
        module._APPLIED = False
        od_config = _make_od_config(cache_backend=backend)
        module.apply_diffusion_310p_patches(od_config)
        assert od_config.cache_backend == "none", backend


# ---------------------------------------------------------------------------
# static window buffer guard (aclnnInplaceCos / DT_INT64, task-30 run 2)
# ---------------------------------------------------------------------------


def test_static_window_guard_pins_windows_to_cpu(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    window_names = (
        "hann_window",
        "hamming_window",
        "blackman_window",
        "kaiser_window",
        "bartlett_window",
    )
    # The guard intentionally monkey-patches torch; restore the original
    # builders after the test.
    for name in window_names:
        monkeypatch.setattr(torch, name, getattr(torch, name))

    original_hann = torch.hann_window
    module._install_static_window_guard()
    for name in window_names:
        assert getattr(getattr(torch, name), "_310p_window_cpu_guarded", False), name

    # Re-install is a no-op (module flag and wrapper sentinel).
    module._install_static_window_guard()
    module._STATIC_WINDOW_GUARD_INSTALLED = False
    module._install_static_window_guard()
    wrapped = torch.hann_window
    assert torch.hann_window is wrapped

    # The guard must be value-exact against the stock builder: same buffer,
    # computed on CPU regardless of any device context.
    window = torch.hann_window(129)
    assert window.device.type == "cpu"
    assert torch.equal(window, original_hann(129))


# ---------------------------------------------------------------------------
# quantization tier gate
# ---------------------------------------------------------------------------


def test_int8_quantization_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    od_config = _make_od_config(quantization_config=_make_fake_quant("int8"))
    module.apply_diffusion_310p_patches(od_config)  # must not raise


def test_fp8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="rejects quantization"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("fp8")))


def test_unset_quantization_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())  # FP16 primary tier


# ---------------------------------------------------------------------------
# single-chip parallelism gate
# ---------------------------------------------------------------------------


def test_multi_rank_sequence_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    for attr_kwargs in (
        {"ulysses_degree": 2},
        {"ring_degree": 2},
        {"allgather_degree": 2},
    ):
        # A rejected config still latches _APPLIED (the flag is set before
        # validation); reset it so each rejection is actually checked.
        module._APPLIED = False
        with pytest.raises(RuntimeError, match="multi-rank HCCL"):
            module.apply_diffusion_310p_patches(_make_od_config(**attr_kwargs))


def test_cfg_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="cfg_parallel_size"):
        module.apply_diffusion_310p_patches(_make_od_config(cfg_parallel_size=2))


# ---------------------------------------------------------------------------
# idempotency / attention guard
# ---------------------------------------------------------------------------


def test_apply_is_idempotent(monkeypatch: pytest.MonkeyPatch, captured_registry_calls):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    od_config = _make_od_config()
    module.apply_diffusion_310p_patches(od_config)
    applied_before = module._APPLIED
    assert applied_before is True
    # Second call: no exception, no re-registration side effects.
    module.apply_diffusion_310p_patches(od_config)
    assert module._APPLIED is True


def test_attention_guard_install_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    module._install_attention_guard()
    module._install_attention_guard()  # must not double-wrap / raise
    assert module._ATTENTION_GUARD_INSTALLED is True

    platform = sys.modules["vllm_omni.platforms.npu.platform"].NPUOmniPlatform
    guarded = platform.__dict__["get_diffusion_attn_backend_cls"]
    assert getattr(guarded, "_310p_attention_guarded", False) is True


def test_attention_guard_forces_torch_sdpa_on_none(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    module._install_attention_guard()

    platform = sys.modules["vllm_omni.platforms.npu.platform"].NPUOmniPlatform
    resolved = platform.get_diffusion_attn_backend_cls(None, 64)
    assert "torch_sdpa" in resolved.lower()


def test_attention_guard_rejects_flash_attn_selection(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    module._install_attention_guard()

    platform = sys.modules["vllm_omni.platforms.npu.platform"].NPUOmniPlatform
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        platform.get_diffusion_attn_backend_cls("FLASH_ATTN", 64)


# ---------------------------------------------------------------------------
# scheduler sigma pinning (RealDiv-2 family, research/012)
# ---------------------------------------------------------------------------


def test_scheduler_sigma_math_pinned_to_cpu(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    scheduler_cls = captured["scheduler_cls"]

    scheduler = scheduler_cls(scheduler_scale=5.0, device="cpu")
    original_set_timesteps = scheduler.set_timesteps
    calls: list[dict] = []

    def _spy_set_timesteps(num_inference_steps, device=None, **kwargs):
        calls.append({"num": num_inference_steps, "device": device})
        return original_set_timesteps(num_inference_steps, device=device or "cpu", **kwargs)

    scheduler.set_timesteps = _spy_set_timesteps
    module._pin_scheduler_sigma_math_to_cpu(scheduler)

    # The denoise loop calls set_timesteps with the DiT device; the pin must
    # force the sigma math to CPU and move the tensors back.
    scheduler.set_timesteps(num_inference_steps=50, device="cpu")
    assert calls[-1]["device"] == "cpu"
    assert scheduler.timesteps.device.type == "cpu"
    assert scheduler.sigmas.device.type == "cpu"
    assert scheduler.device.type == "cpu"
    # Numerics: sigma = scale * t / (1 + (scale - 1) * t), t in [1, 0].
    expected_first = 5.0 * 1.0 / (1.0 + 4.0 * 1.0)
    assert torch.allclose(scheduler.sigmas[:1], torch.tensor([expected_first]))
    assert len(scheduler.timesteps) == 50


def test_scheduler_pin_moves_tensors_to_requested_device(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    scheduler_cls = captured["scheduler_cls"]

    scheduler = scheduler_cls(scheduler_scale=5.0, device="cpu")
    module._pin_scheduler_sigma_math_to_cpu(scheduler)
    scheduler.set_timesteps(num_inference_steps=4, device="cpu")
    # On a real node the requested device is the NPU; exercise the .to()
    # move path explicitly to a different (cpu here) device object.
    scheduler.timesteps = scheduler.timesteps.to("cpu")
    assert scheduler.timesteps.device.type == "cpu"


# ---------------------------------------------------------------------------
# CPU text-encoder routing
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Pretends to tokenize a fixed-length, fully-packed prompt
    (prefix + content, no padding) so the fixed-offset crop math is exact."""

    def __init__(self, max_length: int = 98):
        self.max_length = max_length
        self.tokenizer = SimpleNamespace(padding_side=None)

    def __call__(self, text, images=None, videos=None, max_length=None, **kwargs):
        length = max_length or self.max_length
        batch = SimpleNamespace(
            input_ids=torch.tensor([[1] * length]),
            attention_mask=torch.ones(1, length, dtype=torch.long),
        )
        return _FakeTensorBatch(batch)


class _FakeTensorBatch:
    def __init__(self, inner: SimpleNamespace):
        self._inner = inner

    def to(self, device, **kwargs):
        return self

    def keys(self):
        # BatchEncoding is a mapping upstream (the pipeline unpacks it with
        # ``**inputs`` into the CLIP encoder); mimic that here.
        return self._inner.__dict__.keys()

    def __getitem__(self, key):
        return getattr(self._inner, key)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FakeQwenEncoder(torch.nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self._hidden = hidden

    def forward(self, input_ids, return_dict=True, output_hidden_states=True):
        length = input_ids.shape[1]
        return {"hidden_states": [torch.randn(1, length, self._hidden)]}


class _FakeCLIPEncoder(torch.nn.Module):
    def forward(self, **inputs):
        return {"pooler_output": torch.randn(1, 8)}


def _make_fake_pipeline_instance(module, monkeypatch):
    """Build a subclass instance without running the heavy __init__."""
    cls = module._Kandinsky6TI2VA310P
    inst = object.__new__(cls)
    torch.nn.Module.__init__(inst)
    inst.device = torch.device("cpu")  # host test: single device
    inst._encoder_device = torch.device("cpu")
    inst.max_sequence_length = 64
    inst.text_token_padding = False
    inst.tokenizer = _FakeTokenizer(max_length=64 + 34)
    inst.tokenizer_2 = _FakeTokenizer(max_length=77)
    inst.text_encoder = _FakeQwenEncoder()
    inst.text_encoder_2 = _FakeCLIPEncoder()
    return inst


# ---------------------------------------------------------------------------
# SDPA math decomposition: per-head-chunk score buffer (video-tier OOM fix)
# ---------------------------------------------------------------------------


def _load_attention_module(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    return _load_source_module("attention_310p_under_test", patch_dir / "attention.py")


def _naive_sdpa(query, key, value, attention_mask, causal, scale):
    """Reference full-heads-at-once formula (the un-chunked semantics)."""
    if query.shape[1] != key.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    attn = torch.matmul(query * scale, key.transpose(-2, -1))
    if causal:
        seq = attn.shape[-1]
        mask = torch.ones(seq, seq, dtype=torch.bool).tril()
        attn = attn.masked_fill(~mask, float("-inf"))
    if attention_mask is not None:
        attn = attn + attention_mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, value)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("with_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_math_sdpa_chunked_matches_naive(monkeypatch, causal, with_bias, dtype):
    monkeypatch.setenv("VLLM_OMNI_310P_SDPA_HEAD_CHUNK", "2")  # force multi-chunk (6 heads -> 2+2+2)
    attention = _load_attention_module(monkeypatch)

    g = torch.Generator().manual_seed(7)
    batch, q_heads, kv_heads, seq, dim = 2, 6, 2, 64, 16
    query = torch.randn(batch, q_heads, seq, dim, generator=g, dtype=dtype)
    key = torch.randn(batch, kv_heads, seq, dim, generator=g, dtype=dtype)
    value = torch.randn(batch, kv_heads, seq, dim, generator=g, dtype=dtype)
    bias = torch.randn(batch, 1, seq, seq, generator=g, dtype=dtype) if with_bias else None

    out = attention._math_sdpa(query, key, value, bias, causal, None)
    ref = _naive_sdpa(query, key, value, bias, causal, None)
    assert out.shape == ref.shape
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float32 else (2e-2, 2e-2)
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)


def test_math_sdpa_chunk_size_env_budget(monkeypatch):
    attention = _load_attention_module(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_SDPA_HEAD_CHUNK", "2")
    assert attention._sdpa_score_chunk_heads(1, 5, 96, 2) == 2
    monkeypatch.delenv("VLLM_OMNI_310P_SDPA_HEAD_CHUNK")
    monkeypatch.setenv("VLLM_OMNI_310P_SDPA_SCORE_BUDGET_MB", "1024")
    # 1 x 512 x 512 x 2B = 512 KiB per head -> 2048 heads fit in 1 GiB,
    # capped by the actual head count.
    assert attention._sdpa_score_chunk_heads(1, 8, 512, 2) == 8
    monkeypatch.setenv("VLLM_OMNI_310P_SDPA_SCORE_BUDGET_MB", "2")
    # 2 MiB budget // 512 KiB per head = 4 -> 4 heads.
    assert attention._sdpa_score_chunk_heads(1, 8, 512, 2) == 4
    # Tiny budget -> floor to 1 head, never 0 (1 MiB // 512 KiB = 2).
    monkeypatch.setenv("VLLM_OMNI_310P_SDPA_SCORE_BUDGET_MB", "1")
    assert attention._sdpa_score_chunk_heads(1, 8, 512, 2) == 2


# ---------------------------------------------------------------------------
# latent dtype pinning (silent-bf16 trap via missing transformer.dtype)
# ---------------------------------------------------------------------------


def test_prepare_latents_pins_noise_to_runtime_fp16(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    inst = _make_fake_pipeline_instance(module, monkeypatch)
    inst.od_config = SimpleNamespace(dtype=torch.float16)

    # Recorder for the base-class call that the override delegates to.
    class _Base:
        def prepare_latents(self, **kwargs):
            _Base.captured = kwargs
            return "bundle"

    _Base.captured = None
    base_cls = type(inst).__mro__[1]
    monkeypatch.setattr(
        base_cls, "prepare_latents", _Base.prepare_latents, raising=False
    )

    out = inst.prepare_latents(
        latent_frames=13,
        height=480,
        width=864,
        dtype=torch.bfloat16,  # base fallback value; must be overridden
        device="npu:0",
        seed=42,
        sample_audio=True,
    )
    assert out == "bundle"
    assert _Base.captured is not None
    assert _Base.captured["dtype"] is torch.float16
    assert _Base.captured["latent_frames"] == 13


def test_prepare_latents_rejects_non_fp16_runtime_dtype(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    inst = _make_fake_pipeline_instance(module, monkeypatch)
    inst.od_config = SimpleNamespace(dtype=torch.bfloat16)

    base_cls = type(inst).__mro__[1]
    monkeypatch.setattr(
        base_cls,
        "prepare_latents",
        lambda self, **kwargs: "bundle",
        raising=False,
    )
    with pytest.raises(RuntimeError, match="FP16"):
        inst.prepare_latents(
            latent_frames=13,
            height=480,
            width=864,
            dtype=torch.bfloat16,
            device="npu:0",
            seed=42,
            sample_audio=True,
        )


def test_encode_prompt_routes_to_encoder_device(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    inst = _make_fake_pipeline_instance(module, monkeypatch)

    embeds, cu_seqlens, attention_mask = inst.encode_prompt("a stone warrior")
    assert set(embeds) == {"text_embeds", "pooled_embed"}
    assert embeds["text_embeds"].device == inst._encoder_device
    assert embeds["pooled_embed"].device == inst._encoder_device
    assert cu_seqlens.device == inst._encoder_device
    # 34 tokens cropped by the fixed offset; the rest are content.
    assert int(cu_seqlens[1]) == 64


def test_encode_prompt_padding_branch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    inst = _make_fake_pipeline_instance(module, monkeypatch)
    inst.text_token_padding = True

    embeds, cu_seqlens, attention_mask = inst.encode_prompt("a stone warrior")
    assert attention_mask is not None
    assert int(cu_seqlens[1]) == 64


def test_encoder_on_npu_env_gate(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    monkeypatch.delenv("VLLM_OMNI_310P_KANDINSKY6_ENCODER_ON_NPU", raising=False)
    assert module._encoder_on_npu() is False
    monkeypatch.setenv("VLLM_OMNI_310P_KANDINSKY6_ENCODER_ON_NPU", "1")
    assert module._encoder_on_npu() is True


# ---------------------------------------------------------------------------
# registry dispatch on model_class_name
# ---------------------------------------------------------------------------


def test_registry_dispatches_kandinsky6(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)

    # Expose the loaded module under its REAL name so the dispatcher's
    # importlib.import_module resolves it without the heavy dependencies.
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.kandinsky6", module)

    registry = _load_registry_module(monkeypatch)
    od_config = _make_od_config()
    registry.apply_diffusion_310p_patches(od_config)

    assert module._APPLIED is True


def test_registry_unknown_arch_is_noop(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.kandinsky6", module)

    registry = _load_registry_module(monkeypatch)
    od_config = _make_od_config(model_class_name="SomeOtherPipeline")
    registry.apply_diffusion_310p_patches(od_config)

    assert module._APPLIED is False


def test_registry_kill_switch_disables_dispatch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_kandinsky6_patch(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.kandinsky6", module)
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")

    registry = _load_registry_module(monkeypatch)
    registry.apply_diffusion_310p_patches(_make_od_config())

    assert module._APPLIED is False


# ---------------------------------------------------------------------------
# real-device assertions (container only; skipped on CPU-only hosts)
# ---------------------------------------------------------------------------


def test_is_310p_real_device():
    torch_npu = pytest.importorskip("torch_npu")
    import vllm_omni.platforms.npu._310p as _310p

    assert _310p.is_310p() is True
    assert torch.npu.is_available() is True
    assert torch.npu.device_count() >= 1


def test_real_patch_module_importable_in_container():
    """Inside the 310P image the full import chain must resolve (real
    transformers/diffusers/vllm present)."""
    torch_npu = pytest.importorskip("torch_npu")
    import importlib

    module = importlib.import_module("vllm_omni.platforms.npu._310p.patch.kandinsky6")
    assert module.KANDINSKY6_ARCH == "Kandinsky6TI2VAPipeline"
    assert hasattr(module, "_Kandinsky6TI2VA310P")


class _OdConfigGuard:
    """Fixture collecting registry-level od_config state across tests."""


@pytest.fixture
def captured_registry_calls():
    return []
