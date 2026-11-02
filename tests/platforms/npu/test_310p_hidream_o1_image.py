# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the HiDream-O1-Image 310P patch wiring (task-28).

Loads ``vllm_omni/platforms/npu/_310p/patch/hidream_o1_image.py`` and the 310P
patch registry dispatcher from source with fake ``vllm``/``vllm_omni``
dependencies, so the patch contract (FP16 pin, attention-backend forcing,
cache disabling, quantization tier gate, single-chip parallelism preflight,
pipeline forward CPU-scheduler swap, RoPE / timestep CPU tables, registry
dispatch on ``model_class_name``) is pinned without an NPU, vllm-ascend or
the real model, and without importing the ``vllm_omni`` package at all.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_cosmos3_edge`.
"""

from __future__ import annotations

import importlib.util
import math
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
    dtype: torch.dtype | None = torch.float16,
    model_class_name: str = "HiDreamO1ImagePipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
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
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        cfg_parallel_size=cfg_parallel_size,
    )


def _make_fake_quant(method: str):
    return SimpleNamespace(get_name=lambda method=method: method)


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules hidream_o1_image.py and the registry
    dispatcher import lazily.  Everything resolves through sys.modules, so
    the vllm_omni package is never imported (parent packages stubbed too,
    otherwise the import machinery would run the real __init__ chain)."""
    captured: dict[str, object] = {}

    for parent in (
        "vllm",
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.models",
        "vllm_omni.diffusion.models.hidream_o1_image",
        "vllm_omni.diffusion.attention",
        "vllm_omni.diffusion.attention.backends",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
        "vllm_omni.platforms.npu._310p.patch",
    ):
        _install_fake_module(monkeypatch, parent)

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

    class FakeNPUOmniPlatform:
        @classmethod
        def get_diffusion_attn_backend_cls(cls, selected_backend, head_size, allow_trtllm_default: bool = True):
            return (selected_backend, head_size)

    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu.platform", NPUOmniPlatform=FakeNPUOmniPlatform)
    captured["platform"] = FakeNPUOmniPlatform

    class FakeDiffusionOutput:
        def __init__(self, output, stage_durations=None):
            self.output = output
            self.stage_durations = stage_durations

    fake_pipeline_mod = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image",
        HiDreamO1ImagePipeline=object,
        DiffusionOutput=FakeDiffusionOutput,
        PATCH_SIZE=32,
        T_EPS=0.001,
        build_hidream_o1_scheduler=lambda **kwargs: ("fake-scheduler", kwargs),
    )
    captured["pipeline_module"] = fake_pipeline_mod

    class FakeRotary:
        inv_freq = None
        mrope_section = (2, 1, 1)

    class FakeTimestepEmbedder:
        timestep_embedding = staticmethod(lambda *a, **k: None)

    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer",
        HiDreamO1RotaryEmbedding=FakeRotary,
        TimestepEmbedder=FakeTimestepEmbedder,
    )
    captured["transformer_module"] = sys.modules[
        "vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer"
    ]

    # Engine dispatcher (patch/attention.py) - no-op fake for the registry.
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: None,
    )

    return captured


def _fresh_patch_module(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    patch_mod = _load_source_module(
        "hidream_patch_under_test",
        _repo_root() / "vllm_omni/platforms/npu/_310p/patch/hidream_o1_image.py",
    )
    patch_mod._APPLIED = False
    patch_mod._ATTENTION_GUARD_INSTALLED = False
    patch_mod._PIPELINE_PATCHED = False
    return patch_mod


def _install_fake_pipeline(captured: dict[str, object]) -> None:
    """Give the fake pipeline module a tiny real pipeline class so the
    patched forward can be executed end-to-end on CPU (2x2 patches, 4
    hidden steps, deterministic fake denoiser)."""
    mod = captured["pipeline_module"]

    class FakeScheduler:
        """Deterministic flow step: z <- z + v * dt (v = model_output)."""

        def __init__(self, num_steps: int):
            self.timesteps = torch.linspace(1000.0, 1000.0 / num_steps, num_steps)
            self.sigmas = (self.timesteps / 1000.0).clamp_min(0.001)
            self._i = 0

        def step(self, model_output, timestep, sample, return_dict=True):
            dt = self.sigmas[self._i]
            prev = sample.float() + model_output.float() * dt
            self._i += 1
            if return_dict:
                return SimpleNamespace(prev_sample=prev)
            return (prev,)

    def _build_hidream_o1_scheduler_310p(
        scheduler_name="default",
        num_inference_steps=1000,
        shift=3.0,
        device="cpu",
    ):
        assert str(device).startswith("cpu"), f"scheduler must be built on CPU, got {device}"
        return FakeScheduler(num_inference_steps)

    class FakePipeline:
        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.float16
            self.od_config = SimpleNamespace(parallel_config=SimpleNamespace(cfg_parallel_size=1))
            self._steps_done = 0

        def _resolve_generation_params(self, req):
            return (
                req.get("prompt", "p"),
                64,  # height -> 2x2 patches at PATCH_SIZE 32
                64,  # width
                req.get("steps", 4),
                req.get("seed", 42),
                req.get("guidance", 1.0),
            )

        def _build_and_validate_sample(self, prompt, height, width, expected_image_len):
            return {"prompt": prompt}

        def _prepare_noise_and_patchify(self, height, width, seed, dtype, device):
            gen = torch.Generator("cpu").manual_seed(seed + 1)
            return torch.randn(1, 4, 3072, generator=gen, dtype=dtype, device=device)

        def _forward_once(self, sample, z, timestep):
            self._steps_done += 1
            assert z.device == self.device
            return (z.float() * 1.5).to(self.dtype)

        def progress_bar(self, total=10):
            class _P:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def update(self, n=1):
                    pass

            return _P()

    mod.HiDreamO1ImagePipeline = FakePipeline
    mod.build_hidream_o1_scheduler = _build_hidream_o1_scheduler_310p


def _apply(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, dict[str, object]]:
    """Fresh fakes + fresh patch module + pipeline patch installation."""
    captured = _install_patch_fakes(monkeypatch)
    _install_fake_pipeline(captured)
    patch_mod = _load_source_module(
        "hidream_patch_apply_under_test",
        _repo_root() / "vllm_omni/platforms/npu/_310p/patch/hidream_o1_image.py",
    )
    patch_mod._APPLIED = False
    patch_mod._ATTENTION_GUARD_INSTALLED = False
    patch_mod._PIPELINE_PATCHED = False
    patch_mod._install_pipeline_patches()
    return patch_mod, captured


# ---------------------------------------------------------------------------
# FP16 pin
# ---------------------------------------------------------------------------


def test_pin_fp16_none_pinned(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(dtype=None)
    mod._pin_fp16(cfg)
    assert cfg.dtype == torch.float16


def test_pin_fp16_float16_passthrough(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(dtype=torch.float16)
    mod._pin_fp16(cfg)
    assert cfg.dtype == torch.float16


def test_pin_fp16_bf16_coerced(monkeypatch):
    """bf16 (the engine default on 310P, where no BF16 hardware exists) must
    be coerced to fp16, not rejected."""
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(dtype=torch.bfloat16)
    mod._pin_fp16(cfg)
    assert cfg.dtype == torch.float16


def test_pin_fp16_fp32_rejected(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(dtype=torch.float32)
    with pytest.raises(RuntimeError, match="FP16"):
        mod._pin_fp16(cfg)


# ---------------------------------------------------------------------------
# Attention selection gate
# ---------------------------------------------------------------------------


def test_attention_default_none_becomes_torch_sdpa(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(default_backend=None)
    backend = mod._validate_attention_selection(cfg)
    assert backend == "TORCH_SDPA"
    assert cfg.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_attention_mindiesd_default_rejected(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(default_backend="FLASH_ATTN")
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        mod._validate_attention_selection(cfg)


def test_attention_mindiesd_per_role_rejected(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(per_role={"denoise": "RAINFUSION_ATTN"})
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        mod._validate_attention_selection(cfg)


def test_attention_torch_sdpa_passthrough(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(default_backend="TORCH_SDPA")
    assert mod._validate_attention_selection(cfg) == "TORCH_SDPA"


# ---------------------------------------------------------------------------
# Cache backend gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cache_dit", "tea_cache", "mag_cache", "step_cache"])
def test_cache_backends_disabled(monkeypatch, backend):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(cache_backend=backend)
    mod._disable_numerical_cache_backends(cfg)
    assert cfg.cache_backend == "none"


@pytest.mark.parametrize("backend", [None, "none"])
def test_cache_none_untouched(monkeypatch, backend):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(cache_backend=backend)
    mod._disable_numerical_cache_backends(cfg)
    assert cfg.cache_backend == backend


def test_cache_unknown_left_alone(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    cfg = _make_od_config(cache_backend="future_backend")
    mod._disable_numerical_cache_backends(cfg)
    assert cfg.cache_backend == "future_backend"


# ---------------------------------------------------------------------------
# Quantization tier gate
# ---------------------------------------------------------------------------


def test_quant_none_ok(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    mod._validate_quantization(_make_od_config(quantization_config=None))


def test_quant_int8_admitted(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    mod._validate_quantization(_make_od_config(quantization_config=_make_fake_quant("int8")))


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4", "convrot_int4", "bitsandbytes", "bitsandbytesnf4"])
def test_quant_other_methods_rejected(monkeypatch, method):
    mod = _fresh_patch_module(monkeypatch)
    with pytest.raises(RuntimeError, match="rejects quantization method"):
        mod._validate_quantization(_make_od_config(quantization_config=_make_fake_quant(method)))


# ---------------------------------------------------------------------------
# Single-chip parallelism preflight
# ---------------------------------------------------------------------------


def test_parallel_tp1_ok(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    mod._validate_single_chip_parallelism(_make_od_config())


@pytest.mark.parametrize("attr", ["ulysses_degree", "ring_degree", "allgather_degree"])
def test_parallel_seq_parallel_rejected(monkeypatch, attr):
    mod = _fresh_patch_module(monkeypatch)
    with pytest.raises(RuntimeError, match="single-chip"):
        mod._validate_single_chip_parallelism(_make_od_config(**{attr: 2}))


def test_parallel_cfg_parallel_rejected(monkeypatch):
    mod = _fresh_patch_module(monkeypatch)
    with pytest.raises(RuntimeError, match="cfg_parallel_size"):
        mod._validate_single_chip_parallelism(_make_od_config(cfg_parallel_size=2))


# ---------------------------------------------------------------------------
# Pipeline patches: dtype gate, forward swap, CPU RoPE / timestep
# ---------------------------------------------------------------------------


def test_pipeline_dtype_gate_fp16(monkeypatch):
    _, captured = _apply(monkeypatch)
    mod = captured["pipeline_module"]
    pipe = mod.HiDreamO1ImagePipeline()
    pipe.dtype = torch.float16
    mod.HiDreamO1ImagePipeline._validate_static_config(pipe)  # must pass

    pipe.dtype = torch.bfloat16
    with pytest.raises(NotImplementedError, match="FP16-only"):
        mod.HiDreamO1ImagePipeline._validate_static_config(pipe)


def test_pipeline_dtype_gate_fp32_rejected(monkeypatch):
    _, captured = _apply(monkeypatch)
    mod = captured["pipeline_module"]
    pipe = mod.HiDreamO1ImagePipeline()
    pipe.dtype = torch.float32
    with pytest.raises(NotImplementedError, match="half precision"):
        mod.HiDreamO1ImagePipeline._validate_static_config(pipe)


def test_pipeline_forward_cpu_scheduler_roundtrip(monkeypatch):
    """End-to-end: patched forward with the fake scheduler - the scheduler
    must be built on CPU, the latent round-tripped per step, and same-seed
    runs bit-identical (POLICY section-1 level-3 reproducibility hook)."""
    _, captured = _apply(monkeypatch)
    mod = captured["pipeline_module"]

    pipe = mod.HiDreamO1ImagePipeline()
    out = pipe.forward({"prompt": "cat", "steps": 4, "seed": 42, "guidance": 1.0})
    z, h, w = out.output
    assert (h, w) == (64, 64)
    assert z.shape == (1, 4, 3072)
    assert z.device == torch.device("cpu")
    assert z.dtype == torch.float16
    assert pipe._steps_done == 4  # guidance <= 1.0: single forward/step

    out2 = pipe.forward({"prompt": "cat", "steps": 4, "seed": 42, "guidance": 1.0})
    assert torch.equal(z, out2.output[0]), "same-seed runs must be bit-identical"

    out3 = pipe.forward({"prompt": "cat", "steps": 4, "seed": 43, "guidance": 1.0})
    assert not torch.equal(z, out3.output[0]), "different seed must differ"


def test_pipeline_forward_sequential_cfg(monkeypatch):
    """guidance > 1.0 must take the two-forward sequential-CFG path."""
    _, captured = _apply(monkeypatch)
    mod = captured["pipeline_module"]

    pipe = mod.HiDreamO1ImagePipeline()
    out = pipe.forward({"prompt": "cat", "steps": 3, "seed": 7, "guidance": 5.0})
    assert pipe._steps_done == 2 * 3  # cond + uncond per step
    assert out.output[0].shape == (1, 4, 3072)


def test_rope_forward_cpu_table(monkeypatch):
    """The patched RoPE builds the REAL cos/sin table and returns it on the
    hidden-state device/dtype (012 caliber: REAL tensors uploaded to the NPU).
    Unit-rotation invariant: cos^2 + sin^2 == 1 elementwise."""
    _, captured = _apply(monkeypatch)
    tmod = captured["transformer_module"]

    class FakeSelf:
        inv_freq = torch.rand(4)  # head_dim 8 -> 4 frequencies
        mrope_section = (2, 1, 1)

    head = 8
    seq = 5
    hidden = torch.randn(1, seq, head, dtype=torch.float16)
    pos = torch.arange(seq).view(1, 1, -1).expand(3, 1, -1).contiguous()  # (3, 1, S)
    cos, sin = tmod.HiDreamO1RotaryEmbedding.forward(FakeSelf(), hidden, pos)
    assert cos.device == hidden.device
    assert cos.dtype == hidden.dtype
    assert cos.shape[-1] == head and sin.shape == cos.shape
    unit = cos.float() ** 2 + sin.float() ** 2
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-3)


def test_timestep_embedding_cpu_math(monkeypatch):
    """Patched timestep embedding: CPU math, same result on the target
    device, matching the reference sine feature formula."""
    _, captured = _apply(monkeypatch)
    tmod = captured["transformer_module"]

    t = torch.tensor([0.5, 0.9])
    emb = tmod.TimestepEmbedder.timestep_embedding(t, dim=8)
    assert emb.shape == (2, 8)
    assert emb.device == t.device
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, 4, dtype=torch.float32) / 4)
    args = t[:, None].float() * freqs[None]
    expect = torch.cat([args.cos(), args.sin()], dim=-1)
    assert torch.allclose(emb, expect, atol=1e-5)


# ---------------------------------------------------------------------------
# Registry dispatch (dispatcher loaded from source, model module faked in
# sys.modules - the vllm_omni package is never imported)
# ---------------------------------------------------------------------------


def _fresh_registry(monkeypatch):
    return _load_source_module(
        "registry_under_test",
        _repo_root() / "vllm_omni/platforms/npu/_310p/patch/__init__.py",
    )


def test_registry_dispatches_hidream(monkeypatch):
    """The dispatcher must route HiDreamO1ImagePipeline to our module and
    leave foreign archs untouched."""
    _fresh_patch_module(monkeypatch)  # installs the fakes
    registry = _fresh_registry(monkeypatch)

    calls = []

    class FakeModelModule:
        @staticmethod
        def apply_diffusion_310p_patches(od_config):
            calls.append(od_config)

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms.npu._310p.patch.hidream_o1_image",
        FakeModelModule,
    )
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "1")
    monkeypatch.setenv("VLLM_OMNI_310P_SKIP_WARMUP", "0")

    cfg = _make_od_config(model_class_name="HiDreamO1ImagePipeline")
    registry.apply_diffusion_310p_patches(cfg)
    assert len(calls) == 1 and calls[0] is cfg

    registry.apply_diffusion_310p_patches(_make_od_config(model_class_name="SomeOtherPipeline"))
    assert len(calls) == 1, "foreign archs must not be dispatched"


def test_registry_kill_switch(monkeypatch):
    """VLLM_OMNI_310P_DIFFUSION_PATCHES=0 (POLICY section-1 level-2 hook)
    must skip the whole dispatch."""
    _fresh_patch_module(monkeypatch)
    registry = _fresh_registry(monkeypatch)

    calls = []

    class FakeModelModule:
        @staticmethod
        def apply_diffusion_310p_patches(od_config):
            calls.append(od_config)

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms.npu._310p.patch.hidream_o1_image",
        FakeModelModule,
    )
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    registry.apply_diffusion_310p_patches(_make_od_config())
    assert calls == []


def test_apply_is_idempotent(monkeypatch):
    patch_mod, captured = _apply(monkeypatch)
    cfg = _make_od_config()
    patch_mod.apply_diffusion_310p_patches(cfg)
    mod = captured["pipeline_module"]
    forward_after_first = mod.HiDreamO1ImagePipeline.forward
    patch_mod.apply_diffusion_310p_patches(cfg)  # second call must be a no-op
    assert mod.HiDreamO1ImagePipeline.forward is forward_after_first


# ---------------------------------------------------------------------------
# Attention guard (platform selector wrap)
# ---------------------------------------------------------------------------


def test_attention_guard_forces_torch_sdpa(monkeypatch):
    patch_mod, captured = _apply(monkeypatch)
    patch_mod._install_attention_guard()
    platform = captured["platform"]
    enum = captured["enum"]
    assert platform.get_diffusion_attn_backend_cls(None, 128) == enum.TORCH_SDPA.get_path()


def test_attention_guard_rejects_mindiesd(monkeypatch):
    patch_mod, captured = _apply(monkeypatch)
    patch_mod._install_attention_guard()
    platform = captured["platform"]
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        platform.get_diffusion_attn_backend_cls("FLASH_ATTN", 128)
