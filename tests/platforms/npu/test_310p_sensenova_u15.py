# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the SenseNova-U1.5-8B-MoT 310P patch wiring.

Loads ``vllm_omni/platforms/npu/_310p/patch/sensenova_u15.py`` and the 310P
patch registry from source with fake ``vllm``/``vllm_omni`` dependencies, so
the patch contract (FP16 enforcement, INT8 tier acceptance, single-chip
FP16 over-budget rejection, attention-backend forcing, cache disabling,
CPU build pin, post-load vision CPU pins, registry dispatch on
``model_class_name``) is pinned without an NPU, vllm-ascend or a real
model.  The schedule/denoising arithmetic rewrites additionally get
bit-exact CPU parity tests against the upstream fp32 op sequence.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_flux2_klein` (task-17 keeps the
convergent design of the SDXL/fluxklein 310P branches).
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
    model_class_name: str = "SenseNovaU1Pipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    lora_path: object = None,
    tp: int = 1,
    sp: int | None = None,
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
        lora_path=lora_path,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            sequence_parallel_size=sp,
            ulysses_degree=1,
            ring_degree=1,
            allgather_degree=1,
            vae_patch_parallel_size=1,
            vae_parallel_mode="none",
        ),
    )


class _FakeModule:
    """Tracks .to(device) calls (mimics an nn.Module for placement tests)."""

    def __init__(self, device: str = "npu:0"):
        self.device = device

    def to(self, device):
        self.device = device
        return self


class _FakeVision:
    def __init__(self):
        self.vision_model = _FakeModule()
        self.fm_modules = {"vision_model_mot_gen": _FakeModule()}


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules sensenova_u15.py imports lazily."""
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

    # version.py's warn_if_misaligned_vllm_version() does
    # ``from vllm import __version__, __version_tuple__`` - the fake must carry
    # both (a bare fake module without the names raises ImportError, not
    # ModuleNotFoundError, so the version guard's except clause misses it).
    _install_fake_module(
        monkeypatch,
        "vllm",
        logger=logging.getLogger,
        __version__="0.32.0",
        __version_tuple__=(0, 32, 0),
    )
    _install_fake_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)

    # Fake pipeline module (base class + re-export target).
    class FakeSenseNovaU1Pipeline:
        def __init__(self, od_config=None):
            self.od_config = od_config

    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1",
        SenseNovaU1Pipeline=FakeSenseNovaU1Pipeline,
        get_sensenova_u1_post_process_func=lambda od_config: (lambda x: x),
    )
    captured["fake_pipeline_module"] = sys.modules["vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1"]

    # Fake loader module: initialize_model records the default device it is
    # called under; DiffusersPipelineLoader tracks post-load placement calls.
    devices_seen: list[str] = []

    def orig_initialize_model(od_config):
        devices_seen.append(str(torch.get_default_device()))
        return "pipeline"

    post_load_calls: list[tuple] = []

    class FakeDiffusersPipelineLoader:
        def _process_weights_after_loading(self, model, target_device):
            post_load_calls.append((id(self), model, str(target_device)))

    loader_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.model_loader.diffusers_loader",
        initialize_model=orig_initialize_model,
        DiffusersPipelineLoader=FakeDiffusersPipelineLoader,
    )
    captured["loader_module"] = loader_module
    captured["orig_initialize_model"] = orig_initialize_model
    captured["devices_seen"] = devices_seen
    captured["post_load_calls"] = post_load_calls

    # Fake diffusion registry: captures register_diffusion_model calls.
    registry_calls: list[dict] = []

    def fake_register_diffusion_model(
        model_arch,
        module_name,
        class_name,
        pre_process_func_name=None,
        post_process_func_name=None,
        ir_op_priority_func_name=None,
        action_post_process_func_name=None,
        prefix_cache_func_name=None,
    ):
        registry_calls.append(
            dict(
                model_arch=model_arch,
                module_name=module_name,
                class_name=class_name,
                post_process_func_name=post_process_func_name,
            )
        )

    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.registry",
        register_diffusion_model=fake_register_diffusion_model,
    )
    captured["registry_calls"] = registry_calls

    # Fake SDPA backend module for the shared attention patch.
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.attention.backends.sdpa", _sdpa_attention=lambda *a: a)

    return captured


def _load_sensenova_patch(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    module = _load_source_module("sensenova_u15_310p_under_test", patch_dir / "sensenova_u15.py")
    # Fresh module globals per test: reset the idempotency flags.
    module._APPLIED = False
    module._LOADER_CPU_BUILD_INSTALLED = False
    module._VISION_PIN_INSTALLED = False
    module._ATTENTION_GUARD_INSTALLED = False
    return module


def _int8_config() -> SimpleNamespace:
    return SimpleNamespace(name="int8")


# ---------------------------------------------------------------------------
# dtype / quantization / attention gates
# ---------------------------------------------------------------------------


def test_fp16_and_int8_are_accepted(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.float16, quantization_config=_int8_config())
    with caplog.at_level(logging.WARNING):
        module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default is not None
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert "second tier" not in caplog.text  # not an accident here: documented primary tier
    assert "int8" in caplog.text.lower()


def test_bf16_dtype_is_rejected_on_310p(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="FP16"):
        module.apply_diffusion_310p_patches(_make_od_config(dtype=torch.bfloat16))


def test_explicit_mindiesd_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(
            _make_od_config(default_backend="FLASH_ATTN", quantization_config=_int8_config())
        )

    module._APPLIED = False
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(
            _make_od_config(per_role={"self": "RAINFUSION_ATTN"}, quantization_config=_int8_config())
        )


def test_user_selected_sdpa_is_preserved(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    od_config = _make_od_config(default_backend="TORCH_SDPA", quantization_config=_int8_config())
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_numerical_cache_backends_are_disabled(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    for backend in ("cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache"):
        module._APPLIED = False
        od_config = _make_od_config(cache_backend=backend, quantization_config=_int8_config())
        module.apply_diffusion_310p_patches(od_config)
        assert od_config.cache_backend == "none", backend


def test_quantization_non_int8_rejected(monkeypatch: pytest.MonkeyPatch):
    """FP8/MXFP* give no 310P hardware benefit (research/009) -> rejected."""
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    for method in ("fp8", "mxfp8", "convrot_int4"):
        module._APPLIED = False
        with pytest.raises(RuntimeError, match="rejects quantization"):
            module.apply_diffusion_310p_patches(_make_od_config(quantization_config=SimpleNamespace(name=method)))


def test_single_chip_fp16_without_quant_is_rejected(monkeypatch: pytest.MonkeyPatch):
    """17.53B / 35.07 GB cannot run FP16 on one 310P chip (21589 MB budget):
    fail at patch time instead of mid-load OOM."""
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="21589"):
        module.apply_diffusion_310p_patches(_make_od_config(tp=1, sp=None))


def test_multi_chip_without_quant_is_not_patched_rejected(monkeypatch: pytest.MonkeyPatch):
    """A 2-chip FP16 split is a separate validation scope (TP1-only GO); the
    patch must not hard-fail it, only the quant tiers are gated."""
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config(tp=2, sp=None))
    assert module._APPLIED is True


# ---------------------------------------------------------------------------
# placement / loader wraps
# ---------------------------------------------------------------------------


def test_loader_initialize_model_pinned_to_cpu(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_int8_config()))

    loader_module = captured["loader_module"]
    assert loader_module.initialize_model is not captured["orig_initialize_model"]
    loader_module.initialize_model(_make_od_config())
    assert captured["devices_seen"] == ["cpu"]
    assert getattr(loader_module, "_310p_cpu_build_patched", False) is True


def test_vision_pins_run_after_weight_load(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_OMNI_310P_SENSENOVA_VISION_CPU", "1")
    monkeypatch.setenv("VLLM_OMNI_310P_SENSENOVA_MOTGEN_CPU", "1")
    captured = _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_int8_config()))

    loader_cls = captured["loader_module"].DiffusersPipelineLoader
    model = _FakeVision()
    loader_cls._process_weights_after_loading(object(), model, "npu:0")

    assert len(captured["post_load_calls"]) == 1
    assert model.vision_model.device == "cpu"
    assert model.fm_modules["vision_model_mot_gen"].device == "cpu"


def test_vision_pins_absent_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_OMNI_310P_SENSENOVA_VISION_CPU", raising=False)
    monkeypatch.delenv("VLLM_OMNI_310P_SENSENOVA_MOTGEN_CPU", raising=False)
    captured = _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_int8_config()))

    loader_cls = captured["loader_module"].DiffusersPipelineLoader
    # No env var: the post-load method must be the untouched original.
    assert captured["post_load_calls"] == []
    model = _FakeVision()
    loader_cls._process_weights_after_loading(object(), model, "npu:0")
    assert len(captured["post_load_calls"]) == 1
    assert model.vision_model.device == "npu:0"


def test_platform_guard_forces_sdpa_and_blocks_mindiesd(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_int8_config()))
    platform = captured["loader_module"].__dict__.get("NPUOmniPlatform")  # not captured; use fake module
    fake_platform_module = sys.modules["vllm_omni.platforms.npu.platform"]
    platform = fake_platform_module.NPUOmniPlatform
    expected = captured["enum"].TORCH_SDPA.get_path()

    assert platform.get_diffusion_attn_backend_cls(None, 128) == expected
    assert captured["loader_module"] is not None  # sanity: fakes installed

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        platform.get_diffusion_attn_backend_cls("FLASH_ATTN", 128)

    assert platform.get_diffusion_attn_backend_cls("TORCH_SDPA", 128) == "resolved:TORCH_SDPA"


# ---------------------------------------------------------------------------
# registry dispatch
# ---------------------------------------------------------------------------


def test_registry_dispatch_resolves_sensenova_module(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_FORCE_MATH_SDPA", "0")  # keep torch untouched

    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    registry = _load_source_module("registry_under_test", patch_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.sensenova_u15", module)
    # The registry's attention import would pull the real vllm_omni package
    # (needs vllm/aenum); fake the leaf module instead.
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: None,
    )

    registry.apply_diffusion_310p_patches(_make_od_config(quantization_config=_int8_config()))
    assert module._APPLIED is True


def test_registry_kill_switch_disables_dispatch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")

    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    registry = _load_source_module("registry_under_test2", patch_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.sensenova_u15", module)

    registry.apply_diffusion_310p_patches(_make_od_config(quantization_config=_int8_config()))
    assert module._APPLIED is False


# ---------------------------------------------------------------------------
# model registration (platform.py-untouched design)
# ---------------------------------------------------------------------------


def test_registry_registers_310p_subclass(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_int8_config()))

    assert len(captured["registry_calls"]) == 1
    reg = captured["registry_calls"][0]
    assert reg["model_arch"] == "SenseNovaU1Pipeline"
    assert reg["class_name"] == "_SenseNovaU15Pipeline310P"
    assert reg["post_process_func_name"] == "get_sensenova_u1_post_process_func"

    base_cls = captured["fake_pipeline_module"].SenseNovaU1Pipeline
    cls = module._SenseNovaU15Pipeline310P
    assert issubclass(cls, base_cls)
    mro = [c.__name__ for c in cls.__mro__]
    assert mro.index("_SenseNovaU15Overrides") < mro.index("FakeSenseNovaU1Pipeline")


# ---------------------------------------------------------------------------
# arithmetic parity (bit-exact against the upstream fp32 op sequence)
# ---------------------------------------------------------------------------


def test_apply_time_schedule_is_bit_exact_with_upstream(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_sensenova_patch(monkeypatch)

    class _Host:
        pass

    host = _Host()
    for shift in (1.0, 3.0, 5.0):
        t = torch.linspace(0.0, 1.0, 64).float()
        got = module._SenseNovaU15Overrides._apply_time_schedule(host, t, 1024, shift)
        sigma = 1.0 - t.cpu().float()
        ref = 1.0 - (shift * sigma / (1.0 + (shift - 1.0) * sigma))
        assert torch.equal(got.cpu(), ref), shift


def test_predict_v_denominator_is_bit_exact_with_upstream(monkeypatch: pytest.MonkeyPatch):
    """The 310P denominator rewrite (CPU float -> fp32 scalar tensor) must be
    bit-identical to the upstream ``(x_pred - z) / (1 - t).clamp_min(t_eps)``
    (fp32 0-dim denominator): same fp32 denominator value, and ATen keeps the
    division result FP16 in both forms (a 0-dim scalar does not promote the
    fp16 numerator)."""
    _install_patch_fakes(monkeypatch)

    torch.manual_seed(0)
    z = torch.randn(1, 64, 12, dtype=torch.float16)
    x_pred = torch.randn_like(z)
    t_eps = 0.02
    for t_val in (0.0, 0.02, 0.5, 0.98, 1.0):
        t = torch.tensor(t_val, dtype=torch.float32)
        upstream = (x_pred - z) / (1 - t).clamp_min(t_eps)

        t_f = float(t)
        denom_t = torch.tensor(max(1.0 - t_f, float(t_eps)), dtype=torch.float32)
        rewritten = (x_pred - z) / denom_t

        assert torch.equal(upstream, rewritten), t_val
        assert upstream.dtype == torch.float16  # ATen scalar rule: no promotion
        assert rewritten.dtype == torch.float16


def test_euler_update_delta_is_bit_exact_with_upstream(monkeypatch: pytest.MonkeyPatch):
    """The ``(t_next - t)`` rewrite (CPU floats -> fp32 (1,) tensor) must be
    bit-identical to the upstream fp32 0-dim Sub."""
    _install_patch_fakes(monkeypatch)

    ts = torch.linspace(0.0, 1.0, 51).float()
    for i in range(50):
        upstream_dt = ts[i + 1] - ts[i]  # fp32 0-dim
        dt_f = float(ts[i + 1]) - float(ts[i])
        rewritten_dt = torch.tensor(dt_f, dtype=torch.float32)
        assert torch.equal(upstream_dt, rewritten_dt), i
