# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the FLUX.2-klein 310P patch wiring.

Loads ``vllm_omni/platforms/npu/_310p/patch/flux2_klein.py`` and the 310P
patch registry from source with fake ``vllm``/``vllm_omni`` dependencies, so
the patch contract (FP16 enforcement, attention-backend forcing, cache
disabling, quant/LoRA guards, VAE tiling, registry dispatch on
``model_class_name``) is pinned without an NPU, vllm-ascend or a real model.

Mirrors the fakes-first convention of :mod:`tests.platforms.npu.test_310p_patches`.
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
    model_class_name: str = "Flux2KleinPipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    lora_path: object = None,
    tf_params: dict | None = None,
    ulysses_degree: int = 1,
):
    tf_params = dict(tf_params or {})
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
        tf_model_config=SimpleNamespace(params=tf_params, get=lambda key, default=None: tf_params.get(key, default)),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            sequence_parallel_size=None,
            ulysses_degree=ulysses_degree,
            ring_degree=1,
            allgather_degree=1,
        ),
    )


class _FakeVAE:
    def __init__(self, *, with_switches: bool = True):
        self.with_switches = with_switches
        self.calls: list[str] = []

    def enable_tiling(self):
        if not self.with_switches:
            raise AttributeError("no tiling")
        self.calls.append("enable_tiling")

    def enable_slicing(self):
        if not self.with_switches:
            raise AttributeError("no slicing")
        self.calls.append("enable_slicing")


def _make_fake_pipeline_module(monkeypatch: pytest.MonkeyPatch, vae: object) -> object:
    class FakeEncoder:
        def __init__(self):
            # The pipeline __init__ places components on the local chip.
            self.device = "npu"

        def to(self, device):
            self.device = device
            return self

    class FakeFlux2KleinPipeline:
        def __init__(self):
            self.vae = vae
            self.text_encoder = FakeEncoder()

    def _fake_from_pf(cls_fn, *args, **kwargs):
        return FakeEncoder()

    return _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein",
        Flux2KleinPipeline=FakeFlux2KleinPipeline,
        from_pretrained_with_prefetch=_fake_from_pf,
    )


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules flux2_klein.py imports lazily."""
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

    fake_platform_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu.platform",
        NPUOmniPlatform=FakeNPUOmniPlatform,
    )

    _install_fake_module(monkeypatch, "vllm", logger=logging.getLogger)
    _install_fake_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)

    # Default fake pipeline module: apply_diffusion_310p_patches always wraps
    # Flux2KleinPipeline.__init__ (VAE tiling), which lazily imports this
    # module.  Tests with a specific VAE install their own fake first, which
    # wins via sys.modules overwrite.
    _make_fake_pipeline_module(monkeypatch, _FakeVAE())

    captured["platform_module"] = fake_platform_module
    captured["backend_calls"] = calls
    return captured


def _load_flux2_klein_patch(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    module = _load_source_module("flux2_klein_310p_under_test", patch_dir / "flux2_klein.py")
    # Fresh module globals per test: reset the idempotency flags.
    module._APPLIED = False
    module._ATTENTION_GUARD_INSTALLED = False
    module._VAE_TILING_WRAPPED = False
    return module


def test_fp16_is_accepted_and_config_normalized(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.float16)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default is not None
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert od_config.cache_backend is None


def test_bf16_dtype_is_rejected_on_310p(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="FP16"):
        module.apply_diffusion_310p_patches(_make_od_config(dtype=torch.bfloat16))


def test_explicit_mindiesd_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(default_backend="FLASH_ATTN"))

    module._APPLIED = False
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(per_role={"self": "RAINFUSION_ATTN"}))


def test_user_selected_sdpa_is_preserved(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    od_config = _make_od_config(default_backend="TORCH_SDPA")
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_numerical_cache_backends_are_disabled(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    for backend in ("cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache"):
        module._APPLIED = False
        od_config = _make_od_config(cache_backend=backend)
        module.apply_diffusion_310p_patches(od_config)
        assert od_config.cache_backend == "none", backend


def test_quantization_int8_accepted_as_second_tier(monkeypatch: pytest.MonkeyPatch, caplog):
    """POLICY section 2: INT8 W8A8 is the sanctioned fallback tier
    (research/009) - accepted with a loud warning, no raise."""
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    with caplog.at_level(logging.WARNING):
        module.apply_diffusion_310p_patches(
            _make_od_config(quantization_config=SimpleNamespace(name="int8"))
        )
    assert "second tier" in caplog.text
    assert "int8" in caplog.text.lower()


def test_quantization_non_int8_rejected(monkeypatch: pytest.MonkeyPatch):
    """FP8/MXFP* give no 310P hardware benefit (research/009) -> rejected."""
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    for method in ("fp8", "mxfp8", "convrot_int4"):
        module._APPLIED = False
        with pytest.raises(RuntimeError, match="rejects quantization"):
            module.apply_diffusion_310p_patches(
                _make_od_config(quantization_config=SimpleNamespace(name=method))
            )


def test_lora_warns(monkeypatch: pytest.MonkeyPatch, caplog):
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    with caplog.at_level(logging.WARNING):
        module.apply_diffusion_310p_patches(_make_od_config(lora_path="/tmp/lora"))
    assert "lora" in caplog.text.lower()


def test_ulysses_head_divisibility_preflight(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    # 24 heads / ulysses 3 -> OK.
    module.apply_diffusion_310p_patches(_make_od_config(tf_params={"num_attention_heads": 24}, ulysses_degree=3))
    # 24 heads / ulysses 7 -> rejected.
    module._APPLIED = False
    with pytest.raises(RuntimeError, match="ulysses_degree"):
        module.apply_diffusion_310p_patches(_make_od_config(tf_params={"num_attention_heads": 24}, ulysses_degree=7))


def test_platform_guard_forces_sdpa_and_blocks_mindiesd(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())
    platform = captured["platform_module"].NPUOmniPlatform
    expected = captured["enum"].TORCH_SDPA.get_path()

    # User did not select a backend: the guard must return the SDPA path
    # without consulting the original (mindiesd-sniffing) selector.
    assert platform.get_diffusion_attn_backend_cls(None, 128) == expected
    assert captured["backend_calls"] == []

    # An explicit MindIE-SD-only backend raises before reaching the original.
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        platform.get_diffusion_attn_backend_cls("FLASH_ATTN", 128)
    assert captured["backend_calls"] == []

    # A non-MindIE-SD explicit selection falls through to the original.
    assert platform.get_diffusion_attn_backend_cls("TORCH_SDPA", 128) == "resolved:TORCH_SDPA"
    assert captured["backend_calls"] == [("TORCH_SDPA", 128)]


def test_vae_tiling_is_forced_at_pipeline_init(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    vae = _FakeVAE()
    pipeline_module = _make_fake_pipeline_module(monkeypatch, vae)
    module = _load_flux2_klein_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())

    pipeline = pipeline_module.Flux2KleinPipeline()
    assert vae.calls == ["enable_tiling", "enable_slicing"]
    assert pipeline._310p_vae_tiling is True


def test_vae_tiling_missing_api_warns_but_boots(monkeypatch: pytest.MonkeyPatch, caplog):
    _install_patch_fakes(monkeypatch)
    vae = _FakeVAE(with_switches=False)
    pipeline_module = _make_fake_pipeline_module(monkeypatch, vae)
    module = _load_flux2_klein_patch(monkeypatch)

    with caplog.at_level(logging.WARNING):
        module.apply_diffusion_310p_patches(_make_od_config())

    pipeline = pipeline_module.Flux2KleinPipeline()
    assert vae.calls == []
    assert pipeline._310p_vae_tiling is False
    assert "tiling" in caplog.text.lower()


def test_encoder_cpu_downgrade_env_switch(monkeypatch: pytest.MonkeyPatch):
    """Chain-1 OOM downgrade (lead-fixed order): FLUX2KLEIN_310P_ENCODER_CPU=1
    pins the Qwen3 text encoder on host memory after pipeline construction."""
    _install_patch_fakes(monkeypatch)
    pipeline_module = _make_fake_pipeline_module(monkeypatch, _FakeVAE())
    module = _load_flux2_klein_patch(monkeypatch)
    monkeypatch.setenv("FLUX2KLEIN_310P_ENCODER_CPU", "1")

    module.apply_diffusion_310p_patches(_make_od_config())

    pipeline = pipeline_module.Flux2KleinPipeline()
    assert pipeline.text_encoder.device == "cpu"
    assert pipeline._310p_encoder_cpu is True


def test_encoder_stays_on_chip_by_default(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    pipeline_module = _make_fake_pipeline_module(monkeypatch, _FakeVAE())
    module = _load_flux2_klein_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())

    pipeline = pipeline_module.Flux2KleinPipeline()
    assert pipeline.text_encoder.device == "npu"
    assert not getattr(pipeline, "_310p_encoder_cpu", False)


def test_patch_is_idempotent_per_process(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_flux2_klein_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())
    assert module._APPLIED is True

    # A second config (e.g. a stage restart in the same worker process) must
    # not double-install the guard or re-validate; the module flag is the
    # authority, matching the _WORKER_PATCHED registry convention.
    second = _make_od_config(default_backend="FLASH_ATTN")
    module.apply_diffusion_310p_patches(second)  # must not raise (already applied)
    assert second.diffusion_attention_config.default.backend == "FLASH_ATTN"
    assert captured["platform_module"].NPUOmniPlatform.get_diffusion_attn_backend_cls.__name__ == "_guarded"


def _install_parent_fakes(monkeypatch: pytest.MonkeyPatch):
    """Pre-seed the whole import chain the registry dispatches through.

    The registry does ``importlib.import_module("vllm_omni.platforms.npu.
    _310p.patch.flux2_klein")`` at call time; every dotted parent must
    already be in ``sys.modules`` so the real (vllm-dependent) packages are
    never touched.
    """
    for name in (
        "vllm_omni",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
        "vllm_omni.platforms.npu._310p.patch",
    ):
        _install_fake_module(monkeypatch, name)


def test_registry_dispatches_by_model_class_name(monkeypatch: pytest.MonkeyPatch):
    _install_parent_fakes(monkeypatch)
    _install_patch_fakes(monkeypatch)
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    registry = _load_source_module("flux2_klein_310p_registry_under_test", patch_dir / "__init__.py")
    module = _load_source_module("flux2_klein_310p_under_test2", patch_dir / "flux2_klein.py")
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.flux2_klein", module)

    assert "Flux2KleinPipeline" in registry._DIFFUSION_310P_PATCH_MODULES
    assert registry._DIFFUSION_310P_PATCH_MODULES["Flux2KleinPipeline"] == [
        "vllm_omni.platforms.npu._310p.patch.flux2_klein"
    ]

    # Flux2KleinPipeline -> dispatch applies the registered module's patches.
    od_config = _make_od_config(model_class_name="Flux2KleinPipeline")
    registry.apply_diffusion_310p_patches(od_config)
    assert od_config.diffusion_attention_config.default is not None
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"

    # Unregistered pipeline arch: silent no-op (runs the generic 310P path),
    # no od_config mutation.
    unknown = _make_od_config(model_class_name="SomeFuturePipeline", dtype=torch.bfloat16)
    registry.apply_diffusion_310p_patches(unknown)
    assert unknown.dtype is torch.bfloat16
    assert unknown.diffusion_attention_config.default is None

    # Re-dispatch in the same process is a no-op (module idempotency flag).
    repeat = _make_od_config(model_class_name="Flux2KleinPipeline", dtype=torch.bfloat16)
    registry.apply_diffusion_310p_patches(repeat)
    assert repeat.dtype is torch.bfloat16

class TestEncoderCpuGuard:
    """Chain-1 (FLUX2KLEIN_310P_ENCODER_CPU=1): the Qwen3 text encoder never
    touches the chip; the prompt-embeds forward is retargeted to the encoder
    device and the tiny embeds tensor is moved back to the target."""

    def test_cpu_pinned_encoder_absorbs_to_and_forwards(self, monkeypatch):
        _install_patch_fakes(monkeypatch)
        module = _load_flux2_klein_patch(monkeypatch)
        proxy_cls = module._CpuPinnedEncoder

        class FakeMod:
            dtype = "fp16"

            def forward(self, x):
                return "enc(" + x + ")"

            def __call__(self, x):
                return "enc(" + x + ")"

        m = FakeMod()
        proxy = proxy_cls(m)
        assert proxy.to("npu:0") is proxy
        assert proxy.to("cpu", dtype="fp32") is proxy
        assert proxy.dtype == "fp16"
        assert proxy.forward("a") == "enc(a)"
        assert proxy("b") == "enc(b)"

    def test_retarget_decision_matrix(self, monkeypatch):
        _install_patch_fakes(monkeypatch)
        module = _load_flux2_klein_patch(monkeypatch)

        cpu, npu = SimpleNamespace(type="cpu"), SimpleNamespace(type="npu")

        # env off -> never retarget
        monkeypatch.delenv(module._ENCODER_CPU_ENV, raising=False)
        assert module._encoder_cpu_retarget(cpu, npu) is npu
        assert module._encoder_cpu_retarget(None, npu) is npu

        # env on + encoder on cpu + target npu -> run encoder on host
        monkeypatch.setenv(module._ENCODER_CPU_ENV, "1")
        assert module._encoder_cpu_retarget(cpu, npu) is cpu
        # encoder on npu (not the chain-1 situation) -> keep target
        assert module._encoder_cpu_retarget(npu, npu) is npu
        # target already cpu -> keep
        assert module._encoder_cpu_retarget(cpu, cpu) is cpu
        # no encoder device info -> keep target
        assert module._encoder_cpu_retarget(None, npu) is npu
        monkeypatch.delenv(module._ENCODER_CPU_ENV)

    def test_guard_install_is_env_gated(self, monkeypatch):
        """apply() must not touch the pipeline module unless chain-1 is armed
        (fakes-first discipline: no real vllm_omni import in CPU tests)."""
        monkeypatch.delenv("FLUX2KLEIN_310P_ENCODER_CPU", raising=False)
        _install_patch_fakes(monkeypatch)
        module = _load_flux2_klein_patch(monkeypatch)
        od_config = _make_od_config()
        module.apply_diffusion_310p_patches(od_config)
        assert module._ENCODER_CPU_GUARDED is False
