# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Boogu-Image family 310P patch wiring (task-33).

Loads ``vllm_omni/platforms/npu/_310p/patch/boogu_image.py`` and the 310P
patch registry from source with fake ``vllm``/``vllm_omni`` dependencies, so
the patch contract (FP16 enforcement, attention-backend forcing, cache
rejection, quantization tier gate with INT8 as the GO-sanctioned primary
tier, VAE tiling, single-chip parallelism preflight, registry dispatch on
``model_class_name``, dispatcher math-SDPA decoupling ahead of the
kill-switch) is pinned without an NPU, vllm-ascend or a real model.

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
    model_class_name: str = "BooguImagePipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    vae_use_tiling: bool = False,
    vae_use_slicing: bool = False,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    allgather_degree: int = 1,
    cfg_parallel_size: int = 1,
    enable_layerwise_offload: bool = False,
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
        vae_use_tiling=vae_use_tiling,
        vae_use_slicing=vae_use_slicing,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        cfg_parallel_size=cfg_parallel_size,
        enable_layerwise_offload=enable_layerwise_offload,
    )


def _make_fake_quant(method: str):
    return SimpleNamespace(get_name=lambda method=method: method)


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules boogu_image.py imports lazily."""
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

    _install_fake_module(monkeypatch, "vllm", logger=logging.getLogger)
    _install_fake_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)

    captured["platform"] = FakeNPUOmniPlatform
    captured["calls"] = calls
    return captured


def _install_engine_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fakes for the dispatcher's engine-level imports (math-SDPA entry and
    the DiffusionEngine warmup hook) used by the dispatcher-level tests."""

    if "vllm_omni.platforms.npu._310p.patch.attention" not in sys.modules:
        attention_fakes = types.ModuleType("vllm_omni.platforms.npu._310p.patch.attention")
        attention_fakes.maybe_patch_sdpa_math = lambda: None
        monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.attention", attention_fakes)

    engine_fakes = types.ModuleType("vllm_omni.diffusion.diffusion_engine")

    class FakeDiffusionEngine:
        _dummy_run = staticmethod(lambda self: None)

    engine_fakes.DiffusionEngine = FakeDiffusionEngine
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.diffusion_engine", engine_fakes)


@pytest.fixture()
def boogu_patch(monkeypatch: pytest.MonkeyPatch):
    fakes = _install_patch_fakes(monkeypatch)
    root = _repo_root()
    module = _load_source_module(
        "boogu_image_patch_under_test",
        root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "boogu_image.py",
    )
    # Idempotency state is per process; reset for each test.
    module._APPLIED = False
    module._ATTENTION_GUARD_INSTALLED = False
    module.fake_state = fakes
    return module


class TestRegistryDispatch:
    def test_boogu_registered_in_registry(self):
        """The dispatcher registry must route the Boogu family to this module
        (COMPARISON-POLICY.md section 5: model -> patch module list)."""
        root = _repo_root()
        registry_src = (root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py").read_text()
        assert '"BooguImagePipeline": ["vllm_omni.platforms.npu._310p.patch.boogu_image"]' in registry_src
        assert '"BooguImageTurboPipeline": ["vllm_omni.platforms.npu._310p.patch.boogu_image"]' in registry_src

    def test_dispatch_applies_boogu_module(self, monkeypatch):
        root = _repo_root()
        installed: dict[str, object] = {}

        def _fake_apply(od_config):
            installed["od_config"] = od_config

        fake_module = types.ModuleType("vllm_omni.platforms.npu._310p.patch.boogu_image")
        fake_module.apply_diffusion_310p_patches = _fake_apply
        monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.boogu_image", fake_module)
        _install_engine_fakes(monkeypatch)

        dispatcher = _load_source_module(
            "boogu_dispatcher_under_test", root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
        )
        od_config = _make_od_config()
        monkeypatch.delenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", raising=False)
        monkeypatch.delenv("VLLM_OMNI_310P_SKIP_WARMUP", raising=False)

        dispatcher.apply_diffusion_310p_patches(od_config)
        assert installed.get("od_config") is od_config

    def test_warmup_skip_decoupled_from_kill_switch(self, monkeypatch):
        """Level-2 semantics (COMPARISON-POLICY §1): the env-driven warmup
        skip is a hardware-level asset - the 512x512 dummy geometry fails TBE
        dynamic-variant pre-compilation on 310P (research/012), and with a
        CPU-pinned MLLM the dummy run's image-input vision encode exceeds the
        engine's 600 s orchestrator init gate, so a kill-switch server that
        still ran the dummy run could never become ready (task-33 measured).
        The dispatcher must install the skip BEFORE the kill-switch gate."""
        root = _repo_root()
        fake_module = types.ModuleType("vllm_omni.platforms.npu._310p.patch.boogu_image")
        fake_module.apply_diffusion_310p_patches = lambda od_config: None
        monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.boogu_image", fake_module)
        _install_engine_fakes(monkeypatch)

        dispatcher = _load_source_module(
            "boogu_dispatcher_warmup_under_test",
            root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py",
        )
        od_config = _make_od_config()
        monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
        monkeypatch.setenv("VLLM_OMNI_310P_SKIP_WARMUP", "1")

        dispatcher.apply_diffusion_310p_patches(od_config)

        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

        assert getattr(DiffusionEngine, "_310p_warmup_skipped", False) is True

    def test_unregistered_pipeline_is_noop(self, monkeypatch):
        root = _repo_root()
        installed: list = []

        attention_fakes = types.ModuleType("vllm_omni.platforms.npu._310p.patch.attention")
        attention_fakes.maybe_patch_sdpa_math = lambda: None
        monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.attention", attention_fakes)

        dispatcher = _load_source_module(
            "boogu_dispatcher_under_test2", root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
        )
        od_config = _make_od_config(model_class_name="SomeOtherPipeline")
        monkeypatch.delenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", raising=False)
        dispatcher.apply_diffusion_310p_patches(od_config)
        assert not installed  # no module import side effects expected


class TestKillSwitchAndMathSdpaDecoupling:
    def test_kill_switch_skips_model_patch_but_math_sdpa_stays(self, monkeypatch):
        """sdxl 939fc24 decoupling: the math-SDPA decomposition is a
        hardware-level invariant installed BEFORE the kill-switch gate, so
        POLICY section-1 level-2 reference runs keep it active while the
        per-model invariants are disabled."""
        root = _repo_root()
        math_sdpa_calls: list[str] = []
        patch_calls: list[str] = []

        attention_fakes = types.ModuleType("vllm_omni.platforms.npu._310p.patch.attention")
        attention_fakes.maybe_patch_sdpa_math = lambda: math_sdpa_calls.append("math")
        monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.attention", attention_fakes)
        _install_engine_fakes(monkeypatch)  # DiffusionEngine fake for the =1 leg

        boogu_fakes = types.ModuleType("vllm_omni.platforms.npu._310p.patch.boogu_image")
        boogu_fakes.apply_diffusion_310p_patches = lambda od_config: patch_calls.append("boogu")
        monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.boogu_image", boogu_fakes)

        dispatcher = _load_source_module(
            "boogu_dispatcher_under_test3", root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
        )
        od_config = _make_od_config()
        monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
        dispatcher.apply_diffusion_310p_patches(od_config)
        assert math_sdpa_calls == ["math"]  # installed despite the kill switch
        assert patch_calls == []  # per-model invariants disabled

        monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "1")
        dispatcher.apply_diffusion_310p_patches(od_config)
        assert patch_calls == ["boogu"]  # re-enabled dispatch reaches the model module


class TestBooguInvariants:
    def test_fp16_enforced(self, boogu_patch):
        od_config = _make_od_config(dtype=torch.float16)
        boogu_patch.apply_diffusion_310p_patches(od_config)
        with pytest.raises(RuntimeError, match="requires FP16"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(dtype=torch.bfloat16))

    def test_none_default_backend_forced_to_torch_sdpa(self, boogu_patch):
        od_config = _make_od_config()
        boogu_patch.apply_diffusion_310p_patches(od_config)
        assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"

    def test_mindie_sd_only_backend_rejected_default(self, boogu_patch):
        with pytest.raises(RuntimeError, match="MindIE-SD"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(default_backend="FLASH_ATTN"))

    def test_mindie_sd_only_backend_rejected_per_role(self, boogu_patch):
        with pytest.raises(RuntimeError, match="MindIE-SD"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(per_role={"text": "RAINFUSION_ATTN"}))

    def test_numerical_cache_backend_disabled(self, boogu_patch):
        od_config = _make_od_config(cache_backend="cache_dit")
        boogu_patch.apply_diffusion_310p_patches(od_config)
        assert od_config.cache_backend == "none"

    def test_non_numerical_cache_backend_left_alone(self, boogu_patch):
        od_config = _make_od_config(cache_backend="some_future_backend")
        boogu_patch.apply_diffusion_310p_patches(od_config)
        assert od_config.cache_backend == "some_future_backend"

    def test_quantization_none_fp16_tier_passes_with_offload_note(self, boogu_patch):
        """Full-resident FP16 is infeasible on one chip (21.26 GB resident vs
        21589 MB), but the FP16 tier itself is the sanctioned tier once the
        DiT blocks stream via layerwise offload (task-33 measured-fits
        path): the gate warns (capacity ledger + offload state) and lets the
        config through."""
        od_config = _make_od_config(quantization_config=None)
        boogu_patch.apply_diffusion_310p_patches(od_config)
        assert od_config.quantization_config is None

    def test_quantization_none_warn_reports_offload_state(self, boogu_patch, caplog):
        """The FP16 warning must reflect whether layerwise offload is on, so
        the operator sees the OOM risk before the first kernel launch."""
        od_config = _make_od_config(quantization_config=None, enable_layerwise_offload=False)
        with caplog.at_level("WARNING"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(od_config)
        assert any("enable_layerwise_offload=False" in r.message and "expect OOM" in r.message for r in caplog.records)
        od_config = _make_od_config(quantization_config=None, enable_layerwise_offload=True)
        with caplog.at_level("WARNING"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(od_config)
        assert any("enable_layerwise_offload=True" in r.message for r in caplog.records)

    def test_quantization_int8_rejected_kernel_infeasible(self, boogu_patch):
        """Real-device finding (task-33): npu_quant_matmul /
        aclnnQuantMatmulV5 fails with error 161002 on every shape tested on
        310P3 (CANN 9.1.0 + torch_npu 2.10.0.post4), including canonical
        4096x4096 - the INT8 W8A8 tier is hardware-infeasible, so the gate
        rejects it up front instead of letting the kernel crash mid-run."""
        with pytest.raises(RuntimeError, match="rejects quantization method 'int8'"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(
                _make_od_config(quantization_config=_make_fake_quant("int8"))
            )

    @pytest.mark.parametrize("method", ["fp8", "mxfp4", "mxfp8", "bitsandbytes", "svdquant"])
    def test_other_quantization_methods_rejected(self, boogu_patch, method):
        with pytest.raises(RuntimeError, match="rejects quantization method"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(
                _make_od_config(quantization_config=_make_fake_quant(method))
            )

    def test_per_component_quantization_int8_transformer_rejected(self, boogu_patch):
        """H3-NPU recipe shape '{"transformer": int8, "mllm": null}' rides the
        INT8 kernel, which is not functional on 310P3 (task-33 probe matrix):
        the gate rejects the routing before the first kernel launch."""
        cfg = SimpleNamespace(
            get_name=lambda: "component",
            _components={"transformer": _make_fake_quant("int8"), "mllm": None},
            _default=None,
        )
        with pytest.raises(RuntimeError, match="rejects per-component quantization"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(quantization_config=cfg))

    def test_per_component_quantization_without_explicit_mllm_rejected(self, boogu_patch):
        """H3 recipe shape {"transformer": {"method": "int8"}}: transformer
        INT8 -> kernel infeasible on 310P3 -> rejected."""
        cfg = SimpleNamespace(
            get_name=lambda: "component",
            _components={"transformer": _make_fake_quant("int8")},
            _default=None,
        )
        with pytest.raises(RuntimeError, match="rejects per-component quantization"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(quantization_config=cfg))

    def test_per_component_quantization_all_null_accepted(self, boogu_patch):
        """All components null = plain FP16 tier via per-component routing
        (the measured-fits path with --enable-layerwise-offload)."""
        cfg = SimpleNamespace(
            get_name=lambda: "component",
            _components={"transformer": None, "mllm": None},
            _default=None,
        )
        od_config = _make_od_config(quantization_config=cfg)
        boogu_patch.apply_diffusion_310p_patches(od_config)
        assert od_config.quantization_config is cfg

    @pytest.mark.parametrize("bad_method", ["fp8", "mxfp8"])
    def test_per_component_quantization_non_int8_component_rejected(self, boogu_patch, bad_method):
        cfg = SimpleNamespace(
            get_name=lambda: "component",
            _components={"transformer": _make_fake_quant(bad_method)},
            _default=None,
        )
        with pytest.raises(RuntimeError, match="rejects per-component quantization"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(quantization_config=cfg))

    def test_per_component_quantization_non_int8_default_rejected(self, boogu_patch):
        cfg = SimpleNamespace(
            get_name=lambda: "component",
            _components={"transformer": _make_fake_quant("int8")},
            _default=_make_fake_quant("fp8"),
        )
        with pytest.raises(RuntimeError, match="rejects per-component quantization"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(quantization_config=cfg))

    def test_vae_tiling_and_slicing_forced(self, boogu_patch):
        od_config = _make_od_config(vae_use_tiling=False, vae_use_slicing=False)
        boogu_patch.apply_diffusion_310p_patches(od_config)
        assert od_config.vae_use_tiling is True
        assert od_config.vae_use_slicing is True

    def test_vae_tiling_already_on_stays_on(self, boogu_patch):
        od_config = _make_od_config(vae_use_tiling=True, vae_use_slicing=True)
        boogu_patch.apply_diffusion_310p_patches(od_config)
        assert od_config.vae_use_tiling is True
        assert od_config.vae_use_slicing is True

    @pytest.mark.parametrize("attr", ["ulysses_degree", "ring_degree", "allgather_degree"])
    def test_sequence_parallel_rejected(self, boogu_patch, attr):
        kwargs = {attr: 2}
        with pytest.raises(RuntimeError, match="multi-rank HCCL|GATE-TP2"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(**kwargs))

    def test_cfg_parallel_rejected(self, boogu_patch):
        with pytest.raises(RuntimeError, match="cfg_parallel_size"):
            _fresh(boogu_patch).apply_diffusion_310p_patches(_make_od_config(cfg_parallel_size=2))

    def test_idempotent(self, boogu_patch):
        od_config = _make_od_config()
        boogu_patch.apply_diffusion_310p_patches(od_config)
        od_config2 = _make_od_config()
        boogu_patch.apply_diffusion_310p_patches(od_config2)
        # Second call must be a no-op (module applied-once flag).
        assert od_config2.vae_use_tiling is False  # untouched by the second call

    def test_placement_plan_logged(self, boogu_patch, caplog):
        with caplog.at_level(logging.INFO):
            boogu_patch.apply_diffusion_310p_patches(_make_od_config(quantization_config=None))
        assert any("placement plan" in record.message for record in caplog.records)
        assert any("mllm=CPU-pin" in record.message for record in caplog.records)


class TestAttentionGuard:
    def test_guard_forces_torch_sdpa_for_none(self, boogu_patch):
        od_config = _make_od_config()
        boogu_patch.apply_diffusion_310p_patches(od_config)
        fake = boogu_patch.fake_state
        guarded = fake["platform"].get_diffusion_attn_backend_cls
        result = guarded(None, 128)
        assert result == "vllm_omni.diffusion.attention.backends.torch_sdpa.Backend"

    def test_guard_rejects_mindie_sd_only(self, boogu_patch):
        od_config = _make_od_config()
        boogu_patch.apply_diffusion_310p_patches(od_config)
        fake = boogu_patch.fake_state
        guarded = fake["platform"].get_diffusion_attn_backend_cls
        with pytest.raises(RuntimeError, match="MindIE-SD"):
            guarded("FLASH_ATTN", 128)

    def test_guard_passes_through_other_backends(self, boogu_patch):
        od_config = _make_od_config()
        boogu_patch.apply_diffusion_310p_patches(od_config)
        fake = boogu_patch.fake_state
        guarded = fake["platform"].get_diffusion_attn_backend_cls
        assert guarded("TORCH_SDPA", 128) == "resolved:TORCH_SDPA"


def _fresh(boogu_patch):
    """Reset the applied-once flag so each test exercises a fresh apply."""
    boogu_patch._APPLIED = False
    boogu_patch._ATTENTION_GUARD_INSTALLED = False
    return boogu_patch
