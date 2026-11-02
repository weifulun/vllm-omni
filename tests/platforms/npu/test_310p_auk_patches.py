# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the AuK 310P diffusion patch.

The tests load the patch module from source with fake AuK pipeline
dependencies (no real model weights, no NPU kernels), so they validate the
patch contract the way ``test_310p_patches.py`` does for the Qwen3-TTS line:

* the deploy dtype is forced to FP16 before the pipeline is constructed;
* the 310P pipeline subclass forces both CUDA-graph wrappers off;
* the NPU-only KV-cache quant knob is neutralized;
* the registry dispatch routes ``AuKPipeline`` to this module (and only this
  module) through the shared ``apply_diffusion_310p_patches`` entry point.
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


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _FakeAuKVAE:
    pass


class _FakeAuKPipeline(torch.nn.Module):
    """Minimal stand-in for AuKPipeline: records config, exposes the graph knobs."""

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.dtype = getattr(od_config, "dtype", None) or torch.bfloat16
        self.device = torch.device("cpu")
        eager = bool(getattr(od_config, "enforce_eager", True))
        # The base pipeline builds CUDA-graph wrappers enabled=not enforce_eager;
        # mirror that so the 310P subclass's forced-off behaviour is observable.
        self.cudagraph_wrapper = SimpleNamespace(enabled=not eager)
        self.vae_decode = SimpleNamespace(enabled=not eager)


def _install_auk_fakes(monkeypatch: pytest.MonkeyPatch):
    """Install fake parents so the source-loaded patch module imports cleanly."""
    for name in (
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.models",
        "vllm_omni.diffusion.models.auk",
    ):
        monkeypatch.setitem(sys.modules, name, _install_fake_module(monkeypatch, name))
    # Fake VAE module whose filter taps are built without an explicit device,
    # mirroring the real _kaiser_sinc_filter (device-less tensor creation).
    calls = {"n": 0}

    def _fake_kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
        calls["n"] += 1
        return torch.ones(1, 1, kernel_size, dtype=torch.float32)

    auk_vae = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.auk.auk_vae",
        _kaiser_sinc_filter=_fake_kaiser_sinc_filter,
    )
    auk_vae._kaiser_calls = calls
    sys.modules["vllm_omni.diffusion.models.auk"].auk_vae = auk_vae
    registrations: list[dict] = []
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.registry",
        _install_fake_module(
            monkeypatch,
            "vllm_omni.diffusion.registry",
            register_diffusion_model=lambda *args, **kwargs: registrations.append({"args": args, "kwargs": kwargs}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.models.auk.pipeline_auk",
        _install_fake_module(
            monkeypatch,
            "vllm_omni.diffusion.models.auk.pipeline_auk",
            AuKPipeline=_FakeAuKPipeline,
            get_auk_post_process_func=lambda od_config: ("post", od_config),
        ),
    )
    return registrations


def _load_auk_patch(monkeypatch: pytest.MonkeyPatch):
    registrations = _install_auk_fakes(monkeypatch)
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "auk.py"
    module = _load_source_module("ausk_patch_under_test", path)
    return module, registrations


def _fake_od_config(**overrides) -> SimpleNamespace:
    config = SimpleNamespace(
        model="unused",
        dtype=torch.bfloat16,
        enforce_eager=True,
        diffusion_kv_cache_dtype=None,
        model_class_name="AuKPipeline",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_apply_forces_fp16_and_registers_310p_pipeline(monkeypatch):
    module, registrations = _load_auk_patch(monkeypatch)
    od_config = _fake_od_config(dtype=torch.bfloat16)

    module.apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16
    assert len(registrations) == 1
    entry = registrations[0]
    assert entry["args"] == ("AuKPipeline",)
    assert entry["kwargs"]["module_name"] == module.__name__
    assert entry["kwargs"]["class_name"] == "_AuK310P"
    assert entry["kwargs"]["post_process_func_name"] == "get_auk_post_process_func"
    # The post-process func must be resolvable from the registered module.
    assert callable(module.get_auk_post_process_func)


def test_apply_is_idempotent_for_registration(monkeypatch):
    module, registrations = _load_auk_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_fake_od_config())
    module.apply_diffusion_310p_patches(_fake_od_config(dtype=torch.bfloat16))

    assert len(registrations) == 1
    assert module._PATCHED is True


def test_apply_neutralizes_npu_only_kv_cache_quant(monkeypatch):
    module, _ = _load_auk_patch(monkeypatch)
    od_config = _fake_od_config(diffusion_kv_cache_dtype="fp8")

    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_kv_cache_dtype == "auto"
    assert od_config.dtype is torch.float16


def test_auk310p_forces_graph_wrappers_off(monkeypatch):
    module, _ = _load_auk_patch(monkeypatch)
    # A non-eager deploy would build the CUDA-graph wrappers enabled; the 310P
    # subclass must force them off regardless (CUDA-only capture primitives).
    od_config = _fake_od_config(enforce_eager=False)
    od_config.dtype = torch.float16

    pipeline = module._AuK310P(od_config=od_config)

    assert pipeline.cudagraph_wrapper.enabled is False
    assert pipeline.vae_decode.enabled is False
    assert pipeline.dtype is torch.float16


def test_auk310p_keeps_eager_deploy_state(monkeypatch):
    module, _ = _load_auk_patch(monkeypatch)
    od_config = _fake_od_config(enforce_eager=True)

    pipeline = module._AuK310P(od_config=od_config)

    assert pipeline.cudagraph_wrapper.enabled is False
    assert pipeline.vae_decode.enabled is False


def _load_dispatch_init(monkeypatch: pytest.MonkeyPatch, monkey_module: str):
    """Load the real _310p/patch/__init__.py dispatch table from source.

    The module-level surface of that file is stdlib-only (logging/os); the
    patch submodules it imports lazily are faked so the real sdxl/auk code
    (which drags in pipeline code) is not loaded.
    """
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    return _load_source_module(monkey_module, path)


def test_dispatch_routes_auk_pipeline(monkeypatch):
    init = _load_dispatch_init(monkeypatch, "ausk_patch_init_under_test")
    calls: list[str] = []

    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: calls.append("attention"),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.sdxl",
        apply_diffusion_310p_patches=lambda od_config: calls.append("sdxl"),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.auk",
        apply_diffusion_310p_patches=lambda od_config: calls.append("auk"),
    )

    od_config = _fake_od_config()
    init.apply_diffusion_310p_patches(od_config)

    # The math-SDPA fix is global to every 310P diffusion pipeline; the
    # model-scoped module follows the registry entry.
    assert calls == ["attention", "auk"]

    # Unknown pipelines still get the global attention fix, but no
    # model-scoped module (generic 310P path, no crash).
    calls.clear()
    init.apply_diffusion_310p_patches(_fake_od_config(model_class_name="SomeOtherPipeline"))
    assert calls == ["attention"]

    # The SDXL branch keeps its own routing.
    calls.clear()
    init.apply_diffusion_310p_patches(_fake_od_config(model_class_name="StableDiffusionXLPipeline"))
    assert calls == ["attention", "sdxl"]


def test_auk310p_pins_kaiser_filter_to_cpu(monkeypatch):
    """The VAE's Kaiser-sinc taps must be built on CPU.

    Inside the NPU device context used by the diffusion loader, the real
    ``_kaiser_sinc_filter`` would create NPU tensors and dispatch
    ``aten::kaiser_window`` (no NPU kernel on this torch_npu build ->
    DispatchStub abort).  The 310P subclass must install the CPU-pinned
    wrapper before the VAE is constructed.
    """
    module, _ = _load_auk_patch(monkeypatch)
    auk_vae = sys.modules["vllm_omni.diffusion.models.auk.auk_vae"]
    original = auk_vae._kaiser_sinc_filter

    module._AuK310P(od_config=_fake_od_config(enforce_eager=True))

    assert auk_vae._auk_310p_kaiser_patch is True
    assert auk_vae._kaiser_sinc_filter is not original
    # Idempotent: constructing a second pipeline must not re-wrap.
    first = auk_vae._kaiser_sinc_filter
    module._AuK310P(od_config=_fake_od_config(enforce_eager=True))
    assert auk_vae._kaiser_sinc_filter is first
    # The wrapper forces CPU tensor creation (the fake builds device-less
    # tensors, mirroring the real filter).
    taps = auk_vae._kaiser_sinc_filter(0.5, 0.6, 12)
    assert taps.device.type == "cpu"


def test_dispatch_env_disable(monkeypatch):
    init = _load_dispatch_init(monkeypatch, "ausk_patch_init_env_under_test")
    called = {"flag": False}

    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: called.update(flag=True),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.auk",
        apply_diffusion_310p_patches=lambda od_config: called.update(flag=True),
    )
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")

    init.apply_diffusion_310p_patches(_fake_od_config())

    assert called["flag"] is False


# ---------------------------------------------------------------------------
# AR-worker MRoPE patch (rotary.py)
# ---------------------------------------------------------------------------


def _install_rotary_fakes(monkeypatch, is_310p_value: bool):
    _install_fake_module(
        monkeypatch, "torch_npu",
        npu=SimpleNamespace(get_device_name=lambda idx: "Ascend310P3"),
    )
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        monkeypatch.setitem(sys.modules, name, _install_fake_module(monkeypatch, name))

    # Fresh class objects per test: the patch rewrites class attributes in
    # place, so a module-level shared class would leak state between tests.
    class _FakeBaseMRoPE:
        def forward_oot(self, positions, query, key):  # noqa: N803
            return ("torch-native", positions, query, key)

    class _FakeAscendMRoPE:
        """Stand-in for vllm_ascend.ops.rotary_embedding.AscendMRotaryEmbedding."""

        def forward_oot(self, positions, query, key):  # noqa: N803
            return ("ascend-fast-path", positions, query, key)

    _install_fake_module(
        monkeypatch, "vllm.model_executor.layers.rotary_embedding", MRotaryEmbedding=_FakeBaseMRoPE,
    )
    for name in ("vllm_ascend", "vllm_ascend.ops"):
        monkeypatch.setitem(sys.modules, name, _install_fake_module(monkeypatch, name))
    _install_fake_module(
        monkeypatch, "vllm_ascend.ops.rotary_embedding", AscendMRotaryEmbedding=_FakeAscendMRoPE,
    )
    for name in ("vllm_omni", "vllm_omni.platforms", "vllm_omni.platforms.npu"):
        monkeypatch.setitem(sys.modules, name, _install_fake_module(monkeypatch, name))
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p",
        is_310p=lambda device=None: is_310p_value,
    )
    return _FakeBaseMRoPE, _FakeAscendMRoPE


def _load_rotary_patch(monkeypatch):
    return _load_source_module(
        "auk_rotary_patch_under_test",
        _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "rotary.py",
    )


def test_rotary_310p_routes_mrope_to_torch(monkeypatch):
    """On 310P the fast-path aclnn kernel is replaced by the torch-native
    forward_oot (EZ1001: RopeWithSinCosCacheV2 has no 310P dtype coverage)."""
    _load_rotary_patch(monkeypatch)
    base, ascend_cls = _install_rotary_fakes(monkeypatch, is_310p_value=True)
    module = _load_rotary_patch(monkeypatch)

    module.maybe_patch_310p_mrope()

    assert ascend_cls.forward_oot is base.forward_oot
    assert getattr(ascend_cls, "_auk_310p_mrope_patched", False) is True
    # Idempotent: a second call must be a no-op, not a re-wrap error.
    module.maybe_patch_310p_mrope()
    assert ascend_cls.forward_oot is base.forward_oot


def test_rotary_off_310p_is_noop(monkeypatch):
    _load_rotary_patch(monkeypatch)
    base, ascend_cls = _install_rotary_fakes(monkeypatch, is_310p_value=False)
    module = _load_rotary_patch(monkeypatch)

    module.maybe_patch_310p_mrope()

    # The fast path is untouched: behaviour and patch flag both unchanged.
    assert ascend_cls().forward_oot(None, None, None)[0] == "ascend-fast-path"
    assert not hasattr(ascend_cls, "_auk_310p_mrope_patched")
