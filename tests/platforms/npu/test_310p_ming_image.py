# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Ming-Image family 310P patch wiring (task-36).

Loads ``vllm_omni/platforms/npu/_310p/patch/ming_image.py`` and the 310P
patch registry from source with fake ``vllm``/``vllm_omni`` dependencies, so
the patch contract (FP16 forcing, attention-backend forcing, cache disabling,
quantization tier gate, eager-decode forcing, VAE tiling, KV-quant
neutralization, single-chip parallelism preflight, registry dispatch on
``model_class_name``, kill switch) is pinned without an NPU, vllm-ascend or
a real model.

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
    # Link the fake into its fake parent so ``import a.b.c as x`` attribute
    # lookups succeed without touching the real package tree.
    if "." in name:
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and parent.__name__ == parent_name and getattr(parent, "__fake_", False):
            setattr(parent, child, module)
    module.__fake__ = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return module


#: Parent packages shadowed by fakes so the source-loaded registry never
#: triggers real (NPU-bound) package __init__ chains on a CPU host.
_PACKAGE_PARENTS = (
    "vllm_omni",
    "vllm_omni.diffusion",
    "vllm_omni.diffusion.models",
    "vllm_omni.diffusion.model_loader",
    "vllm_omni.platforms",
    "vllm_omni.platforms.npu",
    "vllm_omni.platforms.npu._310p",
    "vllm_omni.platforms.npu._310p.patch",
)


def _make_od_config(
    *,
    dtype: torch.dtype = torch.float16,
    model_class_name: str = "MingImageDiffusionPipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    vae_use_tiling: bool = False,
    vae_use_slicing: bool = False,
    diffusion_kv_cache_dtype: str | None = None,
    enforce_eager: bool = False,
    diffusion_compile_granularity: str = "regional",
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
        vae_use_tiling=vae_use_tiling,
        vae_use_slicing=vae_use_slicing,
        diffusion_kv_cache_dtype=diffusion_kv_cache_dtype,
        enforce_eager=enforce_eager,
        diffusion_compile_granularity=diffusion_compile_granularity,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        cfg_parallel_size=cfg_parallel_size,
    )


def _make_fake_quant(method: str):
    return SimpleNamespace(get_name=lambda method=method: method)


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules the registry dispatcher and ming_image.py import lazily."""
    for parent in _PACKAGE_PARENTS:
        _install_fake_module(monkeypatch, parent)
    captured: dict[str, object] = {}

    class FakeAttentionSpec:
        def __init__(self, backend: str):
            self.backend = backend

    _install_fake_module(monkeypatch, "vllm_omni.diffusion.data", AttentionSpec=FakeAttentionSpec)

    sdpa_math_calls: list[str] = []

    def maybe_patch_sdpa_math() -> None:
        sdpa_math_calls.append("called")

    # The registry dispatch and the HF-encoder shims import these by real
    # name; isolate them so the registry-under-test never touches the
    # site-packages tree.
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=maybe_patch_sdpa_math,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.utils",
        create_transformers_model=lambda *a, **k: None,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.registry",
        initialize_model=lambda od_config, *a, **k: None,
    )
    captured["sdpa_math_calls"] = sdpa_math_calls
    return captured


def _load_ming_image_patch(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    module = _load_source_module("mingimage_310p_under_test", patch_dir / "ming_image.py")
    # Fresh module globals per test: reset the idempotency flag.
    module._APPLIED = False
    return module


def _load_registry_module(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    return _load_source_module("mingimage_registry_under_test", patch_dir / "__init__.py")


# ---------------------------------------------------------------------------
# dtype / attention / cache / quant / eager / VAE / KV-quant invariants
# ---------------------------------------------------------------------------


def test_fp16_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.float16)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16


def test_bf16_dtype_is_forced_to_fp16(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.bfloat16)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16


def test_explicit_mindiesd_default_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(default_backend="FLASH_ATTN"))


def test_explicit_mindiesd_per_role_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="per-role"):
        module.apply_diffusion_310p_patches(_make_od_config(per_role={"diffusion": "RAINFUSION_ATTN"}))


def test_attention_default_none_becomes_torch_sdpa(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config()
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_numerical_cache_backend_is_disabled(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(cache_backend="cache_dit")
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.cache_backend == "none"


def test_none_cache_backend_is_left_alone(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(cache_backend=None)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.cache_backend is None


def test_int8_quantization_is_accepted_with_warning(monkeypatch: pytest.MonkeyPatch, caplog):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with caplog.at_level(logging.WARNING):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("int8")))

    assert any("int8" in message for message in caplog.messages)


def test_fp8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="fp8"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("fp8")))


def test_mxfp8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="mxfp8"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("mxfp8")))


def test_eager_decode_is_forced_when_deploy_ships_regional_compile(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(enforce_eager=False, diffusion_compile_granularity="regional")
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.enforce_eager is True
    # The granularity flag itself is untouched; enforce_eager alone disengages
    # setup_compile() and CUDA-graph decode capture in the model runner.
    assert od_config.diffusion_compile_granularity == "regional"


def test_eager_already_on_is_preserved(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(enforce_eager=True)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.enforce_eager is True


def test_vae_tiling_and_slicing_are_forced(monkeypatch: pytest.MonkeyPatch, caplog):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(vae_use_tiling=False, vae_use_slicing=False)
    with caplog.at_level(logging.WARNING):
        module.apply_diffusion_310p_patches(od_config)

    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    assert any("vae_use_tiling" in message for message in caplog.messages)


def test_attention_kv_quant_is_neutralized(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(diffusion_kv_cache_dtype="fp8")
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_kv_cache_dtype == "auto"


def test_attention_kv_quant_auto_is_preserved(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(diffusion_kv_cache_dtype="auto")
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_kv_cache_dtype == "auto"


def test_ulysses_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="Ulysses"):
        module.apply_diffusion_310p_patches(_make_od_config(ulysses_degree=2))


def test_ring_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="ring"):
        module.apply_diffusion_310p_patches(_make_od_config(ring_degree=2))


def test_allgather_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="allgather"):
        module.apply_diffusion_310p_patches(_make_od_config(allgather_degree=2))


def test_cfg_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="CFG parallelism"):
        module.apply_diffusion_310p_patches(_make_od_config(cfg_parallel_size=2))


def test_tp1_defaults_pass(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config()
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.vae_use_tiling is True
    assert od_config.enforce_eager is True
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch, caplog):
    _install_patch_fakes(monkeypatch)
    module = _load_ming_image_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING):
        module.apply_diffusion_310p_patches(od_config)
        first_warnings = len(caplog.messages)
        module.apply_diffusion_310p_patches(od_config)
        second_warnings = len(caplog.messages) - first_warnings

    assert od_config.dtype is torch.float16
    assert second_warnings == 0


# ---------------------------------------------------------------------------
# registry dispatch / kill switch
# ---------------------------------------------------------------------------


def test_registry_registers_ming_image_aliases(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    for arch in ("MingImageDiffusionPipeline", "MingImageLayeredDiffusionPipeline"):
        assert arch in registry._DIFFUSION_310P_PATCH_MODULES
        assert registry._DIFFUSION_310P_PATCH_MODULES[arch] == [
            "vllm_omni.platforms.npu._310p.patch.ming_image"
        ]
    # Pre-existing registrations survive (SDXL / Z-Image lines).
    assert registry._DIFFUSION_310P_PATCH_MODULES["ZImagePipeline"] == [
        "vllm_omni.platforms.npu._310p.patch.z_image"
    ]


def test_registry_dispatch_applies_ming_image_patch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    # Make the real import path resolve to the source-loaded module under
    # test so the dispatch's importlib.import_module hits the fake.
    patch_module = _load_ming_image_patch(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.ming_image", patch_module)

    od_config = _make_od_config()
    registry.apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    assert od_config.enforce_eager is True
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_registry_dispatch_is_noop_for_unregistered_pipeline(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    od_config = _make_od_config(model_class_name="SomeOtherPipeline")
    registry.apply_diffusion_310p_patches(od_config)

    assert od_config.vae_use_tiling is False
    assert od_config.diffusion_attention_config.default is None
    assert od_config.enforce_eager is False


def test_kill_switch_env_disables_dispatch(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")

    od_config = _make_od_config()
    registry.apply_diffusion_310p_patches(od_config)

    # Model invariants are skipped ...
    assert od_config.vae_use_tiling is False
    assert od_config.diffusion_attention_config.default is None
    assert od_config.enforce_eager is False
    # ... but the hardware-level MATH-SDPA shim hook still ran (it is outside
    # the kill switch by design).
    assert captured["sdpa_math_calls"] == ["called"]


def test_layered_alias_dispatches_to_same_module(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    patch_module = _load_ming_image_patch(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.ming_image", patch_module)

    od_config = _make_od_config(model_class_name="MingImageLayeredDiffusionPipeline")
    registry.apply_diffusion_310p_patches(od_config)

    assert od_config.vae_use_tiling is True
    assert od_config.enforce_eager is True
