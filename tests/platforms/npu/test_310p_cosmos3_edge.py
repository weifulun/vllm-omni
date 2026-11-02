# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Cosmos3 family 310P patch wiring.

Loads ``vllm_omni/platforms/npu/_310p/patch/cosmos3_edge.py`` and the 310P
patch registry from source with fake ``vllm``/``vllm_omni`` dependencies, so
the patch contract (FP16 enforcement, attention-backend forcing, cache
disabling, quantization tier gate, VAE tiling, guardrails gate, single-chip
parallelism preflight, registry dispatch on ``model_class_name``) is pinned
without an NPU, vllm-ascend or a real model.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_ltx25`.
"""

from __future__ import annotations

import importlib.util
import logging
import os
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
    model_class_name: str = "Cosmos3OmniPipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    model_config: dict | None = None,
    vae_use_tiling: bool = False,
    vae_use_slicing: bool = False,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    allgather_degree: int = 1,
    cfg_parallel_size: int = 1,
    extras: dict | None = None,
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
        vae_use_tiling=vae_use_tiling,
        vae_use_slicing=vae_use_slicing,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        cfg_parallel_size=cfg_parallel_size,
        extras=dict(extras or {}),
    )


def _make_fake_quant(method: str):
    return SimpleNamespace(get_name=lambda method=method: method)


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules cosmos3_edge.py imports lazily."""
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

    # The registry dispatch and kill-switch paths import the attention
    # patch module by real name; isolate it from the installed package so
    # the registry-under-test never touches the site-packages tree.
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: None,
    )

    captured["backend_calls"] = calls
    return captured


def _load_cosmos3_patch(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    module = _load_source_module("cosmos3_310p_under_test", patch_dir / "cosmos3_edge.py")
    # Fresh module globals per test: reset the idempotency flags.
    module._APPLIED = False
    module._ATTENTION_GUARD_INSTALLED = False
    return module


def _load_registry_module(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    return _load_source_module("cosmos3_registry_under_test", patch_dir / "__init__.py")


# ---------------------------------------------------------------------------
# dtype / attention / cache invariants
# ---------------------------------------------------------------------------


def test_fp16_is_accepted_and_config_normalized(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.float16)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default is not None
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert od_config.cache_backend is None


def test_bf16_dtype_is_rejected_on_310p(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="FP16"):
        module.apply_diffusion_310p_patches(_make_od_config(dtype=torch.bfloat16))


def test_explicit_mindiesd_default_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(default_backend="FLASH_ATTN"))


def test_explicit_mindiesd_per_role_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(per_role={"self": "RAINFUSION_ATTN"}))


def test_numerical_cache_backend_is_disabled(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    od_config = _make_od_config(cache_backend="cache_dit")
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.cache_backend == "none"


# ---------------------------------------------------------------------------
# quantization tier gate
# ---------------------------------------------------------------------------


def test_int8_quantization_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    od_config = _make_od_config(quantization_config=_make_fake_quant("int8"))
    module.apply_diffusion_310p_patches(od_config)  # must not raise


def test_fp8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="rejects quantization"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("fp8")))


def test_convrot_int8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="rejects quantization"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("convrot_int8")))


def test_mxfp8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="rejects quantization"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("mxfp8")))


# ---------------------------------------------------------------------------
# VAE tiling forcing
# ---------------------------------------------------------------------------


def test_vae_tiling_and_slicing_are_forced(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    od_config = _make_od_config(vae_use_tiling=False, vae_use_slicing=False)
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True


# ---------------------------------------------------------------------------
# guardrails gate
# ---------------------------------------------------------------------------


def test_guardrails_default_on_is_forced_off(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    od_config = _make_od_config(model_config={"guardrails": True})
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.model_config["guardrails"] is False


def test_guardrails_defaulting_on_is_forced_off_when_absent(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    # model_config without an explicit guardrails key: the upstream default is
    # ON, so the patch must force it off.
    od_config = _make_od_config(model_config={})
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.model_config.get("guardrails", True) is False


def test_guardrails_already_off_is_preserved(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    od_config = _make_od_config(model_config={"guardrails": False})
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.model_config["guardrails"] is False


# ---------------------------------------------------------------------------
# single-chip parallelism preflight
# ---------------------------------------------------------------------------


def test_ulysses_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="sequence parallel"):
        module.apply_diffusion_310p_patches(_make_od_config(ulysses_degree=2))


def test_ring_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="sequence parallel"):
        module.apply_diffusion_310p_patches(_make_od_config(ring_degree=2))


def test_allgather_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="sequence parallel"):
        module.apply_diffusion_310p_patches(_make_od_config(allgather_degree=2))


def test_cfg_parallel_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="CFG parallel"):
        module.apply_diffusion_310p_patches(_make_od_config(cfg_parallel_size=2))


def test_tp1_defaults_pass(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())  # must not raise


# ---------------------------------------------------------------------------
# attention guard (platform selector wrap)
# ---------------------------------------------------------------------------


def test_attention_guard_defaults_to_torch_sdpa(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())

    platform_module = sys.modules["vllm_omni.platforms.npu.platform"]
    path = platform_module.NPUOmniPlatform.get_diffusion_attn_backend_cls(None, 128)
    assert path == "vllm_omni.diffusion.attention.backends.torch_sdpa.Backend"


def test_attention_guard_rejects_mindiesd_only_backend(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())

    platform_module = sys.modules["vllm_omni.platforms.npu.platform"]
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        platform_module.NPUOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN", 128)


def test_attention_guard_passes_other_backends_through(monkeypatch: pytest.MonkeyPatch):
    captured = _install_patch_fakes(monkeypatch)
    module = _load_cosmos3_patch(monkeypatch)

    module.apply_diffusion_310p_patches(_make_od_config())

    platform_module = sys.modules["vllm_omni.platforms.npu.platform"]
    result = platform_module.NPUOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN", 128)
    assert result == "resolved:SAGE_ATTN"
    assert captured["backend_calls"] == [("SAGE_ATTN", 128)]


# ---------------------------------------------------------------------------
# registry dispatch
# ---------------------------------------------------------------------------


def test_registry_registers_cosmos3_aliases(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    for arch in ("Cosmos3OmniPipeline", "Cosmos3OmniDiffusersPipeline"):
        assert arch in registry._DIFFUSION_310P_PATCH_MODULES
        assert registry._DIFFUSION_310P_PATCH_MODULES[arch] == [
            "vllm_omni.platforms.npu._310p.patch.cosmos3_edge"
        ]
    assert set(registry.COSMOS3_ARCHS if hasattr(registry, "COSMOS3_ARCHS") else ()) <= set(
        registry._DIFFUSION_310P_PATCH_MODULES
    )


def test_registry_dispatch_applies_cosmos3_patch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    # Make the real import path resolve to the source-loaded module under
    # test so the dispatch's importlib.import_module hits the fake.
    patch_module = _load_cosmos3_patch(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.cosmos3_edge", patch_module)

    od_config = _make_od_config()
    registry.apply_diffusion_310p_patches(od_config)

    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert od_config.model_config.get("guardrails", True) is False


def test_registry_dispatch_is_noop_for_unregistered_pipeline(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    od_config = _make_od_config(model_class_name="SomeOtherPipeline")
    registry.apply_diffusion_310p_patches(od_config)

    assert od_config.vae_use_tiling is False
    assert od_config.diffusion_attention_config.default is None


def test_kill_switch_env_disables_dispatch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")

    od_config = _make_od_config()
    registry.apply_diffusion_310p_patches(od_config)

    assert od_config.vae_use_tiling is False
    assert od_config.diffusion_attention_config.default is None
