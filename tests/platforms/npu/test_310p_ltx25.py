# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the LTX-2.5 310P patch wiring.

Loads ``vllm_omni/platforms/npu/_310p/patch/ltx25.py`` and the 310P patch
registry from source with fake ``vllm``/``vllm_omni`` dependencies, so the
patch contract (FP16 enforcement, attention-backend forcing, cache
disabling, quantization tier gate, ConvVAE forcing, VAE tiling, encoder CPU
pinning with device-aware forward, registry dispatch on
``model_class_name``) is pinned without an NPU, vllm-ascend or a real model.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_flux2_klein`.
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
    model_class_name: str = "LTX2Pipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    lora_path: object = None,
    tf_params: dict | None = None,
    ulysses_degree: int = 1,
    extras: dict | None = None,
    vae_use_tiling: bool = False,
    vae_use_slicing: bool = False,
    enable_layerwise_offload: bool = False,
    diffusion_offload_config: dict | None = None,
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
        extras=dict(extras or {}),
        vae_use_tiling=vae_use_tiling,
        vae_use_slicing=vae_use_slicing,
        enable_layerwise_offload=enable_layerwise_offload,
        diffusion_offload_config=diffusion_offload_config,
    )


def _make_fake_quant(method: str):
    return SimpleNamespace(get_name=lambda method=method: method)


class _FakeEncoder:
    """Stands in for the Gemma-4 text encoder: tracks device and a forward."""

    def __init__(self):
        self.device = "npu"

    def to(self, device):
        self.device = device
        return self

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Stand-in output mirrors the tensor inputs' device so the wrapper
        # round-trip is observable in tests.
        return SimpleNamespace(hidden_states=input_ids)


def _make_fake_pipeline_module(monkeypatch: pytest.MonkeyPatch) -> type:
    class FakeLTX2Pipeline:
        def __init__(self, *, od_config=None, prefix: str = ""):
            # CPU device so the suite runs on CPU-only torch builds; on the
            # NPU image the real pipeline device is npu:N and the wrapper
            # logic under test is identical.
            self.device = torch.device("cpu")
            self.text_encoder = _FakeEncoder()
            self.vae = SimpleNamespace()

    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.ltx2.pipeline_ltx2",
        LTX2Pipeline=FakeLTX2Pipeline,
    )
    return FakeLTX2Pipeline


def _install_patch_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install the fake modules ltx25.py imports lazily."""
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

    _make_fake_pipeline_module(monkeypatch)

    captured["backend_calls"] = calls
    return captured


def _load_ltx25_patch(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    module = _load_source_module("ltx25_310p_under_test", patch_dir / "ltx25.py")
    # Fresh module globals per test: reset the idempotency flags.
    module._APPLIED = False
    module._ATTENTION_GUARD_INSTALLED = False
    module._PIPELINE_INIT_WRAPPED = False
    return module


# ---------------------------------------------------------------------------
# dtype / attention / cache invariants
# ---------------------------------------------------------------------------


def test_fp16_is_accepted_and_config_normalized(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    od_config = _make_od_config(dtype=torch.float16)
    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_attention_config.default is not None
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert od_config.cache_backend is None


def test_bf16_dtype_is_rejected_on_310p(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="FP16"):
        module.apply_diffusion_310p_patches(_make_od_config(dtype=torch.bfloat16))


def test_explicit_mindiesd_default_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(default_backend="FLASH_ATTN"))


def test_explicit_mindiesd_per_role_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="MindIE-SD"):
        module.apply_diffusion_310p_patches(_make_od_config(per_role={"self": "RAINFUSION_ATTN"}))


def test_numerical_cache_backend_is_disabled(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    od_config = _make_od_config(cache_backend="cache_dit")
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.cache_backend == "none"


# ---------------------------------------------------------------------------
# quantization tier gate
# ---------------------------------------------------------------------------


def test_int8_quantization_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    od_config = _make_od_config(quantization_config=_make_fake_quant("int8"))
    module.apply_diffusion_310p_patches(od_config)  # must not raise


def test_fp8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="rejects quantization"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("fp8")))


def test_convrot_int8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="rejects quantization"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("convrot_int8")))


def test_mxfp8_quantization_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="rejects quantization"):
        module.apply_diffusion_310p_patches(_make_od_config(quantization_config=_make_fake_quant("mxfp8")))


# ---------------------------------------------------------------------------
# ConvVAE / VAE tiling forcing
# ---------------------------------------------------------------------------


def test_conv_vae_is_forced_when_unset(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    od_config = _make_od_config()
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.extras["ltx2_use_conv_vae"] is True


def test_explicit_diffvae_optout_is_rejected_on_310p(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="NATTEN"):
        module.apply_diffusion_310p_patches(_make_od_config(extras={"ltx2_use_conv_vae": False}))


def test_explicit_conv_vae_choice_is_preserved(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    od_config = _make_od_config(extras={"ltx2_use_conv_vae": True})
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.extras["ltx2_use_conv_vae"] is True


def test_vae_tiling_and_slicing_are_forced(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    od_config = _make_od_config(vae_use_tiling=False, vae_use_slicing=False)
    module.apply_diffusion_310p_patches(od_config)
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True


# ---------------------------------------------------------------------------
# encoder CPU pin (chain-1 OOM downgrade)
# ---------------------------------------------------------------------------


def test_encoder_is_pinned_to_cpu_with_device_aware_forward(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)
    fake_pipeline = _make_fake_pipeline_module(monkeypatch)

    od_config = _make_od_config()
    module.apply_diffusion_310p_patches(od_config)

    pipeline = fake_pipeline()
    encoder = pipeline.text_encoder
    assert encoder.device == "cpu"
    assert getattr(encoder.forward, "_310p_wrapped", False)

    # The wrapper moves inputs to the encoder device (CPU) and returns the
    # output on the pipeline device.
    input_ids = torch.arange(4)
    out = encoder.forward(input_ids=input_ids)
    assert out.hidden_states.device == pipeline.device


def test_encoder_pin_is_skipped_with_layerwise_offload(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)
    fake_pipeline = _make_fake_pipeline_module(monkeypatch)

    od_config = _make_od_config(diffusion_offload_config={"strategy": "layer_wise"})
    module.apply_diffusion_310p_patches(od_config)

    pipeline = fake_pipeline()
    encoder = pipeline.text_encoder
    assert encoder.device == "npu"  # not moved to CPU
    assert not getattr(encoder.forward, "_310p_wrapped", False)


def test_encoder_pin_can_be_disabled_by_env_without_offload(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)
    fake_pipeline = _make_fake_pipeline_module(monkeypatch)
    monkeypatch.setenv("LTX25_310P_ENCODER_CPU", "0")

    od_config = _make_od_config()
    module.apply_diffusion_310p_patches(od_config)

    pipeline = fake_pipeline()
    assert pipeline.text_encoder.device == "npu"


def test_encoder_pin_env_default_is_on(monkeypatch: pytest.MonkeyPatch):
    """Without the env var (and without offload) the pin engages."""
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)
    fake_pipeline = _make_fake_pipeline_module(monkeypatch)
    monkeypatch.delenv("LTX25_310P_ENCODER_CPU", raising=False)

    module.apply_diffusion_310p_patches(_make_od_config())

    pipeline = fake_pipeline()
    assert pipeline.text_encoder.device == "cpu"


# ---------------------------------------------------------------------------
# strict-Ulysses preflight
# ---------------------------------------------------------------------------


def test_heads_divisibility_passes(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    module.apply_diffusion_310p_patches(
        _make_od_config(tf_params={"num_attention_heads": 24}, ulysses_degree=2)
    )


def test_heads_divisibility_fails(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    module = _load_ltx25_patch(monkeypatch)

    with pytest.raises(RuntimeError, match="num_attention_heads"):
        module.apply_diffusion_310p_patches(
            _make_od_config(tf_params={"num_attention_heads": 25}, ulysses_degree=2)
        )


# ---------------------------------------------------------------------------
# registry dispatch
# ---------------------------------------------------------------------------


def _load_registry_module(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    return _load_source_module("ltx25_registry_under_test", patch_dir / "__init__.py")


def test_registry_registers_all_ltx25_pipelines(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    for arch in ("LTX2Pipeline", "LTX2DistilledOneStagePipeline", "LTX2TwoStagePipeline", "LTX2DistilledTwoStagePipeline"):
        assert arch in registry._DIFFUSION_310P_PATCH_MODULES
        assert registry._DIFFUSION_310P_PATCH_MODULES[arch] == [
            "vllm_omni.platforms.npu._310p.patch.ltx25"
        ]


def test_registry_dispatch_applies_ltx25_patch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    # Make the real import path resolve to the source-loaded module under
    # test so the dispatch's importlib.import_module hits the fake.
    patch_module = _load_ltx25_patch(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.ltx25", patch_module)

    od_config = _make_od_config()
    registry.apply_diffusion_310p_patches(od_config)

    assert od_config.extras["ltx2_use_conv_vae"] is True
    assert od_config.vae_use_tiling is True
    assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_registry_dispatch_is_noop_for_unregistered_pipeline(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)

    od_config = _make_od_config(model_class_name="SomeOtherPipeline")
    registry.apply_diffusion_310p_patches(od_config)

    assert "ltx2_use_conv_vae" not in od_config.extras
    assert od_config.vae_use_tiling is False


def test_kill_switch_env_disables_dispatch(monkeypatch: pytest.MonkeyPatch):
    _install_patch_fakes(monkeypatch)
    registry = _load_registry_module(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")

    od_config = _make_od_config()
    registry.apply_diffusion_310p_patches(od_config)

    assert "ltx2_use_conv_vae" not in od_config.extras
    assert od_config.vae_use_tiling is False
