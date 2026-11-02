# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the 310P SDXL diffusion patch wiring.

Loads the patch module from source with a fake SDXL pipeline, so the
contract (FP16 forcing, VAE tiling, quant neutralization, CPU encoder pinning,
prompt-embeds device routing, registry override) is validated on CPU without
diffusers / transformers / vllm_ascend.
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


class _FakeEncoderOut:
    def __init__(self, hidden: torch.Tensor, pooled: torch.Tensor):
        self.hidden_states = [hidden, hidden, hidden]
        self._pooled = pooled

    def __getitem__(self, index: int) -> torch.Tensor:
        assert index == 0
        return self._pooled


class _FakeEncoder:
    def __init__(self):
        self.to_calls: list[dict] = []
        self.call_devices: list[torch.device] = []
        self.last_input_dtype = None

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        return self

    def __call__(self, input_ids: torch.Tensor, *, output_hidden_states: bool):
        assert output_hidden_states
        self.call_devices.append(input_ids.device)
        self.last_input_dtype = input_ids.dtype
        return _FakeEncoderOut(
            hidden=torch.zeros(1, 77, 8, dtype=torch.float32),
            pooled=torch.zeros(1, 8, dtype=torch.float32),
        )


class _FakeTokenizer:
    def __call__(self, prompt, *, padding, max_length, truncation, return_tensors):
        assert return_tensors == "pt"
        assert padding == "max_length"
        assert truncation is True
        return SimpleNamespace(input_ids=torch.zeros((1, max_length), dtype=torch.long))


class FakeStableDiffusionXLPipeline(torch.nn.Module):
    """Stand-in for the built-in SDXL pipeline: same attribute surface the
    310P subclass touches, devices resolved to CPU on the test host."""

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.device = torch.device("cpu")  # stands in for the NPU device
        self.tokenizer_max_length = 77
        self.text_encoder = _FakeEncoder()
        self.text_encoder_2 = _FakeEncoder()


def _install_sdxl_patch_fakes(
    monkeypatch: pytest.MonkeyPatch,
    registry_calls: list[dict],
):
    def register_diffusion_model(
        model_arch: str,
        module_name: str,
        class_name: str,
        pre_process_func_name: str | None = None,
        post_process_func_name: str | None = None,
        **_kwargs,
    ) -> None:
        registry_calls.append(
            {
                "model_arch": model_arch,
                "module_name": module_name,
                "class_name": class_name,
                "pre_process_func_name": pre_process_func_name,
                "post_process_func_name": post_process_func_name,
            }
        )

    fake_pipeline = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.sdxl.pipeline_sdxl",
        StableDiffusionXLPipeline=FakeStableDiffusionXLPipeline,
        get_sdxl_image_post_process_func=lambda od_config: lambda images: images,
    )
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models.sdxl")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.registry",
        register_diffusion_model=register_diffusion_model,
    )
    return fake_pipeline


def _load_sdxl_patch(monkeypatch: pytest.MonkeyPatch):
    registry_calls: list[dict] = []
    fake_pipeline = _install_sdxl_patch_fakes(monkeypatch, registry_calls)
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "sdxl.py"
    module = _load_source_module("vllm_omni_test_310p_sdxl_patch", path)
    return module, fake_pipeline, registry_calls


def _make_od_config(**overrides) -> SimpleNamespace:
    base = dict(
        dtype=torch.bfloat16,
        vae_use_tiling=False,
        vae_use_slicing=False,
        diffusion_kv_cache_dtype=None,
        model_class_name="StableDiffusionXLPipeline",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_apply_sdxl_patches_forces_fp16_and_vae_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, registry_calls = _load_sdxl_patch(monkeypatch)
    od_config = _make_od_config()

    module.apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    assert od_config.diffusion_kv_cache_dtype is None
    assert len(registry_calls) == 1
    entry = registry_calls[0]
    assert entry["model_arch"] == "StableDiffusionXLPipeline"
    assert entry["module_name"] == module.__name__
    assert entry["class_name"] == "_StableDiffusionXL310P"
    assert entry["post_process_func_name"] == "get_sdxl_image_post_process_func"


def test_apply_sdxl_patches_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, registry_calls = _load_sdxl_patch(monkeypatch)
    od_config = _make_od_config()

    module.apply_diffusion_310p_patches(od_config)
    module.apply_diffusion_310p_patches(od_config)

    assert len(registry_calls) == 1


def test_apply_sdxl_patches_neutralizes_kv_quant(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, registry_calls = _load_sdxl_patch(monkeypatch)
    od_config = _make_od_config(diffusion_kv_cache_dtype="fp8")

    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_kv_cache_dtype == "auto"
    assert len(registry_calls) == 1


def test_pipeline_pins_text_encoders_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SDXL_ENCODER_ON_NPU", raising=False)
    module, _fake_pipeline, _calls = _load_sdxl_patch(monkeypatch)
    pipe = module._StableDiffusionXL310P(od_config=_make_od_config(dtype=torch.float16))

    assert pipe._encoder_device == torch.device("cpu")
    assert pipe.text_encoder.to_calls == [{"device": torch.device("cpu"), "dtype": torch.float32}]
    assert pipe.text_encoder_2.to_calls == [{"device": torch.device("cpu"), "dtype": torch.float32}]


def test_pipeline_keeps_encoders_on_device_when_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_310P_SDXL_ENCODER_ON_NPU", "1")
    module, _fake_pipeline, _calls = _load_sdxl_patch(monkeypatch)
    pipe = module._StableDiffusionXL310P(od_config=_make_od_config(dtype=torch.float16))

    assert pipe._encoder_device == pipe.device
    assert pipe.text_encoder.to_calls == []
    assert pipe.text_encoder_2.to_calls == []


def test_prompt_embeds_route_through_encoder_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SDXL_ENCODER_ON_NPU", raising=False)
    module, _fake_pipeline, _calls = _load_sdxl_patch(monkeypatch)
    pipe = module._StableDiffusionXL310P(od_config=_make_od_config(dtype=torch.float16))
    tokenizer = _FakeTokenizer()

    prompt_embeds, pooled = pipe._get_clip_prompt_embeds(
        "a cat",
        tokenizer,
        pipe.text_encoder,
        num_images_per_prompt=2,
        return_pooled=True,
    )

    # The encoder saw the tokens on its pinned device, not the pipeline device.
    assert pipe.text_encoder.call_devices == [pipe._encoder_device]
    # Embeds come back on the pipeline device in the runtime FP16 dtype.
    assert prompt_embeds.device == pipe.device
    assert prompt_embeds.dtype is torch.float16
    # repeat(1, 2, 1) -> view gives batch 2 rows.
    assert prompt_embeds.shape[0] == 2
    assert pooled is not None
    assert pooled.device == pipe.device
    assert pooled.dtype is torch.float16
    assert pooled.shape[0] == 2


def test_patch_module_exports_post_process_func(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fake_pipeline, _calls = _load_sdxl_patch(monkeypatch)

    # _load_process_func resolves the post-process func from the registered
    # module; the override must expose the base implementation.
    assert module.get_sdxl_image_post_process_func is fake_pipeline.get_sdxl_image_post_process_func


def test_platform_wiring_calls_sdxl_310p_patches() -> None:
    # Source-level pin: the NPU platform's diffusion runtime hook must stay
    # wired to the 310P SDXL patch (import-level tests cannot reach
    # platform.py on a CPU host because of the vllm_ascend dependency).
    path = (
        _repo_root()
        / "vllm_omni"
        / "platforms"
        / "npu"
        / "platform.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "is_310p()" in source
    assert "disable_jit_compile()" in source
    # The call site must stay registry-generic: model branches register in
    # _310p.patch and never call their own patch module from platform.py
    # (COMPARISON-POLICY.md §5).
    assert "apply_diffusion_310p_patches(od_config)" in source
    assert "patch.sdxl" not in source


def test_diffusion_registry_dispatches_by_model_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Load the patch registry (__init__.py) standalone: it has no heavy
    # top-level imports, so the dispatch contract is testable on CPU.
    init_path = (
        _repo_root()
        / "vllm_omni"
        / "platforms"
        / "npu"
        / "_310p"
        / "patch"
        / "__init__.py"
    )
    patch_init = _load_source_module("vllm_omni_test_310p_patch_init", init_path)

    calls: list[object] = []
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.sdxl",
        apply_diffusion_310p_patches=lambda od: calls.append(od),
    )

    sdxl_od = SimpleNamespace(model_class_name="StableDiffusionXLPipeline")
    patch_init.apply_diffusion_310p_patches(sdxl_od)
    assert calls == [sdxl_od]

    # Unregistered pipelines are a no-op (they keep the generic 310P path).
    other_od = SimpleNamespace(model_class_name="ZImagePipeline")
    patch_init.apply_diffusion_310p_patches(other_od)
    assert calls == [sdxl_od]


def test_diffusion_registry_env_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_OMNI_310P_DIFFUSION_PATCHES=0 selects the generic 310P path (the
    COMPARISON-POLICY.md §1 level-2 same-seed reference run)."""
    init_path = (
        _repo_root()
        / "vllm_omni"
        / "platforms"
        / "npu"
        / "_310p"
        / "patch"
        / "__init__.py"
    )
    patch_init = _load_source_module("vllm_omni_test_310p_patch_init_env", init_path)

    calls: list[object] = []
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.sdxl",
        apply_diffusion_310p_patches=lambda od: calls.append(od),
    )

    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    od = SimpleNamespace(model_class_name="StableDiffusionXLPipeline")
    patch_init.apply_diffusion_310p_patches(od)
    assert calls == []
