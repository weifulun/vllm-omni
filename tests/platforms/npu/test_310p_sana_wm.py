# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the 310P SANA-WM diffusion patch wiring.

Loads the patch module from source with a fake SANA-WM pipeline, so the
contract (FP16 forcing, VAE tiling, quant neutralization, CPU Gemma encoder
pinning, prompt-embeds device routing, registry override) is validated on CPU
without diffusers / transformers / vllm_ascend.
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


class _FakeEncoded(dict):
    """Dict-subclass stand-in for a transformers BatchEncoding: supports
    ``**``-expansion into the encoder AND attribute access (``.to``,
    ``.attention_mask``), like the real object."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self._to_calls: list[torch.device] = []

    def to(self, device: torch.device):
        self["input_ids"] = self.input_ids.to(device)
        self["attention_mask"] = self.attention_mask.to(device)
        self.input_ids = self["input_ids"]
        self.attention_mask = self["attention_mask"]
        self._to_calls.append(device)
        return self


class _FakeTokenizer:
    def __init__(self, *, max_length: int):
        self._max_length = max_length

    def encode(self, text: str) -> list[int]:
        assert isinstance(text, str)
        return [1] * 4

    def __call__(
        self,
        texts: list[str],
        *,
        padding: str,
        max_length: int,
        truncation: bool,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> _FakeEncoded:
        assert return_tensors == "pt"
        assert padding == "max_length"
        assert truncation is True
        return _FakeEncoded(
            input_ids=torch.zeros((1, max_length), dtype=torch.long),
            attention_mask=torch.ones((1, max_length), dtype=torch.long),
        )


class _FakeGemmaEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_calls: list[dict] = []
        self.call_devices: list[torch.device] = []

    def to(self, *args, **kwargs):
        self.to_calls.append(kwargs)
        return self

    def forward(self, **inputs):
        self.call_devices.append(inputs["input_ids"].device)
        return SimpleNamespace(last_hidden_state=torch.zeros(1, 300, 2304, dtype=torch.float32))


class FakeSanaWmPipeline(torch.nn.Module):
    """Stand-in for the built-in SANA-WM pipeline: same attribute surface the
    310P subclass touches, devices resolved to CPU on the test host."""

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.device = torch.device("cpu")  # stands in for the NPU device
        self.tokenizer = _FakeTokenizer(max_length=300)
        self.text_encoder = _FakeGemmaEncoder()
        self.sana_wm_config = SimpleNamespace(chi_prompt=[], model_max_length=300)
        self._last_prompt_attention_mask = None

    def _stage1_prompt_text(self, prompt: dict) -> str:
        return str(prompt.get("prompt") or "")

    def _load_stage1_text_encoder(self, *, dtype: torch.dtype) -> None:
        if self.text_encoder is None:
            self.text_encoder = _FakeGemmaEncoder()

    def _ensure_stage1_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.text_encoder is None:
            self._load_stage1_text_encoder(dtype=dtype)
        self.text_encoder.to(device=device, dtype=dtype)


def _install_sana_wm_patch_fakes(monkeypatch: pytest.MonkeyPatch, registry_calls: list[dict]):
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

    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm",
        SanaWmPipeline=FakeSanaWmPipeline,
        SANA_WM_STAGE1_PROMPT_CHANNELS=2304,
        get_sana_wm_pre_process_func=lambda od_config: (lambda req: req),
    )
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models.sana_wm")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.registry",
        register_diffusion_model=register_diffusion_model,
    )


def _load_sana_wm_patch(monkeypatch: pytest.MonkeyPatch):
    registry_calls: list[dict] = []
    _install_sana_wm_patch_fakes(monkeypatch, registry_calls)
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "sana_wm.py"
    module = _load_source_module("vllm_omni_test_310p_sana_wm_patch", path)
    return module, registry_calls


def _make_od_config(**overrides) -> SimpleNamespace:
    base = dict(
        dtype=torch.bfloat16,
        vae_use_tiling=False,
        vae_use_slicing=False,
        diffusion_kv_cache_dtype=None,
        model_class_name="SanaWmPipeline",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_apply_patches_forces_fp16_and_vae_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU", raising=False)
    module, registry_calls = _load_sana_wm_patch(monkeypatch)
    od_config = _make_od_config()

    module.apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    assert od_config.diffusion_kv_cache_dtype is None
    assert len(registry_calls) == 1
    entry = registry_calls[0]
    assert entry["model_arch"] == "SanaWmPipeline"
    assert entry["module_name"] == module.__name__
    assert entry["class_name"] == "_SanaWm310P"
    # pre/post process funcs keep the existing entries (None = untouched).
    assert entry["pre_process_func_name"] is None
    assert entry["post_process_func_name"] is None


def test_apply_patches_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU", raising=False)
    module, registry_calls = _load_sana_wm_patch(monkeypatch)
    od_config = _make_od_config()

    module.apply_diffusion_310p_patches(od_config)
    module.apply_diffusion_310p_patches(od_config)

    assert len(registry_calls) == 1


def test_apply_patches_neutralizes_kv_quant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU", raising=False)
    module, _calls = _load_sana_wm_patch(monkeypatch)
    od_config = _make_od_config(diffusion_kv_cache_dtype="fp8")

    module.apply_diffusion_310p_patches(od_config)

    assert od_config.diffusion_kv_cache_dtype == "auto"


def test_pipeline_pins_gemma_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU", raising=False)
    module, _calls = _load_sana_wm_patch(monkeypatch)
    pipe = module._SanaWm310P(od_config=_make_od_config(dtype=torch.float16))

    assert pipe._encoder_device() == torch.device("cpu")
    assert pipe.text_encoder.to_calls == [{"device": torch.device("cpu"), "dtype": torch.float16}]


def test_pipeline_keeps_encoder_on_device_when_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU", "1")
    module, _calls = _load_sana_wm_patch(monkeypatch)
    pipe = module._SanaWm310P(od_config=_make_od_config(dtype=torch.float16))

    assert pipe._encoder_device() == pipe.device
    assert pipe.text_encoder.to_calls == []


def test_ensure_stage1_text_encoder_stays_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU", raising=False)
    module, _calls = _load_sana_wm_patch(monkeypatch)
    pipe = module._SanaWm310P(od_config=_make_od_config(dtype=torch.float16))

    # The request path asks for the NPU device; the 310P override must not
    # re-place the encoder there.
    pipe._ensure_stage1_text_encoder(device=pipe.device, dtype=torch.float16)

    assert pipe.text_encoder.to_calls == [
        {"device": torch.device("cpu"), "dtype": torch.float16},
        {"device": torch.device("cpu"), "dtype": torch.float16},
    ]


def test_prompt_embeds_route_through_encoder_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU", raising=False)
    module, _calls = _load_sana_wm_patch(monkeypatch)
    pipe = module._SanaWm310P(od_config=_make_od_config(dtype=torch.float16))

    prompt_embeds = pipe._native_prompt_embeds(
        {"prompt": "a slow forward camera move"},
        device=pipe.device,
        dtype=torch.float16,
    )

    # The encoder saw the tokens on its pinned device (CPU), not the pipeline
    # device; embeds come back on the pipeline device in the runtime FP16.
    assert pipe.text_encoder.call_devices == [torch.device("cpu")]
    assert prompt_embeds.device == pipe.device
    assert prompt_embeds.dtype is torch.float16
    assert prompt_embeds.shape == (1, 300, 2304)
    # All-ones attention mask collapses to None (mask-free cross-attention).
    assert pipe._last_prompt_attention_mask is None


def test_patch_module_reexports_pre_process_func(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _calls = _load_sana_wm_patch(monkeypatch)

    # The pre-process func stays resolvable from the built-in pipeline module
    # (registry registration passes None = keep existing entry); the 310P
    # module re-exports it so module-resolution from the registered module
    # name keeps working.
    fake_pipeline = sys.modules["vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm"]
    assert module.get_sana_wm_pre_process_func is fake_pipeline.get_sana_wm_pre_process_func


def test_platform_wiring_registry_generic() -> None:
    # Source-level pin: the NPU platform's diffusion runtime hook must stay
    # registry-generic (COMPARISON-POLICY.md §5): model branches register in
    # _310p.patch and never touch platform.py.
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "platform.py"
    source = path.read_text(encoding="utf-8")
    assert "is_310p()" in source
    assert "disable_jit_compile()" in source
    assert "apply_diffusion_310p_patches(od_config)" in source
    assert "patch.sana_wm" not in source


def test_diffusion_registry_dispatches_sana_wm(monkeypatch: pytest.MonkeyPatch) -> None:
    # Load the patch registry (__init__.py) standalone and pin the SANA-WM
    # dispatch entry.
    init_path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    patch_init = _load_source_module("vllm_omni_test_310p_sana_wm_patch_init", init_path)

    calls: list[object] = []
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.sana_wm",
        apply_diffusion_310p_patches=lambda od: calls.append(od),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.sdxl",
        apply_diffusion_310p_patches=lambda od: calls.append(("sdxl", od)),
    )
    # Fake the shared attention module: the real one needs torch_npu (NPU
    # driver), so the dispatch contract is pinned on a CPU host instead —
    # the attention patch has its own contract tests (test_310p_sdxl.py).
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch",
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda *a, **k: None,
    )
    # The dispatcher touches shared knobs; keep them inert on the CPU host.
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "1")
    monkeypatch.setenv("VLLM_OMNI_310P_SKIP_WARMUP", "0")

    sana_od = SimpleNamespace(model_class_name="SanaWmPipeline")
    patch_init.apply_diffusion_310p_patches(sana_od)
    assert calls == [sana_od]

    sdxl_od = SimpleNamespace(model_class_name="StableDiffusionXLPipeline")
    patch_init.apply_diffusion_310p_patches(sdxl_od)
    assert calls[-1] == ("sdxl", sdxl_od)

    # Unregistered pipelines are a no-op (they keep the generic 310P path).
    other_od = SimpleNamespace(model_class_name="ZImagePipeline")
    patch_init.apply_diffusion_310p_patches(other_od)
    assert len(calls) == 2


def test_diffusion_registry_env_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_OMNI_310P_DIFFUSION_PATCHES=0 selects the generic 310P path (the
    COMPARISON-POLICY.md §1 level-2 same-seed reference run)."""
    init_path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    patch_init = _load_source_module("vllm_omni_test_310p_sana_wm_patch_init_env", init_path)

    calls: list[object] = []
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.sana_wm",
        apply_diffusion_310p_patches=lambda od: calls.append(od),
    )

    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    od = SimpleNamespace(model_class_name="SanaWmPipeline")
    patch_init.apply_diffusion_310p_patches(od)
    assert calls == []
