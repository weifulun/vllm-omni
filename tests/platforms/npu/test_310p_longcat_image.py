# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the 310P LongCat-Image diffusion patch wiring.

Loads the patch module from source with a fake LongCat pipeline, so the
contract (FP16 forcing, VAE tiling, quant gating, CPU encoder pinning,
prompt-rewrite/encode device routing, post-load transformer placement,
attention-backend guard, registry override) is validated on CPU without
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
from torch import nn

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
    def __init__(self, hidden: torch.Tensor):
        self.hidden_states = [hidden]


class _FakeProcessorBatch:
    """Mapping-like processor output: supports ``.to(device)`` and
    ``**batch`` unpacking (``generate(**inputs, ...)``), like a
    BatchEncoding."""

    def __init__(self, input_ids: torch.Tensor, processor: _FakeProcessor):
        self.input_ids = input_ids
        self._processor = processor

    def to(self, device):
        self._processor.to_devices.append(device)
        return self

    def keys(self):
        return ["input_ids"]

    def __getitem__(self, key):
        if key == "input_ids":
            return self.input_ids
        raise KeyError(key)


class _FakeProcessor:
    """Qwen2.5-VL processor stand-in for the AR prompt-rewrite path."""

    def __init__(self):
        self.to_devices: list[torch.device] = []

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        return "templated prompt"

    def __call__(self, text: list[str], padding: bool, return_tensors: str):
        assert return_tensors == "pt"
        return _FakeProcessorBatch(torch.zeros((len(text), 8), dtype=torch.long), self)

    def batch_decode(self, ids, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> list[str]:
        assert skip_special_tokens
        return [f"rewritten {i}" for i in range(len(ids))]


class _TokenIds:
    """Tokenizer output: supports both ``.input_ids`` and
    ``["input_ids"]`` access (the pipeline uses the subscript form)."""

    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __getitem__(self, key):
        if key == "input_ids":
            return self.input_ids
        raise KeyError(key)


class _FakeTokenizer:
    def __call__(self, text, *, add_special_tokens: bool = True):
        assert add_special_tokens is False
        return _TokenIds([101, 2048, 320])

    def pad(self, batch, *, max_length, padding, return_attention_mask, return_tensors):
        assert return_tensors == "pt"
        return SimpleNamespace(
            input_ids=torch.zeros((1, max_length), dtype=torch.long),
            attention_mask=torch.ones((1, max_length), dtype=torch.long),
        )


class _FakeEncoder(nn.Module):
    """CPU-pinned Qwen2.5-VL stand-in: records ``.to()`` and forward/generate
    devices."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, dtype=torch.float32)
        self.to_calls: list[dict] = []
        self.call_devices: list[torch.device] = []
        self.generate_input_devices: list[torch.device] = []

    def to(self, *args, **kwargs):
        self.to_calls.append(kwargs)
        return super().to(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool):
        assert output_hidden_states
        self.call_devices.append(input_ids.device)
        hidden = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            8,
            dtype=torch.float16,
            device=input_ids.device,
        )
        return _FakeEncoderOut(hidden)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.generate_input_devices.append(input_ids.device)
        return torch.cat(
            [
                input_ids,
                torch.full((input_ids.shape[0], 4), 999, dtype=torch.long, device=input_ids.device),
            ],
            dim=1,
        )


class _FakeVae(nn.Module):
    def __init__(self, scaling_factor: float, shift_factor: float):
        super().__init__()
        self.linear = nn.Linear(4, 4, dtype=torch.float16)
        self.config = SimpleNamespace(scaling_factor=scaling_factor, shift_factor=shift_factor)
        self.use_tiling = True

    @property
    def dtype(self):
        return self.linear.weight.dtype


class _FakeScheduler:
    def __init__(self):
        self.config = {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }


class _RecordingTransformer(nn.Module):
    """DiT stand-in with real FP16 params; records ``.to(device=...)`` calls
    without actually moving (the CPU test host has no NPU backend)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, dtype=torch.float16)
        self.to_calls: list[torch.device] = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is not None:
            self.to_calls.append(device)
        return self


class FakeLongCatImagePipeline(nn.Module):
    """Stand-in for the built-in LongCat pipeline: same attribute surface the
    310P subclass touches, devices resolved to CPU on the test host."""

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.device = torch.device("cpu")  # stands in for the NPU device
        self.tokenizer_max_length = 512
        self.text_encoder = _FakeEncoder()
        self.text_processor = _FakeProcessor()
        self.tokenizer = _FakeTokenizer()
        self.prompt_template_encode_prefix = "<prefix>"
        self.prompt_template_encode_suffix = "<suffix>"
        self.transformer = _RecordingTransformer()
        self.vae = _FakeVae(scaling_factor=0.5, shift_factor=0.25)
        self.scheduler = _FakeScheduler()
        self.vae_scale_factor = 8
        self.default_sample_size = 128
        self._loaded_weights: set[str] = set()

    def rewire_prompt(self, prompt, device):
        """Generic-pipeline stand-in: moves the text encoder onto the target
        device at request time.  The 310P override must shadow this under the
        exact same name (a typo dispatches here and un-pins the encoder)."""
        self.base_rewire_called = True
        self.text_encoder.to(device)
        return prompt

    def load_weights(self, weights) -> set[str]:
        for name, _tensor in weights:
            self._loaded_weights.add(name)
        return self._loaded_weights


class _FakeDiffusionOutput:
    pass


def _install_longcat_patch_fakes(monkeypatch: pytest.MonkeyPatch, registry_calls: list[dict]):
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
        "vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image",
        LongCatImagePipeline=FakeLongCatImagePipeline,
        calculate_shift=lambda *a, **k: 0.5,
        get_longcat_image_post_process_func=lambda od_config: SimpleNamespace(name="post"),
        get_prompt_language=lambda prompt: "en",
        retrieve_timesteps=lambda scheduler, steps, device, **k: (torch.zeros(steps), steps),
        split_quotation=lambda prompt: [(prompt, False)],
    )
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.data", DiffusionOutput=_FakeDiffusionOutput)
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models.longcat_image")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.registry",
        register_diffusion_model=register_diffusion_model,
    )
    # The patch module reads the AR-rewrite system prompts from diffusers at
    # import time; fake the submodule so the contract test stays hermetic
    # across host diffusers versions (the host may predate longcat_image).
    _install_fake_module(monkeypatch, "diffusers")
    _install_fake_module(monkeypatch, "diffusers.pipelines")
    _install_fake_module(monkeypatch, "diffusers.pipelines.longcat_image")
    _install_fake_module(
        monkeypatch,
        "diffusers.pipelines.longcat_image.system_messages",
        SYSTEM_PROMPT_EN="EN PROMPT",
        SYSTEM_PROMPT_ZH="ZH PROMPT",
    )
    return fake_pipeline


def _load_longcat_patch(monkeypatch: pytest.MonkeyPatch):
    registry_calls: list[dict] = []
    fake_pipeline = _install_longcat_patch_fakes(monkeypatch, registry_calls)
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "longcat_image.py"
    module = _load_source_module("vllm_omni_test_310p_longcat_patch", path)
    return module, fake_pipeline, registry_calls


def _install_310p_registry_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare package fakes so the registry's ``from vllm_omni.platforms.npu.
    _310p.patch.attention import ...`` resolves to the leaf fakes installed
    by the individual tests (leaf sys.modules hits skip parent imports)."""
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch")


def _make_od_config(**overrides) -> SimpleNamespace:
    base = dict(
        dtype=torch.bfloat16,
        vae_use_tiling=False,
        vae_use_slicing=False,
        quantization_config=None,
        lora_path=None,
        diffusion_attn_backend_per_role=None,
        diffusion_attn_backend_default=None,
        model_class_name="LongCatImagePipeline",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_pipe(module, monkeypatch: pytest.MonkeyPatch, **od_overrides):
    monkeypatch.delenv("LONGCATIMAGE_310P_ENCODER_ON_NPU", raising=False)
    return module._LongCatImage310P(od_config=_make_od_config(**od_overrides))


def test_apply_forces_fp16_and_vae_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, registry_calls = _load_longcat_patch(monkeypatch)
    od_config = _make_od_config()

    module.apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    assert len(registry_calls) == 1
    entry = registry_calls[0]
    assert entry["model_arch"] == "LongCatImagePipeline"
    assert entry["module_name"] == module.__name__
    assert entry["class_name"] == "_LongCatImage310P"
    assert entry["post_process_func_name"] == "get_longcat_image_post_process_func"


def test_apply_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, registry_calls = _load_longcat_patch(monkeypatch)
    od_config = _make_od_config()

    module.apply_diffusion_310p_patches(od_config)
    module.apply_diffusion_310p_patches(od_config)

    assert len(registry_calls) == 1


def test_apply_rejects_fp8_quant_but_warns_on_int8(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, calls = _load_longcat_patch(monkeypatch)
    od_config = _make_od_config(quantization_config=SimpleNamespace(method="fp8"))
    with pytest.raises(ValueError, match="fp8"):
        module.apply_diffusion_310p_patches(od_config)

    od_config = _make_od_config(quantization_config=SimpleNamespace(method="int8"))
    module.apply_diffusion_310p_patches(od_config)
    assert len(calls) == 1


def test_apply_rejects_mindie_sd_only_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    for backend in ("FLASH_ATTN", "RAINFUSION_ATTN", "FLASH_ATTN_HUB"):
        od_config = _make_od_config(diffusion_attn_backend_default=SimpleNamespace(backend=backend))
        with pytest.raises(ValueError, match="MindIE-SD"):
            module.apply_diffusion_310p_patches(od_config)


def test_apply_pins_default_backend_to_torch_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform's default (None) backend resolution must land on
    TORCH_SDPA even on an image that shipped mindiesd."""
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)

    class _FakeBackendEnum:
        class TORCH_SDPA:
            @staticmethod
            def get_path():
                return "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"

    class _FakeNPUOmniPlatform:
        @classmethod
        def get_diffusion_attn_backend_cls(cls, selected_backend, head_size, allow_trtllm_default=True):
            return f"orig:{selected_backend}"

    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu.platform",
        NPUOmniPlatform=_FakeNPUOmniPlatform,
    )
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.attention")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.attention.backends")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.attention.backends.registry",
        DiffusionAttentionBackendEnum=_FakeBackendEnum,
    )

    module.apply_diffusion_310p_patches(_make_od_config())

    resolved = _FakeNPUOmniPlatform.get_diffusion_attn_backend_cls(None, 128)
    assert resolved == "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
    # Explicit TORCH_SDPA selections pass through to the original resolver.
    assert _FakeNPUOmniPlatform.get_diffusion_attn_backend_cls("TORCH_SDPA", 128) == "orig:TORCH_SDPA"
    # Idempotent: a second apply does not wrap the guard twice.
    module.apply_diffusion_310p_patches(_make_od_config())
    assert _FakeNPUOmniPlatform.get_diffusion_attn_backend_cls(None, 128) == (
        "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
    )


def test_pipeline_pins_encoder_to_cpu_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    assert pipe._encoder_device == torch.device("cpu")
    assert pipe.text_encoder.to_calls == [{"device": torch.device("cpu"), "dtype": torch.float16}]
    # Encoder params are FP16 on CPU (the contract audit's requirement).
    for _name, param in pipe.text_encoder.named_parameters():
        assert param.dtype is torch.float16
        assert param.device.type == "cpu"


def test_pipeline_keeps_encoder_on_device_when_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    monkeypatch.setenv("LONGCATIMAGE_310P_ENCODER_ON_NPU", "1")
    pipe = module._LongCatImage310P(od_config=_make_od_config())

    assert pipe._encoder_device == pipe.device
    assert pipe.text_encoder.to_calls == []


def test_rewire_prompt_keeps_encoder_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    rewritten = pipe.rewire_prompt(["a cat"], pipe.device)

    # The processor batch is moved to the encoder (CPU) device — never the
    # pipeline (NPU) device — and generation runs there.
    assert pipe.text_processor.to_devices == [pipe._encoder_device]
    assert pipe.text_encoder.generate_input_devices == [pipe._encoder_device]
    assert rewritten == ["rewritten 0"]
    # rewire_prompt must not move the encoder module itself.
    assert pipe.text_encoder.to_calls == [{"device": torch.device("cpu"), "dtype": torch.float16}]


def test_rewire_prompt_override_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The override name must match the base method (``rewire_prompt``); a
    typo would silently dispatch to the generic implementation (the fake
    base), which moves the text encoder onto the NPU at request time."""
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)

    pipe = _make_pipe(module, monkeypatch)
    pipe.rewire_prompt(["a cat"], pipe.device)
    assert getattr(pipe, "base_rewire_called", False) is False


def test_encode_prompt_routes_through_encoder_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    embeds = pipe._encode_prompt(["a cat"])

    # The encoder saw the tokens on its pinned device, not the pipeline device.
    assert pipe.text_encoder.call_devices == [pipe._encoder_device]
    # Embeds come back on the pipeline device in the runtime FP16 dtype.
    assert embeds.device == pipe.device
    assert embeds.dtype is torch.float16


def test_scaled_timestep_is_value_identical_and_expandable(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)

    t = torch.tensor(1200.0)
    scaled = module._LongCatImage310P._310p_scaled_timestep(t, torch.float16, torch.device("cpu"))

    assert scaled.shape == (1,)
    assert scaled.dtype is torch.float16
    assert scaled.device == torch.device("cpu")
    assert float(scaled.item()) == pytest.approx(1.2, abs=2e-3)  # fp16 granularity
    # Batch expansion mirrors the upstream ``t.expand(latents.shape[0])``.
    expanded = scaled.expand(3)
    assert expanded.shape == (3,)
    assert torch.allclose(expanded, torch.full((3,), scaled.item(), dtype=torch.float16))


def test_cfg_normalize_runs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    # Case 1: ratio below 1.0 — the FP16 CPU division must be value-identical.
    noise_pred = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    comb_pred = torch.tensor([[4.0, 4.0]], dtype=torch.float16)
    expected_ratio = (
        torch.norm(noise_pred, dim=-1, keepdim=True) / (torch.norm(comb_pred, dim=-1, keepdim=True) + 1e-8)
    ).clamp(min=0.0, max=1.0)

    result = pipe.cfg_normalize_function(noise_pred, comb_pred, cfg_renorm_min=0.0)

    assert result.dtype is torch.float16
    assert result.device == comb_pred.device
    assert torch.allclose(result, (comb_pred * expected_ratio).half(), atol=2e-2)

    # Case 2: ratio above 1.0 — clamped to exactly 1.0 (comb_pred unchanged).
    noise_big = torch.tensor([[4.0, 4.0]], dtype=torch.float16)
    comb_small = torch.tensor([[1.0, 1.0]], dtype=torch.float16)
    result_clamped = pipe.cfg_normalize_function(noise_big, comb_small, cfg_renorm_min=0.0)
    assert torch.allclose(result_clamped, comb_small)


def test_cfg_normalize_respects_renorm_min_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    # ||noise|| / (||comb|| + 1e-8) = 1/8 = 0.125 — the cfg_renorm_min=0.3
    # floor must raise it to exactly 0.3 (with floor 0.0 it stays 0.125).
    noise_pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float16)
    comb_pred = torch.tensor([[4.0, 4.0, 4.0, 4.0]], dtype=torch.float16)

    result_floored = pipe.cfg_normalize_function(noise_pred, comb_pred, cfg_renorm_min=0.3)
    result_unfloored = pipe.cfg_normalize_function(noise_pred, comb_pred, cfg_renorm_min=0.0)

    assert torch.allclose(result_floored, comb_pred * torch.full((1, 1), 0.3, dtype=torch.float16), atol=2e-2)
    assert torch.allclose(result_unfloored, comb_pred * torch.full((1, 1), 0.125, dtype=torch.float16), atol=2e-2)


def test_load_weights_places_transformer_and_audits(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    weights = iter([("transformer.blocks.0.attn.to_q.weight", torch.zeros(4, 4, dtype=torch.float16))])
    loaded = pipe.load_weights(weights)

    # No device move when the transformer already sits on the pipeline device.
    assert pipe.transformer.to_calls == []
    assert loaded == {"transformer.blocks.0.attn.to_q.weight"}
    # The audit ran (no exception) against the FP16/CPU contract.
    for _name, param in pipe.transformer.named_parameters():
        assert param.dtype is torch.float16


def test_load_weights_moves_transformer_to_target_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader leaves the CPU-built transformer on host; the override must
    move it to the pipeline (NPU) device before the audit (the default
    loader path does not relocate unquantized modules)."""
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    audited: list[str] = []
    monkeypatch.setattr(pipe, "_310p_audit_components", lambda: audited.append("ok"))
    # Sentinel stand-in for the NPU target (the CPU test host has no NPU
    # device type registered with torch, so a torch.device("npu") is
    # unconstructible; the placement check is a plain inequality).
    npu_target = SimpleNamespace(type="npu", index=0)
    pipe.device = npu_target

    pipe.load_weights(iter([]))

    assert pipe.transformer.to_calls == [npu_target]
    assert audited == ["ok"]


def test_vae_inv_scale_reciprocal(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _fake_pipeline, _calls = _load_longcat_patch(monkeypatch)
    pipe = _make_pipe(module, monkeypatch)

    assert pipe._310p_vae_inv_scale == pytest.approx(1.0 / 0.5)


def test_patch_module_exports_post_process_func(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fake_pipeline, _calls = _load_longcat_patch(monkeypatch)

    # _load_process_func resolves the post-process func from the registered
    # module; the override must expose the base implementation.
    assert module.get_longcat_image_post_process_func is fake_pipeline.get_longcat_image_post_process_func


def test_platform_wiring_calls_310p_patches() -> None:
    # Source-level pin: the NPU platform's diffusion runtime hook must stay
    # wired to the registry dispatch (import-level tests cannot reach
    # platform.py on a CPU host because of the vllm_ascend dependency).
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "platform.py"
    source = path.read_text(encoding="utf-8")
    assert "is_310p()" in source
    assert "disable_jit_compile()" in source
    # The call site must stay registry-generic: model branches register in
    # _310p.patch and never call their own patch module from platform.py
    # (COMPARISON-POLICY.md §5).
    assert "apply_diffusion_310p_patches(od_config)" in source
    assert "patch.longcat_image" not in source


def test_diffusion_registry_dispatches_by_model_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Load the patch registry (__init__.py) standalone: it has no heavy
    # top-level imports, so the dispatch contract is testable on CPU.
    init_path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    patch_init = _load_source_module("vllm_omni_test_310p_patch_init", init_path)

    calls: list[object] = []
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.longcat_image",
        apply_diffusion_310p_patches=lambda od: calls.append(od),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: None,
    )

    longcat_od = SimpleNamespace(model_class_name="LongCatImagePipeline")
    patch_init.apply_diffusion_310p_patches(longcat_od)
    assert calls == [longcat_od]

    # Unregistered pipelines are a no-op (they keep the generic 310P path).
    other_od = SimpleNamespace(model_class_name="StableDiffusionXLPipeline")
    patch_init.apply_diffusion_310p_patches(other_od)
    assert calls == [longcat_od]


def test_diffusion_registry_env_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_OMNI_310P_DIFFUSION_PATCHES=0 selects the generic 310P path (the
    COMPARISON-POLICY.md §1 level-2 same-seed reference run)."""
    init_path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    patch_init = _load_source_module("vllm_omni_test_310p_patch_init_env", init_path)

    calls: list[object] = []
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.longcat_image",
        apply_diffusion_310p_patches=lambda od: calls.append(od),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: None,
    )

    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    od = SimpleNamespace(model_class_name="LongCatImagePipeline")
    patch_init.apply_diffusion_310p_patches(od)
    assert calls == []


def test_skip_warmup_knob_survives_patch_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_OMNI_310P_SKIP_WARMUP must apply even with the patch registry
    disabled (the level-2 reference run still needs the 512x512 dummy
    warmup off on 310P)."""
    init_path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    patch_init = _load_source_module("vllm_omni_test_310p_patch_init_warmup", init_path)

    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.attention",
        maybe_patch_sdpa_math=lambda: None,
    )

    class _FakeDiffusionEngine:
        _original_dummy_run = staticmethod(lambda self: "warmup")

    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.diffusion_engine",
        DiffusionEngine=_FakeDiffusionEngine,
    )
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")

    monkeypatch.setenv("VLLM_OMNI_310P_SKIP_WARMUP", "1")
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    patch_init.apply_diffusion_310p_patches(SimpleNamespace(model_class_name="LongCatImagePipeline"))

    assert _FakeDiffusionEngine._dummy_run is not _FakeDiffusionEngine._original_dummy_run
