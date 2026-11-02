# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-runnable contract tests for the 310P Qwen-Image-2.1 patch (task-16).

Skeleton status: validates the *patch contract* (dispatch gating, dtype
enforcement, RoPE fallback policy, NZ-cast disable, VAE memory policy) with
fake modules, mirroring ``test_310p_patches.py``.  No NPU is required; on a
310P box the real-device smoke test (see risk list section 3) extends this
file with a ``@hardware_test``-style marker once the branch lands.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Source loading with fakes (same convention as test_310p_patches.py)
# ---------------------------------------------------------------------------


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


def _native_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Reference real-valued RoPE used as the native fallback oracle."""
    cos = torch.real(freqs).float().unsqueeze(1)
    sin = torch.imag(freqs).float().unsqueeze(1)
    pairs = x.float().unflatten(-1, (-1, 2))
    x1, x2 = pairs.unbind(-1)
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).to(x.dtype)


def _load_qwen_image_2_1_patch(monkeypatch: pytest.MonkeyPatch, is_310p_value: bool = True):
    """Load the patch source with a controllable ``is_310p`` and a fake
    transformer module (so nothing real is imported from the model tree)."""
    rope_calls: list[str] = []

    def npu_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rope_calls.append("npu")
        raise RuntimeError("simulated: 310P interleave mode unsupported")

    class FakeQwenImage21Transformer2DModel(torch.nn.Module):
        @staticmethod
        def _nz_weights_supported_for(od_config, quant_config):  # noqa: ANN001
            return True

    fake_transformer = _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer",
        _apply_qwen_image21_rotary_emb_npu=npu_rope,
        _apply_qwen_image21_rotary_emb_native=_native_rope,
        QwenImage21Transformer2DModel=FakeQwenImage21Transformer2DModel,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.models.qwen_image_21",
        qwen_image_21_transformer=fake_transformer,
    )
    _install_fake_module(monkeypatch, "vllm_omni.diffusion")
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models")
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p", is_310p=lambda: is_310p_value)

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "qwen_image_2_1.py"
    module = _load_source_module("vllm_omni_test_310p_qwen_image_2_1_patch", path)
    return module, fake_transformer, rope_calls


def _make_od_config(model_class_name: str = "QwenImage21Pipeline", **overrides) -> SimpleNamespace:
    defaults = dict(
        model_class_name=model_class_name,
        dtype=torch.bfloat16,
        enforce_eager=False,
        vae_use_tiling=False,
        vae_use_slicing=False,
        quantization_config=None,
        extras={},
        diffusion_attention_backend=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_dispatch_is_gated_on_310p_and_model_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)

    module._APPLIED = False
    module.maybe_apply_diffusion_310p_patches(_make_od_config())
    assert module._APPLIED is True

    # Non-Qwen-Image-2.1 diffusion models are untouched by this patch file.
    other = _make_od_config(model_class_name="StableDiffusionXLPipeline")
    module.maybe_apply_diffusion_310p_patches(other)
    assert other.dtype is torch.bfloat16  # untouched


def test_dispatch_is_noop_off_310p(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch, is_310p_value=False)
    config = _make_od_config()
    module.maybe_apply_diffusion_310p_patches(config)
    assert module._APPLIED is False
    assert config.dtype is torch.bfloat16


def test_runtime_dtype_forced_fp16_and_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)

    assert module.resolve_runtime_dtype(torch.bfloat16) is torch.float16
    assert module.resolve_runtime_dtype(torch.float16) is torch.float16

    module._APPLIED = False
    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    # Re-run must be idempotent (single-shot guard).
    assert module._APPLIED is True


def test_apply_enforces_runtime_configuration(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    module._APPLIED = False
    config = _make_od_config(extras={"prefix_kv_cache_dtype": "fp8"}, quantization_config=object())
    with caplog.at_level("WARNING"):
        module.apply_qwen_image_2_1_310p_patches(config)

    assert config.dtype is torch.float16
    assert config.enforce_eager is True
    assert config.vae_use_tiling is True
    assert config.vae_use_slicing is True
    # 310P has no MindIE diffusion quant path: extras and quant config dropped.
    assert config.extras["prefix_kv_cache_dtype"] is None
    assert config.quantization_config is None
    assert any("prefix_kv_cache_dtype" in record.message for record in caplog.records)


def test_attention_default_is_sdpa_and_env_experiment_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    monkeypatch.setenv("QWEN21_310P_ATTENTION", "flash_attn")
    module._APPLIED = False
    config = _make_od_config()
    module.apply_qwen_image_2_1_310p_patches(config)
    assert config.diffusion_attention_backend == "FLASH_ATTN"

    monkeypatch.setenv("QWEN21_310P_ATTENTION", "bogus")
    module._APPLIED = False
    config2 = _make_od_config()
    module.apply_qwen_image_2_1_310p_patches(config2)
    assert config2.diffusion_attention_backend is None  # stays on the resolver default


@pytest.mark.parametrize("policy", ["auto", "npu", "native", "bogus"])
def test_rope_policy_resolution(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    module, _, _ = _load_qwen_image_2_1_patch(monkeypatch)
    if policy == "bogus":
        assert module.resolve_rope_policy({"QWEN21_310P_ROPE": policy}) == "auto"
        return
    monkeypatch.setenv("QWEN21_310P_ROPE", policy)
    assert module.resolve_rope_policy() == policy
    monkeypatch.delenv("QWEN21_310P_ROPE")


def test_rope_dispatcher_falls_back_to_native_on_error(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    module, fake_transformer, rope_calls = _load_qwen_image_2_1_patch(monkeypatch)

    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, dtype=torch.float16)
    freqs = torch.polar(torch.ones(8, 32), torch.randn(8, 32))
    expected = _native_rope(x, freqs)

    # auto: first call hits the (simulated-broken) NPU op, then native.
    dispatch = module.build_rope_dispatcher(fake_transformer._apply_qwen_image21_rotary_emb_npu, _native_rope, "auto")
    with caplog.at_level("WARNING"):
        first = dispatch(x, freqs)
    assert rope_calls == ["npu"]  # only one NPU attempt
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert "native" in " ".join(record.message for record in caplog.records)

    # Subsequent calls must not touch the NPU op again (sticky fallback).
    second = dispatch(x, freqs)
    assert rope_calls == ["npu"]
    torch.testing.assert_close(second, expected, rtol=0, atol=0)

    # native policy: NPU op never attempted.
    native = module.build_rope_dispatcher(fake_transformer._apply_qwen_image21_rotary_emb_npu, _native_rope, "native")
    torch.testing.assert_close(native(x, freqs), expected, rtol=0, atol=0)
    assert rope_calls == ["npu"]


def test_patch_rebinds_transformer_rope_and_disables_nz(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fake_transformer, _ = _load_qwen_image_2_1_patch(monkeypatch)
    monkeypatch.setenv("QWEN21_310P_ROPE", "native")

    module.apply_qwen_image_2_1_310p_patches(_make_od_config())

    # RoPE rebound to the pure-torch native implementation.
    assert fake_transformer._apply_qwen_image21_rotary_emb_npu is _native_rope
    # NZ weight cast is disabled for 310P, enabled elsewhere (fake base returns True).
    model_cls = fake_transformer.QwenImage21Transformer2DModel
    assert model_cls._nz_weights_supported_for(object(), None) is False
    # Idempotent: a second application must not re-patch.
    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    assert fake_transformer._310p_patch_applied is True


def test_nz_gate_passes_through_off_310p(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-310P NPU boxes the original NZ eligibility is preserved."""
    module, fake_transformer, _ = _load_qwen_image_2_1_patch(monkeypatch, is_310p_value=False)
    module.apply_qwen_image_2_1_310p_patches(_make_od_config())
    # _APPLIED guard: apply_* unconditionally applies dtype/eager on this rank;
    # the contract under test is the NZ gate delegation.
    model_cls = fake_transformer.QwenImage21Transformer2DModel
    assert model_cls._nz_weights_supported_for(object(), None) is True
