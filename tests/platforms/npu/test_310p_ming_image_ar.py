# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Ming-Image AR-side 310P patch (task-36b).

Loads ``vllm_omni/platforms/npu/_310p/patch/ming_image_ar.py`` and the 310P
patch registry dispatcher from source with fake
``vllm_omni.model_executor.models`` dependencies, so the pre-add RMSNorm
contract (pre-add vs sum-norm math, FP32 intermediates, shared weight,
backbone norm swap, registry dispatch on the stage-0 arch, idempotency) is
pinned without an NPU, vllm-ascend or a real checkpoint.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_ming_image` (diffusion side) and the
ming3 line's pre-add norm suite (task-27).
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
    if "." in name:
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and parent.__name__ == parent_name and getattr(parent, "__fake_", False):
            setattr(parent, child, module)
    module.__fake__ = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return module


#: Fakes-first dependencies of ``ming_image_ar.py`` (top-level modeling
#: import) and of the registry dispatcher branch (lazy ming_image_ar import
#: resolved from ``sys.modules``).
_PACKAGE_PARENTS = (
    "vllm_omni",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.models",
    "vllm_omni.model_executor.models.ming_flash_omni",
    "vllm_omni.model_executor.models.ming_image",
)


class _FakeBailingMoeV2ForCausalLM:
    """Stand-in for the real backbone class (no vLLM machinery on CPU)."""

    constructed: list["SimpleNamespace"] = []

    def __init__(self, backbone=None, *args, **kwargs):
        self.model = backbone


def _install_ar_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    for parent in _PACKAGE_PARENTS:
        _install_fake_module(monkeypatch, parent)
    captured: dict[str, object] = {}

    fake_model_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.ming_image.model",
        BailingMoeV2ForCausalLM=_FakeBailingMoeV2ForCausalLM,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.ming_flash_omni.modeling_bailing_moe_v2",
        BailingMoeV2ForCausalLM=_FakeBailingMoeV2ForCausalLM,
    )
    captured["fake_model_module"] = fake_model_module
    return captured


def _load_ar_module(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    module = _load_source_module("mingimage_ar_310p_under_test", patch_dir / "ming_image_ar.py")
    module._PATCHED = False
    return module


def _load_registry_module(monkeypatch: pytest.MonkeyPatch):
    patch_dir = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"
    return _load_source_module("mingimage_ar_registry_under_test", patch_dir / "__init__.py")


class _FakeNorm:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(8))
        self.variance_epsilon = 1e-6


class _FakeLayer:
    def __init__(self):
        self.input_layernorm = _FakeNorm()
        self.post_attention_layernorm = _FakeNorm()


class _FakeBackbone:
    def __init__(self, n_layers: int = 3):
        self.layers = [_FakeLayer() for _ in range(n_layers)]


def _hf_preadd_reference(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """HF-transformers pre-add RMSNorm reference math (FP32)."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# _MingPreAddRMSNorm310P math contract
# ---------------------------------------------------------------------------


def test_preadd_norm_without_residual_matches_hf_math(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    x = torch.randn(4, 8).half()
    weight = torch.nn.Parameter(torch.ones(8))
    norm = module._MingPreAddRMSNorm310P(weight, 1e-6)

    out = norm(x)
    expected = _hf_preadd_reference(x, weight, 1e-6)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


def test_preadd_norm_normalizes_pre_add_input_not_sum(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    torch.manual_seed(42)
    x = torch.randn(2, 8).half() * 30.0
    residual = torch.randn(2, 8).half() * 30.0
    weight = torch.nn.Parameter(torch.ones(8))
    norm = module._MingPreAddRMSNorm310P(weight, 1e-6)

    normed, residual_new = norm(x, residual)
    # Pre-add: the normalized branch is rms(x), NOT rms(x + residual).
    expected_normed = _hf_preadd_reference(x, weight, 1e-6)
    torch.testing.assert_close(normed, expected_normed, atol=1e-3, rtol=1e-3)
    sum_normed = _hf_preadd_reference((x.float() + residual.float()).half(), weight, 1e-6)
    assert not torch.allclose(normed, sum_normed, atol=1e-2, rtol=1e-2)
    # Residual stream updated as x + residual (order-equivalent, FP32 math).
    expected_res = (x.float() + residual.float()).to(torch.float16)
    torch.testing.assert_close(residual_new, expected_res, atol=1e-3, rtol=1e-3)


def test_preadd_norm_weight_is_shared_not_copied(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    original = _FakeNorm()
    norm = module._MingPreAddRMSNorm310P(original.weight, original.variance_epsilon)
    assert norm.weight is original.weight
    assert norm.variance_epsilon == original.variance_epsilon


def test_preadd_norm_math_runs_in_fp32(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    # Inputs whose FP16 accumulation would round differently than FP32.
    x = torch.full((1, 8), 255.0, dtype=torch.float16)
    weight = torch.nn.Parameter(torch.ones(8))
    norm = module._MingPreAddRMSNorm310P(weight, 1e-6)
    out = norm(x)
    # rms(255) = 1.0 exactly in either precision; the check is that the
    # result equals the FP32 reference bit-for-bit after the final cast.
    expected = (x.float() / x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()).to(torch.float16)
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# backbone norm swap
# ---------------------------------------------------------------------------


def test_install_swaps_all_decoder_layer_norms(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    backbone = _FakeBackbone(n_layers=3)
    model = module._BailingMoeV2ForCausalLM310P(backbone=backbone)

    assert model.model is backbone
    for lyr in backbone.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(lyr, name)
            assert isinstance(mod, module._MingPreAddRMSNorm310P)
            assert mod.variance_epsilon == 1e-6


def test_install_preserves_original_weights(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    backbone = _FakeBackbone(n_layers=2)
    originals = [
        getattr(lyr, name).weight
        for lyr in backbone.layers
        for name in ("input_layernorm", "post_attention_layernorm")
    ]
    module._BailingMoeV2ForCausalLM310P(backbone=backbone)
    for lyr, weight in zip(backbone.layers, [originals[i * 2 : (i + 1) * 2] for i in range(2)]):
        for name, expected in zip(("input_layernorm", "post_attention_layernorm"), weight):
            assert getattr(lyr, name).weight is expected


def test_install_noop_without_backbone_layers(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    # No backbone attribute at all: must not raise.
    module._BailingMoeV2ForCausalLM310P(backbone=None)
    # Backbone without .layers: must not raise.
    module._BailingMoeV2ForCausalLM310P(backbone=SimpleNamespace(layers=None))


def test_install_skips_missing_norm_attributes(monkeypatch: pytest.MonkeyPatch):
    _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)

    backbone = _FakeBackbone(n_layers=1)
    del backbone.layers[0].post_attention_layernorm
    module._BailingMoeV2ForCausalLM310P(backbone=backbone)

    lyr = backbone.layers[0]
    assert isinstance(lyr.input_layernorm, module._MingPreAddRMSNorm310P)
    assert not hasattr(lyr, "post_attention_layernorm")


# ---------------------------------------------------------------------------
# registry dispatch + idempotency
# ---------------------------------------------------------------------------


def test_apply_patches_swaps_ming_image_module(monkeypatch: pytest.MonkeyPatch):
    fakes = _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)
    fake_model_module = fakes["fake_model_module"]

    assert not module._PATCHED
    module.apply_ming_image_ar_patches()
    assert module._PATCHED
    assert fake_model_module.BailingMoeV2ForCausalLM is module._BailingMoeV2ForCausalLM310P

    # Idempotent: a second call is a no-op.
    module.apply_ming_image_ar_patches()
    assert fake_model_module.BailingMoeV2ForCausalLM is module._BailingMoeV2ForCausalLM310P


def test_registry_dispatch_fires_for_ming_image_stage0_arch(monkeypatch: pytest.MonkeyPatch):
    fakes = _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)
    # The dispatcher imports the patch module by real name; point the import
    # at the source-loaded module under test.
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.ming_image_ar", module)
    registry = _load_registry_module(monkeypatch)

    assert not module._PATCHED
    registry.apply_model_patches(SimpleNamespace(model_arch="MingImageForConditionalGeneration"))
    assert module._PATCHED
    assert fakes["fake_model_module"].BailingMoeV2ForCausalLM is module._BailingMoeV2ForCausalLM310P


def test_registry_dispatch_noop_for_other_archs(monkeypatch: pytest.MonkeyPatch):
    fakes = _install_ar_fakes(monkeypatch)
    module = _load_ar_module(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.ming_image_ar", module)
    registry = _load_registry_module(monkeypatch)

    for arch in ("MingFlashOmniThinkerForConditionalGeneration", "BailingMoeV2ForCausalLM", "MingImageDiffusionPipeline"):
        registry.apply_model_patches(SimpleNamespace(model_arch=arch))
        assert not module._PATCHED
    assert fakes["fake_model_module"].BailingMoeV2ForCausalLM is _FakeBailingMoeV2ForCausalLM


def test_dispatcher_constant_is_stage0_arch(monkeypatch: pytest.MonkeyPatch):
    registry = _load_registry_module(monkeypatch)
    # The branch must key on the pipeline-declared stage-0 arch (ming_image
    # pipeline: model_arch="MingImageForConditionalGeneration").
    assert registry._MING_IMAGE_AR_ARCH == "MingImageForConditionalGeneration"
