# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests: Ming-Image Design checkpoint dispatch (task-36b 310P 真机首跑缺陷修复).

Root cause fixed: a Ming-Image Design checkpoint (AR wrapper root config.json +
``transformer/`` + ``mlp/`` subfolders) resolved to the legacy ``MingImagePipeline``
(via the BailingMM2 architecture branch), which inherits ZImagePipeline and loads
the generic ``DistributedAutoencoderKL`` — shape-assert crash on the vendor
AutoencoderKLQwenImage VAE (weight [384] into parameter [64]).  Design checkpoints
must resolve to ``MingImageDiffusionPipeline``.

Fakes-first: no GPU, no real checkpoint — temp dir layouts + direct function call.
Run: python3 -m pytest tests/platforms/npu/test_310p_ming_image_dispatch.py --noconftest -o addopts=""
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_data_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_310p_test_data_module",
        _REPO_ROOT / "vllm_omni" / "diffusion" / "data.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def data_module(monkeypatch) -> types.ModuleType:
    # Isolate the module namespace so repeated loads are deterministic.
    monkeypatch.syspath_prepend(str(_REPO_ROOT))
    return _load_data_module()


def _write_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _design_layout(root: Path) -> None:
    """Ming-Image Design: AR wrapper config + transformer/ + mlp/ subfolders."""
    _write_config(
        root / "config.json",
        {
            "architectures": ["BailingMM2NativeForConditionalGeneration"],
            "model_type": "bailingmm_moe_v2_lite",
        },
    )
    _write_config(root / "transformer" / "config.json", {"_class_name": "DiffusionTransformer"})
    _write_config(root / "mlp" / "config.json", {"use_identity_mlp": True})


def _legacy_layout(root: Path) -> None:
    """Legacy ming_flash_omni image-gen: AR wrapper config, no transformer/ subdir."""
    _write_config(
        root / "config.json",
        {
            "architectures": ["BailingMM2NativeForConditionalGeneration"],
            "model_type": "ming_flash_omni",
        },
    )


class TestMingImageDesignLayoutDetection:
    def test_design_layout_detected(self, data_module, tmp_path: Path) -> None:
        _design_layout(tmp_path)
        assert data_module._looks_like_ming_image_design(str(tmp_path)) is True

    def test_legacy_layout_not_detected(self, data_module, tmp_path: Path) -> None:
        _legacy_layout(tmp_path)
        assert data_module._looks_like_ming_image_design(str(tmp_path)) is False

    def test_missing_mlp_not_detected(self, data_module, tmp_path: Path) -> None:
        _design_layout(tmp_path)
        (tmp_path / "mlp" / "config.json").unlink()
        assert data_module._looks_like_ming_image_design(str(tmp_path)) is False

    def test_missing_transformer_not_detected(self, data_module, tmp_path: Path) -> None:
        _design_layout(tmp_path)
        (tmp_path / "transformer" / "config.json").unlink()
        assert data_module._looks_like_ming_image_design(str(tmp_path)) is False

    def test_non_local_path_is_false(self, data_module) -> None:
        assert data_module._looks_like_ming_image_design("some-org/ming-image-design") is False
        assert data_module._looks_like_ming_image_design("/nonexistent/path/xyz") is False


class TestResolveModelClassName:
    def test_design_checkpoint_resolves_design_pipeline(self, data_module, tmp_path: Path) -> None:
        _design_layout(tmp_path)
        assert (
            data_module.resolve_model_class_name(str(tmp_path))
            == "MingImageDiffusionPipeline"
        )

    def test_legacy_checkpoint_resolves_legacy_pipeline(self, data_module, tmp_path: Path) -> None:
        _legacy_layout(tmp_path)
        assert data_module.resolve_model_class_name(str(tmp_path)) == "MingImagePipeline"

    def test_none_model_returns_none(self, data_module) -> None:
        assert data_module.resolve_model_class_name(None) is None

    def test_empty_model_returns_none(self, data_module) -> None:
        assert data_module.resolve_model_class_name("") is None
