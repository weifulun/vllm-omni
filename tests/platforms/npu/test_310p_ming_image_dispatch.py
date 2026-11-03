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


def _load_data_module(monkeypatch) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_310p_test_data_module",
        _REPO_ROOT / "vllm_omni" / "diffusion" / "data.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    # Register before exec: data.py has top-level @dataclass decorators whose
    # resolution path does sys.modules[cls.__module__].
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def data_module(monkeypatch) -> types.ModuleType:
    # Isolate the module namespace so repeated loads are deterministic.
    monkeypatch.syspath_prepend(str(_REPO_ROOT))
    return _load_data_module(monkeypatch)


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


class TestEnrichConfigDispatch:
    """enrich_config must not clobber Design dispatch (second defect site).

    The resolver-level fix alone is not enough: OmniDiffusionConfig.enrich_config
    has an inline BailingMM2 mapping that unconditionally overwrote
    model_class_name with the legacy MingImagePipeline.  These tests drive the
    real enrich path on fake checkpoint layouts (no NPU, no GPU).
    """

    def _enrich(self, data_module, model_path: Path):
        cfg = data_module.OmniDiffusionConfig(
            model=str(model_path),
            model_class_name=None,
            diffusion_load_format="default",
        )
        cfg.enrich_config()
        return cfg

    def test_design_layout_enrich_routes_design_pipeline(
        self, data_module, tmp_path: Path
    ) -> None:
        _design_layout(tmp_path)
        cfg = self._enrich(data_module, tmp_path)
        assert cfg.model_class_name == "MingImageDiffusionPipeline"
        # Component DiT config must carry the vendor _class_name marker so
        # _validate_variant_config can confirm the Design variant.
        assert cfg.tf_model_config is not None
        assert getattr(cfg.tf_model_config, "_class_name", None) == "DiffusionTransformer"

    def test_design_layout_enrich_keeps_explicit_class(
        self, data_module, tmp_path: Path
    ) -> None:
        _design_layout(tmp_path)
        cfg = data_module.OmniDiffusionConfig(
            model=str(tmp_path),
            model_class_name="MingImageDiffusionPipeline",
            diffusion_load_format="default",
        )
        cfg.enrich_config()
        assert cfg.model_class_name == "MingImageDiffusionPipeline"

    def test_legacy_layout_enrich_routes_legacy_pipeline(
        self, data_module, tmp_path: Path
    ) -> None:
        _legacy_layout(tmp_path)
        cfg = self._enrich(data_module, tmp_path)
        assert cfg.model_class_name == "MingImagePipeline"


# ---------------------------------------------------------------------------
# Placement surface (recipe A' = TP3 AR + TP1 DiT, lead O1 ruling 11-03).
# Fakes-first: parses the vendored staging yaml snapshot; no device contact.
# Keep in sync with agents/model-mingimage/mingimage-310p.yaml (diff at stage).
# ---------------------------------------------------------------------------

def _load_recipe_fixture() -> dict:
    import yaml

    here = Path(__file__).resolve().parent
    with open(here / "fixtures" / "mingimage-310p.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _split_devices(raw: object) -> list[str]:
    assert isinstance(raw, str) and raw, f"devices must be a csv string, got {raw!r}"
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


class TestPlacementSurface:
    def test_stage0_tp3_devices_consistent(self) -> None:
        stages = _load_recipe_fixture()["stages"]
        stage0 = stages[0]
        assert stage0["tensor_parallel_size"] == 3
        devs = _split_devices(stage0["devices"])
        assert devs == ["0", "1", "3"]
        assert len(devs) == stage0["tensor_parallel_size"]

    def test_stage1_tp1_single_device(self) -> None:
        stages = _load_recipe_fixture()["stages"]
        stage1 = stages[1]
        assert stage1["tensor_parallel_size"] == 1
        devs = _split_devices(stage1["devices"])
        assert devs == ["2"]
        assert len(devs) == stage1["tensor_parallel_size"]

    def test_stages_disjoint_devices(self) -> None:
        stages = _load_recipe_fixture()["stages"]
        dev_sets = [set(_split_devices(s["devices"])) for s in stages]
        for i, j in ((0, 1),):
            assert dev_sets[i].isdisjoint(dev_sets[j])

    def test_fp16_invariant_and_probe_surface(self) -> None:
        doc = _load_recipe_fixture()
        assert doc["dtype"] == "float16"
        stage0 = doc["stages"][0]
        assert stage0["default_sampling_params"] == {
            "max_tokens": 1,
            "temperature": 0,
            "seed": 42,
        }
        assert stage0["enforce_eager"] is True
