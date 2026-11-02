# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-device 310P wiring tests for the SDXL diffusion patch.

``test_310p_sdxl.py`` covers the same contract on CPU with fakes; these tests
run on a real 310P card and assert the end-to-end chain: the runtime reports
310P, the platform runtime hook applies the SDXL 310P overrides (FP16 force,
VAE tiling, registry swap) and the swapped pipeline class is what the
diffusion registry resolves.

The full t2i smoke (seeded 1024x1024, per-chip memory peak, per-step timing)
lives in the model-sdxl run script, not here, so pytest stays lightweight.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_ascend")

import torch  # noqa: E402
import torch_npu  # noqa: E402
from vllm_ascend.device.device_config import get_ascend_device_type  # noqa: E402
from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc  # noqa: E402

from tests.helpers.mark import hardware_marks  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "310P"}, num_cards=1)]


@pytest.fixture(scope="module", autouse=True)
def require_real_310p() -> None:
    runtime = device_type_from_runtime_soc(torch_npu.npu.get_soc_version())
    if runtime is not AscendDeviceType._310P:
        pytest.skip(f"not a 310P host (runtime device type: {runtime})")


def test_runtime_and_omni_agree_on_310p() -> None:
    from vllm_omni.platforms.npu import is_a5
    from vllm_omni.platforms.npu._310p import is_310p

    assert is_310p() is True
    assert is_a5() is False
    assert get_ascend_device_type() is AscendDeviceType._310P


def test_apply_diffusion_310p_patches_swaps_registry_entry() -> None:
    from vllm_omni.diffusion.registry import DiffusionModelRegistry
    from vllm_omni.platforms.npu._310p.patch.sdxl import (
        _StableDiffusionXL310P,
        apply_diffusion_310p_patches,
    )

    od_config = SimpleNamespace(
        dtype=torch.bfloat16,
        vae_use_tiling=False,
        vae_use_slicing=False,
        diffusion_kv_cache_dtype=None,
        model_class_name="StableDiffusionXLPipeline",
    )

    apply_diffusion_310p_patches(od_config)

    assert od_config.dtype is torch.float16
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    resolved = DiffusionModelRegistry._try_load_model_cls("StableDiffusionXLPipeline")
    assert resolved is _StableDiffusionXL310P


def test_310p_pipeline_class_derives_from_real_pipeline() -> None:
    """The 310P override must subclass the real (checkpoint-backed) pipeline so
    loading/forward are inherited; only encoder placement and embed routing
    change.  Placement semantics are pinned on CPU in test_310p_sdxl.py."""
    from vllm_omni.diffusion.models.sdxl.pipeline_sdxl import StableDiffusionXLPipeline
    from vllm_omni.platforms.npu._310p.patch.sdxl import (
        _encoder_on_npu,
        _StableDiffusionXL310P,
        get_sdxl_image_post_process_func,
    )

    assert issubclass(_StableDiffusionXL310P, StableDiffusionXLPipeline)
    assert _StableDiffusionXL310P._get_clip_prompt_embeds is not StableDiffusionXLPipeline._get_clip_prompt_embeds
    assert not _encoder_on_npu()  # default: CLIP encoders pinned to CPU
    assert get_sdxl_image_post_process_func is not None
