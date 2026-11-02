# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU pin for the Boogu-Image Qwen3-VL understanding encoder (310P).

Capacity-feasibility standing (task-33): the ``mllm`` encoder
(``Qwen3VLForConditionalGeneration``, ~17.5 GB) cannot co-reside with the
diffusion transformer on ONE 310P3 chip (21589 MB) in any precision tier -
FP16 alone (17.5 + 20.59 GB) already exceeds the budget, and the INT8 tier
(17.5 + 10.3 GB) still does not.  The encoder therefore runs on CPU; it is
invoked once per request for the instruction embedding, which is then moved
to the execution device by the pipeline itself
(``_get_instruction_feature_embeds`` ends with
``instruction_feats.to(dtype=..., device=...)``).

Standing: same class as the is_310p import fix (container FS patch) - it is
NOT part of the kill-switch-gated per-model patch
(``vllm_omni.platforms.npu._310p.patch.boogu_image``), so the
COMPARISON-POLICY.md section-1 level-2 same-seed reference run stays
comparable (both levels share the CPU encoder).  The container installer
copies this module into the installed ``vllm_omni`` tree and hooks
``BooguImagePipeline._load_mllm`` from the installed boogu_image package
``__init__`` (see deploy/container-mllm-pin.sh); the upstream branch tree is
otherwise untouched.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_INSTALLED = False


class _CpuPinnedMLLM(nn.Module):
    """Wrapper that keeps the wrapped encoder on CPU forever.

    - ``forward``: move tensor inputs to CPU on entry; outputs are returned
      on CPU (the pipeline moves the instruction features to the execution
      device itself).
    - ``_apply``: pinned no-op - any pipeline-wide ``.to(device)`` /
      ``.cuda()`` recursion stops here, so the encoder can never be moved
      off CPU after construction.
    - attribute access (``dtype``, ``config``, ...) delegates to the inner
      module.
    """

    def __init__(self, mllm: nn.Module):
        super().__init__()
        self._mllm = mllm.to("cpu").eval()

    def forward(self, *args, **kwargs):
        args = tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args)
        kwargs = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        return self._mllm(*args, **kwargs)

    def _apply(self, fn, recurse: bool = True):  # noqa: ANN001 - torch signature
        return self  # pinned: never move the encoder off CPU

    def to(self, *args, **kwargs):  # noqa: D102 - torch signature
        return self

    def __getattr__(self, name: str):
        # Standard nn.Module protocol first: registered submodules/parameters/
        # buffers (``_mllm`` is a registered submodule), then delegate to the
        # inner module (``dtype``, ``config``, ...).
        for store in ("_modules", "_parameters", "_buffers"):
            data = self.__dict__.get(store)
            if data is not None and name in data:
                return data[name]
        inner = self.__dict__.get("_modules", {}).get("_mllm")
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(name)


def install_mllm_cpu_pin() -> bool:
    """Wrap ``BooguImagePipeline._load_mllm`` so the encoder stays on CPU.

    Idempotent.  Returns True when the pin was installed (or already was).
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import BooguImagePipeline

    if getattr(BooguImagePipeline._load_mllm, "_310p_cpu_pinned", False):
        _INSTALLED = True
        return True

    original = BooguImagePipeline._load_mllm

    def _load_mllm_pinned(self, model, local_files_only, mllm_quant_config, prefetch_list):
        mllm = original(self, model, local_files_only, mllm_quant_config, prefetch_list)
        logger.warning(
            "[310P][boogu] mllm encoder pinned to CPU (310P capacity feasibility: "
            "17.5 GB encoder + 10.3-20.6 GB transformer cannot co-reside on one "
            "21589 MB chip in any tier; task-33 GO ladder). Container-level "
            "feasibility patch - NOT gated by VLLM_OMNI_310P_DIFFUSION_PATCHES."
        )
        return _CpuPinnedMLLM(mllm)

    _load_mllm_pinned._310p_cpu_pinned = True  # type: ignore[attr-defined]
    BooguImagePipeline._load_mllm = _load_mllm_pinned  # type: ignore[method-assign]
    _INSTALLED = True
    return True
