# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch the SANA-WM diffusion pipeline for the 310P NPU path.

SANA-WM Stage-1 is a camera-controllable first-frame image-to-video world
model: Stage-1 DiT (Gated-DeltaNet hybrid, ~2.05B params) + LTX-2 video VAE
(32x spatial / 8x temporal, forced tiling) + Gemma-2-2b text encoder.  On 310P
there is no MindIE-SD operator surface and no BF16: diffusion runs plain
``torch_npu`` in FP16 and fully eager.  This module follows the SDXL 310P
patch pattern (COMPARISON-POLICY.md §5 registry branch):

* force the 310P runtime dtype (FP16) on the diffusion config before
  ``DiffusersPipelineLoader`` builds the modules, so DiT / VAE / encoders are
  built and loaded in FP16;
* pin the Gemma-2-2b text encoder on CPU (preprocessing-only, runs once per
  prompt branch; the 310P CLIP/VLM op surface is unverified and ~5.2 GB of
  resident FP16 weights are pure budget cost on the 21589 MB/chip budget);
  prompt embeds move back to the NPU in the runtime FP16 dtype;
* keep the forced VAE tiling/slicing on (the pipeline re-asserts tiling in
  ``_ensure_vae``; the config flags here are belt-and-braces, mirroring SDXL);
* refuse NPU attention-KV quantization (MindIE-only path);
* register a 310P-specialized pipeline under the same ``model_arch``.

The attention backend is NOT hard-pinned here: with MindIE-SD absent the NPU
platform already defaults to ``TORCH_SDPA``, and the 310P-wide explicit math
SDPA decomposition (``patch/attention.py``) is applied by the registry
dispatcher before this module loads — both the vllm ``Attention`` softmax
blocks and any direct ``F.scaled_dot_product_attention`` call (diffusers VAE)
route through it.

Placement (mount-recipe.md §4, 21589 MB/chip measured): Stage-1 DiT FP16
~4.1 GB + VAE FP16 ~1.2 GB ≈ 5.3 GB resident on one chip; Gemma-2-2b (~5.2 GB
FP16) stays in host RAM.  **TP1 single chip** — no INT8 fallback and no
multi-chip placement required (research/011 parameter verification).

Warmup note: the pipeline sets ``dummy_run_num_frames = 0``, so the generic
512x512 startup warmup (the research/012 dynamic-TBE failure class) never
runs for SANA-WM; the first real request pre-compiles kernels.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import (
    SanaWmPipeline,
    get_sana_wm_pre_process_func,
)

logger = logging.getLogger(__name__)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_ENCODER_ON_NPU_ENV = "VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU"
_SANA_WM_ARCH = "SanaWmPipeline"
_PATCHED = False

__all__ = [
    "_SanaWm310P",
    "apply_diffusion_310p_patches",
    "get_sana_wm_pre_process_func",
]


def _encoder_on_npu() -> bool:
    """Opt out of CPU encoder pinning for A/B experiments."""
    return os.environ.get(_ENCODER_ON_NPU_ENV, "0") == "1"


class _SanaWm310P(SanaWmPipeline):
    """SANA-WM pipeline specialized for the 310P NPU path.

    Component placement on 310P:

    * ``text_encoder`` (Gemma-2-2b): preprocessing-only, once per prompt
      branch.  Kept on CPU in FP16 (the Qwen3-VL fp16-CPU-forward precedent
      from the Qwen-Image-2.1 line); prompt embeds are moved back to the NPU
      in the runtime FP16 dtype.  ``VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU=1``
      keeps it on-device for experiments.
    * DiT + VAE + latents: NPU, FP16 (forced by the runtime patch).
    * Scheduler: FP32, unchanged (device handling already in the base).
    """

    def __init__(self, *, od_config: Any = None, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if self.text_encoder is not None and not _encoder_on_npu():
            # Preprocessing-only module: keep it off the 310P NPU.
            self.text_encoder.to(device=_CPU_DEVICE, dtype=_RUNTIME_DTYPE)
            logger.info(
                "[310P] SANA-WM Gemma text encoder pinned to CPU (%s); embeds moved to %s in %s.",
                _CPU_DEVICE,
                self.device,
                _RUNTIME_DTYPE,
            )

    def _encoder_device(self) -> torch.device:
        return self.device if _encoder_on_npu() else _CPU_DEVICE

    def _ensure_stage1_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        # 310P: never re-place the encoder onto the NPU.
        super()._ensure_stage1_text_encoder(device=self._encoder_device(), dtype=dtype)

    def _native_prompt_embeds(
        self,
        prompt: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # MIRROR of SanaWmPipeline._native_prompt_embeds (base) with one 310P
        # change: the token tensors route to the encoder's pinned device
        # (CPU) instead of the pipeline device; embeds still return on
        # ``device`` in ``dtype``.  Keep in sync if the base body changes.
        prompt_text = self._stage1_prompt_text(prompt)
        self._ensure_stage1_text_encoder(device=device, dtype=dtype)
        tokenizer = self.tokenizer
        if tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Sana-WM Stage-1 text encoder did not initialize.")
        chi_prompt = "\n".join(part for part in self.sana_wm_config.chi_prompt if part)
        model_max_length = int(self.sana_wm_config.model_max_length)
        max_length_all = model_max_length
        if chi_prompt:
            max_length_all = len(tokenizer.encode(chi_prompt)) + model_max_length - 2
        encoded = tokenizer(
            [prompt_text],
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self._encoder_device())
        hidden_states = self.text_encoder(**encoded).last_hidden_state
        attention_mask = encoded.attention_mask
        if chi_prompt:
            select_index = [0] + list(range(-model_max_length + 1, 0))
            hidden_states = hidden_states[:, select_index]
            attention_mask = attention_mask[:, select_index]
        if hidden_states.shape[-1] != 2304:
            raise ValueError(
                "Sana-WM Stage-1 Gemma hidden size mismatch: expected "
                f"2304, got {hidden_states.shape[-1]}."
            )
        prompt_attention_mask = attention_mask.to(device=device, dtype=torch.float32)
        if bool(prompt_attention_mask.all()):
            prompt_attention_mask = None
        self._last_prompt_attention_mask = prompt_attention_mask
        return hidden_states.to(device=device, dtype=dtype)


def apply_diffusion_310p_patches(od_config) -> None:
    """Apply the SANA-WM 310P diffusion runtime patches (idempotent)."""
    global _PATCHED

    # The registry dispatcher already ran the shared 310P attention patch and
    # the optional warmup skip before this module; nothing to do there.
    if getattr(od_config, "dtype", None) is not _RUNTIME_DTYPE:
        logger.info("[310P] SANA-WM: forcing diffusion runtime dtype to %s.", _RUNTIME_DTYPE)
        od_config.dtype = _RUNTIME_DTYPE

    # VAE tiling is re-asserted per-request by the pipeline itself; the config
    # flags are belt-and-braces (registry.py copies them onto model.vae).
    od_config.vae_use_tiling = True
    od_config.vae_use_slicing = True

    if getattr(od_config, "diffusion_kv_cache_dtype", None) not in (None, "auto"):
        logger.warning(
            "[310P] SANA-WM: ignoring diffusion_kv_cache_dtype=%r (MindIE-only path).",
            od_config.diffusion_kv_cache_dtype,
        )
        od_config.diffusion_kv_cache_dtype = "auto"

    if _PATCHED:
        return

    from vllm_omni.diffusion.registry import register_diffusion_model

    # Override the built-in pipeline with the 310P-specialized class under the
    # same model_arch.  pre_process_func_name stays untouched (None keeps the
    # existing entry): get_sana_wm_pre_process_func is the canonical entry.
    register_diffusion_model(
        _SANA_WM_ARCH,
        module_name=__name__,
        class_name="_SanaWm310P",
    )
    _PATCHED = True
    logger.info("[310P] Registered SANA-WM 310P pipeline override for %s.", _SANA_WM_ARCH)


# Backwards-compatible alias; the registry dispatch (and PR-facing callers)
# use the canonical apply_diffusion_310p_patches name.
apply_sana_wm_310p_patches = apply_diffusion_310p_patches
