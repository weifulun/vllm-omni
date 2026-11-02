# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch the SDXL diffusion pipeline for the 310P NPU path.

The 310P line has no MindIE-SD operators: diffusion runs plain ``torch_npu``
in FP16 and fully eager (NPU exposes no inductor).  This module follows the
same porting pattern as ``qwen3_tts.py``:

* force the 310P runtime dtype (FP16) on the diffusion config before the
  pipeline is constructed, so every component (UNet / encoders / VAE) is
  built and loaded in FP16 by ``DiffusersPipelineLoader``;
* pin preprocessing-only modules (CLIP text encoders) on CPU, mirroring the
  Qwen3-TTS "preprocessing encoder stays on CPU" rule;
* keep the VAE decode peak bounded via forced tiling/slicing;
* refuse NPU attention-KV quantization (MindIE-only path);
* register a 310P-specialized pipeline under the same ``model_arch`` via
  ``register_diffusion_model`` (the platform-optimised override path).

The attention backend itself is NOT hard-pinned here: with MindIE-SD absent
the platform already defaults to ``TORCH_SDPA`` on NPU.  ``FLASH_ATTN``
(native ``npu_fusion_attention``) stays an explicit experiment via
``diffusion_attention_config``; a 310P-native backend (``_npu_flash_attention``
+ FRACTAL_NZ, TTS pattern) is the fallback if neither passes on-device.
"""

from __future__ import annotations

import logging
import os

import torch

from vllm_omni.diffusion.models.sdxl.pipeline_sdxl import (
    StableDiffusionXLPipeline,
    get_sdxl_image_post_process_func,
)

logger = logging.getLogger(__name__)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_ENCODER_ON_NPU_ENV = "VLLM_OMNI_310P_SDXL_ENCODER_ON_NPU"
_SDXL_ARCH = "StableDiffusionXLPipeline"
_PATCHED = False

__all__ = [
    "_StableDiffusionXL310P",
    "apply_diffusion_310p_patches",
    "get_sdxl_image_post_process_func",
]


def _encoder_on_npu() -> bool:
    """Opt out of CPU encoder pinning for A/B experiments."""
    return os.environ.get(_ENCODER_ON_NPU_ENV, "0") == "1"


class _StableDiffusionXL310P(StableDiffusionXLPipeline):
    """SDXL pipeline specialized for the 310P NPU path.

    Component placement on 310P:

    * ``text_encoder`` / ``text_encoder_2`` (CLIP-L + OpenCLIP-ViT-bigG):
      preprocessing-only, run once per prompt.  Kept on CPU in FP32 (the
      310P CLIP op surface is unverified and the 2.2 GB ViT-bigG is pure
      resident cost on NPU); prompt embeds are moved back to the NPU in
      the runtime FP16 dtype.  ``VLLM_OMNI_310P_SDXL_ENCODER_ON_NPU=1``
      keeps them on-device for experiments.
    * UNet + VAE + latents: NPU, FP16 (forced by the runtime patch).
    * Scheduler: FP32, unchanged (device handling already in the base).
    """

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._encoder_device = self.device
        if not _encoder_on_npu():
            # Preprocessing-only modules: keep them off the 310P NPU.
            self._encoder_device = _CPU_DEVICE
            self.text_encoder.to(device=_CPU_DEVICE, dtype=torch.float32)
            self.text_encoder_2.to(device=_CPU_DEVICE, dtype=torch.float32)
            logger.info(
                "[310P] SDXL text encoders pinned to CPU (FP32); embeds moved to %s in %s.",
                self.device,
                _RUNTIME_DTYPE,
            )

    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        tokenizer,
        text_encoder,
        num_images_per_prompt: int = 1,
        return_pooled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Base implementation with encoder-side device routing.

        Tokens go to the encoder's device (CPU by default on 310P); the
        extracted embeds come back to the pipeline device in the runtime
        dtype, matching the Qwen3-TTS CPU-encoder pattern.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        encoder_device = getattr(self, "_encoder_device", self.device)
        outputs = text_encoder(text_input_ids.to(encoder_device), output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-2].to(dtype=self.od_config.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(len(prompt) * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = None
        if return_pooled:
            pooled_prompt_embeds = outputs[0].to(dtype=self.od_config.dtype, device=self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompt) * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds


def apply_diffusion_310p_patches(od_config) -> None:
    """Install the 310P SDXL runtime on top of ``od_config``.

    Registry entry point (COMPARISON-POLICY.md §5): dispatched from
    ``_310p.patch.apply_diffusion_310p_patches`` via the NPU platform's
    ``init_diffusion_model_runner_runtime`` before the diffusion pipeline is
    constructed, on 310P hosts only.  Safe to call more than once; the
    model-class registration happens once.
    """
    global _PATCHED

    # FP16 is the only supported 310P diffusion runtime dtype: force it before
    # DiffusersPipelineLoader builds modules under set_default_torch_dtype.
    if od_config.dtype is not _RUNTIME_DTYPE:
        logger.info("[310P] Forcing SDXL diffusion runtime dtype %s -> %s.", od_config.dtype, _RUNTIME_DTYPE)
        od_config.dtype = _RUNTIME_DTYPE

    # VAE decode at 1024x1024 is the memory peak; keep it tiled on 310P.
    od_config.vae_use_tiling = True
    od_config.vae_use_slicing = True

    # NPU attention KV quantization is a MindIE-only path; refuse it loudly
    # instead of letting the FLASH_ATTN quant branch fail deep in the kernel.
    kv_cache_dtype = getattr(od_config, "diffusion_kv_cache_dtype", None)
    if kv_cache_dtype not in (None, "auto", "float"):
        logger.warning(
            "[310P] diffusion_kv_cache_dtype=%r is not supported on 310P (MindIE-only); neutralizing to 'auto'.",
            kv_cache_dtype,
        )
        od_config.diffusion_kv_cache_dtype = "auto"

    if _PATCHED:
        return

    from vllm_omni.diffusion.registry import register_diffusion_model

    # Override the built-in pipeline with the 310P-specialized class under the
    # same model_arch; the post-process func must be resolvable from this
    # module because _DIFFUSION_MODELS now points here.
    register_diffusion_model(
        _SDXL_ARCH,
        module_name=__name__,
        class_name="_StableDiffusionXL310P",
        post_process_func_name="get_sdxl_image_post_process_func",
    )
    _PATCHED = True
    logger.info("[310P] Registered SDXL 310P pipeline override for %s.", _SDXL_ARCH)


# Backwards-compatible alias; the registry dispatch (and PR-facing callers)
# use the canonical apply_diffusion_310p_patches name.
apply_sdxl_310p_patches = apply_diffusion_310p_patches
