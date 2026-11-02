# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch the Z-Image-Turbo diffusion pipeline for the 310P NPU path.

Same porting pattern as ``sdxl.py`` (COMPARISON-POLICY.md §5 registry-style
dispatch — platform.py and the dispatch in ``_310p.patch`` are untouched):

* force the 310P runtime dtype (FP16) before the pipeline is constructed;
* pin the Qwen3-based text encoder (preprocessing-only, once per prompt) on
  CPU in FP32; the extracted embeds are moved back to the NPU in the runtime
  FP16 dtype;
* keep the VAE decode peak bounded via forced tiling/slicing;
* refuse NPU attention-KV quantization (MindIE-only path);
* register the 310P-specialized pipeline under the same ``model_arch``
  (``ZImagePipeline``) via ``register_diffusion_model``.

Weight note: the official repo ships the 6.15B transformer in FP32
(24.6GB) — the 310P single-chip budget (21.08GiB) requires the FP16
conversion staged at /data/310p-weights/zimage-turbo (see
310p-project/weights-manifest.md "Z-Image-Turbo" section).
"""

from __future__ import annotations

import logging
import os

import torch

from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    ZImagePipeline,
    get_post_process_func,
)
from vllm_omni.platforms.npu._310p.patch.common import maybe_install_phase_tracer

logger = logging.getLogger(__name__)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_ENCODER_ON_NPU_ENV = "VLLM_OMNI_310P_ZIMAGE_ENCODER_ON_NPU"
_ZIMAGE_ARCH = "ZImagePipeline"
_PATCHED = False

__all__ = [
    "_ZImage310P",
    "apply_diffusion_310p_patches",
    "get_post_process_func",
]


def _encoder_on_npu() -> bool:
    """Opt out of CPU encoder pinning for A/B experiments."""
    return os.environ.get(_ENCODER_ON_NPU_ENV, "0") == "1"


class _ZImage310P(ZImagePipeline):
    """Z-Image pipeline specialized for the 310P NPU path.

    Component placement on 310P:

    * ``text_encoder`` (Qwen3-based CausalLM, ~4B, FP32 ≈ 16GB CPU RAM):
      preprocessing-only, run once per prompt (no CFG on Turbo → a single
      encode per request).  Kept on CPU in FP32 (the 310P op surface for a
      30-layer CausalLM is unverified and it is pure resident cost on NPU);
      prompt embeds are moved back to the NPU in the runtime FP16 dtype.
      ``VLLM_OMNI_310P_ZIMAGE_ENCODER_ON_NPU=1`` keeps it on-device for
      experiments.
    * ``transformer`` (ZImageTransformer2DModel, 30-layer single-stream DiT,
      6.15B) + ``vae`` + latents: NPU, FP16 (forced by the runtime patch);
      attention goes through the unified sdpa backend, whose 310P MATH
      decomposition is the hardware-level shim (decoupled from this patch
      suite by design, commit 939fc24).
    * ``scheduler`` (FlowMatchEulerDiscrete): FP32, unchanged.
    """

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._encoder_device = self.device
        if not _encoder_on_npu():
            self._encoder_device = _CPU_DEVICE
            self.text_encoder.to(device=_CPU_DEVICE, dtype=torch.float32)
            logger.info(
                "[310P] Z-Image text encoder pinned to CPU (FP32); embeds moved to %s in %s.",
                self.device,
                _RUNTIME_DTYPE,
            )
        maybe_install_phase_tracer(self, self.device)

    def _encode_prompt(
        self,
        prompt,
        device=None,
        prompt_embeds=None,
        max_sequence_length: int = 512,
    ):
        """Base implementation with encoder-side device routing (CPU by
        default on 310P); embeds return on ``device`` in the runtime dtype."""
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        encoder_device = getattr(self, "_encoder_device", device)
        text_input_ids = text_inputs.input_ids.to(encoder_device)
        prompt_masks = text_inputs.attention_mask.to(encoder_device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(
                prompt_embeds[i][prompt_masks[i]].to(dtype=self.od_config.dtype, device=device)
            )

        return embeddings_list


def apply_diffusion_310p_patches(od_config) -> None:
    """Install the 310P Z-Image runtime on top of ``od_config``.

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
        logger.info("[310P] Forcing Z-Image diffusion runtime dtype %s -> %s.", od_config.dtype, _RUNTIME_DTYPE)
        od_config.dtype = _RUNTIME_DTYPE

    # VAE decode at 1024x1024 is the memory peak; keep it tiled on 310P.
    od_config.vae_use_tiling = True
    od_config.vae_use_slicing = True

    # NPU attention KV quantization is a MindIE-only path; refuse it loudly.
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
        _ZIMAGE_ARCH,
        module_name=__name__,
        class_name="_ZImage310P",
        post_process_func_name="get_post_process_func",
    )
    _PATCHED = True
    logger.info("[310P] Registered Z-Image 310P pipeline override for %s.", _ZIMAGE_ARCH)


# Backwards-compatible alias; the registry dispatch (and PR-facing callers)
# use the canonical apply_diffusion_310p_patches name.
apply_zimage_310p_patches = apply_diffusion_310p_patches
