# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch the ABot-World causal DMD pipeline for the 310P NPU path.

Same porting pattern as ``sdxl.py``/``z_image.py`` (COMPARISON-POLICY.md §5
registry-style dispatch — platform.py and the dispatch in ``_310p.patch`` are
untouched; only the registry line + this module are added by this branch):

* force the 310P runtime dtype (FP16) before the pipeline is constructed;
* pin the UMT5-XXL text encoder (preprocessing-only, once per session) on CPU
  in FP32 — the single-chip memory invariant: 10.54GB DiT + 2.82GB Wan VAE +
  ~3.0GB full-Wan decoder persistent FP32 feature cache already occupy
  16.4GB of the 21.08GiB chip, so the 11.36GB encoder cannot co-reside;
  CPU FP32 follows the 310P SoC CPU compute discipline (ENV-PRECHECK P1-11:
  CPU FP16 GEMM pathology ~80x slower); prompt embeds are moved back to the
  NPU in the runtime FP16 dtype;
* route the pipeline's custom ``encode_prompt`` (which unlike Z-Image does
  not use the base ``_encode_prompt``) through the encoder-side device, and
  set ``_encoder_device`` so the shared HF-encoder offload hook
  (``_310p.patch._offload_hf_text_encoder``) skips the no-op ``_encode_prompt``
  wrap instead of raising AttributeError on this pipeline;
* refuse NPU attention-KV quantization (MindIE-only path);
* register the 310P-specialized pipeline under the same ``model_arch``
  (``ABotWorldCausalPipeline``) via ``register_diffusion_model``; the
  built-in pre/post process funcs are kept (registered with None).

Weight note: the official repo ships the 5.27B DiT in BF16
(10,540,744,232 B) and the UMT5-XXL encoder in BF16
(11,361,920,418 B) — the 310P FP16-only hardware runs the FP16 conversion
at load (construction-time default-dtype pin, same as Z-Image FP32-on-disk
casting).
"""

from __future__ import annotations

import logging
import os

import torch

from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import (
    ABotWorldCausalPipeline,
)
from vllm_omni.platforms.npu._310p.patch.common import maybe_install_phase_tracer

logger = logging.getLogger(__name__)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_ENCODER_ON_NPU_ENV = "VLLM_OMNI_310P_ABOT_ENCODER_ON_NPU"
_ABOT_ARCH = "ABotWorldCausalPipeline"
_PATCHED = False

__all__ = [
    "_ABotWorldCausal310P",
    "apply_diffusion_310p_patches",
]


def _encoder_on_npu() -> bool:
    """Opt out of CPU encoder pinning for A/B experiments (memory risk)."""
    return os.environ.get(_ENCODER_ON_NPU_ENV, "0") == "1"


class _ABotWorldCausal310P(ABotWorldCausalPipeline):
    """ABot-World causal DMD pipeline specialized for the 310P NPU path.

    Component placement on 310P:

    * ``text_encoder`` (UMT5-XXL encoder, 24 layers, ~5.68B, BF16 11.36GB on
      disk): preprocessing-only, run once per session (a single encode per
      rollout).  Kept on CPU in FP32 (ENV-PRECHECK P1-11); prompt embeds are
      moved back to the NPU in the runtime FP16 dtype.
      ``VLLM_OMNI_310P_ABOT_ENCODER_ON_NPU=1`` keeps it on-device for
      experiments (single-chip OOM risk — it cannot co-reside with the DiT).
    * ``transformer`` (ABotWorldCausalTransformer3DModel, 5.27B causal DMD
      DiT, paged KV) + ``vae`` (Wan2.2 VAE / TAEW2.2 tiny decoder): NPU,
      FP16 (forced by the runtime patch); the AR paged self-attention goes
      through the 310P math decomposition (``patch/ar_paged.py``) — the
      fused ``npu_fusion_attention`` kernel family is dead on-card
      (research/012 §F.sdpa).
    * DMD 4-step schedule (timesteps 1000/750/500/250) is computed on CPU in
      float64 (``_build_shifted_flow_schedule``) — no NPU RealDiv surface.
    """

    def __init__(self, *, od_config, prefix: str = "") -> None:
        # 310P hardware constraint (same as Z-Image): aclnn has no BF16
        # support for elementwise initializers; pin the FP16 runtime dtype and
        # the CPU default device around construction (modules are .to() their
        # final devices afterwards, so construction on CPU is exact).
        _prev_dtype = torch.get_default_dtype()
        _had_default_device = hasattr(torch.device, "get_default")
        _prev_device = torch.device.get_default() if _had_default_device else None
        try:
            torch.set_default_dtype(_RUNTIME_DTYPE)
            if _had_default_device:
                torch.device.set_default(_CPU_DEVICE)
            super().__init__(od_config=od_config, prefix=prefix)
        finally:
            torch.set_default_dtype(_prev_dtype)
            if _had_default_device and _prev_device is not None:
                torch.device.set_default(_prev_device)
        if _encoder_on_npu():
            self._encoder_device = self.device
        else:
            self._encoder_device = _CPU_DEVICE
            self.text_encoder.to(device=_CPU_DEVICE, dtype=torch.float32)
            logger.info(
                "[310P] ABot-World text encoder pinned to CPU (FP32); embeds moved to %s in %s.",
                self.device,
                _RUNTIME_DTYPE,
            )
        maybe_install_phase_tracer(self, self.device)

    def encode_prompt(self, prompt: str, *, max_sequence_length: int, dtype: torch.dtype) -> torch.Tensor:
        """Base implementation with encoder-side device routing (CPU by default
        on 310P); embeds return on the execution device in the runtime dtype.

        The stock implementation moves ``input_ids``/``attention_mask`` onto
        ``self.device`` and runs the encoder there; with the encoder pinned to
        CPU in FP32 (P1-11) the ids stay on the encoder device and only the
        extracted embeds cross the PCIe bridge in FP16.
        """
        text_inputs = self.tokenizer(
            [" ".join(prompt.strip().split())],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoder_device = self._encoder_device
        input_ids = text_inputs.input_ids.to(encoder_device)
        attention_mask = text_inputs.attention_mask.to(encoder_device)
        embeds = self.text_encoder(input_ids, attention_mask).last_hidden_state
        embeds = embeds.to(device=self.device, dtype=dtype)
        return embeds * attention_mask.unsqueeze(-1).to(dtype=embeds.dtype)


def apply_diffusion_310p_patches(od_config) -> None:
    """Install the 310P ABot-World runtime on top of ``od_config``.

    Registry entry point (COMPARISON-POLICY.md §5): dispatched from
    ``_310p.patch.apply_diffusion_310p_patches`` via the NPU platform's
    ``init_diffusion_model_runner_runtime`` before the diffusion pipeline is
    constructed, on 310P hosts only.  Safe to call more than once; the
    model-class registration happens once.
    """
    global _PATCHED

    # FP16 is the only supported 310P diffusion runtime dtype: force it
    # before DiffusersPipelineLoader builds modules under set_default_torch_dtype.
    if od_config.dtype is not _RUNTIME_DTYPE:
        logger.info("[310P] Forcing ABot-World diffusion runtime dtype %s -> %s.", od_config.dtype, _RUNTIME_DTYPE)
        od_config.dtype = _RUNTIME_DTYPE

    # The AR-Diffusion streaming decoder owns its temporal feature cache and
    # does not honor od_config tiling (documented upstream); set the flags
    # anyway for any tiling-aware decode path and for parity with the suite.
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
    # same model_arch.  pre/post process funcs are the built-in ones
    # (get_abot_world_pre_process_func / get_abot_world_post_process_func in
    # vllm_omni.diffusion.models.abot_world) and are kept by passing None.
    register_diffusion_model(
        _ABOT_ARCH,
        module_name=__name__,
        class_name="_ABotWorldCausal310P",
    )
    _PATCHED = True
    logger.info("[310P] Registered ABot-World 310P pipeline override for %s.", _ABOT_ARCH)
