# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch the AuK diffusion pipeline for the 310P NPU path.

The 310P line has no MindIE-SD operators and none of the CUDA-only
accelerations the AuK stage ships with (torch.compile of the DiT blocks,
per-shape CUDA graphs for one denoise step, bucketed CUDA graphs for the
codec decode).  On 310P the whole stage runs plain ``torch_npu`` in FP16,
fully eager — the same porting pattern as ``sdxl.py`` / ``qwen3_tts.py``:

* force the 310P runtime dtype (FP16) on the diffusion config before the
  pipeline is constructed, so the DiT (released fp32 in
  ``auk.safetensors``) is cast to FP16 at load and runs under FP16 autocast;
* the BigVGAN-flow VAE stays FP32 on the inference device — the pipeline
  already pins it there (numerically sensitive codec; aclnn supports
  DT_FLOAT on 310P);
* both CUDA-graph wrappers are forced off so a stray non-eager deploy cannot
  attempt CUDA capture on NPU.  The wrappers already fall back to eager on
  non-CUDA tensors; this makes the state explicit and logged;
* attention goes through the global explicit math SDPA decomposition
  (``attention.py``): AuK's single attention site calls
  ``F.scaled_dot_product_attention`` with an additive key-padding bias
  (``[B, 1, 1, K]``, -inf for padded keys), which the math path handles;
* the BigVGAN-flow VAE's Kaiser-sinc filter taps are pinned to CPU: the
  diffusion loader constructs the pipeline inside a ``torch.device`` context
  (``diffusers_loader._init_from_load_format``), so device-less tensor
  creation in ``_kaiser_sinc_filter`` would default to the NPU, and
  ``aten::kaiser_window`` has no NPU kernel on the CANN 9.1.0 / torch_npu
  build (observed on 310P: DispatchStub ``missing kernel for npu`` abort
  during AuKVAE construction).  The taps are a constant buffer the pipeline
  moves to the device anyway (``AuKVAE.from_config(...).to(device=...)``),
  so CPU construction is numerically identical.

No stage-0 (frozen Qwen2.5-Omni thinker encoder) patch is needed: it runs the
standard vLLM AR worker path with the generic 310P worker patch,
``enforce_eager: true`` and the FP16 deploy dtype.
"""

from __future__ import annotations

import logging

import torch

from vllm_omni.diffusion.models.auk.pipeline_auk import (
    AuKPipeline,
    get_auk_post_process_func,
)

logger = logging.getLogger(__name__)

_RUNTIME_DTYPE = torch.float16
_AUK_ARCH = "AuKPipeline"
_PATCHED = False

__all__ = [
    "_AuK310P",
    "apply_diffusion_310p_patches",
    "get_auk_post_process_func",
]


def _patch_auk_vae_kaiser_window() -> None:
    """Pin the VAE's Kaiser-sinc filter taps to CPU construction.

    Called before ``AuKPipeline.__init__`` (the VAE is built inside it).
    Idempotent; safe on non-310P hosts too (CPU taps are identical).
    """
    from vllm_omni.diffusion.models.auk import auk_vae

    if getattr(auk_vae, "_auk_310p_kaiser_patch", False):
        return
    original = auk_vae._kaiser_sinc_filter

    def _cpu_kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
        with torch.device("cpu"):
            return original(cutoff, half_width, kernel_size)

    auk_vae._kaiser_sinc_filter = _cpu_kaiser_sinc_filter
    auk_vae._auk_310p_kaiser_patch = True
    logger.info(
        "[310P] AuKVAE Kaiser-sinc filter taps pinned to CPU "
        "(no NPU kernel for aten::kaiser_window on this torch_npu build)."
    )


class _AuK310P(AuKPipeline):
    """AuK pipeline specialized for the 310P NPU path.

    Component placement on 310P:

    * DiT: NPU, FP16 (forced by the runtime patch before construction).
    * BigVGAN-flow VAE: NPU, FP32 (unchanged from the base pipeline; the
      codec is numerically sensitive and aclnn supports DT_FLOAT).
    * DiT step graph + VAE decode bucket graph: forced off (CUDA-only
      capture primitives); eager execution is the 310P norm.
    """

    def __init__(self, *, od_config, prefix: str = "") -> None:
        _patch_auk_vae_kaiser_window()
        super().__init__(od_config=od_config, prefix=prefix)
        if self.cudagraph_wrapper.enabled:
            self.cudagraph_wrapper.enabled = False
            logger.info("[310P] AuK DiT single-step CUDA graph wrapper forced off (CUDA-only capture).")
        if self.vae_decode.enabled:
            self.vae_decode.enabled = False
            logger.info("[310P] AuK VAE decode bucket CUDA graph forced off (CUDA-only capture).")
        if self.dtype is not _RUNTIME_DTYPE:
            logger.warning(
                "[310P] AuK DiT dtype is %s but the 310P runtime expects %s; check the deploy config.",
                self.dtype,
                _RUNTIME_DTYPE,
            )
        else:
            logger.info(
                "[310P] AuK 310P pipeline ready: DiT=%s VAE=fp32, eager (no CUDA graphs), math-SDPA attention.",
                self.dtype,
            )


def apply_diffusion_310p_patches(od_config) -> None:
    """Install the 310P AuK runtime on top of ``od_config``.

    Registry entry point (COMPARISON-POLICY.md §5): dispatched from
    ``_310p.patch.apply_diffusion_310p_patches`` via the NPU platform's
    ``init_diffusion_model_runner_runtime`` before the diffusion pipeline is
    constructed, on 310P hosts only.  Safe to call more than once; the
    model-class registration happens once.
    """
    global _PATCHED

    # FP16 is the only supported 310P diffusion runtime dtype: force it before
    # the pipeline casts the fp32 DiT checkpoint at load time.
    if od_config.dtype is not _RUNTIME_DTYPE:
        logger.info("[310P] Forcing AuK diffusion runtime dtype %s -> %s.", od_config.dtype, _RUNTIME_DTYPE)
        od_config.dtype = _RUNTIME_DTYPE

    # NPU attention KV quantization is a MindIE-only path; refuse it loudly
    # instead of letting the quant branch fail deep in the kernel.
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
        _AUK_ARCH,
        module_name=__name__,
        class_name="_AuK310P",
        post_process_func_name="get_auk_post_process_func",
    )
    _PATCHED = True
    logger.info("[310P] Registered AuK 310P pipeline override for %s.", _AUK_ARCH)
