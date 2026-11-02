# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch CosyVoice3 for the 310P NPU path.

CosyVoice3 is a 2-stage pipeline that resolves **both** stages from the same
model class (``CosyVoice3Model``), dispatched on ``model_stage``:

* stage 0 (``cosyvoice3_talker``, LLM_AR) — vLLM Qwen2 backbone with
  PagedAttention plus talker-specific ``speech_embedding`` / ``llm_decoder``.
  The engine creates every parameter at the deploy dtype, so ``dtype: float16``
  in the 310P deploy overlay covers the whole stage; the patch only swaps the
  per-token random sampler for the NPU Gumbel variant and verifies the stage
  landed in FP16.
* stage 1 (``cosyvoice3_code2wav``, LLM_GENERATION) — a torch DiT flow
  estimator (CFM, 10 Euler steps) + causal HiFT vocoder, loaded directly from
  ``flow.pt`` / ``hift.pt`` inside ``CosyVoice3Code2Wav.load_weights``.  These
  tensors bypass vLLM's weight machinery entirely, so nothing in the engine
  casts or packs them: the patch forces FP16 (310P aclnn rejects DT_BFLOAT16,
  EZ1001; the checkpoints keep their native dtype) and packs the weights for
  the 310P aclnn kernels (Linear -> NZ, group-1 conv1d/convtranspose1d ->
  FRACTAL_Z), mirroring ``voxcpm2.py`` in this directory.

310P-specific decisions (each with its precedent on this node):

* **FP16 forced** — the 310P line is an FP16-only line (EZ1001 on BF16);
  same rule as the Qwen3-TTS / VoxCPM2 310P patches.
* **Random sampler rebind** — the model-owned RAS loop samples per decode
  token via vLLM's native ``random_sample`` (``torch.multinomial``), which on
  NPU forces a CPU-NPU sync (or is unsupported on the 310P aclnn path); the
  vLLM-Ascend Gumbel variant (``exponential_`` + ``argmax``) is the
  established NPU-safe replacement (used by the Qwen3-TTS 310P CodePredictor
  override and by ``AscendSampler310`` itself).
* **S3 speech-tokenizer pinned to CPU** — ``s3tokenizer`` loads its PyTorch
  model onto ``current_omni_platform.get_torch_device()``; on 310P that
  pushes the reference-audio encoder through the 310P dynamic-TBE
  compilation gap family (research 012).  The model is tiny and runs once
  per reference utterance, so it stays eager on CPU — the same
  "preprocessing pinned CPU" rule the VoxCPM2 310P patch applies.
* **HiFT F0 predictor untouched** — upstream already keeps it on CPU in FP32
  for non-CUDA devices (``load_weights`` picks ``f0_device="cpu"`` when
  ``device.type != "cuda"``), and ``inference`` transfers the mel chunk to
  CPU for the F0 step.  The 310P cast/pack pass must skip it.
* **CUDA/TRT paths naturally inert** — the ``COSYVOICE3_FULL_RESPONSE_*`` /
  ``COSYVOICE3_PACKED_STREAMING`` toggles and the TensorRT estimator require
  Hopper (capability 9) + CUDA; none of them engage on NPU, so no
  neutralization is needed (unlike VoxCPM2's LocDiT NPUGraph wrapper, which
  has no CosyVoice3 NPU equivalent in the first place).

Target revision: vllm-omni v0.32.0 (tag 5917730a4b).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_RUNTIME_DTYPE = torch.float16
_ACL_FORMAT_FRACTAL_Z = 4

# POLICY §1 level-2 reference runs ("310P 通用路径"): with the env var set to
# "0" the patcher installs nothing (native multinomial sampler, checkpoint
# dtypes, no packing) while the deploy dtype stays float16 — the generic NPU
# path the 310P patch must not regress against.
_PATCH_ENV = "VLLM_OMNI_310P_COSYVOICE3_PATCH"

_TALKER_PATCHED = False
_CODE2WAV_PATCHED = False


def _patch_enabled() -> bool:
    if os.environ.get(_PATCH_ENV, "1") == "0":
        logger.info("CosyVoice3 310P patch disabled via %s=0 (POLICY §1 level-2 reference run)", _PATCH_ENV)
        return False
    return True


# ===================================================================
#  dtype / packing helpers (shared shape with voxcpm2.py in this dir)
# ===================================================================


def _cast_subtree_to_dtype(modules: list[nn.Module], dtype: torch.dtype) -> int:
    """Cast float parameters/buffers of ``modules`` to ``dtype`` in place.

    Complex buffers (e.g. DiT RoPE ``freqs_cis``) and integer buffers are left
    untouched: ``nn.Module.to(dtype)`` would silently demote complex64 to
    real float and corrupt the cache.
    """
    converted = 0
    with torch.no_grad():
        for child in modules:
            for name, param in list(child._parameters.items()):
                if param is None or not param.is_floating_point():
                    continue
                child._parameters[name] = nn.Parameter(
                    param.data.to(dtype),
                    requires_grad=param.requires_grad,
                )
                converted += 1
            for name, buf in list(child._buffers.items()):
                if buf is None or not buf.is_floating_point() or buf.is_complex():
                    continue
                child._buffers[name] = buf.to(dtype)
                converted += 1
    return converted


def _cast_module_to_dtype(module: nn.Module, dtype: torch.dtype) -> int:
    return _cast_subtree_to_dtype(list(module.modules()), dtype)


def _pack_npu_weights(modules: list[nn.Module]) -> tuple[int, int]:
    """Pack side-path weights for 310P aclnn kernels.

    Mirrors the VoxCPM2 310P helper: Linear weights go through
    ``maybe_trans_nz`` and group-1 conv1d / convtranspose1d weights are cast
    to the FRACTAL_Z layout (Conv kernels on 310P require it; the 910 path's
    A5 exemption does not apply to 310P).  The flow/hift modules live
    outside vLLM's weight-loading machinery, so the engine's
    ``allow_internal_format`` pass never touches them.
    """
    from vllm_ascend.utils import maybe_trans_nz

    import torch_npu

    linear_count = 0
    conv_count = 0
    with torch.no_grad():
        for module_ in modules:
            if isinstance(module_, nn.Linear):
                module_.weight.data = maybe_trans_nz(module_.weight.data)
                linear_count += 1
            elif isinstance(module_, (nn.Conv1d, nn.ConvTranspose1d)) and module_.groups == 1:
                module_.weight.data = torch_npu.npu_format_cast(
                    module_.weight.data.contiguous(),
                    _ACL_FORMAT_FRACTAL_Z,
                )
                conv_count += 1
    return linear_count, conv_count


# ===================================================================
#  Stage 0: talker (LLM_AR)
# ===================================================================


def _verify_talker_fp16(model: nn.Module) -> int:
    """Defensively cast any stray FP32 talker tensors to FP16.

    The engine constructs the stage at the deploy dtype (``float16`` from
    ``deploy/cosyvoice3_310p.yaml``), so this normally converts nothing; it
    catches a misconfigured deployment before the first decode step instead
    of at ACL kernel dispatch.
    """
    stray = 0
    with torch.no_grad():
        for child in model.modules():
            for name, param in list(child._parameters.items()):
                if (
                    param is not None
                    and param.is_floating_point()
                    and param.dtype != _RUNTIME_DTYPE
                ):
                    child._parameters[name] = nn.Parameter(
                        param.data.to(_RUNTIME_DTYPE),
                        requires_grad=param.requires_grad,
                    )
                    stray += 1
    return stray


def _pin_s3_tokenizer_to_cpu() -> None:
    """Keep the S3Tokenizer speech encoder eager on CPU.

    Upstream loads it onto the platform device (NPU inside the stage-0
    worker).  On 310P that routes the reference-audio encoder through the
    dynamic-TBE compilation gap family (research 012: no public precedent,
    no static-pin switch); the encoder is small and runs once per reference
    utterance, so CPU eager is the safe path (VoxCPM2 precedent:
    preprocessing pinned CPU).
    """
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cv3_module

    processor_cls = cv3_module.CosyVoice3MultiModalProcessor
    if getattr(processor_cls, "_s3_cpu_pinned_310p", False):
        return

    original_load = processor_cls._load_s3_model

    def _load_s3_model_cpu(cls):  # noqa: N802 - mirrors classmethod signature
        model, s3, _device = original_load.__func__(cls)
        model = model.to("cpu").eval()
        logger.info("CosyVoice3 310P: S3 speech tokenizer pinned to CPU (eager, FP32)")
        return model, s3, "cpu"

    processor_cls._load_s3_model = classmethod(_load_s3_model_cpu)  # type: ignore[assignment]
    processor_cls._s3_cpu_pinned_310p = True  # type: ignore[attr-defined]


def apply_talker_310p_patches() -> None:
    """Install the 310P CosyVoice3 talker patch (idempotent).

    Called from ``apply_model_patches`` with ``model_arch ==
    "CosyVoice3Model"`` and ``model_stage == "cosyvoice3_talker"`` inside the
    worker's ``load_model`` (after ``set_device``, before registry
    resolution).
    """
    global _TALKER_PATCHED
    if _TALKER_PATCHED:
        return
    if not _patch_enabled():
        return

    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cv3_module
    from vllm_ascend.sample.sampler import random_sample

    # NPU-safe Gumbel sampler: vLLM's native ``random_sample`` is
    # ``torch.multinomial``-based and forces a CPU-NPU sync per draw.  The
    # talker's RAS loop (``_ras_sample_one`` / ``_nucleus_sample_one``) draws
    # once per decode token, and the standard-sampling fallback draws once
    # per request step; both call sites read the module-global name, so a
    # single rebind covers the whole stage.
    cv3_module.random_sample = random_sample  # type: ignore[assignment]

    _pin_s3_tokenizer_to_cpu()

    base_cls = cv3_module.CosyVoice3Model

    class _CosyVoice3Talker310P(base_cls):  # type: ignore[misc,valid-type]
        """CosyVoice3 talker stage specialized for the 310P NPU path (FP16)."""

        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            stray = _verify_talker_fp16(self)
            if stray:
                logger.warning(
                    "CosyVoice3 310P: talker stage had %d non-FP16 tensors; cast to FP16 "
                    "(deployment dtype should be float16, see deploy/cosyvoice3_310p.yaml)",
                    stray,
                )
            else:
                logger.info("CosyVoice3 310P: talker stage verified FP16")

    cv3_module.CosyVoice3Model = _CosyVoice3Talker310P  # type: ignore[assignment]

    _TALKER_PATCHED = True
    logger.info("Applied 310P patch for CosyVoice3 talker (FP16 verified, Gumbel sampler, S3 on CPU)")


# ===================================================================
#  Stage 1: code2wav (LLM_GENERATION)
# ===================================================================


def _wrap_flow_inference_dtype(flow_model: nn.Module) -> None:
    """Cast the FP32 conditioning tensors to the flow runtime dtype.

    ``CausalMaskedDiffWithDiT.inference`` receives the campplus speaker
    embedding (FP32, 192-d) and the reference mel ``prompt_feat`` (FP32) from
    the talker-stage payload.  Everything inside the flow model is FP16 after
    this patch, and the first op on the embedding is the FP16
    ``spk_embed_affine_layer`` — an FP32 input would die with
    "Input type (float) and bias type (c10::Half) should be the same".
    Casting at the inference boundary covers both the streaming
    (``forward_streaming_batch``) and finalize (``forward``) paths because
    they funnel through ``_forward_mel`` -> ``flow_model.inference``.
    """
    original_inference = flow_model.inference

    def _inference_dtype_cast(*args, **kwargs):
        # Positional signature:
        # (token, token_len, prompt_token, prompt_token_len, prompt_feat,
        #  prompt_feat_len, embedding, streaming, finalize, n_timesteps)
        if len(args) >= 7 and args[6].is_floating_point() and args[6].dtype != _RUNTIME_DTYPE:
            args = args[:6] + (args[6].to(_RUNTIME_DTYPE),) + args[7:]
        elif "embedding" in kwargs and kwargs["embedding"].is_floating_point() and kwargs["embedding"].dtype != _RUNTIME_DTYPE:
            kwargs["embedding"] = kwargs["embedding"].to(_RUNTIME_DTYPE)
        if len(args) >= 5 and args[4].is_floating_point() and args[4].dtype != _RUNTIME_DTYPE:
            args = args[:4] + (args[4].to(_RUNTIME_DTYPE),) + args[5:]
        elif "prompt_feat" in kwargs and kwargs["prompt_feat"].is_floating_point() and kwargs["prompt_feat"].dtype != _RUNTIME_DTYPE:
            kwargs["prompt_feat"] = kwargs["prompt_feat"].to(_RUNTIME_DTYPE)
        return original_inference(*args, **kwargs)

    flow_model.inference = _inference_dtype_cast  # type: ignore[method-assign]


def apply_code2wav_310p_patches() -> None:
    """Install the 310P CosyVoice3 code2wav patch (idempotent).

    Same call site contract as ``apply_talker_310p_patches`` with
    ``model_stage == "cosyvoice3_code2wav"``.
    """
    global _CODE2WAV_PATCHED
    if _CODE2WAV_PATCHED:
        return
    if not _patch_enabled():
        return

    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cv3_module

    base_cls = cv3_module.CosyVoice3Model

    class _CosyVoice3Code2Wav310P(base_cls):  # type: ignore[misc,valid-type]
        """CosyVoice3 code2wav stage specialized for the 310P NPU path.

        FP16 forced + 310P weight packing, applied after
        ``super().load_weights`` because the flow/hift checkpoints are loaded
        by ``CosyVoice3Code2Wav.load_weights`` at their native dtype and the
        HiFT weight-norm folding happens there too (packing must run on the
        folded weights).
        """

        def load_weights(self, weights) -> set[str] | None:  # type: ignore[override]
            loaded = super().load_weights(weights)
            if self.model_stage != "cosyvoice3_code2wav":
                return loaded
            code2wav = self.code2wav
            deploy_dtype = getattr(self.vllm_config.model_config, "dtype", None)
            if deploy_dtype is not None and _RUNTIME_DTYPE != deploy_dtype:
                logger.warning(
                    "CosyVoice3 code2wav on 310P runs best with dtype float16 "
                    "(deploy/cosyvoice3_310p.yaml); deployment dtype is %s, the "
                    "flow/hift side is forced to FP16 by this patch.",
                    deploy_dtype,
                )
            flow_modules = list(code2wav.flow_model.modules())
            flow_cast = _cast_subtree_to_dtype(flow_modules, _RUNTIME_DTYPE)

            # The F0 predictor is deliberately excluded from the cast and
            # pack passes: upstream keeps it on CPU in FP32 for non-CUDA
            # devices and transfers the mel chunk to CPU for the F0 step.
            hift = code2wav.hift
            f0_predictor = getattr(hift, "f0_predictor", None)
            hift_modules = list(hift.modules())
            if f0_predictor is not None:
                f0_subtree = {id(m) for m in f0_predictor.modules()}
                hift_modules = [m for m in hift_modules if id(m) not in f0_subtree]
            hift_cast = _cast_subtree_to_dtype(hift_modules, _RUNTIME_DTYPE)

            flow_linear, flow_conv = _pack_npu_weights(flow_modules)
            hift_linear, hift_conv = _pack_npu_weights(hift_modules)

            _wrap_flow_inference_dtype(code2wav.flow_model)
            logger.info(
                "CosyVoice3 310P: code2wav cast to FP16 (flow=%d hift=%d tensors) and packed "
                "(flow linear=%d conv=%d; hift linear=%d conv=%d; f0 predictor excluded)",
                flow_cast,
                hift_cast,
                flow_linear,
                flow_conv,
                hift_linear,
                hift_conv,
            )
            return loaded

    cv3_module.CosyVoice3Model = _CosyVoice3Code2Wav310P  # type: ignore[assignment]

    _CODE2WAV_PATCHED = True
    logger.info("Applied 310P patch for CosyVoice3 code2wav (FP16 forced, NZ/FRACTAL_Z packing)")
