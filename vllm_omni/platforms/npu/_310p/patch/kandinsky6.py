# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kandinsky 6 Lite TI2VA 310P runtime patch (task-30, v0.32.0 line).

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
diffusion pipeline is constructed.

Kandinsky 6 Lite (kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers) facts
driving the invariants:

- Unified video+audio DiT (``Kandinsky6Transformer3DModel``, 3.2B, 7.48 GB
  stored) jointly denoises HunyuanVideo-VAE video latents and MMAudioVAE
  audio latents; official README tier is 480x864 / 121 frames / 50 steps /
  CFG 5.0 / 44.1 kHz audio.  On ONE 21589 MB 310P3 chip the attention peak
  of the 28-head visual self-attention (math SDPA decomposition, S^2
  materialised) does not fit the official geometry: the 310P baseline tier
  keeps the official resolution and lowers the frame count (precedent:
  task-25 Cosmos3-Edge 49f, task-20 SANA-WM 49f).
- The Qwen2.5-VL text encoder (~16.6 GB) + CLIP text encoder cannot share
  the NPU with the DiT: both are preprocessing-only (run twice per request,
  cond + uncond) and are pinned to CPU in FP32 (SDXL task-11 precedent:
  encoder-on-CPU + embeds moved back to the accelerator in the runtime
  dtype).  ``VLLM_OMNI_310P_KANDINSKY6_ENCODER_ON_NPU=1`` opt-out kept for
  A/B only.
- ``KandinskyFlowMatchScheduler.set_timesteps`` divides on-device
  (``sigma = scale * t / (1 + (scale-1) * t)``) - the research/012
  RealDiv-2 dynamic-TBE family on 310P CANN 9.1.0.  The 310P subclass
  recomputes the sigma schedule on CPU and moves the two small tensors
  back to the DiT device; no division op ever lands on the NPU.
- Fused ``F.scaled_dot_product_attention`` is broken on 310P
  (research/012 dtprobe: ERR01007 on every dtype/mask/length); the explicit
  math decomposition lives in ``patch/attention.py`` and is installed by
  the registry dispatcher before this module runs - this module only gates
  the backend SELECTION (same as the Cosmos3 family).
- Cache backends (MagCache/TeaCache/Cache-DiT are wired for kandinsky6)
  change denoising numerics -> disabled for same-seed comparability
  (POLICY section 1).  Quantization: FP16 primary, INT8 W8A8 sanctioned
  fallback gate only (POLICY section 2, research/009); FP16 fits the
  single-chip budget so INT8 is unnecessary.
- Single-chip GO (task-30): TP1 only; the ported HunyuanVideo VAE ships
  with tiling enabled by default (``use_tiling = True`` in
  ``modeling_kandinsky6_vae.AutoencoderKLHunyuanVideo``), so no separate
  tiling forcing is needed here.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from transformers import AutoConfig, CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor as QwenAutoProcessor

from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import Kandinsky6Transformer3DModel
from vllm_omni.diffusion.models.kandinsky6.modeling_kandinsky6_vae import AutoencoderKLHunyuanVideo
from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import (
    _CLIP_MAX_LENGTH,
    _PROMPT_TEMPLATE,
    _QWEN_CROP_START,
    _build_audio_vae,
    _component_sources,
    _load_json,
    _resolve_bundle_root,
    Kandinsky6TI2VAPipeline,
)
from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6 import KandinskyFlowMatchScheduler

logger = logging.getLogger(__name__)

#: Pipeline class name serving the Kandinsky 6 family (Lite / Pro share
#: ``Kandinsky6TI2VAPipeline``; the HF model_index carries the same name).
KANDINSKY6_ARCH = "Kandinsky6TI2VAPipeline"

_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_ENCODER_ON_NPU_ENV = "VLLM_OMNI_310P_KANDINSKY6_ENCODER_ON_NPU"

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False


# ---------------------------------------------------------------------------
# Config-level invariants (run once per process, at pipeline construction)
# ---------------------------------------------------------------------------


def _force_fp16(od_config: Any) -> None:
    """310P is FP16-only; the Kandinsky 6 line has no bf16 decision."""
    dtype = getattr(od_config, "dtype", None)
    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}); "
        "310P is FP16-only and the Kandinsky 6 line has no bf16 decision "
        "(BF16-only NPU is absent; FP16 per COMPARISON-POLICY.md section 2). "
        "Pass float16 as the diffusion dtype."
    )


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections.

    The math decomposition in ``patch/attention.py`` then replaces both the
    ``SDPAImpl.forward_npu`` entry and ``torch.nn.functional``.
    ``scaled_dot_product_attention`` for anything the pipeline built itself.
    """
    from vllm_omni.diffusion.data import AttentionSpec

    attn_config = getattr(od_config, "diffusion_attention_config", None)
    if attn_config is None:
        return "TORCH_SDPA"

    default_spec = getattr(attn_config, "default", None)
    if default_spec is not None and str(getattr(default_spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
        raise RuntimeError(
            f"diffusion attention backend {default_spec.backend!r} requires MindIE-SD kernels, "
            "which are unavailable on 310P; use TORCH_SDPA."
        )

    per_role = getattr(attn_config, "per_role", None) or {}
    for role, spec in per_role.items():
        if str(getattr(spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention per-role backend for {role!r} is {spec.backend!r}, "
                "which requires MindIE-SD kernels unavailable on 310P; use TORCH_SDPA."
            )

    if default_spec is None:
        attn_config.default = AttentionSpec(backend="TORCH_SDPA")
    return "TORCH_SDPA"


def _disable_numerical_cache_backends(od_config: Any) -> None:
    backend = getattr(od_config, "cache_backend", None)
    if backend is None or backend == "none":
        return
    if backend not in _NUMERICAL_CACHE_BACKENDS:
        logger.warning(
            "[310P][kandinsky6] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][kandinsky6] disabling numerics-changing cache backend %r on 310P "
        "(same-seed reference comparison per COMPARISON-POLICY.md section 1); "
        "re-enable after 310P validation.",
        backend,
    )
    od_config.cache_backend = "none"


def _quantization_method(quant_config: Any) -> str | None:
    """Resolve the quantization method name from a QuantizationConfig object."""
    getter = getattr(quant_config, "get_name", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # pragma: no cover - defensive, contract-level
            pass
    for attr in ("method", "name"):
        value = getattr(quant_config, attr, None)
        if isinstance(value, str):
            return value
    return None


def _validate_quantization(od_config: Any) -> None:
    """Precision-tier gate: FP16 primary, INT8 W8A8 sanctioned fallback only.

    POLICY section 2 (revised 2026-11-02): FP16 is the first choice; when
    weights do not fit the chip budget, INT8 is the fallback tier (no BF16,
    no over-budget force-fitting, the quantization configuration must be
    recorded in the report).  research/009: on 310P only INT8 W8A8 has a
    hardware benefit (``npu_quant_matmul``); FP8/MXFP* fall back to FP16
    matmul with no gain; bitsandbytes is CUDA-only.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][kandinsky6] quantization_config method=%r accepted as the "
            "POLICY section-2 second tier (research/009): online dynamic INT8 "
            "via npu_quant_matmul, layers with output dim > 65535 fall back "
            "to unquantized (mixed-precision semantics). The per-layer "
            "fallback log is MANDATORY in the real-device report, together "
            "with precision evidence vs the FP16 reference. On Kandinsky 6 "
            "Lite (3.2B DiT, 7.48 GB FP16) INT8 is unnecessary - the FP16 "
            "tier fits the single-chip budget.",
            method,
        )
        return
    raise RuntimeError(
        f"310P Kandinsky 6 rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): FP16 is the primary tier "
        "(quantization_config unset; the 3.2B Lite DiT fits the single-chip "
        "21589 MB budget at FP16); only the INT8 W8A8 fallback "
        "(method='int8') is sanctioned on 310P (POLICY section 2, "
        "research/009). FP8/MXFP* give no hardware benefit on 310P (served "
        "with FP16 matmul); bitsandbytes is CUDA-only."
    )


def _validate_single_chip_parallelism(od_config: Any) -> None:
    """GO task-30 is single-chip TP1: reject multi-rank parallel degrees.

    Sequence parallel (Ulysses/Ring/AllGather-KV) and CFG parallel each need
    more than one diffusion worker rank; on one Duo card that routes into the
    multi-worker HCCL path whose gate is CLOSED (GATE-TP2-STATUS.md).  The
    single-worker path handles CFG sequentially (two forwards at
    cfg_parallel_size=1), which is the intended 310P topology.
    """
    for attr in ("ulysses_degree", "ring_degree", "allgather_degree"):
        value = getattr(od_config, attr, 1) or 1
        if value != 1:
            raise RuntimeError(
                f"310P single-chip Kandinsky 6 run rejects {attr}={value}: "
                "sequence parallel needs multi-rank HCCL, whose gate is closed "
                "(GATE-TP2-STATUS.md). Run TP1; CFG is handled sequentially "
                "by the pipeline at cfg_parallel_size=1."
            )
    if (getattr(od_config, "cfg_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            f"310P single-chip Kandinsky 6 run rejects cfg_parallel_size="
            f"{getattr(od_config, 'cfg_parallel_size', None)}: CFG parallel "
            "needs two worker ranks. Run cfg_parallel_size=1 (sequential CFG)."
        )


def _install_attention_guard() -> None:
    """Wrap the NPU platform backend selector so 310P can never resolve to
    MindIE-SD-only kernels, even for per-role or per-request selections that
    bypass the od_config-level default.

    Idempotent across model modules: if a sibling 310P patch module already
    installed the guard (sentinel attribute), do not double-wrap.
    """
    global _ATTENTION_GUARD_INSTALLED
    if _ATTENTION_GUARD_INSTALLED:
        return

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    current = NPUOmniPlatform.__dict__.get("get_diffusion_attn_backend_cls")
    if getattr(current, "_310p_attention_guarded", False):
        _ATTENTION_GUARD_INSTALLED = True
        return

    # Unwrap the classmethod descriptor so ``cls`` is passed exactly once.
    original = getattr(current, "__func__", current)

    @classmethod
    def _guarded(cls, selected_backend, head_size, allow_trtllm_default: bool = True):  # noqa: ANN001
        if selected_backend is None:
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
        if str(selected_backend).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention backend {selected_backend!r} requires MindIE-SD "
                "kernels, which are unavailable on 310P; use TORCH_SDPA."
            )
        return original(cls, selected_backend, head_size, allow_trtllm_default=allow_trtllm_default)

    _guarded._310p_attention_guarded = True  # type: ignore[attr-defined]
    NPUOmniPlatform.get_diffusion_attn_backend_cls = _guarded
    _ATTENTION_GUARD_INSTALLED = True


def _log_placement_plan(od_config: Any) -> None:
    logger.info(
        "[310P][kandinsky6] placement plan: dtype=fp16 attention=TORCH_SDPA "
        "(math decomposition) vae_tiling=built-in(use_tiling=True) "
        "text_encoder=cpu-fp32 text_encoder_2=cpu-fp32 scheduler=cpu-flow-match "
        "parallel=TP1(sequential CFG) model_class=%s",
        getattr(od_config, "model_class_name", None),
    )


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all Kandinsky 6 310P invariants. Idempotent per process.

    Contract name required by the 310P diffusion patch registry
    (``_DIFFUSION_310P_PATCH_MODULES`` dispatch in
    ``vllm_omni.platforms.npu._310p.patch``).
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _force_fp16(od_config)
    _validate_attention_selection(od_config)
    _disable_numerical_cache_backends(od_config)
    _validate_quantization(od_config)
    _validate_single_chip_parallelism(od_config)
    _register_pipeline_override()
    _install_attention_guard()
    _log_placement_plan(od_config)


def _register_pipeline_override() -> None:
    """Swap the built-in pipeline for the 310P-specialized subclass.

    Follows the task-11 (SDXL) platform-optimised override path:
    ``register_diffusion_model`` under the same ``model_arch``; the
    pre/post-process funcs stay the built-ins (they are standalone functions
    of ``od_config`` and do not depend on the pipeline class), so they are
    not re-registered.
    """
    from vllm_omni.diffusion.registry import register_diffusion_model

    register_diffusion_model(
        KANDINSKY6_ARCH,
        module_name=__name__,
        class_name="_Kandinsky6TI2VA310P",
    )
    logger.info("[310P] Registered Kandinsky 6 310P pipeline override for %s.", KANDINSKY6_ARCH)


# ---------------------------------------------------------------------------
# 310P-specialized pipeline
# ---------------------------------------------------------------------------


def _encoder_on_npu() -> bool:
    """Opt out of CPU encoder pinning for A/B experiments."""
    return os.environ.get(_ENCODER_ON_NPU_ENV, "0") == "1"


def _pin_scheduler_sigma_math_to_cpu(scheduler: KandinskyFlowMatchScheduler) -> None:
    """Recompute the sigma schedule on CPU; move the small tensors back.

    ``KandinskyFlowMatchScheduler.set_timesteps`` evaluates
    ``sigma = scale * t / (1 + (scale-1) * t)`` on whatever device the
    denoise loop passes (the DiT device on 310P = NPU).  On-device division
    is the research/012 RealDiv-2 dynamic-TBE pre-compile family on CANN
    9.1.0.  Computing the (num_steps+1)-element schedule on CPU in FP32 and
    moving ``timesteps``/``sigmas`` back is numerically identical for the
    DiT (it only indexes ``sigmas`` and consumes ``timesteps``) and removes
    every division op from the NPU graph.
    """
    original_set_timesteps = scheduler.set_timesteps

    def _set_timesteps_cpu(num_inference_steps, device=None, **kwargs):
        original_set_timesteps(num_inference_steps, device="cpu", **kwargs)
        target = torch.device(device) if device is not None else torch.device("cpu")
        scheduler.timesteps = scheduler.timesteps.to(target)
        scheduler.sigmas = scheduler.sigmas.to(target)

    scheduler.set_timesteps = _set_timesteps_cpu
    scheduler.device = torch.device("cpu")


class _Kandinsky6TI2VA310P(Kandinsky6TI2VAPipeline):
    """Kandinsky 6 TI2VA pipeline specialized for the 310P NPU path.

    Component placement on 310P:

    * ``text_encoder`` (Qwen2.5-VL 7B, ~16.6 GB) + ``text_encoder_2``
      (CLIP): preprocessing-only, run twice per request (cond + uncond).
      Pinned to CPU in FP32 so they cannot co-reside with the DiT on one
      21589 MB chip; prompt embeds are moved back to the NPU in the runtime
      FP16 dtype by ``_move_text_embeds`` (base behaviour).
    * transformer + video VAE + audio VAE/vocoder: NPU, FP16 (forced by the
      runtime patch); the ported HunyuanVideo VAE ships tiling enabled by
      default.
    * scheduler: sigma schedule computed on CPU (RealDiv-2 family,
      research/012), tensors moved back to the DiT device.
    """

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._encoder_device = self.device
        if not _encoder_on_npu():
            self._encoder_device = torch.device("cpu")
            _pin_scheduler_sigma_math_to_cpu(self.scheduler)
            logger.info(
                "[310P] Kandinsky 6 text encoders pinned to CPU (FP32); "
                "sigma schedule computed on CPU; embeds moved to %s in %s.",
                self.device,
                torch.float16,
            )

    def _load_components(self, od_config, prefix):  # noqa: ANN001
        """Mirror of the base method with per-component load devices.

        Base (v0.32.0) decides ONE ``load_device`` for every component
        (NPU when offload is disabled).  On 310P the Qwen2.5-VL text
        encoder (~16.6 GB) must not load onto the same chip as the DiT
        (7.48 GB) + video VAE + audio VAE: the two text encoders therefore
        load on CPU (FP32, matching the SDXL task-11 CPU-encoder tier),
        everything else on the NPU.  Everything else is verbatim.
        """
        del prefix
        from diffusers.models.modeling_utils import no_init_weights

        model_root = _resolve_bundle_root(od_config.model)
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model_config = dict(od_config.model_config or {})

        npu_device = self.device
        encoder_device = self.device if _encoder_on_npu() else torch.device("cpu")
        scheduler_device = "cpu" if not _encoder_on_npu() else self.device

        transformer_dir = os.path.join(model_root, "transformer")
        with open(os.path.join(transformer_dir, "config.json"), encoding="utf-8") as handle:
            transformer_config = json.load(handle)
        with torch.device(npu_device), no_init_weights():
            transformer = Kandinsky6Transformer3DModel.from_diffusers_config(
                transformer_config,
                quant_config=getattr(od_config, "quantization_config", None),
            )

        vae_dir = os.path.join(model_root, "vae")
        with open(os.path.join(vae_dir, "config.json"), encoding="utf-8") as handle:
            vae_config = json.load(handle)
        with torch.device(npu_device), no_init_weights():
            vae = AutoencoderKLHunyuanVideo.from_config(vae_config)
        vae.to(dtype=torch.float16)

        text_encoder_dir = os.path.join(model_root, "text_encoder")
        text_encoder_config = AutoConfig.from_pretrained(text_encoder_dir)
        with torch.device(encoder_device), no_init_weights():
            text_encoder = Qwen2_5_VLForConditionalGeneration(text_encoder_config)
        # CPU FP32: the 310P x86 host has no native FP16 CPU kernels
        # (emulated = slower, not more accurate); embeds are cast to the
        # runtime FP16 dtype when moved back to the NPU.
        text_encoder.to(dtype=torch.float32)

        tokenizer = QwenAutoProcessor.from_pretrained(os.path.join(model_root, "tokenizer"))
        # Pin the padding side the same way the base pipeline does:
        # encode_prompt() crops embeddings at a fixed offset assuming the
        # system-prompt prefix leads and padding trails.
        tokenizer.tokenizer.padding_side = "right"

        clip_dir = os.path.join(model_root, "text_encoder_2")
        clip_config = AutoConfig.from_pretrained(clip_dir)
        with torch.device(encoder_device), no_init_weights():
            text_encoder_2 = CLIPTextModel(clip_config)
        text_encoder_2.to(dtype=torch.float32)
        tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(model_root, "tokenizer_2"))

        audio_vae = None
        audio_vae_dir = os.path.join(model_root, "audio_vae")
        include_audio_vae = bool(model_config.get("sample_audio", True)) and os.path.isdir(audio_vae_dir)
        if include_audio_vae:
            with torch.device(npu_device), no_init_weights():
                audio_vae = _build_audio_vae(audio_vae_dir)
            audio_vae.to(dtype=dtype)

        self.weights_sources = _component_sources(model_root, include_audio_vae=include_audio_vae)

        checkpoint_scheduler_scale = 5.0
        scheduler_config_path = os.path.join(model_root, "scheduler", "scheduler_config.json")
        if os.path.isfile(scheduler_config_path):
            checkpoint_scheduler_scale = float(
                _load_json(model_root, "scheduler/scheduler_config.json").get("shift", checkpoint_scheduler_scale)
            )
        scheduler_scale = float(model_config.get("scheduler_scale", checkpoint_scheduler_scale))
        scheduler = KandinskyFlowMatchScheduler(scheduler_scale=scheduler_scale, device=scheduler_device)

        return transformer, vae, text_encoder, audio_vae, scheduler, tokenizer, text_encoder_2, tokenizer_2

    def encode_prompt(self, text: str):  # noqa: ANN001
        """Base method with encoder inputs routed to ``self._encoder_device``.

        The returned embeds stay on the encoder device; the caller moves them
        to the DiT device via ``_move_text_embeds`` (base behaviour, FP32 ->
        FP16 cast).
        """
        full_text = _PROMPT_TEMPLATE.format(text)
        # Under the 310P CPU-encoder placement the encoders NEVER move to the
        # accelerator (no offload hooks by design): inputs go to the encoder
        # device, not ``self.device``.
        qwen_device = getattr(self, "_encoder_device", self.device)
        inputs = self.tokenizer(
            text=[full_text],
            images=None,
            videos=None,
            max_length=self.max_sequence_length + _QWEN_CROP_START,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(qwen_device)
        qwen_output = self.text_encoder(
            input_ids=inputs["input_ids"],
            return_dict=True,
            output_hidden_states=True,
        )
        embeds = qwen_output["hidden_states"][-1][:, _QWEN_CROP_START:]
        attention = inputs["attention_mask"][:, _QWEN_CROP_START:].to(dtype=torch.bool)
        if self.text_token_padding:
            qwen_embeds = embeds[0]
            qwen_attention = attention[0]
            cu_seqlens = torch.tensor([0, qwen_embeds.shape[0]], dtype=torch.int32, device=qwen_embeds.device)
        else:
            qwen_embeds = embeds[attention]
            qwen_attention = None
            cu_seqlens = torch.tensor([0, int(attention.sum().item())], dtype=torch.int32, device=embeds.device)

        clip_device = qwen_device
        clip_inputs = self.tokenizer_2(
            [text],
            max_length=_CLIP_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(clip_device)
        pooled_embed = self.text_encoder_2(**clip_inputs)["pooler_output"]
        return {"text_embeds": qwen_embeds, "pooled_embed": pooled_embed}, cu_seqlens, qwen_attention
