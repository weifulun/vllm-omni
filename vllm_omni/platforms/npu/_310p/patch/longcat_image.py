# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch LongCat-Image for the 310P NPU diffusion path.

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); ``platform.py`` stays
untouched by this branch.  Dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
pipeline is constructed.

310P diffusion decision: pure ``torch_npu`` + FP16, no MindIE-SD kernels.
Invariants enforced here:

* ``od_config.dtype`` must be ``float16``.  310P is FP16-only (no BF16;
  ``EZ1001`` rejects it) and the whole LongCat-Image checkpoint set
  (transformer / text_encoder / vae) is shipped BF16, so every BF16
  checkpoint must land as FP16: the transformer params are FP16 via the
  loader's default-dtype wrap (weights are cast by the vLLM weight
  loaders), the VAE is cast FP16 after construction, and the Qwen2.5-VL
  text encoder is cast FP16 on host.
* ``text_encoder`` (Qwen2.5-VL-7B, ~16.6 GB) cannot co-reside with the 6B
  DiT (~12.5 GB FP16) on one 21589 MB chip: it is pinned to CPU in FP16
  (same playbook as task-11 SDXL CLIP pinning and task-16 M8).  The
  prompt-rewrite pass (``rewire_prompt`` AR ``generate``) and the embedding
  pass (``_encode_prompt``) run on host; prompt embeds are moved to the NPU
  in the runtime FP16 dtype.
* The pipeline is constructed under an inner ``torch.device("cpu")``
  context (loader-level CPU construction): the generic loader wraps the
  whole pipeline ``__init__`` in ``with target_device:`` so any
  deviceless tensor op during construction (RoPE freq tables, scheduler
  buffers, ``torch.zeros/ones`` initializations) would land on the NPU and
  trigger the 310P dynamic-TBE family (RealDiv at encoder RoPE init,
  OnesLike at VAE init; see the upstream issue family draft and
  research/012).  Constructing on CPU and moving components explicitly
  keeps every init-time deviceless op on host.
* The flow-matching timeline is computed on CPU (research/012 treatment
  chain): the per-step ``timestep / 1000`` division and the CFG-renorm
  ``cond_norm / (noise_norm + 1e-8)`` division are the RealDiv2 candidates
  in this pipeline's production graph.  The 310P subclass rewrites both to
  CPU-side FP16 computation (bit-identical IEEE-half results, no on-device
  Div), and the final VAE-unpack ``latents / scaling_factor`` becomes a
  scalar reciprocal multiply.  Same-seed patch on/off runs stay
  bit-comparable for the level-2 comparison (COMPARISON-POLICY.md §1).
* VAE decode is forced onto the diffusers tiling/slicing path via the
  standard ``od_config.vae_use_tiling / vae_use_slicing`` path (the
  untiled 768x1344 decode peak does not fit the per-chip budget with the
  DiT resident).  The VAE mid-block attention goes through the global 310P
  math-SDPA swap (patch/attention.py), which covers the diffusers
  ``F.scaled_dot_product_attention`` call.
* The diffusion attention backend resolves to ``TORCH_SDPA`` whenever the
  user did not select one (the math-SDPA decomposition patch applies
  regardless); an explicit MindIE-SD-only backend request fails at patch
  time.  Numerics-changing cache backends are disabled so the same-seed
  reference comparison is not broken by step-skipping caches.  Quantization
  tiers follow POLICY section 2 (FP16 primary; INT8 W8A8 the sanctioned
  fallback with mandatory precision evidence); FP8/MXFP* rejected.
* LoRA is accepted with a loud warning (out of scope for the parity
  baseline).

Component placement (logged at startup):

* ``transformer`` (6B DiT): NPU, FP16 (single chip, TP1).
* ``text_encoder`` (Qwen2.5-VL-7B): CPU, FP16 (pinned; AR rewrite included).
* ``vae`` (Flux-style 16-ch VAE): NPU, FP16, tiling+slicing forced.
* ``scheduler``: CPU numpy timeline; timesteps moved to the device as FP32
  (no on-device division anywhere in the denoise graph).

Environment knobs (A/B only, defaults are the 310P baseline):
``LONGCATIMAGE_310P_ENCODER_ON_NPU=1`` keeps the Qwen2.5-VL encoder on the
chip (expected OOM/EZ1001 at 768x1344 with the DiT resident; kept for
operator diagnosis only).
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any

import numpy as np
import torch
from diffusers.pipelines.longcat_image.system_messages import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image import (
    LongCatImagePipeline,
    calculate_shift,
    get_longcat_image_post_process_func,
    get_prompt_language,
    retrieve_timesteps,
    split_quotation,
)

logger = logging.getLogger(__name__)

LONGCAT_IMAGE_ARCH = "LongCatImagePipeline"
_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_ENCODER_ON_NPU_ENV = "LONGCATIMAGE_310P_ENCODER_ON_NPU"

_PATCHED = False

# ``get_longcat_image_post_process_func`` is re-exported so the diffusion
# registry can resolve the post-process func from this module (the 310P
# override registers under the same model_arch; COMPARISON-POLICY.md §5).
__all__ = [
    "_LongCatImage310P",
    "apply_diffusion_310p_patches",
    "apply_longcat_image_310p_patches",
    "get_longcat_image_post_process_func",
]


def _encoder_on_npu() -> bool:
    """Opt out of CPU encoder pinning for A/B experiments."""
    return os.environ.get(_ENCODER_ON_NPU_ENV, "0") == "1"


def _maybe_enable_vae_tiling(vae) -> bool:
    """Enable VAE tiling/slicing if the VAE supports it.

    Returns True if tiling was enabled.  The standard config path
    (``initialize_model`` applies ``od_config.vae_use_tiling``) is the
    primary mechanism; this is the defensive fallback.
    """
    if getattr(vae, "use_tiling", False):
        return True
    enable_tiling = getattr(vae, "enable_tiling", None)
    if callable(enable_tiling):
        try:
            enable_tiling()
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            logger.info("[310P][longcat] VAE tiling/slicing enabled (defensive wrap).")
            return True
        except Exception as exc:  # noqa: BLE001 - defensive only
            logger.warning("[310P][longcat] VAE tiling enable failed: %s", exc)
    logger.warning("[310P][longcat] VAE tiling could not be enabled; VAE decode may OOM.")
    return False


class _LongCatImage310P(LongCatImagePipeline):
    """LongCat-Image pipeline specialized for the 310P NPU path.

    Overrides relative to the generic pipeline:

    * construction under an inner CPU device context (loader-level CPU
      construction; see module docstring);
    * ``text_encoder`` pinned to CPU in FP16 (the BF16 checkpoint is cast
      on host);
    * VAE cast to FP16 after placement (checkpoint is BF16; 310P rejects
      BF16 with EZ1001);
    * VAE tiling/slicing forced (defensive; the od_config path is primary);
    * ``rewire_prompt`` / ``_encode_prompt`` run the Qwen2.5-VL encoder on
      host (its ``.to(device)`` calls are dropped) and move prompt embeds
      to the NPU in the runtime FP16 dtype;
    * the flow-matching timeline (per-step ``timestep / 1000``, CFG-renorm
      division, VAE-unpack division) is computed on CPU — no RealDiv2
      instances anywhere in the denoise graph (research/012 chain);
    * ``load_weights`` audits the final component dtype/device contract
      (transformer+VAE FP16 on NPU, CPU-pinned encoder FP32 — the CPU FP16
      GEMM path is unusable on this SoC, see ``load_weights``).
    """

    def __init__(self, *, od_config, prefix: str = "") -> None:
        # Loader-level CPU construction: the generic loader wraps pipeline
        # __init__ in ``with target_device:`` (NPU here); deviceless tensor
        # ops during that window would land on the NPU and can trip the
        # 310P dynamic-TBE family at model init (encoder RoPE RealDiv,
        # VAE OnesLike).  The inner CPU context keeps every init-time
        # deviceless op on host; explicit ``.to(self.device)`` calls inside
        # the base __init__ (VAE, scheduler timesteps) still place exactly
        # what the base pipeline places.
        with torch.device("cpu"):
            super().__init__(od_config=od_config, prefix=prefix)

        self._encoder_device = self.device
        if not _encoder_on_npu():
            # Preprocessing + AR rewrite module: keep it off the 310P NPU.
            # The checkpoint is BF16; the 310P is FP16-only, so cast on
            # host (BF16 -> FP16 is exact for these weights' ranges).
            self._encoder_device = _CPU_DEVICE
            self.text_encoder.to(device=_CPU_DEVICE, dtype=_RUNTIME_DTYPE)
            logger.info(
                "[310P][longcat] Qwen2.5-VL text encoder pinned to CPU (%s); embeds move to %s in %s.",
                _CPU_DEVICE,
                self.device,
                _RUNTIME_DTYPE,
            )

        # VAE checkpoint is BF16; 310P rejects BF16 (EZ1001).  Cast in place
        # (eager single-op dtype cast, proven safe on-card).
        if self.vae.dtype != _RUNTIME_DTYPE:
            self.vae.to(dtype=_RUNTIME_DTYPE)
            logger.info("[310P][longcat] VAE cast to %s on %s.", _RUNTIME_DTYPE, self.device)

        self._310p_vae_tiling = _maybe_enable_vae_tiling(self.vae)

        # VAE-unpack reciprocal (final ``latents / scaling_factor`` becomes a
        # scalar multiply; see forward override).
        scaling = float(self.vae.config.scaling_factor)
        self._310p_vae_inv_scale = 1.0 / scaling

    # ------------------------------------------------------------------
    # Prompt-rewrite + embedding: encoder runs on host, embeds on NPU.
    # ------------------------------------------------------------------

    def rewire_prompt(self, prompt, device):
        """Prompt rewrite (Qwen2.5-VL AR ``generate``) on the encoder device.

        Mirrors the generic implementation (v0.32.0; keep in sync) but keeps
        the processor outputs and the Qwen2.5-VL module on the encoder
        (host) device: no ``.to(device)`` on the processor batch, no
        ``self.text_encoder.to(device)``.  Only decoded text is returned, so
        no device tensors cross the CPU/NPU boundary.  ``device`` stays in
        the signature for API parity with the generic pipeline.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        all_text = []
        for each_prompt in prompt:
            language = get_prompt_language(each_prompt)
            if language == "zh":
                question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{each_prompt}\n改写后的prompt为："
            else:
                question = SYSTEM_PROMPT_EN + f"\nUser Input: {each_prompt}\nRewritten prompt:"
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = self.text_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            all_text.append(text)

        inputs = self.text_processor(text=all_text, padding=True, return_tensors="pt").to(self._encoder_device)

        generated_ids = self.text_encoder.generate(**inputs, max_new_tokens=self.tokenizer_max_length)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.text_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def _encode_prompt(self, prompt):
        """Embedding pass on the encoder (host) device; embeds to the NPU.

        Mirrors the generic implementation (v0.32.0; keep in sync) with the
        two ``.to(self.device)`` moves redirected to ``self._encoder_device``
        for the encoder forward; the returned hidden states (FP16 on host)
        are moved to the pipeline device in the runtime FP16 dtype — the
        base ``encode_prompt`` wrapper then no-ops its final
        ``.to(self.device)``.
        """
        batch_all_tokens = []

        for each_prompt in prompt:
            all_tokens = []
            for clean_prompt_sub, matched in split_quotation(each_prompt):
                if matched:
                    for sub_word in clean_prompt_sub:
                        tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                        all_tokens.extend(tokens)
                else:
                    tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)

            if len(all_tokens) > self.tokenizer_max_length:
                logger.warning(
                    "Your input was truncated because `max_sequence_length` is set to "
                    f"{self.tokenizer_max_length} input token nums : {len(all_tokens)}"
                )
                all_tokens = all_tokens[: self.tokenizer_max_length]
            batch_all_tokens.append(all_tokens)

        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_all_tokens},
            max_length=self.tokenizer_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        prefix_tokens = self.tokenizer(self.prompt_template_encode_prefix, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_tokens)
        suffix_len = len(suffix_tokens)

        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        batch_size = text_tokens_and_mask.input_ids.size(0)
        prefix_tokens_batch = prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        suffix_tokens_batch = suffix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_mask_batch = prefix_tokens_mask.unsqueeze(0).expand(batch_size, -1)
        suffix_mask_batch = suffix_tokens_mask.unsqueeze(0).expand(batch_size, -1)

        input_ids = torch.cat((prefix_tokens_batch, text_tokens_and_mask.input_ids, suffix_tokens_batch), dim=-1)
        attention_mask = torch.cat((prefix_mask_batch, text_tokens_and_mask.attention_mask, suffix_mask_batch), dim=-1)

        input_ids = input_ids.to(self._encoder_device)
        attention_mask = attention_mask.to(self._encoder_device)

        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]
        return prompt_embeds.to(dtype=_RUNTIME_DTYPE, device=self.device)

    def load_weights(self, weights):
        """Load weights, place the transformer on the NPU, audit the contract.

        This pipeline is constructed under an inner CPU device context
        (``__init__`` above), so the loader's in-place checkpoint copy leaves
        the transformer's parameters on host.  The default loader path performs
        no post-load device move for unquantized modules
        (``_process_weights_after_loading`` only relocates quantized ones), so
        the transformer is moved explicitly — the same "build on CPU first,
        then move" precedent the loader applies to custom pipelines — before
        the component audit runs.
        """
        loaded = super().load_weights(weights)
        first_param = next(self.transformer.parameters(), None)
        if first_param is not None and first_param.device != self.device:
            self.transformer.to(device=self.device)
            logger.info("[310P][longcat] transformer placed on %s after weight load.", self.device)
        if self._encoder_device.type == "cpu":
            # Measured on 125 (aarch64, torch 2.10.0+cpu / torch_npu
            # 2.10.0.post4): the CPU FP16 GEMM path is ~80x slower than the
            # FP32 path and does not scale across OpenMP threads at all
            # (2127x2127x3584 matmul: 95s FP16 vs 1.2s FP32, identical at 1
            # and 16 threads).  The checkpoint is BF16 and the encoder stays
            # host-resident, so upcasting to FP32 costs only host RAM (~2x)
            # and no NPU budget; the FP16 embed hand-off to the NPU is
            # unchanged (see _encode_prompt).
            self.text_encoder.to(dtype=torch.float32)
            logger.info("[310P][longcat] CPU text encoder upcast to float32 (CPU FP16 GEMM path unusable on this SoC).")
        self._310p_audit_components()
        return loaded

    def _310p_audit_components(self) -> None:
        """Verify the 310P dtype/device contract after weight load.

        * transformer: all params FP16 on the NPU (BF16 checkpoint cast by
          the vLLM weight loaders under the loader's FP16 default-dtype wrap);
        * vae: FP16 on the NPU (cast in __init__);
        * text_encoder: FP16 on the encoder (CPU) device.
        """
        problems = []
        dit_bad = sorted(
            {
                name
                for name, param in self.transformer.named_parameters()
                if param.dtype != _RUNTIME_DTYPE or param.device.type != self.device.type
            }
        )[:5]
        if dit_bad:
            problems.append(f"transformer params not ({_RUNTIME_DTYPE} on {self.device}): {dit_bad}")
        vae_bad = [
            name
            for name, param in self.vae.named_parameters()
            if param.dtype != _RUNTIME_DTYPE or param.device.type != self.device.type
        ][:5]
        if vae_bad:
            problems.append(f"vae params not ({_RUNTIME_DTYPE} on {self.device}): {vae_bad}")
        # CPU-pinned encoder runs FP32 (CPU FP16 GEMM is unusable on this
        # SoC; see load_weights); an NPU-pinned encoder stays runtime FP16.
        enc_dtype = torch.float32 if self._encoder_device.type == "cpu" else _RUNTIME_DTYPE
        enc_bad = [
            name
            for name, param in self.text_encoder.named_parameters()
            if param.dtype != enc_dtype or param.device.type != self._encoder_device.type
        ][:5]
        if enc_bad:
            problems.append(f"text_encoder params not ({enc_dtype} on {self._encoder_device}): {enc_bad}")
        if problems:
            raise RuntimeError("[310P][longcat] component contract violated after weight load:\n" + "\n".join(problems))
        logger.info(
            "[310P][longcat] component contract OK: transformer/vae %s on %s; text_encoder %s on %s; vae tiling=%s",
            _RUNTIME_DTYPE,
            self.device,
            enc_dtype,
            self._encoder_device,
            self._310p_vae_tiling,
        )

    # ------------------------------------------------------------------
    # Flow-matching timeline on CPU (research/012 treatment chain).
    # ------------------------------------------------------------------

    def forward(self, req) -> DiffusionOutput:
        """Denoising loop mirroring the generic pipeline (v0.32.0; keep in
        sync) with the 310P flow-matching timeline computed on CPU.

        Surgical changes relative to upstream ``forward``:
        1. ``"timestep": timestep / 1000`` (per step, appears in the
           positive AND negative kwargs) -> ``self._310p_scaled_timestep``
           (quotient computed on host, FP16 device tensor).  Removes the
           RealDiv2 instances from the per-step denoise graph — the 310P
           dynamic-TBE family (research/012; task-16 R13 wall).
        2. ``cfg_normalize_function`` (per step under CFG+renorm) -> the CPU
           ratio variant (``_310p_cfg_normalize_function``): the on-device
           ``cond_norm / (noise_norm + 1e-8)`` division is computed in
           FP16 on host (bit-identical IEEE-half division).
        3. Final VAE unpack ``(latents / scaling_factor) + shift_factor``
           -> scalar reciprocal multiply (``_310p_vae_inv_scale``).
        Everything else (rewire/encode plumbing, prepare_latents,
        retrieve_timesteps/scheduler, CFG combine, VAE decode) is upstream.
        """
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        prompt_embeds = None
        negative_prompt_embeds = None

        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.sampling_params.num_inference_steps or 50
        sigmas = req.sampling_params.sigmas
        generator = req.sampling_params.generator
        guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt if req.sampling_params.num_outputs_per_prompt > 0 else 1
        )
        latents = req.sampling_params.latents

        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        enable_prompt_rewrite: bool = extra_args.get("enable_prompt_rewrite", True)
        enable_cfg_renorm: bool = extra_args.get("enable_cfg_renorm", True)
        cfg_renorm_min: float = extra_args.get("cfg_renorm_min", 0.0)
        joint_attention_kwargs: dict[str, Any] | None = None
        output_type = req.sampling_params.output_type or "pil"

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        batch_size = len(prompt)

        device = self.device
        if enable_prompt_rewrite and prompt is not None:
            prompt = self.rewire_prompt(prompt if isinstance(prompt, list) else [prompt], device)

        negative_prompt = "" if negative_prompt is None else negative_prompt

        (prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
        )
        if self.do_classifier_free_guidance:
            (negative_prompt_embeds, negative_text_ids) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 4. Prepare latent variables
        num_channels_latents = 16
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        self._num_timesteps = len(timesteps)

        # handle guidance
        guidance = None

        if self._joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        prompt_embeds = prompt_embeds.to(device)
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(device)

        # custom partial function with cfg_renorm_min (identical line to
        # upstream; virtual dispatch resolves to the 310P CPU-ratio override).
        self.cfg_normalize_function = partial(self.cfg_normalize_function, cfg_renorm_min=cfg_renorm_min)

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            if self._interrupt:
                continue

            self._current_timestep = t

            # [310P change 1] per-step timestep scaling on CPU (no on-device
            # RealDiv2 in the denoise graph); expanded to the batch for shape
            # parity with the upstream ``t.expand(latents.shape[0])``.
            timestep_scaled = self._310p_scaled_timestep(t, latents.dtype, device).expand(latents.shape[0])

            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep_scaled,
                "guidance": guidance,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "return_dict": False,
            }
            if self.do_classifier_free_guidance:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep_scaled,
                    "guidance": guidance,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "txt_ids": negative_text_ids,
                    "img_ids": latent_image_ids,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=self.do_classifier_free_guidance,
                true_cfg_scale=guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=enable_cfg_renorm,
            )

            # Compute the previous noisy sample x_t -> x_t-1 with automatic CFG sync
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, self.do_classifier_free_guidance)

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            # [310P change 3] reciprocal multiply (no on-device division).
            latents = (latents * self._310p_vae_inv_scale) + self.vae.config.shift_factor

            if latents.dtype != self.vae.dtype:
                latents = latents.to(dtype=self.vae.dtype)

            image = self.vae.decode(latents, return_dict=False)[0]

        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else None
        return DiffusionOutput(output=image, stage_durations=stage_durations)

    @staticmethod
    def _310p_scaled_timestep(t: torch.Tensor, latents_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``t / 1000`` computed on CPU (no on-device RealDiv2).

        ``t`` is a 1-element device tensor (the scheduler's timestep); the
        official flow divides it on-device per step, which places a RealDiv2
        instance in the denoise graph — the 310P dynamic-TBE family that
        failed on task-16 (research/012).  The quotient is computed in
        FP32 on host and materialized as a 1-element FP16 device tensor
        (expanded to the batch at the call site); timesteps are exactly
        representable, so this is value-identical to the on-device division.
        """
        value = float(t.item()) / 1000.0
        return torch.full((1,), value, dtype=latents_dtype, device=device)

    def cfg_normalize_function(self, noise_pred: torch.Tensor, comb_pred: torch.Tensor, cfg_renorm_min: float = 0.0):
        """CFG renorm override: the norm ratio is computed on CPU (no
        on-device Div).

        Mirrors upstream ``cfg_normalize_function``; ``torch.norm`` stays
        on-device (fused Pow/Sum/Rsqrt, no division), the ratio tensor is
        computed in FP16 on host (bit-identical IEEE-half division) and
        moved back for the final scale multiply.  Overriding the base
        method name means every call path (base ``forward`` partial-binding,
        ``combine_cfg_noise``) resolves to the CPU-ratio variant.
        """
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        scale = (
            cond_norm.detach()
            .to("cpu")
            .div(noise_norm.detach().to("cpu") + 1e-8)
            .clamp(min=float(cfg_renorm_min), max=1.0)
            .to(noise_pred.device, noise_pred.dtype)
        )
        return comb_pred * scale


_MINDIE_SD_ONLY_BACKENDS = ("FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN")


def _install_attention_backend_guard(explicit_backend: Any) -> None:
    """Pin 310P diffusion attention resolution to the MindIE-SD-free path.

    Two layers:

    * an explicit user selection of a MindIE-SD-only backend (``FLASH_ATTN*``
      / ``RAINFUSION_ATTN``) raises here, up front, with an actionable
      message — otherwise it would fail deep in a kernel that does not exist
      on 310P (the 310P image ships no ``mindiesd``);
    * ``NPUOmniPlatform.get_diffusion_attn_backend_cls`` is wrapped so a
      default (``None``) selection resolves to ``TORCH_SDPA`` even on an
      image that shipped ``mindiesd`` (the platform default would otherwise
      pick ``FLASH_ATTN``).  The wrapper is installed once (idempotent) and
      is best-effort: if the platform module is unavailable in the current
      environment (contract tests), only the explicit-selection guard stays
      active.
    """
    if explicit_backend is not None and str(explicit_backend).upper() in _MINDIE_SD_ONLY_BACKENDS:
        raise ValueError(
            "310P does not ship MindIE-SD attention kernels; explicit backend "
            f"{explicit_backend!r} is not available on this card. Use TORCH_SDPA "
            "(the 310P math-SDPA decomposition applies automatically)."
        )

    try:
        from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
        from vllm_omni.platforms.npu.platform import NPUOmniPlatform
    except ImportError:
        logger.debug("[310P][longcat] platform backend guard skipped (imports unavailable).")
        return

    original = NPUOmniPlatform.__dict__["get_diffusion_attn_backend_cls"]
    underlying = getattr(original, "__func__", original)
    if getattr(underlying, "_310p_longcat_guarded", False):
        return

    @classmethod
    def _guarded(cls, selected_backend, head_size, allow_trtllm_default=True):
        if selected_backend is None:
            logger.debug("[310P][longcat] defaulting diffusion attention backend to TORCH_SDPA.")
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
        return underlying(cls, selected_backend, head_size, allow_trtllm_default=allow_trtllm_default)

    _guarded._310p_longcat_guarded = True  # type: ignore[attr-defined]
    NPUOmniPlatform.get_diffusion_attn_backend_cls = _guarded
    logger.info("[310P][longcat] diffusion attention backend guard installed (default -> TORCH_SDPA).")


def apply_diffusion_310p_patches(od_config) -> None:
    """Apply the LongCat-Image 310P runtime patches (idempotent).

    Registry entry point (COMPARISON-POLICY.md §5): dispatched from
    ``_310p.patch.apply_diffusion_310p_patches`` via the NPU platform's
    ``init_diffusion_model_runner_runtime`` before the diffusion pipeline
    is constructed, on 310P hosts only.  Safe to call more than once; the
    model-class registration happens once.
    """
    global _PATCHED

    # 1. FP16 enforcement (310P is FP16-only; the checkpoint set is BF16).
    if getattr(od_config, "dtype", None) != _RUNTIME_DTYPE:
        od_config.dtype = _RUNTIME_DTYPE
        logger.info("[310P][longcat] od_config.dtype forced to %s.", _RUNTIME_DTYPE)

    # 2. Attention backend: default to TORCH_SDPA (the 310P math-SDPA swap
    #    applies regardless via patch/attention.py); reject MindIE-SD-only
    #    explicit selections.
    per_role = getattr(od_config, "diffusion_attn_backend_per_role", None)
    backend_map = {getattr(role, "value", role): b for role, b in (per_role.items() if per_role else [])}
    diffusion_config = backend_map.get("diffusion", None)
    default_backend = getattr(od_config, "diffusion_attn_backend_default", None)
    explicit_backend = None
    if diffusion_config is not None:
        explicit_backend = diffusion_config.backend
    elif default_backend is not None:
        explicit_backend = getattr(default_backend, "backend", None)
    _install_attention_backend_guard(explicit_backend)

    # 3. VAE tiling/slicing via the standard config path (applied by
    #    initialize_model after construction) + defensive in-__init__ wrap.
    od_config.vae_use_tiling = True
    od_config.vae_use_slicing = True

    # 4. Quantization tiers (POLICY section 2): FP16 primary; INT8 W8A8
    #    fallback with loud warning; FP8/MXFP* rejected.
    quant_cfg = getattr(od_config, "quantization_config", None)
    if quant_cfg is not None:
        method = str(getattr(quant_cfg, "method", "")).lower()
        if method in ("int8", "w8a8", "int8_w8a8"):
            logger.warning(
                "[310P][longcat] INT8 W8A8 quantization is the sanctioned POLICY section-2 "
                "fallback tier; precision evidence against the FP16 baseline is mandatory in "
                "the real-device report (research/009: no 310P diffusion INT8 e2e precedent)."
            )
        elif method:
            raise ValueError(
                f"310P does not support diffusion quantization method {method!r}; use none (FP16) or int8."
            )

    # 5. LoRA guard.
    if getattr(od_config, "lora_path", None) is not None:
        logger.warning(
            "[310P][longcat] lora_path=%r is out of scope for the 310P reference baseline; "
            "outputs will not match the reference run weights.",
            od_config.lora_path,
        )

    if _PATCHED:
        return

    from vllm_omni.diffusion.registry import register_diffusion_model

    # Override the built-in pipeline with the 310P-specialized class under
    # the same model_arch (platform-optimised override path;
    # COMPARISON-POLICY.md §5).  The post-process func must be resolvable
    # from this module because _DIFFUSION_MODELS now points here.
    register_diffusion_model(
        LONGCAT_IMAGE_ARCH,
        module_name=__name__,
        class_name="_LongCatImage310P",
        post_process_func_name="get_longcat_image_post_process_func",
    )
    _PATCHED = True
    logger.info("[310P][longcat] Registered LongCat-Image 310P pipeline override for %s.", LONGCAT_IMAGE_ARCH)


# Backwards-compatible alias; the registry dispatch uses the canonical name.
apply_longcat_image_310p_patches = apply_diffusion_310p_patches
