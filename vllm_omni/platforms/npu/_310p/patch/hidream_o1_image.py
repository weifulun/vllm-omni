# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HiDream-O1-Image 310P runtime patch (task-28, v0.32.0 line).

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
diffusion pipeline is constructed.

HiDream-O1-Image facts driving the invariants (code-verified on the v0.32.0
tag 5917730 pipeline/transformer sources):

- ``HiDreamO1ImagePipeline`` is a flow-matching refinement loop over ONE
  unified Qwen3-VL-based transformer (8.8 B params).  There is NO separate
  text encoder and NO VAE: pixels are emitted directly as 32x32 RGB patch
  vectors and rearranged to an image on the CPU (``get_hidream_o1_image_
  post_process_func``).  The classic-diffusion "encoder pinned to CPU" and
  "VAE tiling" invariants therefore have no surface here (documented N/A).
- The HF checkpoint stores F32 weights (8 shards, 35.2 GB on disk).  The
  transformer's ``load_weights`` skips ``model.visual.*`` (vision tower,
  unused for T2I) and ``lm_head.weight`` (14.17 GiB loaded FP16 weights).
  MEASURED on 126 C1 (serial 2106030655ZEP5000578, 21.08 GiB chip): the FP16
  tier OOMs - 14.17 GiB weights + ~3.4 GiB attention/softmax peak (2.03 GiB
  S x S fp32 softmax request at 2048x2048), two independent reproductions at
  the same allocation point.  The GO ladder therefore lands on the POLICY
  section-2 second tier: INT8 W8A8 (online dynamic, research/009).
- The pipeline hard-asserts BF16 (``_validate_static_config`` raises unless
  ``od_config.dtype == bfloat16``), and the engine itself sets
  ``od_config.dtype=bfloat16`` by default.  310P is FP16-only (POLICY section
  2, ENV-PRECHECK P1-5) with no BF16 hardware at all: this patch COERCES the
  od_config dtype (None/bfloat16 -> float16) before the pipeline is built and
  widens the gate to accept float16 (the gate still rejects a bf16
  ``self.dtype`` as the hardware boundary).
- ``FlowUniPCMultistepScheduler`` is built with ``device=self.device``
  (NPU): the per-step sigma division (``timesteps/1000``), the ``v =
  (x_pred - z) / sigma`` velocity math and ``sched.step`` (fp32 log/exp/div
  solver math) all land on the NPU - exactly the research/012 RealDiv
  dynamic-TBE family that blocked the qwenimg21 line.  The patched forward
  keeps the SAME fp32 math but executes every scheduler-side op on CPU and
  round-trips the latent (12.6 M floats at 2048x2048, ~50 MB) once per step.
- ``HiDreamO1RotaryEmbedding`` builds cos/sin from a REAL dtype freqs table
  (no ``torch.polar`` / complex64, so the 012 complex-dtype gap does not
  apply), but computes it in fp32 on the hidden-state device (NPU).  The
  patched forward builds the table on CPU and moves the REAL cos/sin tensors
  to the NPU (012 caliber: "extract as REAL tensors, then upload").
- ``TimestepEmbedder.timestep_embedding`` computes its sine features in fp32
  on the timestep device (NPU); the patch forces the feature math onto CPU
  and returns the result to the original device before the (NPU) MLP.
- Fused SDPA is dead on 310P (research/012): the engine dispatcher's math
  decomposition (``patch/attention.py``) covers the attention layer; the
  pipeline's S x S additive mask (TORCH_SDPA advertises
  ``supports_piecewise_spans=False`` so the dense-mask path is taken) is
  consumed by the math path, verified on the sibling SDXL line.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Pipeline class name serving HiDream-O1-Image (auto-resolved by
#: ``vllm_omni.diffusion.data`` - the registry key dispatched by the 310P
#: diffusion patch registry).
HIDREAM_ARCHS = ("HiDreamO1ImagePipeline",)

_MINDIE_SD_ONLY_BACKENDS = frozenset({"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"})

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False
_PIPELINE_PATCHED = False
_FP16_SOFTMAX_SWAPPED = False


def _pipeline_module():
    from vllm_omni.diffusion.models.hidream_o1_image import pipeline_hidream_o1_image as m

    return m


def _pin_fp16(od_config: Any) -> None:
    """310P is FP16-only: coerce the diffusion dtype to float16.

    The engine sets ``od_config.dtype`` to ``torch.bfloat16`` by default (the
    HiDream recipe passes no explicit dtype), and 310P has no BF16 hardware
    at all (hardware-level EZ1001 reject, ENV-PRECHECK P1-5).  FP16 is the
    POLICY section-2 primary tier and the ONLY half precision this card can
    run, so a BF16 dtype - whether the engine default or an explicit request
    - is COERCED to float16 rather than rejected (rejecting would make the
    default config impossible to launch on 310P).  Non-half dtypes (fp32...)
    are still rejected.  The pipeline-level gate keeps rejecting
    ``self.dtype == bfloat16`` as the hardware boundary.
    """
    dtype = getattr(od_config, "dtype", None)
    if dtype is None:
        od_config.dtype = torch.float16
        logger.warning(
            "[310P][hidream-o1] od_config.dtype was None (engine default); "
            "pinned to float16 (310P is FP16-only, ENV-PRECHECK P1-5, "
            "COMPARISON-POLICY.md section 2)."
        )
        return
    if dtype == torch.bfloat16:
        od_config.dtype = torch.float16
        logger.warning(
            "[310P][hidream-o1] od_config.dtype was bfloat16 (engine default); "
            "COERCED to float16 - 310P has no BF16 hardware (ENV-PRECHECK "
            "P1-5, COMPARISON-POLICY.md section 2)."
        )
        return
    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}): 310P has "
        "no BF16 (hardware-level EZ1001 reject) and FP16 is the POLICY "
        "section-2 primary tier."
    )


def _install_pipeline_patches() -> None:
    """Widen the bf16 gate and move the flow-matching loop onto the CPU
    scheduler (research/012 RealDiv family avoidance).  Idempotent per
    process."""
    global _PIPELINE_PATCHED
    if _PIPELINE_PATCHED:
        return

    m = _pipeline_module()
    pipeline = m.HiDreamO1ImagePipeline

    # ---- 1. bf16-only gate: accept FP16 (the 310P tier), keep the
    # cfg_parallel preflight (sequential CFG is the supported 310P
    # topology).
    def _validate_static_config_310p(self) -> None:
        import torch as _t

        if self.dtype not in (_t.float16, _t.bfloat16):
            raise NotImplementedError(f"only half precision is supported; got {self.dtype!r}")
        if self.dtype == _t.bfloat16:
            raise NotImplementedError("310P is FP16-only: bfloat16 is rejected (POLICY section 2)")
        cfg_parallel_size = int(self.od_config.parallel_config.cfg_parallel_size)
        if cfg_parallel_size > 1:
            raise NotImplementedError(f"cfg_parallel_size={cfg_parallel_size} not supported")

    pipeline._validate_static_config = _validate_static_config_310p

    # ---- 2. forward: identical math, scheduler-side ops on CPU.
    PATCH_SIZE = m.PATCH_SIZE
    T_EPS = m.T_EPS
    build_hidream_o1_scheduler = m.build_hidream_o1_scheduler
    DiffusionOutput = m.DiffusionOutput

    def _forward_310p(self, req):
        with torch.inference_mode():
            prompt, height, width, num_inference_steps, seed, guidance_scale = self._resolve_generation_params(req)

            expected_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
            cond_sample = self._build_and_validate_sample(
                prompt,
                height,
                width,
                expected_image_len,
            )
            samples: list = [cond_sample]
            if guidance_scale > 1.0:
                uncond_sample = self._build_and_validate_sample(
                    " ",
                    height,
                    width,
                    expected_image_len,
                )
                samples.append(uncond_sample)

            z = self._prepare_noise_and_patchify(
                height,
                width,
                seed,
                self.dtype,
                self.device,
            )

            # The flow-matching timetable stays on CPU (research/012 RealDiv
            # dynamic-TBE family; the scheduler moves sigmas onto the sample
            # device internally, so a CPU sample keeps all solver math on CPU).
            sched = build_hidream_o1_scheduler(
                scheduler_name="default",
                num_inference_steps=num_inference_steps,
                shift=3.0,
                device=torch.device("cpu"),
            )

            with self.progress_bar(total=len(sched.timesteps)) as pbar:
                for step_t in sched.timesteps:
                    step_t_f32 = step_t.float()  # CPU
                    t_pixeldit = 1.0 - step_t_f32 / 1000.0  # CPU
                    sigma = (step_t_f32 / 1000.0).clamp_min(T_EPS)  # CPU f32

                    z_cpu = z.float().cpu()

                    x_pred_cond = self._forward_once(samples[0], z, t_pixeldit)
                    v_cond = (x_pred_cond.to(torch.float32).cpu() - z_cpu) / sigma

                    if len(samples) > 1:
                        x_pred_uncond = self._forward_once(samples[1], z, t_pixeldit)
                        v_uncond = (x_pred_uncond.to(torch.float32).cpu() - z_cpu) / sigma
                        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v_guided = v_cond

                    model_output = -v_guided

                    z = sched.step(model_output, step_t_f32, z_cpu, return_dict=False)[0].to(self.dtype).to(self.device)
                    pbar.update()

            return DiffusionOutput(
                output=(z, height, width),
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

    pipeline.forward = _forward_310p

    # ---- 3. RoPE cos/sin table: build on CPU, upload REAL tensors.
    from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
        HiDreamO1RotaryEmbedding,
    )

    def _rope_forward_310p(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 012 caliber: the table is REAL dtype (cos/sin, no complex); build it
        # in fp32 on CPU and move the finished REAL tensors to the NPU.
        device = torch.device("cpu")
        inv_freq = self.inv_freq[None, None, :, None].float().to(device)
        inv_freq = inv_freq.expand(3, position_ids.shape[1], -1, 1)
        positions = position_ids[:, :, None, :].float().to(device)
        with torch.autocast(device_type="cpu", enabled=False):
            freqs = (inv_freq @ positions).transpose(2, 3)
            freqs_t = freqs[0]
            for dim, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[dim] * 3
                freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
            emb = torch.cat((freqs_t, freqs_t), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(hidden_states.device, hidden_states.dtype), sin.to(hidden_states.device, hidden_states.dtype)

    HiDreamO1RotaryEmbedding.forward = _rope_forward_310p

    # ---- 4. TimestepEmbedder sine features: CPU math, upload the result.
    from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
        TimestepEmbedder,
    )

    @staticmethod
    def _timestep_embedding_310p(timestep: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        target_device = timestep.device
        t = timestep.cpu()
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device="cpu") / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(target_device)

    TimestepEmbedder.timestep_embedding = _timestep_embedding_310p

    _PIPELINE_PATCHED = True
    logger.info(
        "310P: HiDreamO1 pipeline patched (fp16 gate, CPU flow-matching "
        "scheduler, CPU RoPE table, CPU timestep features)"
    )


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections.

    The math decomposition in ``patch/attention.py`` (installed by the
    registry dispatcher before this module runs) replaces the fused SDPA
    entry points; this only gates the backend SELECTION.
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
            "[310P][hidream-o1] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][hidream-o1] disabling numerics-changing cache backend %r on 310P "
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

    POLICY section 2: FP16 first; INT8 is the fallback tier only when the
    weights do not fit the chip budget (no BF16, no over-budget forcing).
    research/009: on 310P only INT8 W8A8 has a hardware benefit; FP8/MXFP*
    fall back to FP16 matmul with no gain, ConvRot is unvalidated,
    bitsandbytes is CUDA-only.  MEASURED: the FP16 tier OOMs on the 21.08 GiB
    single chip (14.17 GiB FP16 weights + ~3.4 GiB attention peak > capacity),
    so on HiDream-O1-Image INT8 W8A8 is the ACTIVE tier on this card - the
    gate admits it with a warning instead of expecting it to stay unused.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][hidream-o1] quantization_config method=%r accepted as the "
            "POLICY section-2 second tier (research/009): online dynamic INT8 "
            "via npu_quant_matmul, layers with output dim > 65535 fall back "
            "to unquantized (mixed-precision semantics). MEASURED: the FP16 "
            "tier OOMs on the 21.08 GiB chip (14.17 GiB weights + ~3.4 GiB "
            "attention peak), so INT8 is the ACTIVE tier on HiDream-O1-Image "
            "on this card. 310P has no diffusion INT8 e2e precedent - the "
            "per-layer fallback log and precision evidence vs the FP16 "
            "reference are MANDATORY in the real-device report.",
            method,
        )
        return
    raise RuntimeError(
        f"310P HiDream-O1-Image rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): FP16 is the primary tier "
        "(quantization_config unset; HiDream-O1-Image fits the single-chip "
        "21589 MB budget at FP16); only the INT8 W8A8 fallback "
        "(method='int8') is sanctioned on 310P (POLICY section 2, "
        "research/009). FP8/MXFP* give no hardware benefit on 310P (served "
        "with FP16 matmul); ConvRot-INT4 is unvalidated; bitsandbytes is "
        "CUDA-only."
    )


def _validate_single_chip_parallelism(od_config: Any) -> None:
    """GO task-28 is single-chip TP1: reject multi-rank parallel degrees.

    Sequence parallel (Ulysses/Ring/AllGather-KV) and CFG parallel each need
    more than one diffusion worker rank; on one Duo card that routes into the
    multi-worker HCCL path whose gate is CLOSED (GATE-TP2-STATUS.md).  The
    single-worker path handles CFG sequentially (two forwards per step, the
    pipeline's default at cfg_parallel_size=1), which is the intended 310P
    topology.
    """
    for attr in ("ulysses_degree", "ring_degree", "allgather_degree"):
        value = getattr(od_config, attr, 1) or 1
        if value != 1:
            raise RuntimeError(
                f"310P single-chip HiDream-O1 run rejects {attr}={value}: "
                "sequence parallel needs multi-rank HCCL, whose gate is closed "
                "(GATE-TP2-STATUS.md). Run TP1; CFG is handled sequentially "
                "by the pipeline at cfg_parallel_size=1."
            )
    if (getattr(od_config, "cfg_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            f"310P single-chip HiDream-O1 run rejects cfg_parallel_size="
            f"{getattr(od_config, 'cfg_parallel_size', None)}: CFG parallel "
            "needs two worker ranks. Run cfg_parallel_size=1 (sequential CFG)."
        )


def _install_attention_guard() -> None:
    """Wrap the NPU platform backend selector so 310P can never resolve to
    MindIE-SD-only kernels, even for per-role or per-request selections that
    bypass the od_config-level default."""
    global _ATTENTION_GUARD_INSTALLED
    if _ATTENTION_GUARD_INSTALLED:
        return

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    # Unwrap the classmethod descriptor so ``cls`` is passed exactly once.
    original = NPUOmniPlatform.__dict__["get_diffusion_attn_backend_cls"]
    original = getattr(original, "__func__", original)

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

    NPUOmniPlatform.get_diffusion_attn_backend_cls = _guarded
    _ATTENTION_GUARD_INSTALLED = True


def _log_placement_plan(od_config: Any) -> None:
    logger.info(
        "[310P][hidream-o1] placement plan: dtype=fp16(pinned) "
        "attention=TORCH_SDPA(math decomposition) scheduler=cpu-flow-unipc "
        "rope=cpu-table timestep-features=cpu cache=off parallel=TP1 "
        "(sequential CFG, 2x forward/step) encoder=none(unified model; "
        "vision tower skipped by load_weights) vae=none(pixels direct) "
        "model_class=%s",
        getattr(od_config, "model_class_name", None),
    )


def _math_sdpa_fp16_softmax(query, key, value, attention_mask, causal, scale):
    """Capacity fix for the 21.08 GiB single chip (lead-authorized option a).

    The engine-level ``_math_sdpa`` computes the S x S attention matrix and
    then runs a torch ``softmax`` that materializes an fp32 S x S buffer
    (2.03 GiB at 2048x2048, seq ~4113, 32 heads) - the measured OOM point on
    the FP16 tier (14.17 GiB weights + ~3.4 GiB attention peak > 21.08 GiB).
    This variant keeps the SAME GQA expansion, score matmul, causal and
    additive-mask handling, but computes the softmax IN PLACE in fp16 with a
    max-subtraction for numerical stability, so only the one fp16 S x S
    score buffer is ever live (peak buffer halved ~1.09 GiB).

    Precision note (MANDATORY in the report): the softmax accumulate drops
    from fp32 to fp16 - the attention logits here are bounded (~|z| < 5), so
    fp16 exp is numerically safe and the relative error is ~1e-3.  This only
    changes the attention-layer accumulate precision on 310P; it is scoped to
    this model's process and leaves the SDXL/Cosmos lines untouched.
    """
    # GQA: expand KV heads to match query heads (mirrors _math_sdpa).
    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError(
                "GQA requires query heads to be a multiple of KV heads, "
                f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}."
            )
        repeat_num = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat_num, dim=1)
        value = value.repeat_interleave(repeat_num, dim=1)

    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    attn = torch.matmul(query * scale, key.transpose(-2, -1))
    if causal:
        seq = attn.shape[-1]
        causal_mask = torch.ones(seq, seq, dtype=torch.bool, device=attn.device).tril()
        attn = attn.masked_fill(~causal_mask, float("-inf"))
    if attention_mask is not None:
        attn = attn + attention_mask

    # In-place fp16 softmax with max-subtraction (no extra fp32 S x S buffer).
    row_max = attn.max(dim=-1, keepdim=True).values
    attn.sub_(row_max)
    attn.exp_()
    attn.div_(attn.sum(dim=-1, keepdim=True))

    return torch.matmul(attn, value)


def _install_fp16_softmax_swap() -> None:
    """Swap the SDPA backend's math-attention to the fp16 in-place softmax
    variant.  Scoped to the HiDream process (each model runs in its own
    container), so the SDXL/Cosmos engine-level ``_math_sdpa`` is untouched.
    Idempotent per process."""
    global _FP16_SOFTMAX_SWAPPED
    if _FP16_SOFTMAX_SWAPPED:
        return

    from vllm_omni.diffusion.attention.backends import sdpa as sdpa_backend

    if getattr(sdpa_backend._sdpa_attention, "_310p_hidream_fp16_softmax", False):
        return
    swapped = _math_sdpa_fp16_softmax
    setattr(swapped, "_310p_hidream_fp16_softmax", True)
    sdpa_backend._sdpa_attention = swapped
    _FP16_SOFTMAX_SWAPPED = True
    logger.warning(
        "[310P][hidream-o1] swapped SDPA math-attention to the in-place fp16 "
        "softmax variant to avoid the 2.03 GiB fp32 S x S buffer (lead-"
        "authorized option a; attention accumulate fp32->fp16, ~1e-3)."
    )


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all HiDream-O1-Image 310P invariants. Idempotent per process.

    Contract name required by the 310P diffusion patch registry
    (``_DIFFUSION_310P_PATCH_MODULES`` dispatch in
    ``vllm_omni.platforms.npu._310p.patch``).
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _pin_fp16(od_config)
    _validate_attention_selection(od_config)
    _disable_numerical_cache_backends(od_config)
    _validate_quantization(od_config)
    _validate_single_chip_parallelism(od_config)
    _install_pipeline_patches()
    _install_attention_guard()
    if os.environ.get("VLLM_OMNI_310P_HIDREAM_FP16_SOFTMAX", "0") == "1":
        _install_fp16_softmax_swap()
    _log_placement_plan(od_config)
