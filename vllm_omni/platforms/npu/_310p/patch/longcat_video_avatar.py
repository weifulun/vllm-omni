# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LongCat-Video-Avatar 310P runtime patch (task-34, v0.32.0 line).

Registered in ``vllm_omni.platforms.npu._310p.patch`` (the 310P diffusion
patch registry, COMPARISON-POLICY.md section 5); dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
diffusion pipeline is constructed.

LongCat-Video-Avatar-1.5 facts driving the invariants (stage-gate-0 ledger,
paper-verified against the weights on disk):

- DiT (``base_model``): 15.853B params (shard index total_size
  31,705,833,600 B / 2).  FP16 = 31.7 GB does NOT fit one 21589 MB 310P3
  chip.  TP2 is not an option: the v0.32.0 native MVP transformer
  explicitly raises ``NotImplementedError`` for context parallel
  (Attention/RoPE/multi-speaker-mask paths, "not supported in native MVP")
  and the diffusion engine has no DiT weight sharding (single-worker
  runner; GATE-TP2 multi-rank HCCL replicates weights, it does not shard
  them).  The sanctioned POLICY section-2 fallback is the model's OFFICIAL
  pre-quantized ``base_model_int8`` checkpoint (quantization_config on
  disk: ``int8_per_channel_symmetric``) which the pipeline loads through
  ``create_quantized_avatar_dit`` when ``use_int8=True``.  Resident per
  chip: INT8 15.88 GB + bf16 residuals ~1.2 GB (cast to FP16) + DMD
  distill LoRA 2.52 GB (required for v1.5, dim 128 / alpha 64) + Wan VAE
  0.51 GB ~ 20.1 GB vs ~20.2 GiB torch-free budget.
- Text encoder UMT5-XXL (``LongCat-Video`` base repo, 22.7 GB) and the
  Whisper-large-v3 audio encoder (3.09 GB fp16 on disk, 6.17 GB fp32 as
  loaded) are pinned to CPU: the chip cannot carry any of it, and the
  pipeline already hands their outputs back to the device
  (``.to(device)`` at the call sites), so only the input side is coerced
  (M12-style handoff).
- Attention: at the official 480p x 93f geometry the self-attention is
  37,440 tokens - the math SDPA decomposition materializes the full
  (B,H,S,S) score tensor (32 heads x 37440^2 x 2 B ~ 89 GB: infeasible at
  EVERY geometry this chip can serve).  All three attention entry points
  (self / text-cross / audio-cross, all ending in the engine
  ``VllmAttention`` layer) therefore run through a head-grouped, K-chunked
  online-softmax kernel (fp16 QK matmul, fp32 softmax accumulation,
  fp32 accumulator; bounded score tensor).  Fused SDPA stays rejected
  (research/012 ERR01007) via the platform-selector guard.
- RoPE: the 3D RoPE rotates in fp32 on device (``q.float(), k.float()``
  plus on-device fp32 cos/sin per layer - ~1.2 GB transient at 37,440
  tokens; the research/012 RoPE-frequency CPU pattern).  The freqs table
  is precomputed on CPU (bit-pattern-identical fp32 ops, the upstream table
  is REAL-valued - einsum, no complex op) and the rotation runs in the
  compute dtype using the fp32 cos/sin moved to device and cast.
- Norm/modulation: the model normalizes in fp32 (RMSNorm_FP32 /
  LayerNorm_FP32 / modulate_fp32 / FinalLayer_FP32); at 37,440 tokens each
  fp32 cast is a ~0.6 GB transient re-run every layer x step.  On 310P
  the norm chain runs in the compute dtype (FP16) - the upstream FP32
  casts are a GPU-side stability choice, not a numerical contract.
  Gated by env, loud log.
- Generator: ``torch.Generator(device="npu")`` is probed once in-container;
  when unsupported the seeded generator is CPU-side (seed semantics and
  the seed-exact double-run contract are unchanged - documented).

Kill switch: ``VLLM_OMNI_310P_DIFFUSION_PATCHES=0`` disables the whole
dispatch (registry dispatcher, POLICY section 1 level-2 reference).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Pipeline class name the registry dispatches on (HF model_index
#: ``_class_name`` / pipeline registry alias).
LONGCAT_ARCHS = ("LongCatVideoAvatarPipeline",)

_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

#: Chunked-attention kernel geometry (head-group x K-chunk online softmax).
_ATTENTION_HEAD_GROUP = int(os.environ.get("VLLM_OMNI_310P_LONGCAT_ATTN_HEAD_GROUP", "4"))
_ATTENTION_KV_CHUNK = int(os.environ.get("VLLM_OMNI_310P_LONGCAT_ATTN_KV_CHUNK", "1024"))

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False
_PLACEMENT_PATCHED = False


# ---------------------------------------------------------------------------
# od_config-level invariants (run before pipeline construction)
# ---------------------------------------------------------------------------


def _force_fp16(od_config: Any) -> None:
    """310P is FP16-only; the official examples pass bfloat16."""
    dtype = getattr(od_config, "dtype", None)
    if dtype == torch.float16:
        return
    raise RuntimeError(
        f"310P diffusion requires FP16 (od_config.dtype={dtype!r}); "
        "310P is FP16-only (BF16 rejected by EZ1001; COMPARISON-POLICY.md "
        "section 2). The LongCat examples pass dtype='bfloat16'; pass "
        "dtype='float16' (the bf16 residual params of the INT8 DiT are cast "
        "to FP16 by the 310P placement patch)."
    )


def _quantization_method(quant_config: Any) -> str | None:
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
    """Precision-tier gate: online quantization restricted to INT8 W8A8.

    The model's OFFICIAL ``base_model_int8`` checkpoint (POLICY section-2
    fallback tier, selected via ``additional_config.use_int8=True``) is a
    model-side INT8 weight-only checkpoint - it is not an engine
    ``quantization_config`` and is accepted unconditionally.  Engine-level
    online quantization on 310P may only be INT8 W8A8 (research/009);
    FP8/MXFP* give no hardware benefit.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][longcat] engine-level quantization method=%r accepted "
            "as the POLICY section-2 fallback tier (research/009: INT8 W8A8 "
            "only has a hardware benefit on 310P via npu_quant_matmul). The "
            "LongCat official base_model_int8 checkpoint (per-channel "
            "weight-only INT8, quantization_method=int8_per_channel_symmetric "
            "per the on-disk config) runs regardless via use_int8=True.",
            method,
        )
        return
    raise RuntimeError(
        f"310P LongCat-Video-Avatar rejects online quantization method "
        f"{method!r} (quantization_config={quant_config!r}): INT8 W8A8 "
        "(method='int8') is the only sanctioned engine tier on 310P "
        "(POLICY section 2, research/009); FP8/MXFP* fall back to FP16 "
        "matmul with no gain; ConvRot-INT4 is unvalidated; bitsandbytes is "
        "CUDA-only."
    )


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections.

    The 310P chunked kernel replaces the score-tensor materialization for
    this model's attention entry points; the backend selection itself must
    still resolve to TORCH_SDPA (the math decomposition covers the engine
    fallbacks, and the platform guard keeps per-request selections honest).
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
            "[310P][longcat] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][longcat] disabling numerics-changing cache backend %r on 310P "
        "(same-seed reference comparison per COMPARISON-POLICY.md section 1); "
        "re-enable after 310P validation.",
        backend,
    )
    od_config.cache_backend = "none"


def _force_vae_tiling(od_config: Any) -> None:
    """Engage the diffusers tiling/slicing decode path for the Wan 3D VAE.

    Untiled 3D decode of the 480p geometry (60x104 spatial latent, up to 24
    latent frames) peaks close to the chip budget with the 20.1 GB resident
    DiT+LoRA; tiling is the task-11 SDXL / task-25 Cosmos3 precedent.  The
    runner applies ``vae.use_tiling/use_slicing`` from ``od_config`` after
    pipeline construction.
    """
    forced = False
    for attr in ("vae_use_tiling", "vae_use_slicing"):
        if not getattr(od_config, attr, False):
            setattr(od_config, attr, True)
            forced = True
    if forced:
        logger.warning(
            "[310P][longcat] forcing vae_use_tiling=true vae_use_slicing=true "
            "(untiled 3D VAE decode activation peaks do not fit the 21589 MB "
            "chip budget with the 20.1 GB DiT+LoRA resident)."
        )


def _validate_single_chip_parallelism(od_config: Any) -> None:
    """Single-chip TP1 only: reject every multi-rank parallel degree.

    The native MVP transformer raises NotImplementedError for context
    parallel itself (cp_split_hw > 1), and sequence/CFG parallel need
    multi-rank HCCL whose gate is closed (GATE-TP2-STATUS.md); the
    diffusion engine in v0.32.0 has no DiT weight sharding anyway.
    """
    for attr in ("ulysses_degree", "ring_degree", "allgather_degree"):
        value = getattr(od_config, attr, 1) or 1
        if value != 1:
            raise RuntimeError(
                f"310P single-chip LongCat-Video-Avatar run rejects {attr}={value}: "
                "sequence parallel needs multi-rank HCCL, whose gate is closed "
                "(GATE-TP2-STATUS.md), and the native MVP transformer rejects "
                "context parallel (NotImplementedError). Run TP1."
            )
    if (getattr(od_config, "cfg_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            f"310P single-chip LongCat-Video-Avatar run rejects "
            f"cfg_parallel_size={getattr(od_config, 'cfg_parallel_size', None)}: "
            "CFG parallel needs two worker ranks. Run cfg_parallel_size=1."
        )


def _install_attention_guard() -> None:
    """Wrap the NPU platform backend selector so 310P can never resolve to
    MindIE-SD-only kernels, even for per-role or per-request selections."""
    global _ATTENTION_GUARD_INSTALLED
    if _ATTENTION_GUARD_INSTALLED:
        return

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

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


# ---------------------------------------------------------------------------
# Runtime placements / kernels (installed on the pipeline after construction)
# ---------------------------------------------------------------------------


def _make_cpu_pinned(model: Any) -> Any:
    """One-shot instance overlay: redirect every ``.to(...)`` on *model*
    to CPU (mossimage precedent).  The LongCat pipeline chains
    ``.to(self.device)`` on each component inside ``_load_components`` and
    its overridden ``to()`` re-moves the components - all of it must land
    back on CPU for the CPU-pinned encoders."""
    if getattr(model, "_310p_cpu_pinned", False):
        return model
    base = type(model)

    def _to_cpu(self: Any, *args: Any, **kwargs: Any) -> Any:
        torch.nn.modules.module.Module.to(self, "cpu")
        return self

    overlay = type(f"_CPUPinned{base.__name__}", (base,), {"to": _to_cpu})
    model.__class__ = overlay
    model._310p_cpu_pinned = True
    return model


def _cast_module_to_fp16(module: Any, *, keep_fp32_buffers: tuple[str, ...] = ()) -> None:
    """Cast bf16/fp32 params+buffers to FP16 in place.

    INT8 weight buffers stay int8 (the official INT8 checkpoint); fp32
    scale buffers named in *keep_fp32_buffers* stay fp32
    (QuantizedLinear ``weight_scale`` - tiny, cast per-forward at use).
    """

    def _walk(m: Any) -> None:
        for p in m._parameters.values():
            if p is not None and p.dtype in (torch.bfloat16, torch.float32):
                p.data = p.data.to(torch.float16)
        for bname, b in m._buffers.items():
            if b is None:
                continue
            if bname in keep_fp32_buffers and b.dtype == torch.float32:
                continue
            if b.dtype == torch.int8:
                continue
            if b.dtype in (torch.bfloat16, torch.float32):
                b.data = b.data.to(torch.float16)
        for child in m.children():
            _walk(child)

    _walk(module)


def _chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None,
    head_group: int,
    kv_chunk: int,
) -> torch.Tensor:
    """Head-grouped, K-chunked online-softmax attention.

    Inputs/outputs: (B, H, S, D).  Replaces the materialized (B,H,S,S)
    score tensor of the math SDPA decomposition (~89 GB at the official
    480p x 93f geometry) with a bounded (B, head_group, S, kv_chunk)
    kernel:

    - QK matmul in fp16 (310P hardware path), scores upcast to fp32 for the
      online softmax (running max / running sum) and the fp32 accumulator;
    - attn_mask added per chunk (broadcast over the chunk axis);
    - numerics: identical softmax math to the reference kernel with fp32
      accumulation - the documented 310P deviation from the math
      decomposition (same-hardware double runs stay bit-exact).
    """
    batch, heads, seq, depth = query.shape
    out = torch.empty(batch, heads, seq, depth, dtype=query.dtype, device=query.device)
    scale = scale if scale else 1.0 / float(depth) ** 0.5
    # Normalize the mask to (B, Sq, Sk): the score tensor below is
    # (B, S, hg, C) with S=query length (dim1) and C=key chunk (dim3), so a
    # (B, Sq, Sk) mask unsqueezes at dim2 to (B, Sq, 1, C) and broadcasts
    # straight into the score tensor.  Accepts 2-D (B, Sk) key-only masks and
    # the 4-D (B, 1, Sq, Sk) torch-SDPA broadcast layout.
    msk = None
    if attn_mask is not None:
        msk = attn_mask
        if msk.dim() == 4:
            msk = msk[:, 0]  # (B, Sq, Sk)
        if msk.dim() == 3 and msk.shape[1] == 1:
            msk = msk[:, 0]  # (B, 1, Sk) key-only mask -> (B, Sk)

    for h0 in range(0, heads, head_group):
        h1 = min(h0 + head_group, heads)
        q = query[:, h0:h1].transpose(1, 2)  # (B, S, hg, D)
        m = torch.full((batch, seq, h1 - h0), float("-inf"), dtype=torch.float32, device=query.device)
        l = torch.zeros((batch, seq, h1 - h0), dtype=torch.float32, device=query.device)
        acc = torch.zeros(batch, seq, h1 - h0, depth, dtype=torch.float32, device=query.device)
        for c0 in range(0, seq, kv_chunk):
            c1 = min(c0 + kv_chunk, seq)
            kc = key[:, h0:h1, c0:c1].transpose(1, 2)  # (B, C, hg, D)
            vc = value[:, h0:h1, c0:c1].transpose(1, 2)
            s = torch.einsum("bsgd,bcgd->bsgc", q, kc).float() * scale  # (B, S, hg, C) fp32
            if msk is not None:
                if msk.dim() == 2:
                    s = s + msk[:, c0:c1].float().view(batch, 1, 1, -1)
                else:
                    s = s + msk[:, :, c0:c1].float().unsqueeze(2)
            # Online (chunked) softmax; m/l track the running max/sum over keys.
            m_new = torch.maximum(m, s.amax(dim=-1))  # (B, S, hg)
            p = torch.exp(s - m_new.unsqueeze(-1))  # (B, S, hg, C)
            alpha = torch.exp(m - m_new)  # (B, S, hg)
            acc = acc * alpha.unsqueeze(-1) + torch.einsum("bsgc,bcgd->bsgd", p.to(vc.dtype), vc)
            l = l * alpha + p.sum(dim=-1)  # (B, S, hg)
            m = m_new
        out[:, h0:h1] = (acc / l.clamp_min(1e-30).unsqueeze(-1)).to(query.dtype).transpose(1, 2)
    return out


def _wrap_attention_instances(transformer: Any) -> None:
    """Wrap every VllmAttention ``forward`` inside the transformer with the
    chunked kernel.  Covers self / text-cross / audio-cross attention in one
    pass (all three end in ``self.attn(q, k, v, [metadata])`` with
    (B,H,S,D) layout)."""
    if transformer is None:
        return
    wrapped = 0
    for module in transformer.modules():
        if getattr(module, "_310p_chunked_wrapped", False):
            continue
        # The engine's attention layer is ``vllm_omni.diffusion.attention.layer
        # .Attention`` (instances carry ``softmax_scale``); the LongCat model
        # classes have other names (AvatarSelfAttention & co).  Identify by
        # class name + the engine attribute set rather than the exact import
        # path so a registry rename does not silently skip the patch.
        if module.__class__.__name__ != "Attention" or not hasattr(module, "softmax_scale"):
            continue
        original = module.forward

        def _forward_310p(
            self: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            metadata: Any = None,
            _original=original,
        ) -> torch.Tensor:
            # The LongCat attention sites construct the engine layer with
            # qkv_layout="BSND" (all three roles), so query/key/value arrive
            # and must leave as (B, S, H, D); the kernel works in (B, H, S,
            # D) - transpose at the boundary (views, no copies).
            attn_mask = getattr(metadata, "attn_mask", None) if metadata is not None else None
            mask = None
            if attn_mask is not None:
                mask = attn_mask
                if mask.dim() == 2:
                    mask = mask[:, None, None, :]
            out = _chunked_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                float(getattr(self, "softmax_scale", 0.0) or 0.0),
                mask,
                _ATTENTION_HEAD_GROUP,
                _ATTENTION_KV_CHUNK,
            )
            return out.transpose(1, 2)

        module.forward = _forward_310p.__get__(module, type(module))
        module._310p_chunked_wrapped = True
        wrapped += 1
    if wrapped:
        logger.info(
            "[310P][longcat] chunked head-grouped attention installed on %d attention instances "
            "(head_group=%d kv_chunk=%d; replaces materialized (B,H,S,S) scores).",
            wrapped,
            _ATTENTION_HEAD_GROUP,
            _ATTENTION_KV_CHUNK,
        )


def _patch_rope_cpu_tables() -> None:
    """Precompute the 3D RoPE table on CPU and rotate in the compute dtype
    (research/012 RoPE-frequency CPU pattern).

    Upstream: ``precompute_freqs_cis_3d`` builds the REAL-valued fp32 freqs
    table (einsum over linspace grids; no complex op) on the tensor device
    and ``forward`` does ``q.float(), k.float()`` plus on-device fp32
    ``cos()/sin()`` per layer (~1.2 GB transient at 37,440 tokens).
    Patched: the identical table ops run on CPU (fp32, deterministic), the
    cos/sin are moved to device cast to the compute dtype, and the rotation
    runs in the compute dtype - no fp32 q/k copies, no on-device fp32 trig.
    """
    from vllm_omni.diffusion.models.longcat_video import transformer as longcat_transformer

    cls = longcat_transformer.RotaryPositionalEmbedding
    if getattr(cls, "_310p_rope_patched", False):
        return

    repeat = longcat_transformer.repeat
    broadcast_concat = longcat_transformer.broadcast_concat
    rearrange = longcat_transformer.rearrange
    np = longcat_transformer.np

    def _precompute_freqs_cis_3d_310p(self: Any, grid_size: Any, frame_index: Any = None, num_ref_latents: Any = None) -> Any:
        """CPU mirror of the upstream table (fp32 real, identical ops)."""
        num_frames, height, width = grid_size
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2, device="cpu")[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2, device="cpu")[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2, device="cpu")[: (dim_w // 2)].float() / dim_w))
        if frame_index is not None and num_ref_latents is not None:
            grid_t = torch.cat(
                [
                    torch.tensor([frame_index], dtype=torch.float32, device="cpu"),
                    torch.arange(0, num_frames - num_ref_latents, dtype=torch.float32, device="cpu"),
                ],
                dim=0,
            )
        else:
            grid_t = torch.from_numpy(np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)).float()
        grid_h = torch.from_numpy(np.linspace(0, height, height, endpoint=False, dtype=np.float32)).float()
        grid_w = torch.from_numpy(np.linspace(0, width, width, endpoint=False, dtype=np.float32)).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcast_concat(
            (freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1
        )
        freqs = rearrange(freqs, "T H W D -> (T H W) D")
        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            raise NotImplementedError("LongCat-Video-Avatar context parallel RoPE is not supported in native MVP.")
        return freqs

    def _forward_rope_310p(
        self: Any,
        q: Any,
        k: Any,
        grid_size: Any,
        frame_index: Any = None,
        num_ref_latents: Any = None,
    ) -> Any:
        key_name = (
            tuple(grid_size) if not isinstance(grid_size, tuple) else grid_size,
            frame_index,
            num_ref_latents,
        )
        self.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)
        freqs_cis = self.freqs_dict[key_name]
        # fp32 table on CPU -> cos/sin on CPU (bit-identical trig) -> move
        # and cast to the compute dtype; rotate in the compute dtype.
        cos = freqs_cis.cos().to(q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = freqs_cis.sin().to(q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        q_ = q * cos + longcat_transformer.rotate_half(q) * sin
        k_ = k * cos + longcat_transformer.rotate_half(k) * sin
        return q_, k_

    cls.precompute_freqs_cis_3d = _precompute_freqs_cis_3d_310p
    cls.forward = _forward_rope_310p
    cls._310p_rope_patched = True  # type: ignore[attr-defined]
    logger.info("[310P][longcat] 3D RoPE freqs/cos/sin tables pinned to CPU; rotation in compute dtype.")


def _patch_norm_compute_dtype() -> None:
    """Run the FP32 norm/modulation chain in the compute dtype (FP16) on
    310P.

    Upstream normalizes in fp32 (RMSNorm_FP32 / LayerNorm_FP32 /
    modulate_fp32 / linear_fp32 / silu_linear_fp32) as a GPU-side stability
    choice; every fp32 cast at 37,440 tokens is a multi-GB transient re-run
    every layer x step.  The patch swaps the module-level helpers for
    compute-dtype equivalents (torch's LayerNorm/linear kernels keep fp32
    accumulators internally, so the math is equivalent within fp16
    tolerance).  ``TimestepEmbedder`` keeps its fp32 output contract (tiny
    tensor; downstream ``assert t.dtype == torch.float32``).
    Gated by ``VLLM_OMNI_310P_LONGCAT_FP16_NORMS`` (default on); the kill
    switch restores the upstream FP32 chain for level-2 comparison runs.
    """
    if os.environ.get("VLLM_OMNI_310P_LONGCAT_FP16_NORMS", "1") == "0":
        logger.info("[310P][longcat] FP16 norms disabled by env; upstream FP32 chain kept.")
        return
    from vllm_omni.diffusion.models.longcat_video import transformer as norm_mod

    if not getattr(norm_mod.RMSNorm_FP32, "_310p_fp16_norm", False):

        def _forward_rms_310p(self: Any, x: Any) -> Any:
            normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return normed * self.weight.to(x.dtype)

        norm_mod.RMSNorm_FP32.forward = _forward_rms_310p
        norm_mod.RMSNorm_FP32._310p_fp16_norm = True  # type: ignore[attr-defined]

    if not getattr(norm_mod.LayerNorm_FP32, "_310p_fp16_norm", False):

        def _forward_ln_310p(self: Any, inputs: Any) -> Any:
            out = torch.nn.functional.layer_norm(
                inputs,
                self.normalized_shape,
                None if self.weight is None else self.weight.to(inputs.dtype),
                None if self.bias is None else self.bias.to(inputs.dtype),
                self.eps,
            )
            return out.to(inputs.dtype)

        norm_mod.LayerNorm_FP32.forward = _forward_ln_310p
        norm_mod.LayerNorm_FP32._310p_fp16_norm = True  # type: ignore[attr-defined]

    def _linear_310p(linear: Any, x: Any) -> Any:
        if not hasattr(linear, "weight"):
            return linear(x)
        bias = None if linear.bias is None else linear.bias.to(x.dtype)
        return torch.nn.functional.linear(x, linear.weight.to(x.dtype), bias)

    def _silu_linear_310p(module: Any, x: Any) -> Any:
        return _linear_310p(module[1], torch.nn.functional.silu(x))

    def _modulate_310p(norm_func: Any, x: Any, shift: Any, scale: Any) -> Any:
        # Upstream affine form x * (scale + 1) + shift; the affine now runs
        # in the compute dtype.  Callers pre-shape shift/scale (no unsqueeze
        # here, mirroring the upstream signature).
        dtype = x.dtype
        x_normed = norm_func(x).to(dtype)
        return x_normed * (scale.to(dtype) + 1.0) + shift.to(dtype)

    # Module-global reassignment: every in-module call site (blocks,
    # FinalLayer, TimestepEmbedder) resolves these names dynamically.
    norm_mod.linear_fp32 = _linear_310p
    norm_mod.silu_linear_fp32 = _silu_linear_310p
    norm_mod.modulate_fp32 = _modulate_310p

    te = norm_mod.TimestepEmbedder
    if not getattr(te, "_310p_fp16_norm", False):
        orig_te_forward = te.forward

        def _te_forward_310p(self: Any, t: Any, dtype: Any) -> Any:
            # Tiny-t path: keep the fp32 output contract downstream expects.
            return orig_te_forward(self, t, dtype).to(torch.float32)

        te.forward = _te_forward_310p
        te._310p_fp16_norm = True  # type: ignore[attr-defined]

    logger.info(
        "[310P][longcat] norm/modulate/linear chain runs in compute dtype (FP16) "
        "on 310P (upstream FP32 casts would be multi-GB transients x layer x step); "
        "env VLLM_OMNI_310P_LONGCAT_FP16_NORMS=0 restores the upstream chain."
    )


def _probe_npu_generator() -> bool:
    """Probe torch.Generator(device='npu') support (once per process)."""
    if hasattr(_probe_npu_generator, "_result"):
        return _probe_npu_generator._result  # type: ignore[attr-defined]
    try:
        gen = torch.Generator(device=torch.device("npu", 0))
        gen.manual_seed(42)
        _probe_npu_generator._result = True  # type: ignore[attr-defined]
        logger.info("[310P][longcat] torch.Generator(device='npu') supported; seed path unchanged.")
    except Exception as exc:
        _probe_npu_generator._result = False  # type: ignore[attr-defined]
        logger.warning(
            "[310P][longcat] torch.Generator(device='npu') unsupported (%s); "
            "the pipeline generator falls back to a CPU generator (seed "
            "semantics unchanged, documented in the report).",
            exc,
        )
    return _probe_npu_generator._result  # type: ignore[attr-defined]


def _install_generator_fallback(pipeline_cls: Any) -> None:
    """Route the pipeline's seeded generator to CPU when the NPU generator
    is unsupported.

    The engine path creates ``torch.Generator(device=self.device)`` inside
    ``generate`` when ``req.sampling_params.generator is None``, and the
    official ``end2end.py`` passes an explicitly constructed
    ``torch.Generator(device="npu")``.  Wrapping ``generate`` covers both:
    whenever an NPU generator is present (and the probe says the NPU
    generator is unsupported) it is swapped for a CPU generator carrying
    the same seed (``sampling_params.seed`` or the upstream default 42),
    and the None case gets the same CPU generator.  Seed-exact double-run
    contract is therefore unaffected either way.
    """
    if _probe_npu_generator():
        return
    if getattr(pipeline_cls, "_310p_generator_patched", False):
        return
    original_generate = pipeline_cls.generate

    def _cpu_generator_for(seed: int) -> Any:
        return torch.Generator(device="cpu").manual_seed(seed)

    def _generate_310p(self: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
        sampling_params = getattr(req, "sampling_params", None)
        if sampling_params is None:
            return original_generate(self, req, *args, **kwargs)
        seed = getattr(sampling_params, "seed", None)
        seed = 42 if seed is None else seed
        generator = getattr(sampling_params, "generator", None)
        generator_device = getattr(generator, "device", None)
        device_type = getattr(generator_device, "type", None) or getattr(generator_device, "device_type", None)
        if generator is None or device_type == "npu":
            sampling_params.generator = _cpu_generator_for(int(seed))
        return original_generate(self, req, *args, **kwargs)

    pipeline_cls.generate = _generate_310p
    pipeline_cls._310p_generator_patched = True  # type: ignore[attr-defined]
    logger.info("[310P][longcat] NPU generator fallback armed (CPU generator via generate() wrap).")


def _patch_prompt_embeds_device(pipeline: Any) -> None:
    """Keep UMT5 inputs coherent with the CPU-pinned encoder.

    ``_get_t5_prompt_embeds`` moves ``input_ids``/``attention_mask`` to the
    requested device (NPU) and then calls the CPU-pinned encoder - a
    device split.  Wrap the bound method: run the upstream body on the
    encoder's device, then hand the embeddings back to the requested
    device (the final ``.to(device)`` targets NPU as upstream intended).
    """
    original = pipeline._get_t5_prompt_embeds

    def _get_t5_prompt_embeds_310p(
        self: Any,
        prompt: Any,
        num_videos_per_prompt: int,
        max_sequence_length: int,
        device: Any,
        dtype: Any,
    ) -> tuple[Any, Any]:
        prompt_embeds, mask = original(
            prompt,
            num_videos_per_prompt,
            max_sequence_length,
            torch.device("cpu"),
            dtype,
        )
        return prompt_embeds.to(device=device), mask.to(device=device)

    pipeline._get_t5_prompt_embeds = _get_t5_prompt_embeds_310p.__get__(pipeline, type(pipeline))


def _patch_pipeline_to_override(pipeline: Any) -> None:
    """The pipeline's ``to()`` override re-moves the four components to
    ``self.device``; the CPU-pinned encoders must stay on CPU."""
    original_to = pipeline.to

    def _to_310p(*args: Any, **kwargs: Any) -> Any:
        result = original_to(*args, **kwargs)
        target = result if result is not None else pipeline
        for name in ("text_encoder", "audio_encoder"):
            module = getattr(target, name, None)
            if module is not None:
                torch.nn.modules.module.Module.to(module, "cpu")
        return result

    pipeline.to = _to_310p.__get__(pipeline, type(pipeline))


def _install_placement_patches() -> None:
    """Pin the big encoders to CPU, cast NPU-bound modules to FP16, and
    install the chunked-attention wrapper + generator fallback.

    Installed by wrapping ``LongCatVideoAvatarPipeline.__init__``: the
    upstream ``_load_components`` builds under a CPU build-context by
    default and then moves EVERYTHING to ``self.device``; this wrapper
    re-pins text_encoder/audio_encoder to CPU, casts transformer
    (bf16 residual -> fp16, int8/scales untouched) and VAE (bf16 -> fp16)
    in place, and wraps the attention instances.  Idempotent per process.
    """
    global _PLACEMENT_PATCHED
    from vllm_omni.diffusion.models.longcat_video import pipeline_longcat_video_avatar

    cls = pipeline_longcat_video_avatar.LongCatVideoAvatarPipeline
    if getattr(cls, "_310p_placement_patched", False):
        return
    original_init = cls.__init__

    def _init_then_place(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)

        # 1) CPU pin: UMT5-XXL text encoder (22.7 GB) + Whisper audio encoder
        #    (6.17 GB fp32 as loaded).  The chip budget (20.2 GiB torch-free)
        #    is consumed by the INT8 DiT + LoRA + VAE; the encoders hand their
        #    outputs back to the device at the call sites (.to(device)).
        for name in ("text_encoder", "audio_encoder"):
            module = getattr(self, name, None)
            if module is None:
                continue
            torch.nn.modules.module.Module.to(module, "cpu")
            _make_cpu_pinned(module)
            module.eval()
        logger.info(
            "[310P][longcat] text_encoder (UMT5-XXL, 22.7 GB) and audio_encoder "
            "(Whisper-large-v3) pinned to CPU; .to() redirected."
        )

        # 2) FP16 cast: bf16 residuals of the INT8 DiT, LoRA, and the VAE.
        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            _cast_module_to_fp16(transformer, keep_fp32_buffers=("weight_scale",))
            # LoRA networks live in a plain dict (not registered submodules);
            # enable_loras() already moved them to the model device/dtype
            # (bf16) - cast to FP16 explicitly.
            lora_dict = getattr(transformer, "lora_dict", None) or {}
            for key, network in lora_dict.items():
                _cast_module_to_fp16(network)
                logger.info("[310P][longcat] LoRA network %r cast to FP16.", key)
        vae = getattr(self, "vae", None)
        if vae is not None:
            _cast_module_to_fp16(vae)

        # 3) Chunked attention on every VllmAttention instance (self /
        #    text-cross / audio-cross all end in self.attn(q, k, v)).
        _wrap_attention_instances(transformer)

        # 4) Input-side coercion: keep the tokenizer tensors on the
        #    CPU-pinned encoder (M12-style handoff).
        _patch_prompt_embeds_device(self)

        # 5) Guard the pipeline's own to() override: it re-moves the four
        #    components to self.device; keep the two CPU-pinned ones on CPU.
        _patch_pipeline_to_override(self)

        _log_placement_plan(self)

    cls.__init__ = _init_then_place
    cls._310p_placement_patched = True  # type: ignore[attr-defined]
    _install_generator_fallback(cls)
    _PLACEMENT_PATCHED = True


def _log_placement_plan(pipeline: Any) -> None:
    device = getattr(pipeline, "device", None)
    od_config = getattr(pipeline, "od_config", None)
    logger.info(
        "[310P][longcat] placement plan: dtype=fp16 (forced) attention=TORCH_SDPA "
        "(chunked head-grouped kernel; head_group=%d kv_chunk=%d) vae_tiling=forced "
        "text_encoder=cpu UMT5-XXL audio_encoder=cpu Whisper fp32 "
        "dit=INT8 official checkpoint (base_model_int8, int8_per_channel_symmetric) "
        "lora=fp16 on-device rope=cpu-tables norms=compute-dtype parallel=TP1 "
        "device=%s model_class=%s",
        _ATTENTION_HEAD_GROUP,
        _ATTENTION_KV_CHUNK,
        device,
        getattr(od_config, "model_class_name", None) if od_config is not None else None,
    )


# ---------------------------------------------------------------------------
# Registry entry point
# ---------------------------------------------------------------------------


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all LongCat-Video-Avatar 310P invariants. Idempotent per process.

    Contract name required by the 310P diffusion patch registry
    (``vllm_omni.platforms.npu._310p.patch``).  Model-side runtime
    installations (placement, attention wrapper, RoPE tables, norm chain,
    generator fallback) are deferred to the pipeline-construction hook so
    they survive engine reloads; od_config gates run here, before
    construction.
    """
    global _APPLIED

    _force_fp16(od_config)
    _validate_attention_selection(od_config)
    _disable_numerical_cache_backends(od_config)
    _validate_quantization(od_config)
    _force_vae_tiling(od_config)
    _validate_single_chip_parallelism(od_config)
    _install_attention_guard()

    if _APPLIED:
        return
    _patch_rope_cpu_tables()
    _patch_norm_compute_dtype()
    _install_placement_patches()
    _APPLIED = True
