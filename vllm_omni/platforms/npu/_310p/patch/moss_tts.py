# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch MOSS-TTS (full family: TTS / TTSD / SoundEffect / VoiceGenerator) for the 310P NPU path.

Ported from the 910 NPU path following the Qwen3-TTS 310P template
(``_310p/patch/qwen3_tts.py``):

* FP16 forced for the model parts that are not vLLM-managed (codec decoder,
  audio heads/embeddings caches, LUTs).
* Random sampling operators are not ACL-graph-capturable on 310P and
  ``torch.multinomial`` adds a CPU-NPU sync: the talker's per-step audio-code
  sampling is switched to ``vllm_ascend.sample.sampler.random_sample``.
* The stage-1 codec's 910 NPUGraph streaming-decode wrapper is disabled:
  ``_streaming_graph_batch_sizes_from_compilation_config`` returns ``[]`` so
  ``_MossCodecStreamSession`` never builds ``NPUGraphStreamingDecoderWrapper``
  and every decode step runs eagerly.
* The codec's two 910 fused ops (``npu_rotary_mul`` RoPE and
  ``npu_fusion_attention``) are re-targeted at 310P semantics through a
  drop-in ``torch_npu`` shim installed on ``audio_tokenizer_v2._get_torch_npu``:
  RoPE -> portable neox half-rotation in fp32 (the decode loop is eager, so
  the graph-only fused aclnn RoPE is not required); attention ->
  ``torch_npu._npu_flash_attention`` (token-major fp16, 16-token alignment,
  additive FRACTAL_NZ mask — the Qwen3-TTS 310P pattern).

The generic model modules stay unchanged; registration swaps the classes only
when the 310P platform applies the patch (see ``_310p/patch/__init__.py``).

Stage-0 talker arches (``model_config.model_arch``):
  * ``MossTTSDelayModel``  — MOSS-TTS 8B (n_vq=32), MOSS-TTSD 8B (n_vq=16),
    MOSS-SoundEffect 8B (n_vq=16), MOSS-VoiceGenerator 1.7B (n_vq=16)
  * ``MossTTSRealtime`` / ``MossTTSLocalModel`` — adjacent variants sharing
    the same codec stage; covered minimally (FP16 recast) so the patch never
    breaks them.

Stage-1 codec arch (shared by every variant): ``MossTTSCodecDecoder``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu
from vllm_ascend.sample.sampler import random_sample
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    aligned_16,
    nd_to_nz_2d,
)

from vllm_omni.model_executor.models.moss_tts import (
    audio_tokenizer_v2,
    modeling_moss_tts_codec,
    modeling_moss_tts_talker,
)

_RUNTIME_DTYPE = torch.float16
_NEG_INF_FP16 = -torch.finfo(torch.float16).max

_PATCHED_TALKER = False
_PATCHED_CODEC = False


# ===================================================================
#  Stage-0 talker patches
# ===================================================================


def _cast_stacked_audio_weights(module, dtype: torch.dtype) -> None:
    """Recast the vectorised audio-head/embedding caches and raw heads.

    The stacked caches are built in ``load_weights`` from module weights and
    are then used directly by the per-step sampling matmul; forcing them to
    the 310P runtime dtype keeps the whole audio path in one dtype.
    """
    with torch.no_grad():
        for name in ("_stacked_audio_head_w", "_stacked_audio_emb_w"):
            tensor = getattr(module, name, None)
            if tensor is not None and tensor.dtype != dtype:
                setattr(module, name, tensor.to(dtype=dtype).contiguous())
        for submodule in (getattr(module, "audio_heads", None), getattr(module, "audio_embeddings", None)):
            if submodule is not None:
                submodule.to(dtype=dtype)


class _MossTTSDelayTalker310P(modeling_moss_tts_talker.MossTTSDelayTalkerForGeneration):
    """Delay-pattern talker specialised for the 310P NPU path.

    Covers MOSS-TTS / MOSS-TTSD / MOSS-SoundEffect / MOSS-VoiceGenerator.
    Differences from the generic talker:

    * audio-code sampling uses ``random_sample`` (310P-safe Gumbel-max over an
      NPU exponential generator; no CPU-NPU sync, ACL-graph friendly) instead
      of ``torch.multinomial``;
    * softmax for the sampling distribution runs in fp32;
    * the stacked head/embedding caches are recast to FP16 after load.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._audio_runtime_dtype = _RUNTIME_DTYPE
        # Talker-MTP graph hooks (present on some sibling arches) must stay
        # eager on 310P: random sampling inside a captured loop is unsupported.
        if hasattr(self, "talker_mtp_graph_safe"):
            self.talker_mtp_graph_safe = False

    @staticmethod
    def _sample_with_top_k(
        logits: torch.Tensor,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        """Top-k sampling without ``torch.multinomial`` (310P-safe)."""
        if temperature > 0:
            logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0 and top_k < logits.shape[-1]:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            kth = top_vals[..., -1:].expand_as(logits)
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if temperature <= 0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        flat = probs.reshape(-1, probs.shape[-1])
        # One shared global generator: parity with the 910 path, which also
        # sampled the audio heads with the device-global RNG (request seeds
        # only drive the vLLM-managed text head).
        sampled = random_sample(flat, {}).reshape(probs.shape[:-1])
        return sampled

    def _sample_audio_codes(
        self,
        last_h: torch.Tensor,  # (1, H)
        state: dict,
    ) -> torch.Tensor:  # (n_vq,)
        """310P audio-code sampling: same delay-pattern masks, 310P-safe RNG."""
        device = last_h.device
        n_vq = self.n_vq
        audio_lengths = int(state.get("audio_lengths", 0))
        delayed_lengths = int(state.get("delayed_lengths", -1))

        idx = torch.arange(n_vq, device=device)
        pre_mask = idx < audio_lengths
        if delayed_lengths == -1:
            post_mask = torch.ones(n_vq, dtype=torch.bool, device=device)
        else:
            post_mask = idx > (delayed_lengths - 1)
        sampling_mask = pre_mask & post_mask

        codes = torch.full((n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
        if not bool(sampling_mask.any()):
            return codes

        audio_top_k = 25
        audio_temp = 1.7

        if self._stacked_audio_head_w is not None:
            # The stacked cache is forced to the 310P runtime dtype; match the
            # backbone hidden dtype so mixed-precision matmul cannot occur when
            # the engine was not launched with ``dtype=float16``.
            last_h = last_h.to(self._stacked_audio_head_w.dtype)
            all_logits = self._stacked_audio_head_w @ last_h.reshape(-1)  # (n_vq, V+1)
            all_logits[:, -1] = float("-inf")  # invalid sentinel
            all_logits[:, self.audio_pad_code] = float("-inf")

            active_idx = sampling_mask.nonzero(as_tuple=True)[0]  # (n_active,)
            active_logits = all_logits[active_idx]  # (n_active, V+1)
            active_logits = active_logits / max(audio_temp, 1e-6)
            if audio_top_k > 0 and audio_top_k < active_logits.shape[-1]:
                top_vals, _ = torch.topk(active_logits, audio_top_k, dim=-1)
                kth = top_vals[:, -1:].expand_as(active_logits)
                active_logits = torch.where(
                    active_logits < kth,
                    torch.full_like(active_logits, float("-inf")),
                    active_logits,
                )
            probs = F.softmax(active_logits, dim=-1, dtype=torch.float32)
            codes[active_idx] = random_sample(probs, {}).long()
            return codes

        # Fallback: per-head loop (used before load_weights() completes).
        for i in range(n_vq):
            if not bool(sampling_mask[i]):
                continue
            head = self.audio_heads[i]
            logits = head(last_h).reshape(-1)  # (V,)
            logits[..., -1] = float("-inf")
            logits[..., self.audio_pad_code] = float("-inf")
            sampled = self._sample_with_top_k(logits, audio_top_k, audio_temp)
            codes[i] = sampled.long()
        return codes

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        _cast_stacked_audio_weights(self, _RUNTIME_DTYPE)
        return loaded


class _MossTTSRealtimeTalker310P(modeling_moss_tts_talker.MossTTSRealtimeTalkerForGeneration):
    """Minimal 310P override for the Realtime variant (adjacent scope).

    Realtime sampling runs through the shared talker-MTP hooks (eager on NPU
    already); only the FP16 recast and graph-safety flag are added.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._audio_runtime_dtype = _RUNTIME_DTYPE
        if hasattr(self, "talker_mtp_graph_safe"):
            self.talker_mtp_graph_safe = False

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        _cast_stacked_audio_weights(self, _RUNTIME_DTYPE)
        return loaded


class _MossTTSLocalTalker310P(modeling_moss_tts_talker.MossTTSLocalTalkerForGeneration):
    """Minimal 310P override for the Local-Transformer variant (adjacent scope).

    The Local talker already forces ``talker_mtp_graph_safe = not is_npu()``;
    on 310P the platform is still ``npu`` so its MTP loop is eager — matching
    the "random sampling loops run eager on 310P" rule.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._audio_runtime_dtype = _RUNTIME_DTYPE
        if hasattr(self, "talker_mtp_graph_safe"):
            self.talker_mtp_graph_safe = False

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        _cast_stacked_audio_weights(self, _RUNTIME_DTYPE)
        return loaded


# ===================================================================
#  Stage-1 codec patches
# ===================================================================
#
# The codec is the vendored MOSS Audio Tokenizer (v1 FP32 / v2 bf16 decoder)
# with a ring-KV streaming state. On 910 the attention layers use
# ``npu_fusion_attention`` + ``npu_rotary_mul`` and the streaming decode loop
# is NPUGraph-captured.  On 310P:
#
#   1. NPUGraph streaming capture is disabled (eager decode);
#   2. the v2 decoder + LUT are recast to FP16 (v1 stays FP32, as elsewhere);
#   3. the two 910 fused ops are re-targeted at 310P ops through the shim
#      below, installed by monkey-patching ``audio_tokenizer_v2._get_torch_npu``.


def _cast_codec_decoder_to_310p_dtype(codec) -> None:
    """Recast the v2 codec decoder and its decode LUT to FP16.

    v1 checkpoints emit FP32 quantizer outputs and keep an FP32 decoder on
    every platform; leave them untouched.
    """
    if codec is None:
        return
    if not isinstance(codec, audio_tokenizer_v2.MossAudioTokenizerModel):
        return
    codec.decoder.to(dtype=_RUNTIME_DTYPE)
    quantizer = getattr(codec, "quantizer", None)
    build_lut = getattr(quantizer, "build_decode_lut", None)
    if callable(build_lut):
        num_quantizers = int(getattr(quantizer, "num_quantizers", 0) or 0)
        if num_quantizers > 0:
            build_lut(num_quantizers, dtype=_RUNTIME_DTYPE)


class _MossTTSCodecDecoder310P(modeling_moss_tts_codec.MossTTSCodecDecoder):
    """MOSS codec stage specialised for the 310P NPU path."""

    def _streaming_graph_batch_sizes_from_compilation_config(self) -> list[int]:
        # 310P: the streaming decode loop is not NPUGraph-captured.  Returning
        # an empty batch list keeps ``_MossCodecStreamSession`` from building
        # the graph wrapper, so ``session.step`` always takes the eager
        # ``decode_streaming_batch`` branch.
        return []

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        _cast_codec_decoder_to_310p_dtype(self._codec)
        return loaded


# ------------------------------------------------------------------
#  310P ``torch_npu`` shim for the codec's fused ops
# ------------------------------------------------------------------
#
# ``audio_tokenizer_v2`` reaches ``torch_npu`` only through the module-level
# helper ``_get_torch_npu()`` (RoPE in ``apply_rope`` and attention in
# ``MossAudioTokenizerMultiheadAttention.forward``).  Installing the shim
# there retargets both 910 ops without touching the shared model code.
#
# Parts that still need real-device confirmation are marked TODO(310P-real):
#   * 16-token alignment semantics for the ring mask's padded tail;
#   * FRACTAL_NZ additive-mask numerics for the per-offset streaming mask;
#   * RoPE perf: portable rotation vs the fused aclnn op (graph-pattern op
#     needs head_dim 64/128; codec head_dim must be confirmed on-device).


class _TorchNpu310PShim:
    """Drop-in replacement for the two ``torch_npu`` ops the codec uses.

    ``apply_rope`` calls ``npu_rotary_mul(q)`` and ``npu_rotary_mul(k)``
    separately (each returns its own rotated tensor), so the shim must be a
    stateless per-tensor rotation — no q/k staging across calls.
    """

    # -- RoPE ---------------------------------------------------------------

    def npu_rotary_mul(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """910: fused ``npu_rotary_mul`` rotation on (B, H, T, D) neox layout.

        310P: the codec decode loop runs EAGER (NPUGraph disabled), so the
        fused ``npu_apply_rotary_pos_emb`` graph pattern is not required here;
        apply the same neox half-rotation portably (fp32 math, downcast).
        ``cos``/``sin`` arrive as (B, 1, T, D) full-length halves-cat, which
        broadcasts directly over the head dimension.

        TODO(310P-real): if per-step profiling shows RoPE launch overhead is
        significant, swap to the fused aclnn op (one call per (q, k) pair,
        4D BSND half-rotation layout) and re-verify numerics.
        """
        x32 = x.float()
        x1, x2 = x32.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        out = x32 * cos.float() + rotated * sin.float()
        return out.to(x.dtype)

    # -- attention ----------------------------------------------------------

    def npu_fusion_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_num: int,
        input_layout: str = "BNSD",
        atten_mask: torch.Tensor | None = None,
        scale: float | None = None,
        keep_prob: float = 1.0,
        pre_tockens: int | None = None,
        next_tockens: int | None = None,
        sparse_mode: int = 0,
    ) -> tuple[torch.Tensor, ...]:
        """910: BNSD fused attention with a bool mask (True = masked).

        310P: token-major fp16 flash attention with an additive FRACTAL_NZ
        mask, mirroring the Qwen3-TTS CodePredictor attention override.
        """
        assert input_layout == "BNSD", f"unexpected layout {input_layout}"
        bsz, num_heads, seq_len, head_dim = q.shape
        kv_heads = k.shape[1]
        device = q.device

        q16 = q.to(_RUNTIME_DTYPE).contiguous()
        k16 = k.to(_RUNTIME_DTYPE).contiguous()
        v16 = v.to(_RUNTIME_DTYPE).contiguous()

        # Token-major token alignment (16-token blocks on 310P).
        real_tokens = int(bsz) * int(seq_len)
        q_f = aligned_16(q16.transpose(1, 2).reshape(real_tokens, num_heads, head_dim))
        k_f = aligned_16(k16.transpose(1, 2).reshape(real_tokens, kv_heads, head_dim))
        v_f = aligned_16(v16.transpose(1, 2).reshape(real_tokens, kv_heads, head_dim))
        aligned_tokens = int(q_f.shape[0])

        seq_lens = torch.full((int(bsz),), int(seq_len), dtype=torch.int32, device="cpu")
        if aligned_tokens > real_tokens:
            seq_lens[-1] += aligned_tokens - real_tokens

        nz_mask = None
        if atten_mask is not None:
            nz_mask = self._bool_mask_to_nz(atten_mask, q_f.shape[0], k_f.shape[0], device)

        out = torch.empty(
            (aligned_tokens, num_heads, head_dim),
            dtype=_RUNTIME_DTYPE,
            device=device,
        )
        torch_npu._npu_flash_attention(
            query=q_f.contiguous(),
            key=k_f.contiguous(),
            value=v_f.contiguous(),
            mask=nz_mask,
            seq_len=seq_lens,
            scale_value=float(scale) if scale is not None else 1.0,
            num_heads=int(num_heads),
            num_kv_heads=int(kv_heads),
            out=out,
        )
        attn_out = out[:real_tokens].reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        return (attn_out.to(q.dtype),)

    @staticmethod
    def _bool_mask_to_nz(
        mask: torch.Tensor,
        q_tokens: int,
        kv_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Bool (True = masked) (B, 1, Tq, Tk) -> additive FRACTAL_NZ fp16.

        ``aligned_16`` pads the query and key/value token dims independently;
        the mask is extended to ``(B, 1, q_tokens, kv_tokens)`` with attend
        (0.0) padding so the padded tail stays numerically neutral.  Padded
        query rows attend to everything and their outputs are cropped
        downstream by the codec's audio-length bookkeeping
        (TODO(310P-real): confirm padding rows are never fed back into the KV
        ring — on 910 the graph wrapper owns that invariant).
        """
        additive = torch.where(mask.to(dtype=torch.bool), _NEG_INF_FP16, 0.0).to(_RUNTIME_DTYPE)
        pad_kv = max(0, int(kv_tokens) - additive.shape[-1])
        if pad_kv:
            additive = F.pad(additive, (0, pad_kv), value=0.0)
        pad_q = max(0, int(q_tokens) - additive.shape[-2])
        if pad_q:
            additive = F.pad(additive, (0, 0, 0, pad_q), value=0.0)
        nz = nd_to_nz_2d(additive)
        return torch_npu.npu_format_cast(nz, ACL_FORMAT_FRACTAL_NZ)


def _install_310p_torch_npu_shim():
    shim = _TorchNpu310PShim()

    def _get_torch_npu():
        return shim

    audio_tokenizer_v2._get_torch_npu = _get_torch_npu  # type: ignore[assignment]
    return shim


# ===================================================================
#  Patch registration
# ===================================================================


def apply_talker_patches() -> None:
    """Install the 310P MOSS-TTS talker patches (all three talker arches)."""
    global _PATCHED_TALKER

    if _PATCHED_TALKER:
        return

    modeling_moss_tts_talker.MossTTSDelayTalkerForGeneration = _MossTTSDelayTalker310P
    modeling_moss_tts_talker.MossTTSRealtimeTalkerForGeneration = _MossTTSRealtimeTalker310P
    modeling_moss_tts_talker.MossTTSLocalTalkerForGeneration = _MossTTSLocalTalker310P

    _PATCHED_TALKER = True


def apply_codec_patches() -> None:
    """Install the 310P MOSS-TTS codec patches (decoder + fused-op shim)."""
    global _PATCHED_CODEC

    if _PATCHED_CODEC:
        return

    modeling_moss_tts_codec.MossTTSCodecDecoder = _MossTTSCodecDecoder310P
    _install_310p_torch_npu_shim()

    _PATCHED_CODEC = True
