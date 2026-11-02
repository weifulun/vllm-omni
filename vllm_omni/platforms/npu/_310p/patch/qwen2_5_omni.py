# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen2.5-Omni (thinker / talker / token2wav) for the 310P NPU path.

Qwen2.5-Omni is a three-stage AR model (thinker -> talker -> code2wav), all
stages loading the same entry class with a different ``model_stage``.  The
310P constraints addressed here:

* 310P is FP16-only.  The thinker/talker reference path runs BF16 on
  910/GPU, and the token2wav (DiT + BigVGAN) reference path explicitly runs
  FP32.  Every 310P runtime tensor is pinned to FP16 (``_RUNTIME_DTYPE``):
  cached preprocessing-only embeddings, the speaker-conditioning tensors,
  and the DiT/BigVGAN sub-model after weight load.
* The token2wav DiT attention runs portable block-windowed torch SDPA in the
  reference path.  The 310P override routes short mel sequences to the
  proven 310P flash-attention family (token-major fp16, 16-token alignment,
  FRACTAL_NZ additive mask — same op family as the Qwen3-TTS 310P patch) and
  falls back to the portable SDPA path for long sequences where a dense
  FRACTAL_NZ mask would exceed the chip budget.
* The thinker/talker AR stages run on the generic vllm-ascend 310P runner
  machinery (standard attention backend, MRoPE handled by the shared entry
  forward); no custom kernels are required for them.  The talker stays eager
  per the deploy YAML (``enforce_eager: true`` inherited by the NPU
  override), matching the Qwen3-TTS "keep the codec loop graph-safe"
  precedent.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
import torch_npu
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, aligned_16, nd_to_nz_2d

from vllm_omni.model_executor.models.qwen2_5_omni import (
    qwen2_5_omni,
    qwen2_5_omni_talker,
    qwen2_5_omni_thinker,
    qwen2_5_omni_token2wav,
)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")

# DiT mel sequences longer than this (aligned tokens) keep the portable SDPA
# path: the FRACTAL_NZ mask is dense (1, 1, A, A) fp16 and at A=8192 it
# already costs ~134 MB per cached shape, while the streaming chunk path
# (the production code2wav loop) stays well below it.
_FLASH_MAX_ALIGNED_TOKENS = int(os.environ.get("VLLM_OMNI_310P_QWEN25OMNI_DIT_FLASH_MAX_TOKENS", "8192"))
# Force the portable SDPA path (real-machine verification knob, see RISKS.md
# TODO(310P-real) item on torch SDPA availability on 310P).
_USE_SDPA_FALLBACK = os.environ.get("VLLM_OMNI_310P_QWEN25OMNI_DIT_SDPA", "0") == "1"

_PATCHED_ENTRY = False
_PATCHED_THINKER = False
_PATCHED_TALKER = False
_PATCHED_CODE2WAV = False

_FP16_NEG_MAX = -float(torch.finfo(_RUNTIME_DTYPE).max)


def _recast_tensors_inplace(value, dtype: torch.dtype):
    """Recast every tensor inside ``value`` (nested dict/list/tuple aware)."""
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    if isinstance(value, dict):
        return {k: _recast_tensors_inplace(v, dtype) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_recast_tensors_inplace(v, dtype) for v in value)
    return value


# ===================================================================
#  Entry class (all stages)
# ===================================================================


class _Qwen2_5OmniEntry310P(qwen2_5_omni.Qwen2_5OmniForConditionalGeneration):
    """Entry override pinning the 310P runtime dtype to FP16.

    The shared model code stays portable; this override only (a) recasts the
    cached preprocessing-only embeddings (speaker/codec special tokens) that
    are consumed by the talker prefill path, (b) recasts the token2wav
    sub-model and its speaker-conditioning tensors after weight load, and
    (c) pins the dummy speaker-conditioning fallback to FP16 so the
    no-spk_dict path cannot mix FP32 with the FP16 DiT.
    """

    @property
    def _npu_runtime_dtype(self) -> torch.dtype:
        return _RUNTIME_DTYPE

    def _init_special_tokens_embeddings(self):
        loaded = super()._init_special_tokens_embeddings()
        # Cached per-token embeddings feed the talker prefill on NPU; they
        # must stay in the 310P runtime dtype even though the checkpoint
        # stores them in BF16.
        for name in (
            "embed_text_bos_token",
            "embed_text_eos_token",
            "embed_text_pad_token",
            "embed_codec_bos_token",
            "embed_codec_eos_token",
            "embed_codec_pad_token",
        ):
            tensor = getattr(self, name, None)
            if torch.is_tensor(tensor):
                setattr(self, name, tensor.to(dtype=_RUNTIME_DTYPE))
        embed_text_spk_tokens = getattr(self, "embed_text_spk_tokens", None)
        if isinstance(embed_text_spk_tokens, dict):
            self.embed_text_spk_tokens = _recast_tensors_inplace(embed_text_spk_tokens, _RUNTIME_DTYPE)
        return loaded

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loaded = super().load_weights(weights)
        if self.model_stage == "code2wav" and self.token2wav is not None:
            # The reference path runs the DiT/BigVGAN in FP32 and the
            # checkpoint stores BF16; 310P needs everything in FP16.
            self.token2wav.to(dtype=_RUNTIME_DTYPE)
            spk_dict = getattr(self.token2wav, "spk_dict", None)
            if spk_dict is not None:
                self.token2wav.spk_dict = _recast_tensors_inplace(spk_dict, _RUNTIME_DTYPE)
            for attr in ("_token2wav_conds", "_token2wav_ref_mels"):
                mapping = getattr(self, attr, None)
                if isinstance(mapping, dict):
                    setattr(self, attr, _recast_tensors_inplace(mapping, _RUNTIME_DTYPE))
        return loaded

    def _codec_to_audio(self, codec_tokens, speaker=qwen2_5_omni.DEFAULT_SPEAKER):
        # Pre-seed FP16 dummies so the upstream FP32 fallback is never taken
        # on 310P (the fallback would feed FP32 noise into the FP16 DiT).
        voice = next(
            (key for key in self._token2wav_conds if key.lower() == speaker.lower()),
            speaker,
        )
        if voice not in self._token2wav_conds:
            device = self._module_device(self.token2wav)
            self._token2wav_conds[voice] = torch.zeros(
                (1, self.token2wav_config.dit_config.enc_emb_dim),
                device=device,
                dtype=_RUNTIME_DTYPE,
            )
        if voice not in self._token2wav_ref_mels:
            device = self._module_device(self.token2wav)
            self._token2wav_ref_mels[voice] = torch.zeros(
                (1, 300, self.token2wav_config.dit_config.mel_dim),
                device=device,
                dtype=_RUNTIME_DTYPE,
            )
        return super()._codec_to_audio(codec_tokens, speaker)


# ===================================================================
#  Thinker (stage 0)
# ===================================================================


class _Qwen2_5OmniThinker310P(qwen2_5_omni_thinker.Qwen2_5OmniThinkerForConditionalGeneration):
    """Thinker override: multimodal towers pinned to the 310P runtime dtype.

    The language model stays engine-managed (the engine loads it in the
    deploy dtype, FP16 on 310P).  The audio tower and vision transformer are
    transformers-backed modules; the loader may keep them in the checkpoint
    dtype (BF16), which the 310P path must not consume, so they are recast
    after load.
    """

    @property
    def _npu_runtime_dtype(self) -> torch.dtype:
        return _RUNTIME_DTYPE

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        for tower in (self.audio_tower, self.visual):
            if tower is not None:
                tower.to(dtype=_RUNTIME_DTYPE)
        return loaded


# ===================================================================
#  Talker (stage 1)
# ===================================================================


class _Qwen2_5OmniTalker310P(qwen2_5_omni_talker.Qwen2_5OmniTalkerForConditionalGeneration):
    """Talker override for the 310P NPU path.

    The talker AR loop stays eager (deploy YAML: stage 1 inherits
    ``enforce_eager: true`` and the NPU override does not change it), which
    mirrors the Qwen3-TTS 310P precedent of keeping codec-token sampling off
    ACL graph capture.  Only the FP16 pinning of the projection/embedding
    path is 310P-specific.
    """

    @property
    def _npu_runtime_dtype(self) -> torch.dtype:
        return _RUNTIME_DTYPE

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loaded = super().load_weights(weights)
        self.thinker_to_talker_proj.to(dtype=_RUNTIME_DTYPE)
        return loaded


# ===================================================================
#  Token2Wav DiT attention (stage 2)
# ===================================================================


def _block_window_additive_mask(
    batch_size: int,
    num_blocks: int,
    block_size: int,
    seq_len: int,
    aligned_tokens: int,
    look_backward_block: int,
    look_ahead_block: int,
    device: torch.device,
) -> torch.Tensor:
    """Dense additive mask (1, 1, A, A) for the 310P flash-attention path.

    The token stream is ``batch_size`` rows of ``num_blocks * block_size``
    block-padded tokens plus the 16-alignment pad appended after the last
    row.  Key validity follows the upstream block windowing: the last
    ``stream_tokens - seq_len`` tokens of every row (block padding) and the
    trailing alignment pad are masked.  Alignment-pad query rows clamp to
    the last block's window so no row is fully masked.
    """
    stream_tokens = num_blocks * block_size
    real_tokens = stream_tokens * batch_size
    pad = stream_tokens - seq_len  # block-padded tokens per row (0..block_size-1)

    key_valid = torch.ones(aligned_tokens, dtype=torch.bool, device=device)
    if pad > 0:
        for r in range(batch_size):
            start = r * stream_tokens + (stream_tokens - pad)
            key_valid[start:start + pad] = False
    if aligned_tokens > real_tokens:
        key_valid[real_tokens:] = False

    # Row-local block indices: the kernel isolates rows via seq_len and only
    # reads the mask's per-row diagonal slice, so the window must be expressed
    # in each row's own block numbering.  Alignment-pad tokens clamp to the
    # last block (they stay masked as keys, and as queries they keep a valid
    # own-window).
    tokens = torch.arange(aligned_tokens, device=device)
    row = torch.clamp(tokens // stream_tokens, max=batch_size - 1)
    pos_in_row = tokens - row * stream_tokens
    local_block = torch.clamp(pos_in_row // block_size, max=num_blocks - 1)

    lo = (local_block[:, None] - look_backward_block).clamp(min=0)
    hi = (local_block[:, None] + look_ahead_block).clamp(max=num_blocks - 1)
    window_ok = (local_block[None, :] >= lo) & (local_block[None, :] <= hi)
    mask = torch.where(
        window_ok & key_valid[None, :],
        torch.zeros((), dtype=_RUNTIME_DTYPE, device=device),
        torch.full((), _FP16_NEG_MAX, dtype=_RUNTIME_DTYPE, device=device),
    )
    return mask.view(1, 1, aligned_tokens, aligned_tokens)


def _dit_block_sparse_attention_310p(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    look_backward_block: int,
    look_ahead_block: int,
    mask_cache: dict,
) -> torch.Tensor:
    """Block-windowed attention with the 310P flash kernel (or SDPA fallback).

    ``query/key/value`` are ``(batch, heads, seq, dim)`` fp16.  Mirrors the
    semantics of :func:`qwen2_5_omni_token2wav.block_sparse_attention`; the
    only differences are the 310P flash kernel, 16-token alignment, and the
    FRACTAL_NZ additive mask.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_blocks = -(-seq_len // block_size)
    pad = num_blocks * block_size - seq_len
    stream_tokens = num_blocks * block_size

    # Long mel sequences keep the portable SDPA path (dense FRACTAL_NZ mask
    # would exceed the chip budget).
    if _USE_SDPA_FALLBACK:
        return qwen2_5_omni_token2wav.block_sparse_attention(
            query,
            key,
            value,
            block_size=block_size,
            look_backward_block=look_backward_block,
            look_ahead_block=look_ahead_block,
        )

    aligned_tokens = (stream_tokens * batch_size + 15) // 16 * 16
    if aligned_tokens > _FLASH_MAX_ALIGNED_TOKENS:
        return qwen2_5_omni_token2wav.block_sparse_attention(
            query,
            key,
            value,
            block_size=block_size,
            look_backward_block=look_backward_block,
            look_ahead_block=look_ahead_block,
        )

    # Block-pad the token stream (zeros) then 16-align it, token-major.
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
    real_tokens = batch_size * stream_tokens
    q_f = aligned_16(query.transpose(1, 2).reshape(real_tokens, num_heads, head_dim))
    k_f = aligned_16(key.transpose(1, 2).reshape(real_tokens, num_heads, head_dim))
    v_f = aligned_16(value.transpose(1, 2).reshape(real_tokens, num_heads, head_dim))
    aligned_tokens = int(q_f.shape[0])

    cache_key = (aligned_tokens, block_size, look_backward_block, look_ahead_block, query.device)
    mask = mask_cache.get(cache_key)
    if mask is None:
        additive = _block_window_additive_mask(
            batch_size,
            num_blocks,
            block_size,
            seq_len,
            aligned_tokens,
            look_backward_block,
            look_ahead_block,
            query.device,
        )
        mask = torch_npu.npu_format_cast(nd_to_nz_2d(additive), ACL_FORMAT_FRACTAL_NZ)
        mask_cache[cache_key] = mask

    seq_lens = torch.full((batch_size,), stream_tokens, dtype=torch.int32, device=_CPU_DEVICE)
    if aligned_tokens > real_tokens:
        seq_lens[-1] += aligned_tokens - real_tokens

    out = torch.empty(
        (aligned_tokens, num_heads, head_dim),
        dtype=_RUNTIME_DTYPE,
        device=query.device,
    )
    torch_npu._npu_flash_attention(
        query=q_f.contiguous(),
        key=k_f.contiguous(),
        value=v_f.contiguous(),
        mask=mask,
        seq_len=seq_lens,
        scale_value=float(head_dim) ** -0.5,
        num_heads=int(num_heads),
        num_kv_heads=int(num_heads),
        out=out,
    )
    return (
        out[:real_tokens].reshape(batch_size, stream_tokens, num_heads, head_dim).transpose(1, 2)[:, :, :seq_len]
    )


class _DiTAttention310P(qwen2_5_omni_token2wav.DiTAttention):
    """DiT attention override for the 310P NPU path.

    The prologue (packed QKV, head reshape, first-head RoPE) keeps the
    portable reference implementation: the DiT loop is eager on 310P (no
    graph capture) and the interleaved-pair RoPE convention of this model
    matches no 310P fused rotary op, so the portable rotation is retained
    (TODO(310P-real): benchmark the rotation cost; fall back to a fused op
    only if it shows up in the profile).  The block-windowed attention is
    routed through :func:`_dit_block_sparse_attention_310p`.
    """

    def __init__(self, config: object, prefix: str = "") -> None:
        super().__init__(config, prefix=prefix)
        self._mask_cache_310p: dict = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_size: int | None = None,
        look_backward_block: int = 0,
        look_ahead_block: int = 0,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.inner_dim, self.inner_dim, self.inner_dim], dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        # Due to the training process, only the first head is rotated.
        query[:, :1], key[:, :1] = qwen2_5_omni_token2wav.apply_rotary_pos_emb(
            query[:, :1], key[:, :1], cos, sin
        )

        attention_weights = _dit_block_sparse_attention_310p(
            query,
            key,
            value,
            block_size=block_size,
            look_backward_block=look_backward_block,
            look_ahead_block=look_ahead_block,
            mask_cache=self._mask_cache_310p,
        )
        attention_weights = attention_weights.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        attention_weights = attention_weights.to(query.dtype)

        attention_output = self.to_out[0](attention_weights)
        attention_output = self.to_out[1](attention_output)
        return attention_output


# ===================================================================
#  Patch registration
# ===================================================================


def apply_entry_patches() -> None:
    """Install the 310P entry-class override (shared by all three stages)."""
    global _PATCHED_ENTRY

    if _PATCHED_ENTRY:
        return

    qwen2_5_omni.Qwen2_5OmniForConditionalGeneration = _Qwen2_5OmniEntry310P
    _PATCHED_ENTRY = True


def apply_thinker_patches() -> None:
    """Install the 310P thinker patches (entry + thinker sub-model)."""
    global _PATCHED_THINKER

    if _PATCHED_THINKER:
        return

    apply_entry_patches()
    qwen2_5_omni_thinker.Qwen2_5OmniThinkerForConditionalGeneration = _Qwen2_5OmniThinker310P

    _PATCHED_THINKER = True


def apply_talker_patches() -> None:
    """Install the 310P talker patches (entry + talker sub-model)."""
    global _PATCHED_TALKER

    if _PATCHED_TALKER:
        return

    apply_entry_patches()
    qwen2_5_omni_talker.Qwen2_5OmniTalkerForConditionalGeneration = _Qwen2_5OmniTalker310P

    _PATCHED_TALKER = True


def apply_code2wav_patches() -> None:
    """Install the 310P code2wav patches (entry + DiT attention)."""
    global _PATCHED_CODE2WAV

    if _PATCHED_CODE2WAV:
        return

    apply_entry_patches()
    qwen2_5_omni_token2wav.DiTAttention = _DiTAttention310P

    _PATCHED_CODE2WAV = True
