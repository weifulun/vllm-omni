# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wan-Dancer twin-DiT (task-40 port).

Port of the Wan-Dancer-14B DiT (Wan-AI/Wan-Dancer-14B, global + local twin
checkpoints, Apache-2.0) from the official diffsynth repo (Wan-Video/Wan-Dancer)
to a self-contained torch module that runs on Ascend 310P (FP16-only, no
complex dtype on-device, no ``F.scaled_dot_product_attention``).

Faithful to the official semantics (``diffsynth/models/wan_video_dit.py`` +
``diffsynth/pipelines/wan_video_new.py::model_fn_wan_video`` / ``__call__``):

* dim 5120 / heads 40 / depth 40 / ffn 13824 / patch (1,2,2) / in_dim 36 /
  out_dim 16 — one ``DancerTransformer3DModel`` per checkpoint (the global and
  local checkpoints share an identical key set, 1424 tensors each).
* Twin unimodel heads: the global stage (``int(fps+0.5) != 30``) uses
  ``patch_embedding_global``/``head_global``; the local stage uses
  ``patch_embedding``/``head``.  ``freqs`` end=2794 with dynamic-fps
  resampling of the temporal axis (official ``get_single_freqs``, complex
  space, first/last row pinned).
* I2V conditioning: ``y = cat(msk(4ch), vae_latents(16ch))`` concatenated with
  the noise along the channel axis (in_dim 36); CLIP feature (257 tokens,
  dim 1280) through ``img_emb`` and prepended to the text context; optional
  ``refimage`` (CLIP) through ``img_emb_refimage`` prepended again (it becomes
  the ``img`` branch of the cross-attention).
* Music conditioning: ``music_projection`` Linear(35->256), two LODGE
  ``MusicEncoderLayer`` (norm-first, relu, full-dim rotary), bilinear resize
  of the (n, 256) sequence to (149, 4800) then to
  (latent_f * 8, context_dim=5120); 8 ``Injector`` cross-attention units after
  DiT blocks [0, 4, 8, 12, 16, 20, 24, 27], applied per temporal frame group
  (video tokens rearranged to (b*t, n, c) attending each frame's single music
  token — the official degenerate single-key attention).
* ``enable_skip_layer_nega``: block index 9 is skipped on the negative CFG
  pass only (official ``__call__`` sets the flag per pass).

310P adaptations (documented in the report): RoPE applied with real-valued
cos/sin tables built on CPU from the official complex tables (complex dtype is
forbidden on-device); every attention goes through ``math_sdpa`` (chunked
matmul + fp32 softmax); the block loop supports an explicit per-block device
list for the dual-die (davinci6/7) placement.

``reference_latents`` (``ref_conv`` prepend) is not used by the official
benchmark configuration (``enable_refimagevae=False``); the module is kept for
weight mapping but the prepend path is not implemented (the official freqs
handling for it is ill-defined, see report §known-issues).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.models.wan2_2.wan_dancer_attention import math_sdpa

__all__ = ["DancerTransformer3DModel", "DancerDitConfig", "sinusoidal_embedding_1d"]

MUSIC_INJECT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
SKIP_LAYER_INDEX = 9
MUSIC_RESIZE_ROWS = 149
MUSIC_RESIZE_COLS = 4800


class DancerDitConfig:
    """Static geometry of the Wan-Dancer 14B DiT (both checkpoints)."""

    def __init__(self) -> None:
        self.dim = 5120
        self.ffn_dim = 13824
        self.num_heads = 40
        self.head_dim = self.dim // self.num_heads
        self.depth = 40
        self.in_dim = 36
        self.out_dim = 16
        self.patch_size = (1, 2, 2)
        self.eps = 1e-6
        self.freq_dim = 256
        self.freqs_end = 2794
        self.text_dim = 4096
        self.clip_dim = 1280
        self.music_dim = 35
        self.music_latent = 256
        self.music_heads = 4
        self.music_ffn = 1024


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Official diffsynth ``sinusoidal_embedding_1d`` (fp64 -> input dtype)."""
    assert dim % 2 == 0
    half = dim // 2
    position_dtype = position.dtype
    position = position.float()
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(10000.0, -torch.arange(half, dtype=torch.float64).div(half)).to(position.device),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position_dtype)


def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Official 1-D RoPE table: complex64 (end, dim/2) pairs."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64)[: (dim // 2)] / dim))
    angles = torch.outer(torch.arange(end, dtype=torch.float64), freqs)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def _apply_rope_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Pairwise RoPE in real arithmetic.

    ``x``: (b, s, n*d); ``cos``/``sin``: last two dims (s, d) broadcastable to
    (1, s, 1, d).  Exact real form of the official complex multiplication
    ``view_as_complex(x) * freqs`` with the (d0, d1) pair layout.
    """
    b, s, _ = x.shape
    xf = x.to(torch.float32).view(b, s, num_heads, -1)
    x0 = xf[..., 0::2]
    x1 = xf[..., 1::2]
    # Normalize cos/sin to (1, s, 1, d) regardless of leading batch dims.
    c = cos.to(torch.float32).reshape(s, -1).view(1, s, 1, -1)
    s_ = sin.to(torch.float32).reshape(s, -1).view(1, s, 1, -1)
    c0 = c[..., 0::2]
    c1 = c[..., 1::2]
    s0 = s_[..., 0::2]
    s1 = s_[..., 1::2]
    out0 = x0 * c0 - x1 * s0
    out1 = x0 * s0 + x1 * c1
    return torch.stack([out0, out1], dim=-1).flatten(-2).to(x.dtype)


class DancerSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = _apply_rope_real(q, freqs_cos, freqs_sin, self.num_heads)
        k = _apply_rope_real(k, freqs_cos, freqs_sin, self.num_heads)
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        attn = math_sdpa(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o(attn)


class DancerCrossAttention(nn.Module):
    """Official CrossAttention with optional image branch (has_image_input).

    Context layout: ``y = cat([img(257), ctx])`` when the image branch is on;
    the image branch adds a second attention with ``k_img``/``v_img`` over the
    first 257 tokens.
    """

    def __init__(self, dim: int, num_heads: int, eps: float, has_image_input: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.has_image_input = has_image_input
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, s, _ = q.shape
        sk = k.shape[1]
        n, d = self.num_heads, self.head_dim
        q = q.view(b, s, n, d).transpose(1, 2)
        k = k.view(b, sk, n, d).transpose(1, 2)
        v = v.view(b, sk, n, d).transpose(1, 2)
        return math_sdpa(q, k, v).transpose(1, 2).reshape(b, s, n * d)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            img = None
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        out = self._attn(q, k, v)
        if img is not None:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            out = out + self._attn(q, k_img, v_img)
        return self.o(out)


class DancerDiTBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float):
        super().__init__()
        self.self_attn = DancerSelfAttention(dim, num_heads, eps)
        self.cross_attn = DancerCrossAttention(dim, num_heads, eps, has_image_input=True)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = _modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs_cos, freqs_sin)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class _MLP(nn.Module):
    """Official MLP: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm (no pos emb)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.eps = eps
        p_t, p_h, p_w = patch_size
        self.dim = out_dim * p_t * p_h * p_w
        self.head = nn.Linear(dim, self.dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (b, dim) — the time_embedding output (official semantics).
        (shift, scale) = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        x = _modulate(self.norm(x), shift.squeeze(1), scale.squeeze(1))
        return self.head(x)


class MusicEncoderLayer(nn.Module):
    """LODGE norm-first encoder layer with rotary self-attention (relu FFN)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.register_buffer("_inv_freq", inv_freq, persistent=False)

    def _rotary(self, x: torch.Tensor) -> torch.Tensor:
        # Official LODGE ``rotate_queries_or_keys`` on the full feature dim:
        # angles = arange(s) outer inv_freq, repeat_interleave(2);
        # t = t * cos(angles) + rotate_half(t) * sin(angles).
        b, s, d = x.shape
        angles = torch.outer(torch.arange(s, dtype=torch.float32, device=x.device), self._inv_freq.to(x.device))
        freqs = torch.repeat_interleave(angles, 2, dim=-1)  # (s, d)
        xf = x.to(torch.float32)
        x0 = xf[..., 0::2]
        x1 = xf[..., 1::2]
        c0 = freqs[..., 0::2].cos()
        c1 = freqs[..., 1::2].cos()
        s0 = freqs[..., 0::2].sin()
        s1 = freqs[..., 1::2].sin()
        out0 = x0 * c0 - x1 * s0
        out1 = x0 * s0 + x1 * c1
        return torch.stack([out0, out1], dim=-1).flatten(-2).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Official MusicEncoderLayer (norm_first).

        ``qk = rotary(norm1(x))``; MHA(query=qk, key=qk, value=norm1(x)) —
        the rotated tensor feeds BOTH q and k (through the fused in_proj
        blocks), while v comes from the unrotated normalized input.  2D
        (n, d) input is handled exactly like nn.MultiheadAttention's
        unbatched path (single sequence of length n).
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        h = self.norm1(x)
        qk = self._rotary(h)  # rotated shared q/k source
        q = self.q(qk)
        k = self.k(qk)
        v = self.v(h)  # value = unrotated normalized input
        b, s, _ = q.shape
        n, d = self.nhead, self.head_dim
        attn = math_sdpa(
            q.view(b, s, n, d).transpose(1, 2),
            k.view(b, s, n, d).transpose(1, 2),
            v.view(b, s, n, d).transpose(1, 2),
        )
        attn = attn.transpose(1, 2).reshape(b, s, self.d_model)
        x = x + self.out_proj(attn)
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        if squeeze:
            return x.squeeze(0)
        return x


class DancerInjector(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, inject_layers: tuple[int, ...] = MUSIC_INJECT_LAYERS):
        super().__init__()
        self.injector = nn.ModuleList(
            [DancerCrossAttention(dim, num_heads, eps, has_image_input=False) for _ in inject_layers]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [nn.LayerNorm(dim, eps=eps, elementwise_affine=False) for _ in inject_layers]
        )
        self.injected_block_id: dict[int, int] = {block_id: i for i, block_id in enumerate(inject_layers)}

    def apply(
        self,
        hidden_states: torch.Tensor,
        audio_emb: torch.Tensor,
        block_id: int,
    ) -> torch.Tensor:
        """Official ``after_transformer_block`` rearrange semantics.

        ``hidden_states``: (b, t*n, c); ``audio_emb``: (b, t, c).  Video tokens
        are regrouped per temporal frame ((b*t, n, c)) and each frame attends
        to its single music token ((b*t, 1, c)).
        """
        if block_id not in self.injected_block_id:
            return hidden_states
        attn_id = self.injected_block_id[block_id]
        b, total, c = hidden_states.shape
        t = audio_emb.shape[1]  # official: num_frames = audio_emb.shape[1]
        n = total // t
        # official einops: "b (t n) c -> (b t) n c" (pure reshape, no transpose)
        hs = hidden_states.reshape(b, t, n, c).reshape(b * t, n, c)
        # "b t c -> (b t) 1 c"
        ae = audio_emb.reshape(b * t, 1, c)
        q = self.injector_pre_norm_feat[attn_id](hs)
        out = self.injector[attn_id](q, ae)
        out = out.reshape(b, t, n, c).reshape(b, t * n, c)  # "(b t) n c -> b (t n) c"
        return hidden_states + out


class DancerTransformer3DModel(nn.Module):
    """One Wan-Dancer DiT checkpoint (global or local — identical key set)."""

    def __init__(self, block_devices: torch.device | None = None, cfg: DancerDitConfig | None = None):
        super().__init__()
        self.cfg = cfg or DancerDitConfig()
        c = self.cfg
        self.dim = c.dim
        self.freq_dim = c.freq_dim
        self.patch_size = c.patch_size

        self.patch_embedding = nn.Conv3d(c.in_dim, c.dim, kernel_size=c.patch_size, stride=c.patch_size)
        self.patch_embedding_global = nn.Conv3d(c.in_dim, c.dim, kernel_size=c.patch_size, stride=c.patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(c.text_dim, c.dim),
            nn.GELU(),
            nn.Linear(c.dim, c.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(c.freq_dim, c.dim),
            nn.GELU(),
            nn.Linear(c.dim, c.dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(c.dim, 6 * c.dim))

        self.blocks = nn.ModuleList([DancerDiTBlock(c.dim, c.ffn_dim, c.num_heads, c.eps) for _ in range(c.depth)])
        self.head = _Head(c.dim, c.out_dim, c.patch_size, c.eps)
        self.head_global = _Head(c.dim, c.out_dim, c.patch_size, c.eps)

        self.img_emb = _MLP(c.clip_dim, c.dim)
        self.img_emb_refimage = _MLP(c.clip_dim, c.dim)
        self.ref_conv = nn.Conv2d(16, c.dim, (2, 2))

        self.music_projection = nn.Linear(c.music_dim, c.music_latent)
        self.music_encoder = nn.Sequential(
            MusicEncoderLayer(c.music_latent, c.music_heads, c.music_ffn),
            MusicEncoderLayer(c.music_latent, c.music_heads, c.music_ffn),
        )
        self.music_injector = DancerInjector(c.dim, c.num_heads, c.eps)

        # RoPE tables: CPU fp32 real tables from the official complex tables.
        head_dim = c.head_dim
        t_dim = head_dim - 2 * (head_dim // 3)
        h_dim = w_dim = head_dim // 3
        f_cis = _precompute_freqs_cis(t_dim, c.freqs_end)
        h_cis = _precompute_freqs_cis(h_dim, c.freqs_end)
        w_cis = _precompute_freqs_cis(w_dim, c.freqs_end)
        # Pairwise RoPE: each frequency covers a (2j, 2j+1) dimension pair, so
        # the per-pair cos/sin is duplicated across the pair (repeat_interleave
        # 2) — matches the official complex ``view_as_complex(x) * freqs`` form
        # and vllm-omni's ``freqs.cos().repeat_interleave(2, dim=-1)``.
        self.register_buffer(
            "freqs_cos",
            torch.cat([f_cis.real, h_cis.real, w_cis.real], dim=-1).repeat_interleave(2, dim=-1).float(),
            persistent=False,
        )
        self.register_buffer(
            "freqs_sin",
            torch.cat([f_cis.imag, h_cis.imag, w_cis.imag], dim=-1).repeat_interleave(2, dim=-1).float(),
            persistent=False,
        )
        self._t_dim = t_dim
        self._h_dim = h_dim
        self._w_dim = w_dim

        # Per-block device placement (dual-die).  None -> single device (all
        # blocks stay where the module lives).
        self.block_devices: list[torch.device] | None = None
        if block_devices is not None:
            assert len(block_devices) == c.depth
            self.block_devices = list(block_devices)

    # ------------------------------------------------------------------ #
    # patchify / unpatchify
    # ------------------------------------------------------------------ #
    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        p_t, p_h, p_w = self.patch_size
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        return x, (f // p_t, h // p_h, w // p_w)

    def patchify_global(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        p_t, p_h, p_w = self.patch_size
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = self.patch_embedding_global(x)
        x = x.flatten(2).transpose(1, 2)
        return x, (f // p_t, h // p_h, w // p_w)

    def unpatchify(self, x: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        p_t, p_h, p_w = self.patch_size
        b, s, c = x.shape
        f, h, w = grid
        x = x.view(b, f, h, w, c // (p_t * p_h * p_w), p_t, p_h, p_w)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(b, c // (p_t * p_h * p_w), f * p_t, h * p_h, w * p_w)

    # ------------------------------------------------------------------ #
    # RoPE grid build (CPU tables -> device; complex-space dynamic-fps resample)
    # ------------------------------------------------------------------ #
    def _resample_temporal_freqs(self, f: int, fps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Official ``get_single_freqs`` linear resample of the temporal (f) RoPE
        table -> (f_cos, f_sin) each (f, t_dim), fp32.

        The official code resamples a complex64 table and then takes
        ``.real``/``.imag``; linear interpolation commutes with taking real /
        imaginary parts (and with the pairwise ``repeat_interleave(2)``), so
        resampling the fp32 cos / sin tables directly is exactly equivalent.
        """
        total_frame = int(30.0 / (fps + 1e-6) * f + 0.5)
        total_frame = min(total_frame, self.cfg.freqs_end)
        interval = 30.0 / (fps + 1e-6)
        t_dim = self._t_dim
        cos0 = self.freqs_cos[:total_frame, :t_dim].float()  # (T, t_dim)
        sin0 = self.freqs_sin[:total_frame, :t_dim].float()  # (T, t_dim)
        out_cos = torch.zeros(f, t_dim, device=cos0.device)
        out_sin = torch.zeros(f, t_dim, device=sin0.device)
        out_cos[0] = cos0[0]
        out_sin[0] = sin0[0]
        out_cos[-1] = cos0[total_frame - 1]
        out_sin[-1] = sin0[total_frame - 1]
        for i in range(1, f - 1):
            pos = i * interval
            low = int(pos)
            high = min(low + 1, total_frame - 1)
            wh = pos - low
            wl = 1.0 - wh
            out_cos[i] = cos0[low] * wl + cos0[high] * wh
            out_sin[i] = sin0[low] * wl + sin0[high] * wh
        return out_cos, out_sin

    def build_freqs(self, grid: tuple[int, int, int], fps: float, enable_dynamicfps: bool) -> tuple[torch.Tensor, torch.Tensor]:
        f, h, w = grid
        t_dim, h_dim, w_dim = self._t_dim, self._h_dim, self._w_dim
        if enable_dynamicfps and int(fps + 0.5) != 30:
            f_cos, f_sin = self._resample_temporal_freqs(f, fps)  # (f, t_dim) fp32
        else:
            f_cos = self.freqs_cos[:f, :t_dim]
            f_sin = self.freqs_sin[:f, :t_dim]
        h_cos = self.freqs_cos[:h, t_dim : t_dim + h_dim]
        w_cos = self.freqs_cos[:w, t_dim + h_dim :]
        h_sin = self.freqs_sin[:h, t_dim : t_dim + h_dim]
        w_sin = self.freqs_sin[:w, t_dim + h_dim :]

        f_cos = f_cos.view(f, 1, 1, -1).expand(f, h, w, -1)
        h_cos = h_cos.view(1, h, 1, -1).expand(f, h, w, -1)
        w_cos = w_cos.view(1, 1, w, -1).expand(f, h, w, -1)
        f_sin = f_sin.view(f, 1, 1, -1).expand(f, h, w, -1)
        h_sin = h_sin.view(1, h, 1, -1).expand(f, h, w, -1)
        w_sin = w_sin.view(1, 1, w, -1).expand(f, h, w, -1)
        cos = torch.cat([f_cos, h_cos, w_cos], dim=-1).reshape(1, f * h * w, -1)
        sin = torch.cat([f_sin, h_sin, w_sin], dim=-1).reshape(1, f * h * w, -1)
        return cos, sin

    # ------------------------------------------------------------------ #
    # music conditioning
    # ------------------------------------------------------------------ #
    def encode_music(
        self,
        music_feature: torch.Tensor,
        latent_f: int,
        device: torch.device,
        dtype: torch.dtype,
        resize_global: bool = True,
    ) -> torch.Tensor:
        """Official music-feature encoding (``interp_mode='bilinear'`` path).

        Stage A (gated by ``enable_global or enable_dynamicfps or
        enable_unimodel``): (n, 35) -> projection -> encoder -> (1, 149, 4800).
        Stage B (``enable_music_inject``): bilinear to
        (1, latent_f*8, context_dim=5120).
        """
        music_feature = music_feature.to(device=device, dtype=dtype)
        emb = self.music_projection(music_feature)  # (n, 256)
        emb = self.music_encoder(emb)  # (n, 256)
        if resize_global:
            emb = emb.unsqueeze(0).unsqueeze(0)  # (1, 1, n, 256)
            emb = F.interpolate(emb, size=(MUSIC_RESIZE_ROWS, MUSIC_RESIZE_COLS), mode="bilinear", align_corners=False)
            emb = emb.squeeze(1)  # (1, 149, 4800)
        else:
            emb = emb.unsqueeze(0)  # (1, n, 256)
        target_rows = latent_f * 8
        emb = F.interpolate(
            emb.unsqueeze(1),  # (1, 1, 149, 4800) or (1, 1, n, 256)
            size=(target_rows, self.cfg.dim),
            mode="bilinear",
            align_corners=False,
        )
        return emb.squeeze(1)  # (1, latent_f*8, 5120)

    # ------------------------------------------------------------------ #
    # forward (official model_fn_wan_video semantics, single CFG branch)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        refimage_feature: torch.Tensor | None = None,
        music_feature: torch.Tensor | None = None,
        fps: float = 30.0,
        enable_music_inject: bool = True,
        enable_refimage: bool = False,
        enable_dynamicfps: bool = False,
        do_skip_layer_nega: bool = False,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        is_global = int(fps + 0.5) != 30

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if y is not None:
            x = torch.cat([latents, y], dim=1)  # (1, 36, f, h, w)
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)
        else:
            x = latents

        if enable_refimage and refimage_feature is not None:
            refimage_embedding = self.img_emb_refimage(refimage_feature)
            context = torch.cat([refimage_embedding, context], dim=1)

        if is_global:
            x, grid = self.patchify_global(x)
        else:
            x, grid = self.patchify(x)

        freqs_cos, freqs_sin = self.build_freqs(grid, fps, enable_dynamicfps)

        audio_emb = None
        if enable_music_inject and music_feature is not None:
            # Official Stage-A gate: enable_global or enable_dynamicfps or
            # enable_unimodel — the benchmark runs always have unimodel on, so
            # the fps/dynamicfps flags below are its exact equivalent here.
            audio_emb = self.encode_music(
                music_feature, grid[0], device, latents.dtype,
                resize_global=is_global or enable_dynamicfps,
            )

        head = self.head_global if is_global else self.head
        x_dev0 = self.block_devices[0] if self.block_devices else device
        x = x.to(x_dev0)
        for i, block in enumerate(self.blocks):
            dev = self.block_devices[i] if self.block_devices else device
            if dev is not None:
                x = x.to(dev)
                freqs_cos = freqs_cos.to(dev)
                freqs_sin = freqs_sin.to(dev)
                if audio_emb is not None:
                    audio_emb = audio_emb.to(dev)
                t_mod = t_mod.to(dev)
                context = context.to(dev)
            if do_skip_layer_nega and i == SKIP_LAYER_INDEX:
                continue
            x = block(x, context, t_mod, freqs_cos, freqs_sin)
            if audio_emb is not None:
                x = self.music_injector.apply(x, audio_emb, i)

        x = head(x, t.to(device=x.device, dtype=x.dtype))
        return self.unpatchify(x, grid)
