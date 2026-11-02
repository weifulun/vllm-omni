# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch MiniMax Music 3 for the 310P NPU path (task-32).

MiniMax Music 3 is a two-stage line: an AR talker (Qwen3-8B backbone plus an
RVQ depth decoder, stage 0) and an acoustic flow-matching DiT plus vocoder
(stage 1). The upstream code targets CUDA and hits five 310P walls, each
verified or inferred from the fleet's operator-gap catalog (012):

* The deterministic seeded sampler (MurmurHash3 int64 + float64 double-log
  Gumbel) needs FP64, which 310P AICore does not implement. The sampler runs
  on the narrow ``[rows, vocab]`` logits, so moving those to the host costs a
  couple of megabytes per step and keeps the upstream arithmetic bit-exact.
* ``DepthDecoderBlock`` selects ``sdpa_kernel([FLASH_ATTENTION, MATH])``; on
  torch_npu every selection lands on the fused ``npuFusionAttention`` kernel,
  which 310P does not expose (ERR01007, task-16 machine-verified). The
  sequence is at most eight positions, so plain matmul attention is
  indistinguishable in cost.
* The acoustic stage loads its DiT and vocoder hard-coded in float32; 310P is
  an FP16/INT8 device and the POLICY ladder prefers an FP16 pin over keeping
  the checkpoint precision the hardware cannot execute at full speed.
* ``aligned_condition`` resamples the conditioning with
  ``F.interpolate(mode="nearest")``, which dispatches to the ResizeD family
  whose dynamic-shape JSON is missing on 310P (EZ1013, 012). The nearest
  mapping is replaced by a host-side index gather with torch's exact formula.
* The streaming resampler runs ``torchaudio.functional.resample`` on the
  device; torchaudio-on-NPU has no fleet precedent. The polyphase resample
  moves to the host, where it is bit-identical to the whole-stream resample
  property the class documents.

The DiT's initial-noise draw uses a device ``torch.Generator``; if the NPU
randn-with-generator path faults (no fleet precedent), the draw latches to
the host with the same seed and the same ``philox`` value stream.
* The 310P embedding method keeps a second, fractal-NZ copy of every
  embedding weight (``layer.weight_nz``) while the Linear path converts in
  place. The talker's untied embed_tokens and lm_head (vocab 200000)
  would therefore pin ~3.3 GB of permanent duplicates that the 21 GB chip
  cannot absorb on top of the 17.17 GB of weights, so the seam drops the
  NZ copy and runs ``F.linear`` on the plain fp16 weight.
* The FP16 weights alone (17.17 GB) plus the ~1.3 GB runtime baseline
  leave ~2.6 GB on the 21 GB chip, which the prefill profiling run
  consumes before any KV block exists ("No available memory for the cache
  blocks"). The untied embed_tokens and lm_head (2 x 1.64 GB) are
  therefore offloaded to the host after loading: the lm_head logits
  already live on the host for the sampler seams, and the embedding moves
  are batch x 4096 fp16 in each direction.

Everything here is installed only when the platform already confirmed
``is_310p()`` and the model architecture is one of the MiniMax Music 3
stages, and the whole surface can be disabled with
``VLLM_OMNI_310P_MINIMAX_MUSIC3_PATCH=0`` for the POLICY kill-switch run.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.minimax_music3 import (
    acoustic as _acoustic_module,
    dav as _dav_module,
    dit as _dit_module,
    rvq_decoder as _rvq_decoder_module,
    sampling as _sampling_module,
    talker as _talker_module,
)
from vllm_omni.model_executor.models.minimax_music3.chunking import (
    crop_sample_bounds,
    overlap_mel_length,
)
from vllm_omni.model_executor.models.minimax_music3.constants import (
    DIT_LATENT_CHANNELS,
    DEFAULT_DIT_CFG_SCALE,
    DEFAULT_DIT_STEPS,
)
from vllm_omni.model_executor.models.minimax_music3.dav import remove_weight_norm
from vllm_omni.model_executor.models.minimax_music3.dit import remap_transformer_state
from vllm_omni.model_executor.models.minimax_music3.dit import _SIGMA_FLOOR
from vllm_omni.model_executor.models.minimax_music3.acoustic import _derive_seed
from vllm_omni.model_executor.models.minimax_music3.weights import (
    CONDITION_ENCODER_DIR,
    TRANSFORMER_DIR,
    VOCODER_DIR,
    load_component_state,
    resolve_repo_root,
)

logger = init_logger(__name__)

_KILL_SWITCH_ENV = "VLLM_OMNI_310P_MINIMAX_MUSIC3_PATCH"
_RUNTIME_DTYPE = torch.float16

_TALKER_PATCHED = False
_ACOUSTIC_PATCHED = False
_NOISE_FALLBACK = False


def _kill_switch_off() -> bool:
    return os.environ.get(_KILL_SWITCH_ENV, "1") == "0"


# ===================================================================
#  Stage 0 (AR talker): sampler CPU seams + depth-decoder attention
# ===================================================================


def _cpu_faithful(orig):
    """Wrap a sampler primitive so NPU inputs run on the host.

    310P AICore has no FP64 and the sampler's deterministic semantics are
    built on float64 double-log Gumbel noise plus int64 MurmurHash3. The
    primitives only ever see the narrow ``[rows, vocab]`` logits and small
    index tensors, so staging them through the host is a few megabytes at
    most and preserves the upstream arithmetic exactly: the draw is
    deterministic across processes on 310P and equal to what the CUDA path
    computes for the same seed.

    Non-tensor arguments (e.g. the ``scale`` and ``top_k`` keyword scalars)
    pass through untouched; the wrapped original runs on the host and the
    result returns to the inputs' device.
    """

    def wrapper(*args, **kwargs):
        device = None
        cpu_args = []
        for arg in args:
            if isinstance(arg, Tensor) and arg.device.type == "npu":
                if device is None:
                    device = arg.device
                cpu_args.append(arg.cpu())
            else:
                cpu_args.append(arg)
        out = orig(*cpu_args, **kwargs)
        if isinstance(out, Tensor) and device is not None:
            out = out.to(device)
        return out

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    wrapper.__doc__ = f"310P host-side seam around {getattr(orig, '__name__', orig)!r}."
    return wrapper


def _depth_decoder_block_forward_310p(self: _rvq_decoder_module.DepthDecoderBlock, x: Tensor) -> Tensor:
    """The upstream block forward with the fused-SDPA call replaced.

    torch_npu routes every ``scaled_dot_product_attention`` selection to the
    fused ``npuFusionAttention`` kernel, which 310P does not expose, so the
    causal attention is written out as ``softmax(q k^T / sqrt(d)) v``. The
    sequence is at most eight positions, making the extra ``[B, H, S, S]``
    tensor trivially small.
    """
    h = self.input_layernorm(x)
    bsz, seq, hidden = h.shape
    q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
    q = q.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
    causal = torch.tril(torch.ones((seq, seq), dtype=torch.bool, device=q.device))
    attn = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    h = torch.matmul(attn, v)
    h = h.transpose(1, 2).contiguous().reshape(bsz, seq, hidden)
    x = x + self.o_proj(h)
    h = self.post_attention_layernorm(x)
    h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
    return x + h


def _embed_process_weights_noop_310p(self, layer: nn.Module) -> None:
    """Skip the fractal-NZ copy; the plain fp16 weight stays loaded."""
    return None


def _embed_apply_plain_310p(self, layer: nn.Module, x: Tensor, bias: Tensor | None = None) -> Tensor:
    """F.linear on the plain fp16 weight, tracking the weight's device.

    After the host offload the weight lives on CPU and the hidden states
    arrive on the NPU; the logits then stay on the host, which is where the
    310P sampler seams already run.
    """
    return F.linear(x.to(layer.weight.device), layer.weight, bias)


def _embed_embedding_310p(self, layer: nn.Module, input_: Tensor) -> Tensor:
    """Embedding lookup that follows the weight's device and returns to
    the input's device (the backbone layers stay on the NPU)."""
    weight = layer.weight
    out = F.embedding(input_.to(weight.device), weight)
    return out.to(input_.device)


_TALKER_ARCH_NAME = "MiniMaxMusic3TalkerForConditionalGeneration"


def _offload_talker_embeddings_310p(worker) -> None:
    """Move the untied embed_tokens and lm_head to the host after loading.

    They are 2 x 1.64 GB of fp16 (vocab 200000 x 4096, untied); keeping
    them on the NPU would leave no room for the KV pool on the 21 GB chip.
    Per-step traffic is the lm_head logits (already host-side for the
    sampler seams) plus a batch x 4096 embedding move both ways.
    """
    if _kill_switch_off():
        return
    model = getattr(worker, "model", None)
    if model is None:
        runner = getattr(worker, "model_runner", None)
        model = getattr(runner, "model", None) if runner is not None else None
    if model is None or type(model).__name__ != _TALKER_ARCH_NAME:
        return
    moved = []
    targets = (
        ("embed_tokens", getattr(getattr(model, "model", None), "embed_tokens", None)),
        ("lm_head", getattr(model, "lm_head", None)),
    )
    for name, module in targets:
        if module is None:
            continue
        param = next(module.parameters(), None)
        if param is None or param.device.type == "cpu":
            continue
        module.to("cpu")
        moved.append(f"{name} {tuple(param.shape)} fp16")
    # The c0 logit-index buffer is consumed inside sample() against the
    # (now host-side) logits; the other index buffers stay NPU-side with the
    # forward path.
    c0_ids = getattr(model, "c0_logit_ids", None)
    if c0_ids is not None and c0_ids.device.type != "cpu":
        model.c0_logit_ids = c0_ids.cpu()
        moved.append("c0_logit_ids int64")
    if moved:
        logger.info(
            "MiniMax Music 3 310P: talker embeddings offloaded to host: %s",
            ", ".join(moved),
        )


def _wrap_worker_profile_310p() -> None:
    """Run the talker embedding offload right before the 310P memory profile.

    ``NPUWorker310.determine_available_memory`` runs the dummy prefill and
    sizes the KV pool from whatever is resident at that moment; the model
    weights are already on the NPU and the profile call is a separate method
    dispatch, so a class-level wrap installed here takes effect in the same
    process. Other architectures pass through the hook untouched.
    """
    try:
        from vllm_ascend._310p import worker_310p as _w310
    except ImportError:
        return  # no 310P worker in this environment (host contract tests)
    worker_cls = getattr(_w310, "NPUWorker310", None)
    if worker_cls is None or getattr(worker_cls, "_MM3_PROFILE_WRAPPED", False):
        return
    original = worker_cls.determine_available_memory

    def determine_available_memory_310p(self) -> int:
        _offload_talker_embeddings_310p(self)
        return original(self)

    worker_cls.determine_available_memory = determine_available_memory_310p
    worker_cls._MM3_PROFILE_WRAPPED = True


def apply_talker_patches() -> None:
    """Install the MiniMax Music 3 stage-0 310P patches.

    Swaps the five sampler primitives in the talker namespace for host-side
    seams and the depth-decoder block forward for its decomposed-attention
    twin. The generic model modules stay unchanged, and the ``DepthGraphCache``
    needs no patch: it already refuses capture unless its inputs are CUDA
    tensors, so on NPU it stays on the eager loop.

    Idempotent. No-ops with a warning when the kill switch is set.
    """
    global _TALKER_PATCHED

    if _TALKER_PATCHED:
        return
    if _kill_switch_off():
        logger.warning(
            "MiniMax Music 3 310P talker patches disabled by %s=0; "
            "the stock CUDA-oriented sampler and fused-SDPA paths will run",
            _KILL_SWITCH_ENV,
        )
        return

    for name in ("apply_c0_cfg", "apply_depth_cfg", "draw_from_gumbel", "gumbel_noise", "sample_topk_seeded"):
        original = getattr(_talker_module, name)
        setattr(_talker_module, name, _cpu_faithful(getattr(_sampling_module, name)))
    _rvq_decoder_module.DepthDecoderBlock.forward = _depth_decoder_block_forward_310p

    # The 310P embedding method keeps BOTH the fp16 weight and a fractal-NZ
    # copy (``layer.weight_nz = maybe_trans_nz(layer.weight)``), unlike the
    # Linear path whose NZ conversion is in-place. On this 21 GB chip the
    # talker's 17.17 GB of weights plus the two permanent ~1.64 GB NZ copies
    # of embed_tokens and lm_head (untied, vocab 200000) leave no room for
    # the KV pool, so the embedding/LM-head matmuls run on the plain fp16
    # weight via the generic NPU kernel: numerically identical, a few
    # gigaflops per step at most.
    try:
        from vllm_ascend._310p.ops import vocab_parallel_embedding as _vpe310
    except ImportError:
        _vpe310 = None
    if _vpe310 is not None:
        _method_cls = _vpe310.AscendUnquantizedEmbeddingMethod310
        _method_cls.process_weights_after_loading = _embed_process_weights_noop_310p
        _method_cls.apply = _embed_apply_plain_310p
        _method_cls.embedding = _embed_embedding_310p
        logger.info(
            "MiniMax Music 3 310P: embedding/LM-head fractal-NZ copies disabled "
            "(plain-fp16 F.linear seam); linear NZ conversion left in place"
        )
    _wrap_worker_profile_310p()

    _TALKER_PATCHED = True
    logger.info(
        "MiniMax Music 3 310P talker patches installed: sampler seams on %s, "
        "depth-decoder attention decomposed, embedding NZ seam",
        ", ".join(("apply_c0_cfg", "apply_depth_cfg", "draw_from_gumbel", "gumbel_noise", "sample_topk_seeded")),
    )


# ===================================================================
#  Stage 1 (acoustic): FP16 pin + attention/RoPE/resize/noise seams
# ===================================================================


def _attend_310p(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
    """Plain matmul attention: 310P exposes no fused SDPA kernel at all."""
    seq_k = k.shape[-2]
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        causal = torch.tril(torch.ones((q.shape[-2], seq_k), dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.matmul(scores.softmax(dim=-1), v)


def _rope_310p(self, seq_len: int, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
    """Build the RoPE tables on the host, then move them once.

    The upstream version derives the table with a device-side
    ``arange/einsum`` chain over the ``inv_freq`` buffer (the 012 RealDiv
    family). The tables are small and cached per shape, so computing them
    in float32 on the host and transferring is the safe form; the
    upstream cache key and entry layout are kept unchanged.
    """
    key = (seq_len, dtype, device)
    cached = self._rope_cache.get(key)
    if cached is None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.detach().cpu())
        # Upstream casts to the runtime dtype *before* the trig, so the cast
        # is replicated here and cos/sin run in that dtype (on the host).
        freqs = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype)
        cached = (
            freqs.cos().to(device=device).unsqueeze(1),
            freqs.sin().to(device=device).unsqueeze(1),
        )
        self._rope_cache[key] = cached
    return cached


def _nearest_gather_310p(align: Tensor, mel_len: int) -> Tensor:
    """Replicate torch 1-D ``mode="nearest"`` with an exact index formula.

    ``F.interpolate`` 1-D nearest with ``align_corners=False`` maps output
    index ``i`` to ``floor(i * in_len / out_len)`` (machine-verified against
    the CPU kernel on a range of ratios). Computing that mapping on the host
    as the integer division ``i * in_len // out_len`` plus one gather keeps
    310P off the ResizeD dynamic-shape JSON gap (EZ1013, 012) while staying
    bit-identical to the upstream op.
    """
    in_len = int(align.shape[-1])
    if mel_len <= 0:
        raise ValueError(f"aligned condition length must be positive, got {mel_len}")
    if in_len <= 0:
        raise ValueError("aligned condition has an empty time axis")
    index = (torch.arange(mel_len, dtype=torch.int64) * in_len) // mel_len
    return align.index_select(-1, index.clamp(max=in_len - 1))


def _draw_noise_310p(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    seed: int,
) -> Tensor:
    """Draw the initial noise, latching to the host if the NPU draw faults.

    The NPU randn-with-generator path has no fleet precedent on 310P. The
    first failure latches ``_NOISE_FALLBACK`` so every later window takes
    the host path; a host ``philox`` draw seeded with the same integer is
    the same value stream as the device one, so the song depends on the
    seed alone and double runs stay bit-identical.
    """
    global _NOISE_FALLBACK
    if _NOISE_FALLBACK:
        cpu_generator = torch.Generator()
        cpu_generator.manual_seed(seed)
        return torch.randn(shape, generator=cpu_generator).to(device=device, dtype=dtype)
    try:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    except Exception as exc:
        _NOISE_FALLBACK = True
        logger.warning(
            "MiniMax Music 3 310P acoustic: NPU randn with device generator "
            "failed (%s); latching the initial-noise draw to the host with the "
            "same seed",
            exc,
        )
        cpu_generator = torch.Generator()
        cpu_generator.manual_seed(seed)
        return torch.randn(shape, generator=cpu_generator).to(device=device, dtype=dtype)


def _load_components_310p(self: _acoustic_module.MiniMaxMusic3AcousticForConditionalGeneration) -> set[str]:
    """The upstream component loader with the 310P FP16 pin.

    Upstream moves the DiT and vocoder to the device in float32; 310P is an
    FP16/INT8 device, so the pin becomes float16 (POLICY section 2: FP16
    first; the diffusion side has no 310P INT8 e2e precedent yet).
    """
    root = resolve_repo_root(
        self.model_path,
        revision=self.vllm_config.model_config.revision,
        download_dir=self.vllm_config.load_config.download_dir,
    )
    encoder = self.dit.condition_encoder
    encoder.load_state_dict(load_component_state(root, CONDITION_ENCODER_DIR), strict=True)
    transformer_state = remap_transformer_state(load_component_state(root, TRANSFORMER_DIR))
    self.dit.transformer.load_state_dict(transformer_state, strict=True)
    self.vocoder.load_state_dict(load_component_state(root, VOCODER_DIR), strict=True)

    device = self.vllm_config.device_config.device
    self.dit.to(device=device, dtype=_RUNTIME_DTYPE).eval()
    self.vocoder.to(device=device, dtype=_RUNTIME_DTYPE).eval()
    folded = remove_weight_norm(self.vocoder)
    self._loaded = True

    loaded = {f"dit.{name}" for name, _ in self.dit.named_parameters()}
    loaded |= {f"vocoder.{name}" for name, _ in self.vocoder.named_parameters()}
    logger.info(
        "MiniMax Music 3 acoustic (310P) loaded from %s device=%s dtype=%s dit_steps=%d "
        "dit_cfg_scale=%.3f attention=decomposed folded_weight_norms=%d parameters=%d",
        root,
        device,
        _RUNTIME_DTYPE,
        self.dit_steps,
        self.dit_cfg_scale,
        folded,
        sum(p.numel() for p in self.parameters()),
    )
    return loaded


def _decode_window_310p(
    self: _acoustic_module.MiniMaxMusic3AcousticForConditionalGeneration,
    state,
    hidden: Tensor,
    window,
    seed: int,
) -> Tensor:
    """The upstream window decode with the 310P seams applied.

    The conditioning cast becomes float16, the DiT solve goes through the
    310P forward (host-resilient noise draw), and the rest of the method is
    copied verbatim so a later upstream change to the overlap/crop
    arithmetic shows up as a diff here.
    """
    device = self.vllm_config.device_config.device
    hidden = hidden.unsqueeze(0).to(device=device, dtype=_RUNTIME_DTYPE)
    noise_seed = _derive_seed(seed, "dit", window.index)
    try:
        generator = torch.Generator(device=device).manual_seed(noise_seed)
    except Exception:
        generator = torch.Generator().manual_seed(noise_seed)
    align = self.dit.aligned_condition(hidden)
    latent = self.dit(
        align,
        generator=generator,
        noise_seed=noise_seed,
        initial_latent=state.last_latent,
        initial_condition=state.last_condition,
        num_steps=self.dit_steps,
        cfg_scale=self.dit_cfg_scale,
    )
    waveform = self.vocoder(latent)

    overlap = overlap_mel_length()
    start = max(0, latent.shape[-1] - 2 * overlap)
    end = max(start, latent.shape[-1] - overlap)
    state.last_latent = latent[..., start:end].clone()
    state.last_condition = align[..., start:end].clone()

    wave = waveform[0].clamp(-1.0, 1.0).float()
    left, right = crop_sample_bounds(window)
    stop = wave.shape[-1] - right if right else wave.shape[-1]
    return wave[:, left:stop]


def _dit_forward_310p(
    self: _dit_module.MiniMaxMusic3FlowMatchingDiT,
    align: Tensor,
    *,
    generator: torch.Generator,
    noise_seed: int | None = None,
    initial_latent: Tensor | None = None,
    initial_condition: Tensor | None = None,
    num_steps: int = DEFAULT_DIT_STEPS,
    cfg_scale: float = DEFAULT_DIT_CFG_SCALE,
) -> Tensor:
    """The upstream Euler solve with the noise draw routed through 310P.

    ``noise_seed`` defaults to the generator's state so callers that do not
    thread the seed still get a deterministic draw; the seed is only used
    when the NPU draw latches to the host.
    """
    if num_steps < 1:
        raise ValueError("MiniMax Music 3 DiT num_steps must be positive")
    mel_len = align.shape[-1]
    if noise_seed is None:
        noise_seed = int(generator.initial_seed()) if hasattr(generator, "initial_seed") else 0
    x = _draw_noise_310p(
        (align.shape[0], DIT_LATENT_CHANNELS, mel_len),
        align.device,
        align.dtype,
        generator,
        noise_seed,
    )

    left = 0
    latent_prompt: Tensor | None = None
    noise_prompt: Tensor | None = None
    if initial_latent is not None:
        left = min(initial_latent.shape[-1], mel_len)
        if left <= 0:
            left = 0
        else:
            latent_prompt = initial_latent[..., :left].to(x)
            noise_prompt = x[..., :left].clone()
            if initial_condition is not None:
                align[..., :left] = initial_condition[..., :left].to(align)

    dt = 1.0 / num_steps
    cond_cfg = torch.zeros((2, *align.shape[1:]), device=align.device, dtype=align.dtype)
    cond_cfg[0].copy_(align[0])
    schedule = torch.tensor(
        [step / num_steps for step in range(num_steps)],
        device=align.device,
        dtype=align.dtype,
    )

    for step in range(num_steps):
        t = schedule[step].expand(x.shape[0])
        if left and latent_prompt is not None and noise_prompt is not None:
            x[..., :left] = (1.0 - (1.0 - _SIGMA_FLOOR) * t[0]) * noise_prompt + t[0] * latent_prompt
        d = self.transformer(x.expand(2, -1, -1), t.expand(2), cond_cfg)
        d = cfg_scale * d[:1] + (1.0 - cfg_scale) * d[1:2]
        x = x + dt * d

    if left and latent_prompt is not None:
        x[..., :left] = latent_prompt
    return x


def _aligned_condition_310p(self: _dit_module.MiniMaxMusic3FlowMatchingDiT, hidden: Tensor) -> Tensor:
    """The upstream conditioning with the nearest-resample on the host."""
    hidden = hidden.to(dtype=next(self.parameters()).dtype)
    align = self.condition(hidden)
    mel_len = self.aligned_mel_length(hidden.shape[1])
    return _nearest_gather_310p(align.cpu(), mel_len).to(hidden.device)


def _resampler_push_310p(self: _dav_module.StreamingResampler, chunks, *, flush: bool = False) -> Tensor:
    """The upstream streaming resampler with the polyphase stage on the host.

    ``torchaudio.functional.resample`` on an NPU device has no fleet
    precedent, so the resample runs on the host. The buffered pieces are at
    most a few hundred kilobytes per window, and the host polyphase keeps
    the class's documented invariant: the output is bit-identical to
    resampling the whole stream at once.
    """
    import torchaudio.functional as audio_functional

    parts = [chunk for chunk in chunks if chunk.numel()]
    if self._pending is not None:
        parts.insert(0, self._pending)
        self._pending = None
    if not parts:
        return self._empty(self._context)

    buffered = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    buffered = buffered.cpu()
    total = int(buffered.shape[-1])
    if flush:
        emit_len, lookahead = total, 0
    else:
        emit_len = max(0, ((total - self._margin) // self._block) * self._block)
        lookahead = min(self._margin, total - emit_len)
    if emit_len == 0:
        self._pending = buffered
        return self._empty(buffered)

    padded = buffered[..., : emit_len + lookahead]
    start = 0
    if self._context is not None:
        padded = torch.cat((self._context, padded), dim=-1)
        start = int(self._context.shape[-1]) * self.new_freq // self.orig_freq
    resampled = audio_functional.resample(padded.float(), self.orig_freq, self.new_freq)
    stop = start + -(-emit_len * self.new_freq // self.orig_freq)

    if flush:
        self.reset()
    else:
        self._context = buffered[..., max(0, emit_len - self._context_len) : emit_len].clone()
        self._pending = buffered[..., emit_len:].clone() if emit_len < total else None
    return resampled[..., start:stop]


def apply_acoustic_patches() -> None:
    """Install the MiniMax Music 3 stage-1 310P patches.

    Rebinds:

    * ``MiniMaxMusic3AcousticForConditionalGeneration._load_components`` to
      the FP16 pin,
    * ``MiniMaxMusic3AcousticForConditionalGeneration._decode_window`` to the
      FP16 + host-resilient noise variant,
    * ``MiniMaxMusic3FlowMatchingDiT.forward`` (Euler solve) to route the
      initial noise through ``_draw_noise_310p``,
    * ``MiniMaxMusic3FlowMatchingDiT.aligned_condition`` to the host gather
      resample,
    * ``Attention._attend`` (DiT blocks) to decomposed matmul attention,
    * ``MiniMaxMusic3Transformer1DModel._rope`` to host-built tables,
    * ``StreamingResampler.push`` to the host polyphase.

    Idempotent. No-ops with a warning when the kill switch is set.
    """
    global _ACOUSTIC_PATCHED

    if _ACOUSTIC_PATCHED:
        return
    if _kill_switch_off():
        logger.warning(
            "MiniMax Music 3 310P acoustic patches disabled by %s=0; "
            "the stock float32 load and CUDA-oriented ops will run",
            _KILL_SWITCH_ENV,
        )
        return

    acoustic_cls = _acoustic_module.MiniMaxMusic3AcousticForConditionalGeneration
    acoustic_cls._load_components = _load_components_310p
    acoustic_cls._decode_window = _decode_window_310p

    dit_cls = _dit_module.MiniMaxMusic3FlowMatchingDiT
    dit_cls.forward = _dit_forward_310p
    dit_cls.aligned_condition = _aligned_condition_310p
    _dit_module.MiniMaxMusic3Transformer1DModel._rope = _rope_310p
    _dit_module.Attention._attend = _attend_310p

    _dav_module.StreamingResampler.push = _resampler_push_310p

    _ACOUSTIC_PATCHED = True
    logger.info(
        "MiniMax Music 3 310P acoustic patches installed: FP16 pin, "
        "decomposed DiT attention, host RoPE tables, host nearest-resample, "
        "host streaming resampler, host-resilient noise draw"
    )
