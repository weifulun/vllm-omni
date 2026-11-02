# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""310P patches for MiniCPM-o 4.5 (thinker / talker / code2wav).

MiniCPM-o 4.5 runs three stages, co-located on one Duo 310P card (see
``deploy/minicpmo_4_5_310p.yaml``):

* stage 0 — thinker (``MiniCPMO45OmniForConditionalGeneration``): ~8.7B AR
  LLM plus SigLIP and Whisper encoders, ``model_stage="llm"``;
* stage 1 — talker (``MiniCPMO45OmniTTSForConditionalGeneration``): ~190M
  codec LM, ``model_stage="tts"``;
* stage 2 — code2wav (``MiniCPMO45Code2Wav``): CFM flow-matching DiT + HiFT
  vocoder, ``model_stage="code2wav"``.

The 910 deployment leans on two things 310P does not have:

* **Triton.** triton-ascend does not target 310P (research/003), so every
  Triton kernel in the model path is replaced by an eager torch
  implementation. The talker's ``codec_state.CodecState`` kernels are the
  hot path (per decode step, batched) and get a full torch re-implementation
  with identical semantics (see ``CodecState310P`` below). The talker
  imports ``codec_state`` lazily at call time (``from .codec_state import
  ...`` inside ``make_omni_output``), so patching the module attributes is
  enough; if the Triton import itself fails on 310P a shim module is
  installed so the lazy import keeps working.
* **NPUGraph capture.** Not graph-safe on 310P (research/004; MOSS/VOXCPM
  precedent): encoder NPUGraph, Code2Wav CFM/HiFT/encoder graphs and the
  connector-side graph toggles are all switched off in the deploy overlay;
  this module only defends against a deployment that forgets the overlay.

The FP16-only rule (POLICY section 2) is carried by the overlay's
``dtype: float16`` plus the small dtype corners below.

Idempotence: every ``apply_*`` function is a no-op the second time it runs
(``_PATCHED_*`` flags), matching the ``qwen3_tts.py`` template.

Intentionally NOT enabled on 310P (see RISKS.md for the real-device
follow-ups marked ``TODO(310P-real)``):

* encoder NPUGraph (``encoder_enable_npu_graph``) — overlay only;
* Code2Wav NPUGraph stack (``code2wav_enable_npu_graph``, ``enable_hift_graph``,
  ``enable_cfm_graph``, ``enable_code2wav_encoder_graph``) — overlay only;
* fused talker codec prelude (``talker_codec_fused_prelude``) — overlay plus
  the process latch forced off below.

K-step multi-frame speculative decode (``talker_kstep_frames: 8``) is kept
enabled: it is an NPU-generic torch path and the 310P model runner owns its
own graph capture. Whether the 310P runner captures the FULL_DECODE_ONLY
mode cleanly is a real-device question — fallback is dropping stage 1's
``speculative_config`` (one codec frame per step, correct but slower).
TODO(310P-real)
"""

from __future__ import annotations

import logging
import sys

import torch

logger = logging.getLogger(__package__)

#: 310P is FP16-only (POLICY section 2). The overlay already sets
#: ``dtype: float16``; this constant only pins the dtype corners the engine
#: dtype cannot reach (hard-coded tensors, streaming-encoder gates).
_RUNTIME_DTYPE = torch.float16

_MINICPMO_PKG = "vllm_omni.model_executor.models.minicpmo_4_5"

_PATCHED_THINKER = False
_PATCHED_TALKER = False
_PATCHED_CODE2WAV = False


# ---------------------------------------------------------------------------
# Talker: eager torch CodecState (drop-in for the Triton kernels)
# ---------------------------------------------------------------------------


class CodecState310P:
    """Eager-torch replacement for ``codec_state.CodecState`` on 310P.

    Mirrors the Triton kernel semantics of ``_update_codec_state`` and
    ``_mask_embeddings`` (see ``model_executor/models/minicpmo_4_5/codec_state.py``)
    without any Triton dependency. Per-request slot state is
    ``[step, finished, limit, minimum, drain]`` (int64) plus a rolling
    ``window``-wide codec history ring (oldest at index 0).

    The public surface matches ``CodecState`` exactly: ``__init__``,
    ``prepare``, ``update``, ``indices``, ``mask_embeddings``, ``release``.
    """

    def __init__(self, device, vocab_size, capacity=4096, window=16):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.vocab_size = vocab_size
        self.window = window
        self.capacity = capacity
        self.state = torch.zeros((capacity, 5), dtype=torch.long, device=self.device)
        self.history = torch.full(
            (capacity, window), vocab_size, dtype=torch.long, device=self.device
        )
        self.slots = {}
        self._index_cache = {}
        self.free_slots = list(range(capacity - 1, -1, -1))

    def prepare(self, rows):
        """Allocate slots for new requests and stage their configs/device history."""
        new_slots, configs, histories = [], [], []
        for request_id, state in rows:
            if "_gpu_slot" in state:
                continue
            slot = self.slots.get(request_id)
            if slot is None:
                if not self.free_slots:
                    # Preempted requests retain state but no longer count
                    # against max_num_seqs. Grow without reading it on CPU.
                    capacity = max(1, self.capacity * 2)
                    state_storage = self.state.new_empty((capacity, 5))
                    history_storage = self.history.new_empty((capacity, self.window))
                    state_storage[: self.capacity].copy_(self.state)
                    history_storage[: self.capacity].copy_(self.history)
                    self.state, self.history = state_storage, history_storage
                    self.free_slots.extend(range(capacity - 1, self.capacity - 1, -1))
                    self.capacity = capacity
                slot = self.free_slots.pop()
                self.slots[request_id] = slot
            state["_gpu_slot"] = slot
            new_slots.append(slot)
            configs.append(
                [
                    int(state.get("step", 0)),
                    bool(state.get("finished")),
                    int(state.get("max_tokens") or (2**31 - 1)) - 1,
                    int(state.get("min_tokens") or 0),
                    int(state.get("turn_end_drain") or 0),
                ]
            )
            recent = state.get("recent_codes", [])
            recent = torch.as_tensor(recent, dtype=torch.long).to(self.device, non_blocking=True).reshape(-1)
            padding = torch.full((self.window,), self.vocab_size, device=self.device, dtype=torch.long)
            histories.append(torch.cat([padding, recent])[-self.window :])
        if new_slots:
            slots = torch.tensor(new_slots, dtype=torch.long, device=self.device)
            self.state.index_copy_(0, slots, torch.tensor(configs, dtype=torch.long, device=self.device))
            self.history.index_copy_(0, slots, torch.stack(histories).to(self.device, non_blocking=True))

    def update(
        self,
        rows,
        ids,
        *,
        eos,
        cadence,
        boundary,
        code_classes,
        silent_frames,
        quiet_stop_frames,
    ):
        """Advance one decode step (torch port of ``_update_codec_state``)."""
        self.prepare([(request_id, state) for request_id, state, _ in rows])
        slots = torch.tensor(
            [state["_gpu_slot"] for _, state, _ in rows], dtype=torch.long, device=self.device
        )
        decode = torch.tensor(
            [code is not None for _, _, code in rows], dtype=torch.bool, device=self.device
        )
        ids = ids.to(self.device, non_blocking=True).long()
        code = ids

        step0 = self.state[slots, 0]
        finished = self.state[slots, 1] != 0
        limit = self.state[slots, 2]
        minimum = self.state[slots, 3]
        drain = self.state[slots, 4]

        valid = decode & ~finished & (code != eos)
        step = step0 + valid
        done = finished | (decode & (code == eos)) | (step >= limit)

        hist0 = self.history[slots]
        offsets = torch.arange(self.window, device=self.device)
        # Load every old value before overwriting the history in place:
        # shifted[o] <- hist0[o + 1] (tail element replaced by the new code).
        shifted = torch.where(
            (offsets == self.window - 1).unsqueeze(0),
            code.unsqueeze(1),
            torch.roll(hist0, shifts=-1, dims=1),
        )
        updated = torch.where(valid.unsqueeze(1), shifted, hist0)

        classes = code_classes[updated]
        current = offsets.unsqueeze(0) >= (self.window - step).unsqueeze(1)
        silent = (classes & 1).bool() & current & (offsets.unsqueeze(0) >= self.window - silent_frames)
        quiet = (classes & 2).bool() & current & (offsets.unsqueeze(0) >= self.window - quiet_stop_frames)
        settled = silent.sum(dim=1) == silent_frames
        until_silent = drain == 2
        done = done | (until_silent & (quiet.sum(dim=1) == quiet_stop_frames))
        boundary = (drain == 1) & (step >= cadence) & ((step % cadence) < boundary)
        masked = ~done & ((step < minimum) | boundary | (until_silent & ~settled))

        self.history[slots] = updated
        self.state[slots, 0] = step
        self.state[slots, 1] = done.to(torch.long)
        return updated, valid, done, masked

    def indices(self, slots, flags):
        key = (slots, flags)
        cached = self._index_cache.get(key)
        if cached is None:
            if len(self._index_cache) >= 16:
                self._index_cache.clear()
            cached = (
                torch.tensor(slots, dtype=torch.long, device=self.device),
                torch.tensor(flags, dtype=torch.bool, device=self.device),
            )
            self._index_cache[key] = cached
        return cached

    def mask_embeddings(self, embeds, slots, rows):
        """Zero the embedding rows of requests whose slot is finished (torch port of ``_mask_embeddings``)."""
        slot_ids, _ = self.indices(tuple(slots), (True,) * len(slots))
        row_ids, _ = self.indices(tuple(rows), (True,) * len(rows))
        # slots[r] / rows[r] are parallel: row r owns embeds row rows[r] and
        # belongs to slot slots[r].
        slot_finished = self.state[slot_ids, 1] != 0
        finished_rows = row_ids[slot_finished]
        if finished_rows.numel():
            embeds[finished_rows] = 0.0

    def release(self, request_id):
        slot = self.slots.pop(request_id, None)
        if slot is not None:
            self.free_slots.append(slot)


def mask_logits_310p(logits, forced, masked, eos):
    """Torch port of ``_mask_logits``: force rows to the codec EOS, mask rows lose it."""
    vocab = logits.shape[1]
    tokens = torch.arange(vocab, device=logits.device)
    force = forced.to(torch.bool).unsqueeze(1)
    mask = masked.to(torch.bool).unsqueeze(1)
    values = torch.where(force, torch.where(tokens == eos, 0.0, float("-inf")), logits)
    values = torch.where(mask & (tokens == eos), float("-inf"), values)
    logits.copy_(values.to(logits.dtype))
    return logits


def apply_window_penalty_310p(logits, histories, penalties):
    """Torch port of ``_window_penalty``: alpha = penalty ** frequency-in-window."""
    penalties = penalties.to(torch.float32)
    frequencies = (
        histories.to(torch.long).unsqueeze(2) == torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    ).sum(dim=1)
    alpha = torch.exp(torch.log(penalties).unsqueeze(1) * frequencies.to(torch.float32))
    values = logits.to(torch.float32)
    return torch.where(values < 0, values * alpha, values / alpha).to(logits.dtype)


def _install_codec_state_310p() -> None:
    """Point the lazily-imported ``codec_state`` module at the torch fallbacks."""
    try:
        from vllm_omni.model_executor.models.minicpmo_4_5 import codec_state

        codec_state.CodecState = CodecState310P
        codec_state.mask_logits = mask_logits_310p
        codec_state.apply_window_penalty = apply_window_penalty_310p
        return
    except Exception as exc:  # triton import fails on Triton-less 310P hosts
        logger.info(
            "310P: codec_state module unavailable (%s); installing an eager-torch shim",
            exc,
        )

    import types

    shim = types.ModuleType(f"{_MINICPMO_PKG}.codec_state")
    shim.CodecState = CodecState310P
    shim.mask_logits = mask_logits_310p
    shim.apply_window_penalty = apply_window_penalty_310p
    sys.modules[f"{_MINICPMO_PKG}.codec_state"] = shim
    try:
        import importlib

        package = importlib.import_module(_MINICPMO_PKG)
        setattr(package, "codec_state", shim)
    except Exception:
        pass


def _force_fused_prelude_off() -> None:
    """Force the talker's fused-prelude process latch off on 310P.

    ``talker_codec_sample._FUSED_PRELUDE_OK`` is a permanent process-wide
    latch. The deployment overlay already sets ``talker_codec_fused_prelude:
    false`` (which keeps ``prepare_codec_logits`` on the eager chain); the
    latch here also covers deployments that forgot the overlay. Without
    triton the fused path would fall back safely anyway (``_HAS_TRITON``),
    but with a triton-ascend build that is not 310P-verified we do not want
    an unverified kernel on the hot sampling path (POLICY: unverified fast
    paths off).
    """
    import vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample as tcs

    if tcs._FUSED_PRELUDE_OK:
        tcs._FUSED_PRELUDE_OK = False
        logger.info("310P: talker fused codec prelude disabled (eager torch chain)")


def _allow_fp16_streaming_audio() -> None:
    """Let the streaming (duplex) audio encoder accept FP16 on 310P.

    The batched streaming Whisper encoder validates its weight dtype against
    ``_SUPPORTED_DTYPES = (bfloat16, float32)``. 310P is FP16-only, so FP16 is
    added to the accepted set. (bf16 stays accepted: the gate only rejects
    dtypes, it does not cast.) TODO(310P-real): quality-check streaming duplex
    audio in FP16 before trusting it; fallback is running the streaming
    encoder in FP32 on CPU.
    """
    import vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder as sae

    if _RUNTIME_DTYPE not in sae._SUPPORTED_DTYPES:
        sae._SUPPORTED_DTYPES = tuple(sae._SUPPORTED_DTYPES) + (_RUNTIME_DTYPE,)
        logger.info("310P: streaming audio encoder dtype gate now accepts float16")


# ---------------------------------------------------------------------------
# apply_* entry points (idempotent; called from _310p/patch/__init__.py)
# ---------------------------------------------------------------------------


def apply_thinker_310p_patches() -> None:
    """Stage 0 (thinker): FP16 dtype corners. The AR body itself loads at the
    engine dtype (``float16`` from the overlay); the SigLIP/Whisper encoders
    follow the same loader. NPUGraph for the encoders is disabled by the
    overlay (``encoder_enable_npu_graph: false``) — a capture there would not
    be graph-safe on 310P."""
    global _PATCHED_THINKER
    if _PATCHED_THINKER:
        return
    _allow_fp16_streaming_audio()
    _PATCHED_THINKER = True
    logger.info("310P: MiniCPM-o 4.5 thinker patches applied")


def apply_talker_310p_patches() -> None:
    """Stage 1 (talker): swap the Triton codec-state kernels for torch and
    force the fused-prelude latch off. K-step multi-frame decode stays enabled
    (NPU-generic torch path; the 310P runner owns its own graph capture).
    TODO(310P-real): verify K=8 speculative capture on the 310P runner;
    fallback is dropping stage 1 ``speculative_config`` (one frame per step)."""
    global _PATCHED_TALKER
    if _PATCHED_TALKER:
        return
    _install_codec_state_310p()
    _force_fused_prelude_off()
    _PATCHED_TALKER = True
    logger.info("310P: MiniCPM-o 4.5 talker patches applied")


def apply_code2wav_310p_patches() -> None:
    """Stage 2 (code2wav): NPUGraph stack stays off (overlay), TF32 stays off
    (310P has no TF32 path; the default ``token2wav_allow_tf32: tf32`` is a
    no-op off SM80+ but is disabled explicitly in the overlay). The eager
    CFM/HiFT bodies are plain torch and run in FP16."""
    global _PATCHED_CODE2WAV
    if _PATCHED_CODE2WAV:
        return
    _PATCHED_CODE2WAV = True
    logger.info("310P: MiniCPM-o 4.5 code2wav patches applied (graphs off via overlay)")
