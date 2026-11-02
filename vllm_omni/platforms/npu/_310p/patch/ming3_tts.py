# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Ming-3.0-TTS (tiny, MoE) for the 310P NPU path.

The tiny variant's CUDA recipe is a single-stage Model Runner V2 pipeline
whose LLM backbone is the upstream ``BailingMoeV3ForCausalLM`` (18 KDA /
6 MLA hybrid).  On 310P the patch surface has four parts:

1. **KDA ops.**  The BailingMoeV3 KDA layers call FLA Triton kernels that
   do not exist on 310P (no Triton in the image; the ``vllm_ascend``
   ``patch_triton`` swaps are gated on ``HAS_TRITON`` and therefore never
   run here).  :mod:`ming3_tts_kda` rebinds pure-torch gate/recurrence/
   causal-conv ops onto the ``bailing_moe_v3`` module.  The rebind happens
   at worker patch time (``apply_patches``), before model construction,
   and is guarded by an upstream-layout check so a changed vllm layout
   fails loud instead of silently mixing paths.

2. **FP16 + eager pins.**  310P has no BF16, and the single-chip memory
   budget (FP16 weights ~= 80% of one 21.6 GiB chip) forbids CUDA graph
   capture: the deploy overlay sets ``enforce_eager: true`` /
   ``cudagraph_mode: NONE`` and runs ``model_runner: v1`` under
   ``platforms.npu`` (the tag tree rejects MRv2 on NPU).  The 310P
   subclass re-pins the flow head to FP16 and forces the CFM CUDAGraph
   executor off.

3. **CFM-API compatibility.**  ``Ming3TTSLLMModel`` (backported from
   main) calls the CFM-graph-cache API of main's ``MingLLMModel``; the
   tag tree carries the older dense ``MingLLMModel``.  The 310P subclass
   provides the missing methods (graph executor never engages because
   ``_maybe_build_cfm_graph`` returns None).

4. **Streaming audio decode on the V1 data plane.**  The single-stage
   recipe decodes its latents per step with ``StepVAEDecoder`` inside the
   MRv2 state machine (``Ming3ModelState``, created only by the
   ``worker_v2`` factory) — unreachable on NPU, which serves the legacy
   V1 data plane (both the tag-tree gate and vllm_ascend force V1 on
   310P).  ``StepVAEDecoder`` also requires ``vllm.vllm_flash_attn``
   (CUDA-only), so weight loading crashes on 310P.  The 310P line
   replaces it with an in-model streaming decode: the wrapper builds the
   plain ``AudioVAE`` decoder only, and the LLM subclass decodes each
   step's latent incrementally through the existing causal
   ``AudioVAE.decode(stream_state, last_chunk)`` API (the same call
   pattern the proven dense-line stage-1 uses on NPU) and emits the
   single-stage audio payload contract used by the Qwen3-TTS fused line
   (``{"model_outputs": [per-request PCM], "sr": [sr]}``).

Kill switch: ``VLLM_OMNI_310P_MING3TTS_PATCH=0`` disables the whole
patch (KDA rebind + model/wrapper swap) and falls back to the upstream
path.
"""

from __future__ import annotations

import logging
import os

import torch

from vllm_omni.model_executor.models.ming3_tts import ming3_tts as _ming3_tts_wrapper
from vllm_omni.model_executor.models.ming3_tts import ming3_tts_llm
from vllm_omni.model_executor.models.ming3_tts.ming3_tts_llm import (
    KEY_REQUEST_ID,
    _get_request_token_counts,
    _normalize_request_infos,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms.npu._310p.patch.ming3_tts_kda import install_kda_310p_ops

_PATCH_ENV = "VLLM_OMNI_310P_MING3TTS_PATCH"
_PATCHED = False
_KDA_INSTALLED = False

logger = logging.getLogger(__name__)

_RUNTIME_DTYPE = torch.float16


class _Ming3TTSLLMModel310P(ming3_tts_llm.Ming3TTSLLMModel):
    """Tiny-line LLM model pinned for the 310P NPU path.

    The base ``Ming3TTSLLMModel`` already skips the sync-free KDA prefill
    install when ``enforce_eager`` is set (the 310P overlay sets it), so
    the CUDA-graph Triton prefill never engages; this subclass pins the
    flow-head dtype, the CFM graph, and adds the V1 streaming audio
    decode (see module docstring, part 4).
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.fm_dtype = _RUNTIME_DTYPE
        self.flowloss.to(dtype=_RUNTIME_DTYPE)
        self.linear_proj_audio.to(dtype=_RUNTIME_DTYPE)
        self.stop_head.to(dtype=_RUNTIME_DTYPE)
        self.spk_head.to(dtype=_RUNTIME_DTYPE)
        # 310P streaming decode state (part 4): the AudioVAE decoder is
        # attached by the wrapper subclass in _load_step_vae; per-request
        # causal-decode state lives here between steps.
        self._npu310_vae: object | None = None
        self._npu310_stream_state: dict[str, tuple] = {}
        self._npu310_past_kv: dict[str, object] = {}

    def _npu310_attach_audio_vae(self, vae: object) -> None:
        self._npu310_vae = vae

    def _maybe_build_cfm_graph(self, z_diff_cond: torch.Tensor) -> None:
        # Hard off on 310P (single-chip budget forbids graph capture; the
        # eager flow head is the contract).
        del z_diff_cond
        return None

    # ------------------------------------------------------------------
    # CFM-API compatibility shims.
    #
    # The backported ``Ming3TTSLLMModel`` (main) calls the CFM-graph-cache
    # API that main's ``MingLLMModel`` (ming_tts_llm.py) provides
    # (``_init_cfm_graph_state`` / ``_warmup_cfm_graphs`` /
    # ``_draw_request_cfm_noises``); the tag tree carries the older dense
    # ``MingLLMModel`` whose only CFM-graph method is the single-graph
    # ``_maybe_build_cfm_graph``.  The 310P subclass therefore provides
    # the missing methods verbatim from main (with the graph executor
    # never engaging because ``_maybe_build_cfm_graph`` above returns
    # None), so the main model code runs unmodified on the tag tree.
    # ------------------------------------------------------------------

    def _init_cfm_graph_state(self) -> None:
        # Enum-vs-str tolerant comparison (``cudagraph_mode`` may be a
        # CUDAGraphMode member or a plain string depending on vllm version);
        # no module-level vllm import so the CPU contract tests can load
        # this file without a vllm install.
        compilation_config = self.vllm_config.compilation_config
        raw_mode = getattr(compilation_config, "cudagraph_mode", None)
        mode = getattr(raw_mode, "value", raw_mode)
        self._cfm_graph_enabled = (
            not self.vllm_config.model_config.enforce_eager
            and str(mode).lower() != "none"
        )
        self._cfm_graphs: dict[int, object] = {}
        self._cfm_graph_failed_batch_sizes: set[int] = set()
        self._cfm_graph_warmed_up = False

        raw_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None) or [1]
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        max_num_seqs = max(1, int(getattr(scheduler_config, "max_num_seqs", 1)))
        self._cfm_graph_batch_sizes = sorted(
            {int(size) for size in raw_sizes if 0 < int(size) <= max_num_seqs}
        )
        self._cfm_graph_num_warmups = max(
            1, int(getattr(compilation_config, "cudagraph_num_of_warmups", 1))
        )

    def _warmup_cfm_graphs(self) -> None:
        # No graph warmup on 310P: ``_maybe_build_cfm_graph`` never
        # returns an executor (eager-only flow head).
        self._cfm_graph_warmed_up = True

    def _draw_request_cfm_noises(
        self,
        generators: list[torch.Generator | None],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Verbatim from main's MingLLMModel (RNG call boundaries are the
        # determinism contract: an unseeded row set shares one global draw,
        # a seeded row draws per-generator with the batch-1 step shape).
        # ``DEFAULT_CFM_STEPS == 10`` (main) matches the dense line's
        # ``_CFM_STEPS == 10``.
        default_cfm_steps = 10
        shape = (1, self.ming_config.patch_size, self.ming_config.latent_dim)
        if all(generator is None for generator in generators):
            count = len(generators)
            initial = torch.randn((count, *shape[1:]), device=device, dtype=torch.float32)
            solver = torch.randn((default_cfm_steps, count, *shape[1:]), device=device, dtype=torch.float32)
            return initial, solver
        initial_noise = []
        solver_noise = []
        for generator in generators:
            initial_noise.append(
                torch.randn(
                    (1, self.ming_config.latent_dim, self.ming_config.patch_size),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                ).transpose(1, 2)
            )
            if self._cfm_graph_enabled:
                request_solver_noise = torch.randn(
                    (default_cfm_steps, *shape),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            else:
                request_solver_noise = torch.stack(
                    [
                        torch.randn(
                            shape,
                            device=device,
                            dtype=torch.float32,
                            generator=generator,
                        )
                        for _ in range(default_cfm_steps)
                    ]
                )
            solver_noise.append(request_solver_noise)
        return torch.cat(initial_noise, dim=0), torch.cat(solver_noise, dim=1)

    # ------------------------------------------------------------------
    # Part 4: V1 streaming audio decode.
    # ------------------------------------------------------------------

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput | object,
        *,
        model_intermediate_buffer: list[dict[str, object]] | None = None,
        seq_token_counts: list[int] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        request_sample_eligible: list[bool] | None = None,
        **kwargs: object,
    ) -> OmniOutput | torch.Tensor | object:
        out = super().make_omni_output(
            model_outputs,
            model_intermediate_buffer=model_intermediate_buffer,
            seq_token_counts=seq_token_counts,
            request_token_spans=request_token_spans,
            request_sample_eligible=request_sample_eligible,
            **kwargs,
        )
        if self._npu310_vae is None or not isinstance(out, OmniOutput):
            return out
        if out.multimodal_outputs is None or out.text_hidden_states is None:
            # Prefill step (no latent rows yet) or non-AR output: nothing
            # to decode.
            return out
        latent_rows = out.multimodal_outputs.get(self.latent_patch_key)
        has_patch = out.multimodal_outputs.get(self.has_patch_key)
        if latent_rows is None or has_patch is None:
            return out

        hidden = out.text_hidden_states
        total_tokens = hidden.shape[0]
        runtime_information = (
            model_intermediate_buffer
            if model_intermediate_buffer is not None
            else kwargs.get("runtime_additional_information")
        )
        request_infos = _normalize_request_infos(runtime_information)
        token_counts = _get_request_token_counts(
            hidden, request_infos, seq_token_counts, request_token_spans
        )

        sr_t = torch.tensor(int(self.ming_config.sample_rate), dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32, device=hidden.device)
        payloads: list[torch.Tensor] = [empty.clone() for _ in range(len(token_counts))]

        cursor = 0
        for req_idx, token_count in enumerate(token_counts):
            end = min(cursor + token_count, total_tokens)
            if end <= cursor:
                continue
            cursor = end
            if request_sample_eligible is not None and not request_sample_eligible[req_idx]:
                continue
            row = end - 1
            if not bool(has_patch[row]):
                continue
            req_info = request_infos[req_idx] if req_idx < len(request_infos) else {}
            req_id = req_info.get(KEY_REQUEST_ID)
            if not isinstance(req_id, str):
                continue
            # The base set this step's stop decision before returning.
            finished = req_id in self._pending_postprocess_updates
            latent = latent_rows[row : row + 1]
            waveform, stream_state, past_kv = self._npu310_vae.decode(  # type: ignore[union-attr]
                latent,
                past_key_values=self._npu310_past_kv.get(req_id),
                use_cache=True,
                stream_state=self._npu310_stream_state.get(req_id, (None, None, None)),
                last_chunk=finished,
            )
            if finished:
                self._npu310_stream_state.pop(req_id, None)
                self._npu310_past_kv.pop(req_id, None)
            else:
                self._npu310_stream_state[req_id] = stream_state
                self._npu310_past_kv[req_id] = past_kv
            payloads[req_idx] = waveform.reshape(-1).to(torch.float32)

        return out._replace(
            multimodal_outputs={
                "model_outputs": payloads,
                "sr": [sr_t.clone() for _ in range(len(token_counts))],
            }
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        super().on_requests_finished(finished_req_ids)
        for req_id in finished_req_ids:
            self._npu310_stream_state.pop(req_id, None)
            self._npu310_past_kv.pop(req_id, None)


class _Ming3TTSWrapper310P(_ming3_tts_wrapper.Ming3TTSForConditionalGeneration):
    """Tiny-line wrapper pinned for the 310P NPU path.

    The only delta from the base is ``_load_step_vae``: the base builds
    ``StepVAEDecoder`` (whose ``PackedQwen2`` streaming attention imports
    ``vllm.vllm_flash_attn``, CUDA-only) — on 310P only the plain
    ``AudioVAE`` decoder is built, and the LLM subclass does the
    per-step streaming decode instead.
    """

    def _load_step_vae(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        param = next(self.parameters())
        vae = _ming3_tts_wrapper.AudioVAE(
            self.ming_config.audio_tokenizer_config, build_encoder=False
        ).eval()
        state = vae.state_dict()
        with torch.no_grad():
            for name, value in weights:
                key = name[len("audio.") :]
                if key not in state:
                    raise RuntimeError(
                        f"Ming3 single-stage AudioVAE received an unknown tensor: {name}"
                    )
                state[key].copy_(value)
        missing = {f"audio.{key}" for key in state} - {name for name, _ in weights}
        if missing:
            raise RuntimeError(
                f"Ming3 single-stage AudioVAE left tensors unloaded: {sorted(missing)[:8]}"
            )
        vae.to(device=param.device, dtype=self.vllm_config.model_config.dtype)
        vae.decoder.upsampling.frame_holdback = True
        # Registered as a submodule so the runner counts its memory before
        # profiling.
        self.audio_vae = vae
        # No StepVAEDecoder on 310P: PackedQwen2 requires the CUDA flash
        # attention extension. The LLM model decodes the per-step latents
        # with the causal AudioVAE.decode API instead.
        self.step_vae = None
        attach = getattr(self.model, "_npu310_attach_audio_vae", None)
        if callable(attach):
            attach(vae)
        return {f"audio_vae.{name}" for name in vae.state_dict()}


def _kill_switch_off() -> bool:
    return os.environ.get(_PATCH_ENV, "1") == "0"


def install_kda_ops() -> None:
    """Worker-level hook: rebind KDA/conv ops (see module docstring)."""
    global _KDA_INSTALLED
    if _KDA_INSTALLED:
        return
    if _kill_switch_off():
        logger.info("Ming3 310P patch disabled by %s=0; upstream KDA path kept", _PATCH_ENV)
        return
    install_kda_310p_ops()
    _KDA_INSTALLED = True


def apply_ming3_tts_tiny_patches() -> None:
    """Model-level hook: swap the tiny-line LLM and wrapper classes."""
    global _PATCHED
    if _PATCHED:
        return
    if _kill_switch_off():
        logger.info("Ming3 310P patch disabled by %s=0; model swap skipped", _PATCH_ENV)
        return
    ming3_tts_llm.Ming3TTSLLMModel = _Ming3TTSLLMModel310P
    _ming3_tts_wrapper.Ming3TTSForConditionalGeneration = _Ming3TTSWrapper310P
    _PATCHED = True
    logger.info(
        "Ming3 310P patch installed (Ming3TTSLLMModel -> 310P pin, wrapper -> 310P V1 decode)"
    )
