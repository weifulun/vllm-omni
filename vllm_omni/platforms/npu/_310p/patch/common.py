# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared helpers for 310P diffusion patches (all model branches)."""

from __future__ import annotations

import json
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


def maybe_install_phase_tracer(pipeline, device) -> None:
    """Diagnostic: per-phase wall time from inside the diffusion worker.

    The Omni client process cannot reach the worker-side pipeline object, so
    the engine-side phase timers live in the 310P patches.  When
    ``VLLM_OMNI_310P_TRACE_PHASES=1`` this wraps the pipeline's DiT forward
    (``unet.forward`` or ``transformer.forward``) and ``vae.decode`` with
    NPU-synced timers and appends one JSON line per call to
    ``VLLM_OMNI_310P_PHASES_FILE`` (default ``/tmp/310p_phases.jsonl``); the
    runner ingests and clears the file between repeats.
    """
    if os.environ.get("VLLM_OMNI_310P_TRACE_PHASES", "0") != "1":
        return
    path = os.environ.get("VLLM_OMNI_310P_PHASES_FILE", "/tmp/310p_phases.jsonl")

    def _record(kind: str, fn):
        def traced(*args, **kwargs):
            torch.npu.synchronize(device)
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.npu.synchronize(device)
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({kind: time.perf_counter() - t0}) + "\n")
            except OSError:
                pass
            return out

        return traced

    dit = getattr(pipeline, "unet", None) or getattr(pipeline, "transformer", None)
    dit_name = "unet_s" if dit is getattr(pipeline, "unet", None) else "transformer_s"
    if dit is not None:
        dit.forward = _record(dit_name, dit.forward)
    pipeline.vae.decode = _record("vae_decode_s", pipeline.vae.decode)
    logger.info("[310P] phase tracer -> %s (%s)", path, dit_name)
