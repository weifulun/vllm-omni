# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P runtime patches for the Wan-Dancer diffusion pipeline (task-40).

Registered in ``_DIFFUSION_310P_PATCH_MODULES`` under ``WanDancerPipeline``
(COMPARISON-POLICY.md §5): the model branch keeps all logic here and never
edits platform.py.

What the patch installs on a 310P host:

* **math-SDPA gate on ``torch.nn.functional.scaled_dot_product_attention``**
  (device-gated: NPU tensors take the explicit matmul + fp32-softmax + matmul
  path with per-head 256 MiB row chunking; CPU/CUDA tensors pass through
  unchanged).  This is the research/012 F.sdpa global dead-zone fix
  ("can not cast format when output is input") for every direct ``F.sdpa``
  caller in the process — the Wan-Dancer VAE mid-block attention, the
  diffsynth ``flash_attention`` fallback, and any sibling pipeline.  The
  Dancer DiT itself calls ``math_sdpa`` explicitly, so it is wall-agnostic.
* **FP16-only dtype pin** (310P has no BF16; ``VLLM_OMNI_310P_DEFAULT_FP16=1``
  default, zimage/fluxklein precedent) applied by the loader, not here.
* **placement knobs** (read by the runner, resolved here): per-block dual-die
  DiT placement, VAE device, encoder CPU-pin defaults.

Kill switch: ``VLLM_OMNI_310P_DIFFUSION_PATCHES=0`` (handled by the registry
dispatcher — the generic 310P path becomes the level-2 same-seed reference).
Warmup skip: ``VLLM_OMNI_310P_SKIP_WARMUP=1`` (owned by the SDXL shared-asset
chain, ``_maybe_skip_startup_warmup``).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_APPLIED = False

DEFAULT_DIT_SPLIT = "20,20"  # blocks -> npu:0, npu:1 (davinci6 / davinci7)


def _is_npu_device(device) -> bool:
    try:
        return device.type == "npu"
    except AttributeError:
        return False


def _is_npu_tensor(t) -> bool:
    try:
        return _is_npu_device(t.device)
    except AttributeError:
        return False


def _math_sdpa_gated(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    """Replacement body for ``torch.nn.functional.scaled_dot_product_attention``."""
    global _orig_sdpa
    enable_gqa = kwargs.pop("enable_gqa", False)
    if not _is_npu_tensor(query) or kwargs or dropout_p != 0.0:
        # CPU/CUDA passthrough, or unsupported kwargs (keep the fused kernel).
        return _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_attention import math_sdpa

    if enable_gqa and query.shape[2] != key.shape[2]:
        if query.shape[2] % key.shape[2] != 0:
            raise ValueError(f"GQA mismatch: q_heads={query.shape[2]} kv_heads={key.shape[2]}")
        rep = query.shape[2] // key.shape[2]
        key = key.repeat_interleave(rep, dim=2)
        value = value.repeat_interleave(rep, dim=2)
    return math_sdpa(query, key, value, attn_mask=attn_mask, causal=is_causal, scale=scale)


_orig_sdpa = None


def _install_math_sdpa_gate() -> None:
    """Rebind ``torch.nn.functional.scaled_dot_product_attention`` (device-gated)."""
    global _orig_sdpa, _APPLIED
    import torch

    if _orig_sdpa is not None:
        return
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = _math_sdpa_gated
    logger.warning("310P Wan-Dancer: torch.nn.functional.scaled_dot_product_attention gated to math SDPA on NPU")


def resolve_dit_block_devices(depth: int = 40) -> list:
    """Per-block device list from ``VLLM_OMNI_310P_DIT_BLOCK_DEVICES``.

    Comma-split counts per device index, e.g. ``"20,20"`` ->
    ``[npu:0]*20 + [npu:1]*20`` (the dual-die placement); ``"40"`` -> all
    ``npu:0``; empty/unset -> ``None`` (single-device, model decides).
    """
    import torch

    raw = os.environ.get("VLLM_OMNI_310P_DIT_BLOCK_DEVICES", DEFAULT_DIT_SPLIT)
    if not raw.strip() or raw.strip() == "single":
        return None
    counts = [int(p) for p in raw.split(",") if p.strip()]
    if sum(counts) != depth:
        raise ValueError(f"VLLM_OMNI_310P_DIT_BLOCK_DEVICES={raw!r} does not sum to {depth}")
    devices: list[torch.device] = []
    for idx, count in enumerate(counts):
        devices.extend([torch.device(f"npu:{idx}")] * count)
    return devices


def resolve_vae_device():
    import torch

    raw = os.environ.get("VLLM_OMNI_310P_VAE_DEVICE", "1")
    if raw.strip() == "":
        return None
    return torch.device(f"npu:{int(raw)}")


def apply_diffusion_310p_patches(od_config) -> None:
    """Registry entry point (engine-driven pipeline construction)."""
    global _APPLIED
    arch = getattr(od_config, "model_class_name", None)
    if arch != "WanDancerPipeline":
        return
    _maybe_apply()


def maybe_apply_diffusion_310p_patches(arch: str | None = "WanDancerPipeline") -> None:
    """Runner-driven entry point: self-gated, idempotent, kill-switch aware.

    No-op off 310P, for other archs, or with the kill switch set.
    """
    global _APPLIED
    if _APPLIED:
        return
    if os.environ.get("VLLM_OMNI_310P_DIFFUSION_PATCHES", "1") == "0":
        logger.info("310P Wan-Dancer: diffusion patches disabled (VLLM_OMNI_310P_DIFFUSION_PATCHES=0)")
        return
    if arch is not None and arch != "WanDancerPipeline":
        return
    from vllm_omni.platforms.npu._310p import is_310p

    if not is_310p():
        return
    _maybe_apply()
    _APPLIED = True


def _maybe_apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _install_math_sdpa_gate()
    split = os.environ.get("VLLM_OMNI_310P_DIT_BLOCK_DEVICES", DEFAULT_DIT_SPLIT)
    vae_dev = os.environ.get("VLLM_OMNI_310P_VAE_DEVICE", "1")
    logger.warning(
        "310P Wan-Dancer patches active: dit_block_devices=%s vae_device=npu:%s "
        "math_sdpa_gate=on encoders=cpu-pin (T5 bf16 / CLIP fp32)",
        split,
        vae_dev,
    )
