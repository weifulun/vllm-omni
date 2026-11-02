# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch the SANA-WM diffusion pipeline for the 310P NPU path.

SANA-WM Stage-1 is a camera-controllable first-frame image-to-video world
model: Stage-1 DiT (Gated-DeltaNet hybrid, ~2.05B params) + LTX-2 video VAE
(32x spatial / 8x temporal, forced tiling) + Gemma-2-2b text encoder.  On 310P
there is no MindIE-SD operator surface and no BF16: diffusion runs plain
``torch_npu`` in FP16 and fully eager.  This module follows the SDXL 310P
patch pattern (COMPARISON-POLICY.md §5 registry branch):

* force the 310P runtime dtype (FP16) on the diffusion config before
  ``DiffusersPipelineLoader`` builds the modules, so DiT / VAE / encoders are
  built and loaded in FP16;
* pin the Gemma-2-2b text encoder on CPU (preprocessing-only, runs once per
  prompt branch; the 310P CLIP/VLM op surface is unverified and ~5.2 GB of
  resident FP16 weights are pure budget cost on the 21589 MB/chip budget);
  prompt embeds move back to the NPU in the runtime FP16 dtype;
* keep the forced VAE tiling/slicing on (the pipeline re-asserts tiling in
  ``_ensure_vae``; the config flags here are belt-and-braces, mirroring SDXL);
* refuse NPU attention-KV quantization (MindIE-only path);
* register a 310P-specialized pipeline under the same ``model_arch``.

The attention backend is NOT hard-pinned here: with MindIE-SD absent the NPU
platform already defaults to ``TORCH_SDPA``, and the 310P-wide explicit math
SDPA decomposition (``patch/attention.py``) is applied by the registry
dispatcher before this module loads — both the vllm ``Attention`` softmax
blocks and any direct ``F.scaled_dot_product_attention`` call (diffusers VAE)
route through it.

Placement (mount-recipe.md §4, 21589 MB/chip measured): Stage-1 DiT FP16
~4.1 GB + VAE FP16 ~1.2 GB ≈ 5.3 GB resident on one chip; Gemma-2-2b (~5.2 GB
FP16) stays in host RAM.  **TP1 single chip** — no INT8 fallback and no
multi-chip placement required (research/011 parameter verification).

Warmup note: the pipeline sets ``dummy_run_num_frames = 0``, so the generic
512x512 startup warmup (the research/012 dynamic-TBE failure class) never
runs for SANA-WM; the first real request pre-compiles kernels.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

import vllm_omni.diffusion.models.sana_wm.ucpe as _sana_wm_ucpe
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import (
    SANA_WM_STAGE1_PROMPT_CHANNELS,
    SanaWmPipeline,
    get_sana_wm_pre_process_func,
)
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    SanaWmSelfAttention,
    SanaWmWanRotaryPosEmbed,
)

logger = logging.getLogger(__name__)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_ENCODER_ON_NPU_ENV = "VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU"
_SANA_WM_ARCH = "SanaWmPipeline"
_PATCHED = False
_ORIGINAL_TORCH_CROSS: Any = None
# Effective "never tile vertically" threshold for the LTX-2 VAE (pixels).
_VAE_NO_VERTICAL_TILE_MIN_HEIGHT = 10_000_000
# Optional benchmark timing file (JSON lines), see _SanaWm310P._install_timing_hooks.
_TIMING_FILE_ENV = "VLLM_OMNI_310P_SANA_WM_TIMING_FILE"

__all__ = [
    "_SanaWm310P",
    "apply_diffusion_310p_patches",
    "get_sana_wm_pre_process_func",
]


def _encoder_on_npu() -> bool:
    """Opt out of CPU encoder pinning for A/B experiments."""
    return os.environ.get(_ENCODER_ON_NPU_ENV, "0") == "1"


# ---------------------------------------------------------------------------
# 310P RoPE: real-arithmetic replacements for the upstream float64/complex128
# path.  Upstream builds the Wan-style RoPE table with ``torch.polar`` on
# float64 and applies it via ``view_as_complex`` + complex128 multiply
# (``sana_wm_transformer.py::_build_1d_freq`` / ``SanaWmSelfAttention.
# _apply_rotary_emb`` / ``ucpe.py::_apply_complex_rope``).  On 310P the model
# is constructed inside an NPU device context, so the bare ``torch.arange``
# lands on npu:0 and ``aclnnPolar`` rejects DT_DOUBLE (supported list is
# [DT_FLOAT]); complex128 / DT_DOUBLE kernels are unavailable.  The table is
# rebuilt on the CPU (fp64 arithmetic, numerically identical to the upstream
# formula) and stored complex64; both apply sites decompose the complex
# multiply into four real fp32 MACs per pair — exact algebra, accumulation
# promoted from fp64 to fp32 (below the fp16 output noise floor).
# ---------------------------------------------------------------------------


def _build_1d_freq_310p(self, dim: int, positions: torch.Tensor) -> torch.Tensor:
    """310P CPU/fp32 replacement for ``SanaWmWanRotaryPosEmbed._build_1d_freq``.

    MIRROR of the upstream body with three 310P changes: all arithmetic on
    the CPU (positions detached to CPU fp64), the table returned complex64
    (upstream complex128).  Formula is unchanged.  Restored after pipeline
    construction; only active while ``_SanaWm310P.__init__`` builds modules.
    """
    if dim <= 0:
        return torch.empty(positions.shape[0], 0, dtype=torch.complex64, device=_CPU_DEVICE)
    positions_cpu = positions.detach().to(device=_CPU_DEVICE, dtype=torch.float64)
    # Explicit CPU device: the worker runs under an NPU default-device
    # context, so a bare arange would land on npu:0 and mix devices against
    # the CPU positions tensor.
    exp = torch.arange(0, dim, 2, dtype=torch.float64, device=_CPU_DEVICE)[: dim // 2] / dim
    freqs = 1.0 / (self.theta ** exp)
    phase = torch.outer(positions_cpu, freqs)
    return torch.polar(torch.ones_like(phase), phase).to(torch.complex64)


def _rope_apply_real_forward(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """310P real-arithmetic replacement for ``SanaWmSelfAttention._apply_rotary_emb``.

    Mirrors the upstream body line-for-line (permute / unflatten / flatten /
    permute order preserved) with the complex multiply decomposed:
    ``x * (c + i*s) = (re*c - im*s) + i*(re*s + im*c)`` in fp32, cast back to
    the input dtype.  Works for the ``freqs`` layouts passed at both call
    sites (GDN q/k and softmax q/k).
    """
    x = hidden_states.permute(0, 1, 3, 2).to(torch.float32).unflatten(3, (-1, 2))
    cos = freqs.real.to(torch.float32)
    sin = freqs.imag.to(torch.float32)
    re, im = x[..., 0], x[..., 1]
    out = torch.stack((re * cos - im * sin, re * sin + im * cos), dim=-1)
    return out.flatten(3, 4).permute(0, 1, 3, 2).to(hidden_states.dtype)


def _torch_cross_310p(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """310P replacement for ``torch.cross`` (decomposed into mul/sub).

    The CANN 9.1.0 310P op package ships no dynamic-shape kernel JSON for
    ``aclnnLinalgCross`` (EZ1013: "JSON configuration file of operator
    aclnnLinalgCross cannot be found" in the dynamic shape scenario), so the
    NPU call fails at kernel init.  The SANA-WM camera-geometry paths
    (``camera_control`` Plücker moments, ``ucpe.world_to_ray_mats`` basis
    build) call it on small per-request tensors; the vector product is
    decomposed into six muls and three subs (supported fp16/fp32 kernels).
    ``dim`` is the axis of the 3-vector; both SANA-WM call sites pass -1.
    """
    if a.shape[dim] != 3 or b.shape[dim] != 3:
        raise ValueError("torch.cross inputs must have a size-3 axis")
    ax, ay, az = torch.unbind(a, dim=dim)
    bx, by, bz = torch.unbind(b, dim=dim)
    return torch.stack(
        (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), dim=dim
    )


def _ucpe_rope_apply_real(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """310P real-arithmetic replacement for ``ucpe._apply_complex_rope``.

    Same decomposition as ``_rope_apply_real_forward`` with the ``inverse``
    (conjugate) branch sign-flipping ``sin``; fp32 MACs, cast back to the
    input dtype (mirrors the upstream ``x_out.to(hidden_states.dtype)``).
    """
    x = hidden_states.to(torch.float32).contiguous().unflatten(-1, (-1, 2))
    cos = freqs.real.to(torch.float32)
    sin = freqs.imag.to(torch.float32)
    if inverse:
        sin = -sin
    re, im = x[..., 0], x[..., 1]
    out = torch.stack((re * cos - im * sin, re * sin + im * cos), dim=-1)
    return out.flatten(-2, -1).to(hidden_states.dtype)


class _SanaWm310P(SanaWmPipeline):
    """SANA-WM pipeline specialized for the 310P NPU path.

    Component placement on 310P:

    * ``text_encoder`` (Gemma-2-2b): preprocessing-only, once per prompt
      branch.  Kept on CPU in FP16 (the Qwen3-VL fp16-CPU-forward precedent
      from the Qwen-Image-2.1 line); prompt embeds are moved back to the NPU
      in the runtime FP16 dtype.  ``VLLM_OMNI_310P_SANA_WM_ENCODER_ON_NPU=1``
      keeps it on-device for experiments.
    * DiT + VAE + latents: NPU, FP16 (forced by the runtime patch).
    * Scheduler: FP32, unchanged (device handling already in the base).
    """

    def __init__(self, *, od_config: Any = None, prefix: str = "") -> None:
        # 310P: model construction runs inside an NPU device context, where
        # the upstream RoPE table build (float64 + torch.polar) hits the
        # aclnnPolar DT_DOUBLE rejection.  Swap in the CPU/fp32 builder for
        # the duration of module construction only (restored in finally).
        _orig_build_1d_freq = SanaWmWanRotaryPosEmbed._build_1d_freq
        SanaWmWanRotaryPosEmbed._build_1d_freq = _build_1d_freq_310p
        try:
            super().__init__(od_config=od_config, prefix=prefix)
        finally:
            SanaWmWanRotaryPosEmbed._build_1d_freq = _orig_build_1d_freq
        # Runtime-rebuild shadow: ``SanaWmWanRotaryPosEmbed.forward`` rebuilds
        # the table for shapes beyond the cached 1024 (line 614-615 of the
        # upstream module) via ``self._build_1d_freq``; keep the CPU builder
        # alive on this instance so that path never touches the NPU kernel.
        rope = getattr(getattr(self, "transformer", None), "rope", None)
        if rope is not None:
            rope._build_1d_freq = lambda dim, positions: _build_1d_freq_310p(
                rope, dim, positions
            )
        self._install_310p_rope_apply()
        self._fixup_vae_tiling_310p()
        self._install_timing_hooks()
        if self.text_encoder is not None and not _encoder_on_npu():
            # Preprocessing-only module: keep it off the 310P NPU.
            self.text_encoder.to(device=_CPU_DEVICE, dtype=_RUNTIME_DTYPE)
            logger.info(
                "[310P] SANA-WM Gemma text encoder pinned to CPU (%s); embeds moved to %s in %s.",
                _CPU_DEVICE,
                self.device,
                _RUNTIME_DTYPE,
            )

    def _install_310p_rope_apply(self) -> None:
        """Shadow ``SanaWmSelfAttention._apply_rotary_emb`` on every attention
        instance of the constructed transformer with the fp32 real-arithmetic
        version (the upstream staticmethod casts to float64 and runs
        complex128 multiply, neither of which 310P implements).

        Instance attributes shadow the class staticmethod for instance-level
        calls only (both call sites are ``self._apply_rotary_emb(...)``), so
        other pipelines in the process are untouched.
        """
        transformer = getattr(self, "transformer", None)
        if transformer is None:
            return
        installed = 0
        for module in transformer.modules():
            if isinstance(module, SanaWmSelfAttention):
                module._apply_rotary_emb = _rope_apply_real_forward
                installed += 1
        if installed:
            logger.info(
                "[310P] SANA-WM: real-arithmetic RoPE apply installed on %d attention module(s).",
                installed,
            )

    def _install_timing_hooks(self) -> None:
        """Optional per-phase wall timing for the benchmark runner.

        The engine builds its model replica internally, so the runner cannot
        wrap the live instance from outside; when
        ``VLLM_OMNI_310P_SANA_WM_TIMING_FILE`` names a path, this instance
        appends one JSON line per call (``{"phase": ..., "s": ...}``) to it.
        DiT/VAE calls bracket an NPU sync so the wall includes kernel
        completion; the CPU-pinned encoder needs none.
        """
        import json
        import os
        import time

        path = os.environ.get(_TIMING_FILE_ENV)
        if not path:
            return

        def _sync() -> None:
            npu = getattr(torch, "npu", None)
            if npu is not None:
                try:
                    npu.synchronize()
                except Exception:  # noqa: BLE001 - timing must never break the run
                    pass

        def _record(phase: str, t0: float) -> None:
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"phase": phase, "s": round(time.perf_counter() - t0, 4)}) + "\n")
            except OSError:
                logger.warning("[310P] SANA-WM timing file write failed; phase timing lost.")

        def _wrap(module_obj: Any, attr: str, phase: str, sync: bool) -> None:
            orig = getattr(module_obj, attr)

            def timed(*args: Any, **kwargs: Any) -> Any:
                if sync:
                    _sync()
                t0 = time.perf_counter()
                out = orig(*args, **kwargs)
                if sync:
                    _sync()
                _record(phase, t0)
                return out

            setattr(module_obj, attr, timed)

        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            _wrap(transformer, "forward", "dit", sync=True)
        vae = getattr(self, "vae", None)
        if vae is not None:
            _wrap(vae, "decode", "vae_decode", sync=True)
        if self.text_encoder is not None:
            _wrap(self.text_encoder, "forward", "encoder", sync=False)

    def _fixup_vae_tiling_310p(self) -> None:
        """310P: disable the LTX-2 VAE's *vertical* tiling in tiled decode.

        ``aclnnReflectionPad3d`` requires ``pad < dim`` on 310P, while the
        LTX-2 tile loop (tile 16 latent rows, overlap 2 -> stride 14) produces
        a degenerate 1-row edge tile for the 480px tier (15 latent rows:
        rows 0..14 and 14..14); the conv-in reflect pad (pad 1) on that
        1-row axis is rejected by the kernel (EZ1001 "padding size should be
        less than the corresponding self dimention").  With
        ``tile_sample_min_height`` *and* ``tile_sample_stride_height`` raised
        above any tier height the H axis becomes a single full-height tile (the
        row loop steps by the stride; both must be huge), with no vertical
        seams to blend; the width tiling (16 of 26 latent columns) is
        preserved, so the memory profile stays well under the full decode.
        The pipeline only re-asserts ``use_tiling`` per request, so these
        construction-time values persist across requests.
        """
        vae = getattr(self, "vae", None)
        if vae is None or not hasattr(vae, "tile_sample_min_height"):
            return
        vae.tile_sample_min_height = _VAE_NO_VERTICAL_TILE_MIN_HEIGHT
        if hasattr(vae, "tile_sample_stride_height"):
            vae.tile_sample_stride_height = _VAE_NO_VERTICAL_TILE_MIN_HEIGHT
        logger.info(
            "[310P] SANA-WM VAE vertical tiling disabled (tile_sample_min_height=%d): "
            "aclnnReflectionPad3d rejects the 1-row edge tile at 480px height.",
            _VAE_NO_VERTICAL_TILE_MIN_HEIGHT,
        )

    def _encoder_device(self) -> torch.device:
        return self.device if _encoder_on_npu() else _CPU_DEVICE

    def _ensure_stage1_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        # 310P: never re-place the encoder onto the NPU.
        super()._ensure_stage1_text_encoder(device=self._encoder_device(), dtype=dtype)

    def _native_prompt_embeds(
        self,
        prompt: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # MIRROR of SanaWmPipeline._native_prompt_embeds (base) with one 310P
        # change: the token tensors route to the encoder's pinned device
        # (CPU) instead of the pipeline device; embeds still return on
        # ``device`` in ``dtype``.  Keep in sync if the base body changes.
        prompt_text = self._stage1_prompt_text(prompt)
        self._ensure_stage1_text_encoder(device=device, dtype=dtype)
        tokenizer = self.tokenizer
        if tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Sana-WM Stage-1 text encoder did not initialize.")
        chi_prompt = "\n".join(part for part in self.sana_wm_config.chi_prompt if part)
        model_max_length = int(self.sana_wm_config.model_max_length)
        max_length_all = model_max_length
        if chi_prompt:
            max_length_all = len(tokenizer.encode(chi_prompt)) + model_max_length - 2
        encoded = tokenizer(
            [prompt_text],
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self._encoder_device())
        hidden_states = self.text_encoder(**encoded).last_hidden_state
        attention_mask = encoded.attention_mask
        if chi_prompt:
            select_index = [0] + list(range(-model_max_length + 1, 0))
            hidden_states = hidden_states[:, select_index]
            attention_mask = attention_mask[:, select_index]
        if hidden_states.shape[-1] != SANA_WM_STAGE1_PROMPT_CHANNELS:
            raise ValueError(
                "Sana-WM Stage-1 Gemma hidden size mismatch: expected "
                f"{SANA_WM_STAGE1_PROMPT_CHANNELS}, got {hidden_states.shape[-1]}."
            )
        prompt_attention_mask = attention_mask.to(device=device, dtype=torch.float32)
        if bool(prompt_attention_mask.all()):
            prompt_attention_mask = None
        self._last_prompt_attention_mask = prompt_attention_mask
        return hidden_states.to(device=device, dtype=dtype)


def apply_diffusion_310p_patches(od_config) -> None:
    """Apply the SANA-WM 310P diffusion runtime patches (idempotent)."""
    global _PATCHED

    # The registry dispatcher already ran the shared 310P attention patch and
    # the optional warmup skip before this module; nothing to do there.
    if getattr(od_config, "dtype", None) is not _RUNTIME_DTYPE:
        logger.info("[310P] SANA-WM: forcing diffusion runtime dtype to %s.", _RUNTIME_DTYPE)
        od_config.dtype = _RUNTIME_DTYPE

    # VAE tiling is re-asserted per-request by the pipeline itself; the config
    # flags are belt-and-braces (registry.py copies them onto model.vae).
    od_config.vae_use_tiling = True
    od_config.vae_use_slicing = True

    if getattr(od_config, "diffusion_kv_cache_dtype", None) not in (None, "auto"):
        logger.warning(
            "[310P] SANA-WM: ignoring diffusion_kv_cache_dtype=%r (MindIE-only path).",
            od_config.diffusion_kv_cache_dtype,
        )
        od_config.diffusion_kv_cache_dtype = "auto"

    if _PATCHED:
        return

    # The UCPE camera path (ucpe.py, SANA-WM-exclusive module) applies the
    # same complex-RoPE in float64 (`_apply_complex_rope`, incl. the inverse
    # conjugate branch).  Swap the module-level function for the fp32
    # real-arithmetic version; `_prepare_ray_apply_fns` resolves the global
    # by name at call time, so every `partial(...)` picks up the replacement.
    _sana_wm_ucpe._apply_complex_rope = _ucpe_rope_apply_real

    # aclnnLinalgCross has no 310P dynamic-shape kernel (EZ1013); the camera
    # geometry (Plücker moments + ray basis) calls torch.cross per request.
    # Swap it process-wide for the mul/sub decomposition (same pattern as
    # the attention.py SDPA swap; only SANA-WM loads in this process).
    global _ORIGINAL_TORCH_CROSS
    if _ORIGINAL_TORCH_CROSS is None:
        _ORIGINAL_TORCH_CROSS = torch.cross  # save before the swap
    torch.cross = _torch_cross_310p  # idempotent re-assignment

    from vllm_omni.diffusion.registry import register_diffusion_model

    # Override the built-in pipeline with the 310P-specialized class under the
    # same model_arch.  pre_process_func_name stays untouched (None keeps the
    # existing entry): get_sana_wm_pre_process_func is the canonical entry.
    register_diffusion_model(
        _SANA_WM_ARCH,
        module_name=__name__,
        class_name="_SanaWm310P",
    )
    _PATCHED = True
    logger.info("[310P] Registered SANA-WM 310P pipeline override for %s.", _SANA_WM_ARCH)


# Backwards-compatible alias; the registry dispatch (and PR-facing callers)
# use the canonical apply_diffusion_310p_patches name.
apply_sana_wm_310p_patches = apply_diffusion_310p_patches
