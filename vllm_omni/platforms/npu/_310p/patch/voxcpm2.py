# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch VoxCPM2 for the 310P NPU path.

Port template: ``qwen3_tts.py`` in this directory (the Qwen3-TTS 310P patch).
What the template establishes for 310P is applied here with the VoxCPM2
equivalents:

* **FP16 forced.**  The 310P line is an FP16 line.  VoxCPM2 ships
  ``deploy/voxcpm2.yaml`` with ``dtype: bfloat16`` (correct for 910B / H-series);
  on 310P the deployment must run ``deploy/voxcpm2_310p.yaml`` (or
  ``--dtype float16``) so the AR side loads FP16, and the tts side-path
  (LocDiT estimator / AudioVAE / projections) is cast to FP16 regardless of the
  native checkpoint dtype (``VoxCPM.from_pretrained`` keeps it native).
* **LocDiT NPUGraph -> eager.**  On 910 the NPU adapter wraps
  ``tts.feat_decoder.estimator.forward`` in ``NPUExactGraphRunner``
  (``platforms/npu/models/voxcpm2_talker.py``).  On 310P the estimator's
  portable-SDPA + RoPE path is not a capturable ACL graph target and a failed
  capture kills the stage process (see ``graph_tools.NPUExactGraphRunner.run``).
  So on 310P the NPUGraph wrapper is neutralized and LocDiT stays eager.  The
  per-request CFM noise (``normal_``) lives *outside* the estimator forward in
  either case, so sampling semantics are unchanged.
* **Preprocessing pinned CPU (resample only).**  Unlike Qwen3-TTS (dedicated
  Mimi encoder module), VoxCPM2's reference/prompt-audio encoder *is* the
  AudioVAE, which also decodes on the NPU — it cannot be pinned to CPU.  The
  only device-sensitive preprocessing step is ``torchaudio.Resample`` inside
  ``_encode_raw_audio``; on 310P it is forced onto the CPU before the H2D copy.
* **VAE decode latent cast.**  The talker computes the AR-side latent in FP32
  (numerically-stable projection), then hands it to ``audio_vae.decode``.  The
  910 line never sees the mismatch because tts modules run uniformly in BF16;
  once this patch casts the AudioVAE to FP16, the FP32 latent hits FP16 conv
  biases (``Input type (float) and bias type (c10::Half) should be the
  same``).  The 310P talker subclass therefore re-casts the latent to the
  VAE parameter dtype at the ``_run_vae_decode`` boundary before calling
  super.

Device-verification TODOs (kept OFF by default, see the ``_LOC_DIT_*`` flags):
the Qwen3-TTS template's 310P flash-attention recipe
(``_npu_flash_attention`` + FRACTAL_NZ mask + fused ``npu_apply_rotary_pos_emb``)
is the expected speed-up for the 10-step CFM loop, but it is not enabled until
qualified on real hardware; eager portable SDPA is the safe baseline.

Target revision: vllm-omni v0.32.0 (tag 5917730a4b) — the VoxCPM2 talker there
precedes the main-branch ``_install_dtype_aware_encode`` wrapper, so the
310P encode variant below pins the VAE input dtype explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_RUNTIME_DTYPE = torch.float16
_ACL_FORMAT_FRACTAL_Z = 4

# Optimization toggles: OFF until qualified on a real 310P card.
_LOC_DIT_FLASH_ATTN_310P = False

_PATCHED = False


# ===================================================================
#  dtype helpers
# ===================================================================


def _cast_module_to_dtype(module: nn.Module, dtype: torch.dtype) -> int:
    """Cast float parameters/buffers of ``module`` to ``dtype`` in place.

    Complex buffers (e.g. RoPE ``freqs_cis``) and integer buffers are left
    untouched: ``nn.Module.to(dtype)`` would silently demote complex64 to
    real float and corrupt the cache.
    """
    converted = 0
    with torch.no_grad():
        for child in module.modules():
            for name, param in list(child._parameters.items()):
                if param is None or not param.is_floating_point():
                    continue
                child._parameters[name] = nn.Parameter(
                    param.data.to(dtype),
                    requires_grad=param.requires_grad,
                )
                converted += 1
            for name, buf in list(child._buffers.items()):
                if buf is None or not buf.is_floating_point() or buf.is_complex():
                    continue
                child._buffers[name] = buf.to(dtype)
                converted += 1
    return converted


def _pack_npu_weights(module: nn.Module) -> tuple[int, int]:
    """Pack the tts side-path weights for 310P aclnn kernels.

    Mirrors the 910 Qwen3-TTS Code2Wav weight preparation
    (``platforms/npu/models/qwen3_tts.py::_prepare_npu_decoder_weights``):
    Linear weights go through ``maybe_trans_nz`` and group-1 conv1d /
    convtranspose1d weights are cast to the FRACTAL_Z layout (``Conv`` kernels
    on 310P require it; the 910 path's A5 exemption does not apply to 310P).
    The tts modules live outside vLLM's weight-loading machinery, so the
    engine's ``allow_internal_format`` pass never touches them.
    """
    from vllm_ascend.utils import maybe_trans_nz

    import torch_npu

    linear_count = 0
    conv_count = 0
    with torch.no_grad():
        for module_ in module.modules():
            if isinstance(module_, nn.Linear):
                module_.weight.data = maybe_trans_nz(module_.weight.data)
                linear_count += 1
            elif isinstance(module_, (nn.Conv1d, nn.ConvTranspose1d)) and module_.groups == 1:
                module_.weight.data = torch_npu.npu_format_cast(
                    module_.weight.data.contiguous(),
                    _ACL_FORMAT_FRACTAL_Z,
                )
                conv_count += 1
    return linear_count, conv_count


def _pack_fused_buffers(model: nn.Module) -> int:
    """Pack post-load ND buffers created by the LocDiT fusion rewrites.

    ``_install_locdit_fused_qkv`` registers a ``_voxcpm2_fused_qkv_weight``
    buffer after weight loading, so the engine's ND->NZ pass misses it.
    """
    from vllm_ascend.utils import maybe_trans_nz

    packed = 0
    with torch.no_grad():
        for module_ in model.modules():
            for name, buf in list(module_._buffers.items()):
                if name != "_voxcpm2_fused_qkv_weight" or buf is None:
                    continue
                module_._buffers[name] = maybe_trans_nz(buf)
                packed += 1
    return packed


# ===================================================================
#  310P encode path (resample on CPU)
# ===================================================================


def _encode_raw_audio_310p(
    tts: nn.Module,
    samples: "list[float] | torch.Tensor | bytes",
    sr: int,
    padding_mode: str = "right",
    keep_on_device: bool = False,
) -> torch.Tensor:
    """310P variant of ``voxcpm2_talker._encode_raw_audio``.

    Keeps the upstream semantics, with two changes:

    1. When the incoming sample rate differs from the VAE encoder rate,
       resampling happens on the CPU *before* any device transfer instead of
       running ``torchaudio.Resample`` on the NPU.
    2. The VAE input is explicitly cast to the VAE parameter dtype before
       ``encode`` (upstream relies on the native VoxCPM package handling the
       fp32 input; with the tts side cast to FP16 by this patch, the input
       dtype must follow the parameters explicitly).

    Keep this in sync with upstream ``_encode_raw_audio``.
    """
    import numpy as np

    if isinstance(samples, bytes):
        if len(samples) % np.dtype(np.float32).itemsize:
            raise ValueError("VoxCPM2 audio bytes must contain float32 samples")
        audio = torch.from_numpy(np.frombuffer(samples, dtype=np.float32).copy())
    elif isinstance(samples, list):
        audio = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    else:
        audio = samples.float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    vae_device = next(tts.audio_vae.parameters()).device
    encode_sr = tts._encode_sample_rate
    if sr != encode_sr:
        # CPU resample first: the 310P NPU path must not run torchaudio
        # Resample on-device.
        import torchaudio

        audio = torchaudio.transforms.Resample(sr, encode_sr)(audio.cpu())
    if keep_on_device:
        audio = audio.to(vae_device, non_blocking=True)

    patch_len = tts.patch_size * tts.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
        audio = torch.nn.functional.pad(audio, pad)

    vae_device = next(tts.audio_vae.parameters()).device
    vae_dtype = next(tts.audio_vae.parameters()).dtype
    feat = tts.audio_vae.encode(audio.to(device=vae_device, dtype=vae_dtype), encode_sr)
    if not keep_on_device:
        feat = feat.cpu()
    return feat.view(tts.audio_vae.latent_dim, -1, tts.patch_size).permute(1, 2, 0)


# ===================================================================
#  310P talker subclass
# ===================================================================


def _force_310p_fp16(self) -> None:
    """Force the tts side-path to FP16 after ``__init__``.

    On the v0.32.0 line the tts modules keep the native checkpoint dtype
    (``VoxCPM.from_pretrained`` -> ``.to(device)`` only); nothing in the
    engine casts them, so cast them here.  ``_side_dtype`` is re-derived
    afterwards because CFM buffers / condition tensors follow it.
    """
    deploy_dtype = getattr(self.vllm_config.model_config, "dtype", None)
    if deploy_dtype is not None and torch.float16 != deploy_dtype:
        logger.warning(
            "VoxCPM2 on 310P runs best with --dtype float16 (deploy/voxcpm2_310p.yaml); "
            "deployment dtype is %s: the AR side loads %s while the tts side is forced to "
            "FP16 by this patch.",
            deploy_dtype,
            deploy_dtype,
        )
    converted = _cast_module_to_dtype(self._tts, _RUNTIME_DTYPE)
    self._side_dtype = _RUNTIME_DTYPE
    logger.info("VoxCPM2 310P: tts side cast to FP16 (%d tensors)", converted)


def _neutralize_loc_dit_npugraph() -> None:
    """Stop the 910 LocDiT NPUGraph wrapper from engaging on 310P.

    ``apply_voxcpm2_talker_patch`` (installed at ``set_device``) wraps
    ``VoxCPM2TalkerForConditionalGeneration.__init__`` so that every talker
    ends construction with ``setup_voxcpm2_loc_dit_npu_graph(self)``.  The
    adapter resolves that name through its module globals at call time, so
    replacing it with a logged no-op is enough to keep LocDiT eager.
    """
    from vllm_omni.platforms.npu.models import voxcpm2_talker as npu_adapter

    def _setup_eager(model) -> None:
        del model
        logger.info(
            "VoxCPM2 310P: LocDiT NPUGraph neutralized; estimator stays eager "
            "(310P random-sampling / portable-SDPA path is not graph-capturable)"
        )

    npu_adapter.setup_voxcpm2_loc_dit_npu_graph = _setup_eager  # type: ignore[assignment]


def _wrap_lazy_optimization_setup(self) -> None:
    """Pack the LocDiT fused weights once the lazy fusion rewrites ran."""
    if getattr(self, "_voxcpm2_310p_setup_wrapped", False):
        return
    original = self._setup_execution_optimizations

    def _setup_then_pack():
        original()
        _pack_fused_buffers(self._tts)

    self._setup_execution_optimizations = _setup_then_pack
    self._voxcpm2_310p_setup_wrapped = True


def apply_voxcpm2_310p_patches() -> None:
    """Install the 310P VoxCPM2 runtime patch (idempotent).

    Called from ``vllm_omni.platforms.npu._310p.patch.apply_model_patches``
    with ``model_arch == "VoxCPM2TalkerForConditionalGeneration"`` inside the
    worker's ``load_model`` (the MRV1 NPU runner on the v0.32.0 line — the
    NPU AR worker has a single runner class there), i.e. after ``set_device``
    (which installed the 910 adapter) and before the model class is resolved
    from the registry.
    """
    global _PATCHED
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as vox_module

    _neutralize_loc_dit_npugraph()
    vox_module._encode_raw_audio = _encode_raw_audio_310p  # type: ignore[assignment]

    base_cls = vox_module.VoxCPM2TalkerForConditionalGeneration

    class _VoxCPM2Talker310P(base_cls):  # type: ignore[misc,valid-type]
        """VoxCPM2 talker specialized for the 310P NPU path (FP16, LocDiT eager)."""

        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            # super().__init__ runs the 910 adapter's patched __init__
            # (original init + neutralized LocDiT NPUGraph setup).
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            _force_310p_fp16(self)
            linear_count, conv_count = _pack_npu_weights(self._tts)
            logger.info(
                "VoxCPM2 310P: packed tts weights (linear=%d conv=%d)",
                linear_count,
                conv_count,
            )
            _wrap_lazy_optimization_setup(self)

        def _run_vae_decode(self, feat: torch.Tensor) -> torch.Tensor:
            """Cast the latent to the AudioVAE parameter dtype before decode.

            After ``_force_310p_fp16`` the VAE convolutions are FP16, but the
            latent produced by the AR-side projection is FP32; the 910 line
            never meets this (uniform BF16).  Without the cast the first
            decode step dies in ``F.conv1d`` with an input/bias dtype
            mismatch.  The eager and CUDA/NPU graph branches both go through
            this boundary, so casting here covers both.
            """
            vae = self.tts.audio_vae
            ref = next(vae.parameters(), None)
            if ref is not None and feat.dtype != ref.dtype:
                feat = feat.to(ref.dtype)
            return super()._run_vae_decode(feat)

    vox_module.VoxCPM2TalkerForConditionalGeneration = _VoxCPM2Talker310P  # type: ignore[assignment]

    _PATCHED = True
    logger.info("Applied 310P patch for VoxCPM2 (FP16 forced, LocDiT eager, resample on CPU)")
