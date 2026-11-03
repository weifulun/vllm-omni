# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Component loaders for the Wan-Dancer 310P port (task-40).

Loads the Wan-AI/Wan-Dancer-14B repo (non-diffusers layout) into the ported
torch modules:

* twin DiT checkpoints (``global_model.safetensors`` / ``local_model.safetensors``)
  — BF16 on disk, cast to FP16 on load (310P FP16-only);
* ``Wan2.1_VAE.pth`` — diffsynth Wan video VAE (ported verbatim);
* ``models_t5_umt5-xxl-enc-bf16.pth`` — UMT5-XXL encoder (transformers);
* ``models_clip_xlm-roberta-large-vit-huge-14.pth`` — OpenCLIP ViT-H/14 visual
  tower (transformers ``CLIPVisionModel``);
* music feature extraction (official ``get_music_base_feature``, librosa,
  FP32 numpy per ENV-PRECHECK P1-11).

All placement is explicit (no implicit ``.to()`` drift): the DiT takes a
per-block device list (dual-die), encoders stay CPU-pinned by default.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import (
    DancerDitConfig,
    DancerTransformer3DModel,
)
from vllm_omni.diffusion.models.wan2_2.wan_dancer_vae import WanVideoVAE

__all__ = [
    "load_dancer_dit",
    "load_wan_vae",
    "load_t5_encoder",
    "load_clip_visual",
    "get_music_base_feature",
    "crop_and_resize",
    "preprocess_image",
    "preprocess_video",
]

_MUSIC_DTYPE = torch.float16


# --------------------------------------------------------------------------- #
# DiT
# --------------------------------------------------------------------------- #
def _map_dit_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """diffsynth naming -> ported module naming.

    The ported modules use the official submodule names, so the mapping is
    identity except for the two music encoder layers, whose official
    ``nn.MultiheadAttention`` fused ``in_proj_{weight,bias}`` is split into the
    ported separate q/k/v Linears (first third = q, second = k, third = v).
    """
    out: dict[str, torch.Tensor] = {}
    latent = 256  # music_latent (checkpoint geometry)
    for key, value in sd.items():
        if key.startswith("music_encoder."):
            parts = key.split(".")
            idx = parts[1]
            rest = parts[2]
            if rest.startswith("self_attn.in_proj_weight"):
                w = value  # (3*latent, latent)
                out[f"music_encoder.{idx}.q.weight"] = w[:latent]
                out[f"music_encoder.{idx}.k.weight"] = w[latent : 2 * latent]
                out[f"music_encoder.{idx}.v.weight"] = w[2 * latent :]
                continue
            if rest.startswith("self_attn.in_proj_bias"):
                b = value
                out[f"music_encoder.{idx}.q.bias"] = b[:latent]
                out[f"music_encoder.{idx}.k.bias"] = b[latent : 2 * latent]
                out[f"music_encoder.{idx}.v.bias"] = b[2 * latent :]
                continue
            if rest.startswith("self_attn."):
                # e.g. "self_attn.out_proj.weight" -> "music_encoder.{i}.out_proj.weight"
                out[f"music_encoder.{idx}.{rest[len('self_attn.'):]}"] = value
                continue
            out[key] = value
            continue
        out[key] = value
    return out


def load_dancer_dit(
    checkpoint_path: str,
    dtype: torch.dtype = _MUSIC_DTYPE,
    block_devices: torch.device | list[torch.device] | None = None,
    device: torch.device | None = None,
) -> DancerTransformer3DModel:
    """Load one twin checkpoint (global or local) with an explicit placement.

    ``block_devices``: a single device (all blocks) or a 40-entry per-block
    list for the dual-die placement.  ``device`` defaults the buffers and any
    blocks not covered by ``block_devices``.
    """
    from safetensors.torch import load_file

    if block_devices is not None and not isinstance(block_devices, (list, tuple)):
        block_devices = [block_devices] * DancerDitConfig().depth
    model = DancerTransformer3DModel(block_devices=block_devices)
    sd = _map_dit_state_dict(load_file(checkpoint_path))
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"dit mapping mismatch: {missing[:3]} {unexpected[:3]}"
    if device is not None:
        model = model.to(device)
    # Cast to the on-device dtype (FP16 on 310P; BF16 allowed off-device).
    model = model.to(dtype=dtype)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #
def load_wan_vae(checkpoint_path: str, dtype: torch.dtype = _MUSIC_DTYPE, device: torch.device | None = None) -> WanVideoVAE:
    vae = WanVideoVAE(z_dim=16)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not any(k.startswith("model.") for k in sd):
        sd = {f"model.{k}": v for k, v in sd.items()}
    vae.load_state_dict(sd, strict=True)
    vae = vae.to(dtype=dtype)
    if device is not None:
        vae = vae.to(device)
    vae.eval()
    return vae


# --------------------------------------------------------------------------- #
# T5 (UMT5-XXL encoder)
# --------------------------------------------------------------------------- #
_T5_CONFIG = dict(
    d_model=4096,
    d_ff=18432,
    num_layers=24,
    num_heads=16,
    d_kv=64,
    dropout=0.0,
    layer_norm_epsilon=1e-6,
    vocab_size=256384,
    use_relative_position=True,
    relative_attention_num_buckets=32,
)


def load_t5_encoder(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the diffsynth UMT5-XXL encoder .pth into transformers UMT5.

    The diffsynth checkpoint uses the HF/transformers UMT5 key layout
    (``encoder.block.N.layer.M.*``); ``shared`` (embedding) maps to
    ``encoder.embed_tokens``.  Any residual name drift is reported loudly.
    """
    from transformers import UMT5EncoderModel

    model = UMT5EncoderModel(_T5_CONFIG)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mapped: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key == "shared.weight":
            mapped["shared.weight"] = value
        elif key.startswith("encoder."):
            mapped[key] = value
        else:
            raise KeyError(f"unmapped T5 checkpoint key: {key}")
    # UMT5EncoderModel keeps `shared` as the embedding (tie_word_embeddings).
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    fatal = [m for m in missing if "embed_tokens" not in m and "shared" not in m]
    if fatal or unexpected:
        raise RuntimeError(f"T5 mapping mismatch: missing={fatal[:4]} unexpected={unexpected[:4]}")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# CLIP visual tower (OpenCLIP ViT-H/14 -> transformers CLIPVisionModel)
# --------------------------------------------------------------------------- #
_CLIP_CONFIG = dict(
    hidden_size=1280,
    num_hidden_layers=32,
    num_attention_heads=16,
    intermediate_size=5120,
    image_size=224,
    patch_size=14,
    projection_dim=1024,
)


def load_clip_visual(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    from transformers import CLIPVisionModel

    model = CLIPVisionModel(_CLIP_CONFIG)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    visual = {k[len("visual.") :]: v for k, v in sd.items() if k.startswith("visual.")}
    if not visual:
        raise KeyError("OpenCLIP checkpoint has no visual.* keys")
    mapped: dict[str, torch.Tensor] = {}
    for key, value in visual.items():
        if key == "class_embedding":
            mapped["vision_model.embeddings.class_token"] = value
        elif key == "position_embedding":
            mapped["vision_model.embeddings.position_embedding"] = value
        elif key == "patch_embedding.weight":
            mapped["vision_model.embeddings.patch_embedding.weight"] = value
        elif key == "ln_final.weight":
            mapped["vision_model.post_layernorm.weight"] = value
        elif key == "ln_final.bias":
            mapped["vision_model.post_layernorm.bias"] = value
        elif key == "proj":
            mapped["visual_projection.weight"] = value
        elif ".attn.qkv.weight" in key or key.endswith(".attn.qkv.weight"):
            blk, rest = key.split(".attn.qkv.weight")
            n = int(blk)
            w = value  # (3*1280, 1280)
            dim = w.shape[0] // 3
            mapped[f"vision_model.layers.{n}.attention.attention.query.weight"] = w[:dim]
            mapped[f"vision_model.layers.{n}.attention.attention.key.weight"] = w[dim : 2 * dim]
            mapped[f"vision_model.layers.{n}.attention.attention.value.weight"] = w[2 * dim :]
        elif key.endswith(".attn.qkv.bias"):
            blk = key.split(".attn.qkv.bias")[0]
            n = int(blk)
            b = value
            dim = b.shape[0] // 3
            mapped[f"vision_model.layers.{n}.attention.attention.query.bias"] = b[:dim]
            mapped[f"vision_model.layers.{n}.attention.attention.key.bias"] = b[dim : 2 * dim]
            mapped[f"vision_model.layers.{n}.attention.attention.value.bias"] = b[2 * dim :]
        elif key.endswith(".attn.out_proj.weight"):
            n = int(key.split(".attn.out_proj.weight")[0])
            mapped[f"vision_model.layers.{n}.attention.out_proj.weight"] = value
        elif key.endswith(".attn.out_proj.bias"):
            n = int(key.split(".attn.out_proj.bias")[0])
            mapped[f"vision_model.layers.{n}.attention.out_proj.bias"] = value
        elif key.endswith(".mlp.fc1.weight"):
            n = int(key.split(".mlp.fc1.weight")[0])
            mapped[f"vision_model.layers.{n}.mlp.fc1.weight"] = value
        elif key.endswith(".mlp.fc1.bias"):
            n = int(key.split(".mlp.fc1.bias")[0])
            mapped[f"vision_model.layers.{n}.mlp.fc1.bias"] = value
        elif key.endswith(".mlp.fc2.weight"):
            n = int(key.split(".mlp.fc2.weight")[0])
            mapped[f"vision_model.layers.{n}.mlp.fc2.weight"] = value
        elif key.endswith(".mlp.fc2.bias"):
            n = int(key.split(".mlp.fc2.bias")[0])
            mapped[f"vision_model.layers.{n}.mlp.fc2.bias"] = value
        elif key.endswith(".ln1.weight"):
            n = int(key.split(".ln1.weight")[0])
            mapped[f"vision_model.layers.{n}.layernorm1.weight"] = value
        elif key.endswith(".ln1.bias"):
            n = int(key.split(".ln1.bias")[0])
            mapped[f"vision_model.layers.{n}.layernorm1.bias"] = value
        elif key.endswith(".ln2.weight"):
            n = int(key.split(".ln2.weight")[0])
            mapped[f"vision_model.layers.{n}.layernorm2.weight"] = value
        elif key.endswith(".ln2.bias"):
            n = int(key.split(".ln2.bias")[0])
            mapped[f"vision_model.layers.{n}.layernorm2.bias"] = value
        else:
            raise KeyError(f"unmapped CLIP visual key: {key}")
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    fatal = [m for m in missing if "layernorm" not in m.split(".")[-1]]
    if fatal or unexpected:
        raise RuntimeError(f"CLIP mapping mismatch: missing={fatal[:4]} unexpected={unexpected[:4]}")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# music features (official get_music_base_feature, verbatim semantics)
# --------------------------------------------------------------------------- #
def get_music_base_feature(music_path: str, output_path: str, fps: int = 30) -> torch.Tensor:
    """Official librosa feature (35-dim, ~30 fps rows).  FP32 numpy, OMP>=16 caller duty."""
    import numpy as np
    import librosa

    hop_length = 512
    sr = fps * hop_length  # 15360
    data, _ = librosa.load(music_path, sr=sr)
    sr = 22050  # official quirk: features computed with sr=22050 on 15360 Hz data
    envelope = librosa.onset.onset_strength(y=data, sr=sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=data, sr=sr, hop_length=hop_length, n_chroma=12).T
    peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=sr, hop_length=hop_length)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0
    start_bpm = librosa.beat.tempo(y=librosa.load(music_path)[0])[0]
    _, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=sr, hop_length=hop_length, start_bpm=start_bpm, tightness=100
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0
    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]], axis=-1
    )
    assert audio_feature.shape[1] == 35, f"music feature dim {audio_feature.shape[1]} != 35"
    np.save(output_path, audio_feature)
    return torch.from_numpy(audio_feature.astype(np.float32))


# --------------------------------------------------------------------------- #
# image / keyframe helpers (official semantics)
# --------------------------------------------------------------------------- #
def crop_and_resize(image, target_width: int, target_height: int):
    """Official crop_and_resize: center-fit resize + gray(127) pad to target."""
    from PIL import Image
    import numpy as np
    import torchvision.transforms.functional as TF

    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resized_height = round(height * scale)
    resized_width = round(width * scale)
    image = TF.resize(image, (resized_height, resized_width), interpolation=TF.InterpolationMode.BILINEAR)
    target_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 127
    tl_x = (target_width - resized_width) // 2
    tl_y = (target_height - resized_height) // 2
    br_x = tl_x + resized_width
    br_y = tl_y + resized_height
    target_image[tl_y:br_y, tl_x:br_x, :] = np.array(image, dtype=np.uint8)
    return Image.fromarray(target_image), (tl_x, tl_y, br_x, br_y)


def preprocess_image(image, torch_dtype: torch.dtype, device, pattern: str = "B C H W") -> torch.Tensor:
    """Official preprocess_image: HWC uint8 -> [-1, 1] tensor."""
    import numpy as np

    image = torch.Tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=torch_dtype, device=device)
    image = image * (2.0 / 255.0) - 1.0
    if pattern == "B C H W":
        return image.unsqueeze(0).permute(0, 3, 1, 2)
    return image.permute(2, 0, 1)


def preprocess_video(video, torch_dtype: torch.dtype, device) -> torch.Tensor:
    """List[PIL] -> (1, 3, T, H, W) in [-1, 1]."""
    frames = [preprocess_image(im, torch_dtype=torch_dtype, device=device, pattern="C H W") for im in video]
    return torch.stack(frames, dim=1).unsqueeze(0)
