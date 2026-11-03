# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WanDancerPipeline — two-stage Wan-Dancer 310P inference (task-40).

Standalone (runner-driven) pipeline implementing the official two-stage flow:

* **Global stage** — whole-song choreography: 149 frames @ 720x1280 at
  ``input_fps = 30 / round(N_music / 149)``, 48 steps, cfg 5, sigma_shift 5,
  ``patchify_global``/``head_global`` (unimodel), music injection from the
  full-song 35-dim feature.
* **Local stage** — per 5 s music segment: 149 frames @ 30 fps, 24 steps,
  cfg 5, sigma_shift 5, keyframes anchored to the global video
  (``process_global_video_firstlastframe``), refimage = input image (CLIP
  ``img`` branch), ``enable_skip_layer`` (block 9 skipped on the negative
  pass), ``enable_dynamicfps=True`` (inactive at fps==30).

Component placement is explicit: DiT blocks follow the per-block device list
(dual-die on 310P), VAE is FP16 on NPU, T5 stays CPU-pinned BF16 and CLIP
CPU-pinned FP32 (ENCODERS-CPU-PIN-DEFAULT per qwenimg21/zimage precedent).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import build_wan_scheduler
from vllm_omni.diffusion.models.wan2_2.wan_dancer_components import (
    crop_and_resize,
    get_music_base_feature,
    load_clip_visual,
    load_dancer_dit,
    load_t5_encoder,
    load_wan_vae,
    preprocess_video,
)
from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import DancerTransformer3DModel

__all__ = ["WanDancerPipeline", "WAN_DANCER_NEGATIVE_PROMPT", "process_global_video_firstlastframe"]

WAN_DANCER_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

TOKEN_MAX_LEN = 512


def _npu_device(index: int) -> torch.device:
    return torch.device(f"npu:{index}")


class WanDancerPipeline:
    """Two-stage Wan-Dancer generation on explicit device placement."""

    def __init__(
        self,
        repo_dir: str,
        dit_global: DancerTransformer3DModel,
        dit_local: DancerTransformer3DModel,
        vae,
        t5_encoder,
        clip_visual,
        tokenizer,
        device: torch.device,
        dit_dtype: torch.dtype = torch.float16,
    ):
        self.repo_dir = repo_dir
        self.dit_global = dit_global
        self.dit_local = dit_local
        self.vae = vae
        self.t5_encoder = t5_encoder
        self.clip_visual = clip_visual
        self.tokenizer = tokenizer
        self.device = device
        self.dit_dtype = dit_dtype
        self.scheduler = build_wan_scheduler("euler", 5.0)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_repo(
        cls,
        repo_dir: str,
        global_dit: str | None = None,
        local_dit: str | None = None,
        vae_path: str | None = None,
        t5_path: str | None = None,
        clip_path: str | None = None,
        block_devices_global: torch.device | list[torch.device] | None = None,
        block_devices_local: torch.device | list[torch.device] | None = None,
        vae_device: torch.device | None = None,
        dit_dtype: torch.dtype = torch.float16,
        load_local: bool = True,
    ) -> "WanDancerPipeline":
        from transformers import T5TokenizerFast

        repo_dir = os.path.abspath(repo_dir)
        global_dit = global_dit or os.path.join(repo_dir, "global_model.safetensors")
        local_dit = local_dit or os.path.join(repo_dir, "local_model.safetensors")
        vae_path = vae_path or os.path.join(repo_dir, "Wan2.1_VAE.pth")
        t5_path = t5_path or os.path.join(repo_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        clip_path = clip_path or os.path.join(repo_dir, "models_clip_xlm-roberta-large-vit-huge-14.pth")

        def _dev(dev):
            if isinstance(dev, (list, tuple)):
                return [torch.device(d) for d in dev]
            return dev

        print(f"[wandancer] loading global DiT <- {global_dit}", flush=True)
        dit_global = load_dancer_dit(global_dit, dtype=dit_dtype, block_devices=_dev(block_devices_global))
        print(f"[wandancer] DiT global loaded: {sum(p.numel() for p in dit_global.parameters()) / 1e9:.3f}B", flush=True)

        dit_local = None
        if load_local and os.path.exists(local_dit):
            print(f"[wandancer] loading local DiT <- {local_dit}", flush=True)
            dit_local = load_dancer_dit(local_dit, dtype=dit_dtype, block_devices=_dev(block_devices_local))
            print(f"[wandancer] DiT local loaded: {sum(p.numel() for p in dit_local.parameters()) / 1e9:.3f}B", flush=True)

        print(f"[wandancer] loading VAE <- {vae_path}", flush=True)
        vae = load_wan_vae(vae_path, dtype=dit_dtype, device=_dev(vae_device))

        print(f"[wandancer] loading T5 <- {t5_path} (CPU bf16)", flush=True)
        t5 = load_t5_encoder(t5_path)
        print(f"[wandancer] loading CLIP visual <- {clip_path} (CPU fp32)", flush=True)
        clip = load_clip_visual(clip_path)

        tokenizer = T5TokenizerFast.from_pretrained(os.path.join(repo_dir, "google", "umt5-xxl"))

        device = _dev(block_devices_global)[0] if block_devices_global else torch.device("cpu")
        return cls(
            repo_dir=repo_dir,
            dit_global=dit_global,
            dit_local=dit_local,
            vae=vae,
            t5_encoder=t5,
            clip_visual=clip,
            tokenizer=tokenizer,
            device=device,
            dit_dtype=dit_dtype,
        )

    # ------------------------------------------------------------------ #
    # encoders (CPU-pinned)
    # ------------------------------------------------------------------ #
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(
            prompt, max_length=TOKEN_MAX_LEN, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        with torch.no_grad():
            emb = self.t5_encoder(input_ids=ids).last_hidden_state  # (1, 512, 4096) bf16 CPU
        return emb

    def encode_image(self, image, size: tuple[int, int] | None = None) -> torch.Tensor:
        """PIL image -> (1, 257, 1280) CLIP tokens (CPU).

        ``size``: official pipelines ``.resize((width, height))`` the input
        before ``preprocess_image`` (no-op when already target-sized).
        """
        from vllm_omni.diffusion.models.wan2_2.wan_dancer_components import preprocess_image

        if size is not None and image.size != tuple(size):
            image = image.resize(tuple(size))
        x = preprocess_image(image, torch_dtype=torch.float32, device="cpu", pattern="C H W")
        with torch.no_grad():
            out = self.clip_visual(pixel_values=x.unsqueeze(0)).last_hidden_state  # (1, 257, 1280)
        return out

    # ------------------------------------------------------------------ #
    # keyframe latents (official KeyframesLatents unit)
    # ------------------------------------------------------------------ #
    def build_keyframe_y(self, keyframes, keyframes_mask, num_frames: int, height: int, width: int):
        """keyframes: list[PIL]; mask: torch int32 (num_frames,).

        Returns (clip_feature (1,257,1280) CPU, y (1,20,f_lat,h/8,w/8) NPU fp16).
        """
        first = keyframes[0]
        clip_feature = self.encode_image(first)  # CPU fp32
        msk = torch.zeros(1, num_frames, height // 8, width // 8)
        msk[:, keyframes_mask == 1, :, :] = 1.0
        frames = [
            preprocess_video([im], torch_dtype=torch.float32, device="cpu").squeeze(0) for im in keyframes
        ]
        images = torch.cat(frames, dim=1)  # (3, num_frames, H, W) — CPU
        # official mask expansion: repeat first-frame mask 4x, reshape to 4-channel causal layout
        msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8).transpose(1, 2)[0]  # (4, f_lat, h, w)
        vid = images.to(dtype=self.dit_dtype, device=self.device, non_blocking=True)
        y_lat = self.vae.encode([vid], self.device, tiled=True)  # (1, 16, f, h, w) (official tiled=True)
        y_lat = y_lat[0]  # (16, f, h, w)
        y = torch.cat([msk.to(self.device, non_blocking=True), y_lat.to(self.device)], dim=0)  # (20, f, h, w)
        y = y.unsqueeze(0).to(dtype=self.dit_dtype)
        return clip_feature, y

    # ------------------------------------------------------------------ #
    # denoise (one stage, one CFG pair)
    # ------------------------------------------------------------------ #
    def _noise(self, shape, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        return torch.randn(shape, generator=gen, dtype=torch.float32, device=self.device).to(self.dit_dtype)

    def denoise_stage(
        self,
        dit: DancerTransformer3DModel,
        seed: int,
        steps: int,
        cfg_scale: float,
        sigma_shift: float,
        height: int,
        width: int,
        num_frames: int,
        prompt: str,
        negative_prompt: str,
        clip_feature: torch.Tensor,
        y: torch.Tensor,
        music_feature: torch.Tensor | None,
        fps: float,
        refimage_feature: torch.Tensor | None = None,
        enable_skip_layer: bool = False,
        enable_dynamicfps: bool = True,
        stage_name: str = "stage",
    ) -> torch.Tensor:
        f_lat = (num_frames - 1) // 4 + 1
        h_lat = height // 8
        w_lat = width // 8
        latents = self._noise((1, 16, f_lat, h_lat, w_lat), seed)

        ctx_pos = self.encode_prompt(prompt).to(self.dit_dtype).to(self.device, non_blocking=True)
        ctx_neg = self.encode_prompt(negative_prompt).to(self.dit_dtype).to(self.device, non_blocking=True)
        clip_npu = clip_feature.to(self.dit_dtype).to(self.device, non_blocking=True)
        ref_npu = refimage_feature.to(self.dit_dtype).to(self.device, non_blocking=True) if refimage_feature is not None else None
        music_npu = music_feature.to(self.device, non_blocking=True) if music_feature is not None else None

        self.scheduler.set_shift(sigma_shift)
        self.scheduler.set_timesteps(steps, device=self.device)

        t0 = time.time()
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            t = timestep.unsqueeze(0).to(dtype=self.dit_dtype)
            pred_pos = dit(
                latents, t, ctx_pos,
                clip_feature=clip_npu, y=y,
                refimage_feature=ref_npu, music_feature=music_npu, fps=fps,
                enable_music_inject=music_feature is not None,
                enable_refimage=ref_npu is not None,
                enable_dynamicfps=enable_dynamicfps,
                do_skip_layer_nega=False,
            )
            if cfg_scale != 1.0:
                pred_neg = dit(
                    latents, t, ctx_neg,
                    clip_feature=clip_npu, y=y,
                    refimage_feature=ref_npu, music_feature=music_npu, fps=fps,
                    enable_music_inject=music_feature is not None,
                    enable_refimage=ref_npu is not None,
                    enable_dynamicfps=enable_dynamicfps,
                    do_skip_layer_nega=enable_skip_layer,
                )
                pred = pred_neg + cfg_scale * (pred_pos - pred_neg)
            else:
                pred = pred_pos
            latents = self.scheduler.step(pred, timestep, latents).prev_sample
            if progress_id % 4 == 0:
                elapsed = time.time() - t0
                print(f"[{stage_name}] step {progress_id}/{steps}  t={float(timestep):.1f}  {elapsed:.1f}s", flush=True)
        return latents

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
    def generate_global(
        self,
        input_image,
        prompt: str,
        music_feature: torch.Tensor,
        seed: int = 42,
        height: int = 1280,
        width: int = 720,
        num_frames: int = 149,
        steps: int = 48,
        cfg_scale: float = 5.0,
        sigma_shift: float = 5.0,
        input_fps: float | None = None,
        latent_sink: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Official global stage.  Returns (video (1,3,T,H,W) in [-1,1], fps)."""
        if input_fps is None:
            input_fps = 30.0 / int(music_feature.shape[0] / 149.0 + 0.5)
        prompt = f"{prompt}帧率是{input_fps}"
        from PIL import Image

        keyframes = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        keyframes[0] = np.array(input_image, dtype=np.uint8)
        keyframes_list = [Image.fromarray(k) for k in keyframes]
        mask = torch.zeros(num_frames, dtype=torch.int32)
        mask[0] = 1
        clip_feature, y = self.build_keyframe_y(keyframes_list, mask, num_frames, height, width)
        latents = self.denoise_stage(
            self.dit_global, seed, steps, cfg_scale, sigma_shift, height, width, num_frames,
            prompt, WAN_DANCER_NEGATIVE_PROMPT, clip_feature, y, music_feature,
            fps=float(input_fps), enable_dynamicfps=True, stage_name="global",
        )
        if latent_sink is not None:
            np.save(latent_sink, latents.detach().cpu().float().numpy())
            print(f"[global] latents saved -> {latent_sink}", flush=True)
        video = self.vae.decode(latents, self.device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
        video = video.squeeze(0).float().clamp(-1, 1).cpu()  # (3, T, H, W)
        return video, float(input_fps)

    def generate_local(
        self,
        refimage,
        keyframes,
        keyframes_mask,
        prompt: str,
        music_feature: torch.Tensor,
        seed: int = 42,
        height: int = 1280,
        width: int = 720,
        num_frames: int = 149,
        steps: int = 24,
        cfg_scale: float = 5.0,
        sigma_shift: float = 5.0,
        latent_sink: str | None = None,
    ) -> torch.Tensor:
        """Official local stage (one 5 s segment)."""
        if self.dit_local is None:
            raise RuntimeError("local DiT not loaded")
        prompt = f"{prompt}, 帧率是30fps。"
        refimage_feature = self.encode_image(refimage, size=(width, height))
        clip_feature, y = self.build_keyframe_y(keyframes, keyframes_mask, num_frames, height, width)
        latents = self.denoise_stage(
            self.dit_local, seed, steps, cfg_scale, sigma_shift, height, width, num_frames,
            prompt, WAN_DANCER_NEGATIVE_PROMPT, clip_feature, y, music_feature,
            fps=30.0, refimage_feature=refimage_feature,
            enable_skip_layer=True, enable_dynamicfps=True, stage_name="local",
        )
        if latent_sink is not None:
            np.save(latent_sink, latents.detach().cpu().float().numpy())
            print(f"[local] latents saved -> {latent_sink}", flush=True)
        video = self.vae.decode(latents, self.device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
        return video.squeeze(0).float().clamp(-1, 1).cpu()


def process_global_video_firstlastframe(video_path: str, height: int, width: int, total_frames: int):
    """Official local-stage conditioning: keyframes/masks for each 149-frame segment."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(frames)
    seg_num = int(np.ceil(total_frames / 149))
    frame_interval = float(total_frames) / n

    keyframes_list = []
    keyframes_mask_list = []
    for i in range(seg_num):
        mask = np.zeros(149, dtype=np.int32)
        if i != seg_num - 1:
            cnt = 0
            while cnt * frame_interval < 149 - frame_interval:
                mask[int(np.ceil(frame_interval * cnt))] = 1
                cnt += 1
        else:
            end_index = total_frames - 149 * i - 1
            mask[end_index] = 1
            cnt = 0
            while cnt * frame_interval < end_index - frame_interval:
                mask[int(np.ceil(frame_interval * cnt))] = 1
                cnt += 1
        keyframes_mask_list.append(mask)

    index = 0
    for mask in keyframes_mask_list:
        keyframes = np.zeros((149, height, width, 3), dtype=np.uint8)
        keyframes = [Image.fromarray(img) for img in keyframes]
        for j in range(len(mask)):
            if mask[j] == 1:
                frame = Image.fromarray(frames[index].astype("uint8"))
                frame, _ = crop_and_resize(frame, target_width=width, target_height=height)
                keyframes[j] = frame
                index += 1
        keyframes_list.append(keyframes)

    for i in range(len(keyframes_list) - 1):
        keyframes_list[i][-1] = keyframes_list[i + 1][0]
        keyframes_mask_list[i][-1] = 1
    return keyframes_list, keyframes_mask_list
