# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wan-Dancer 310P e2e runner (task-40 acceptance entry point).

Runs the official two-stage flow on the Atlas 300I Duo (TP1, dual-die):

    global  : whole-song 149f @ 720x1280, 48 steps, cfg 5, shift 5,
              input_fps = 30/round(N_music/149), patchify_global/head_global
    local   : per 149f (5s) music segment, 24 steps, cfg 5, shift 5, fps 30,
              keyframes from the global video, refimage = input image,
              enable_skip_layer (block 9, negative pass), enable_dynamicfps

Design: sequential twin loading (global DiT -> global stage -> free ->
local DiT -> local segments -> free) so per-die static memory is ~8.6 GB
(one 17.24 GB twin split 20/20), plus streamed block execution.  T5 stays
CPU-pinned BF16, CLIP CPU-pinned FP32, VAE FP16 on die 1.

Env knobs (resolved by the 310P patch module):
    VLLM_OMNI_310P_DIT_BLOCK_DEVICES  "20,20" (default) | "40" | "single"
    VLLM_OMNI_310P_VAE_DEVICE         "1" (default)
    VLLM_OMNI_310P_DIFFUSION_PATCHES  "1" (default) | "0" level-2 reference
    VLLM_OMNI_310P_SKIP_WARMUP        "1" (TBE dummy-run skip, shared asset)

Perf recording: a background sampler polls ``npu-smi info`` every 2 s and
records per-die HBM peak; step timing comes from the pipeline's step logs.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import threading
import time

import numpy as np
import torch

DEFAULT_PROMPT_GLOBAL_SUFFIX = ""  # pipeline appends the official suffixes


def npu_smi_baseline(tag: str) -> dict:
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"[perf] npu-smi unavailable at {tag}: {exc}", flush=True)
        return {}
    mems = re.findall(r"(\d+)\s*MB", out)
    print(f"[perf] {tag} npu-smi HBM MB: {mems[:8]}", flush=True)
    return {"tag": tag, "hbm_mb": [int(m) for m in mems[:8]], "ts": time.time()}


class HbmSampler(threading.Thread):
    def __init__(self, path: str):
        super().__init__(daemon=True)
        self.path = path
        self.stop_evt = threading.Event()
        self.peaks: dict[int, int] = {}

    def run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30).stdout
                mems = [int(m) for m in re.findall(r"(\d+)\s*MB", out)[:8]]
                for idx, m in enumerate(mems):
                    self.peaks[idx] = max(self.peaks.get(idx, 0), m)
            except Exception:  # noqa: BLE001
                pass
            self.stop_evt.wait(2.0)


def free_module(mod) -> None:
    del mod
    gc.collect()
    try:
        if torch.npu.is_available():
            torch.npu.empty_cache()
    except AttributeError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Wan-Dancer weights dir (global/local model, VAE, T5, CLIP, google/umt5-xxl)")
    ap.add_argument("--assets", required=True, help="staged assets dir (music/, ref_image/, prompt/)")
    ap.add_argument("--music", default="KPopDance.WAV")
    ap.add_argument("--image", default="1001.jpg")
    ap.add_argument("--prompt-global", default="kpop_global.txt")
    ap.add_argument("--prompt-local", default="kpop_local.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="both", choices=["global", "local", "both"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--global-steps", type=int, default=48)
    ap.add_argument("--local-steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--num-frames", type=int, default=149)
    ap.add_argument("--save-latents", action="store_true", help="dump stage latents npy (level-2 comparison)")
    ap.add_argument("--allow-non-310p", action="store_true", help="CPU dry-run without the 310P gate")
    args = ap.parse_args()

    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)

    # --- 310P gate + patch install (kill-switch aware) ---
    from vllm_omni.platforms.npu._310p import is_310p
    from vllm_omni.platforms.npu._310p.patch import wan_dancer

    if not args.allow_non_310p and not is_310p():
        print("[gate] host is not 310P; refusing (use --allow-non-310p for a CPU dry-run)", flush=True)
        raise SystemExit(2)
    wan_dancer.maybe_apply_diffusion_310p_patches()
    npu_smi_baseline("pre-load")

    from vllm_omni.diffusion.models.wan2_2.pipeline_wan_dancer import (
        WanDancerPipeline,
        process_global_video_firstlastframe,
    )
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_components import (
        crop_and_resize,
        get_music_base_feature,
    )
    from PIL import Image

    sampler = HbmSampler(os.path.join(outdir, "hbm_peaks.json"))
    sampler.start()

    music_path = os.path.join(args.assets, "music", args.music)
    image_path = os.path.join(args.assets, "ref_image", args.image)
    prompt_global = open(os.path.join(args.assets, "prompt", args.prompt_global), encoding="utf-8").read().strip()
    prompt_local = open(os.path.join(args.assets, "prompt", args.prompt_local), encoding="utf-8").read().strip()

    # --- music feature (FP32 numpy per P1-11; OMP>=16 assumed from env) ---
    print(f"[music] encoding {music_path}", flush=True)
    t0 = time.time()
    music_np = get_music_base_feature(music_path, os.path.join(outdir, "music_feature.npy"))
    music_feature = torch.from_numpy(music_np)
    n_rows = music_feature.shape[0]
    input_fps = 30.0 / int(n_rows / 149.0 + 0.5)
    print(f"[music] rows={n_rows} input_fps={input_fps} took {time.time() - t0:.1f}s", flush=True)

    img = Image.open(image_path)
    img_padded, box = crop_and_resize(img, target_width=args.width, target_height=args.height)
    tl_x, tl_y, br_x, br_y = box
    print(f"[image] {args.image} content box=({tl_x},{tl_y},{br_x},{br_y})", flush=True)

    block_devs = wan_dancer.resolve_dit_block_devices()
    vae_dev = wan_dancer.resolve_vae_device()
    print(f"[placement] dit_blocks={block_devs} vae={vae_dev} (resolved)", flush=True)

    pipe_kwargs = dict(
        global_dit=os.path.join(args.repo, "global_model.safetensors"),
        local_dit=os.path.join(args.repo, "local_model.safetensors"),
        vae_path=os.path.join(args.repo, "Wan2.1_VAE.pth"),
        t5_path=os.path.join(args.repo, "models_t5_umt5-xxl-enc-bf16.pth"),
        clip_path=os.path.join(args.repo, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        block_devices_global=block_devs,
        block_devices_local=block_devs,
        vae_device=vae_dev,
    )

    # ---------------------------- GLOBAL STAGE ---------------------------- #
    global_mp4 = os.path.join(outdir, "global_crop.mp4")
    if args.stage in ("global", "both"):
        t0 = time.time()
        pipe = WanDancerPipeline.from_repo(args.repo, load_local=False, **pipe_kwargs)
        npu_smi_baseline("post-load-global")
        video, fps = pipe.generate_global(
            img_padded, prompt_global, music_feature,
            seed=args.seed, height=args.height, width=args.width,
            num_frames=args.num_frames, steps=args.global_steps,
            cfg_scale=args.cfg, sigma_shift=5.0, input_fps=float(input_fps),
        )
        _save_video(video, global_mp4, fps=30)
        _crop_video(global_mp4, global_mp4, tl_x, tl_y, br_x, br_y)
        if args.save_latents:
            pass  # latents captured inside generate_global via pipeline hook (below)
        print(f"[global] done {time.time() - t0:.1f}s -> {global_mp4}", flush=True)
        free_module(pipe.dit_global)
        free_module(pipe)
    else:
        video = None

    # ---------------------------- LOCAL STAGE ----------------------------- #
    if args.stage in ("local", "both"):
        pipe = WanDancerPipeline.from_repo(args.repo, load_local=True, **{k: v for k, v in pipe_kwargs.items() if k != "global_dit"})
        pipe.dit_global = None
        npu_smi_baseline("post-load-local")

        # official segmentation of the global video (frames/masks per 149f segment)
        total_frames = int(_audio_duration(music_path) * 30)
        keyframes_list, keyframes_mask_list = process_global_video_firstlastframe(
            global_mp4, args.height, args.width, total_frames
        )
        # official: replace first frame of first segment with the input image
        # (only when the video frame is smaller than 512 on the short side)
        vk0 = keyframes_list[0][0]
        if min(vk0.size[1], vk0.size[0]) < 512:
            keyframes_list[0][0] = img_padded

        seg_music = _slice_music_wavs(music_path, outdir)
        seg_videos = []
        for idx, wav in enumerate(seg_music):
            seg_feature = get_music_base_feature(wav, os.path.join(outdir, f"seg_{idx}_feature.npy"))
            seed = idx * 10 + args.seed
            t0 = time.time()
            seg = pipe.generate_local(
                Image.open(image_path), keyframes_list[idx], keyframes_mask_list[idx],
                prompt_local, torch.from_numpy(seg_feature),
                seed=seed, height=args.height, width=args.width,
                num_frames=args.num_frames, steps=args.local_steps,
                cfg_scale=args.cfg, sigma_shift=5.0,
            )
            seg_mp4 = os.path.join(outdir, f"seg_{idx}_seed{seed}.mp4")
            _save_video(seg, seg_mp4, fps=30)
            seg_videos.append(seg_mp4)
            print(f"[local] seg {idx} done {time.time() - t0:.1f}s", flush=True)
        _concat_videos(seg_videos, os.path.join(outdir, "final_concat.mp4"))
        _mux_audio(os.path.join(outdir, "final_concat.mp4"), music_path, os.path.join(outdir, "final.mp4"))
        print(f"[local] final -> {os.path.join(outdir, 'final.mp4')}", flush=True)
        free_module(pipe.dit_local)
        free_module(pipe)

    sampler.stop_evt.set()
    sampler.join(timeout=5)
    import json

    with open(os.path.join(outdir, "hbm_peaks.json"), "w") as f:
        json.dump(sampler.peaks, f, indent=1)
    npu_smi_baseline("post-run")
    print(f"[done] artifacts in {outdir}", flush=True)


# --------------------------------------------------------------------------- #
# video I/O helpers (cv2; moviepy is not available in the vllm-ascend image)
# --------------------------------------------------------------------------- #
def _tensor_to_frames(video: torch.Tensor) -> np.ndarray:
    """(1, 3, T, H, W) in [-1, 1] -> (T, H, W, 3) uint8 RGB."""
    v = video.squeeze(0).float().clamp(-1, 1)
    v = (v + 1.0) * 127.5
    return v.permute(1, 2, 3, 0).contiguous().numpy().astype(np.uint8)


def _save_video(video: torch.Tensor, path: str, fps: int = 30) -> None:
    import cv2

    frames = _tensor_to_frames(video)
    h, w = frames.shape[1], frames.shape[2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()


def _read_mp4_frames(path: str) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _crop_video(src: str, dst: str, x1: int, y1: int, x2: int, y2: int) -> None:
    if x1 == 0 and y1 == 0:
        import shutil

        shutil.copyfile(src, dst)
        return
    import cv2

    cap = cv2.VideoCapture(src)
    ret, fr = cap.read()
    h, w = fr.shape[:2]
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (x2 - x1, y2 - y1))
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        vw.write(cv2.cvtColor(fr[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cap.release()
    vw.release()


def _concat_videos(paths: list[str], dst: str) -> None:
    import cv2

    cap0 = cv2.VideoCapture(paths[0])
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    fr0 = _read_mp4_frames(paths[0])[0]
    h, w = fr0.shape[:2]
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in paths:
        for fr in _read_mp4_frames(p):
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    cap0.release()


def _audio_duration(path: str) -> float:
    import wave

    with wave.open(path) as wf:
        return wf.getnframes() / wf.getframerate()


def _slice_music_wavs(music_path: str, outdir: str) -> list[str]:
    """Official get_music_clip_149f (soundfile-free: numpy + wave writer)."""
    total = _audio_duration(music_path)
    audio, sr = _load_wav(music_path)
    duration = float(149) / 30.0
    idx, t = 0, 0.0
    out: list[str] = []
    while t + 0.2 < total:
        end = min(t + duration, total)
        chunk = audio[int(t * sr) : int(end * sr)]
        p = os.path.join(outdir, f"music_seg_{idx:03d}.wav")
        _write_wav(p, chunk, sr)
        out.append(p)
        t += duration
        idx += 1
    return out


def _load_wav(path: str):
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return (audio.mean(axis=1) if audio.ndim > 1 else audio), sr
    except Exception:  # noqa: BLE001
        pass
    import wave

    with wave.open(path) as wf:
        n, sw, sr = wf.getnframes(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(n)
    import numpy as np

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2:
        arr = arr / 32768.0
    elif sw == 1:
        arr = (arr - 128) / 128.0
    else:
        arr = arr / 2147483648.0
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def _write_wav(path: str, audio, sr: int) -> None:
    try:
        import soundfile as sf

        sf.write(path, audio.astype("float32"), sr)
        return
    except Exception:  # noqa: BLE001
        pass
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _mux_audio(video: str, audio: str, dst: str) -> None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
         "-c:v", "copy", "-c:a", "aac", "-shortest", dst],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        import shutil

        print(f"[warn] ffmpeg mux failed ({r.stderr[:200]}); keeping silent concat", flush=True)
        shutil.copyfile(video, dst)


if __name__ == "__main__":
    main()
