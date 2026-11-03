# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU contract tests for the Wan-Dancer 310P port (task-40).

No NPU required: the registry/gate logic is exercised with a faked
``is_310p``, the DiT weight mapping is checked against the real safetensors
headers (shapes/dtypes), RoPE is checked against a direct complex reference,
and the full DiT forward is smoke-run at tiny geometry on CPU.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

WAN_DANCER_HDR = os.environ.get(
    "WANDANCER_HDR", "/root/310p-downloads/wan-dancer/global_model_hdr.json"
)


@pytest.fixture(autouse=True)
def _restore_sdpa():
    """The gate test rebinds torch.nn.functional.scaled_dot_product_attention."""
    import torch.nn.functional as F

    orig = F.scaled_dot_product_attention
    yield
    F.scaled_dot_product_attention = orig


# --------------------------------------------------------------------------- #
# registry / kill-switch / gate
# --------------------------------------------------------------------------- #
def test_registry_dispatch_wandancer(monkeypatch):
    from vllm_omni.platforms.npu._310p import patch as patch_pkg
    from vllm_omni.platforms.npu._310p.patch import wan_dancer

    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "1")

    class _Cfg:
        model_class_name = "WanDancerPipeline"

    # Force the 310P gate to true (CPU box).
    import vllm_omni.platforms.npu._310p as npu_310p

    monkeypatch.setattr(npu_310p, "is_310p", lambda: True)
    patch_pkg._DIFFUSION_310P_PATCH_MODULES["WanDancerPipeline"] = [
        "vllm_omni.platforms.npu._310p.patch.wan_dancer"
    ]
    wan_dancer._APPLIED = False
    patch_pkg.apply_diffusion_310p_patches(_Cfg())
    assert wan_dancer._orig_sdpa is not None, "math-SDPA gate must be installed on 310P"


def test_registry_kill_switch(monkeypatch):
    import torch.nn.functional as F
    from vllm_omni.platforms.npu._310p.patch import wan_dancer

    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    wan_dancer._APPLIED = False
    wan_dancer._orig_sdpa = None

    class _Cfg:
        model_class_name = "WanDancerPipeline"

    wan_dancer.apply_diffusion_310p_patches(_Cfg())
    assert wan_dancer._orig_sdpa is None, "kill switch must leave the gate uninstalled"


def test_gate_cpu_passthrough_and_npu_math(monkeypatch):
    """Gate: CPU tensors call the original; faked-NPU tensors take the math path."""
    import torch.nn.functional as F
    from vllm_omni.platforms.npu._310p.patch import wan_dancer

    torch.manual_seed(0)
    orig = F.scaled_dot_product_attention
    wan_dancer._orig_sdpa = orig
    b, h, s, d = 1, 4, 64, 64
    q = torch.randn(b, h, s, d, dtype=torch.float16)
    k = torch.randn(b, h, s, d, dtype=torch.float16)
    v = torch.randn(b, h, s, d, dtype=torch.float16)

    # CPU passthrough (original kernel).
    out_cpu = wan_dancer._math_sdpa_gated(q, k, v)
    ref_cpu = orig(q, k, v)
    assert torch.equal(out_cpu, ref_cpu)

    # Faked NPU device -> math path (compare against naive fp32 reference).
    monkeypatch.setattr(wan_dancer, "_is_npu_tensor", lambda t: True)
    out_npu = wan_dancer._math_sdpa_gated(q, k, v)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (d**0.5)
    ref = torch.softmax(scores, dim=-1) @ v.float()
    assert (out_npu.float() - ref).abs().max().item() < 1.8e-3


# --------------------------------------------------------------------------- #
# math_sdpa equivalence + chunking
# --------------------------------------------------------------------------- #
def test_math_sdpa_equivalence_and_chunking(monkeypatch):
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_attention import math_sdpa

    torch.manual_seed(0)
    b, h, s, d = 2, 8, 96, 64
    q = torch.randn(b, h, s, d, dtype=torch.float16)
    k = torch.randn(b, h, s, d, dtype=torch.float16)
    v = torch.randn(b, h, s, d, dtype=torch.float16)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (d**0.5)
    ref = torch.softmax(scores, dim=-1) @ v.float()

    out_fast = math_sdpa(q, k, v)
    assert (out_fast.float() - ref).abs().max().item() < 1.8e-3

    # Force the chunked path with a tiny budget.
    monkeypatch.setenv("VLLM_OMNI_310P_ATTN_HEAD_CHUNK_BYTES", str(1024 * 1024))
    out_chunk = math_sdpa(q, k, v)
    assert (out_chunk.float() - ref).abs().max().item() < 1.8e-3
    assert (out_fast - out_chunk).abs().max().item() < 2e-3


def test_math_sdpa_gqa_and_mask():
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_attention import math_sdpa

    torch.manual_seed(1)
    b, h, s, d = 1, 2, 40, 32
    q = torch.randn(b, h, s, d, dtype=torch.float16)
    hk, sk = 4, 33
    k = torch.randn(b, hk, sk, d, dtype=torch.float16)
    v = torch.randn(b, hk, sk, d, dtype=torch.float16)
    # Expanded-KV GQA (the 310P rule) before the call.
    k_exp = k.repeat_interleave(h // hk, dim=1)
    v_exp = v.repeat_interleave(h // hk, dim=1)
    out = math_sdpa(q, k_exp, v_exp)
    scores = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) / (d**0.5)
    ref = torch.softmax(scores, dim=-1) @ v_exp.float()
    assert (out.float() - ref).abs().max().item() < 1.8e-3

    # Additive mask path.
    mask = torch.zeros(b, h, s, sk, dtype=torch.float16)
    mask[:, :, :, sk // 2 :] = -10000.0
    out_m = math_sdpa(q, k_exp, v_exp, attn_mask=mask)
    scores = scores + mask.float()
    ref_m = torch.softmax(scores, dim=-1) @ v_exp.float()
    assert (out_m.float() - ref_m).abs().max().item() < 3e-3


# --------------------------------------------------------------------------- #
# DiT weight mapping (real headers from the 101 download)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(WAN_DANCER_HDR), reason="Dancer safetensors headers not downloaded")
def test_dit_weight_mapping():
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_components import _map_dit_state_dict
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import DancerTransformer3DModel

    hdr = json.load(open(WAN_DANCER_HDR))
    hdr = {k: v for k, v in hdr.items() if k != "__metadata__"}
    assert len(hdr) == 1424
    # Fake tensors from header shapes (dtypes as stored: BF16).
    fake_sd = {
        k: torch.zeros(tuple(v["shape"]), dtype=torch.bfloat16) for k, v in hdr.items()
    }
    mapped = _map_dit_state_dict(fake_sd)
    model = DancerTransformer3DModel()
    missing, unexpected = model.load_state_dict(mapped, strict=True, assign=True)
    assert not missing and not unexpected
    n_params = sum(p.numel() for p in model.parameters())
    # 17.245B per twin checkpoint (measured from headers).
    assert 17.0e9 < n_params < 17.5e9, n_params


# --------------------------------------------------------------------------- #
# RoPE vs complex reference
# --------------------------------------------------------------------------- #
def test_rope_tables_and_apply_reference():
    torch.manual_seed(0)
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import (
        DancerTransformer3DModel,
        _apply_rope_real,
        _precompute_freqs_cis,
    )

    model = DancerTransformer3DModel()
    f, h, w = 7, 5, 4
    cos, sin = model.build_freqs((f, h, w), fps=30.0, enable_dynamicfps=True)
    assert cos.shape == (1, f * h * w, 128)

    # Direct complex reference (official semantics, fp64).
    head_dim = 128
    t_dim = head_dim - 2 * (head_dim // 3)
    f_cis = _precompute_freqs_cis(t_dim, model.cfg.freqs_end)
    h_cis = _precompute_freqs_cis(head_dim // 3, model.cfg.freqs_end)
    w_cis = _precompute_freqs_cis(head_dim // 3, model.cfg.freqs_end)
    ref = torch.cat(
        [
            f_cis[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_cis[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_cis[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, -1)  # complex64 (s, 64)

    # My tables must satisfy cos = freqs.real repeated pairwise.
    ref_cos = ref.repeat_interleave(2, dim=-1).real  # (s, 128)
    ref_sin = ref.repeat_interleave(2, dim=-1).imag
    assert (cos[0] - ref_cos.float()).abs().max().item() < 2e-4
    assert (sin[0] - ref_sin.float()).abs().max().item() < 2e-4

    # Apply equivalence on random x.
    s = f * h * w
    x = torch.randn(1, s, 40 * 128, dtype=torch.float32)
    mine = _apply_rope_real(x, cos[0], sin[0], 40)
    # complex reference: view_as_complex per head
    xf = x.view(1, s, 40, 128)
    xc = torch.view_as_complex(xf.double().view(1, s, 40, 64, 2))
    freqs_c = ref.double().view(1, s, 1, 64)
    ref_x = torch.view_as_real(xc * freqs_c).reshape(1, s, 40 * 128)
    assert (mine.double() - ref_x).abs().max().item() < 2e-4


def test_dynamicfps_resample_reference():
    torch.manual_seed(0)
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import DancerTransformer3DModel

    from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import _precompute_freqs_cis

    model = DancerTransformer3DModel()
    f, fps = 149, 7.5
    # Official get_single_freqs (complex, loop) as reference.
    total_frame = int(30.0 / (fps + 1e-6) * f + 0.5)
    interval = 30.0 / (fps + 1e-6)
    t_dim = model._t_dim
    base = _precompute_freqs_cis(t_dim, model.cfg.freqs_end)[:total_frame]  # complex64 (T, d_f)
    ref = torch.zeros(f, base.shape[-1], dtype=torch.complex64)
    ref[0] = base[0]
    ref[-1] = base[total_frame - 1]
    for i in range(1, f - 1):
        pos = i * interval
        low = int(pos)
        high = min(low + 1, total_frame - 1)
        wh = pos - low
        ref[i] = base[low] * (1.0 - wh) + base[high] * wh
    f_cos, f_sin = model._resample_temporal_freqs(f, fps)
    ref_pair = ref.repeat_interleave(2, dim=-1)  # pairwise layout of the stored tables
    assert (f_cos - ref_pair.real).abs().max().item() == 0.0
    assert (f_sin - ref_pair.imag).abs().max().item() == 0.0


# --------------------------------------------------------------------------- #
# tiny-geometry forward smoke (CPU, random weights)
# --------------------------------------------------------------------------- #
def _tiny_forward(**kw):
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import DancerTransformer3DModel

    torch.manual_seed(0)
    model = DancerTransformer3DModel()
    model.eval()
    # latent (1, 16, f, h, w) with h/w even for patch (1,2,2): f=3, h=w=8 -> grid (3,4,4)
    latents = torch.randn(1, 16, 3, 8, 8, dtype=torch.float16)
    y = torch.randn(1, 20, 3, 8, 8, dtype=torch.float16)
    clip = torch.randn(1, 257, 1280, dtype=torch.float16)
    ctx = torch.randn(1, 8, 4096, dtype=torch.float16)  # short prompt for speed
    music = torch.randn(17, 35, dtype=torch.float32)
    t = torch.tensor([512.0], dtype=torch.float16)
    with torch.no_grad():
        out = model(
            latents, t, ctx,
            clip_feature=clip, y=y, music_feature=music,
            fps=30.0, enable_music_inject=True, enable_refimage=False,
            enable_dynamicfps=True, do_skip_layer_nega=kw.get("skip", False),
        )
    return out


def test_tiny_forward_shapes():
    out = _tiny_forward()
    assert out.shape == (1, 16, 3, 8, 8)
    assert torch.isfinite(out).all()


def test_tiny_forward_global_head_and_skip():
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import DancerTransformer3DModel

    torch.manual_seed(0)
    model = DancerTransformer3DModel()
    model.eval()
    latents = torch.randn(1, 16, 3, 8, 8, dtype=torch.float16)
    y = torch.randn(1, 20, 3, 8, 8, dtype=torch.float16)
    clip = torch.randn(1, 257, 1280, dtype=torch.float16)
    ctx = torch.randn(1, 8, 4096, dtype=torch.float16)
    music = torch.randn(17, 35, dtype=torch.float32)
    t = torch.tensor([512.0], dtype=torch.float16)
    with torch.no_grad():
        # global head path (fps != 30)
        out_g = model(latents, t, ctx, clip_feature=clip, y=y, music_feature=music,
                      fps=7.5, enable_music_inject=True, enable_dynamicfps=True)
        # skip layer on (negative pass) vs not — must differ
        out_s = model(latents, t, ctx, clip_feature=clip, y=y, music_feature=music,
                      fps=30.0, enable_music_inject=True, enable_dynamicfps=True, do_skip_layer_nega=True)
        out_n = model(latents, t, ctx, clip_feature=clip, y=y, music_feature=music,
                      fps=30.0, enable_music_inject=True, enable_dynamicfps=True, do_skip_layer_nega=False)
    assert out_g.shape == (1, 16, 3, 8, 8) and torch.isfinite(out_g).all()
    assert (out_s - out_n).abs().max().item() > 0, "skip_layer must change the negative pass"


def test_block_devices_identity():
    """Per-block device list (all same device) must be bit-identical to no list."""
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_transformer import DancerTransformer3DModel

    torch.manual_seed(0)
    latents = torch.randn(1, 16, 3, 8, 8, dtype=torch.float16)
    y = torch.randn(1, 20, 3, 8, 8, dtype=torch.float16)
    clip = torch.randn(1, 257, 1280, dtype=torch.float16)
    ctx = torch.randn(1, 8, 4096, dtype=torch.float16)
    t = torch.tensor([512.0], dtype=torch.float16)

    m1 = DancerTransformer3DModel()
    m2 = DancerTransformer3DModel(block_devices=[torch.device("cpu")] * 40)
    m2.load_state_dict(m1.state_dict())
    m1.eval()
    m2.eval()
    with torch.no_grad():
        o1 = m1(latents, t, ctx, clip_feature=clip, y=y)
        o2 = m2(latents, t, ctx, clip_feature=clip, y=y)
    assert torch.equal(o1, o2)


# --------------------------------------------------------------------------- #
# VAE smoke (CPU, random weights)
# --------------------------------------------------------------------------- #
def test_vae_encode_decode_shapes():
    from vllm_omni.diffusion.models.wan2_2.wan_dancer_vae import WanVideoVAE

    torch.manual_seed(0)
    vae = WanVideoVAE(z_dim=16)
    vae.eval()
    video = torch.randn(3, 5, 64, 64, dtype=torch.float16)
    with torch.no_grad():
        lat = vae.encode([video], "cpu")
        assert lat.shape == (1, 16, 2, 8, 8), lat.shape
        dec = vae.decode(lat, "cpu")
    # official: out_T = T_lat * 4 - 3 -> 2*4-3 = 5
    assert dec.shape == (1, 3, 5, 512, 512), dec.shape
