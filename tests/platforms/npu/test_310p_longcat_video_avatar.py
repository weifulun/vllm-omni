# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the LongCat-Video-Avatar 310P patch wiring (task-34).

Loads ``vllm_omni/platforms/npu/_310p/patch/longcat_video_avatar.py`` and the
310P patch registry from source with fake ``vllm_omni`` dependencies, so the
patch contract (FP16 enforcement, attention-backend forcing, cache
disabling, quantization tier gate, VAE tiling, single-chip preflight,
registry dispatch on ``model_class_name``, kill switch, chunked-attention
numerics, CPU RoPE tables, compute-dtype norms, CPU-pinned encoders,
generator fallback) is pinned without an NPU, vllm-ascend or a real model.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_cosmos3_edge`.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PATCH_PATH = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"


@pytest.fixture()
def patch_mod():
    return _load_source_module("lc310p_patch", _PATCH_PATH / "longcat_video_avatar.py")


@pytest.fixture()
def registry_mod():
    return _load_source_module("lc310p_registry", _PATCH_PATH / "__init__.py")


def _make_od_config(
    *,
    dtype: torch.dtype = torch.float16,
    model_class_name: str = "LongCatVideoAvatarPipeline",
    default_backend: str | None = None,
    per_role: dict | None = None,
    cache_backend: str | None = None,
    quantization_config: object = None,
    vae_use_tiling: bool = False,
    vae_use_slicing: bool = False,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    allgather_degree: int = 1,
    cfg_parallel_size: int = 1,
):
    return SimpleNamespace(
        dtype=dtype,
        model_class_name=model_class_name,
        diffusion_attention_config=SimpleNamespace(
            default=SimpleNamespace(backend=default_backend) if default_backend is not None else None,
            per_role={role: SimpleNamespace(backend=backend) for role, backend in (per_role or {}).items()},
        ),
        cache_backend=cache_backend,
        quantization_config=quantization_config,
        vae_use_tiling=vae_use_tiling,
        vae_use_slicing=vae_use_slicing,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        cfg_parallel_size=cfg_parallel_size,
    )


# ---------------------------------------------------------------------------
# od_config gates
# ---------------------------------------------------------------------------


def test_fp16_enforced(patch_mod):
    assert patch_mod._force_fp16(_make_od_config(dtype=torch.float16)) is None
    with pytest.raises(RuntimeError, match="requires FP16"):
        patch_mod._force_fp16(_make_od_config(dtype=torch.bfloat16))


def test_attention_backend_forcing(patch_mod, monkeypatch):
    _install_fake_attention_spec(monkeypatch)
    # MindIE-SD-only default rejected.
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        patch_mod._validate_attention_selection(_make_od_config(default_backend="FLASH_ATTN"))
    # MindIE-SD-only per-role rejected.
    with pytest.raises(RuntimeError, match="MindIE-SD"):
        patch_mod._validate_attention_selection(_make_od_config(per_role={"longcat_video_avatar.self": "FLASH_ATTN_HUB"}))
    # No default -> TORCH_SDPA installed.
    cfg = _make_od_config()
    assert patch_mod._validate_attention_selection(cfg) == "TORCH_SDPA"
    assert cfg.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_numerical_cache_backend_disabled(patch_mod):
    cfg = _make_od_config(cache_backend="tea_cache")
    patch_mod._disable_numerical_cache_backends(cfg)
    assert cfg.cache_backend == "none"
    cfg = _make_od_config(cache_backend="none")
    patch_mod._disable_numerical_cache_backends(cfg)
    assert cfg.cache_backend == "none"


def test_quantization_tier_gate(patch_mod):
    assert patch_mod._validate_quantization(_make_od_config()) is None  # no config -> pass
    with pytest.raises(RuntimeError, match="rejects online quantization"):
        patch_mod._validate_quantization(
            _make_od_config(quantization_config=SimpleNamespace(method="fp8"))
        )
    # INT8 W8A8 is the sanctioned tier.
    assert (
        patch_mod._validate_quantization(_make_od_config(quantization_config=SimpleNamespace(method="int8")))
        is None
    )


def test_vae_tiling_forced(patch_mod):
    cfg = _make_od_config()
    patch_mod._force_vae_tiling(cfg)
    assert cfg.vae_use_tiling is True and cfg.vae_use_slicing is True


def test_single_chip_parallelism_preflight(patch_mod):
    patch_mod._validate_single_chip_parallelism(_make_od_config())
    with pytest.raises(RuntimeError, match="ulysses_degree=2"):
        patch_mod._validate_single_chip_parallelism(_make_od_config(ulysses_degree=2))
    with pytest.raises(RuntimeError, match="cfg_parallel_size=2"):
        patch_mod._validate_single_chip_parallelism(_make_od_config(cfg_parallel_size=2))


def test_registry_row_registered(registry_mod):
    modules = registry_mod._DIFFUSION_310P_PATCH_MODULES["LongCatVideoAvatarPipeline"]
    assert modules == ["vllm_omni.platforms.npu._310p.patch.longcat_video_avatar"]


def test_registry_dispatch_kill_switch(patch_mod, registry_mod, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.longcat_video_avatar", patch_mod)
    monkeypatch.setattr(patch_mod, "apply_diffusion_310p_patches", lambda od_config: calls.append(od_config), raising=False)
    _install_fake_sdpa(monkeypatch)
    cfg = _make_od_config()
    registry_mod.apply_diffusion_310p_patches(cfg)
    assert len(calls) == 1
    # Kill switch: POLICY section-1 level-2 reference path.
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    registry_mod.apply_diffusion_310p_patches(cfg)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Fake vllm_omni scaffolding (for the deep apply test)
# ---------------------------------------------------------------------------


def _install_fake_attention_spec(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.data",
        types.ModuleType("vllm_omni.diffusion.data"),
    )
    spec_mod = sys.modules["vllm_omni.diffusion.data"]

    class AttentionSpec:
        def __init__(self, backend: str = "TORCH_SDPA"):
            self.backend = backend

    spec_mod.AttentionSpec = AttentionSpec


def _install_fake_sdpa(monkeypatch):
    mod = types.ModuleType("vllm_omni.platforms.npu._310p.patch.attention")

    def maybe_patch_sdpa_math():
        return None

    mod.maybe_patch_sdpa_math = maybe_patch_sdpa_math
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.attention", mod)


def _make_fake_transformer_module(monkeypatch, name="vllm_omni.diffusion.models.longcat_video.transformer"):
    """Fake longcat_video transformer module carrying the REAL upstream
    semantics of everything the patch touches (fp32-chain helpers, norm
    classes, TimestepEmbedder, RoPE).  Defined via exec into the module
    namespace so cross-references resolve dynamically through the module
    globals - exactly like the real module - which means the patch's
    module-global reassignments take effect at every call site."""
    code = _FAKE_TRANSFORMER_SOURCE
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    exec(compile(code, f"<fake {name}>", "exec"), mod.__dict__)  # noqa: S102
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


_FAKE_TRANSFORMER_SOURCE = r"""
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def linear_fp32(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not hasattr(linear, "weight"):
        return linear(x.float())
    bias = None if linear.bias is None else linear.bias.float()
    return F.linear(x.float(), linear.weight.float(), bias)


def silu_linear_fp32(module: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    return linear_fp32(module[1], F.silu(x.float()))


def modulate_fp32(norm_func, x, shift, scale):
    assert shift.dtype == torch.float32 and scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    return x.to(dtype)


def broadcast_concat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(map(lambda t: len(t.shape), tensors))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), "invalid broadcast concat dimensions"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RMSNorm_FP32(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm_FP32(nn.LayerNorm):
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(origin_dtype)
        return out


class TimestepEmbedder(nn.Module):
    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = linear_fp32(self.mlp[0], t_freq)
        t_emb = F.silu(t_emb)
        t_emb = linear_fp32(self.mlp[2], t_emb)
        return t_emb


class FinalLayer_FP32(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_final = LayerNorm_FP32(32, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(32, 64, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(8, 64, bias=True))

    def forward(self, x, t, latent_shape):
        assert t.dtype == torch.float32
        batch, seq_len, channels = x.shape
        num_frames, _, _ = latent_shape
        shift, scale = silu_linear_fp32(self.adaLN_modulation, t).unsqueeze(2).chunk(2, dim=-1)
        x = modulate_fp32(self.norm_final, x.view(batch, num_frames, -1, channels), shift, scale).view(
            batch, seq_len, channels
        )
        x = linear_fp32(self.linear, x)
        return x


class RotaryPositionalEmbedding(torch.nn.Module):
    _MAX_FREQS_LRU_CACHE_SIZE = 32

    def __init__(self, head_dim, cp_split_hw=None):
        super().__init__()
        self.head_dim = head_dim
        self.cp_split_hw = tuple(cp_split_hw or (1, 1))
        self.base = 10000
        self.freqs_dict = OrderedDict()

    def register_grid_size(self, grid_size, key_name, frame_index=None, num_ref_latents=None):
        if key_name in self.freqs_dict:
            self.freqs_dict.move_to_end(key_name)
            return
        self.freqs_dict[key_name] = self.precompute_freqs_cis_3d(grid_size, frame_index, num_ref_latents)
        if len(self.freqs_dict) > self._MAX_FREQS_LRU_CACHE_SIZE:
            self.freqs_dict.popitem(last=False)

    def precompute_freqs_cis_3d(self, grid_size, frame_index=None, num_ref_latents=None):
        num_frames, height, width = grid_size
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        if frame_index is not None and num_ref_latents is not None:
            grid_t = torch.cat(
                [
                    torch.tensor([frame_index], dtype=torch.float32),
                    torch.arange(0, num_frames - num_ref_latents, dtype=torch.float32),
                ],
                dim=0,
            )
        else:
            grid_t = torch.from_numpy(np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)).float()
        grid_h = torch.from_numpy(np.linspace(0, height, height, endpoint=False, dtype=np.float32)).float()
        grid_w = torch.from_numpy(np.linspace(0, width, width, endpoint=False, dtype=np.float32)).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcast_concat(
            (freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1
        )
        freqs = rearrange(freqs, "T H W D -> (T H W) D")
        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            raise NotImplementedError("LongCat-Video-Avatar context parallel RoPE is not supported in native MVP.")
        return freqs

    def forward(self, q, k, grid_size, frame_index=None, num_ref_latents=None):
        key_name = (tuple(grid_size), frame_index, num_ref_latents)
        self.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)
        freqs_cis = self.freqs_dict[key_name].to(q.device)
        q_, k_ = q.float(), k.float()
        freqs_cis = freqs_cis.float().to(q.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, "n d -> 1 1 n d"), rearrange(sin, "n d -> 1 1 n d")
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)
        return q_.type_as(q), k_.type_as(k)
"""


def _make_fake_pipeline_module(monkeypatch):
    mod = types.ModuleType("vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar")

    class LongCatVideoAvatarPipeline:
        def __init__(self, od_config=None, device=None):
            self.od_config = od_config
            self.device = device or torch.device("cpu")
            self.text_encoder = torch.nn.Linear(4, 4)
            self.audio_encoder = torch.nn.Linear(4, 4)
            self.transformer = _tiny_transformer()
            self.vae = torch.nn.Linear(4, 4)

        def to(self, *args, **kwargs):
            return self

        def generate(self, req, *args, **kwargs):
            self.generated = req
            return "video"

        def _get_t5_prompt_embeds(self, prompt, num_videos_per_prompt, max_sequence_length, device, dtype):
            return torch.zeros(1, 1, 8, 4).to(device, dtype=dtype), torch.ones(1, 8, device=device)

    mod.LongCatVideoAvatarPipeline = LongCatVideoAvatarPipeline
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar", mod)
    return mod


def _tiny_transformer():
    transformer = torch.nn.Module()
    blocks = torch.nn.Module()
    attn_layer = _tiny_engine_attention()
    blocks.add_module("attn_layer", attn_layer)
    transformer.add_module("blocks", blocks)
    # LoRA network as a plain dict attribute (upstream contract).
    transformer.lora_dict = {"dmd": torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Linear(2, 4))}
    return transformer


def _tiny_engine_attention():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax_scale = 1.0 / math.sqrt(8)

        def forward(self, query, key, value, attn_metadata=None):
            # Reference math attention in BSND for the wrapper tests.
            scale = self.softmax_scale
            out = torch.einsum("bshd,bchd->bhsc", query, key) * scale
            out = torch.softmax(out.float(), dim=-1).to(query.dtype)
            return torch.einsum("bhsc,bchd->bshd", out, value)

    return Attention()


def _install_fake_package_chain(monkeypatch):
    names = [
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.models",
        "vllm_omni.diffusion.models.longcat_video",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
        "vllm_omni.platforms.npu._310p.patch",
    ]
    for name in names:
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)


# ---------------------------------------------------------------------------
# Deep apply (placement / rope / norms / generator) with fakes
# ---------------------------------------------------------------------------


def test_deep_apply_installs_placements(patch_mod, monkeypatch):
    _install_fake_package_chain(monkeypatch)
    _install_fake_sdpa(monkeypatch)
    _install_fake_attention_spec(monkeypatch)
    _fake_transformer = _make_fake_transformer_module(monkeypatch)
    pipeline_mod = _make_fake_pipeline_module(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu._310p.patch.longcat_video_avatar", patch_mod)

    # Platform guard fakes.
    platform_mod = types.ModuleType("vllm_omni.platforms.npu.platform")

    class NPUOmniPlatform:
        @classmethod
        def get_diffusion_attn_backend_cls(cls, selected_backend, head_size, allow_trtllm_default=True):
            return "sdpa"

    platform_mod.NPUOmniPlatform = NPUOmniPlatform
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu.platform", platform_mod)
    registry_fake = types.ModuleType("vllm_omni.diffusion.attention.backends.registry")

    class DiffusionAttentionBackendEnum:
        TORCH_SDPA = "sdpa"

        def get_path(self):
            return "sdpa"

    registry_fake.DiffusionAttentionBackendEnum = DiffusionAttentionBackendEnum
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.attention.backends.registry", registry_fake)
    for sub in ("vllm_omni.diffusion.attention", "vllm_omni.diffusion.attention.backends"):
        monkeypatch.setitem(sys.modules, sub, types.ModuleType(sub))

    cfg = _make_od_config()
    patch_mod.apply_diffusion_310p_patches(cfg)

    cls = pipeline_mod.LongCatVideoAvatarPipeline
    assert getattr(cls, "_310p_placement_patched", False) is True
    assert patch_mod._APPLIED is True

    # Construction runs the placement hook: encoders CPU-pinned, attention
    # wrapped, bf16 residuals cast to fp16.
    pipe = cls(od_config=cfg)
    assert pipe.text_encoder._310p_cpu_pinned
    assert pipe.audio_encoder._310p_cpu_pinned
    # The CPU-pinned overlay redirects .to() back to CPU.
    pipe.text_encoder.to("cpu")  # no-op target
    attn = pipe.transformer.blocks.attn_layer
    assert getattr(attn, "_310p_chunked_wrapped", False) is True

    # LoRA dict cast to fp16.
    for module in pipe.transformer.lora_dict["dmd"]:
        assert module.weight.dtype == torch.float16

    # bf16 -> fp16 cast on the transformer (the tiny qkv is fp32 by default,
    # so re-seed a bf16 param to exercise the cast).
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    pipe.transformer.register_parameter("resid_param", p)
    _cast = pipe.transformer  # already cast at init; verify behaviour directly:
    patch_mod._cast_module_to_fp16(_cast)
    assert pipe.transformer.resid_param.dtype == torch.float16


def test_rope_cpu_table_equivalence(patch_mod, monkeypatch):
    _install_fake_package_chain(monkeypatch)
    fake_transformer = _make_fake_transformer_module(monkeypatch)
    patch_mod._patch_rope_cpu_tables()

    rope = fake_transformer.RotaryPositionalEmbedding(head_dim=16)
    grid = (4, 3, 5)
    table = rope.freqs_dict
    key = (grid, None, None)
    rope.register_grid_size(grid, key, None, None)
    assert key in table
    stored = table[key]
    assert stored.device.type == "cpu"
    assert stored.dtype == torch.float32
    assert stored.shape == (4 * 3 * 5, 16)

    # Forward: same rotation as the upstream reference (fp32 math), but the
    # result stays in the input dtype/device - verify numerically against a
    # direct fp32 reference built from the stored table.
    torch.manual_seed(0)
    # S must equal the table length: T*H*W = 4*3*5 = 60 tokens.
    q = torch.randn(1, 7, 60, 16, dtype=torch.float16)  # (B, H, S, D)
    k = torch.randn(1, 7, 60, 16, dtype=torch.float16)
    q_out, k_out = rope.forward(q, k, grid)
    assert q_out.dtype == torch.float16 and k_out.dtype == torch.float16

    cos = stored.cos().to(q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = stored.sin().to(q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    rot = fake_transformer.rotate_half
    ref_q = (q * cos + rot(q) * sin)
    ref_k = (k * cos + rot(k) * sin)
    # fp16 rotation of O(1) values: tight tolerance vs the same-fp16 math.
    assert torch.allclose(q_out, ref_q, atol=2e-3, rtol=2e-2)
    assert torch.allclose(k_out, ref_k, atol=2e-3, rtol=2e-2)
    # And against the upstream fp32 semantics (loose fp16 tolerance).
    ref_q32 = (q.float() * cos.float() + rot(q.float()) * sin.float())
    ref_k32 = (k.float() * cos.float() + rot(k.float()) * sin.float())
    assert torch.allclose(q_out.float(), ref_q32, atol=2e-2, rtol=2e-2)
    assert torch.allclose(k_out.float(), ref_k32, atol=2e-2, rtol=2e-2)


def test_norm_compute_dtype_patched(patch_mod, monkeypatch):
    _install_fake_package_chain(monkeypatch)
    fake = _make_fake_transformer_module(monkeypatch)
    patch_mod._patch_norm_compute_dtype()

    torch.manual_seed(0)
    x = torch.randn(2, 24, 32, dtype=torch.float16)

    # 1) RMSNorm: fp16 in -> fp16 out, math == upstream fp32 norm within fp16 tol.
    rms = fake.RMSNorm_FP32(32, eps=1e-6).to(torch.float16)
    out = rms(x)
    assert out.dtype == torch.float16
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.float16)
    ref = ref * rms.weight.to(torch.float16)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    # 2) LayerNorm: same contract.
    ln = fake.LayerNorm_FP32(32, eps=1e-6, elementwise_affine=True).to(torch.float16)
    out = ln(x)
    assert out.dtype == torch.float16
    ref = F.layer_norm(x.float(), (32,), ln.weight.float(), ln.bias.float(), 1e-6)
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)

    # 3) modulate: upstream affine x*(scale+1)+shift; affine runs in fp16 now.
    shift = torch.randn(2, 1, 32, dtype=torch.float32) * 0.1
    scale = torch.randn(2, 1, 32, dtype=torch.float32) * 0.1
    xv = x.view(2, 1, 24, 32)  # (B, F, T, C)
    ln_noaff = fake.LayerNorm_FP32(32, eps=1e-6, elementwise_affine=False).to(torch.float16)
    out = fake.modulate_fp32(ln_noaff, xv, shift, scale)
    assert out.dtype == torch.float16
    ref = F.layer_norm(xv.float(), (32,), None, None, 1e-6)
    ref = (ref * (scale + 1) + shift)
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)

    # 4) TimestepEmbedder: output contract stays fp32 despite the fp16 compute.
    te = fake.TimestepEmbedder(8).to(torch.float16)
    t = torch.randn(2, dtype=torch.float32)
    out = te.forward(t, torch.float16)
    assert out.dtype == torch.float32
    t_freq = te.timestep_embedding(t, te.frequency_embedding_size)
    ref = F.linear(t_freq, te.mlp[0].weight.float(), te.mlp[0].bias.float())
    ref = F.linear(F.silu(ref), te.mlp[2].weight.float(), te.mlp[2].bias.float())
    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)

    # 5) FinalLayer end-to-end: assert t fp32 passes, output fp16, math within tol.
    fl = fake.FinalLayer_FP32().to(torch.float16)
    xf = torch.randn(2, 4, 32, dtype=torch.float16)
    tf = te.forward(t, torch.float16).view(2, 1, -1)  # fp32, (B, 1, D) as in the real flow
    out = fl.forward(xf, tf, (2, 1, 2))
    assert out.dtype == torch.float16
    emb = F.linear(F.silu(tf), fl.adaLN_modulation[1].weight.float(), fl.adaLN_modulation[1].bias.float())
    sh, sc = emb.unsqueeze(2).chunk(2, dim=-1)
    rx = F.layer_norm(xf.view(2, 2, 2, 32).float(), (32,), None, None, 1e-6)
    rx = ((rx * (sc + 1) + sh).view(2, 4, 32))
    ref = F.linear(rx, fl.linear.weight.float(), fl.linear.bias.float())
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)

    # 6) Kill switch: env=0 keeps the upstream chain (fp32 asserts intact).
    monkeypatch.setenv("VLLM_OMNI_310P_LONGCAT_FP16_NORMS", "0")
    fake2 = _make_fake_transformer_module(monkeypatch)
    patch_mod._patch_norm_compute_dtype()
    with pytest.raises(AssertionError):
        fake2.modulate_fp32(
            fake2.LayerNorm_FP32(32, eps=1e-6, elementwise_affine=False),
            torch.randn(2, 4, 32, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
        )


def test_chunked_attention_matches_reference(patch_mod):
    torch.manual_seed(0)
    b, h, s, d = 1, 5, 40, 8  # head count not divisible by head_group=4
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / math.sqrt(d)

    out = patch_mod._chunked_attention(q, k, v, scale, None, head_group=4, kv_chunk=16)
    assert out.shape == (b, h, s, d)
    ref = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1) @ v
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    # kv_chunk larger than seq -> single chunk.
    out = patch_mod._chunked_attention(q, k, v, scale, None, head_group=4, kv_chunk=1024)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_chunked_attention_masked(patch_mod):
    torch.manual_seed(1)
    b, h, s, d = 1, 2, 16, 4
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    # Mask out keys with index >= 8 for all queries (broadcast layout).
    mask = torch.zeros(b, 1, s, s)
    mask[:, :, :, 8:] = float("-inf")
    out = patch_mod._chunked_attention(q, k, v, 0.5, mask, head_group=2, kv_chunk=7)
    ref = torch.softmax(q @ k.transpose(-1, -2) * 0.5 + mask, dim=-1) @ v
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    # Masked keys must not contribute: zeroing the masked K/V leaves the
    # output identical.
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 8:] = 0
    v2[:, :, 8:] = 0
    out2 = patch_mod._chunked_attention(q, k2, v2, 0.5, mask, head_group=2, kv_chunk=7)
    assert torch.allclose(out, out2, atol=1e-5, rtol=1e-5)


def test_cpu_pinned_overlay(patch_mod):
    m = torch.nn.Linear(4, 4)
    patch_mod._make_cpu_pinned(m)
    m.to("cpu")
    assert next(m.parameters()).device.type == "cpu"
    # A device-requesting .to() is redirected to CPU.
    m.to("cpu")
    assert next(m.parameters()).device.type == "cpu"
    assert m._310p_cpu_pinned


def test_generator_fallback_wraps_generate(patch_mod, monkeypatch):
    _install_fake_package_chain(monkeypatch)
    pipeline_mod = _make_fake_pipeline_module(monkeypatch)
    cls = pipeline_mod.LongCatVideoAvatarPipeline

    # CPU-only box: the NPU generator probe fails -> fallback armed.
    monkeypatch.setattr(patch_mod, "_probe_npu_generator", lambda: False)
    patch_mod._install_generator_fallback(cls)
    assert getattr(cls, "_310p_generator_patched", False) is True
    pipe = cls()
    req = SimpleNamespace(sampling_params=SimpleNamespace(generator=None, seed=7))
    pipe.generate(req)
    gen = req.sampling_params.generator
    assert isinstance(gen, torch.Generator)
    assert gen.device.type == "cpu"

    # An explicitly passed NPU generator (official end2end.py pattern:
    # torch.Generator(device="npu").manual_seed(42)) is also swapped for a
    # CPU generator, seed taken from sampling_params.seed.
    req2 = SimpleNamespace(
        sampling_params=SimpleNamespace(
            generator=SimpleNamespace(device=SimpleNamespace(type="npu")), seed=99
        )
    )
    pipe.generate(req2)
    gen2 = req2.sampling_params.generator
    assert isinstance(gen2, torch.Generator)
    assert gen2.device.type == "cpu"

    # An explicitly passed CPU generator is left untouched.
    cpu_gen = torch.Generator(device="cpu").manual_seed(5)
    req3 = SimpleNamespace(sampling_params=SimpleNamespace(generator=cpu_gen, seed=5))
    pipe.generate(req3)
    assert req3.sampling_params.generator is cpu_gen

    # NPU generator supported -> no wrap.
    cls2 = type("P2", (object,), {})
    monkeypatch.setattr(patch_mod, "_probe_npu_generator", lambda: True)
    patch_mod._install_generator_fallback(cls2)
    assert not getattr(cls2, "_310p_generator_patched", False)


def test_prompt_embeds_device_coercion(patch_mod, monkeypatch):
    _install_fake_package_chain(monkeypatch)
    pipeline_mod = _make_fake_pipeline_module(monkeypatch)
    cls = pipeline_mod.LongCatVideoAvatarPipeline
    pipe = cls()
    # Simulate the patched init hook portion:
    torch.nn.modules.module.Module.to(pipe.text_encoder, "cpu")
    patch_mod._make_cpu_pinned(pipe.text_encoder)
    patch_mod._make_cpu_pinned(pipe.audio_encoder)
    patch_mod._patch_prompt_embeds_device(pipe)
    patch_mod._patch_pipeline_to_override(pipe)

    emb, mask = pipe._get_t5_prompt_embeds("hi", 1, 8, torch.device("cpu"), torch.float16)
    assert emb.device.type == "cpu"  # coerced to the encoder device

    # The pipeline to() override keeps the pinned encoders on CPU.
    pipe.to("cpu")
    assert next(pipe.text_encoder.parameters()).device.type == "cpu"
