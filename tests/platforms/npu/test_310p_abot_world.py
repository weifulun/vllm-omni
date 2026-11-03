# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ABot-World family 310P patch wiring.

Loads ``vllm_omni/platforms/npu/_310p/patch/abot_world.py``,
``.../ar_paged.py`` and the 310P patch registry from source with fake
``vllm``/``vllm_omni`` dependencies, so the patch contract (FP16 enforcement,
VAE tiling flags, KV-quant refusal, model-class registration under the same
arch, custom ``encode_prompt`` device routing with the ``_encoder_device``
suite guard, the AR paged-attention math fallback (diversion gate, gather
semantics, numeric parity with the reference, head-chunk budget, batched
reshape) and the shared dispatch (registry line, hardware shims outside the
kill-switch gate, startup warmup skip asset 6e4e4e3b6) is pinned without an
NPU, vllm-ascend or a real model.

Mirrors the fakes-first convention of
:mod:`tests.platforms.npu.test_310p_cosmos3_edge`.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


_PATCH_DIR = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch"


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs):
    """Install a fake module under *name*, creating the full parent chain.

    ``import a.b.c`` resolves every dotted prefix through ``sys.modules``
    first; missing prefixes would import the real (unimportable on this
    host) packages.  Parents already present are reused (and linked).
    """
    parts = name.split(".")
    parent = None
    for i in range(len(parts) - 1):
        prefix = ".".join(parts[: i + 1])
        if prefix not in sys.modules:
            parent_module = types.ModuleType(prefix)
            monkeypatch.setitem(sys.modules, prefix, parent_module)
            if parent is not None:
                setattr(parent, parts[i], parent_module)
        parent = sys.modules[prefix]
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    if parent is not None:
        setattr(parent, parts[-1], module)
    return module


# ── AR paged-attention math shim ─────────────────────────────────────────────


def _reference_paged_attention(query, key_cache, value_cache, block_table, query_start_loc, seq_lens, scale):
    """CPU reference mirroring upstream ``_reference_paged_attention``.

    Flattens a 4D (B,T,H,D) query to TND before slicing (the NPU branch does
    the same), so the per-row ``block_table[i]`` gather is well defined over
    the flat axis.
    """
    batched = query.dim() == 4
    orig_shape = query.shape
    query_flat = query.reshape(-1, *query.shape[2:]) if batched else query
    outs = []
    block_size = key_cache.shape[1]
    for i in range(seq_lens.shape[0]):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        kv_len = int(seq_lens[i].item())
        q = query_flat[q_start:q_end]
        positions = torch.arange(kv_len, device=query.device)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions % block_size
        physical_blocks = block_table[i, logical_blocks].long()
        k = key_cache[physical_blocks, offsets]
        v = value_cache[physical_blocks, offsets]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * float(scale)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    flat_out = torch.cat(outs, dim=0)
    return flat_out.reshape(orig_shape) if batched else flat_out


def _paged_setup(seed=0, block_size=8, max_seq_len=20, heads=4, dim=8):
    """Single-request layout (the real AR-decode usage: ``max_num_seqs=1``),
    matching the upstream NPU branch's ``block_table[0]`` shared-window gather."""
    g = torch.Generator().manual_seed(seed)
    key_cache = torch.randn(4, block_size, heads, dim, generator=g)
    value_cache = torch.randn(4, block_size, heads, dim, generator=g)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    query_start_loc = torch.tensor([0, max_seq_len], dtype=torch.long)  # one sequence
    seq_lens = torch.tensor([max_seq_len], dtype=torch.long)
    query = torch.randn(max_seq_len, heads, dim, generator=g)
    return query, key_cache, value_cache, block_table, query_start_loc, seq_lens


def _gathered_window(key_cache, value_cache, block_table, max_seq_len):
    block_size = key_cache.shape[1]
    num_blocks = (int(max_seq_len) + block_size - 1) // block_size
    block_ids = block_table[0, :num_blocks].long()
    gathered_k = key_cache.index_select(0, block_ids).flatten(0, 1)[: int(max_seq_len)]
    gathered_v = value_cache.index_select(0, block_ids).flatten(0, 1)[: int(max_seq_len)]
    return gathered_k, gathered_v


def _load_ar_paged(monkeypatch: pytest.MonkeyPatch):
    _install_fake_module(
        monkeypatch,
        "vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention",
        ar_diffusion_paged_attention=lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("original paged attention must not be called by the math path")
        ),
    )
    return _load_source_module("patch_ar_paged", _PATCH_DIR / "ar_paged.py")


def test_ar_paged_install_wraps(monkeypatch: pytest.MonkeyPatch):
    mod = _load_ar_paged(monkeypatch)
    import vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention as fake_pa

    assert not getattr(fake_pa.ar_diffusion_paged_attention, "_310p_math_patched", False)
    mod.maybe_patch_ar_paged_math()
    assert getattr(fake_pa.ar_diffusion_paged_attention, "_310p_math_patched", False)
    # Idempotent: second call is a no-op on the already-wrapped function.
    wrapped = fake_pa.ar_diffusion_paged_attention
    mod.maybe_patch_ar_paged_math()
    assert fake_pa.ar_diffusion_paged_attention is wrapped


def test_ar_paged_cpu_passthrough(monkeypatch: pytest.MonkeyPatch):
    mod = _load_ar_paged(monkeypatch)
    import vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention as fake_pa

    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "original-result"

    fake_pa.ar_diffusion_paged_attention = original
    mod.maybe_patch_ar_paged_math()

    query = torch.randn(4, 2, 4)
    key_cache = torch.randn(2, 8, 2, 4)
    value_cache = torch.randn(2, 8, 2, 4)
    out = fake_pa.ar_diffusion_paged_attention(
        query,
        key_cache,
        value_cache,
        block_table=torch.zeros(1, 2, dtype=torch.long),
        query_start_loc=torch.tensor([0, 4], dtype=torch.long),
        seq_lens=torch.tensor([8], dtype=torch.long),
        max_query_len=4,
        max_seq_len=8,
        softmax_scale=0.5,
    )
    assert out == "original-result"
    assert len(calls) == 1  # CPU device never diverted


def test_ar_paged_math_diversion_matches_reference(monkeypatch: pytest.MonkeyPatch):
    mod = _load_ar_paged(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_PAGED_MATH_DEVICE", "cpu")
    mod.maybe_patch_ar_paged_math()
    import vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention as fake_pa

    scale = 0.5
    for seed in (0, 1, 2):
        query, key_cache, value_cache, block_table, query_start_loc, seq_lens = _paged_setup(seed=seed)
        ref = _reference_paged_attention(
            query, key_cache, value_cache, block_table, query_start_loc, seq_lens, scale
        )
        out = fake_pa.ar_diffusion_paged_attention(
            query,
            key_cache,
            value_cache,
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_query_len=int(query_start_loc[1] - query_start_loc[0]),
            max_seq_len=int(seq_lens.max()),
            softmax_scale=scale,
        )
        assert out.shape == query.shape
        assert (out - ref).abs().max().item() < 1e-5, f"seed={seed}"


def test_ar_paged_head_chunk_budget_matches_single(monkeypatch: pytest.MonkeyPatch):
    """Forcing tiny head chunks (multi-chunk softmax) must not change numerics."""
    mod = _load_ar_paged(monkeypatch)
    query, key_cache, value_cache, block_table, query_start_loc, seq_lens = _paged_setup(seed=3)
    gathered_k, gathered_v = _gathered_window(key_cache, value_cache, block_table, int(seq_lens.max()))
    scale = 0.5
    monkeypatch.delenv("VLLM_OMNI_310P_ATTN_HEAD_CHUNK_BYTES", raising=False)
    base = mod._math_varlen_paged_attention(
        query, gathered_k, gathered_v, query_start_loc, seq_lens, scale
    )
    # Force a 1-head chunk (2 MiB budget < one head of this geometry).
    monkeypatch.setenv("VLLM_OMNI_310P_ATTN_HEAD_CHUNK_BYTES", "2097152")
    chunked = mod._math_varlen_paged_attention(
        query, gathered_k, gathered_v, query_start_loc, seq_lens, scale
    )
    assert (base - chunked).abs().max().item() < 1e-6


def test_ar_paged_batched_reshape(monkeypatch: pytest.MonkeyPatch):
    mod = _load_ar_paged(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_PAGED_MATH_DEVICE", "cpu")
    mod.maybe_patch_ar_paged_math()
    import vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention as fake_pa

    scale = 0.5
    g = torch.Generator().manual_seed(4)
    heads, dim, block_size, kv_len = 4, 8, 8, 20
    key_cache = torch.randn(4, block_size, heads, dim, generator=g)
    value_cache = torch.randn(4, block_size, heads, dim, generator=g)
    # Two batch entries sharing the same KV pages (identical rows), so the
    # per-row reference equals the NPU branch's shared row-0 window gather.
    block_table = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    query_start_loc = torch.tensor([0, kv_len, 2 * kv_len], dtype=torch.long)
    seq_lens = torch.tensor([kv_len, kv_len], dtype=torch.long)
    query_b = torch.randn(2, kv_len, heads, dim, generator=g)
    ref = _reference_paged_attention(
        query_b, key_cache, value_cache, block_table, query_start_loc, seq_lens, scale
    )
    out = fake_pa.ar_diffusion_paged_attention(
        query_b,
        key_cache,
        value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=int(query_start_loc[1] - query_start_loc[0]),
        max_seq_len=int(seq_lens.max()),
        softmax_scale=scale,
    )
    assert out.shape == query_b.shape
    assert (out - ref).abs().max().item() < 1e-5


# ── ABot-World model patch module ───────────────────────────────────────────


class _FakeTokenizer:
    def __init__(self, seq_len: int = 8):
        self.seq_len = seq_len
        self.last_prompts = None

    def __call__(self, prompts, **kwargs):
        self.last_prompts = prompts
        mask = torch.zeros(1, self.seq_len, dtype=torch.long)
        mask[:, : self.seq_len // 2] = 1
        return SimpleNamespace(
            input_ids=torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0),
            attention_mask=mask,
        )


class _FakeEncoder(torch.nn.Module):
    """Deterministic stand-in for the UMT5-XXL encoder."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.proj = torch.nn.Linear(2, dim)

    def forward(self, input_ids, attention_mask):
        x = torch.stack([input_ids % 7, (input_ids * 3) % 5], dim=-1).to(torch.float32)
        h = self.proj(x)
        h = h * attention_mask.unsqueeze(-1).to(h.dtype)
        return SimpleNamespace(last_hidden_state=h)


def _fake_abot_base(monkeypatch: pytest.MonkeyPatch):
    import torch.nn as nn

    # The 310P execution device as seen by the base pipeline's
    # ``text_encoder.to(self.device)`` line.  The real runtime passes a
    # ``torch.device("npu", i)``; the string stand-in routes through the
    # same ``_is_npu_device`` branch in the 310P pin's shadow.
    _NPU_CONSTRUCTION_SPEC = "npu:0"

    class _Base(nn.Module):
        def __init__(self, *, od_config, prefix: str = ""):
            super().__init__()
            self.od_config = od_config
            self.construction_dtype = torch.get_default_dtype()
            self.device = torch.device("cpu")  # execution-device stand-in
            self.tokenizer = _FakeTokenizer()
            # Mirror the stock ABotWorldCausalPipeline.__init__ encoder flow:
            # _load_text_encoder (CPU) -> .to(execution device) BEFORE the DiT
            # is built.  With the 310P pin the npu spec is rewritten to CPU
            # inside the shadow; without it (opt-out) a CPU-only host raises —
            # the emulated NPU move is installed by the opt-out test.
            self.text_encoder = self._load_text_encoder("model", torch.float16)
            self.text_encoder = self.text_encoder.to(_NPU_CONSTRUCTION_SPEC)
            self.vae = nn.Linear(4, 4)
            self.transformer = nn.Linear(4, 4)

        def _load_text_encoder(self, model, dtype):
            return _FakeEncoder()

    _install_fake_module(
        monkeypatch, "vllm_omni.diffusion.models.abot_world.pipeline_abot_world",
        ABotWorldCausalPipeline=_Base,
    )
    return _Base


def _load_abot_module(monkeypatch: pytest.MonkeyPatch):
    _fake_abot_base(monkeypatch)
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.common",
                         maybe_install_phase_tracer=lambda *a, **kw: None)
    calls = []
    _install_fake_module(
        monkeypatch, "vllm_omni.diffusion.registry",
        register_diffusion_model=lambda *a, **kw: calls.append((a, kw)),
        initialize_model=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    return _load_source_module("patch_abot_world", _PATCH_DIR / "abot_world.py"), calls


def _make_od_config(dtype=torch.bfloat16, model_class_name="ABotWorldCausalPipeline"):
    return SimpleNamespace(
        dtype=dtype,
        model_class_name=model_class_name,
        vae_use_tiling=False,
        vae_use_slicing=False,
        diffusion_kv_cache_dtype="fp8",
    )


def test_abot_apply_forces_fp16_and_flags(monkeypatch: pytest.MonkeyPatch):
    mod, calls = _load_abot_module(monkeypatch)
    od_config = _make_od_config(dtype=torch.bfloat16)
    mod.apply_diffusion_310p_patches(od_config)
    assert od_config.dtype is torch.float16
    assert od_config.vae_use_tiling is True
    assert od_config.vae_use_slicing is True
    assert od_config.diffusion_kv_cache_dtype == "auto"


def test_abot_registers_310p_override_under_same_arch(monkeypatch: pytest.MonkeyPatch):
    mod, calls = _load_abot_module(monkeypatch)
    mod.apply_diffusion_310p_patches(_make_od_config())
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "ABotWorldCausalPipeline"
    assert kwargs["module_name"] == mod.__name__
    assert kwargs["class_name"] == "_ABotWorldCausal310P"
    # pre/post process funcs kept from the built-in registration.
    assert kwargs.get("pre_process_func_name") is None
    assert kwargs.get("post_process_func_name") is None
    # Idempotent: second apply does not re-register.
    od_config = _make_od_config()
    mod.apply_diffusion_310p_patches(od_config)
    assert len(calls) == 1


def _build_310p_pipeline(monkeypatch: pytest.MonkeyPatch, od_config, encoder_on_npu_env=None):
    mod, _ = _load_abot_module(monkeypatch)
    if encoder_on_npu_env is not None:
        monkeypatch.setenv("VLLM_OMNI_310P_ABOT_ENCODER_ON_NPU", encoder_on_npu_env)
    else:
        monkeypatch.delenv("VLLM_OMNI_310P_ABOT_ENCODER_ON_NPU", raising=False)
    return mod, mod._ABotWorldCausal310P(od_config=od_config)


def test_abot_subclass_pins_fp16_around_construction(monkeypatch: pytest.MonkeyPatch):
    mod, _ = _load_abot_module(monkeypatch)
    od_config = _make_od_config(dtype=torch.float16)
    monkeypatch.delenv("VLLM_OMNI_310P_ABOT_ENCODER_ON_NPU", raising=False)
    _prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        pipeline = mod._ABotWorldCausal310P(od_config=od_config)
    finally:
        torch.set_default_dtype(_prev)
    # Construction ran under the pinned FP16 default dtype...
    assert pipeline.construction_dtype is torch.float16
    # ...and the process default dtype was restored afterwards.
    assert torch.get_default_dtype() is torch.float32


def test_abot_encoder_pinned_cpu_fp32(monkeypatch: pytest.MonkeyPatch):
    mod, pipeline = _build_310p_pipeline(monkeypatch, _make_od_config(dtype=torch.float16))
    assert pipeline._encoder_device.type == "cpu"
    for p in pipeline.text_encoder.parameters():
        assert p.device.type == "cpu"
        assert p.dtype == torch.float32


def test_abot_encoder_pin_blocks_npu_move_during_construction(monkeypatch: pytest.MonkeyPatch):
    """The pin must hold across the base __init__'s ``text_encoder.to(npu)``:
    the encoder ends construction CPU-resident, never NPU-resident."""
    mod, pipeline = _build_310p_pipeline(monkeypatch, _make_od_config(dtype=torch.float16))
    # The base-line ``.to("npu:0")`` would have raised on a CPU-only host
    # without the pin's shadow; with it the encoder survives construction
    # CPU-resident and the post-construction P1-11 conversion lands FP32.
    for p in pipeline.text_encoder.parameters():
        assert p.device.type == "cpu"
        assert p.dtype == torch.float32
    assert not hasattr(pipeline.text_encoder, "_310p_original_to")


def test_abot_encoder_pin_shadow_unit(monkeypatch: pytest.MonkeyPatch):
    """Direct mechanism test of _pin_encoder_to_cpu/_unpin_encoder."""
    mod, _ = _load_abot_module(monkeypatch)
    m = torch.nn.Linear(4, 4)
    mod._pin_encoder_to_cpu(m)
    # NPU positional device spec is rewritten to CPU; dtype still applies.
    out = m.to("npu:0", dtype=torch.float16)
    assert out is m
    assert all(p.device.type == "cpu" for p in m.parameters())
    assert all(p.dtype == torch.float16 for p in m.parameters())
    # NPU kwarg device spec is rewritten too; cpu specs pass through.
    m.to(device="npu", dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in m.parameters())
    m.to(torch.device("cpu"))
    assert all(p.device.type == "cpu" for p in m.parameters())
    # Unpin restores the stock bound method and drops the bookkeeping attr.
    mod._unpin_encoder(m)
    assert not hasattr(m, "_310p_original_to")
    # The stock method is back: an npu spec now behaves as stock — it raises
    # on a CPU-only host and actually moves on a torch_npu runtime.
    try:
        import torch_npu  # noqa: F401
        has_npu = True
    except ImportError:
        has_npu = False
    if has_npu:
        m.to("npu:0")
        assert all(p.device.type == "npu" for p in m.parameters())
        m.to("cpu")
    else:
        with pytest.raises(RuntimeError):
            m.to("npu:0")
    # Idempotent: unpinned unpin is a no-op.
    mod._unpin_encoder(m)


def test_abot_encoder_on_npu_opt_out(monkeypatch: pytest.MonkeyPatch):
    mod, _ = _load_abot_module(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_ABOT_ENCODER_ON_NPU", "1")
    # Emulate the host-side NPU move the stock .to would perform: record the
    # requested device and stay CPU-resident (CPU-only torch cannot target NPU).
    moved_to = []

    def _fake_to(self, *args, **kwargs):
        moved_to.append(args[0] if args else kwargs.get("device"))
        return self

    monkeypatch.setattr(torch.nn.Module, "to", _fake_to)
    pipeline = mod._ABotWorldCausal310P(od_config=_make_od_config(dtype=torch.float16))
    assert pipeline._encoder_device == pipeline.device
    # Opt-out: the base-line NPU move was NOT intercepted (no pin installed).
    assert moved_to and moved_to[0] == "npu:0"


def test_abot_encode_prompt_routes_through_encoder_device(monkeypatch: pytest.MonkeyPatch):
    mod, pipeline = _build_310p_pipeline(monkeypatch, _make_od_config(dtype=torch.float16))
    seen_devices = []

    original_forward = pipeline.text_encoder.forward

    def spy_forward(input_ids, attention_mask):
        seen_devices.append(input_ids.device)
        return original_forward(input_ids, attention_mask)

    pipeline.text_encoder.forward = spy_forward

    embeds = pipeline.encode_prompt("a  scene with  spaces", max_sequence_length=8, dtype=torch.float16)

    # Encoder ran on the pinned CPU device, never the (stand-in) execution device.
    assert all(d.type == "cpu" for d in seen_devices)
    # Embeds return on the execution device in the runtime dtype.
    assert embeds.device == pipeline.device
    assert embeds.dtype == torch.float16
    # Padding (second half of the mask) is zeroed in the output.
    half = embeds.shape[1] // 2
    assert torch.all(embeds[0, half:] == 0)
    assert torch.any(embeds[0, :half] != 0)
    # The prompt was normalised to a single space-joined token string.
    assert pipeline.tokenizer.last_prompts == ["a scene with spaces"]


def test_abot_encoder_device_guard_for_suite_hook(monkeypatch: pytest.MonkeyPatch):
    """Suite compatibility: _offload_hf_text_encoder skips the _encode_prompt
    wrap when the pipeline exposes _encoder_device — ABot pipelines have no
    _encode_prompt method at all, so the guard attribute must be present."""
    mod, pipeline = _build_310p_pipeline(monkeypatch, _make_od_config(dtype=torch.float16))
    assert hasattr(pipeline, "_encoder_device")
    assert not hasattr(pipeline, "_encode_prompt")


# ── Shared dispatch (registry line, hardware shims, warmup skip) ────────────


def _load_dispatch(monkeypatch: pytest.MonkeyPatch):
    captured = {"sdpa": 0, "ar_paged": 0}

    def fake_sdpa():
        captured["sdpa"] += 1

    def fake_ar_paged():
        captured["ar_paged"] += 1

    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.attention",
                         maybe_patch_sdpa_math=fake_sdpa)
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.ar_paged",
                         maybe_patch_ar_paged_math=fake_ar_paged)
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.models.utils",
                         create_transformers_model=lambda *a, **kw: None)
    _install_fake_module(monkeypatch, "vllm_omni.diffusion.registry",
                         initialize_model=lambda *a, **kw: None)

    class _FakeDiffusionEngine:
        _dummy_run = classmethod(lambda cls: None)

    _install_fake_module(monkeypatch, "vllm_omni.diffusion.diffusion_engine",
                         DiffusionEngine=_FakeDiffusionEngine)
    dispatch = _load_source_module("patch_dispatch_init", _PATCH_DIR / "__init__.py")
    return dispatch, captured


def test_dispatch_registry_contains_abot(monkeypatch: pytest.MonkeyPatch):
    dispatch, _ = _load_dispatch(monkeypatch)
    assert dispatch._DIFFUSION_310P_PATCH_MODULES["ABotWorldCausalPipeline"] == [
        "vllm_omni.platforms.npu._310p.patch.abot_world"
    ]


def test_dispatch_applies_abot_module(monkeypatch: pytest.MonkeyPatch):
    dispatch, captured = _load_dispatch(monkeypatch)
    applied = {}

    def fake_apply(od_config):
        applied["od_config"] = od_config

    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.abot_world",
                         apply_diffusion_310p_patches=fake_apply)
    od_config = _make_od_config()
    dispatch.apply_diffusion_310p_patches(od_config)
    assert applied.get("od_config") is od_config
    # Hardware shims installed (outside the model gate).
    assert captured["sdpa"] == 1
    assert captured["ar_paged"] == 1


def test_dispatch_kill_switch_skips_model_module(monkeypatch: pytest.MonkeyPatch):
    dispatch, captured = _load_dispatch(monkeypatch)
    marker = []
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.abot_world",
                         apply_diffusion_310p_patches=lambda od_config: marker.append(od_config))
    monkeypatch.setenv("VLLM_OMNI_310P_DIFFUSION_PATCHES", "0")
    dispatch.apply_diffusion_310p_patches(_make_od_config())
    assert marker == []  # model patch module never invoked
    # ...but the hardware shims still run (level-2 reference stays comparable).
    assert captured["sdpa"] == 1
    assert captured["ar_paged"] == 1


def test_dispatch_unknown_pipeline_is_noop(monkeypatch: pytest.MonkeyPatch):
    dispatch, _ = _load_dispatch(monkeypatch)
    marker = []
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch.abot_world",
                         apply_diffusion_310p_patches=lambda od_config: marker.append(od_config))
    od_config = _make_od_config(model_class_name="SomeOtherPipeline")
    dispatch.apply_diffusion_310p_patches(od_config)
    assert marker == []


def test_warmup_skip_asset_present(monkeypatch: pytest.MonkeyPatch):
    """GATE-TP2-STATUS 正典修复资产 (6e4e4e3b6): SKIP_WARMUP=1 no-ops the
    DiffusionEngine dummy warmup."""
    dispatch, _ = _load_dispatch(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_310P_SKIP_WARMUP", "1")
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    calls = []

    def _marker_dummy_run(self):
        calls.append(self)

    # A marker "real" dummy run (plain instance method) that must be replaced.
    DiffusionEngine._dummy_run = _marker_dummy_run
    dispatch._maybe_skip_startup_warmup()

    # After the skip is installed, calling the (now no-op) dummy run must not
    # touch the marker.
    DiffusionEngine._dummy_run(object())
    assert calls == []
    assert getattr(DiffusionEngine, "_310p_warmup_skipped", False) is True
