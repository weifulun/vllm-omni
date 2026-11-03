# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the MiniMax Music 3 310P patches.

The host has no NPU and (in CI) no importable vllm, so the minimax_music3
model modules are faked in ``sys.modules`` and the patch module is loaded
from source, the same way the Qwen3-TTS patch contract test does it. Where
a reference implementation matters (the sampler arithmetic, the streaming
resampler, the depth-decoder block) the *real* upstream source is carved
out of the repository files and executed, so the tests compare the 310P
seams against upstream behavior rather than against a copy.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATCH_SOURCE = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch" / "minimax_music3.py"
_DISPATCH_SOURCE = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
_MODEL_DIR = Path("vllm_omni") / "model_executor" / "models" / "minimax_music3"


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


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _class_from_source(path: Path, class_name: str, namespace: dict) -> type:
    """Carve one class out of a real source file and exec it standalone."""
    source = path.read_text()
    tree = ast.parse(source)
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if node is None:
        raise AssertionError(f"{class_name} not found in {path}")
    ns = dict(namespace)
    exec(compile(ast.get_source_segment(source, node), str(path), "exec"), ns)
    return ns[class_name]


def _fake_logger(name: str):
    import logging

    return logging.getLogger(name)


class _FakeAcousticModel(nn.Module):
    """Just enough surface for the patched loader test."""

    def __init__(self, dit: nn.Module, vocoder: nn.Module) -> None:
        super().__init__()
        self.dit = dit
        self.vocoder = vocoder
        self.model_path = "/fake/model"
        self.vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(revision=None),
            load_config=types.SimpleNamespace(download_dir=None),
            device_config=types.SimpleNamespace(device=torch.device("cpu")),
        )
        self.dit_steps = 2
        self.dit_cfg_scale = 1.7
        self._loaded = False


class _FakeRequestARState:
    """The talker's per-pair state surface the symmetric-pair-stop seam sees."""

    def __init__(self, *, seed: int, max_frames: int, prompt_tokens: int = 0) -> None:
        self.seed = seed
        self.max_frames = max_frames
        self.prompt_tokens = prompt_tokens


def _fake_state_for(self, pair_id, row):
    """The stock ``_state_for`` semantics against the fake state class."""
    state = self._states.get(pair_id)
    if state is None:
        state = _FakeRequestARState(
            seed=self._row_seeds[row],
            max_frames=self._row_max_frames[row],
            prompt_tokens=self._row_prompt_tokens[row],
        )
        self._states[pair_id] = state
    return state


def _install_minimax_fakes(monkeypatch: pytest.MonkeyPatch, root: Path):
    """Install fake minimax_music3 modules; return (patch module, fakes)."""
    pkg_name = "vllm_omni.model_executor.models.minimax_music3"

    # Fake the whole parent chain so the import machinery never touches the
    # real vllm_omni package (whose __init__ imports vllm, which is broken on
    # the py3.9 host and absent under CI pythons without a full env).
    omni = _install_fake_module(monkeypatch, "vllm_omni")
    model_executor = _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    models = _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    omni.model_executor = model_executor
    model_executor.models = models

    # Real sampling source (torch-only imports) under the fake dotted name:
    # the seams must be tested against the upstream arithmetic itself.
    _install_fake_module(
        monkeypatch,
        f"{pkg_name}.constants",
        DIT_LATENT_CHANNELS=128,
        DEFAULT_DIT_STEPS=30,
        DEFAULT_DIT_CFG_SCALE=1.7,
        AR_CFG_SCALE=1.5,
        AR_CFG_TOP_K=50,
        AR_TOP_K=50,
    )
    sampling = _load_source_module(f"{pkg_name}.sampling", root / _MODEL_DIR / "sampling.py")
    monkeypatch.setitem(sys.modules, f"{pkg_name}.sampling", sampling)

    dav_path = root / _MODEL_DIR / "dav.py"
    real_resampler = _class_from_source(
        dav_path,
        "StreamingResampler",
        namespace={
            "torch": torch,
            "Tensor": Tensor,
            "nn": nn,
            "math": math,
            "Sequence": Sequence,
            "init_logger": _fake_logger,
            "DAV_SAMPLE_RATE": 44_100,
            "OUTPUT_SAMPLE_RATE": 32_000,
            "OUTPUT_CHANNELS": 2,
        },
    )

    def _remove_weight_norm(module: nn.Module) -> int:
        return 0

    def _derive_seed(seed: int, *parts: object) -> int:
        value = seed
        for part in parts:
            value = (value * 131 + sum(map(ord, str(part))) + len(str(part))) & 0xFFFFFFFF
        return value

    pkg = _install_fake_module(monkeypatch, pkg_name)
    pkg.__path__ = []  # marks it as a package

    acoustic = _install_fake_module(
        monkeypatch,
        f"{pkg_name}.acoustic",
        MiniMaxMusic3AcousticForConditionalGeneration=_FakeAcousticModel,
        _derive_seed=_derive_seed,
    )
    dav = _install_fake_module(
        monkeypatch,
        f"{pkg_name}.dav",
        StreamingResampler=real_resampler,
        remove_weight_norm=_remove_weight_norm,
    )
    dit = _install_fake_module(
        monkeypatch,
        f"{pkg_name}.dit",
        MiniMaxMusic3FlowMatchingDiT=type("MiniMaxMusic3FlowMatchingDiT", (nn.Module,), {}),
        MiniMaxMusic3Transformer1DModel=type("MiniMaxMusic3Transformer1DModel", (nn.Module,), {}),
        Attention=type("Attention", (nn.Module,), {}),
        remap_transformer_state=lambda state: state,
        _SIGMA_FLOOR=1e-6,
    )
    rvq = _install_fake_module(
        monkeypatch,
        f"{pkg_name}.rvq_decoder",
        DepthDecoderBlock=type("DepthDecoderBlock", (nn.Module,), {}),
    )
    class _FakePinnedIndexStage:
        def __init__(self, rows: int, capacity: int, device) -> None:
            self.capacity = capacity

        def push(self, values, count):
            return torch.tensor([list(values[0])[:count]], dtype=torch.int64)

    class _FakeTalker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frame_feedback_active = torch.zeros(4, dtype=torch.bool)
            self.frame_feedback_codes = torch.zeros((4, 3), dtype=torch.long)
            self._feedback_stage = None
            self._feedback_positions = None
            self._feedback_codes = None
            self._max_rows = 4

        def _write_frame_feedback(self, feedback_rows, feedback_codes, device):
            raise AssertionError("stock frame-feedback writer must be replaced")

    _FakeTalker._state_for = _fake_state_for

    talker = _install_fake_module(
        monkeypatch,
        f"{pkg_name}.talker",
        apply_c0_cfg=sampling.apply_c0_cfg,
        apply_depth_cfg=sampling.apply_depth_cfg,
        draw_from_gumbel=sampling.draw_from_gumbel,
        gumbel_noise=sampling.gumbel_noise,
        sample_topk_seeded=sampling.sample_topk_seeded,
        MiniMaxMusic3TalkerForConditionalGeneration=_FakeTalker,
        PinnedIndexStage=_FakePinnedIndexStage,
        RequestARState=_FakeRequestARState,
    )
    _install_fake_module(
        monkeypatch,
        f"{pkg_name}.chunking",
        crop_sample_bounds=lambda window: (window.start, window.end),
        overlap_mel_length=lambda: 24,
    )
    _install_fake_module(
        monkeypatch,
        f"{pkg_name}.weights",
        CONDITION_ENCODER_DIR="condition_encoder",
        TRANSFORMER_DIR="transformer",
        VOCODER_DIR="vocoder",
        load_component_state=lambda root, component, **kw: {},
        resolve_repo_root=lambda *a, **kw: "/fake/root",
    )
    for name, module in (
        ("acoustic", acoustic),
        ("dav", dav),
        ("dit", dit),
        ("rvq_decoder", rvq),
        ("talker", talker),
    ):
        setattr(pkg, name, module)
    models.minimax_music3 = pkg

    # vllm is broken on the py3.9 host; the patch module only needs its logger.
    if "vllm" not in sys.modules or not hasattr(sys.modules.get("vllm", None), "logger"):
        _install_fake_module(monkeypatch, "vllm")
        _install_fake_module(monkeypatch, "vllm.logger", init_logger=_fake_logger)

    patch = _load_source_module("vllm_omni.platforms.npu._310p.patch.minimax_music3", root / _PATCH_SOURCE)
    return patch, {"sampling": sampling, "acoustic": acoustic, "dav": dav, "dit": dit, "rvq": rvq, "talker": talker}


def _reset_patch_state(patch: types.ModuleType) -> None:
    patch._TALKER_PATCHED = False
    patch._ACOUSTIC_PATCHED = False
    patch._NOISE_FALLBACK = False
    if hasattr(patch, "_ORIG_STATE_FOR"):
        patch._ORIG_STATE_FOR = None


def _fresh(monkeypatch: pytest.MonkeyPatch, root: Path):
    """Install fakes, load the patch module, and reset its latch state."""
    patch, fakes = _install_minimax_fakes(monkeypatch, root)
    _reset_patch_state(patch)
    return patch, fakes


# ---------------------------------------------------------------------------
#  Stage 0: sampler seams + depth-decoder attention
# ---------------------------------------------------------------------------


def test_talker_apply_rebinds_and_is_idempotent(monkeypatch):
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    talker, sampling = fakes["talker"], fakes["sampling"]

    patch.apply_talker_patches()
    patch.apply_talker_patches()  # idempotent

    for name in ("apply_c0_cfg", "apply_depth_cfg", "draw_from_gumbel", "gumbel_noise", "sample_topk_seeded"):
        assert getattr(talker, name) is not getattr(sampling, name), name
    # The generic sampling module itself is never mutated.
    assert sampling.gumbel_noise.__name__ == "gumbel_noise"


def test_talker_kill_switch_leaves_talker_untouched(monkeypatch):
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    talker, sampling = fakes["talker"], fakes["sampling"]
    monkeypatch.setenv(patch._KILL_SWITCH_ENV, "0")

    patch.apply_talker_patches()

    for name in ("apply_c0_cfg", "apply_depth_cfg", "draw_from_gumbel", "gumbel_noise", "sample_topk_seeded"):
        assert getattr(talker, name) is getattr(sampling, name), name
    assert not patch._TALKER_PATCHED


def test_sampler_seams_are_exact_cpu_passthrough(monkeypatch):
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    talker, sampling = fakes["talker"], fakes["sampling"]
    patch.apply_talker_patches()

    torch.manual_seed(7)
    rows, vocab = 3, 64
    logits = torch.randn(rows, vocab)
    seeds = torch.tensor([11, 42, 9001])
    positions = torch.tensor([0, 5, 99])

    # CPU inputs: the seams must call the upstream primitives unchanged and
    # return bit-identical results (the deterministic draws are the contract).
    assert torch.equal(
        talker.gumbel_noise(seeds, positions, vocab),
        sampling.gumbel_noise(seeds, positions, vocab),
    )
    gumbel = sampling.gumbel_noise(seeds, positions, vocab)
    assert torch.equal(talker.draw_from_gumbel(logits, gumbel), sampling.draw_from_gumbel(logits, gumbel))
    cond, uncond = torch.randn(rows, vocab), torch.randn(rows, vocab)
    assert torch.equal(talker.apply_c0_cfg(cond, uncond), sampling.apply_c0_cfg(cond, uncond))
    assert torch.equal(talker.apply_depth_cfg(cond, uncond), sampling.apply_depth_cfg(cond, uncond))
    assert torch.equal(
        talker.sample_topk_seeded(logits, seeds, positions),
        sampling.sample_topk_seeded(logits, seeds, positions),
    )
    # Determinism across calls is what the seed-exact double run relies on.
    assert torch.equal(
        talker.sample_topk_seeded(logits, seeds, positions),
        talker.sample_topk_seeded(logits, seeds, positions),
    )


def test_depth_decoder_attention_matches_upstream_sdpa(monkeypatch):
    root = _repo_root()
    patch, _ = _fresh(monkeypatch, root)

    rvq_path = root / _MODEL_DIR / "rvq_decoder.py"
    block_cls = _class_from_source(
        rvq_path,
        "DepthDecoderBlock",
        namespace={
            "torch": torch,
            "Tensor": Tensor,
            "nn": nn,
            "F": F,
            "SDPBackend": torch.nn.attention.SDPBackend,
            "sdpa_kernel": torch.nn.attention.sdpa_kernel,
            "_SDPA_BACKENDS": [torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.MATH],
            "AUDIO_VOCAB_SIZE": 1024,
            "BACKBONE_HIDDEN_SIZE": 4096,
            "NUM_CODEBOOKS": 8,
        },
    )
    torch.manual_seed(3)
    block = block_cls(hidden_size=32, num_heads=4, intermediate_size=64)
    x = torch.randn(2, 6, 32)

    h_patched = patch._depth_decoder_block_forward_310p(block, x)
    ref = block_cls.forward(block, x)  # upstream body, stock sdpa on CPU
    assert torch.allclose(h_patched, ref, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
#  Stage 1: FP16 pin, resample/attention/RoPE/noise seams
# ---------------------------------------------------------------------------


def test_acoustic_apply_rebinds_and_is_idempotent(monkeypatch):
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    acoustic, dit, dav = fakes["acoustic"], fakes["dit"], fakes["dav"]

    patch.apply_acoustic_patches()
    patch.apply_acoustic_patches()  # idempotent

    assert (
        acoustic.MiniMaxMusic3AcousticForConditionalGeneration._load_components
        is patch._load_components_310p
    )
    assert acoustic.MiniMaxMusic3AcousticForConditionalGeneration._decode_window is patch._decode_window_310p
    assert dit.MiniMaxMusic3FlowMatchingDiT.forward is patch._dit_forward_310p
    assert dit.MiniMaxMusic3FlowMatchingDiT.aligned_condition is patch._aligned_condition_310p
    assert dit.MiniMaxMusic3Transformer1DModel._rope is patch._rope_310p
    assert dit.Attention._attend is patch._attend_310p
    assert dav.StreamingResampler.push is patch._resampler_push_310p


def test_acoustic_kill_switch_leaves_classes_untouched(monkeypatch):
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    dit, dav = fakes["dit"], fakes["dav"]
    monkeypatch.setenv(patch._KILL_SWITCH_ENV, "0")

    patch.apply_acoustic_patches()

    assert not patch._ACOUSTIC_PATCHED
    assert dav.StreamingResampler.push.__name__ == "push"
    assert not hasattr(dit.MiniMaxMusic3FlowMatchingDiT, "aligned_condition")


def test_load_components_pins_float16(monkeypatch):
    root = _repo_root()
    patch, _ = _fresh(monkeypatch, root)

    class _SwallowLoad(nn.Linear):
        """Linear whose load_state_dict accepts the fake (empty) checkpoints."""

        def load_state_dict(self, state, strict=True):
            return types.SimpleNamespace(out_keys=[], missing_keys=[], unexpected_keys=[])

    class _FakeDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.condition_encoder = _SwallowLoad(4, 4, bias=False)
            self.transformer = _SwallowLoad(4, 4, bias=False)

    dit = _FakeDiT()
    vocoder = _SwallowLoad(8, 8, bias=False)
    model = _FakeAcousticModel(dit, vocoder)

    patch._load_components_310p(model)

    assert model._loaded
    assert all(p.dtype == torch.float16 for p in model.dit.parameters())
    assert all(p.dtype == torch.float16 for p in model.vocoder.parameters())
    assert all(p.dtype == torch.float16 for p in model.parameters())


def test_nearest_gather_is_bit_exact_vs_interpolate(monkeypatch):
    root = _repo_root()
    patch, _ = _fresh(monkeypatch, root)
    torch.manual_seed(11)
    for in_len in (7, 33, 250, 1000):
        for mel_len in (1, 3, in_len, in_len * 2, 689):
            align = torch.randn(1, 16, in_len)
            expected = F.interpolate(align, size=mel_len, mode="nearest")
            assert torch.equal(patch._nearest_gather_310p(align, mel_len), expected), (in_len, mel_len)


def test_attend_310p_matches_sdpa(monkeypatch):
    root = _repo_root()
    patch, _ = _fresh(monkeypatch, root)
    torch.manual_seed(5)
    q = torch.randn(2, 4, 16, 8)
    k = torch.randn(2, 4, 16, 8)
    v = torch.randn(2, 4, 16, 8)
    carrier = types.SimpleNamespace()
    for is_causal in (False, True):
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        got = patch._attend_310p(carrier, q, k, v, is_causal=is_causal)
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4), is_causal


def test_rope_310p_matches_upstream_tables(monkeypatch):
    root = _repo_root()
    patch, _ = _fresh(monkeypatch, root)
    torch.manual_seed(13)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2).float() / 32))
    carrier = types.SimpleNamespace(inv_freq=inv_freq, _rope_cache={})

    for seq_len in (1, 17, 689):
        for dtype in (torch.float32, torch.float16):
            got = patch._rope_310p(carrier, seq_len, dtype=dtype, device=torch.device("cpu"))
            # Upstream arithmetic, computed the way the stock method does it.
            t = torch.arange(seq_len, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype)
            expected = (freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1))
            assert got[0].shape == expected[0].shape
            assert torch.allclose(got[0], expected[0], atol=1e-3, rtol=1e-3)
            assert torch.allclose(got[1], expected[1], atol=1e-3, rtol=1e-3)
            # Cached on the second call under the same key.
            again = patch._rope_310p(carrier, seq_len, dtype=dtype, device=torch.device("cpu"))
            assert again[0] is got[0]


def test_noise_draw_cpu_fallback_is_deterministic(monkeypatch):
    root = _repo_root()
    patch, _ = _fresh(monkeypatch, root)
    patch._NOISE_FALLBACK = True
    a = patch._draw_noise_310p((2, 8, 32), torch.device("cpu"), torch.float16, torch.Generator(), 42)
    b = patch._draw_noise_310p((2, 8, 32), torch.device("cpu"), torch.float16, torch.Generator(), 42)
    c = patch._draw_noise_310p((2, 8, 32), torch.device("cpu"), torch.float16, torch.Generator(), 43)
    assert a.dtype == torch.float16
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    patch._NOISE_FALLBACK = False


def test_resampler_push_keeps_streaming_semantics(monkeypatch):
    # The streaming resampler runs torchaudio's polyphase resample; the
    # container (which has torchaudio) is the authoritative environment for
    # this test.
    pytest.importorskip("torchaudio")
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    resampler_cls = fakes["dav"].StreamingResampler
    original_push = resampler_cls.push  # captured before the rebind

    torch.manual_seed(21)
    ref = resampler_cls(orig_freq=44100, new_freq=32000)
    patched = resampler_cls(orig_freq=44100, new_freq=32000)
    resampler_cls.push = patch._resampler_push_310p

    expected: list[Tensor] = []
    got: list[Tensor] = []
    for i in range(6):
        chunk = torch.randn(2, 173 * (i + 1))
        flush = i == 5
        expected.append(original_push(ref, [chunk], flush=flush))
        got.append(patched.push([chunk], flush=flush))

    assert len(expected) == len(got)
    for e, g in zip(expected, got):
        assert e.shape == g.shape
        assert torch.equal(e, g)


# ---------------------------------------------------------------------------
#  Dispatch + ipc defense line
# ---------------------------------------------------------------------------


def test_dispatch_routes_each_arch(monkeypatch):
    root = _repo_root()

    calls = []

    class _FakeApply:
        def __init__(self, name):
            self.name = name

        def __call__(self):
            calls.append(self.name)

    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p.patch")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=_FakeApply("qwen3_tts.talker"),
        apply_code2wav_patches=_FakeApply("qwen3_tts.code2wav"),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.minimax_music3",
        apply_talker_patches=_FakeApply("mm3.talker"),
        apply_acoustic_patches=_FakeApply("mm3.acoustic"),
    )
    dispatch = _load_source_module(
        "vllm_omni.platforms.npu._310p.patch.__init__",
        root / _DISPATCH_SOURCE,
    )

    def _arch(name: str):
        return types.SimpleNamespace(model_arch=name)

    dispatch.apply_model_patches(_arch("Qwen3TTSTalkerForConditionalGeneration"))
    dispatch.apply_model_patches(_arch("MiniMaxMusic3TalkerForConditionalGeneration"))
    dispatch.apply_model_patches(_arch("MiniMaxMusic3ForConditionalGeneration"))  # dual spelling
    dispatch.apply_model_patches(_arch("MiniMaxMusic3AcousticForConditionalGeneration"))
    dispatch.apply_model_patches(_arch("SomeOtherArchitecture"))

    assert calls == [
        "qwen3_tts.talker",
        "mm3.talker",
        "mm3.talker",
        "mm3.acoustic",
    ]


def test_ipc_routes_npu_storages_to_shm():
    root = _repo_root()
    ipc_path = root / "vllm_omni" / "diffusion" / "ipc.py"
    source = ipc_path.read_text()

    # The defense line exists: NPU storages are unpicklable and the pack
    # decision consults it.
    assert "def _tensor_storage_unpicklable(t: torch.Tensor) -> bool:" in source
    assert 'return t.device.type == "npu"' in source
    assert "or _tensor_storage_unpicklable(val)" in source

    # Load only the pure predicate from source and exercise it.
    tree = ast.parse(source)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tensor_storage_unpicklable"
    )
    ns = {"torch": torch}
    exec(compile(ast.get_source_segment(source, node), str(ipc_path), "exec"), ns)
    unpicklable = ns["_tensor_storage_unpicklable"]

    class _Dev:
        def __init__(self, t):
            self.type = t

    class _FakeTensor:
        def __init__(self, t):
            self.device = _Dev(t)

    assert unpicklable(_FakeTensor("npu"))
    assert not unpicklable(_FakeTensor("cpu"))
    assert not unpicklable(_FakeTensor("cuda"))


def test_talker_embedding_nz_seam(monkeypatch):
    """The 310P embedding method must not keep a fractal-NZ duplicate.

    The talker's untied embed_tokens and lm_head (vocab 200000) would pin
    ~3.3 GB of permanent NZ copies on the 21 GB chip, so the seam skips
    the copy and applies F.linear on the plain fp16 weight.
    """
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)

    class _Method:
        def process_weights_after_loading(self, layer):
            layer.weight_nz = layer.weight.flip(0).clone()  # stand-in NZ copy

        def apply(self, layer, x, bias=None):
            return F.linear(x, layer.weight_nz, bias)

    ascend = _install_fake_module(monkeypatch, "vllm_ascend")
    p310 = _install_fake_module(monkeypatch, "vllm_ascend._310p")
    ops = _install_fake_module(monkeypatch, "vllm_ascend._310p.ops")
    vpe = _install_fake_module(
        monkeypatch,
        "vllm_ascend._310p.ops.vocab_parallel_embedding",
        AscendUnquantizedEmbeddingMethod310=_Method,
    )
    ascend._310p = p310
    p310.ops = ops
    ops.vocab_parallel_embedding = vpe

    patch.apply_talker_patches()

    cls = vpe.AscendUnquantizedEmbeddingMethod310
    inst = cls()
    layer = nn.Linear(4, 8)
    inst.process_weights_after_loading(layer)
    assert not hasattr(layer, "weight_nz"), "the NZ copy must not be created"
    x = torch.randn(2, 4)
    out = inst.apply(layer, x)
    assert out.device == x.device, "logits must round-trip to the input device"
    assert torch.equal(out, F.linear(x, layer.weight))


def test_embedding_seam_follows_weight_device(monkeypatch):
    """The quant_method.embedding seam must look up on the weight's device
    and hand the result back on the input's device."""
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)

    class _Method:
        def embedding(self, layer, input_):
            return F.embedding(input_, layer.weight)

        def apply(self, layer, x, bias=None):
            return F.linear(x, getattr(layer, "weight_nz", layer.weight), bias)

    ascend = _install_fake_module(monkeypatch, "vllm_ascend")
    p310 = _install_fake_module(monkeypatch, "vllm_ascend._310p")
    ops = _install_fake_module(monkeypatch, "vllm_ascend._310p.ops")
    vpe = _install_fake_module(
        monkeypatch,
        "vllm_ascend._310p.ops.vocab_parallel_embedding",
        AscendUnquantizedEmbeddingMethod310=_Method,
    )
    ascend._310p = p310
    p310.ops = ops
    ops.vocab_parallel_embedding = vpe

    patch.apply_talker_patches()

    inst = vpe.AscendUnquantizedEmbeddingMethod310()
    layer = nn.Embedding(16, 4)
    ids = torch.tensor([[3, 7, 1]])
    out = inst.embedding(layer, ids)
    assert out.device == ids.device
    assert torch.equal(out, F.embedding(ids, layer.weight))


def test_talker_embedding_offload_hook(monkeypatch):
    """The post-load offload moves embed_tokens and lm_head to the host
    for the talker architecture and leaves other models untouched."""
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)

    class _Talker(nn.Module):
        def __init__(self):
            super().__init__()
            inner = nn.Module()
            inner.embed_tokens = nn.Embedding(16, 4)
            self.model = inner
            self.lm_head = nn.Linear(4, 16)

    worker = types.SimpleNamespace(model=_Talker())
    patch._offload_talker_embeddings_310p(worker)
    assert worker.model.lm_head.weight.device.type == "cpu"
    assert worker.model.model.embed_tokens.weight.device.type == "cpu"

    class _NotMusic3(nn.Module):
        def __init__(self):
            super().__init__()
            self.touched = False

    other = types.SimpleNamespace(model=_NotMusic3())
    patch._offload_talker_embeddings_310p(other)
    assert not other.model.touched


def test_write_frame_feedback_twin_fills_flags_without_index_fill(monkeypatch):
    """The 310P twin must publish frame feedback without the index_fill_
    kernel (ACLNN 161002 gap) and keep the upstream flag semantics."""
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    patch.apply_talker_patches()

    talker_cls = fakes["talker"].MiniMaxMusic3TalkerForConditionalGeneration
    assert talker_cls._write_frame_feedback is patch._write_frame_feedback_310p

    model = talker_cls()
    codes = [torch.full((1, 3), 7), torch.full((1, 3), 9)]
    model._write_frame_feedback([1, 3], codes, torch.device("cpu"))
    assert model.frame_feedback_active.tolist() == [False, True, False, True]
    assert model._feedback_positions is not None
    assert model.frame_feedback_codes[1].tolist() == [7, 7, 7]
    assert model.frame_feedback_codes[3].tolist() == [9, 9, 9]

    # Empty step: flags reset and positions cleared.
    model._write_frame_feedback([], [], torch.device("cpu"))
    assert not model.frame_feedback_active.any()
    assert model._feedback_positions is None


# ---------------------------------------------------------------------------
#  Stage 0: symmetric pair stop (KV-thin CFG budget)
# ---------------------------------------------------------------------------


def _pair_stop_stub(max_frames: int):
    class _Stub:
        _states = {}
        _row_seeds = [42]
        _row_max_frames = [max_frames]
        _row_prompt_tokens = [1380]

    return _Stub()


def test_pair_stop_shrinks_frame_budget_exactly_once(monkeypatch):
    """The seam shortens the model-side frame budget by the control-token
    slack at state creation and never applies it twice (cached pair state)."""
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    monkeypatch.delenv(patch._PAIR_STOP_SLACK_ENV, raising=False)
    patch.apply_talker_patches()

    talker_cls = fakes["talker"].MiniMaxMusic3TalkerForConditionalGeneration
    assert talker_cls._state_for is not _fake_state_for  # wrapped
    stub = _pair_stop_stub(3200)
    state = talker_cls._state_for(stub, "pair-1", 0)
    assert state.max_frames == 3200 - patch._PAIR_STOP_SLACK_DEFAULT
    # Same pair: the cached state comes back untouched.
    again = talker_cls._state_for(stub, "pair-1", 0)
    assert again is state
    assert again.max_frames == 3200 - patch._PAIR_STOP_SLACK_DEFAULT
    # A fresh pair is shrunk from the FULL budget again.
    fresh = talker_cls._state_for(stub, "pair-2", 0)
    assert fresh.max_frames == 3200 - patch._PAIR_STOP_SLACK_DEFAULT


def test_pair_stop_slack_zero_keeps_stock_budget(monkeypatch):
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    monkeypatch.setenv(patch._PAIR_STOP_SLACK_ENV, "0")
    patch.apply_talker_patches()

    state = fakes["talker"].MiniMaxMusic3TalkerForConditionalGeneration._state_for(
        _pair_stop_stub(3200), "pair-1", 0
    )
    assert state.max_frames == 3200


def test_pair_stop_kill_switch_leaves_state_for_stock(monkeypatch):
    root = _repo_root()
    patch, fakes = _fresh(monkeypatch, root)
    monkeypatch.setenv(patch._KILL_SWITCH_ENV, "0")
    talker_cls = fakes["talker"].MiniMaxMusic3TalkerForConditionalGeneration
    stock = talker_cls._state_for
    patch.apply_talker_patches()  # no-op with the kill switch
    assert talker_cls._state_for is stock
