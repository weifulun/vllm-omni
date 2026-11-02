# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the MiniCPM-o 4.5 310P patch contract.

The tests load the patch module from source (its only module-level import is
``torch``) and validate:

* the registry dispatch in ``_310p/patch/__init__.py`` routes the four
  MiniCPM-o archs to the right stage patch (thinker/talker share the
  pipeline-level arch and are disambiguated by ``model_stage``);
* ``CodecState310P`` matches the documented semantics of the Triton
  ``_update_codec_state`` / ``_mask_embeddings`` kernels against an
  independent per-row Python reference (CPU, no triton);
* the torch ``mask_logits`` / ``apply_window_penalty`` ports match the
  ``_mask_logits`` / ``_window_penalty`` kernel math;
* the codec-state shim installs over the lazily-imported module;
* the fused-prelude latch and the streaming-encoder dtype gate land;
* the ``minicpmo_4_5_310p.yaml`` overlay carries the 310P config contract.

No NPU, triton, or vllm imports are required (markers: core_model + cpu).
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATCH_DIR = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / _PATCH_DIR).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {_PATCH_DIR}")


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


def _load_minicpmo_patch(monkeypatch: pytest.MonkeyPatch):
    path = _repo_root() / _PATCH_DIR / "minicpmo.py"
    return _load_source_module("vllm_omni_test_310p_minicpmo_patch", path)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_registry_routes_minicpmo_stages_and_stays_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"thinker": 0, "talker": 0, "code2wav": 0, "worker": 0}
    registry_path = _repo_root() / _PATCH_DIR / "__init__.py"
    registry = _load_source_module("vllm_omni_test_310p_minicpmo_registry", registry_path)

    # Fake the whole parent chain so the registry's `from ... import ...`
    # never executes the real (heavy) vllm_omni package __init__ chain.
    for name in (
        "vllm_omni",
        "vllm_omni.platforms",
        "vllm_omni.platforms.npu",
        "vllm_omni.platforms.npu._310p",
        "vllm_omni.platforms.npu._310p.patch",
    ):
        _install_fake_module(monkeypatch, name)  # force: never run the real package __init__

    fake_worker = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.worker",
        apply_patch=lambda: calls.__setitem__("worker", calls["worker"] + 1),
    )
    fake_minicpmo = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.minicpmo",
        apply_thinker_310p_patches=lambda: calls.__setitem__("thinker", calls["thinker"] + 1),
        apply_talker_310p_patches=lambda: calls.__setitem__("talker", calls["talker"] + 1),
        apply_code2wav_310p_patches=lambda: calls.__setitem__("code2wav", calls["code2wav"] + 1),
    )
    # Bind the submodules as attributes so `from ...patch import worker /
    # minicpmo` resolves without an attribute-import fallback.
    patch_pkg = sys.modules["vllm_omni.platforms.npu._310p.patch"]
    setattr(patch_pkg, "worker", fake_worker)
    setattr(patch_pkg, "minicpmo", fake_minicpmo)

    registry.apply_patches()
    registry.apply_patches()  # worker patch once
    registry.apply_model_patches(SimpleNamespace(model_arch="MiniCPMO45Code2Wav", model_stage="code2wav"))
    registry.apply_model_patches(SimpleNamespace(model_arch="MiniCPMO45OmniForConditionalGeneration", model_stage="tts"))
    registry.apply_model_patches(SimpleNamespace(model_arch="MiniCPMO45OmniForConditionalGeneration", model_stage="llm"))
    # The explicit TTS arch also routes to the talker patch.
    registry.apply_model_patches(SimpleNamespace(model_arch="MiniCPMO45OmniTTSForConditionalGeneration", model_stage="tts"))
    # Unknown archs (including other omni models) are untouched.
    registry.apply_model_patches(SimpleNamespace(model_arch="SomeOtherOmni", model_stage="tts"))
    registry.apply_model_patches(SimpleNamespace(model_arch=None))

    assert calls == {"thinker": 1, "talker": 2, "code2wav": 1, "worker": 1}


def _install_minicpmo_parent_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.model_executor.models.minicpmo_4_5",
    ):
        if name not in sys.modules:
            _install_fake_module(monkeypatch, name)


def test_apply_entrypoints_are_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_minicpmo_patch(monkeypatch)
    _install_minicpmo_parent_fakes(monkeypatch)

    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.minicpmo_4_5.codec_state",
        CodecState=object,
        mask_logits=lambda *a: None,
        apply_window_penalty=lambda *a: None,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample",
        _FUSED_PRELUDE_OK=True,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder",
        _SUPPORTED_DTYPES=(torch.bfloat16, torch.float32),
    )

    module.apply_thinker_310p_patches()
    module.apply_thinker_310p_patches()
    module.apply_talker_310p_patches()
    module.apply_talker_310p_patches()
    module.apply_code2wav_310p_patches()
    module.apply_code2wav_310p_patches()

    codec_state_module = sys.modules["vllm_omni.model_executor.models.minicpmo_4_5.codec_state"]
    assert codec_state_module.CodecState is module.CodecState310P
    assert codec_state_module.mask_logits is module.mask_logits_310p
    assert codec_state_module.apply_window_penalty is module.apply_window_penalty_310p
    talker = sys.modules["vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample"]
    assert talker._FUSED_PRELUDE_OK is False
    streaming = sys.modules["vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder"]
    assert torch.float16 in streaming._SUPPORTED_DTYPES
    assert torch.bfloat16 in streaming._SUPPORTED_DTYPES  # original dtypes preserved


# ---------------------------------------------------------------------------
# CodecState310P vs the documented kernel semantics
# ---------------------------------------------------------------------------

_WINDOW = 4
_VOCAB = 16
_EOS = _VOCAB - 1


def _reference_step(state_cols, history, *, decode, code, eos, window, cadence, boundary_steps,
                    code_classes, silent_frames, quiet_stop_frames):
    """Per-row Python reference of the ``_update_codec_state`` kernel math."""
    step, finished, limit, minimum, drain = (int(x) for x in state_cols)
    valid = bool(decode and not finished and code != eos)
    step += int(valid)
    done = bool(finished or (decode and code == eos) or (step >= limit))
    old = [history[min(o + 1, window - 1)] for o in range(window)]
    shifted = [code if o == window - 1 else v for o, v in enumerate(old)]
    updated = shifted if valid else list(history)
    classes = [code_classes[c] for c in updated]
    current = [o >= window - step for o in range(window)]
    silent = [bool(classes[o] & 1) and current[o] and o >= window - silent_frames for o in range(window)]
    quiet = [bool(classes[o] & 2) and current[o] and o >= window - quiet_stop_frames for o in range(window)]
    settled = sum(silent) == silent_frames
    until_silent = drain == 2
    done = bool(done or (until_silent and sum(quiet) == quiet_stop_frames))
    boundary = (drain == 1) and (step >= cadence) and ((step % cadence) < boundary_steps)
    masked = (not done) and ((step < minimum) or boundary or (until_silent and not settled))
    return updated, valid, done, masked, (step, done)


def _scenario(seed: int):
    """Random multi-request scenario shared by the torch path and the reference."""
    rng = random.Random(seed)
    code_classes = torch.tensor(
        [rng.choice([0, 1, 2, 3]) for _ in range(_VOCAB + 1)], dtype=torch.int8
    )
    requests = {}
    for rid in ("r0", "r1", "r2", "r3"):
        requests[rid] = {
            "max_tokens": rng.choice([None, 6, 8]),
            "min_tokens": rng.choice([0, 1, 2]),
            "turn_end_drain": rng.choice([0, 0, 1, 2]),
            "recent_codes": [rng.randrange(_VOCAB) for _ in range(rng.randrange(_WINDOW))],
        }
    code_classes_np = code_classes.tolist()
    # Reference initial history ring: padding + recent codes, last `window` kept
    # (matches CodecState.prepare's staging).
    for rid, cfg in requests.items():
        cfg["init_history"] = ([_VOCAB] * _WINDOW + list(cfg["recent_codes"]))[-_WINDOW:]
    steps = []
    for _ in range(14):
        rows = []
        for rid in ("r0", "r1", "r2", "r3"):
            if rng.random() < 0.85:
                code = _EOS if rng.random() < 0.15 else rng.randrange(_VOCAB)
            else:
                code = None
            rows.append((rid, requests[rid], code))
        steps.append(rows)
    return requests, code_classes, code_classes_np, steps


def test_codec_state_310p_update_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_minicpmo_patch(monkeypatch)
    requests, code_classes, code_classes_np, steps = _scenario(seed=1234)

    codec = module.CodecState310P("cpu", _VOCAB, capacity=8, window=_WINDOW)
    # Reference mirrors the prepare() slot staging.
    ref = {
        rid: {
            "state": [0, 0, int(cfg.get("max_tokens") or (2**31 - 1)) - 1, int(cfg.get("min_tokens") or 0),
                      int(cfg.get("turn_end_drain") or 0)],
            "history": list(cfg["init_history"]),
        }
        for rid, cfg in requests.items()
    }

    cadence, boundary_steps, silent_frames, quiet_stop_frames = 4, 1, 2, 3
    for step_no, rows in enumerate(steps):
        ids = torch.tensor([code if code is not None else 0 for _, _, code in rows], dtype=torch.long)
        out_history, valid, done, masked = codec.update(
            rows,
            ids,
            eos=_EOS,
            cadence=cadence,
            boundary=boundary_steps,
            code_classes=code_classes,
            silent_frames=silent_frames,
            quiet_stop_frames=quiet_stop_frames,
        )

        assert out_history.shape == (len(rows), _WINDOW)
        assert out_history.dtype is torch.long
        for (rid, _, code), hist_row, v, d, m in zip(rows, out_history, valid, done, masked):
            updated, ref_valid, ref_done, ref_masked, ref_state = _reference_step(
                ref[rid]["state"],
                ref[rid]["history"],
                decode=code is not None,
                code=code if code is not None else 0,
                eos=_EOS,
                window=_WINDOW,
                cadence=cadence,
                boundary_steps=boundary_steps,
                code_classes=code_classes_np,
                silent_frames=silent_frames,
                quiet_stop_frames=quiet_stop_frames,
            )
            ref[rid]["history"] = updated
            ref[rid]["state"][0] = ref_state[0]
            ref[rid]["state"][1] = int(ref_state[1])
            assert bool(v) == ref_valid, (step_no, rid, "valid")
            assert bool(d) == ref_done, (step_no, rid, "done")
            assert bool(m) == ref_masked, (step_no, rid, "masked")
            assert hist_row.tolist() == updated, (step_no, rid, "history")
        # The device state ring stays in sync with the reference.
        for rid in {rid for rid, _, _ in rows}:
            slot = codec.slots[rid]
            assert codec.state[slot, 0].item() == ref[rid]["state"][0]
            assert codec.state[slot, 1].item() == ref[rid]["state"][1]
            assert codec.history[slot].tolist() == ref[rid]["history"]


def test_codec_state_310p_slot_growth_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_minicpmo_patch(monkeypatch)
    codec = module.CodecState310P("cpu", _VOCAB, capacity=2, window=_WINDOW)

    for rid in ("a", "b", "c"):  # third request forces the capacity doubling
        codec.update(
            [(rid, {"max_tokens": None, "min_tokens": 0, "turn_end_drain": 0, "recent_codes": []}, 3)],
            torch.tensor([3]),
            eos=_EOS,
            cadence=4,
            boundary=1,
            code_classes=torch.zeros(_VOCAB + 1, dtype=torch.int8),
            silent_frames=2,
            quiet_stop_frames=3,
        )
    assert codec.capacity == 4
    assert set(codec.slots) == {"a", "b", "c"}
    # Slots 0/1 held a/b; growth added 2/3; c popped 2, so only 3 is free.
    assert sorted(codec.free_slots) == [3]

    codec.release("a")
    assert codec.slots.keys() == {"b", "c"}
    codec.release("a")  # idempotent
    assert sorted(codec.free_slots) == [0, 3]


def test_codec_state_310p_mask_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_minicpmo_patch(monkeypatch)
    codec = module.CodecState310P("cpu", _VOCAB, capacity=4, window=_WINDOW)

    rows = [("r0", {"turn_end_drain": 0}, 1), ("r1", {"turn_end_drain": 0}, _EOS)]
    out = codec.update(rows, torch.tensor([1, _EOS]), eos=_EOS, cadence=4, boundary=1,
                       code_classes=torch.zeros(_VOCAB + 1, dtype=torch.int8), silent_frames=2,
                       quiet_stop_frames=3)
    del out
    assert codec.state[codec.slots["r1"], 1].item() == 1  # r1 hit eos -> finished

    embeds = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    codec.mask_embeddings(embeds, [codec.slots["r0"], codec.slots["r1"]], [0, 1])
    torch.testing.assert_close(embeds[0], torch.arange(3, dtype=torch.float32))
    torch.testing.assert_close(embeds[1], torch.zeros(3))


# ---------------------------------------------------------------------------
# mask_logits / apply_window_penalty torch ports
# ---------------------------------------------------------------------------


def test_mask_logits_310p_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_minicpmo_patch(monkeypatch)
    rng = random.Random(7)
    vocab = 16
    rows = 4
    logits = torch.tensor([[rng.uniform(-3, 3) for _ in range(vocab)] for _ in range(rows)])
    forced = torch.tensor([True, False, False, True])
    masked = torch.tensor([False, True, False, False])
    eos = 15

    result = module.mask_logits_310p(logits.clone(), forced, masked, eos)
    for r in range(rows):
        expected = torch.tensor(logits[r].tolist())
        if forced[r]:
            expected = torch.full((vocab,), float("-inf"))
            expected[eos] = 0.0
        elif masked[r]:
            expected[eos] = float("-inf")
        torch.testing.assert_close(result[r], expected)
    # In-place contract: same storage as the input.
    logits2 = torch.randn(2, vocab)
    assert module.mask_logits_310p(logits2, torch.zeros(2, dtype=torch.bool),
                                   torch.zeros(2, dtype=torch.bool), eos) is logits2


def test_apply_window_penalty_310p_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_minicpmo_patch(monkeypatch)
    rng = random.Random(11)
    rows, vocab, window = 3, 8, 4
    logits = torch.tensor([[rng.uniform(-4, 4) for _ in range(vocab)] for _ in range(rows)])
    histories = torch.tensor([[rng.randrange(vocab) for _ in range(window)] for _ in range(rows)])
    penalties = torch.tensor([1.0, 1.1, 0.95])

    result = module.apply_window_penalty_310p(logits.clone(), histories, penalties)
    assert result.shape == logits.shape
    for r in range(rows):
        frequencies = torch.bincount(histories[r], minlength=vocab).to(torch.float32)
        alpha = penalties[r] ** frequencies
        values = logits[r].to(torch.float32)
        expected = torch.where(values < 0, values * alpha, values / alpha)
        torch.testing.assert_close(result[r].to(torch.float32), expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# codec_state shim (Triton-import-failure path)
# ---------------------------------------------------------------------------


def test_codec_state_shim_installs_when_module_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_minicpmo_patch(monkeypatch)
    name = "vllm_omni.model_executor.models.minicpmo_4_5.codec_state"
    # Marking the module None makes `from ... import codec_state` raise, which
    # forces the eager-torch shim. monkeypatch restores sys.modules on teardown.
    monkeypatch.setitem(sys.modules, name, None)

    module._install_codec_state_310p()

    shim = sys.modules[name]
    assert shim.CodecState is module.CodecState310P
    assert shim.mask_logits is module.mask_logits_310p
    assert shim.apply_window_penalty is module.apply_window_penalty_310p


# ---------------------------------------------------------------------------
# deploy overlay contract
# ---------------------------------------------------------------------------


def test_310p_overlay_disables_graphs_and_tunes_fp16_budget() -> None:
    path = _repo_root() / "vllm_omni" / "deploy" / "minicpmo_4_5_310p.yaml"
    overlay = yaml.safe_load(path.read_text())

    assert overlay["base_config"] == "minicpmo_4_5.yaml"
    assert overlay["dtype"] == "float16"

    stages = {s["stage_id"]: s for s in overlay["stages"]}
    assert stages[0]["tensor_parallel_size"] == 2
    assert stages[0]["devices"] == "0,1"
    assert stages[0]["max_model_len"] == 8192
    assert stages[1]["devices"] == "0"
    assert stages[2]["devices"] == "1"

    npu = {s["stage_id"]: s for s in overlay["platforms"]["npu"]["stages"]}
    assert npu[0]["additional_config"]["encoder_enable_npu_graph"] is False
    assert npu[2]["additional_config"]["code2wav_enable_npu_graph"] is False
    assert npu[1]["kv_cache_memory_bytes"] == 1073741824  # 1 GiB, right-sized for 4k talker context

    extra = overlay["connectors"]["connector_of_shared_memory"]["extra"]
    for key in ("talker_codec_fused_prelude", "enable_hift_graph", "enable_cfm_graph",
                "enable_code2wav_encoder_graph"):
        assert extra[key] is False, key


def test_310p_overlay_merges_over_base_without_dropping_contract() -> None:
    """Re-implement the repo's stage/platform/connectors merge for the overlay
    and verify the merged 310P deploy keeps the inherited runtime contract
    (speculative decode, stop ids, codec chunk settings)."""
    root = _repo_root() / "vllm_omni" / "deploy"
    base = yaml.safe_load((root / "minicpmo_4_5.yaml").read_text())
    overlay = yaml.safe_load((root / "minicpmo_4_5_310p.yaml").read_text())

    def deep_merge(original, update):
        merged = dict(original)
        for key, value in (update or {}).items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def merge_stage_lists(base_stages, overlay_stages):
        by_id = {s["stage_id"]: dict(s) for s in (base_stages or [])}
        for stage in overlay_stages or []:
            sid = stage["stage_id"]
            by_id[sid] = deep_merge(by_id.get(sid, {}), stage)
        return list(by_id.values())

    stages = merge_stage_lists(base.get("stages"), overlay.get("stages"))
    base_platforms = (base.get("platforms") or {}).get("npu", {})
    overlay_platforms = (overlay.get("platforms") or {}).get("npu", {})
    npu_stages = merge_stage_lists(base_platforms.get("stages"), overlay_platforms.get("stages"))
    extra = deep_merge(
        (base.get("connectors") or {}).get("connector_of_shared_memory", {}).get("extra"),
        (overlay.get("connectors") or {}).get("connector_of_shared_memory", {}).get("extra"),
    )

    by_id = {s["stage_id"]: s for s in stages}
    assert by_id[0]["tensor_parallel_size"] == 2
    assert by_id[0]["devices"] == "0,1"
    assert by_id[0]["interleave_mm_strings"] is True  # inherited
    assert by_id[1]["max_model_len"] == 4096  # inherited talker context
    assert by_id[2]["enforce_eager"] is True  # inherited
    assert by_id[2]["max_model_len"] == 65536  # inherited code2wav context

    npu_by_id = {s["stage_id"]: s for s in npu_stages}
    assert npu_by_id[0]["additional_config"]["encoder_enable_npu_graph"] is False
    assert npu_by_id[0]["additional_config"]["encoder_max_npu_graphs"] == 4  # inherited key, superseded below
    assert npu_by_id[0]["compilation_config"]["cudagraph_mode"] == "PIECEWISE"  # inherited
    assert npu_by_id[1]["speculative_config"]["method"] == "ngram"  # inherited K-step
    assert npu_by_id[1]["default_sampling_params"]["stop_token_ids"] == [1]
    assert npu_by_id[1]["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert npu_by_id[1]["kv_cache_memory_bytes"] == 1073741824
    assert npu_by_id[2]["additional_config"]["cfm_graph_batch_buckets"] == [1, 2, 4, 8]  # inherited
    assert npu_by_id[2]["additional_config"]["code2wav_enable_npu_graph"] is False

    assert extra["talker_codec_fused_prelude"] is False
    assert extra["enable_hift_graph"] is False
    assert extra["enable_cfm_graph"] is False
    assert extra["enable_code2wav_encoder_graph"] is False
    assert extra["cfm_fused_body"] is True  # inherited NPU fused body (ACLNN ops, 310P-safe)
    assert extra["codec_chunk_frames"] == 25  # inherited
