# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch MOSS-TTS-Nano (single-stage 0.1B AR TTS) for the 310P NPU path.

Nano is architecturally different from the 8B family (``moss_tts.py``): it is
a **single stage** whose ``forward()`` shell drives a trust_remote_code
transformers LM (``AutoModelForCausalLM``, GPT-2 decoder) plus the
MOSS-Audio-Tokenizer-Nano codec inside ``inference_stream()``.  Neither the LM
nor the codec is vLLM-managed; both run eagerly (deploy yaml
``enforce_eager: true``, ``max_num_seqs: 1`` — there is **no** NPUGraph
wrapper and no CUDA-graph path on NPU, the tokenizer's flash-kvcache branch
is CUDA-only).

Against the 8B 310P skeleton this patch therefore:

* **reuses**: the FP16-forcing semantics (``_RUNTIME_DTYPE`` = fp16 for
  model parts that are not vLLM-managed) and the
  ``vllm_ascend.sample.sampler.random_sample`` sampling operator;
* **trims**: the ``torch_npu`` shim (Nano's remote code contains no
  ``npu_fusion_attention`` / ``npu_rotary_mul`` hooks — the GPT-2 decoder and
  the tokenizer both run plain SDPA with portable RoPE, with the GPT-2
  attention implementation forced to ``sdpa`` by
  ``_configure_attention_implementation`` on NPU) and the NPUGraph
  ``_streaming_graph_batch_sizes_from_compilation_config`` override (nothing
  to disable — there is no graph wrapper in this model);
* **adds**: a scoped ``torch.multinomial`` guard.  The remote LM samples both
  text and audio tokens with a plain ``torch.multinomial(probs,
  num_samples=1)`` on device tensors (remote
  ``_sample_next_token``); on 310P that op is the same CPU-NPU-sync /
  non-capturable hazard the 8B patch removes at the vllm-ascend sampler
  level, but Nano's sampling happens *inside* remote code, so the only
  reachable patch point is ``torch.multinomial`` itself.  The guard only
  intercepts the exact single-sample/replacement/NPU-tensor form the remote
  LM uses and rewrites it to ``random_sample`` (Gumbel-max + NPU exponential
  generator, device-global RNG — parity with the 910 path); every other
  call shape delegates to the original implementation untouched.

Generic model modules stay unchanged; registration swaps the class only when
the 310P platform applies the patch (see ``_310p/patch/__init__.py``).

Stage-0 arch (``model_config.model_arch``): ``MossTTSNanoForCausalLM``.
"""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.moss_tts_nano import modeling_moss_tts_nano

_RUNTIME_DTYPE = torch.float16

_NANO_PATCHED = False
_MULTINOMIAL_GUARD_INSTALLED = False
_ORIGINAL_MULTINOMIAL = None


def _cast_module_to_310p_dtype(module) -> None:
    """Force a not-vLLM-managed module (remote LM / audio tokenizer) to fp16.

    910/CUDA run these sub-models in bf16/fp32; 310P is FP16-first, so both
    sub-models are recast wholesale (params + floating-point buffers, which
    includes rotary ``inv_freq``).

    TODO(310P-real): verify on-device SDPA additive-mask backend, fp16 RoPE
    (``inv_freq`` cast to fp16 changes cos/sin numerics vs the 910 fp32
    baseline) and tokenizer streaming-state dtypes; if audio quality regresses,
    keep ``inv_freq``-style buffers in fp32 via a targeted allow-list.
    """
    if module is None:
        return
    with torch.no_grad():
        module.to(dtype=_RUNTIME_DTYPE)


def _multinomial_310p_path(input: torch.Tensor, num_samples: int, replacement: bool, out) -> bool:
    """Whether this ``torch.multinomial`` call should use the 310P rewrite.

    Only the exact form used by the Nano remote LM
    (``torch.multinomial(probs, num_samples=1)`` on an NPU floating-point
    tensor) is intercepted; everything else falls through to the original
    implementation.  Kept as a module-level hook so contract tests can force
    either branch without a real NPU tensor.
    """
    return (
        input.device.type == "npu"
        and input.dim() >= 1
        and input.is_floating_point()
        and num_samples == 1
        and replacement
        and out is None
    )


def _install_multinomial_guard() -> None:
    """Wrap ``torch.multinomial`` for NPU-tensor single-sample calls.

    Idempotent.  The wrapped function keeps a reference to the original so a
    second install (or a future consumer) never double-wraps.
    """
    global _MULTINOMIAL_GUARD_INSTALLED, _ORIGINAL_MULTINOMIAL
    if _MULTINOMIAL_GUARD_INSTALLED:
        return
    _ORIGINAL_MULTINOMIAL = torch.multinomial

    def multinomial_310p(input, num_samples=1, replacement=True, *, out=None):
        if _multinomial_310p_path(input, num_samples, replacement, out):
            from vllm_ascend.sample.sampler import random_sample

            flat = input.reshape(-1, input.shape[-1])
            # ``{}`` = device-global generator: parity with the 910 path,
            # which also sampled from the device-global RNG state.
            sampled = random_sample(flat, {})  # (n_rows,) long
            # torch.multinomial contract: input.shape[:-1] + (num_samples,)
            return sampled.reshape(*input.shape[:-1], 1)
        return _ORIGINAL_MULTINOMIAL(input, num_samples, replacement=replacement, out=out)

    torch.multinomial = multinomial_310p
    _MULTINOMIAL_GUARD_INSTALLED = True


def _wrap_audio_tokenizer_encode(at: torch.nn.Module) -> None:
    """310P: the remote ``_call_audio_encode`` force-casts the prompt waveform
    to fp32 (``torch.as_tensor(waveform, dtype=torch.float32)``), while the
    recast tokenizer parameters are fp16 — aclnn is strict about
    weightOptional/input dtype (EZ1001) and has no implicit cast.

    Cast the encode input to the tokenizer's parameter dtype at the
    ``batch_encode`` boundary. ``_prepare_waveform_batch`` derives the padded
    buffer dtype from ``wav_list[0].dtype``, so the cast propagates through
    the whole encoder. Semantics-preserving: fp32 -> fp16 on the encode path
    only (identical to what the fp16 weights already imply numerically).
    """
    if not hasattr(at, "batch_encode"):
        # Test fakes / tokenizers without an encode path: nothing to cast.
        return
    orig = at.batch_encode

    def _param_dtype() -> torch.dtype | None:
        for p in at.parameters():
            if p.is_floating_point():
                return p.dtype
        return None

    def batch_encode_310p(wav_list, *args, **kwargs):
        dt = _param_dtype()
        if dt is not None:
            casted = []
            for w in wav_list:
                w = torch.as_tensor(w)
                if w.is_floating_point() and w.dtype != dt:
                    w = w.to(dt)
                casted.append(w)
            wav_list = casted
        return orig(wav_list, *args, **kwargs)

    at.batch_encode = batch_encode_310p


class _MossTTSNano310P(modeling_moss_tts_nano.MossTTSNanoForGeneration):
    """Nano on 310P: FP16-forced remote LM + audio tokenizer.

    The base class builds both sub-models in ``__init__`` (eager, so that
    ``load_format: dummy`` works) — we hook right after that and recast them.
    Attention stays on the NPU SDPA path chosen by the base class'
    ``_configure_attention_implementation`` (``sdpa`` on NPU); nothing else is
    changed.
    """

    def __init__(self, vllm_config, prefix: str = "") -> None:
        # Upstream __init__ is keyword-only (``def __init__(self, *, vllm_config, prefix)``);
        # a positional super() call raises TypeError at engine init.
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _cast_module_to_310p_dtype(self._lm)
        _cast_module_to_310p_dtype(self._audio_tokenizer)
        _wrap_audio_tokenizer_encode(self._audio_tokenizer)


def apply_nano_patches() -> None:
    """Swap in the 310P Nano class and install the multinomial guard.

    Idempotent: safe to call on every model load of this arch.
    """
    global _NANO_PATCHED
    if _NANO_PATCHED:
        return

    modeling_moss_tts_nano.MossTTSNanoForGeneration = _MossTTSNano310P
    _install_multinomial_guard()
    _NANO_PATCHED = True
