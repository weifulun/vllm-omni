# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P patch for the OmniVoice diffusion TTS pipeline (task-23).

Gate contract (COMPARISON-POLICY.md §5): this file registers ONLY its
self-gated entry point; ``NPUOmniPlatform.init_diffusion_model_runner_runtime``
calls it once before the diffusion pipeline is built.  It is a no-op unless
``is_310p()`` AND ``od_config.model_class_name`` is the OmniVoice pipeline.

Why this hook at all: ``apply_model_patches(model_config)`` only runs inside
the AR ``OmniNPUModelRunner.load_model``; OmniVoice is a single-stage
*diffusion* pipeline (``StageExecutionType.DIFFUSION``) built by
``DiffusionModelRunner.load_model``, which calls
``init_diffusion_model_runner_runtime`` first.

310P-specific hazards found in the v0.32.0 code (machine-verified context
from sibling tasks on the same image / CANN 9.1.0 stack):

1. **Hard-coded BF16 attention cast.** ``OmniVoiceAttention.forward``
   permutes Q/K/V and casts them to ``torch.bfloat16`` unconditionally.
   310P is FP16-only (POLICY §2); a BF16 tensor fed to the NPU SDPA path is
   a guaranteed failure.  ``_OmniVoiceAttention310P`` re-casts to the
   *runtime* (hidden-state) dtype instead — fp16 under the 310P deploy
   overlay, and identical to upstream on 910 (where hidden states are
   already bf16).

2. **Fused SDPA kernel unsupported on 310P3 (machine-verified on 126/129,
   task-16):** ``F.scaled_dot_product_attention`` fails ERR01007
   "OPS feature not supported" for fp16/bf16/fp32, masked and unmasked.
   The TORCH_SDPA backend's module entry ``sdpa._sdpa_attention`` is
   re-bound to an eager decomposition (scale -> matmul -> mask -> softmax
   -> matmul), numerically equivalent; chunked online-softmax above the
   dense byte budget.

3. **RoPE table RealDiv on NPU (012 family).** ``_precompute_rope_table``
   computes ``1.0 / (theta ** arange)`` device-locally; on 310P the TBE
   dynamic real_div pre-compile gap (research/012) makes any NPU-side
   RealDiv in the loader path a first-forward hazard.  The table is tiny
   (4096 x 128 fp32): it is now precomputed on CPU and moved by the caller
   (``_rope_table_for`` already does ``.to(device, dtype)``).

4. **Device RNG.** ``torch.Generator(device="npu")`` + device-local
   ``torch.rand(generator=...)`` is used for Gumbel sampling.  If the
   real-device probe reports the NPU RNG unusable, generator creation is
   re-routed to CPU and ``_gumbel_sample`` draws on CPU, then moves the
   noise to the device — same distribution, same seed sequence.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from vllm_omni.platforms.npu._310p import is_310p

logger = logging.getLogger(__name__)

# Registry contract (COMPARISON-POLICY.md §5): platform.py maps
# model_class_name -> patch function through this entry point.
#
# Both registry keys resolve to the same pipeline class (diffusion/registry.py);
# the *resolved* ``od_config.model_class_name`` is the architecture name from
# config.json (``architectures: ["OmniVoice"]``), so the gate must accept both
# spellings.  Machine-verified on 126/C3 (2026-11-03): the resolved value is
# "OmniVoice".
MODEL_ARCH = "OmniVoice"
MODEL_ARCH_NAMES = ("OmniVoice", "OmniVoicePipeline")

_SDPA_DENSE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
_SDPA_CHUNK_SIZE = 512

_APPLIED = False
_NPU_RNG_OK: bool | None = None  # probe result cache (None = not probed)


# ---------------------------------------------------------------------------
# Pure, CPU-testable decision helpers
# ---------------------------------------------------------------------------


def should_patch(od_config: Any) -> bool:
    """True only on 310P hardware for the OmniVoice diffusion pipeline."""
    model_arch = getattr(od_config, "model_class_name", None)
    if model_arch not in MODEL_ARCH_NAMES:
        return False
    return is_310p()


def probe_npu_rng() -> bool:
    """Once per process: can ``torch.Generator(device='npu')`` + device rand
    actually run here?  Cached on the module."""
    global _NPU_RNG_OK
    if _NPU_RNG_OK is not None:
        return _NPU_RNG_OK
    try:
        if not torch.npu.is_available():
            _NPU_RNG_OK = False
            return _NPU_RNG_OK
        device = torch.device("npu", torch.npu.current_device())
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        torch.rand((8,), generator=gen, device=device, dtype=torch.float32)
        torch.npu.synchronize(device)
        _NPU_RNG_OK = True
    except Exception as exc:  # noqa: BLE001 - any failure means "use CPU RNG"
        logger.info("310P OmniVoice: NPU RNG probe failed (%s); using CPU RNG", exc)
        _NPU_RNG_OK = False
    return _NPU_RNG_OK


# ---------------------------------------------------------------------------
# 2. Manual SDPA (eager decomposition) — machine-verified on 310P3
# ---------------------------------------------------------------------------


def _sdpa_gqa_expand(query: Any, key: Any, value: Any):
    """GQA expansion parity with upstream ``_sdpa_attention``."""
    if query.shape[1] == key.shape[1]:
        return query, key, value
    if query.shape[1] % key.shape[1]:
        raise ValueError(
            "GQA requires query heads a multiple of KV heads, "
            f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}."
        )
    repeat = query.shape[1] // key.shape[1]
    return query, key.repeat_interleave(repeat, dim=1), value.repeat_interleave(repeat, dim=1)


def manual_sdpa_forward(
    query: Any, key: Any, value: Any,
    attention_mask: Any, causal: bool, scale: float,
) -> Any:
    """Eager SDPA decomposition for 310P3.

    Same contract as the backend's ``_sdpa_attention``: (B,H,S,D) in,
    (B,H,S,D) out; ``attention_mask`` may be a bool keep-mask (True = keep,
    torch SDPA semantics) or a float additive bias; ``causal`` adds the
    lower-triangular mask.  Dense path below the byte budget; chunked
    online-softmax above it.
    """
    query, key, value = _sdpa_gqa_expand(query, key, value)
    batch, heads, seq, dim = query.shape
    dense_bytes = batch * heads * seq * key.shape[2] * query.element_size()
    if dense_bytes > _SDPA_DENSE_LIMIT_BYTES:
        return _chunked_sdpa_forward(query, key, value, attention_mask, causal, scale)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if causal:
        # Top-left alignment (row i keeps keys j <= i): the semantics torch's
        # CPU/MATH backend applies to ``is_causal``.  Identical to bottom-right
        # for the square shapes OmniVoice actually uses (dense mask, never
        # causal; the branch exists for defensive parity only).
        sq, sk = scores.shape[-2], scores.shape[-1]
        tri = torch.ones(sq, sk, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~tri, float("-inf"))
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)


def _chunked_sdpa_forward(
    query: Any, key: Any, value: Any,
    attention_mask: Any, causal: bool, scale: float,
) -> Any:
    """Online-softmax attention over key chunks (fp32 accumulators)."""
    batch, heads, seq, dim = query.shape
    seq_k = key.shape[2]
    out = torch.zeros(batch, heads, seq, dim, device=query.device, dtype=torch.float32)
    rowmax = torch.full((batch, heads, seq, 1), float("-inf"), device=query.device, dtype=torch.float32)
    rowsum = torch.zeros_like(rowmax)
    for start in range(0, seq_k, _SDPA_CHUNK_SIZE):
        stop = min(start + _SDPA_CHUNK_SIZE, seq_k)
        chunk = stop - start
        s = torch.matmul(query, key[:, :, start:stop].transpose(-2, -1)).float()
        s = s * scale
        if causal:
            # Top-left: global key start+col is kept iff start+col <= i,
            # i.e. col <= i - start.
            tri = torch.ones(seq, chunk, dtype=torch.bool, device=s.device).tril(-start)
            s = s.masked_fill(~tri, float("-inf"))
        if attention_mask is not None:
            part = attention_mask[..., start:stop]
            if part.dtype == torch.bool:
                s = s.masked_fill(~part, float("-inf"))
            else:
                s = s + part.float()
        new_max = torch.maximum(rowmax, s.amax(dim=-1, keepdim=True))
        with torch.no_grad():
            alpha = torch.exp(rowmax - new_max)
        probs = torch.exp(s - new_max)
        rowsum = rowsum * alpha + probs.sum(dim=-1, keepdim=True)
        out = out * alpha + probs @ value[:, :, start:stop].float()
        rowmax = new_max
    return (out / rowsum).to(query.dtype)


def _patch_attention_manual_sdpa() -> None:
    """Swap the TORCH_SDPA backend entry for the eager decomposition."""
    from vllm_omni.diffusion.attention.backends import sdpa as sdpa_mod

    if getattr(sdpa_mod, "_310p_manual_sdpa_patched", False):
        return
    sdpa_mod._sdpa_attention = manual_sdpa_forward
    sdpa_mod._310p_manual_sdpa_patched = True
    logger.info(
        "310P OmniVoice: TORCH_SDPA backend swapped to eager manual "
        "decomposition (fused SDPA kernel unsupported on 310P3, CANN 9.1.0)."
    )


# ---------------------------------------------------------------------------
# 1. Attention runtime-dtype cast (upstream hard-casts BF16)
# ---------------------------------------------------------------------------


def _patch_attention_dtype() -> None:
    """Rebind ``OmniVoiceAttention`` to the 310P subclass.

    The only behavioral change versus the parent: the Q/K/V prologue output
    is cast to the hidden-state (runtime) dtype instead of a hard-coded
    ``torch.bfloat16``.  On 910 the hidden states are bf16, so this is a
    no-op there; on 310P (FP16 overlay) it keeps the attention inputs fp16.
    """
    from vllm_omni.model_executor.models.omnivoice import omnivoice_generator as gen_mod
    from vllm_omni.model_executor.models.omnivoice.fused_qkv_rope import fused_qkv_norm_rope

    if getattr(gen_mod, "_310p_attention_dtype_patched", False):
        return

    OmniVoiceAttention = gen_mod.OmniVoiceAttention

    class _OmniVoiceAttention310P(OmniVoiceAttention):
        def forward(
            self,
            hidden_states: torch.Tensor,
            rope_table: torch.Tensor,
            attn_metadata: Any,
        ) -> torch.Tensor:
            seq_len, _ = hidden_states.shape
            qkv = self.qkv_proj(hidden_states).view(1, seq_len, self.num_qkv_heads, self.head_dim)
            q, k, v = fused_qkv_norm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                rope_table,
                self.q_norm.eps,
                self.num_heads,
                self.num_kv_heads,
            )
            # 310P: runtime dtype (fp16 under the overlay) instead of BF16.
            dtype = hidden_states.dtype
            q, k, v = (t.permute(0, 2, 1, 3).contiguous().to(dtype) for t in (q, k, v))
            if self.use_packed_varlen:
                out = self.attention_op(q, k, v, attn_metadata)
            else:
                out = self.attention_op.sdpa_fallback.forward(q, k, v, attn_metadata)
            out = out.squeeze(0).to(hidden_states.dtype).reshape(seq_len, self.num_heads * self.head_dim)
            return self.o_proj(out)

    _OmniVoiceAttention310P.__name__ = "OmniVoiceAttention310P"
    _OmniVoiceAttention310P.__qualname__ = "OmniVoiceAttention310P"
    gen_mod.OmniVoiceAttention = _OmniVoiceAttention310P
    gen_mod._310p_attention_dtype_patched = True
    logger.info("310P OmniVoice: attention cast re-bound to runtime dtype (was hard-coded BF16).")


# ---------------------------------------------------------------------------
# 3. RoPE table precomputed on CPU (012 family: no NPU RealDiv in loader path)
# ---------------------------------------------------------------------------


def _precompute_rope_table_cpu(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """CPU-side twin of ``omnivoice_generator._precompute_rope_table``.

    Same layout (``[max_seq_len, head_dim]``, first half cos, second half
    sin), computed entirely on CPU.  Callers (``_rope_table_for``) already
    move/cast the table to the target device and dtype.
    """
    del device
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def _patch_rope_table_cpu() -> None:
    from vllm_omni.model_executor.models.omnivoice import omnivoice_generator as gen_mod

    if getattr(gen_mod, "_310p_rope_cpu_patched", False):
        return
    gen_mod._precompute_rope_table = _precompute_rope_table_cpu
    gen_mod._310p_rope_cpu_patched = True
    logger.info("310P OmniVoice: RoPE table precomputed on CPU (avoids NPU RealDiv, 012 family).")


# ---------------------------------------------------------------------------
# 4. Device-RNG fallback (activated only when the probe fails)
# ---------------------------------------------------------------------------


def gumbel_sample_cpu(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> torch.Tensor:
    """Device-agnostic Gumbel sampling.

    Upstream draws ``torch.rand`` on the tensor's device with a matching
    device generator.  When the NPU generator is unusable (fallback mode)
    the generator lives on CPU; draw there and move the noise — same
    distribution, same per-seed stream.
    """
    same_device = generator is not None and generator.device.type == logits.device.type
    if same_device:
        noise = -torch.log(
            -torch.log(
                torch.rand(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype).clamp(min=1e-8)
            )
        )
    else:
        noise = -torch.log(
            -torch.log(
                torch.rand(logits.shape, generator=generator, dtype=logits.dtype).clamp(min=1e-8)
            )
        )
        noise = noise.to(device=logits.device, dtype=logits.dtype)
    return logits / max(temperature, 1e-8) + noise


def _patch_rng_fallback() -> None:
    """Re-route generator creation + Gumbel draws through the CPU fallback.

    Installed only when ``probe_npu_rng()`` reports the NPU RNG unusable.
    ``torch.Generator`` is re-bound process-wide to a thin subclass that
    accepts ``device='npu'`` but materializes a CPU generator; because the
    only generator consumers in this process are the two OmniVoice sites
    (generator.forward / pipeline prepare-denoise), the blast radius is
    exactly the model's own sampling.
    """
    from vllm_omni.model_executor.models.omnivoice import omnivoice_generator as gen_mod

    if getattr(torch.Generator, "_310p_cpu_compat", False):
        return

    _RealGenerator = torch.Generator

    class _Generator310PCompat(_RealGenerator):
        def __new__(cls, device=None):
            dev_type = None
            if device is not None:
                try:
                    dev_type = torch.device(device).type
                except RuntimeError:
                    # torch builds without an NPU backend cannot parse the
                    # device string; classify by prefix instead (CPU test
                    # hosts exercise this branch).
                    dev_type = str(device).split(":")[0].split("(")[0].strip()
            if dev_type == "npu":
                return _RealGenerator.__new__(cls)  # CPU generator
            return _RealGenerator.__new__(cls, device)

    _Generator310PCompat._310p_cpu_compat = True  # type: ignore[attr-defined]
    torch.Generator = _Generator310PCompat  # type: ignore[misc,assignment]

    # The pipeline passes generator objects into the generator's denoise
    # path; the only Gumbel draw site is ``omnivoice_generator._gumbel_sample``
    # (resolved from module globals at call time), so one rebind covers both
    # the eager forward and the step-execution path.
    gen_mod._gumbel_sample = gumbel_sample_cpu
    logger.info("310P OmniVoice: device RNG re-routed to CPU fallback (NPU generator unusable).")


# ---------------------------------------------------------------------------
# 5. Startup dummy-run skip (main-process side; the warmup executes in the
#    engine factory, not in the worker where the hook above lands)
# ---------------------------------------------------------------------------


def maybe_skip_startup_warmup_310p(od_config: Any) -> None:
    """Self-gated: neutralize ``DiffusionEngine.run_startup_warmup`` on 310P.

    Called from ``DiffusionEngine.make_engine`` right before the warmup call
    (the main process).  The dummy forward hits a GE format-cast wall on this
    stack (``can not cast format when output is input`` + ERR01007 at dummy
    graph build, machine-verified on 126/C3 2026-11-02); the real request
    drives its own eager compile.  Pattern: task-16
    ``_patch_skip_startup_warmup`` (Qwen-Image-2.1, same box).  Re-enable once
    the CANN-side gap is fixed.
    """
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    if getattr(DiffusionEngine, "_310p_warmup_skipped", False):
        return
    if not should_patch(od_config):
        return

    def _skip_startup_warmup(self):
        logger.info(
            "310P OmniVoice: startup dummy-run warmup skipped (GE format-cast "
            "wall at dummy graph build); the first real request compiles "
            "eagerly and pays the cold-compile cost."
        )

    DiffusionEngine.run_startup_warmup = _skip_startup_warmup
    DiffusionEngine._310p_warmup_skipped = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def maybe_apply_omnivoice_310p_patches(od_config: Any) -> None:
    """Self-gated 310P patch entry for the OmniVoice diffusion pipeline.

    No-op off 310P or for any other model_class_name; idempotent.
    """
    global _APPLIED
    if _APPLIED or not should_patch(od_config):
        return
    _APPLIED = True

    logger.info("310P OmniVoice: applying 310P patches (model_class_name=%s)", MODEL_ARCH)
    _patch_attention_dtype()
    _patch_attention_manual_sdpa()
    _patch_rope_table_cpu()
    if not probe_npu_rng():
        _patch_rng_fallback()
    else:
        logger.info("310P OmniVoice: NPU device RNG verified; keeping upstream generator path.")
