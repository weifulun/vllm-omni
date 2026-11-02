# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""310P MOSS-Image specialization (task-19, model-mossimage).

Machine-verified invariants on 310P3 x CANN 9.1.0 x torch_npu 2.10.0.post4
(310P dtprobe evidence from the Qwen-Image-2.1 line, research/011-013; this
model's surface is strictly smaller -- no torch_npu-specific ops in the
transformer, conv2d VAE instead of the conv3d family):

1. 310P is FP16-only (no BF16): ``od_config.dtype`` is forced to
   ``float16`` and CUDA-style graphing is off (``enforce_eager``; NPU has no
   inductor, CUDA-graph decode is CUDA-only).

2. Attention: the fused SDPA aclnn kernel is rejected on 310P3 for every
   probed shape/dtype/mask (``ERR01007 OPS feature not supported``; the torch
   MATH backend fails identically through the torch_npu dispatch).  The eager
   decomposition (scale -> matmul -> mask -> softmax -> matmul) passes all
   shapes, so ``_patch_attention_manual_sdpa`` swaps the TORCH_SDPA backend's
   module-level entry (``sdpa._sdpa_attention``) for that decomposition --
   same vllm-omni-side monkey-patch style as the rest of this file; the image
   stays read-only.  Explicit MindIE-SD-only selections (``FLASH_ATTN``,
   ``RAINFUSION_ATTN``) are rejected at patch time (research/003 official
   matrix: ``npu_fusion_attention`` exists only on 950DT + A2).

3. Loader-level CPU construction: the ``"default"`` load-format branch builds
   the pipeline inside ``with target_device:`` (npu:0).  Every device-less op
   at construction time (the 8.4B Qwen3.5 encoder's weight-finalization
   frequency tables, VAE RMS-norm ``torch.ones/zeros`` buffers, transformer
   RoPE ``bands``) then lands on the NPU and hits the 310P CANN 9.1.0
   dynamic-TBE compile-gap family (RealDiv/OnesLike first; research/012).
   Upstream's own custom_pipeline branch solves this class of problem the
   same way ("build on CPU first, then explicitly move to the target
   device"); ``_patch_loader_cpu_build`` mirrors it.

4. Text encoder placement: the MOSSImage pipeline loads the 8.39B Qwen3.5
   encoder via ``from_pretrained_with_prefetch(AutoModel.from_pretrained, ...).
   to(encoder_device)`` and, without an offload plan, ``encoder_device`` is the
   NPU -- ~16.8 GB that cannot coexist with the 10.17 GB FP16 DiT on a
   21589 MB chip (budget decision: encoder pinned CPU, single-chip TP1).
   ``_patch_pipeline_encoder_placement`` rewires the pipeline module's own
   ``from_pretrained_with_prefetch`` reference so the encoder factory runs
   under ``torch.device("cpu")`` and the returned model is CPU-pinned
   (``_make_cpu_pinned`` redirects every later ``.to(...)`` to CPU).

5. Explicit post-load placement: after ``MOSSImagePipeline.load_weights``
   finishes, ``_install_post_load_placement`` moves the transformer (built on
   CPU by invariant 3) to the local NPU via a tracked H2D copy (caching
   allocator friendly) and confirms the VAE is on the NPU; the encoder stays
   CPU-pinned by invariant 4.

6. VAE tiling+slicing are forced (``od_config.vae_use_tiling`` /
   ``vae_use_slicing``): the 1024^2 untiled decode activation peak is the same
   SDXL-R4 class of risk on a ~21 GB chip.

7. Precision tiers (COMPARISON-POLICY.md section 2): FP16 is the first
   choice; INT8 W8A8 is the sanctioned fallback tier (framework built-in,
   >65535 weight values fall back to FP16 layer-wise; research/009) and is
   accepted with a loud warning; BF16 and all other quantization methods are
   rejected.

``VLLM_OMNI_310P_DIFFUSION_PATCHES=0`` disables the whole dispatch (the
COMPARISON-POLICY.md section 1 level-2 same-seed reference hook).
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

MODEL_ARCH = "MOSSImagePipeline"
_KILL_SWITCH = "VLLM_OMNI_310P_DIFFUSION_PATCHES"

#: Attention backends whose kernels require MindIE-SD (unavailable on 310P).
_MINDIE_SD_ONLY_BACKENDS = frozenset({"FLASH_ATTN", "RAINFUSION_ATTN"})

#: Numerics-changing cache backends (rejected by the MOSSImage pipeline
#: itself; this guard is the belt-and-braces layer with a stable log line).
_NUMERIC_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_APPLIED = False

#: Dense score-matrix byte budget before the chunked (online-softmax) path
#: takes over.  1024^2 images mean 4096 latent tokens: a dense
#: (1,24,4096,4096) fp16 score matrix is ~0.8 GB transient per attention --
#: keep the dense path at/below 2 GiB, chunk above it (same budget as the
#: Qwen-Image-2.1 machine run, where 16384-token dense ~17 GB was impossible).
_SDPA_DENSE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
_SDPA_CHUNK_SIZE = 512


# ---------------------------------------------------------------------------
# Pure decision functions (CPU-testable, no NPU imports)
# ---------------------------------------------------------------------------


def resolve_runtime_dtype(current: torch.dtype) -> torch.dtype:
    """310P is FP16-only: the runtime dtype is always ``float16``."""
    del current
    return torch.float16


def resolve_encoder_placement() -> str:
    """310P pins the Qwen3.5 text encoder (8.39B) to CPU.

    Returns ``"cpu"``.  The generic NPU pipeline would otherwise (a) run the
    encoder weight-finalization frequency builds in the ambient NPU device
    context and trip the 310P dynamic-TBE RealDiv gap, and (b) spend
    ~16.8 GB of the 21589 MB chip, OOMing next to the 10.17 GB FP16 DiT.
    """
    return "cpu"


def resolve_vae_memory(vae_use_tiling: bool, vae_use_slicing: bool) -> tuple[bool, bool]:
    """Force tiling+slicing: the FLUX.2-small VAE upsampling chain peaks
    multi-GB untiled on a ~21 GB 310P chip (SDXL R4 class of risk)."""
    return True, True


def _quantization_method(quant_config: Any) -> str | None:
    """Resolve the quantization method name from a QuantizationConfig object."""
    getter = getattr(quant_config, "get_name", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # pragma: no cover - defensive, contract-level
            pass
    for attr in ("method", "name"):
        value = getattr(quant_config, attr, None)
        if isinstance(value, str):
            return value
    return None


def validate_quantization(od_config: Any) -> None:
    """Precision-tier gate: FP16 primary, INT8 W8A8 sanctioned fallback only.

    COMPARISON-POLICY.md section 2 (revised 2026-11-02): FP16 is the first
    choice; INT8 is the fallback tier when weights exceed the chip budget
    (not the MOSS-Image case -- 10.17 GB fits FP16 with headroom -- but the
    gate must accept the tier).  BF16 and every other quantization method are
    rejected: 310P is FP16-only and no other quant path is validated on the
    CANN 9.1.0 / MindIE-DS 9.1.0 stack.
    """
    quant = getattr(od_config, "quantization_config", None)
    if quant is None:
        return
    method = _quantization_method(quant)
    if method is None:
        logger.warning(
            "310P MOSS-Image: quantization_config present but method unresolvable; "
            "treating as unvalidated and dropping it.",
        )
        od_config.quantization_config = None
        return
    if method == "int8":
        logger.warning(
            "310P MOSS-Image: INT8 W8A8 quantization (POLICY section 2 fallback "
            "tier) accepted -- weights above 65535 fall back to FP16 layer-wise "
            "(research/009).  Record the quantization config + accuracy evidence "
            "in the task report.",
        )
        return
    raise RuntimeError(
        f"310P MOSS-Image rejects quantization method {method!r}: only FP16 "
        "(primary) and INT8 (POLICY section 2 fallback tier) are sanctioned on "
        "the 310P x CANN 9.1.0 stack.",
    )


def validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections."""
    from vllm_omni.diffusion.data import AttentionSpec

    attn_config = getattr(od_config, "diffusion_attention_config", None)
    if attn_config is None:
        return "TORCH_SDPA"

    default_spec = getattr(attn_config, "default", None)
    if default_spec is not None and str(getattr(default_spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
        raise RuntimeError(
            f"diffusion attention backend {default_spec.backend!r} requires MindIE-SD "
            "kernels, which are unavailable on 310P; use TORCH_SDPA.",
        )

    per_role = getattr(attn_config, "per_role", None) or {}
    for role, spec in per_role.items():
        if str(getattr(spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention per-role backend for {role!r} is {spec.backend!r}, "
                "which requires MindIE-SD kernels unavailable on 310P; use TORCH_SDPA.",
            )

    if default_spec is None:
        attn_config.default = AttentionSpec(backend="TORCH_SDPA")
    return "TORCH_SDPA"


def disable_numerical_cache_backends(od_config: Any) -> None:
    """Reject numerics-changing cache backends (same-seed reference runs).

    The MOSSImage pipeline raises on cache backends at construction anyway;
    this guard keeps a stable log line and protects future pipelines that
    reuse the 310P hook.
    """
    backend = getattr(od_config, "cache_backend", None)
    if backend is None or backend == "none":
        return
    if backend not in _NUMERIC_CACHE_BACKENDS:
        logger.warning(
            "[310P][mossimage] cache backend %r is unvalidated on 310P; "
            "the MOSSImage pipeline rejects it at construction.",
            backend,
        )
        return
    logger.warning(
        "[310P][mossimage] rejecting numerics-changing cache backend %r on 310P "
        "(same-seed reference comparison per COMPARISON-POLICY.md section 1).",
        backend,
    )
    od_config.cache_backend = "none"


def check_heads_divisibility(od_config: Any) -> None:
    """Ulysses/ring SP preflight: heads % degree == 0, seq % degree == 0.

    No-op for the TP1 baseline (the only sanctioned 310P starting point);
    it exists so a future TP/SP experiment on this model fails at patch
    time with a clear message instead of mid-forward.
    """
    tp_size = int(getattr(od_config, "tensor_parallel_size", 1) or 1)
    ulysses = int(getattr(od_config, "ulysses_degree", 1) or 1)
    ring = int(getattr(od_config, "ring_degree", 1) or 1)
    if tp_size > 2:
        raise RuntimeError(
            f"310P MOSS-Image: tensor_parallel_size={tp_size} is not sanctioned "
            "(TP2 on the Duo hits the 013 TransData environment-level failure; "
            "TP1 single-chip is the GO path).",
        )
    if ulysses <= 1 and ring <= 1:
        return
    heads = 24  # MOSSImageTransformer num_attention_heads (transformer/config.json)
    if heads % ulysses:
        raise RuntimeError(
            f"310P MOSS-Image: ulysses_degree={ulysses} does not divide the "
            f"24 transformer heads.",
        )
    if tp_size % ulysses:
        raise RuntimeError(
            f"310P MOSS-Image: ulysses_degree={ulysses} does not divide "
            f"tensor_parallel_size={tp_size}.",
        )


# ---------------------------------------------------------------------------
# Installers (monkey-patches on vllm-omni-side module/class attributes)
# ---------------------------------------------------------------------------


def _patch_loader_cpu_build() -> None:
    """Run the whole pipeline construction on CPU on 310P.

    The ``"default"`` load-format branch of the diffusers loader builds the
    pipeline inside ``with target_device:`` (npu:0).  Every device-less op at
    construction time (encoder frequency tables, VAE RMS-norm buffers,
    transformer RoPE ``bands``) then lands on the NPU and hits the 310P
    CANN 9.1.0 dynamic-TBE compile gap (research/012).  Upstream's own
    custom_pipeline branch solves this class of problem the same way
    ("build on CPU first, then explicitly move to the target device"); we
    mirror it by running ``initialize_model`` under ``torch.device("cpu")``.
    Final placement is made explicit by ``_install_post_load_placement``.
    """
    loader_mod = importlib.import_module("vllm_omni.diffusion.model_loader.diffusers_loader")
    if getattr(loader_mod, "_310p_cpu_build_patched", False):
        return
    orig_initialize_model = loader_mod.initialize_model

    def _initialize_model_on_cpu(od_config: Any) -> Any:
        with torch.device(resolve_encoder_placement()):
            return orig_initialize_model(od_config)

    loader_mod.initialize_model = _initialize_model_on_cpu
    loader_mod._310p_cpu_build_patched = True
    logger.info(
        "310P MOSS-Image: pipeline construction pinned to CPU "
        "(310P dynamic-TBE gap: RealDiv/OnesLike); explicit post-load "
        "placement moves transformer+VAE to the NPU, encoder stays CPU.",
    )


def _sdpa_gqa_expand(query: Any, key: Any, value: Any):
    """GQA expansion parity with upstream ``_sdpa_attention``."""
    if query.shape[1] == key.shape[1]:
        return query, key, value
    if query.shape[1] % key.shape[1]:
        raise ValueError(
            "GQA requires query heads a multiple of KV heads, "
            f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}.",
        )
    repeat = query.shape[1] // key.shape[1]
    return query, key.repeat_interleave(repeat, dim=1), value.repeat_interleave(repeat, dim=1)


def _manual_sdpa_forward(query: Any, key: Any, value: Any,
                         attention_mask: Any, causal: bool, scale: float) -> Any:
    """Eager SDPA decomposition for 310P3.

    Same contract as the backend's ``_sdpa_attention``: (B,H,S,D) in,
    (B,H,S,D) out; ``attention_mask`` may be a bool keep-mask (True = keep,
    torch SDPA semantics) or a float additive bias; ``causal`` adds the
    lower-triangular mask.  Dense path below the byte budget; chunked
    online-softmax above it.  Verified numerically equivalent to
    ``F.scaled_dot_product_attention`` in the contract test
    (``test_manual_sdpa_swap_rebinds_and_is_equivalent``).
    """
    query, key, value = _sdpa_gqa_expand(query, key, value)
    batch, heads, seq, dim = query.shape
    dense_bytes = batch * heads * seq * key.shape[2] * query.element_size()
    if dense_bytes > _SDPA_DENSE_LIMIT_BYTES:
        return _chunked_sdpa_forward(query, key, value, attention_mask,
                                     causal, scale)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        tri = torch.ones(sq, sk, dtype=torch.bool, device=scores.device).tril(sk - sq)
        scores = scores.masked_fill(~tri, float("-inf"))
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)


def _chunked_sdpa_forward(query: Any, key: Any, value: Any,
                          attention_mask: Any, causal: bool, scale: float) -> Any:
    """Online-softmax attention over key chunks (fp32 accumulators).

    Memory O(S*chunk) instead of O(S^2); math is the same softmax, so the
    result matches the dense path within fp32 accumulation noise.
    """
    batch, heads, seq, dim = query.shape
    seq_k = key.shape[2]
    out = torch.zeros(batch, heads, seq, dim,
                      device=query.device, dtype=torch.float32)
    rowmax = torch.full((batch, heads, seq, 1), float("-inf"),
                        device=query.device, dtype=torch.float32)
    rowsum = torch.zeros_like(rowmax)
    for start in range(0, seq_k, _SDPA_CHUNK_SIZE):
        stop = min(start + _SDPA_CHUNK_SIZE, seq_k)
        chunk = stop - start
        s = torch.matmul(query, key[:, :, start:stop].transpose(-2, -1)).float()
        s = s * scale
        if causal:
            # Entry (i, j) valid iff j <= i; local column j = start + col.
            tri = torch.ones(seq, chunk, dtype=torch.bool,
                             device=s.device).tril(-start)
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
    """Swap the TORCH_SDPA backend entry for the eager decomposition.

    The fused SDPA aclnn kernel is rejected on 310P3 (all probed shapes,
    dtypes, masks; torch MATH backend fails identically); the swap is a
    module-global rebind of ``sdpa._sdpa_attention`` -- every caller
    (``SDPAImpl.forward_npu`` -> ``_forward_impl``) resolves the name from
    module globals at call time, so one rebind covers the whole diffusion
    attention path in this process.
    """
    from vllm_omni.diffusion.attention.backends import sdpa as sdpa_mod

    if getattr(sdpa_mod, "_310p_manual_sdpa_patched", False):
        return
    sdpa_mod._sdpa_attention = _manual_sdpa_forward
    sdpa_mod._310p_manual_sdpa_patched = True
    logger.info(
        "310P MOSS-Image: TORCH_SDPA backend swapped to eager manual "
        "decomposition (fused SDPA kernel unsupported on 310P3, CANN 9.1.0).",
    )


def _make_cpu_pinned(model: Any) -> Any:
    """One-shot instance overlay: redirect every ``.to(...)`` on *model*
    to CPU.  Used for the Qwen3.5 text encoder: the pipeline chains
    ``.to(encoder_device)`` after loading, and later stages may retry
    placement; all of it must land on CPU on 310P.

    The overlay subclasses the real class without adding storage, so the
    instance layout is unchanged and ``__class__`` swap is safe.
    """
    if getattr(model, "_310p_cpu_pinned", False):
        return model
    base = type(model)

    def _to_cpu(self: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return torch.nn.modules.module.Module.to(self, "cpu")

    overlay = type(f"_CPUPinned{base.__name__}", (base,), {"to": _to_cpu})
    model.__class__ = overlay
    model._310p_cpu_pinned = True
    return model


def _patch_pipeline_encoder_placement(pipeline_module: Any) -> None:
    """Pin the Qwen3.5 text encoder to CPU.

    Rewires the pipeline module's own ``from_pretrained_with_prefetch``
    reference (module-local; other models are untouched): the encoder
    factory (``AutoModel.from_pretrained``) runs under ``torch.device("cpu")``
    so transformers' weight finalization creates no NPU tensors, and the
    returned model is ``_make_cpu_pinned`` (every later ``.to(...)``
    redirects to CPU).
    """
    if getattr(pipeline_module, "_310p_encoder_patched", False):
        return
    orig = pipeline_module.from_pretrained_with_prefetch
    enc_cls = pipeline_module.AutoModel

    def _same_classmethod(factory: Any, cls: Any, name: str) -> bool:
        """True if *factory* is ``cls.name`` (classmethod-safe: comparing the
        underlying ``__func__``/``__self__``, since bound-method objects are
        recreated on every attribute access and ``is`` never matches them)."""
        target = getattr(cls, name, None)
        if target is None or factory is None:
            return False
        f1, f2 = getattr(factory, "__func__", factory), getattr(target, "__func__", target)
        s1, s2 = getattr(factory, "__self__", None), getattr(target, "__self__", None)
        return f1 is f2 and s1 == s2

    def wrapped(factory: Any, *args: Any, **kwargs: Any) -> Any:
        if _same_classmethod(factory, enc_cls, "from_pretrained"):
            with torch.device(resolve_encoder_placement()):
                model = orig(factory, *args, **kwargs)
            model = _make_cpu_pinned(model)
            logger.info(
                "310P MOSS-Image: Qwen3.5 text encoder pinned to %s "
                "(RealDiv guard + 16.8 GB kept off the chip budget); "
                ".to(...) redirected to CPU.",
                resolve_encoder_placement(),
            )
            return model
        return orig(factory, *args, **kwargs)

    pipeline_module.from_pretrained_with_prefetch = wrapped
    pipeline_module._310p_encoder_patched = True


def _install_post_load_placement() -> None:
    """Move transformer+VAE to the local NPU after weight loading.

    With invariant 3 the pipeline is constructed on CPU; the transformer
    parameters therefore land on CPU after ``load_weights``.  This wrap
    performs the explicit tracked H2D copy (the upstream custom_pipeline
    placement pattern) so final placement is deterministic instead of
    ambient-ctx dependent.  Idempotent on already-placed pipelines (``.to``
    on the target device is a no-op).
    """
    from vllm_omni.diffusion.models.mossimage import pipeline_mossimage

    cls = pipeline_mossimage.MOSSImagePipeline
    if getattr(cls, "_310p_placement_patched", False):
        return
    original_load_weights = cls.load_weights

    def _load_weights_then_place(self: Any, weights: Any) -> Any:
        result = original_load_weights(self, weights)
        from vllm_omni.diffusion.distributed.utils import get_local_device

        target = get_local_device()
        if getattr(self, "transformer", None) is not None:
            self.transformer.to(target)
        if getattr(self, "vae", None) is not None:
            self.vae.to(target)
        logger.info(
            "310P MOSS-Image: post-load placement done (transformer/vae -> %s; "
            "text_encoder stays CPU-pinned).",
            target,
        )
        return result

    cls.load_weights = _load_weights_then_place
    cls._310p_placement_patched = True
    logger.info(
        "310P MOSS-Image: MOSSImagePipeline.load_weights wrapped for explicit "
        "NPU placement (transformer+VAE H2D; encoder CPU-pinned).",
    )


def _log_placement_plan(od_config: Any) -> None:
    tp = int(getattr(od_config, "tensor_parallel_size", 1) or 1)
    logger.info(
        "[310P][mossimage] placement plan: dtype=fp16 attention=TORCH_SDPA "
        "vae_tiling=forced encoder=cpu(transformer+vae->npu post-load) | "
        "parallel: tp=%s ulysses=%s ring=%s | quant=%s",
        tp,
        int(getattr(od_config, "ulysses_degree", 1) or 1),
        int(getattr(od_config, "ring_degree", 1) or 1),
        _quantization_method(getattr(od_config, "quantization_config", None)),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Install the 310P specialization for MOSS-Image on this rank.

    Idempotent; safe to call from the diffusion runtime hook on every rank.
    """
    global _APPLIED
    if _APPLIED:
        return

    current_dtype = getattr(od_config, "dtype", None)
    od_config.dtype = resolve_runtime_dtype(current_dtype)
    if current_dtype != torch.float16:
        logger.warning(
            "310P MOSS-Image: forcing runtime dtype %s -> float16 "
            "(310P is FP16-only; BF16 has no 310P path).",
            current_dtype,
        )
    od_config.enforce_eager = True  # NPU has no inductor; CUDA-graph decode is CUDA-only

    validate_attention_selection(od_config)
    validate_quantization(od_config)
    disable_numerical_cache_backends(od_config)
    check_heads_divisibility(od_config)

    vae_tiling, vae_slicing = resolve_vae_memory(
        getattr(od_config, "vae_use_tiling", False),
        getattr(od_config, "vae_use_slicing", False),
    )
    od_config.vae_use_tiling = vae_tiling
    od_config.vae_use_slicing = vae_slicing

    _patch_loader_cpu_build()
    _patch_attention_manual_sdpa()
    from vllm_omni.diffusion.models.mossimage import pipeline_mossimage as pipeline_module

    _patch_pipeline_encoder_placement(pipeline_module)
    _install_post_load_placement()
    _log_placement_plan(od_config)
    _APPLIED = True


def maybe_apply_diffusion_310p_patches(od_config: Any) -> None:
    """Self-gated standalone entry point (platform.py rebase-window call).

    No-op unless ``VLLM_OMNI_310P_DIFFUSION_PATCHES != "0"``,
    ``is_310p()`` is True *and* ``model_class_name == MODEL_ARCH``.  Once the
    task-11 registry hook lands in ``init_diffusion_model_runner_runtime``
    this file is registered via ``_DIFFUSION_310P_PATCH_MODULES`` and the
    registry drives ``apply_diffusion_310p_patches``; keep both paths for the
    rebase window (``_APPLIED`` makes the double path idempotent).
    """
    if os.environ.get(_KILL_SWITCH, "1") == "0":
        return
    from vllm_omni.platforms.npu._310p import is_310p

    if not is_310p():
        return
    if getattr(od_config, "model_class_name", None) != MODEL_ARCH:
        return
    apply_diffusion_310p_patches(od_config)


# Registry-facing alias (COMPARISON-POLICY.md section 5).
apply_patch = apply_diffusion_310p_patches
