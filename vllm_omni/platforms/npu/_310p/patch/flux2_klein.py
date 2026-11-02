# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch FLUX.2-klein for the 310P NPU diffusion path.

Registered in ``vllm_omni.platforms.npu._310p.patch.__init__`` (the 310P
diffusion patch registry, COMPARISON-POLICY.md section 5); ``platform.py``
stays untouched by this branch. Dispatched from
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` on 310P, before the
pipeline is constructed.

310P diffusion decision: pure ``torch_npu`` + FP16, no MindIE-SD kernels.
Invariants enforced here:

* ``od_config.dtype`` is forced to ``float16``.  310P is FP16-only (no BF16
  kernel path); a non-FP16 incoming dtype is corrected in place at the
  platform layer (SDXL-line precedent: patch-forced dtype).
* The diffusion attention backend resolves to ``TORCH_SDPA`` whenever the
  user did not select one: MindIE-SD's ``FLASH_ATTN`` / ``RAINFUSION_ATTN``
  kernels do not exist on 310P, and the platform's ``find_spec("mindiesd")``
  fallback cannot be trusted on this card.  An explicit request for a
  MindIE-SD-only backend fails at patch time with a clear error instead of
  deep inside a kernel call.  The native ``torch_npu`` fusion-attention
  candidate is retired: research/003 shows ``npu_fusion_attention`` is
  officially limited to 950DT + A2 training series, unavailable on 310P.
* Numerics-changing cache backends are disabled while the 310P path is
  unvalidated, so the same-seed reference comparison (COMPARISON-POLICY.md
  section 1) is not broken by step-skipping caches.
* Precision tiers (POLICY section 2, revised 2026-11-02): FP16 (no
  ``quantization_config``) is the primary tier.  When weights exceed the
  chip budget, the sanctioned second tier is INT8 W8A8 (method ``int8``,
  DiffusionInt8Config online dynamic quantization through
  ``torch_npu.npu_quant_matmul``, per-layer fallback of layers wider than
  65535 to unquantized - research/009): accepted with a loud warning;
  precision evidence against the FP16 baseline is mandatory in the
  real-device report.  FP8/MXFP* (no 310P hardware benefit), ConvRot-INT4
  (unvalidated) and bitsandbytes (CUDA-only) are rejected.
* LoRA is accepted with a loud warning (out of scope for the parity
  baseline; weights differ from the reference run by design).
* Chain-1 OOM downgrade (lead-fixed order): when the chip's 21589 MB budget
  is exceeded, ``FLUX2KLEIN_310P_ENCODER_CPU=1`` pins the Qwen3 text
  encoder on host memory (same playbook as model-qwenimg21 M8/R12); the
  wrapper moves it back after pipeline construction.  Chain-2 (SP=2) uses the
  intra-card HCCS chip pair (supported link; feasibility judged against the
  task-10 HCCL topology table's intra-card pair bandwidth, falling back to
  CPU-offload when insufficient).
* VAE decode is forced onto the diffusers tiling/slicing path before the
  pipeline is loaded: the untiled 1024^2 decode peak is 8 GB+ of activations,
  which does not fit the 21589 MB per-chip budget with weights resident
  (same finding as task-11's SDXL risk list).

Component placement (logged at startup, matches the pipeline's own
``get_local_device()`` behavior):

* ``transformer``: sharded across the configured TP/SP group (Ulysses strict
  mode requires ``seq_len % sp == 0`` and ``heads % ulysses_degree == 0``).
* ``text_encoder`` (Qwen3ForCausalLM) and ``vae`` (AutoencoderKLFlux2):
  replicated on every rank's local chip, forward-per-rank.  The 4B t2i
  baseline fits on a single 310P chip (chip0 budget 21589 MB per the
  mount recipe, with a thin activation margin); SP=2 scale-out uses the
  two chips of ONE Duo card over the intra-card HCCS link (supported,
  standard path) - feasibility is judged against the task-10 HCCL topology
  table's intra-card pair bandwidth, cross-card (PCIe) SP stays a separate
  task-5/task-7 validation scope.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: ``od_config.model_class_name`` value for the FLUX.2-klein pipeline.
FLUX2_KLEIN_ARCH = "Flux2KleinPipeline"

#: Backends whose NPU kernels ship only with MindIE-SD (absent on 310P).
_MINDIE_SD_ONLY_BACKENDS = frozenset(
    {"FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "RAINFUSION_ATTN"}
)

#: Cache backends that change denoising numerics (step skipping / reuse).
_NUMERICAL_CACHE_BACKENDS = frozenset(
    {"cache_dit", "tea_cache", "mag_cache", "sea_cache", "step_cache", "stepcache", "step_cache_dit"}
)

_APPLIED = False
_ATTENTION_GUARD_INSTALLED = False
_VAE_TILING_WRAPPED = False
_ENCODER_CPU_GUARDED = False
_DIT_FP16_GUARDED = False
_LOADER_FP16_GUARDED = False

#: Chain-1 OOM downgrade: pin the Qwen3 text encoder on host memory.
_ENCODER_CPU_ENV = "FLUX2KLEIN_310P_ENCODER_CPU"


def _evidence(tag: str, payload: str) -> None:
    # File-level probe: spawn workers swallow the logger in flight; the
    # evidence file survives and makes patch application observable.
    try:
        with open("/tmp/310p_flux2_evidence.log", "a") as fh:
            fh.write(f"{tag} {payload} (pid={os.getpid()})\n")
    except OSError:
        pass


def _force_fp16(od_config: Any) -> None:
    """Force FP16 (310P invariant; SDXL-line precedent: patch-forced dtype).

    310P is FP16-only hardware — there is no bf16 decision this line could
    make, so a non-FP16 incoming dtype is corrected in place (and logged)
    rather than rejected: the platform layer is the enforcement point for
    the hardware invariant, and the config plumbing above it evolves across
    vllm-omni lines."""
    dtype = getattr(od_config, "dtype", None)
    if dtype == torch.float16:
        return
    od_config.dtype = torch.float16
    logger.warning(
        "[310P][flux2-klein] forced od_config.dtype=%r -> float16 "
        "(310P is FP16-only hardware; no bf16 kernel path exists)",
        dtype,
    )


def _validate_attention_selection(od_config: Any) -> str:
    """Force TORCH_SDPA on 310P and reject MindIE-SD-only selections."""
    from vllm_omni.diffusion.data import AttentionSpec

    attn_config = getattr(od_config, "diffusion_attention_config", None)
    if attn_config is None:
        return "TORCH_SDPA"

    default_spec = getattr(attn_config, "default", None)
    if default_spec is not None and str(getattr(default_spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
        raise RuntimeError(
            f"diffusion attention backend {default_spec.backend!r} requires MindIE-SD kernels, "
            "which are unavailable on 310P; use TORCH_SDPA."
        )

    per_role = getattr(attn_config, "per_role", None) or {}
    for role, spec in per_role.items():
        if str(getattr(spec, "backend", "")).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention per-role backend for {role!r} is {spec.backend!r}, "
                "which requires MindIE-SD kernels unavailable on 310P; use TORCH_SDPA."
            )

    if default_spec is None:
        attn_config.default = AttentionSpec(backend="TORCH_SDPA")
    return "TORCH_SDPA"


def _disable_numerical_cache_backends(od_config: Any) -> None:
    backend = getattr(od_config, "cache_backend", None)
    if backend is None or backend == "none":
        return
    if backend not in _NUMERICAL_CACHE_BACKENDS:
        logger.warning(
            "[310P][flux2-klein] cache backend %r is unvalidated on 310P; leaving as configured.",
            backend,
        )
        return
    logger.warning(
        "[310P][flux2-klein] disabling numerics-changing cache backend %r on 310P "
        "(same-seed reference comparison per COMPARISON-POLICY.md section 1); "
        "re-enable after 310P validation.",
        backend,
    )
    od_config.cache_backend = "none"


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


def _validate_quantization(od_config: Any) -> None:
    """Precision-tier gate: FP16 primary, INT8 W8A8 sanctioned fallback only.

    POLICY section 2 (revised 2026-11-02): FP16 is the first choice; when
    weights do not fit the chip budget, INT8 is the fallback tier (no BF16,
    no over-budget force-fitting, quantization config must be recorded in
    the report).  research/009: on 310P only INT8 W8A8 has a hardware
    benefit (npu_quant_matmul, 280 TOPS INT8 per card); FP8/MXFP* fall back
    to FP16 matmul with no gain, ConvRot-INT4 NPU applicability is
    unvalidated, bitsandbytes is CUDA-only.
    """
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    method = _quantization_method(quant_config)
    if method == "int8":
        logger.warning(
            "[310P][flux2-klein] quantization_config method=%r accepted as the "
            "POLICY section-2 second tier (research/009): online dynamic INT8 "
            "via npu_quant_matmul, layers with output dim > 65535 fall back to "
            "unquantized (mixed-precision semantics). Precision evidence vs "
            "the FP16 baseline is MANDATORY in the real-device report; no "
            "upstream 310P diffusion INT8 precedent exists (first-run).",
            method,
        )
        return
    raise RuntimeError(
        f"310P FLUX.2-klein rejects quantization method {method!r} "
        f"(quantization_config={quant_config!r}): FP16 is the primary tier "
        "(quantization_config unset); only the INT8 W8A8 fallback "
        "(method='int8') is sanctioned on 310P (POLICY section 2, "
        "research/009). FP8/MXFP* give no hardware benefit on 310P "
        "(served with FP16 matmul); ConvRot-INT4 is unvalidated; "
        "bitsandbytes is CUDA-only."
    )

def _check_heads_divisibility(od_config: Any) -> None:
    """Strict-Ulysses preflight: head count must divide the ulysses degree."""
    parallel = getattr(od_config, "parallel_config", None)
    ulysses = getattr(parallel, "ulysses_degree", 1) if parallel is not None else 1
    if not ulysses or ulysses <= 1:
        return
    tf_config = getattr(od_config, "tf_model_config", None)
    if tf_config is None:
        return
    heads = tf_config.get("num_attention_heads", None)
    if heads is None:
        return
    if heads % ulysses != 0:
        raise RuntimeError(
            f"strict Ulysses on 310P needs num_attention_heads % ulysses_degree == 0, "
            f"got heads={heads}, ulysses_degree={ulysses}. "
            "Single-chip TP=1 baseline is unaffected."
        )


def _maybe_enable_vae_tiling(vae: Any) -> bool:
    """Put the VAE onto diffusers' tiling/slicing decode path.

    Returns True when both switches engaged.  Guarded with ``getattr`` so a
    diffusers revision without the switches still boots (the warning keeps
    the untiled-decode memory risk on the record).
    """
    enabled = True
    for attr in ("enable_tiling", "enable_slicing"):
        fn = getattr(vae, attr, None)
        if not callable(fn):
            logger.warning(
                "[310P][flux2-klein] VAE %s has no %r; untiled decode activation peak "
                "(8 GB+ at 1024^2) is a memory risk on 24 GB chips.",
                type(vae).__name__,
                attr,
            )
            enabled = False
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - boot must not die on a tiling flag
            logger.warning("[310P][flux2-klein] VAE %s() failed: %s", attr, exc)
            enabled = False
    return enabled


class _CpuPinnedEncoder:
    """Chain-1 proxy: the Qwen3 text encoder load stays on host memory and the
    pipeline's ``.to(execution_device)`` cast is absorbed (no 8 GB NPU round
    trip).  Attributes/calls forward to the wrapped module so the pipeline
    code is untouched; per-request H2D of the small prompt tensors happens in
    ``_get_qwen3_prompt_embeds`` (retargeted while env-gated)."""

    def __init__(self, module: Any) -> None:
        self._m = module

    def to(self, *args: Any, **kwargs: Any) -> "_CpuPinnedEncoder":
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._m, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._m(*args, **kwargs)


def _encoder_cpu_retarget(encoder_device: Any, device: Any) -> Any:
    """Chain-1 device decision for the Qwen3 prompt-embeds forward: while the
    encoder lives on host (FLUX2KLEIN_310P_ENCODER_CPU=1), run the encoder
    on its own device instead of the NPU; the tiny embeds tensor is moved to
    the target afterwards (see _get_qwen3_prompt_embeds wrap)."""
    if os.environ.get(_ENCODER_CPU_ENV, "0") != "1":
        return device
    enc_type = getattr(encoder_device, "type", None)
    dev_type = getattr(device, "type", None)
    if enc_type == "cpu" and dev_type is not None and dev_type != "cpu":
        return encoder_device
    return device


def _install_encoder_cpu_guard() -> None:
    """Chain-1 runtime guard (FLUX2KLEIN_310P_ENCODER_CPU=1):

    1. the pipeline module's ``from_pretrained_with_prefetch`` returns a
       ``_CpuPinnedEncoder`` proxy for the ``text_encoder`` subfolder, so the
       Qwen3 8 GB load + ``.to(execution_device)`` never touches the chip;
    2. ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` runs the encoder on its
       own device and moves the resulting embeds to the requested device.
    """
    global _ENCODER_CPU_GUARDED
    if _ENCODER_CPU_GUARDED:
        return
    if os.environ.get(_ENCODER_CPU_ENV, "0") != "1":
        return  # chain-1 not armed: no pipeline-module touch, no import side effects
    _ENCODER_CPU_GUARDED = True

    import sys as _sys

    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
        Flux2KleinPipeline,
    )

    _pl = _sys.modules.get(
        "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein"
    )
    if _pl is None:  # pragma: no cover - defensive; the import above populated it
        raise RuntimeError("pipeline module missing from sys.modules after import")

    _orig_from_pf = getattr(_pl, "from_pretrained_with_prefetch", None)
    if _orig_from_pf is not None:

        def _guarded_from_pf(cls_fn: Any, *args: Any, **kwargs: Any) -> Any:
            module = _orig_from_pf(cls_fn, *args, **kwargs)
            if os.environ.get(_ENCODER_CPU_ENV, "0") == "1" and kwargs.get("subfolder") == "text_encoder":
                return _CpuPinnedEncoder(module)
            return module

        _pl.from_pretrained_with_prefetch = _guarded_from_pf

    # NOTE: the pipeline defines _get_qwen3_prompt_embeds as a @staticmethod
    # (no self); the wrap must keep the plain-function arity or the call
    # sites in the pipeline break.
    _orig_embeds = getattr(Flux2KleinPipeline, "_get_qwen3_prompt_embeds", None)

    def _get_qwen3_prompt_embeds_310p(
        text_encoder: Any,
        tokenizer: Any,
        prompt: Any,
        hidden_states_layers: Any,
        dtype: Any = None,
        device: Any = None,
        max_sequence_length: int = 512,
    ) -> Any:
        target = device
        device = _encoder_cpu_retarget(getattr(text_encoder, "device", None), device)
        out = _orig_embeds(
            text_encoder,
            tokenizer,
            prompt,
            hidden_states_layers,
            dtype,
            device,
            max_sequence_length,
        )
        if target is not None and out is not None and getattr(out, "device", None) != target:
            out = out.to(target)
        return out

    if _orig_embeds is not None:
        # The original is a @staticmethod; rebind as a staticmethod so the
        # pipeline's self._get_qwen3_prompt_embeds(...) call site keeps the
        # plain-function arity (no implicit self bind).
        Flux2KleinPipeline._get_qwen3_prompt_embeds = staticmethod(
            _get_qwen3_prompt_embeds_310p
        )


def _install_vae_tiling_wrap() -> None:
    """Wrap ``Flux2KleinPipeline.__init__`` so the VAE switches on before
    the first request (the runner loads the pipeline right after the patch
    runs)."""
    global _VAE_TILING_WRAPPED
    if _VAE_TILING_WRAPPED:
        return

    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline

    original_init = Flux2KleinPipeline.__init__

    def _init_with_vae_tiling(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._310p_vae_tiling = _maybe_enable_vae_tiling(self.vae)
        # 310P hardware invariant, enforced at the platform layer: the Qwen3
        # encoder loads in its on-disk BF16 dtype and the aclnn embedding
        # kernel rejects it ("Tensor weight not implemented for DT_BFLOAT16"
        # -> EZ1001 at the first encode).  Cast to FP16 here; chain-1
        # additionally moves the (now FP16) encoder to host memory.
        enc_dtype = next(iter(getattr(self.text_encoder, "parameters", lambda: [])()), None)
        if os.environ.get(_ENCODER_CPU_ENV, "0") == "1":
            enc = self.text_encoder
            if isinstance(enc, _CpuPinnedEncoder):
                enc = enc._m  # the loader proxy was effective (load stayed on host)
            if enc_dtype is not None and enc_dtype is not torch.float16:
                logger.warning(
                    "[310P][flux2-klein] text encoder loaded as %s; casting to "
                    "float16 (310P is FP16-only)",
                    enc_dtype,
                )
            self.text_encoder = enc.to("cpu", dtype=torch.float16)
            self._310p_encoder_cpu = True
            logger.warning(
                "[310P][flux2-klein] text encoder pinned to CPU (fp16) via %s "
                "(chain-1 OOM downgrade, chip budget 21589 MB); per-request "
                "H2D of the small prompt embeds only.",
                _ENCODER_CPU_ENV,
            )
        elif enc_dtype is not None and enc_dtype is not torch.float16:
            logger.warning(
                "[310P][flux2-klein] text encoder loaded as %s; casting to "
                "float16 in place (310P is FP16-only; aclnn rejects BF16 "
                "weights)",
                enc_dtype,
            )
            self.text_encoder = self.text_encoder.to(dtype=torch.float16)
        try:
            enc_p = next(iter(self.text_encoder.parameters()))
            vae_p = next(iter(self.vae.parameters()))
            _evidence(
                "post-init",
                f"enc_dev={enc_p.device} enc_dtype={enc_p.dtype} "
                f"vae_dev={vae_p.device} tiling={self._310p_vae_tiling} "
                f"env={os.environ.get(_ENCODER_CPU_ENV, '0')}",
            )
        except Exception as exc:  # pragma: no cover - probe must never break init
            _evidence("post-init", f"probe-error {exc!r}")

    Flux2KleinPipeline.__init__ = _init_with_vae_tiling
    _VAE_TILING_WRAPPED = True


def _log_placement_plan(od_config: Any) -> None:
    parallel = getattr(od_config, "parallel_config", None)
    tp = getattr(parallel, "tensor_parallel_size", 1) if parallel is not None else 1
    sp = getattr(parallel, "sequence_parallel_size", None) if parallel is not None else None
    ulysses = getattr(parallel, "ulysses_degree", 1) if parallel is not None else 1
    ring = getattr(parallel, "ring_degree", 1) if parallel is not None else 1
    allgather = getattr(parallel, "allgather_degree", 1) if parallel is not None else 1
    logger.info(
        "[310P][flux2-klein] dtype=fp16 attention=TORCH_SDPA vae_tiling=forced | parallel: tp=%s "
        "sp=%s (ulysses=%s ring=%s allgather=%s) | placement: transformer sharded across the "
        "TP/SP group; text_encoder (Qwen3) and vae (AutoencoderKLFlux2) replicated per rank "
        "on each rank's local chip; intra-card SP=2 rides HCCS (standard); "
        "cross-card SP awaits task-5/task-7 PCIe validation.",
        tp,
        sp,
        ulysses,
        ring,
        allgather,
    )


def _install_attention_guard() -> None:
    """Wrap the NPU platform backend selector so 310P can never resolve to
    MindIE-SD-only kernels, even for per-role or per-request selections that
    bypass the od_config-level default."""
    global _ATTENTION_GUARD_INSTALLED
    if _ATTENTION_GUARD_INSTALLED:
        return

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    # Unwrap the classmethod descriptor so ``cls`` is passed exactly once.
    original = NPUOmniPlatform.__dict__["get_diffusion_attn_backend_cls"]
    original = getattr(original, "__func__", original)

    @classmethod
    def _guarded(cls, selected_backend, head_size, allow_trtllm_default: bool = True):  # noqa: ANN001
        if selected_backend is None:
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
        if str(selected_backend).upper() in _MINDIE_SD_ONLY_BACKENDS:
            raise RuntimeError(
                f"diffusion attention backend {selected_backend!r} requires MindIE-SD "
                "kernels, which are unavailable on 310P; use TORCH_SDPA."
            )
        return original(cls, selected_backend, head_size, allow_trtllm_default=allow_trtllm_default)

    NPUOmniPlatform.get_diffusion_attn_backend_cls = _guarded
    _ATTENTION_GUARD_INSTALLED = True


def _install_dit_fp16_guard() -> None:
    """One-shot BF16->FP16 cast for the DiT, trapped at the first forward.

    The FLUX.2-klein 4B transformer ships BF16 on disk (verified:
    transformer/context_embedder.weight dtype=BF16) and the engine's weight
    loader does not cast it to the od_config dtype; 310P aclnn rejects BF16
    tensors outright (EZ1001 at the first matmul/cat).  Weights land after
    pipeline __init__, so the cast is deferred to the first forward: a
    one-shot in-place FP16 cast (weights already resident), then the
    original forward runs.
    """
    global _DIT_FP16_GUARDED
    if _DIT_FP16_GUARDED:
        return
    _DIT_FP16_GUARDED = True

    import sys as _sys

    from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
        Flux2Transformer2DModel,
    )

    _orig_load_weights = getattr(Flux2Transformer2DModel, "load_weights", None)
    if _orig_load_weights is not None:

        def _load_weights_310p_fp16(self: Any, weights: Any, *a: Any, **k: Any) -> Any:
            loaded = _orig_load_weights(self, weights, *a, **k)
            p = next(iter(self.parameters()), None)
            if p is not None and p.dtype is not torch.float16:
                logger.warning(
                    "[310P][flux2-klein] DiT weights loaded as %s; casting to "
                    "float16 in place (310P is FP16-only hardware)",
                    p.dtype,
                )
                self.to(dtype=torch.float16)
                _evidence("dit-load-cast", f"loaded {p.dtype} -> fp16")
            return loaded

        Flux2Transformer2DModel.load_weights = _load_weights_310p_fp16
        _sys.modules[
            "vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer"
        ].Flux2Transformer2DModel.load_weights = _load_weights_310p_fp16

    _orig_forward = Flux2Transformer2DModel.forward

    def _forward_310p_fp16(self: Any, *args: Any, **kwargs: Any) -> Any:
        p = next(iter(self.parameters()), None)
        if p is not None and p.dtype is not torch.float16:
            logger.warning(
                "[310P][flux2-klein] DiT weights loaded as %s; casting to "
                "float16 in place (310P is FP16-only hardware)",
                p.dtype,
            )
            self.to(dtype=torch.float16)
            _evidence("dit-forward-cast", f"loaded {p.dtype} -> fp16")
        return _orig_forward(self, *args, **kwargs)

    _sys.modules[
        "vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer"
    ].Flux2Transformer2DModel.forward = _forward_310p_fp16
    Flux2Transformer2DModel.forward = _forward_310p_fp16


def _install_loader_fp16_guard() -> None:
    """Catch-all BF16->FP16 cast at the one weight-loading choke point.

    Every diffusion component (text encoder, DiT, VAE) lands through
    DiffusersPipelineLoader.load_weights; the loader does not cast the
    on-disk dtype to the od_config dtype, and the loader telemetry then
    PRINTS a bf16 Parameter (repr -> torch.cat on the NPU -> EZ1001),
    killing the run before the first forward.  Wrap the choke point and
    cast each component to FP16 the moment its weights land.
    """
    global _LOADER_FP16_GUARDED
    if _LOADER_FP16_GUARDED:
        return
    _LOADER_FP16_GUARDED = True

    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )

    _orig_load_weights = DiffusersPipelineLoader.load_weights

    def _load_weights_310p_fp16(self: Any, model: Any, **kwargs: Any) -> None:
        _orig_load_weights(self, model, **kwargs)
        p = next(iter(model.parameters()), None)
        if p is not None and p.dtype is not torch.float16:
            logger.warning(
                "[310P][flux2-klein] %s weights loaded as %s; casting to "
                "float16 in place (310P is FP16-only hardware)",
                type(model).__name__,
                p.dtype,
            )
            model.to(dtype=torch.float16)
            _evidence("loader-cast", f"{type(model).__name__} {p.dtype} -> fp16")
        else:
            _evidence("loader-ok", f"{type(model).__name__} already-fp16")

    DiffusersPipelineLoader.load_weights = _load_weights_310p_fp16


def apply_diffusion_310p_patches(od_config: Any) -> None:
    """Apply all FLUX.2-klein 310P invariants. Idempotent per process.

    Contract name required by the 310P diffusion patch registry
    (``_DIFFUSION_310P_PATCH_MODULES`` dispatch in
    ``vllm_omni.platforms.npu._310p.patch``).
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    _evidence("apply", f"dtype={getattr(od_config, 'dtype', None)}")

    _force_fp16(od_config)
    _validate_attention_selection(od_config)
    _disable_numerical_cache_backends(od_config)
    _validate_quantization(od_config)
    if getattr(od_config, "lora_path", None) is not None:
        logger.warning(
            "[310P][flux2-klein] lora_path=%r is out of scope for the 310P reference "
            "baseline; outputs will not match the reference run weights.",
            od_config.lora_path,
        )
    _check_heads_divisibility(od_config)
    _install_vae_tiling_wrap()
    _install_encoder_cpu_guard()
    _install_dit_fp16_guard()
    _install_loader_fp16_guard()
    _install_attention_guard()
    _log_placement_plan(od_config)
