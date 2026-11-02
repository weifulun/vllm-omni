# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""310P patch for the Qwen-Image-2.1 diffusion pipeline (task-16).

SKELETON STATUS (local prep, 2026-11-02)
----------------------------------------
This file is the 310P adaptation hook for ``QwenImage21Pipeline``.

Integration contract (COMPARISON-POLICY.md §5, lead-approved 2026-11-02)
-------------------------------------------------------------------------
task-11 (SDXL) lands the registry-style diffusion 310P hook in
``NPUOmniPlatform.init_diffusion_model_runner_runtime`` first
(model_class_name -> [patch function]); this branch rebases on it and
registers ONLY through this file -- it must NOT edit ``platform.py``.  The
registry consumes the names below (``maybe_apply_diffusion_310p_patches``
stays as the self-gated standalone entry point until the registry lands):

    MODEL_ARCH = "QwenImage21Pipeline"
    apply_patch = apply_qwen_image_2_1_310p_patches

Why this hook at all: ``apply_model_patches(model_config)`` only runs inside
the AR ``OmniNPUModelRunner.load_model``; diffusion pipelines are built by
``DiffusionModelRunner.load_model``, which calls
``init_diffusion_model_runner_runtime`` first (same convention as task-11).

Code-level findings that shape this patch (v0.32.0, branch feat/310p-qwen-image-2.1)
--------------------------------------------------------------------------------------
Under an FP16 runtime (the 310P policy) *every* BF16-gated NPU fast path in
``qwen_image_21`` is disabled by its own dtype guard, so the diff versus the
910B behavior is mostly "fast paths stay off", plus three real hazards:

1. RoPE: ``qwen_image_21_transformer._apply_qwen_image21_rotary_emb_npu``
   calls ``torch_npu.npu_rotary_mul(x.float(), cos, sin,
   rotary_mode="interleave")`` with no fallback.  ``npu_rotary_mul`` is proven
   on 310P only in *half* mode (``_310p/patch/qwen3_tts.py`` and
   ``platforms/npu/layers/rotary_embedding.py`` never pass "interleave").
   If the 310P CANN build rejects interleave mode the whole DiT crashes at
   the first block -> guarded dispatcher with native real-valued fallback.
   ADDITIONALLY (v7f, machine-verified): the freqs table is complex
   (``torch.polar``) and 310P AICore has **no complex-dtype compute** -- the
   stock ``torch.real/imag`` on the NPU-resident table dies with
   ``Current settings do not support Complex dtype`` / ERR01007.  The
   dispatcher therefore extracts real cos/sin tables on CPU and never executes
   a complex op on the device, for every policy.
2. NZ weight cast: ``load_weights`` casts every BF16 2-D linear weight via
   ``npu_format_cast(w, 29)`` (29 == ACL_FORMAT_FRACTAL_NZ) when internal
   formats are enabled.  BF16 gates it off under FP16, but if the model is
   ever built BF16 on 310P the cast target is 910-oriented and unvalidated ->
   defensively disabled for 310P.
3. Attention: default backend on NPU without MindIE-SD is TORCH_SDPA
   (``platform.py:get_diffusion_attn_backend_cls``), which drives
   ``SDPAImpl.forward_npu`` (dense-mask ``full_qk`` path because
   ``supports_piecewise_spans`` is False) -- the block-causal mask of
   Qwen-Image-2.1 is therefore a dense 4-D boolean mask on 310P.  The
   ``QwenImage21SDPAImpl`` fusion fast path requires BF16 and stays off.
   Everything else in the transformer/VAE is portable eager torch.

 3b. Attention execution (machine-verified on 126 C4 + 129 cross-node,
    2026-11-02): the torch_npu fused SDPA kernel is UNSUPPORTED on 310P3
    (CANN 9.1.0): ``F.scaled_dot_product_attention`` fails ``ERR01007 OPS
    feature not supported`` + "can not cast format when output is input" for
    fp16/bf16/fp32, masked and unmasked, across all probed shapes. The torch
    MATH backend fails identically through the torch_npu dispatch. An eager
    decomposition (scale -> matmul -> mask -> softmax -> matmul) passes all
    shapes/dtypes on both nodes, so ``_patch_attention_manual_sdpa`` swaps
    the TORCH_SDPA backend's module-level entry (``sdpa._sdpa_attention``)
    for that decomposition -- same vllm-omni-side monkey-patch style as the
    rest of this file; the image stays read-only.

4. Text encoder placement (machine-verified on 126, 2026-11-02):
   the generic pipeline loads the 8.8B Qwen3-VL encoder via
   ``from_pretrained_with_prefetch(...).to(self.device)`` INSIDE the
   loader's ``with target_device:`` (npu:0) context, which breaks on
   310P twice over:

   a) transformers v5 ``from_pretrained`` finalization runs
      ``initialize_weights`` -> Qwen3-VL ``compute_axial_rope_parameters``:
      ``inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2) /
      spatial_dim))``.  Deviceless ``torch.arange`` honors the ambient NPU
      device context, so the RealDiv lands on the NPU and CANN 9.1.0's
      310P TBE cannot pre-compile the *dynamic* real_div kernel
      (``Failed to compile Op RealDiv2 ... ops_legacy/dynamic/real_div.py``
      -> graph build fail -> ``Orchestrator initialization failed``).
      Upstream 910B3 CI (identical version stack, vllm-omni PR #9288)
      does not trip it -- 310P soc-specific TBE gap (research/012, first
      public report).
   b) Memory: the encoder occupies ~18 GB of the 21.589 GB chip, leaving
      no headroom for the 14.23 GB DiT on the same chip.

   The 310P policy therefore pins the encoder to CPU: the factory call is
   re-routed under ``torch.device("cpu")`` and every subsequent ``.to``
   on the instance is redirected to CPU.  The pipeline already casts
   encoder output to the transformer dtype (pipeline L713), so the
   BF16-CPU encoder feeds the FP16 DiT cleanly.

CUDA-graph decode (``decode_graph.py``) is CUDA-only by construction
(``device.type != "cuda"`` -> eager) and needs no 310P handling.
"""

from __future__ import annotations

import os
import logging
import math
from typing import Any

import torch

from vllm_omni.platforms.npu._310p import is_310p

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001 - logging library unavailable
    logger = logging.getLogger(__name__)

# Registry contract (COMPARISON-POLICY.md §5): the task-11 platform.py hook
# maps model_class_name -> patch function; this file registers the entry
# below and nothing else.
MODEL_ARCH = "QwenImage21Pipeline"

# Env knobs (all checked once at patch time; machine-phase A/B experiments use
# them instead of editing code):
#   QWEN21_310P_ROPE=auto|npu|native   (default auto: probe once, fall back)
#   QWEN21_310P_GENERATOR_DEVICE=cpu|npu (default npu: keep device RNG)
#   QWEN21_310P_ATTENTION=sdpa|flash_attn (default sdpa; flash_attn is an
#       experiment knob only, npu_fusion_attention on 310P is unvalidated)

_APPLIED = False


# ---------------------------------------------------------------------------
# Pure, CPU-testable decision functions
# ---------------------------------------------------------------------------


def resolve_runtime_dtype(current: torch.dtype) -> torch.dtype:
    """310P diffusion runs FP16 (project policy; MindIE-SD free)."""
    del current
    return torch.float16


def resolve_rope_policy(environ: dict[str, str] | None = None) -> str:
    """Pick the RoPE execution policy for 310P.

    Returns one of ``"auto"``, ``"npu"`` or ``"native"``.  All policies run
    with real cos/sin tables extracted on CPU (310P AICore has no complex-dtype
    compute -- see ``build_rope_dispatcher``).  ``"auto"`` attempts
    ``npu_rotary_mul`` (interleave) once on first use and sticks to the
    real-valued algebra on failure; ``"npu"`` fails loudly instead.
    """
    env = os.environ if environ is None else environ
    policy = env.get("QWEN21_310P_ROPE", "auto").strip().lower()
    if policy not in ("auto", "npu", "native"):
        logger.warning("Unknown QWEN21_310P_ROPE=%r; falling back to auto", policy)
        return "auto"
    return policy


def resolve_vae_memory(vae_use_tiling: bool, vae_use_slicing: bool) -> tuple[bool, bool]:
    """Force tiling+slicing: the Qwen-Image-2.1 VAE upsampling chain peaks
    multi-GB untiled on a ~21 GB 310P chip (same class of risk as SDXL R4)."""
    return True, True


def resolve_encoder_placement() -> str:
    """310P pins the Qwen3-VL text encoder to CPU (see docstring finding 4).

    Returns ``"cpu"``.  The generic NPU pipeline would otherwise (a) run the
    axial-RoPE table build on the NPU device context and trip the 310P TBE
    RealDiv gap, and (b) spend ~18 GB of the 21.589 GB chip, OOMing the DiT.
    """
    return "cpu"


def should_disable_nz(od_config: Any) -> bool:
    """The FRACTAL_NZ linear-weight cast (29) is 910-oriented; never run it
    on 310P.  Under the FP16 runtime the BF16 gate already skips it, this is
    the defensive second layer."""
    return is_310p()


def resolve_quantization_override(od_config: Any) -> None:
    """310P has no MindIE diffusion quant path: drop KV-cache quant extras
    and component quantization configs with a loud warning (mirrors SDXL R6)."""
    extras = getattr(od_config, "extras", None)
    if isinstance(extras, dict):
        for key in ("prefix_kv_cache_dtype",):
            if extras.get(key):
                logger.warning(
                    "310P Qwen-Image-2.1: dropping extras[%r]=%r (no 310P KV-quant path).",
                    key,
                    extras[key],
                )
                extras[key] = None
    quant = getattr(od_config, "quantization_config", None)
    if quant is not None:
        logger.warning("310P Qwen-Image-2.1: ignoring quantization_config=%r (unvalidated).", quant)
        od_config.quantization_config = None


def probe_npu_rotary_interleave() -> bool:
    """One-time capability probe for ``npu_rotary_mul(..., rotary_mode="interleave")``.

    Runs on the default NPU device with a minimal FP32 sentinel.  The TTS
    310P path proves ``npu_rotary_mul`` exists on 310P (half mode); interleave
    mode is the open question this probe answers.  Never raises: any error
    means "use native RoPE".
    """
    try:
        import torch_npu

        if not hasattr(torch_npu, "npu_rotary_mul"):
            return False
        device = torch.device("npu")
        x = torch.zeros(1, 1, 1, 4, dtype=torch.float32, device=device)
        cos = torch.ones(1, 1, 1, 4, dtype=torch.float32, device=device)
        sin = torch.zeros(1, 1, 1, 4, dtype=torch.float32, device=device)
        torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")
        torch.npu.synchronize(device)
        return True
    except Exception as exc:  # noqa: BLE001 - capability probe must never raise
        logger.info("310P npu_rotary_mul interleave probe failed (%s); using native RoPE.", exc)
        return False


def build_rope_dispatcher(original_npu_impl, native_impl, policy: str):
    """Return a 310P-safe RoPE implementation bound to *policy*.

    310P AICore has **no complex-dtype compute**: the transformer's freqs
    table is complex (``torch.polar``) and both stock implementations run
    ``torch.real/imag`` on it *where freqs lives* (NPU) -- which the device
    rejects (``Current settings do not support Complex dtype`` /
    ERR01007).  This dispatcher therefore never executes complex ops on the
    device: real cos/sin tables are extracted on CPU and moved to the
    operand device as REAL tensors.

    * ``"auto"``   -> ``npu_rotary_mul`` (interleave) with those real
      tables; on any failure permanently switch to the real-valued algebra
      (kept algebraically identical to the transformer's native fallback).
    * ``"native"`` -> always the real-valued algebra.
    * ``"npu"``    -> always ``npu_rotary_mul`` with real tables (fail
      loudly if broken).

    ``original_npu_impl`` / ``native_impl`` are retained for API
    compatibility (and as the algebraic reference implementations).
    """
    if policy not in ("auto", "native", "npu"):
        raise ValueError(f"unknown 310P RoPE policy: {policy!r}")

    state = {"mode": policy, "probe_done": False}
    table_cache: dict[str, Any] = {"ref": None, "id": None, "cos": None, "sin": None}

    def _real_tables(freqs: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU-side ``torch.real/imag`` of the (complex) freqs table.

        Cached per freqs identity; the strong reference keeps the id stable
        against GC recycling (same hazard the stock ``_NPU_ROPE`` table
        documents).
        """
        fid = id(freqs)
        if table_cache["id"] != fid:
            base = freqs.detach()
            if base.device.type != "cpu":
                base = base.cpu()
            table_cache["cos"] = torch.real(base).float().to(device)
            table_cache["sin"] = torch.imag(base).float().to(device)
            table_cache["ref"] = freqs
            table_cache["id"] = fid
        return table_cache["cos"], table_cache["sin"]

    def _native_algebra(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Interleaved rotate in FP32 (cos/sin shaped ``[S, D // 2]``) --
        identical algebra to ``_apply_qwen_image21_rotary_emb_native``."""
        x_pairs = x.float().unflatten(-1, (-1, 2))
        x1, x2 = x_pairs.unbind(-1)
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
        return out.flatten(-2).to(x.dtype)

    def _npu_algebra(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        import torch_npu

        c = cos.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)
        s = sin.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)
        return torch_npu.npu_rotary_mul(x.float(), c, s, rotary_mode="interleave").to(x.dtype)

    def dispatch(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        cos, sin = _real_tables(freqs, x.device)
        if state["mode"] == "native":
            return _native_algebra(x, cos, sin)
        try:
            return _npu_algebra(x, cos, sin)
        except Exception as exc:  # noqa: BLE001
            if state["mode"] == "npu":
                raise
            state["mode"] = "native"
            state["probe_done"] = True
            logger.warning(
                "310P Qwen-Image-2.1: npu_rotary_mul interleave failed (%s); "
                "using real-valued algebraic RoPE (CPU-extracted tables) "
                "for this process.",
                exc,
            )
            return _native_algebra(x, cos, sin)

    return dispatch


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def _patch_transformer_module() -> None:
    """Rebind the transformer's NPU RoPE function and disable NZ casting.

    Both are monkey-patches on module/class attributes (same style as
    ``_310p/patch/qwen3_tts.py``); the model modules stay pristine.
    """
    from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

    if getattr(transformer, "_310p_patch_applied", False):
        return

    policy = resolve_rope_policy()
    native_impl = transformer._apply_qwen_image21_rotary_emb_native
    if policy == "auto":
        # Defer the probe to the first real call (patch time is platform init
        # time; the NPU device may not be bound to this rank yet).
        transformer._apply_qwen_image21_rotary_emb_npu = build_rope_dispatcher(
            transformer._apply_qwen_image21_rotary_emb_npu,
            native_impl,
            "auto",
        )
    else:
        transformer._apply_qwen_image21_rotary_emb_npu = (
            transformer._apply_qwen_image21_rotary_emb_npu if policy == "npu" else native_impl
        )
    if policy == "auto":
        # Record probe status lazily so logs carry the actual verdict.
        transformer._310p_rope_probe_pending = True

    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        QwenImage21Transformer2DModel,
    )

    original_supported = QwenImage21Transformer2DModel._nz_weights_supported_for

    @staticmethod
    def _nz_weights_supported_for_310p(od_config, quant_config):
        if is_310p():
            return False
        return original_supported(od_config, quant_config)

    QwenImage21Transformer2DModel._nz_weights_supported_for = _nz_weights_supported_for_310p  # type: ignore[assignment]
    _patch_rope_tables_cpu()
    transformer._310p_patch_applied = True
    logger.info(
        "310P Qwen-Image-2.1 patches installed: rope=%s, nz-cast=disabled, decode-graphs=cpu-only(eager).",
        policy,
    )



def _patch_rope_tables_cpu() -> None:
    """Keep the complex RoPE frequency tables on the CPU.

    ``QwenImage21Rope.forward`` moves the complex table to the operand
    device (``freq.to(device)``) and the stock algebra then runs complex
    ops where it lands; on 310P the AICore cannot transfer or compute
    complex tensors ("Current settings do not support Complex dtype",
    ERR01007 OPS feature not supported).  The RoPE dispatcher installed by
    :func:`_patch_transformer_module` consumes the table through
    CPU-side ``torch.real/imag`` and moves only the REAL cos/sin tables
    to the operand device, so the complex tensor must never leave the
    host.  This wrapper re-implements the stock token-index arithmetic on
    the CPU and returns the selected complex rows on the CPU.
    """
    from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

    rope_cls = transformer.QwenImage21Rope
    if getattr(rope_cls, "_310p_forward_patched", False):
        return

    def forward_tables_on_cpu(
        self,
        img_shapes: list[tuple[int, int, int]],
        image_pad_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # Same token-index arithmetic as the stock forward, executed on the
        # CPU so the complex table never touches the NPU.  ``device`` stays
        # in the signature for call-site compatibility.
        frame_index, height_index, width_index = [], [], []
        image_height_index, image_width_index = [], []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()

        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend(
                [h for h in range(-(height - height // 2), height // 2) for _ in range(width)]
            )
            image_width_index.extend(
                [w for _ in range(height) for w in range(-(width - width // 2), width // 2)]
            )

        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))

        pad_mask = image_pad_mask.cpu().bool()
        frame_index = torch.tensor(frame_index, dtype=torch.long)
        height_index = frame_index.clone()
        width_index = frame_index.clone()
        height_index[pad_mask] = torch.tensor(image_height_index, dtype=torch.long)
        width_index[pad_mask] = torch.tensor(image_width_index, dtype=torch.long)

        height_index = torch.where(
            height_index < 0, height_index + self.freqs[1].shape[0], height_index
        )
        width_index = torch.where(
            width_index < 0, width_index + self.freqs[2].shape[0], width_index
        )
        return torch.cat(
            [
                self.freqs[0].index_select(0, frame_index),
                self.freqs[1].index_select(0, height_index),
                self.freqs[2].index_select(0, width_index),
            ],
            dim=-1,
        )

    rope_cls.forward = forward_tables_on_cpu  # type: ignore[assignment]
    rope_cls._310p_forward_patched = True
    logger.info("310P Qwen-Image-2.1: QwenImage21Rope.forward keeps complex freqs tables on CPU.")


def _patch_loader_cpu_build() -> None:
    """Run the whole pipeline construction on CPU on 310P.

    The ``"default"`` load-format branch of the diffusers loader builds the
    pipeline inside ``with target_device:`` (npu:0).  Every device-less op at
    construction time (encoder axial-RoPE arange/div, VAE RMS-norm
    ``torch.ones/zeros``, transformer RoPE ``freqs``) then lands on the NPU
    and hits the 310P CANN 9.1.0 dynamic-TBE compile gap (RealDiv first,
    then OnesLike; upstream 910B CI on the identical version stack never
    trips it -- research/012).

    Upstream's own ``custom_pipeline`` branch solves this class of problem
    the same way ("build on CPU first, then explicitly move to the target
    device").  We mirror it by running ``initialize_model`` under
    ``torch.device("cpu")``.  NOTE: placement is NOT a side effect of the
    stock loader -- the non-quant branch of
    ``_process_weights_after_loading`` moves nothing (quant-method modules
    only); the stock flow gets placement from the build context itself.
    ``_patch_loader_final_placement`` therefore performs the explicit
    post-load CPU->NPU move, while the encoder stays CPU-pinned
    (finding 4).
    """
    import importlib

    loader_mod = importlib.import_module(
        "vllm_omni.diffusion.model_loader.diffusers_loader"
    )
    if getattr(loader_mod, "_310p_cpu_build_patched", False):
        return
    orig_initialize_model = loader_mod.initialize_model

    def _initialize_model_on_cpu(od_config: Any) -> Any:
        with torch.device(resolve_encoder_placement()):
            return orig_initialize_model(od_config)

    loader_mod.initialize_model = _initialize_model_on_cpu
    loader_mod._310p_cpu_build_patched = True
    logger.info(
        "310P Qwen-Image-2.1: pipeline construction pinned to CPU "
        "(310P dynamic-TBE gap: RealDiv/OnesLike); loader post-load path "
        "handles NPU placement."
    )


def _patch_torch_sdpa_npu_gate() -> None:
    """Gate torch's ``F.scaled_dot_product_attention`` on 310P.

    The VAE mid-block attention (``QwenImage21AttentionBlock``, decode
    path) calls ``F.scaled_dot_product_attention`` directly, bypassing
    the DiT backend swap.  CANN 9.1.0's fused attention on 310P3
    rejects its fp16 path (``can not cast format when output is
    input``, ERR01007 OPS feature not supported).  The wrapper
    decomposes NPU calls into the verified manual dense/chunked path
    and forwards every non-NPU call to the stock function unchanged
    (CPU text-encoder parity).
    """
    F = torch.nn.functional
    if getattr(F, "_310p_sdpa_npu_gated", False):
        return
    original = F.scaled_dot_product_attention

    def sdpa_npu_gated(query, key, value, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, **kwargs):
        if query.device.type != "npu":
            return original(query, key, value, attn_mask=attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal,
                            scale=scale, **kwargs)
        if dropout_p:
            raise RuntimeError("310P manual SDPA does not support dropout_p > 0")
        if scale is None:
            scale = 1.0 / math.sqrt(query.size(-1))
        return _manual_sdpa_forward(query, key, value, attn_mask, is_causal, scale)

    F.scaled_dot_product_attention = sdpa_npu_gated
    F._310p_sdpa_npu_gated = True
    logger.info(
        "310P Qwen-Image-2.1: torch F.scaled_dot_product_attention gated "
        "(NPU -> manual eager path, non-NPU passthrough) -- CANN 9.1.0 "
        "fused SDPA rejects the fp16 VAE mid-block attention."
    )


def _patch_loader_final_placement() -> None:
    """Move still-CPU components onto the local NPU after weight loading.

    The stock flow gets its NPU placement from the loader's
    ``with target_device:`` build context -- params are born on the NPU.
    With the CPU-build patch in place, every param is born on the host and
    the non-quant branch of ``_process_weights_after_loading`` moves
    nothing, so the transformer (14.23 GB of weights) would otherwise stay
    on the CPU and meet NPU inputs at its first matmul (``Expected all
    tensors to be on the same device``).  The wrapper moves every
    transformer/vae attribute whose first parameter is still off-NPU; the
    CPU-pinned encoder is excluded by construction (and its ``.to`` is
    CPU-redirected anyway).
    """
    import importlib

    loader_mod = importlib.import_module(
        "vllm_omni.diffusion.model_loader.diffusers_loader"
    )
    loader_cls = getattr(loader_mod, "DiffusersPipelineLoader", None)
    if loader_cls is None or getattr(loader_cls, "_310p_final_placement_patched", False):
        return

    orig_load_model = loader_cls.load_model

    def load_model_310p(self, *args: Any, **kwargs: Any):
        loaded = orig_load_model(self, *args, **kwargs)
        moved = []
        for name in ("transformer", "vae"):
            comp = getattr(loaded, name, None)
            if comp is None:
                continue
            try:
                first = next(comp.parameters())
            except StopIteration:
                continue
            if first.device.type == "npu":
                continue
            target = torch.device("npu")
            if name == "vae":
                # 1024² hard-degradation ①: pin the VAE to a second die
                # (e.g. "npu:1") when the single-die budget is exceeded.
                # Opt-in via env so the default placement is unchanged.
                vae_dev = os.environ.get("VLLM_OMNI_310P_VAE_DEVICE")
                if vae_dev:
                    target = torch.device(vae_dev)
            comp.to(target)
            moved.append(f"{name}->{target}")
        if moved:
            logger.info("310P Qwen-Image-2.1: post-load placement moved %s.", moved)
        return loaded

    loader_cls.load_model = load_model_310p  # type: ignore[assignment]
    loader_cls._310p_final_placement_patched = True


# Dense score matrix byte budget before the chunked (online-softmax) path
# takes over.  The DiT is patch_size=1, so 1024^2 images mean 16384 tokens:
# a dense (1,32,16384,16384) fp16 score matrix is ~17 GB -- impossible next
# to the 14.23 GB of DiT weights on a 21.5 GB chip.  4096 tokens (512^2) is
# ~1.1 GB transient and stays dense.
_SDPA_DENSE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
_SDPA_CHUNK_SIZE = 512


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


def _manual_sdpa_forward(query: Any, key: Any, value: Any,
                         attention_mask: Any, causal: bool, scale: float) -> Any:
    """Eager SDPA decomposition for 310P3 (finding 3b).

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

    The fused SDPA aclnn kernel is rejected on 310P3 (finding 3b); the
    swap is a module-global rebind of ``sdpa._sdpa_attention`` -- every
    caller (``SDPAImpl.forward_npu`` -> ``_forward_impl``) resolves the name
    from module globals at call time, so one rebind covers the whole
    diffusion attention path on this process.
    """
    from vllm_omni.diffusion.attention.backends import sdpa as sdpa_mod

    if getattr(sdpa_mod, "_310p_manual_sdpa_patched", False):
        return
    sdpa_mod._sdpa_attention = _manual_sdpa_forward
    sdpa_mod._310p_manual_sdpa_patched = True
    logger.info(
        "310P Qwen-Image-2.1: TORCH_SDPA backend swapped to eager manual "
        "decomposition (fused SDPA kernel unsupported on 310P3, CANN 9.1.0)."
    )


def _make_cpu_pinned(model: Any, dtype_shadow: Any = None) -> Any:
    """One-shot instance overlay: redirect every ``.to(...)`` on *model*
    to CPU.  Used for the Qwen3-VL encoder (docstring finding 4): the
    pipeline chains ``.to(self.device)`` right after loading, and later
    stages may retry placement; all of it must land on CPU on 310P.

    With ``dtype_shadow`` the overlay also reports that dtype for
    ``.dtype`` (see the attribute docstring inside).

    The overlay subclasses the real class without adding storage, so the
    instance layout is unchanged and ``__class__`` swap is safe.
    """
    if getattr(model, "_310p_cpu_pinned", False):
        return model
    base = type(model)

    def _to_cpu(self: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return torch.nn.modules.module.Module.to(self, "cpu")

    attrs: dict[str, Any] = {"to": _to_cpu}
    if dtype_shadow is not None:
        # Report the 310P runtime dtype while the parameters keep their
        # native (CPU-friendly BF16) dtype: the pipeline reads
        # ``self.text_encoder.dtype`` to choose the ``prompt_embeds``
        # dtype, so shadowing it moves the bf16->fp16 cast to the DiT
        # boundary instead of running a CPU FP16 encoder forward.
        def _shadowed_dtype(self: Any, _d: Any = dtype_shadow) -> Any:
            return _d

        attrs["dtype"] = property(_shadowed_dtype)
    overlay = type(f"_CPUPinned{base.__name__}", (base,), attrs)
    model.__class__ = overlay
    model._310p_cpu_pinned = True
    return model


class _CPUProcessorProxy:
    """Forwarding proxy for the Qwen3-VL processor: its ``__call__`` output
    (a ``BatchFeature``) normally gets ``.to(self.device)`` (= NPU) by the
    pipeline; with the encoder pinned to CPU the inputs must stay on CPU.
    NPU-directed ``.to`` calls are redirected to CPU, everything else is
    untouched."""

    def __init__(self, processor: Any) -> None:
        object.__setattr__(self, "_p", processor)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._p(*args, **kwargs)
        return _CPUFeatureProxy(out)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._p, name)


class _CPUFeatureProxy:
    """BatchFeature wrapper: NPU-directed ``.to`` -> CPU, rest forwarded."""

    def __init__(self, feature: Any) -> None:
        object.__setattr__(self, "_f", feature)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        dev = args[0] if args else kwargs.get("device")
        if dev is not None and "npu" in str(dev).lower():
            return self._f.to("cpu")
        return self._f.to(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._f, name)



def _patch_encode_prompt_boundary(pipeline_module: Any) -> None:
    """Move the encoder outputs to the DiT device at the CPU->NPU boundary.

    With the encoder pinned to the CPU (310P), ``encode_prompt`` returns
    the prompt embeds/masks on the host; the stock
    ``prompt_embeds.to(dtype=...)`` cast is dtype-only and leaves the
    tensors on the CPU, so the DiT (NPU) meets mixed-device operands at
    its first matmul (``Expected all tensors to be on the same device``).
    The wrapper moves every returned tensor to the transformer device --
    a real-dtype HCCS copy, no complex involved.
    """
    pipeline_cls = getattr(pipeline_module, "QwenImage21Pipeline", None)
    if pipeline_cls is None or getattr(pipeline_cls, "_310p_encode_boundary_patched", False):
        return

    original = pipeline_cls.encode_prompt

    def _dit_device(pipeline_self: Any) -> torch.device:
        t = pipeline_self.transformer
        d = getattr(t, "device", None)
        if isinstance(d, torch.device):
            return d
        try:
            return next(t.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_prompt_310p(self, *args: Any, **kwargs: Any):
        result = original(self, *args, **kwargs)
        device = _dit_device(self)
        items = list(result) if isinstance(result, (tuple, list)) else [result]
        moved = [t.to(device) if torch.is_tensor(t) and t.device != device else t for t in items]
        return type(result)(moved) if isinstance(result, (tuple, list)) else moved[0]

    pipeline_cls.encode_prompt = encode_prompt_310p  # type: ignore[assignment]
    pipeline_cls._310p_encode_boundary_patched = True
    logger.info("310P Qwen-Image-2.1: encode_prompt moves outputs to the DiT device (CPU->NPU boundary).")


def _patch_pipeline_encoder_placement(pipeline_module: Any) -> None:
    """Pin the Qwen3-VL text encoder (and its processor inputs) to CPU.

    Rewires the pipeline module's own ``from_pretrained_with_prefetch``
    reference (module-local; other models are untouched): the encoder
    factory runs under ``torch.device("cpu")`` so transformers' weight
    finalization (axial-RoPE table build) creates no NPU tensors, and the
    returned model is ``_make_cpu_pinned``; the processor is wrapped in
    ``_CPUProcessorProxy``.
    """
    if getattr(pipeline_module, "_310p_encoder_patched", False):
        return
    orig = pipeline_module.from_pretrained_with_prefetch
    enc_cls = pipeline_module.Qwen3VLForConditionalGeneration
    proc_cls = pipeline_module.Qwen3VLProcessor

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

    def _cast_to_runtime_dtype(model: Any, what: str) -> Any:
        """Cast *model* weights to the 310P runtime dtype (FP16).

        HF ``from_pretrained`` honours the checkpoint's ``torch_dtype``
        (BFloat16 for Qwen3-VL / the VAE), but 310P has no BF16 compute
        path and the DiT is pinned to FP16 by ``resolve_runtime_dtype``:
        a BF16 component then meets FP16 tensors at the first boundary
        matmul and the NPU refuses it (``m1/m2 ... c10::BFloat16 !=
        c10::Half``).  Casting on CPU, before any placement redirect,
        keeps every pipeline component on one dtype.
        """
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            return model
        runtime_dtype = resolve_runtime_dtype(param_dtype)
        if param_dtype != runtime_dtype:
            model = model.to(dtype=runtime_dtype)
            logger.info(
                "310P Qwen-Image-2.1: %s cast %s -> %s "
                "(310P has no BF16 compute path; single-dtype pipeline).",
                what, param_dtype, runtime_dtype,
            )
        return model

    vae_cls = getattr(pipeline_module, "DistributedAutoencoderKLQwenImage21", None)

    def _param_dtype(model: Any) -> Any:
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return None

    def wrapped(factory: Any, *args: Any, **kwargs: Any) -> Any:
        if _same_classmethod(factory, enc_cls, "from_pretrained"):
            with torch.device(resolve_encoder_placement()):
                model = orig(factory, *args, **kwargs)
            native = _param_dtype(model)
            if native is None:
                model = _make_cpu_pinned(model)
            else:
                runtime_dtype = resolve_runtime_dtype(native)
                if native != runtime_dtype:
                    # Keep the CPU encoder weights in their native BF16 (CPU
                    # BF16 kernels are solid; CPU FP16 forwards overflow-prone
                    # and slow) and move the bf16->fp16 cast to the DiT
                    # boundary: shadowing .dtype makes the pipeline's
                    # ``prompt_embeds.to(dtype=self.text_encoder.dtype)`` cast
                    # the embeddings to the transformer's FP16.
                    model = _make_cpu_pinned(model, dtype_shadow=runtime_dtype)
                    logger.info(
                        "310P Qwen-Image-2.1: Qwen3-VL text encoder keeps %s "
                        "CPU weights; .dtype shadowed to %s so prompt "
                        "embeddings are cast at the DiT boundary.",
                        native, runtime_dtype,
                    )
                else:
                    model = _make_cpu_pinned(model)
            logger.info(
                "310P Qwen-Image-2.1: Qwen3-VL text encoder pinned to %s "
                "(RealDiv guard + memory budget); .to(...) redirected to CPU.",
                resolve_encoder_placement(),
            )
            return model
        if vae_cls is not None and _same_classmethod(factory, vae_cls, "from_pretrained"):
            model = orig(factory, *args, **kwargs)
            return _cast_to_runtime_dtype(model, "VAE")
        if _same_classmethod(factory, proc_cls, "from_pretrained"):
            return _CPUProcessorProxy(orig(factory, *args, **kwargs))
        return orig(factory, *args, **kwargs)

    pipeline_module.from_pretrained_with_prefetch = wrapped
    pipeline_module._310p_encoder_patched = True


def apply_qwen_image_2_1_310p_patches(od_config: Any) -> None:
    """Install the 310P specialization for Qwen-Image-2.1 on this rank.

    Idempotent; safe to call from the diffusion runtime hook on every rank.
    """
    global _APPLIED
    if _APPLIED:
        return

    od_config.dtype = resolve_runtime_dtype(getattr(od_config, "dtype", torch.bfloat16))
    od_config.enforce_eager = True  # NPU has no inductor; CUDA-graph decode is CUDA-only
    resolve_quantization_override(od_config)

    vae_tiling, vae_slicing = resolve_vae_memory(
        getattr(od_config, "vae_use_tiling", False),
        getattr(od_config, "vae_use_slicing", False),
    )
    od_config.vae_use_tiling = vae_tiling
    od_config.vae_use_slicing = vae_slicing

    attn = os.environ.get("QWEN21_310P_ATTENTION", "sdpa").strip().lower()
    if attn not in ("sdpa", "flash_attn"):
        logger.warning("Unknown QWEN21_310P_ATTENTION=%r; using sdpa.", attn)
        attn = "sdpa"
    if attn == "flash_attn":
        od_config.diffusion_attention_backend = "FLASH_ATTN"  # experiment knob only
    else:
        # Leave the resolver's default (TORCH_SDPA on NPU without mindiesd).
        logger.info("310P Qwen-Image-2.1: attention backend = TORCH_SDPA (dense block-causal mask path).")

    _patch_transformer_module()
    _patch_loader_cpu_build()
    _patch_loader_final_placement()
    _patch_attention_manual_sdpa()
    _patch_torch_sdpa_npu_gate()
    from vllm_omni.diffusion.models.qwen_image_21 import (
        pipeline_qwen_image_21 as pipeline_module,
    )

    _patch_pipeline_encoder_placement(pipeline_module)
    _patch_encode_prompt_boundary(pipeline_module)
    _APPLIED = True


def maybe_apply_diffusion_310p_patches(od_config: Any) -> None:
    """Self-gated standalone entry point.

    No-op unless ``is_310p()`` is True *and* ``model_class_name == MODEL_ARCH``.
    Once the task-11 registry hook lands in ``init_diffusion_model_runner_runtime``
    this file is registered via ``(MODEL_ARCH, apply_patch)`` and the registry
    drives the same function; keep both paths for the rebase window.
    """
    if not is_310p():
        logger.warning("310P gate DIAG: is_310p()=False; patch entry no-op.")
        return
    if getattr(od_config, "model_class_name", None) != MODEL_ARCH:
        logger.warning(
            "310P gate DIAG: model_class_name=%r != %r; patch entry no-op.",
            getattr(od_config, "model_class_name", None), MODEL_ARCH,
        )
        return
    logger.warning(
        "310P gate DIAG: applying Qwen-Image-2.1 310P patches "
        "(model_class_name=%r).", MODEL_ARCH,
    )
    apply_qwen_image_2_1_310p_patches(od_config)


# Registry-facing alias (COMPARISON-POLICY.md §5).
apply_patch = apply_qwen_image_2_1_310p_patches
