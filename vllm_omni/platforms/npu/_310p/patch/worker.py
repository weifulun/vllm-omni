# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Omni NPU worker device initialization for 310P.

This patches:
    vllm_omni.platforms.npu.worker.base.OmniNPUWorkerBase

Omni AR/generation workers share this base initialization path, so the 310P
compile-mode setup is applied here after device initialization.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.sample.sampler import Sampler
from vllm_ascend._310p.sample.sampler import AscendSampler310

from vllm_omni.platforms.npu._310p import disable_jit_compile
from vllm_omni.platforms.npu.worker import base as worker_base

logger = init_logger(__name__)


def _force_hccl_comm_early(worker) -> None:
    """Create the HCCL comm once, while the chip is still near-idle.

    task-8 real-device finding (128, 2026-11-02): torch_npu's hccl backend
    creates the real HCCL comm lazily, at the FIRST collective. In vLLM v1
    that first collective is a TP allreduce inside
    ``determine_available_memory -> profile_run -> _dummy_run`` — i.e. AFTER
    ~10.4 GiB of weights + replicated encoders are resident per chip. At that
    point the HCCL module's device-memory request fails deterministically on
    the 310P3 driver with ``EL0019 Resource_Error_Insufficient_Device_Memory``
    ("Failed to allocate 2122720 bytes") even though several GiB of HBM remain
    free, on BOTH tested Duo cards (C1 and C2). A standalone 2-rank
    comm-init + allreduce probe passes at 0/6/10/13 GiB of pre-allocation, so
    the failure correlates with the caching-allocator pool state at
    comm-creation time, not with free capacity.

    Creating a comm here (1-element allreduce, right after distributed-env
    init and before any model allocation) is numerically neutral; the comm
    persists on the group's ProcessGroup, so all later collectives reuse it.

    NOTE: vLLM creates SEPARATE ProcessGroups for the world group and the TP
    group (``ensure_model_parallel_initialized`` -> ``new_group``), and
    vllm-ascend can add further optional groups (kvpp/otp/lmtp/emtp/mlptp);
    each has its own lazy HCCL comm. The collective that crashes vLLM v1 is
    the TP group's (``get_tp_group().all_reduce`` ->
    ``torch.ops.vllm.all_reduce``), so EVERY multi-rank group that exists at
    init time is forced here (deduplicated by ProcessGroup identity).
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_world_size() <= 1:
        return
    if getattr(worker, "_hccl_comm_forced", False):
        return
    dev = torch.device(f"npu:{worker.local_rank}")
    reserved = torch.npu.memory_reserved(dev) / 1024**3
    allocated = torch.npu.memory_allocated(dev) / 1024**3
    seen_pgs: set[int] = set()
    failures: list[str] = []

    def _force(label: str, pg, allreduce_fn) -> None:
        if pg is None or id(pg) in seen_pgs:
            return
        try:
            allreduce_fn(torch.zeros(1, device=dev))
            torch.npu.synchronize()
            seen_pgs.add(id(pg))
            logger.info(
                "[310P][rank%d] forced early HCCL comm (%s) at reserved=%.2fGiB alloc=%.2fGiB",
                worker.rank, label, reserved, allocated,
            )
        except Exception:
            logger.exception(
                "[310P][rank%d] early HCCL comm forcing (%s) failed; continuing "
                "with lazy comm creation at first collective",
                worker.rank, label,
            )
            failures.append(label)

    # 1. world/default group
    try:
        world_pg = dist.group.WORLD if hasattr(dist, "group") else None
    except Exception:  # noqa: BLE001
        world_pg = None
    _force("world/default", world_pg, dist.all_reduce)

    # 2. every group registered in vllm's parallel-state registry
    #    (the lookup table behind torch.ops.vllm.all_reduce)
    try:
        import vllm.distributed.parallel_state as vps

        for name, factory in list(getattr(vps, "_groups", {}).items()):
            try:
                gc = factory()
            except Exception:  # noqa: BLE001
                continue
            if gc is None or getattr(gc, "world_size", 1) <= 1:
                continue
            _force(f"vllm/{name}", getattr(gc, "device_group", None), gc.all_reduce)
    except Exception:  # noqa: BLE001
        logger.exception(
            "[310P][rank%d] early HCCL comm forcing (vllm group registry) unavailable",
            worker.rank,
        )

    # 3. optional vllm-ascend groups (kvpp/p-tp/otp/lmtp/emtp/mlptp/mc2)
    try:
        import vllm_ascend.distributed.parallel_state as vaps

        for name in ("_KVPP", "_P_TP", "_OTP", "_LMTP", "_EMBED_TP", "_MLP_TP", "_MC2"):
            gc = getattr(vaps, name, None)
            if gc is None or getattr(gc, "world_size", 1) <= 1:
                continue
            _force(f"ascend/{name}", getattr(gc, "device_group", None), gc.all_reduce)
    except Exception:  # noqa: BLE001
        logger.exception(
            "[310P][rank%d] early HCCL comm forcing (ascend groups) unavailable",
            worker.rank,
        )

    worker._hccl_comm_forced = not failures


class _OmniNPUWorkerBase310P(worker_base.OmniNPUWorkerBase):
    def _init_device(self):
        device = super()._init_device()
        disable_jit_compile()
        # 310P: HCCL comm is created lazily at first collective (see
        # _force_hccl_comm_early docstring); force it while the chip is idle.
        _force_hccl_comm_early(self)
        return device


def _install_empty_cache_noop() -> None:
    """310P EH0012 workaround for the standard image (model-mossfull, 2026-11-02).

    On the standard 310P image, the control graph submitted by
    ``torch_npu``'s ``empty_cache()`` poisons later ``aclrt`` allocator
    lookups on the multi-worker (TP>1) path:
    ``Invalid_Argument(EH0012): aclrtAllocatorGetByStream failed ... The
    stream is not registered with any allocator`` is raised around the
    KV-cache reshape op (``ReshapeCacheOperation``) during worker
    initialization. Skipping the flush is semantically lossless
    (``empty_cache`` only releases uncached allocator blocks) and avoids
    the defective control-graph submission. Installed on
    ``torch.accelerator.empty_cache`` (vLLM's model-loader call site) and
    ``torch.npu.empty_cache`` (the underlying implementation).
    """
    import logging

    import torch

    logger = logging.getLogger(__name__)

    def _wrap(name: str, module) -> None:
        fn = getattr(module, "empty_cache", None)
        if fn is None or getattr(fn, "_moss_310p_ec_noop", False):
            return

        def noop(*args, **kwargs):
            logger.info(
                "[310P] skipping empty_cache (%s) - EH0012 workaround", name
            )

        noop._moss_310p_ec_noop = True
        setattr(module, "empty_cache", noop)
        logger.info("[310P] empty_cache no-op installed for %s", name)

    _wrap("torch.accelerator", torch.accelerator)
    try:
        import torch.npu as _npu

        _wrap("torch.npu", _npu)
    except ImportError:  # torch_npu optional at import time
        pass


def _install_shared_chip_kv_fix() -> None:
    """310P multi-stage same-chip KV budget fix (task-8, 128 C1, 2026-11-03).

    The vendor 310P worker sizes the KV budget as
    ``(requested_memory - profile_peak - non_torch_cleared) // 2`` where
    ``requested_memory = gpu_memory_utilization * total``. The profile
    measurement reflects the PHYSICAL chip occupancy at profile time, so on a
    chip that an earlier stage already occupies (MiniCPM-o places stage 0 at
    TP2 on both chips, the stage 1 talker on chip 0, stage 2 on chip 1) the
    late-stage worker sees the predecessor's residue inside the measured peak
    while its own ``gpu_memory_utilization`` stays small (base 0.10 for
    stage 1): the formula goes strongly negative (real-device:
    "Available KV cache memory: -6.33 GiB") and the engine aborts with
    "No available memory for the cache blocks" even though several GiB are
    physically free. Raising the late stage's mmu cannot fix it — the vendor
    ``_init_device`` free check (``free < requested -> raise``) rejects the
    mmu values that would make the formula positive.

    For workers whose chip carries more than 50% external residue at worker
    init, replace the budget with the physically free-based one:
    free-after-profile minus a 0.5 GiB safety margin (retained caching
    allocator blocks are already outside ``free`` and hence already
    accounted for), capped by the stage's explicit
    ``kv_cache_memory_bytes`` when the deployment sets one. Fresh-chip
    workers (residue <= 50%, e.g. stage 0 on the idle chips at boot) keep
    the vendor path completely untouched.
    """
    import logging

    import torch

    logger = logging.getLogger(__name__)

    try:
        import vllm_ascend._310p.worker_310p as w310
    except ImportError:  # pragma: no cover - vendor module optional here
        return

    worker_cls = getattr(w310, "NPUWorker310", None)
    if worker_cls is None or getattr(worker_cls, "_omni_310p_shared_chip_kv", False):
        return

    original = worker_cls.determine_available_memory

    def determine_available_memory(self):
        snap = self.init_snapshot
        residue = snap.total_memory - snap.free_memory
        if residue <= 0.5 * snap.total_memory:
            return original(self)
        gib = 1024**3
        rank = getattr(self, "rank", "?")
        logger.info(
            "[310P][rank%s] shared-chip KV budget mode: %.2f GiB external "
            "residue on %.2f GiB chip at worker init (vendor mmu formula is "
            "ill-posed on an occupied chip)",
            rank, residue / gib, snap.total_memory / gib,
        )
        free_before, _ = torch.npu.mem_get_info()
        self.model_runner.profile_run()
        torch.npu.synchronize()
        free_after, _ = torch.npu.mem_get_info()
        margin = int(0.5 * gib)
        budget = int(max(free_after - margin, 0))
        explicit = getattr(self.cache_config, "kv_cache_memory_bytes", None)
        if explicit:
            budget = min(budget, int(explicit))
        try:
            budget = int(self._scale_kv_cache_memory_for_multi_group(budget))
        except Exception:  # noqa: BLE001 - single-KV-group models: keep raw
            pass
        self.available_kv_cache_memory_bytes = budget
        logger.info(
            "[310P][rank%s] Available KV cache memory: %.2f GiB "
            "(free-based, shared chip; physical free %.2f -> %.2f GiB "
            "across profile)",
            rank, budget / gib, free_before / gib, free_after / gib,
        )
        return budget

    determine_available_memory._omni_310p_shared_chip_kv = True
    worker_cls.determine_available_memory = determine_available_memory
    worker_cls._omni_310p_shared_chip_kv = True
    logger.info(
        "[310P] shared-chip KV budget fix installed on %s", worker_cls.__name__
    )


def apply_patch() -> None:
    # Triton-Ascend does not target 310P; use vLLM's native penalty path.
    AscendSampler310.apply_penalties = staticmethod(Sampler.apply_penalties)
    worker_base.OmniNPUWorkerBase = _OmniNPUWorkerBase310P
    _install_empty_cache_noop()
    _install_shared_chip_kv_fix()
