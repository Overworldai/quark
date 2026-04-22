"""Occupancy calculator.

Given a kernel's footprint (registers, smem) and the target SM specs,
compute how many blocks can fit per SM and the resulting occupancy.

This is conservative — we use the upper-bound register count from the
PTX program (every .reg we declared). ptxas may reduce via SSA coalescing,
but that's optimization on top of our worst case.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SMSpec:
    """Hardware limits for one SM (Streaming Multiprocessor)."""

    name: str
    sm_arch: int  # e.g. 120 for sm_120 (RTX 5090)
    n_sms: int  # SMs on the device
    max_threads_per_sm: int
    max_blocks_per_sm: int
    max_warps_per_sm: int
    max_regs_per_sm: int  # 32-bit registers
    max_regs_per_thread: int
    max_smem_per_sm: int  # bytes
    max_smem_per_block: int  # bytes (driver allocatable)
    warp_size: int = 32
    reg_alloc_unit: int = 8  # per-thread reg granularity (8 on Volta+)

    @property
    def max_warps_total(self) -> int:
        return self.max_warps_per_sm


# Common GPU specs
RTX_5090 = SMSpec(
    name="RTX 5090",
    sm_arch=120,
    n_sms=170,
    max_threads_per_sm=1536,
    max_blocks_per_sm=24,  # Blackwell
    max_warps_per_sm=48,
    max_regs_per_sm=65536,
    max_regs_per_thread=255,
    max_smem_per_sm=102400,  # 100 KB
    max_smem_per_block=102400,
)

RTX_4090 = SMSpec(
    name="RTX 4090",
    sm_arch=89,
    n_sms=128,
    max_threads_per_sm=1536,
    max_blocks_per_sm=24,
    max_warps_per_sm=48,
    max_regs_per_sm=65536,
    max_regs_per_thread=255,
    max_smem_per_sm=102400,
    max_smem_per_block=101376,  # ~99 KB
)


# ───────────────────────────────────────────────────────────────────────
# Device queries (cached) — read once from cupy and reuse forever
# ───────────────────────────────────────────────────────────────────────
#
# Kernel ``is_valid()`` checks and ``SharedLayout(capacity=...)`` calls
# all need a "max usable dynamic smem per block" cap. We used to
# hardcode 100 KB which happens to match the 5090, but it's wrong on
# the 4090 (~99 KB) and any future device. Read it from
# ``cudaDevAttrMaxSharedMemoryPerBlockOptin`` instead.

_smem_cap_cache: int | None = None
_n_sms_cache: int | None = None
_compute_cap_cache: int | None = None


def current_device_smem_cap() -> int:
    """Max dynamic smem per block (bytes) for the current device.

    New code should prefer
    ``popcorn.device.current_device().caps.max_smem_per_block``.
    """
    global _smem_cap_cache
    if _smem_cap_cache is None:
        from popcorn.device import current_device

        _smem_cap_cache = int(current_device().caps.max_smem_per_block)
    return _smem_cap_cache


def current_device_sm_count() -> int:
    """Number of SMs on the current device (cached)."""
    global _n_sms_cache
    if _n_sms_cache is None:
        from popcorn.device import current_device

        _n_sms_cache = int(current_device().caps.compute_unit_count)
    return _n_sms_cache


def current_device_compute_capability() -> int:
    """Compute capability as a packed int (e.g. 120 for sm_120)."""
    global _compute_cap_cache
    if _compute_cap_cache is None:
        from popcorn.device import current_device

        cc = current_device().caps.compute_capability
        if cc is None:
            raise RuntimeError("current_device_compute_capability: no CC (non-CUDA device?)")
        major, minor = cc
        _compute_cap_cache = major * 10 + minor
    return _compute_cap_cache


@dataclass
class Occupancy:
    """Result of an occupancy computation."""

    threads_per_block: int
    regs_per_thread: int  # estimated upper bound
    smem_per_block: int  # bytes
    blocks_per_sm: int  # min of all limits
    warps_per_sm: int
    occupancy_pct: float  # warps_per_sm / max_warps_per_sm
    limits: dict[str, int]  # what each constraint allows
    bottleneck: str  # name of the binding constraint

    def __repr__(self):
        lines = [
            "Occupancy(",
            f"  threads/block:     {self.threads_per_block}",
            f"  regs/thread (est): {self.regs_per_thread}",
            f"  smem/block:        {self.smem_per_block} bytes",
            f"  blocks/SM:         {self.blocks_per_sm}",
            f"  warps/SM:          {self.warps_per_sm}",
            f"  occupancy:         {self.occupancy_pct:.1%}",
            f"  bottleneck:        {self.bottleneck}",
            f"  limits:            {self.limits}",
            ")",
        ]
        return "\n".join(lines)


def compute_occupancy(
    *,
    threads_per_block: int,
    regs_per_thread: int,
    smem_per_block: int,
    sm: SMSpec,
) -> Occupancy:
    """Compute occupancy for a kernel with the given footprint."""
    warp_size = sm.warp_size
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size

    # Round registers up to allocation unit
    rounded_regs = (
        (regs_per_thread + sm.reg_alloc_unit - 1) // sm.reg_alloc_unit * sm.reg_alloc_unit
    )
    if rounded_regs > sm.max_regs_per_thread:
        rounded_regs = sm.max_regs_per_thread

    # Limits
    by_blocks = sm.max_blocks_per_sm
    by_warps = sm.max_warps_per_sm // warps_per_block if warps_per_block > 0 else 0
    by_threads = sm.max_threads_per_sm // threads_per_block if threads_per_block > 0 else 0

    regs_per_block = rounded_regs * threads_per_block
    by_regs = sm.max_regs_per_sm // regs_per_block if regs_per_block > 0 else 0

    if smem_per_block > 0:
        by_smem = sm.max_smem_per_sm // smem_per_block
    else:
        by_smem = sm.max_blocks_per_sm

    if smem_per_block > sm.max_smem_per_block:
        # Block doesn't even fit
        by_smem = 0

    limits = {
        "max_blocks": by_blocks,
        "max_warps": by_warps,
        "max_threads": by_threads,
        "max_regs": by_regs,
        "max_smem": by_smem,
    }

    blocks_per_sm = min(limits.values())
    bottleneck = min(limits.items(), key=lambda kv: kv[1])[0]
    warps_per_sm = blocks_per_sm * warps_per_block
    occupancy_pct = warps_per_sm / sm.max_warps_per_sm

    return Occupancy(
        threads_per_block=threads_per_block,
        regs_per_thread=rounded_regs,
        smem_per_block=smem_per_block,
        blocks_per_sm=blocks_per_sm,
        warps_per_sm=warps_per_sm,
        occupancy_pct=occupancy_pct,
        limits=limits,
        bottleneck=bottleneck,
    )
