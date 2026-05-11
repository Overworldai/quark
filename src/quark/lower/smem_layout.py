"""Shared-memory layout pass — interval-graph coloring for SmemAllocOps.

Backend-neutral. Both the PTX and MSL lowerers consult the resulting
``SmemLayoutPlan`` to know where each region lives in the smem pool.
Regions whose lifetimes don't overlap get assigned the same slot
(physical storage), shrinking peak smem usage.

The pass runs OFFLINE on the IR — no codegen here, just analysis.
The lowerers then walk SmemAllocOps in their existing order and use
``plan.offset_for(alloc_id)`` instead of running their own offset
counter.

## Interval model

Every SmemAllocOp gets a half-open op-index interval ``[start, end)``
based on its declared ``Lifetime``:

  * ``KERNEL``     → ``[0, total_ops)`` — never aliased.
  * ``AUTO``       → ``[first_use_idx, last_use_idx + 1)`` from a
                     dataflow walk over the function body. Falls back
                     to ``[alloc_idx, total_ops)`` if no use is found
                     (defensive — DCE should have removed it anyway).
  * ``IN_REGION(R)``     → ``[R_start, R_end)``.
  * ``BEFORE_REGION(R)`` → ``[0, R_start)``.
  * ``AFTER_REGION(R)``  → ``[R_end, total_ops)``.

Op indices come from a pre-order flat walk of the function body. A
control-flow op (ForLoopOp, IfRegionOp, WhileLoopOp) reserves indices
for itself + its body's ops; the body's ops get sequential indices
inside that span. The op indices are stable across passes given a
fixed IR.

## Coloring

Greedy first-fit on intervals sorted by start. For each region, we
look for the lowest-id slot whose currently-assigned regions don't
intersect this region's interval. If none fits, we create a new slot.
Same-slot regions share the slot's offset; the slot's size is the
max region size in the slot (with each region's own align respected
when assigning the slot's base).

This is O(n²) worst case for n regions, but n is small (<100 in
practice) and the constant factor is tiny — no need for a fancier
interval-graph algorithm.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from quark.ir import Function
from quark.ir.lifetime import Lifetime, LifetimeKind
from quark.ir.op import (
    SmemAllocOp,
)
from quark.ir.region import Region

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Interval:
    """Half-open op-index interval [start, end). end <= total_ops."""

    start: int
    end: int

    def overlaps(self, other: _Interval) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class SmemLayoutPlan:
    """Outcome of the layout pass — consumed by the per-backend lowerers
    to emit smem decls at the right offsets.

    Allocations sharing a slot share storage (the smem layout aliases
    them). The aliasing is safe because the pass proved their lifetimes
    don't overlap in op-index order.
    """

    #: Maps each SmemAllocOp's backing-Value id to its slot id.
    region_to_slot: dict[int, int] = field(default_factory=dict)
    #: Slot id → byte offset within the threadgroup pool.
    slot_offsets: dict[int, int] = field(default_factory=dict)
    #: Slot id → byte size (max of regions placed in the slot).
    slot_sizes: dict[int, int] = field(default_factory=dict)
    #: Total smem usage in bytes.
    total_bytes: int = 0

    def offset_for(self, alloc_id: int) -> int:
        return self.slot_offsets[self.region_to_slot[alloc_id]]

    def size_for_slot(self, alloc_id: int) -> int:
        return self.slot_sizes[self.region_to_slot[alloc_id]]

    def slots_aliasing(self, alloc_id: int) -> list[int]:
        """Return all SmemAllocOp ids sharing the slot of ``alloc_id``."""
        slot = self.region_to_slot[alloc_id]
        return [aid for aid, s in self.region_to_slot.items() if s == slot]


# ---------------------------------------------------------------------------
# Op-index assignment + interval extraction
# ---------------------------------------------------------------------------


def _flat_op_indices(fn: Function) -> tuple[dict[int, int], dict[int, _Interval], int]:
    """Walk the function body in pre-order and assign each op a flat
    index. Returns:

      * ``op_index``: maps ``id(op)`` → flat index.
      * ``region_span``: maps ``id(region)`` → ``[start, end)`` covering
        all ops in the region.
      * ``total_ops``: count of ops walked.
    """
    op_index: dict[int, int] = {}
    region_span: dict[int, _Interval] = {}
    counter = [0]

    def walk_region(r: Region) -> _Interval:
        start = counter[0]
        for op in r.ops:
            op_index[id(op)] = counter[0]
            counter[0] += 1
            for sub in op.regions:
                walk_region(sub)
        end = counter[0]
        span = _Interval(start=start, end=end)
        region_span[id(r)] = span
        return span

    walk_region(fn.body)
    return op_index, region_span, counter[0]


def _value_use_indices(fn: Function, op_index: dict[int, int]) -> dict[int, list[int]]:
    """For each Value.id, the sorted list of op indices that USE it.

    Three classes of uses are recorded:
      * Direct ``operands`` of any op.
      * SharedRegion / GlobalTensor referenced via ``attrs['tensor']``
        on Load/Store/VecLoad/VecStore/AtomicRmw — the underlying
        ``alloc`` Value gets a use credit at this op's index. This is
        what makes AUTO lifetime inference catch real smem usage on
        load/store sites (operands carry only the index Values).
      * Same for ``src_tensor`` / ``dst_tensor`` on LoadMatrixOp /
        StoreMatrixOp.
    """
    from quark.ir.tensor import SharedRegion

    uses: dict[int, list[int]] = {}

    def credit_tensor(t, idx: int) -> None:
        if isinstance(t, SharedRegion):
            uses.setdefault(t.alloc.id, []).append(idx)
            if t.dyn_offset is not None:
                uses.setdefault(t.dyn_offset.id, []).append(idx)
            if t.warp_dyn_offset is not None:
                uses.setdefault(t.warp_dyn_offset.id, []).append(idx)

    def walk_region(r: Region) -> None:
        for op in r.ops:
            idx = op_index[id(op)]
            for v in op.operands:
                uses.setdefault(v.id, []).append(idx)
            # Tensor-attr uses — Load/Store/VecLoad/VecStore/AtomicRmw all
            # carry a SharedRegion/GlobalTensor in attrs['tensor'].
            for key in ("tensor", "src_tensor", "dst_tensor"):
                t = op.attrs.get(key)
                if t is not None:
                    credit_tensor(t, idx)
            for sub in op.regions:
                walk_region(sub)

    walk_region(fn.body)
    return uses


def _interval_for_alloc(
    op: SmemAllocOp,
    *,
    op_index: dict[int, int],
    region_span: dict[int, _Interval],
    uses: dict[int, list[int]],
    total_ops: int,
) -> _Interval:
    """Resolve the lifetime interval for one SmemAllocOp."""
    lifetime: Lifetime | None = op.attrs.get("lifetime")
    if lifetime is None:
        # Old-style SmemAllocOp without a lifetime attr — treat as KERNEL.
        return _Interval(start=0, end=total_ops)
    kind = lifetime.kind
    alloc_idx = op_index[id(op)]
    if kind == LifetimeKind.KERNEL:
        return _Interval(start=0, end=total_ops)
    if kind == LifetimeKind.AUTO:
        backing_id = op.results[0].id
        u = uses.get(backing_id, [])
        if not u:
            # No uses — conservatively span alloc → end. DCE should have
            # killed this, but we don't crash.
            return _Interval(start=alloc_idx, end=total_ops)
        return _Interval(start=min(alloc_idx, u[0]), end=max(u) + 1)
    region = lifetime.region
    if region is None:
        raise ValueError(f"Lifetime kind {kind.name} requires a region")
    span = region_span.get(id(region))
    if span is None:
        # Region not part of this function's body — fall back to KERNEL.
        return _Interval(start=0, end=total_ops)
    if kind == LifetimeKind.IN_REGION:
        return span
    if kind == LifetimeKind.BEFORE_REGION:
        return _Interval(start=0, end=span.start)
    if kind == LifetimeKind.AFTER_REGION:
        return _Interval(start=span.end, end=total_ops)
    raise ValueError(f"Unknown lifetime kind: {kind}")


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _size_bytes_for_alloc(op: SmemAllocOp) -> int:
    """Element count × dtype.bytes, accounting for ``pad`` on the row
    stride for 2D allocations. Mirrors the per-backend ``_emit_smem_allocs``
    sizing logic — both backends MUST agree on this number for the same op.
    """
    from quark.ir.types import DType

    dtype: DType = op.attrs["dtype"]
    shape = tuple(op.attrs["shape"])
    pad = int(op.attrs.get("pad", 0))
    if len(shape) == 2:
        row_stride = shape[1] + pad
        elems = shape[0] * row_stride
    else:
        elems = 1
        for s in shape:
            elems *= s
    return elems * dtype.bytes


def _align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


# ---------------------------------------------------------------------------
# Coloring
# ---------------------------------------------------------------------------


@dataclass
class _Slot:
    """One physical storage slot in the smem pool. May host multiple
    SmemAllocOps if they have non-overlapping intervals."""

    members: list[tuple[int, _Interval, int]] = field(default_factory=list)
    """List of (alloc_value_id, interval, size_bytes) currently assigned."""

    align: int = 16
    """Slot's required alignment (max of any member's align_bytes)."""

    def fits(self, interval: _Interval) -> bool:
        return all(not iv.overlaps(interval) for _, iv, _ in self.members)

    @property
    def size(self) -> int:
        return max((sz for _, _, sz in self.members), default=0)


def compute_smem_layout(
    fn: Function,
    *,
    enable_aliasing: bool = True,
) -> SmemLayoutPlan:
    """Run the smem layout analysis on a Function. Returns a plan the
    backend lowerers consume.

    With ``enable_aliasing=False`` every region gets its own slot
    (passthrough mode — used for the initial bring-up to verify the
    pass doesn't change behavior).
    """
    op_index, region_span, total_ops = _flat_op_indices(fn)
    uses = _value_use_indices(fn, op_index)

    # Gather (alloc_op, interval, size_bytes, align) tuples in op-index order.
    entries: list[tuple[SmemAllocOp, _Interval, int, int]] = []
    for op in fn.smem_allocs:
        if id(op) not in op_index:
            # Defensive: alloc op not yet attached — skip.
            continue
        interval = _interval_for_alloc(
            op,
            op_index=op_index,
            region_span=region_span,
            uses=uses,
            total_ops=total_ops,
        )
        size = _size_bytes_for_alloc(op)
        align = int(op.attrs.get("align_bytes", 16))
        entries.append((op, interval, size, align))

    # Sort by interval start (then by size descending for stability + tighter
    # packing — large early regions create slots big enough to absorb later
    # smaller co-residents).
    entries.sort(key=lambda e: (e[1].start, -e[2]))

    slots: list[_Slot] = []
    region_to_slot: dict[int, int] = {}
    for op, interval, size, align in entries:
        backing_id = op.results[0].id
        slot_idx = -1
        if enable_aliasing:
            for i, slot in enumerate(slots):
                if slot.fits(interval):
                    slot_idx = i
                    break
        if slot_idx == -1:
            slot_idx = len(slots)
            slots.append(_Slot())
        slots[slot_idx].members.append((backing_id, interval, size))
        slots[slot_idx].align = max(slots[slot_idx].align, align)
        region_to_slot[backing_id] = slot_idx

    # Assign offsets (sequential, aligned per slot).
    slot_offsets: dict[int, int] = {}
    slot_sizes: dict[int, int] = {}
    cursor = 0
    for i, slot in enumerate(slots):
        cursor = _align_up(cursor, slot.align)
        slot_offsets[i] = cursor
        slot_sizes[i] = slot.size
        cursor += slot.size

    return SmemLayoutPlan(
        region_to_slot=region_to_slot,
        slot_offsets=slot_offsets,
        slot_sizes=slot_sizes,
        total_bytes=cursor,
    )


def iter_aliased_pairs(plan: SmemLayoutPlan) -> Iterator[tuple[int, int]]:
    """Yield (alloc_id_a, alloc_id_b) for every pair of allocs that
    share a slot. Used by the validator to check for missing barriers
    between aliased writers/readers."""
    by_slot: dict[int, list[int]] = {}
    for alloc_id, slot in plan.region_to_slot.items():
        by_slot.setdefault(slot, []).append(alloc_id)
    for ids in by_slot.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                yield ids[i], ids[j]


def dump_smem_layout(fn: Function, plan: SmemLayoutPlan, *, label: str = "") -> str:
    """Render a human-readable summary of the smem layout for one
    function. Use this to diagnose aliasing surprises (e.g. why two
    regions with overlapping uses ended up sharing a slot, or why a
    config that should fit doesn't).

    Reports per-region: name, declared shape, dtype, padding, lifetime
    interval (op-index half-open), size in bytes, and the slot it was
    placed in. Then per-slot: members + slot size + slot offset, so
    aliasing is visible at a glance.
    """
    op_index, region_span, total_ops = _flat_op_indices(fn)
    uses = _value_use_indices(fn, op_index)

    # Build a {alloc_value_id: SmemAllocOp} map for lookup.
    alloc_by_id: dict[int, SmemAllocOp] = {}
    for op in fn.smem_allocs:
        alloc_by_id[op.results[0].id] = op

    # Per-region info.
    rows: list[str] = []
    rows.append(f"=== smem layout {label} (fn @{fn.name}) ===")
    rows.append(f"  total_ops={total_ops}  total_bytes={plan.total_bytes}")
    rows.append("")
    rows.append("  regions:")
    for op in fn.smem_allocs:
        backing_id = op.results[0].id
        if backing_id not in plan.region_to_slot:
            continue
        slot = plan.region_to_slot[backing_id]
        try:
            interval = _interval_for_alloc(
                op,
                op_index=op_index,
                region_span=region_span,
                uses=uses,
                total_ops=total_ops,
            )
        except Exception:
            interval = _Interval(start=-1, end=-1)
        size = _size_bytes_for_alloc(op)
        name = op.attrs.get("name", "?")
        shape = tuple(op.attrs.get("shape", ()))
        dtype = op.attrs.get("dtype")
        pad = int(op.attrs.get("pad", 0))
        lifetime = op.attrs.get("lifetime")
        lt_kind = lifetime.kind.name if lifetime is not None else "?"
        rows.append(
            f"    {name:24s} shape={shape!s:14s} dtype={dtype!s:6s} pad={pad:<3d} "
            f"lt={lt_kind:14s} live=[{interval.start},{interval.end})  "
            f"size={size}B  slot=#{slot}"
        )

    rows.append("")
    rows.append("  slots:")
    by_slot: dict[int, list[int]] = {}
    for alloc_id, slot in plan.region_to_slot.items():
        by_slot.setdefault(slot, []).append(alloc_id)
    for slot_idx in sorted(by_slot):
        members = by_slot[slot_idx]
        member_names = []
        for aid in members:
            op = alloc_by_id.get(aid)
            n = op.attrs.get("name", "?") if op is not None else "?"
            member_names.append(n)
        offset = plan.slot_offsets.get(slot_idx, -1)
        size = plan.slot_sizes.get(slot_idx, -1)
        flag = " ALIASED" if len(members) > 1 else ""
        rows.append(
            f"    #{slot_idx:<3d} offset={offset:<6d} size={size:<6d}B{flag}  "
            f"members=[{', '.join(member_names)}]"
        )
    rows.append("")
    return "\n".join(rows)


__all__ = (
    "SmemLayoutPlan",
    "compute_smem_layout",
    "dump_smem_layout",
    "iter_aliased_pairs",
)
