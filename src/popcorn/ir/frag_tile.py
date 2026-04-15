"""Register-level fragment tile abstraction.

A ``RegisterTile`` is a Python-level wrapper around an IR ``Value`` that
represents a per-warp fragment of a matrix tile. It carries the extra
metadata needed to express backend-agnostic fragment operations:

  * which MMA shape it belongs to (``shape_id``),
  * which operand role the fragment plays (``layout`` — A, B, or C/D),
  * which dtype each element is.

The abstraction exists because PTX and MSL disagree on how per-lane
fragment storage maps to matrix positions:

  * **PTX** uses the lane layout from PTX ISA §9.7.14.5: for m16n8
    accumulators, lane ``L`` holds four f32 regs at
    ``(gid + dr, tig*2 + dc)`` where ``gid = L/4``, ``tig = L%4``,
    and ``(dr, dc)`` comes from the shape's ``cd_offsets``.
  * **MSL** uses Apple's ``simdgroup_matrix<T, 8, 8>`` layout, a
    2×2×2 bit-swizzled mapping verified in
    ``tests/lower/msl/test_frag_layout.py``:
        row  = ((L >> 4) & 1) * 4 + ((L >> 1) & 3)
        col0 = ((L >> 3) & 1) * 4 + (L & 1) * 2

Callers work in logical (row, col) coordinates. The per-backend lane
map translates to the carrier's per-lane storage. Operations
(``map``, ``reduce_along_cols``, ``convert``) lower to the right
per-lane shuffle / arith pattern for each backend.

The ``RegisterTile`` doesn't introduce a new IR Value type — it
wraps an existing ``Value`` and the IR ops it emits carry the
layout metadata on their attrs. That keeps the IR type system
unchanged while giving callers a higher-level surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builder import Builder
    from .types import DType
    from .value import Value


class FragLayout(Enum):
    """Operand role a fragment fills in an MMA.

    Each role has a distinct per-lane storage layout on every
    backend (see ``LANE_MAPS`` below). ``FragLayout`` + ``shape_id``
    + ``backend`` is enough to resolve the full mapping from a
    per-lane register index to a matrix (row, col).
    """

    #: C/D operand — the accumulator. f32 carrier on PTX/MSL.
    #: On PTX: ``c_regs`` f32 regs/lane at (gid+dr, tig*2+dc).
    #: On MSL: simdgroup_matrix<f32, 8, 8>[mf*nf], 2 elements/lane/tile.
    ACCUMULATOR = "acc"

    #: A operand — matrix-A fragment. Packed bf16×2 (or fp8×4) b32
    #: carrier on PTX; simdgroup_matrix<bf16, 8, 8>[mf*kf] on MSL.
    A_FRAG = "a_frag"

    #: B operand — matrix-B fragment. Less common on the user-facing
    #: surface since ``load_matrix`` produces these directly; included
    #: for future code that needs to manipulate them explicitly.
    B_FRAG = "b_frag"


# ---------------------------------------------------------------------------
# Lane-map tables
#
# For each (backend, layout) pair we provide two functions:
#   lane_to_positions(lane, reg_idx, shape) → tuple[(row, col), ...]
#     Given a lane index and a per-lane register index (or tile index
#     on MSL), returns the (row, col) positions that per-lane slot
#     holds. May return 1 position (one scalar per reg on PTX acc) or
#     2 positions (bf16×2 per reg on PTX a_frag; two elements per tile
#     on MSL).
#
#   positions_to_lane_slot(row, col, shape) → (lane, reg_idx, elem_idx)
#     Inverse: given a matrix position, returns which lane holds it
#     in which per-lane slot.
#
# The ``shape`` argument is the ``MmaShape`` dataclass (providing
# cd_offsets / a_offsets / msl tiling).
#
# These tables are the ONE place per-backend layout knowledge lives.
# Any fragment op's MSL / PTX lowering consults them.
# ---------------------------------------------------------------------------


def apple_lane_to_tile_position(lane: int) -> tuple[int, int, int, int]:
    """Apple ``simdgroup_matrix<T, 8, 8>`` per-lane mapping.

    Returns ``(row0, col0, row1, col1)`` — the two matrix positions
    lane's ``thread_elements()[0]`` and ``[1]`` map to. Verified
    empirically in ``test_frag_layout.py``.
    """
    row = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3)
    col0 = ((lane >> 3) & 1) * 4 + (lane & 1) * 2
    return row, col0, row, col0 + 1


def apple_position_to_lane_slot(row: int, col: int) -> tuple[int, int]:
    """Inverse of ``apple_lane_to_tile_position``. Returns
    ``(lane, element_index)`` — which Apple lane holds (row, col)
    in which ``thread_elements()`` slot (0 or 1)."""
    assert 0 <= row < 8 and 0 <= col < 8, f"({row}, {col}) outside 8×8 tile"
    # Decompose row and col into the bit fields the forward formula uses.
    row_hi = (row >> 2) & 1  # → lane bit 4
    row_lo = row & 3  # → lane bits 1, 2
    col_hi = (col >> 2) & 1  # → lane bit 3
    col_pair = (col >> 1) & 1  # → lane bit 0
    lane = (row_hi << 4) | (col_hi << 3) | ((row_lo >> 1) << 2) | ((row_lo & 1) << 1) | col_pair
    elem = col & 1
    return lane, elem


def ptx_lane_to_acc_position(lane: int, reg_idx: int, cd_offsets: tuple) -> tuple[int, int]:
    """PTX C/D accumulator per-lane mapping. Returns ``(row, col)``
    for the element lane's ``reg_idx``-th c_reg holds."""
    gid = lane >> 2
    tig = lane & 3
    dr, dc = cd_offsets[reg_idx]
    return gid + dr, tig * 2 + dc


# Butterfly shuffle patterns for cross-lane reductions on accumulators.
#
# For "reduce along cols" (axis="row" → produce per-row scalars): we need
# to combine values across the lanes that share a row. The pattern is the
# XOR distances used to fold a group of N lanes pairwise in log2(N) steps.
#
# PTX m16n8 acc: 4 lanes per row (tig=0..3 within a group of 4). Butterfly
# pattern: [1, 2] — pair up ((0,1),(2,3)), then ((0,2),(1,3)).
#
# Apple 8x8 acc: 4 lanes per row, indexed by the 2 free bits (bit 0 and
# bit 3) that DON'T appear in the row formula. Butterfly pattern:
# [1, 8] — XOR bit 0 first, then bit 3. Matches the 2x2 col-block layout
# (bit 0 flips col_pair inside a col-block of 4; bit 3 flips col-block).
PTX_ACC_ROW_REDUCE_BUTTERFLY: tuple[int, ...] = (1, 2)
APPLE_ACC_ROW_REDUCE_BUTTERFLY: tuple[int, ...] = (1, 8)


def ptx_lane_to_a_frag_positions(
    lane: int, reg_idx: int, a_offsets: tuple, elements_per_reg: int = 2
) -> tuple[tuple[int, int], ...]:
    """PTX A-fragment per-lane mapping. Each b32 reg packs
    ``elements_per_reg`` elements (bf16×2 → 2, fp8×4 → 4). Returns
    the (row, col) position of EACH element inside the reg."""
    gid = lane >> 2
    tig = lane & 3
    dr, dc = a_offsets[reg_idx]
    row = gid + dr
    col_base = tig * elements_per_reg + dc
    return tuple((row, col_base + k) for k in range(elements_per_reg))


# ---------------------------------------------------------------------------
# RegisterTile — the user-facing wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisterTile:
    """A per-warp fragment tile with logical layout metadata.

    Generalises over multi-tile register regions (ThunderKittens'
    ``rt<...>`` pattern). A RegisterTile holds one IR Value per MMA
    tile it spans — for a single ``m16n8`` accumulator that's one
    Value; for a ``BM=32, BN=16`` accumulator grid that's four
    (2 m-tiles × 2 n-tiles). The ``tile_grid`` attribute records
    the (tiles_m, tiles_n) shape so operations like ``.map()`` and
    ``.reduce_along_cols()`` can iterate the right dimension.

    Callers treat this as an opaque handle; the builder methods
    (``.map()``, ``.reduce_along_cols()``, ``.convert()``) hide the
    per-backend details. Access the underlying IR Values via
    ``.values`` (tuple) or ``.value`` (for single-tile convenience).

    Attributes:
      values:        tuple of IR Values, one per MMA tile, in row-major
                     tile order: values[mt * tile_grid[1] + nt] is the
                     (mt, nt) tile. Carrier dtype depends on layout ×
                     backend (see FragLayout).
      layout:        ACCUMULATOR / A_FRAG / B_FRAG.
      shape_id:      MMA shape name each individual tile belongs to
                     (e.g. "m16n8k16_bf16"). The tile_grid scales this.
      tile_grid:     (tiles_m, tiles_n) — number of MMA tiles along
                     each dimension. Product must equal len(values).
      logical_shape: (rows, cols) — derived from tile_grid × MMA tile.
                     For a single m16n8 ACCUMULATOR tile this is
                     (16, 8). For a 2×2 grid: (32, 16).
    """

    values: tuple[Value, ...]
    layout: FragLayout
    shape_id: str
    tile_grid: tuple[int, int]
    logical_shape: tuple[int, int]
    # Optional builder reference — attached when the tile flows through
    # a Builder-aware method (e.g. ``bld.register_tile_from_frag``). Lets
    # fluent methods like ``.map()`` emit IR without the caller re-passing
    # the builder. Tests and low-level constructors may leave this None
    # and use the Builder-level API instead.
    _builder: Builder | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        m, n = self.tile_grid
        if m * n != len(self.values):
            raise ValueError(
                f"RegisterTile: tile_grid {self.tile_grid} implies "
                f"{m * n} values, got {len(self.values)}"
            )

    @property
    def rows(self) -> int:
        return self.logical_shape[0]

    @property
    def cols(self) -> int:
        return self.logical_shape[1]

    @property
    def dtype(self) -> DType:
        return self.values[0].dtype

    @property
    def value(self) -> Value:
        """Convenience accessor for single-tile RegisterTiles. Raises
        if the tile spans multiple MMA tiles — callers that handle
        the grid explicitly should use ``.values``."""
        if len(self.values) != 1:
            raise ValueError(
                f"RegisterTile.value: this tile is multi-tile "
                f"(tile_grid={self.tile_grid}); use .values"
            )
        return self.values[0]

    def tile_at(self, mt: int, nt: int) -> Value:
        """Return the Value for the (mt, nt) MMA tile."""
        m, n = self.tile_grid
        if not (0 <= mt < m and 0 <= nt < n):
            raise IndexError(f"RegisterTile.tile_at({mt}, {nt}) outside grid {self.tile_grid}")
        return self.values[mt * n + nt]

    def with_builder(self, builder: Builder) -> RegisterTile:
        """Return a RegisterTile bound to a Builder so fluent methods
        (``.map()`` etc.) can emit IR without the caller re-passing it.
        Useful when a tile is constructed by low-level machinery that
        doesn't have the builder (e.g. IR printer tests) and later needs
        fluent-API methods in an IR-building context.
        """
        return replace(self, _builder=builder)

    def map(
        self,
        fn: Callable[[Value], Value],
        *,
        builder: Builder | None = None,
    ) -> RegisterTile:
        """Apply ``fn`` element-wise to every logical (row, col) in the
        tile. ``fn`` takes and returns a scalar IR Value of the tile's
        dtype — the transformation is inlined per storage slot at lower
        time (per c_reg on PTX, per thread_elements slot on MSL).

        Only valid for ``ACCUMULATOR`` tiles today; ``A_FRAG`` / ``B_FRAG``
        carry packed storage and will be handled through
        ``FragConvertOp`` (proposal §1.6 Step 3).

        If ``builder`` is not passed, falls back to ``self._builder`` (set
        via ``.with_builder(b)`` or by Builder-aware constructors).
        """
        b = builder if builder is not None else self._builder
        if b is None:
            raise RuntimeError(
                "RegisterTile.map: no Builder reference — either call "
                ".with_builder(b) first or pass builder=b explicitly"
            )
        if self.layout is not FragLayout.ACCUMULATOR:
            raise NotImplementedError(
                f"RegisterTile.map: only ACCUMULATOR tiles supported "
                f"today (got {self.layout}). Use FragConvertOp for "
                f"A/B-frag transformations."
            )
        new_values = tuple(b.frag_apply(self.shape_id, v, fn) for v in self.values)
        return replace(self, values=new_values)

    def for_each(
        self,
        fn: Callable[[Value, Value, Value], None],
        cd_offsets: tuple[tuple[int, int], ...],
        *,
        builder: Builder | None = None,
    ) -> None:
        """Apply side-effect ``fn(elem, row, col)`` to every storage slot.

        ``row`` and ``col`` are tile-local U32 IR Values lane-bound at
        lower time (PTX: ``gid + dr``, MSL: Apple formula + tile offset).
        ``fn`` emits stores or atomics using ``(elem, row, col)`` and
        returns None. No output fragment.

        Typical epilogue:

            tile.for_each(
                lambda elem, r, c: b.store(
                    gmem, elem,
                    b.add(row_base, r),
                    b.add(col_base, c),
                ),
                cd_offsets=cd_offsets,
            )
        """
        b = builder if builder is not None else self._builder
        if b is None:
            raise RuntimeError("RegisterTile.for_each: no Builder reference")
        if self.layout is not FragLayout.ACCUMULATOR:
            raise NotImplementedError("for_each: only ACCUMULATOR tiles supported today")
        for v in self.values:
            b.frag_for_each(self.shape_id, v, fn, cd_offsets)

    def reduce_along_cols(
        self,
        kind: str,
        cd_offsets: tuple[tuple[int, int], ...],
        *,
        builder: Builder | None = None,
    ) -> tuple[Value, ...]:
        """Reduce across columns (axis='row'), returning one scalar per
        row class. For a single-tile m16n8 bf16 accumulator with 2 row
        classes, returns a 2-tuple of scalar Values — each replicated
        across every lane that shares the row.

        ``kind`` ∈ ``{"max", "min", "add", "mul"}``.

        For multi-tile ``tile_grid=(MT, NT)`` with ``NT > 1`` the nt
        tiles sharing a row are first folded element-wise (same per-lane
        slot layout across n-tiles, so no cross-lane shuffle is needed);
        then one ``frag_reduce`` runs on the folded tile per m-tile row.
        The returned tuple is flat in row-major tile order:
        ``(tile_m0_rc0, tile_m0_rc1, tile_m1_rc0, tile_m1_rc1, …)``.
        """
        b = builder if builder is not None else self._builder
        if b is None:
            raise RuntimeError("RegisterTile.reduce_along_cols: no Builder reference")
        if self.layout is not FragLayout.ACCUMULATOR:
            raise NotImplementedError("reduce_along_cols: only ACCUMULATOR tiles supported")
        mt, nt = self.tile_grid
        flat: list[Value] = []
        for m in range(mt):
            folded = self.values[m * nt]
            for n in range(1, nt):
                folded = _combine_acc_frags(b, folded, self.values[m * nt + n], kind)
            results = b.frag_reduce(
                self.shape_id,
                folded,
                kind=kind,
                axis="row",
                cd_offsets=cd_offsets,
            )
            flat.extend(results)
        return tuple(flat)

    def map_per_row_class(
        self,
        scales,
        fn: Callable[[Value, Value], Value],
        cd_offsets: tuple[tuple[int, int], ...],
        *,
        builder: Builder | None = None,
    ) -> RegisterTile:
        """Apply ``fn(elem, rc_scale)`` per element, with ``rc_scale``
        bound per slot to the row-class scale.

        For m16n8 bf16 cd_offsets, there are two row classes (dr ∈ {0, 8}).
        Passing ``scales=[rescale_rc0, rescale_rc1]`` with
        ``fn=lambda x,s: x*s`` is the O-rescale pattern: a pure
        ``FragApplyOp`` with per-slot selectors bound to row-class scales.
        """
        b = builder if builder is not None else self._builder
        if b is None:
            raise RuntimeError("RegisterTile.map_per_row_class: no Builder reference")
        if self.layout is not FragLayout.ACCUMULATOR:
            raise NotImplementedError(
                f"RegisterTile.map_per_row_class: only ACCUMULATOR tiles "
                f"supported (got {self.layout})"
            )
        dr_vals = sorted({dr for dr, _ in cd_offsets})
        scales_tup = tuple(scales)
        if len(scales_tup) != len(dr_vals):
            raise ValueError(
                f"map_per_row_class: expected {len(dr_vals)} scales (one per "
                f"row class from cd_offsets), got {len(scales_tup)}"
            )
        slot_to_selector_idx = tuple(dr_vals.index(dr) for dr, _ in cd_offsets)
        new_values = tuple(
            b.frag_apply(
                self.shape_id,
                v,
                fn,
                selectors=scales_tup,
                slot_to_selector_idx=slot_to_selector_idx,
            )
            for v in self.values
        )
        return replace(self, values=new_values)


def _combine_acc_frags(b: Builder, a: Value, c: Value, kind: str) -> Value:
    """Element-wise combine two same-shape accumulator fragments.

    Each accumulator fragment Value is a vec of ``c_regs`` f32 elements.
    All n-tiles in a grid share the same per-lane storage layout, so
    folding them across the N axis reduces to per-slot f32 scalar ops
    (max/min/add/mul) — no cross-lane shuffle required. The result is
    another same-shape vec ready for ``frag_reduce`` along the row axis.

    Used by :meth:`RegisterTile.reduce_along_cols` when ``tile_grid[1] > 1``.
    """
    assert a.shape == c.shape, f"_combine_acc_frags: shape mismatch {a.shape} vs {c.shape}"
    width = a.shape.width
    if kind == "max":
        op = b.max
    elif kind == "min":
        op = b.min
    elif kind == "add":
        op = b.add
    elif kind == "mul":
        op = b.mul
    else:
        raise ValueError(f"_combine_acc_frags: unsupported kind {kind!r}")
    slots_a = [b.vec_extract(a, i) for i in range(width)]
    slots_c = [b.vec_extract(c, i) for i in range(width)]
    return b.vec_build([op(sa, sc) for sa, sc in zip(slots_a, slots_c, strict=True)])


__all__ = (
    "FragLayout",
    "RegisterTile",
    "apple_lane_to_tile_position",
    "apple_position_to_lane_slot",
    "ptx_lane_to_a_frag_positions",
    "ptx_lane_to_acc_position",
)
