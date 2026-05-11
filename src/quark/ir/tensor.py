"""Tensor types: GlobalTensor, SharedRegion, FragTensor.

EXEMPT FROM 500-LINE RULE: these three types plus their subscript /
view / copy_from / stage / warp_lane_view helpers ARE the tensor
API. Splitting by tensor kind loses the shared index-materialization
and subscript-normalization helpers; splitting by helper category
duplicates the per-kind boilerplate.

Three distinct concepts that already exist in the codebase (and that
Metal/OpenCL force on us). Each has its own allocation story, lifetime,
and lowering path — we do NOT unify them into a single `MemRef` with a
tag, because every lowerer would then branch on the tag and re-derive
the information.

See QUARK_IR_PROPOSAL.md §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

from .lifetime import Lifetime
from .types import DType

if TYPE_CHECKING:
    from .module import Param
    from .value import Value


# ---------------------------------------------------------------------------
# Base tensor protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tensor:
    """Common base: a typed, shaped view over some storage backing.

    Strides are in elements, not bytes. Byte arithmetic is strictly a
    lowering concern — the IR never sees it.
    """

    dtype: DType
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.stride):
            raise ValueError(
                f"Tensor shape {self.shape} and stride {self.stride} must have the same rank"
            )
        for s in self.shape:
            if s < 0:
                raise ValueError(f"Tensor shape entries must be non-negative: {self.shape}")


# ---------------------------------------------------------------------------
# Subscript normalization for the Pythonic ``__getitem__`` / ``__setitem__``
# sugar on tensor types. A subscript is one of:
#   tensor[r, c]              → scalar load/store at (r, c)
#   tensor[r, c:c+W]          → vec load/store of width W along col axis
#   tensor[r, c:c+W:1]        → ditto, explicit step (must be 1)
# Only the last axis may be a slice — strided slicing on more than one
# axis isn't a meaningful pattern for a tile load.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubscriptRange:
    """Resolved subscript: list of base indices for each axis, plus an
    optional vec width / dtype if the last axis was a slice."""

    bases: tuple
    is_vec: bool
    width: int = 1
    vec_dtype: Optional[DType] = None


def _materialize_index_bases(bld, bases: tuple) -> tuple:
    """Convert any Python-int bases in a subscript to ``U32`` ConstOp
    Values via ``bld.const`` so the underlying ``LoadOp`` / ``StoreOp``
    sees a uniform tuple of Values.
    """
    from .value import Value as _Value

    out = []
    for b in bases:
        if isinstance(b, int):
            out.append(bld.const(DType.U32, b))
        elif isinstance(b, _Value):
            out.append(b)
        else:
            raise TypeError(f"_materialize_index_bases: unexpected type {type(b).__name__}")
    return tuple(out)


def _normalize_subscript(key, rank: int) -> _SubscriptRange:
    """Resolve a Python subscript into ``(bases, is_vec, width, vec_dtype)``.

    Accepts an int / Value (rank-1 tensor) or a tuple of ints / Values /
    one trailing slice (rank-N tensor). The trailing slice triggers vec
    semantics: ``slice(start, start+W)`` ⇒ width=W vec load/store.
    """
    from .value import Value as _Value

    if not isinstance(key, tuple):
        key = (key,)
    if len(key) != rank:
        raise IndexError(f"tensor subscript: expected {rank} indices, got {len(key)}")
    is_vec = False
    width = 1
    bases_list: list = []
    for axis, k in enumerate(key):
        if isinstance(k, slice):
            if axis != rank - 1:
                raise IndexError(
                    f"tensor subscript: slice only allowed on the last axis (axis {axis} of {rank})"
                )
            if k.step is not None and k.step != 1:
                raise IndexError(f"tensor subscript: slice step must be 1 (got {k.step})")
            start = k.start if k.start is not None else 0
            stop = k.stop
            if stop is None:
                raise IndexError("tensor subscript: slice stop required for vec width")
            if isinstance(start, _Value) or isinstance(stop, _Value):
                raise IndexError(
                    "tensor subscript: vec slice bounds must be Python ints "
                    "(width must be statically known); use vec_load directly "
                    "if the base is a runtime Value"
                )
            width = stop - start
            if width < 1:
                raise IndexError(f"tensor subscript: vec slice width {width} < 1")
            is_vec = True
            bases_list.append(start)
        elif isinstance(k, (int, _Value)):
            bases_list.append(k)
        else:
            raise TypeError(f"tensor subscript axis {axis}: unsupported type {type(k).__name__}")
    return _SubscriptRange(
        bases=tuple(bases_list),
        is_vec=is_vec,
        width=width,
    )


# ---------------------------------------------------------------------------
# GlobalTensor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalTensor(Tensor):
    """A gmem tensor bound to a kernel parameter.

    The base pointer is the u64 kernel parameter `param` (resolved at
    lowering time). Row/col offsets accumulate through `view()` — the
    common pattern being `g_x.view(row=m_block * BM)` to carve out the
    block's rows without threading a u32 offset through every load.

    ``tile(row_base, col_base, shape)`` returns a sub-region typed for
    the unified tile-load primitive — what we call a "GMemTile" in the
    higher-level mental model. It's the source side of
    ``smem_tile.copy_from(gmem_tile)``.
    """

    name: str
    param: Param
    static_row_offset: int = 0
    static_col_offset: int = 0
    dyn_row_offset: Optional[Value] = None
    dyn_col_offset: Optional[Value] = None

    # ------------------------------------------------------------------
    # Pythonic access (sugar over ``bld.load`` / ``bld.store`` /
    # ``bld.vec_load`` / ``bld.vec_store``).
    # ------------------------------------------------------------------
    def __getitem__(self, key) -> Value:
        from .value import _active_builder

        bld = _active_builder()
        rng = _normalize_subscript(key, self.rank)
        bases = _materialize_index_bases(bld, rng.bases)
        if rng.is_vec:
            return bld.vec_load(self, *bases, width=rng.width)
        return bld.load(self, *bases)

    def __setitem__(self, key, value) -> None:
        from .value import _active_builder

        bld = _active_builder()
        rng = _normalize_subscript(key, self.rank)
        bases = _materialize_index_bases(bld, rng.bases)
        if rng.is_vec:
            bld.vec_store(self, value, *bases)
            return
        bld.store(self, value, *bases)

    def tile(
        self,
        *,
        row: int | Value = 0,
        col: int | Value = 0,
        shape: tuple[int, int],
    ) -> GlobalTensor:
        """Carve out a typed 2D tile at ``(row, col)`` with the given
        shape. Convenience wrapper over ``view(row=, col=, shape=)``.

        Returns a GlobalTensor that the unified ``smem_tile.copy_from``
        / ``store_to`` primitives consume directly.
        """
        return self.view(row=row, col=col, shape=shape)

    def view(
        self,
        *,
        row: int | Value = 0,
        col: int | Value = 0,
        shape: tuple[int, ...] | None = None,
    ) -> GlobalTensor:
        """Carve out a sub-region with an added row/col offset.

        Integer offsets accumulate statically; Value offsets chain
        through the dyn_*_offset fields. Mixing is allowed — the static
        part stays on the new tensor's static offsets and the Value
        replaces any existing dyn offset (the caller is expected to
        fold previous dyn offsets via an ArithOp if needed).
        """
        from .value import Value as _Value

        new_static_row = self.static_row_offset
        new_static_col = self.static_col_offset
        new_dyn_row = self.dyn_row_offset
        new_dyn_col = self.dyn_col_offset

        if isinstance(row, int):
            new_static_row += row
        elif isinstance(row, _Value):
            if new_dyn_row is not None:
                raise ValueError(
                    "GlobalTensor.view: this tensor already has a dyn_row_offset; "
                    "fold it into an ArithOp before taking another dyn row view"
                )
            new_dyn_row = row
        else:
            raise TypeError(
                f"GlobalTensor.view: row must be int or Value, got {type(row).__name__}"
            )

        if isinstance(col, int):
            new_static_col += col
        elif isinstance(col, _Value):
            if new_dyn_col is not None:
                raise ValueError(
                    "GlobalTensor.view: this tensor already has a dyn_col_offset; "
                    "fold it into an ArithOp before taking another dyn col view"
                )
            new_dyn_col = col
        else:
            raise TypeError(
                f"GlobalTensor.view: col must be int or Value, got {type(col).__name__}"
            )

        return replace(
            self,
            shape=shape if shape is not None else self.shape,
            static_row_offset=new_static_row,
            static_col_offset=new_static_col,
            dyn_row_offset=new_dyn_row,
            dyn_col_offset=new_dyn_col,
        )


# ---------------------------------------------------------------------------
# SharedRegion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedRegion(Tensor):
    """A typed, block-scoped smem region backed by a SmemAlloc.

    Multiple SharedRegions can share one SmemAlloc backing — for
    double-buffered pipelines (each stage is a view) AND for
    lifetime-disjoint regions the smem layout pass auto-aliases.

    Attributes beyond the base tensor:
      * ``alloc``: the SmemAllocOp result this region is backed by.
      * ``static_offset`` / ``dyn_offset``: byte-equivalent element
        offsets into the alloc (for views & pipeline stages).
      * ``pad``: per-row bank-conflict padding folded into the stride.
      * ``align_bytes``: minimum guaranteed alignment of the region's
        base. The smem_layout pass enforces this when assigning offsets.
      * ``lifetime``: when this region is live. Default
        ``Lifetime.auto()`` lets the layout pass infer from
        earliest/latest use. ``Lifetime.in_region(R)`` pins lifetime to
        a specific control-flow region — see ``ir/lifetime.py``.
      * ``readonly_after_init``: marks regions written once at kernel
        entry and only read thereafter; lets the layout pass elide some
        post-init barriers across aliasing boundaries.
      * ``warp_dyn_offset``: MSL-only — a uniform per-simdgroup offset
        used by ``simdgroup_load`` (which needs a SIMD-uniform base).
        PTX folds this into ``dyn_offset``.
    """

    name: str
    alloc: Value  # result of a SmemAllocOp
    static_offset: int = 0
    dyn_offset: Optional[Value] = None
    pad: int = 0
    align_bytes: int = 16
    lifetime: Lifetime = field(default_factory=Lifetime.auto)
    readonly_after_init: bool = False
    warp_dyn_offset: Optional[Value] = None

    def view(
        self,
        *,
        static_offset_add: int = 0,
        dyn_offset: Optional[Value] = None,
        shape: tuple[int, ...] | None = None,
        name: str | None = None,
    ) -> SharedRegion:
        """Carve out a sub-region — e.g. one pipeline stage.

        Passing a `dyn_offset` replaces any existing one (callers fold
        the old+new into an ArithOp first if needed). Static offsets
        always accumulate. Layout attrs (``align_bytes``, ``lifetime``,
        ``readonly_after_init``) inherit from the parent — a view shares
        the underlying region's storage and lifetime.
        """
        if dyn_offset is None:
            new_dyn = self.dyn_offset
        else:
            if self.dyn_offset is not None:
                raise ValueError(
                    "SharedRegion.view: this tensor already has a dyn_offset; "
                    "fold it into an ArithOp before taking another dyn view"
                )
            new_dyn = dyn_offset

        return replace(
            self,
            name=name if name is not None else self.name,
            shape=shape if shape is not None else self.shape,
            static_offset=self.static_offset + static_offset_add,
            dyn_offset=new_dyn,
        )

    # ------------------------------------------------------------------
    # 3D ↔ 2D — pipeline staging
    # ------------------------------------------------------------------
    def stage(self, idx: int | Value) -> SharedRegion:
        """Slice a 3D region ``(n_stages, rows, cols)`` to a 2D region
        at stage ``idx``. The 2D view shares storage with the parent;
        ``idx`` is the stage selector and may be either a Python int
        (folds into ``static_offset``) or a runtime ``Value`` (chains
        through ``dyn_offset``).

        Raises if the region isn't 3D — pipeline staging is the only
        legal interpretation of the leading dim today.
        """
        from .value import Value as _Value

        if self.rank != 3:
            raise ValueError(
                f"SharedRegion.stage: only 3D regions support .stage() "
                f"(this region has shape {self.shape}, rank {self.rank}). "
                "Allocate via ``bld.smem_region(name, dt, (n_stages, rows, cols))``."
            )
        n_stages, rows, cols = self.shape
        # 2D view: drop the stage dim. Stride flattens to the trailing
        # row-major (rows + pad, 1) — the same contract the existing
        # 2D loaders consume.
        new_shape = (rows, cols)
        new_stride = (cols + self.pad, 1)
        if isinstance(idx, int):
            stage_elems = rows * (cols + self.pad)
            return replace(
                self,
                shape=new_shape,
                stride=new_stride,
                static_offset=self.static_offset + idx * stage_elems,
            )
        if isinstance(idx, _Value):
            # Multiply by per-stage element count and chain through dyn_offset.
            from .value import _active_builder

            bld = _active_builder()
            stage_elems_const = bld.const(idx.dtype, rows * (cols + self.pad))
            stage_off = bld.mul(idx, stage_elems_const)
            if self.dyn_offset is not None:
                stage_off = bld.add(self.dyn_offset, stage_off)
            return replace(
                self,
                shape=new_shape,
                stride=new_stride,
                dyn_offset=stage_off,
            )
        raise TypeError(f"SharedRegion.stage: idx must be int or Value, got {type(idx).__name__}")

    def lane_view(
        self,
        *,
        lane_col_step: int = 2,
    ) -> SharedRegion:
        """Per-lane view for MMA fragment loads (PTX ldmatrix / MSL
        simdgroup_load). Chains ``groupID * row_stride + tidIG * step``
        into ``dyn_offset`` — the per-lane PTX convention — leaving the
        existing view's other offsets intact.

        On MSL, ``simdgroup_load`` ignores ``dyn_offset`` and reads
        ``warp_dyn_offset`` instead — see ``warp_lane_view`` for the
        combined warp+lane case that populates both.
        """
        if self.rank != 2:
            raise ValueError(f"SharedRegion.lane_view: requires 2D region, got shape {self.shape}")
        from .value import _active_builder

        bld = _active_builder()
        gid = bld.group_id()
        tig = bld.thread_id_in_group()
        per_lane = gid * self.stride[0] + tig * lane_col_step
        # Chain through any existing dyn_offset.
        new_dyn = per_lane if self.dyn_offset is None else self.dyn_offset + per_lane
        return replace(self, dyn_offset=new_dyn)

    def warp_lane_view(
        self,
        rows: int,
        *,
        warp_id: Value,
        lane_col_step: int | None = None,
        lane_offset: Value | None = None,
    ) -> SharedRegion:
        """Combined per-warp + per-lane view. Sets BOTH offsets in one
        pass — the canonical way to address a per-warp MMA fragment load
        on either backend:

          * ``dyn_offset``     = warp_off + <per-lane offset>
                                 (PTX ldmatrix per-lane address)
          * ``warp_dyn_offset`` = warp_off
                                 (MSL simdgroup_load SIMD-uniform base)

        The per-lane component comes from ONE of:
          * ``lane_col_step``: standard ``groupID * row_stride + tidIG * step``
            used by the default / unshuffled B-fragment layout.
          * ``lane_offset``: caller-supplied explicit Value for custom
            layouts (e.g. B-shuffled: ``lane_id * frag_elems``).

        Exactly one of ``lane_col_step`` or ``lane_offset`` must be given.
        Replaces the old ``_attach_warp_b_lane`` pattern (view + dataclasses.replace).
        """
        if self.rank != 2:
            raise ValueError(
                f"SharedRegion.warp_lane_view: requires 2D region, got shape {self.shape}"
            )
        if (lane_col_step is None) == (lane_offset is None):
            raise ValueError(
                "warp_lane_view: pass exactly one of ``lane_col_step`` or ``lane_offset``"
            )
        from .value import _active_builder

        bld = _active_builder()
        row_stride = self.stride[0]
        warp_off = warp_id * (rows * row_stride)
        if lane_offset is not None:
            per_lane = lane_offset
        else:
            gid = bld.group_id()
            tig = bld.thread_id_in_group()
            per_lane = gid * row_stride + tig * lane_col_step
        full = warp_off + per_lane
        # Chain through any existing dyn_offset for composability.
        new_dyn = full if self.dyn_offset is None else self.dyn_offset + full
        return replace(
            self,
            shape=(rows, self.shape[1]),
            dyn_offset=new_dyn,
            warp_dyn_offset=warp_off,
        )

    def warp_view(
        self,
        rows: int,
        *,
        warp_id: Value | None = None,
    ) -> SharedRegion:
        """Carve out a per-warp horizontal slice covering ``rows`` rows
        starting at ``warp_id * rows``.

        Sets ``dyn_offset`` (the per-lane address the load/store ops
        consume). For MSL ``simdgroup_load`` / ldmatrix-style
        cooperative loads that need a SIMD-uniform base, call
        ``.view(warp_dyn_offset=...)`` directly — those operations
        ignore ``dyn_offset`` on MSL and read ``warp_dyn_offset``
        instead.
        """
        if self.rank != 2:
            raise ValueError(
                f"SharedRegion.warp_view: needs a 2D region, got shape {self.shape}. "
                "Call .stage(i) first if this is a 3D pipeline region."
            )
        cols = self.shape[1]
        row_stride_elems = cols + self.pad

        from .value import _active_builder

        if warp_id is None:
            # No warp_id → return the full region with the requested rows.
            return replace(self, shape=(rows, cols))
        bld = _active_builder()
        warp_off = bld.mul(warp_id, bld.const(warp_id.dtype, rows * row_stride_elems))
        # Chain through any existing dyn_offset to preserve composability.
        if self.dyn_offset is not None:
            warp_off = bld.add(self.dyn_offset, warp_off)
        return replace(
            self,
            shape=(rows, cols),
            dyn_offset=warp_off,
        )

    # ------------------------------------------------------------------
    # Pythonic load/store sugar (sugar over bld.load/store/vec_*).
    # ------------------------------------------------------------------
    def __getitem__(self, key) -> Value:
        from .value import _active_builder

        bld = _active_builder()
        rng = _normalize_subscript(key, self.rank)
        bases = _materialize_index_bases(bld, rng.bases)
        if rng.is_vec:
            return bld.vec_load(self, *bases, width=rng.width)
        return bld.load(self, *bases)

    def __setitem__(self, key, value) -> None:
        from .value import _active_builder

        bld = _active_builder()
        rng = _normalize_subscript(key, self.rank)
        bases = _materialize_index_bases(bld, rng.bases)
        if rng.is_vec:
            bld.vec_store(self, value, *bases)
            return
        bld.store(self, value, *bases)

    # ------------------------------------------------------------------
    # Tile-load primitive — gmem→smem the high-level way.
    # ------------------------------------------------------------------
    def copy_from(
        self,
        gmem_tile: GlobalTensor,
        *,
        tid: Value,
        n_threads: int,
        async_load: bool = False,
        cast: DType | None = None,
        pred: Value | None = None,
    ) -> None:
        """Cooperatively copy a ``GlobalTensor`` tile into this SMemTile.

        ``rows`` and ``cols`` come from ``self.shape`` (must match the
        gmem tile's logical shape — B-shuffled destinations need the
        shuffled shape pre-computed on both sides). The per-thread
        split is ``rows * cols / n_threads`` contiguous elements; the
        caller is responsible for ensuring ``n_threads`` divides the
        tile size.

        ``async_load=True`` emits ``cp.async`` on PTX (per-line 16-byte
        transfers, requires 16-byte-aligned cols); scalar loads
        otherwise. ``cast`` forces the scalar path and performs the
        element conversion during the load. ``pred`` gates the whole
        copy.

        B-shuffled destinations: pass a SharedRegion whose ``shape`` /
        ``stride`` already reflect the permuted layout (the tensor
        declaration chooses ``(N, (K//BK)*(BK+pad))`` when shuffled).
        This primitive copies logical-layout contiguous elements; the
        shuffle / permutation happens at the declaration level, not
        here.
        """
        from quark.blocks.l0.tile_loader import emit_tile_load

        from .value import _active_builder

        bld = _active_builder()
        if self.rank != 2:
            raise ValueError(
                f"SharedRegion.copy_from: destination must be 2D "
                f"(got shape {self.shape}). Call .stage(i) first for a "
                f"3D pipeline region."
            )
        rows, cols = self.shape
        # gmem_tile's static_row_offset / dyn_row_offset already carry
        # the tile-base. Pass 0/0 as gmem_row_base/gmem_col_base; the
        # LoadOp picks up the view offsets on top of whatever row/col
        # the tile loader passes.
        zero = bld.const(DType.U32, 0)
        emit_tile_load(
            bld,
            dst_smem=self,
            src_gmem=gmem_tile,
            rows=rows,
            cols=cols,
            gmem_row_base=zero,
            gmem_col_base=zero,
            tid=tid,
            n_threads=n_threads,
            use_async=async_load,
            pred=pred,
            cast=cast,
        )


# ---------------------------------------------------------------------------
# FragTensor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FragTensor(Tensor):
    """An MMA fragment — a tile stored across warp lanes as per-thread regs.

    The layout of which lane holds which element is determined by
    `(shape_id, which)`; backends consult their Mma resolution table
    to turn that into concrete register counts and load/store patterns.
    `regs` holds the SSA values that carry the per-lane elements of
    the tile.
    """

    shape_id: str
    which: str  # "a" | "b" | "c" | "d"
    regs: tuple[Value, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.which not in ("a", "b", "c", "d"):
            raise ValueError(f"FragTensor.which must be one of a/b/c/d, got {self.which!r}")
