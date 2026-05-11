"""SmemTile — declarative smem allocation with auto-allocation and
bundled gmem→smem load primitives (``.load_from`` / ``.gather_from``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import quark.lang as qk
from quark.blocks.dsl.context import active_bctx
from quark.ir import Builder, DType, SharedRegion


class SmemTileSpec(NamedTuple):
    """Frozen factory spec for a :class:`SmemTile` — dtype + shape + pad
    + lane_col_step, no ``name``. Used by :meth:`Stage.staged` to
    template per-stage tile construction without eagerly allocating.
    """

    dtype: DType
    shape: tuple[int, ...]
    pad: int = 0
    lane_col_step: int = 2


@dataclass
class SmemTile:
    """Declarative smem tile spec.

    Describes a shared memory allocation with optional padding and
    per-lane view for MMA fragment loads. The underlying
    ``qk.smem_alloc`` happens automatically in ``__post_init__`` via
    the active builder — callers no longer need to invoke ``.emit(bld)``
    explicitly. For the uncommon case where a SmemTile needs to be
    described but not yet allocated (e.g. factory closures in
    ``Pipeline.smem_factory`` run lazily under a builder context),
    the class still works as long as *some* builder is active when
    the SmemTile is constructed.
    """

    name: str
    dtype: DType
    shape: tuple[int, ...]
    pad: int = 0
    lane_col_step: int = 2

    # Populated during __post_init__ (or by lifetime of the emit()
    # back-compat path). Callers access via .smem / .lane.
    _smem: SharedRegion | None = field(default=None, repr=False)
    _lane: SharedRegion | None = field(default=None, repr=False)

    @classmethod
    def spec(
        cls,
        dtype: DType,
        shape: tuple[int, ...],
        *,
        pad: int = 0,
        lane_col_step: int = 2,
    ) -> SmemTileSpec:
        """Return an un-allocated :class:`SmemTileSpec` for use with
        :meth:`Stage.staged`. The concrete SmemTile (and its smem
        allocation) is emitted once the stage factory hands the spec
        back with a per-stage name.
        """
        return SmemTileSpec(dtype=dtype, shape=shape, pad=pad, lane_col_step=lane_col_step)

    def __post_init__(self) -> None:
        """Auto-allocate via the active Builder if one is registered.
        Falls back to the lazy ``emit()`` path for call sites that
        construct SmemTile specs outside a builder context (e.g. a
        ``Pipeline.smem_factory`` lambda captured then called later).
        """
        from quark.ir.value import _ACTIVE_BUILDER

        bld = _ACTIVE_BUILDER.get(None)
        if bld is not None and self._smem is None:
            self._allocate(bld)

    def _allocate(self, bld: Builder) -> None:
        from quark.blocks.l0.smem_base import emit_smem_base

        self._smem = qk.smem_alloc(self.name, self.dtype, self.shape, pad=self.pad)
        self._lane = emit_smem_base(bld, self._smem, self.lane_col_step)

    @property
    def smem(self) -> SharedRegion:
        assert self._smem is not None, f"SmemTile {self.name!r} not emitted yet"
        return self._smem

    @smem.setter
    def smem(self, v: SharedRegion) -> None:
        self._smem = v

    @property
    def lane(self) -> SharedRegion:
        assert self._lane is not None, f"SmemTile {self.name!r} lane not emitted yet"
        return self._lane

    @lane.setter
    def lane(self, v: SharedRegion) -> None:
        self._lane = v

    def load_from(
        self,
        gmem: Any,
        *,
        row: Any = 0,
        col: Any = 0,
        cast: DType | None = None,
        use_async: bool = True,
    ) -> None:
        """Cooperative gmem→smem tile load into this SmemTile.

        The canonical load primitive: ``smem.load_from(gmem_tensor,
        row=, col=, cast=)``. Replaces the old L1 ``TileLoad`` Block.

        Shape is derived from ``self.shape`` — no need to pass ``rows`` /
        ``cols``. Pass either a ``GlobalTensor`` + ``row``/``col`` bases
        (the helper carves a ``gmem.tile(...)`` internally), or hand in
        a pre-carved ``gmem_tile`` as ``gmem`` with ``row=0, col=0``.

        ``use_async`` toggles cp.async; ``cast=`` converts elements on
        the way in (forces the scalar path — cp.async can't cast).
        """
        from quark.ir import GlobalTensor

        rows, cols = self.shape
        if isinstance(gmem, GlobalTensor):
            gmem_tile = gmem.tile(row=row, col=col, shape=(rows, cols))
        else:
            # Caller passed a pre-carved tile; ignore row/col if zero.
            gmem_tile = gmem

        bctx = active_bctx()
        # cp.async can't cast; force scalar when a dtype change is asked.
        is_async = use_async and cast is None
        self.smem.copy_from(
            gmem_tile,
            tid=bctx.tid,
            n_threads=bctx.n_threads,
            async_load=is_async,
            cast=cast,
        )

    def gather_from(
        self,
        gmem: Any,
        *,
        index: Any,
        col: Any = 0,
        cast: DType | None = None,
        use_async: bool = True,
    ) -> None:
        """Gmem→smem tile load where each row's gmem address comes from
        ``index`` (an ``IndexCache`` or raw smem region).

        Replaces the old ``GatheredTileLoad`` L1 Block. Shape is derived
        from ``self.shape``; ``col`` is the gmem column base (row addresses
        come from the index).
        """
        from quark.blocks.l0.gathered_tile_loader import emit_gathered_tile_load
        from quark.ir import Value

        rows, cols = self.shape
        index_smem = index.smem if hasattr(index, "smem") else index
        bctx = active_bctx()
        col_base = col if isinstance(col, Value) else bctx.c(col)
        emit_gathered_tile_load(
            bctx.bld,
            dst_smem=self.smem,
            src_gmem=gmem,
            index_smem=index_smem,
            rows=rows,
            cols=cols,
            gmem_col_base=col_base,
            tid=bctx.tid,
            n_threads=bctx.n_threads,
            cast=cast,
            use_async=use_async,
        )

    def emit(self, bld: Builder) -> None:
        """Back-compat: explicitly allocate under ``bld``. No-op if
        ``__post_init__`` already handled it (idempotent)."""
        if self._smem is None:
            self._allocate(bld)
