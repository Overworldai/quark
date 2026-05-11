"""BlockContext + Block / SetupBlock base classes.

``BlockContext`` caches thread-identity Values (``tid``, ``gid``,
``warp_id``, ``lane_id``, ``tig``) and constants per kernel invocation;
every free helper and L1/L2 block pulls it via ``active_bctx()``.
``Block`` / ``SetupBlock`` are the minimal base classes composable
blocks inherit from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import quark.lang as qk
from quark.blocks.dsl.context import _ACTIVE_BCTX
from quark.ir import Builder, DType, Value

if TYPE_CHECKING:
    from quark.kernels.gemm.mma_shapes import MmaConfig


class BlockContext:
    """Provides builder + thread identity + constants to blocks.

    Created once per kernel and passed to every block's emit().
    Thread identity values (tid, gid, tig) are computed lazily
    and cached — no redundant IR ops.
    """

    def __init__(self, bld: Builder, n_threads: int, mma_cfg: MmaConfig | None = None):
        self.bld = bld
        self.n_threads = n_threads
        self.mma_cfg: MmaConfig = cast("MmaConfig", mma_cfg)
        self._tid: Value | None = None
        self._gid: Value | None = None
        self._tig: Value | None = None
        self._tig_x2: Value | None = None
        self._hoist_cache: dict[str, Value] = {}
        self._warp_id: Value | None = None

    @property
    def tid(self) -> Value:
        if self._tid is None:
            self._tid = qk.thread_idx("x")
        return self._tid

    @property
    def gid(self) -> Value:
        if self._gid is None:
            self._gid = qk.group_id()
        return self._gid

    @property
    def tig(self) -> Value:
        if self._tig is None:
            self._tig = qk.thread_id_in_group()
        return self._tig

    @property
    def tig_x2(self) -> Value:
        if self._tig_x2 is None:
            self._tig_x2 = qk.mul(self.tig, self.const(DType.U32, 2))
        return self._tig_x2

    @property
    def warp_id(self) -> Value:
        if self._warp_id is None:
            self._warp_id = qk.subgroup_id()
        return self._warp_id

    @property
    def lane_id(self) -> Value:
        return self.hoist("lane_id", lambda: qk.lane_id())

    def hoist(self, name: str, compute_fn: Any) -> Value:
        """Compute an expression once and cache by name.

        Usage:
            val = ctx.hoist("warp_b_offset", lambda: qk.mul(warp_id, ctx.c(64)))

        The lambda is called exactly once. Subsequent calls with the same
        name return the cached Value — no duplicate IR ops emitted.
        """
        if name not in self._hoist_cache:
            self._hoist_cache[name] = compute_fn()
        return self._hoist_cache[name]

    def const(self, dtype: DType, value: int | float) -> Value:
        """Deduplicated constant materialization.

        Dedup is done by ``Builder.const`` via the region-scoped CSE
        stack — reusing a const Value from an outer region inside an
        inner region is legal (SSA dominance holds), but leaking an
        inner-region const out to an outer use is not. The CSE stack
        enforces this automatically; callers can use this (or
        ``qk.const``) freely inside nested regions.
        """
        return self.bld.const(dtype, value)

    def c(self, value: int | float, dtype: DType | None = None) -> Value:
        """Shorthand: auto-typed constant."""
        if dtype is None:
            if isinstance(value, float):
                dtype = DType.F32
            elif value < 0:
                dtype = DType.S32
            else:
                dtype = DType.U32
        return self.const(dtype, value)


class Block:
    """Base class for composable blocks.

    Subclasses implement emit(ctx) to produce IR. Blocks are specs
    until emit() is called — they can be inspected, composed, and
    reordered before any IR is generated.
    """

    def emit(self, ctx: BlockContext) -> Any:
        raise NotImplementedError


class SetupBlock(Block):
    """Block that emits at construction time when a BlockContext is active.

    Subclasses are one-shot setup primitives (smem allocation, index
    caching, work-list load) whose construction always coincides with
    their emission. Used via dataclass inheritance — the subclass
    declares fields with ``@dataclass``; ``__post_init__`` here fires
    ``self.emit(active_bctx())`` automatically when a kernel build is
    in flight. Outside a kernel build (tests, REPL) construction stays
    inert and ``self.emit(bctx)`` remains callable for manual use.

    ``emit()`` may return a value; if it does, subclasses are
    responsible for stashing the return on ``self`` (e.g. ``IndexCache``
    sets ``self._smem``; ``WorkListLoad`` sets ``self.grp_start`` /
    ``self.expert``). The auto-emit path discards the return.
    """

    def __post_init__(self) -> None:
        bctx = _ACTIVE_BCTX.get()
        if bctx is not None:
            self.emit(bctx)
