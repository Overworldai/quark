"""Declarative kernel authoring DSL.

The `dsl` subpackage is split across focused modules for readability;
this ``__init__.py`` re-exports the public surface kernel authors
import as ``from quark.blocks.dsl import ...`` (or more commonly
via ``quark.blocks``'s flat re-export):

* :mod:`context`          — ContextVars + ``active_kctx`` / ``active_bctx``
* :mod:`tensors`          — :class:`TensorDecl`, :class:`C`
* :mod:`accumulators`     — :class:`Accumulators`
* :mod:`carry`            — :class:`Stage`, :class:`Carry`
* :mod:`smem_tile`        — :class:`SmemTile`
* :mod:`block_context`    — :class:`BlockContext`, :class:`Block`, :class:`SetupBlock`
* :mod:`kernel_context`   — :class:`KernelContext`

Module-level free functions (``c``, ``barrier``, ``tid``, ``gid``,
``tig``, ``warp_id``, ``lane_id``, ``block_idx``, ``block_base``) live
here — they're thin dispatchers over the active BlockContext /
KernelContext.
"""

from __future__ import annotations

from quark.blocks.dsl.accumulators import Accumulators
from quark.blocks.dsl.block_context import Block, BlockContext, SetupBlock
from quark.blocks.dsl.carry import Carry, Stage
from quark.blocks.dsl.context import _ACTIVE_BCTX, _ACTIVE_KCTX, active_bctx, active_kctx
from quark.blocks.dsl.kernel_context import KernelContext
from quark.blocks.dsl.smem_tile import SmemTile, SmemTileSpec
from quark.blocks.dsl.smem_vector import SmemVector
from quark.blocks.dsl.tensors import C, TensorDecl, _to_value
from quark.ir import DType, Value

# ── Module-level helpers: dispatch to the active BlockContext ──────
#
# Every kernel used to start with `bld = ctx.bld; bctx = ctx.make_ctx(...)`
# just to get the names into scope. With an active-ctx ContextVar we
# can expose these as free functions so emit bodies read as pure
# kernel logic.


def c(value: int | float, dtype: DType | None = None) -> Value:
    """Auto-typed, deduplicated constant on the active BlockContext."""
    return active_bctx().c(value, dtype)


def barrier(scope: str = "block") -> None:
    active_bctx().bld.barrier(scope)


def tid() -> Value:
    return active_bctx().tid


def gid() -> Value:
    return active_bctx().gid


def tig() -> Value:
    return active_bctx().tig


def warp_id() -> Value:
    return active_bctx().warp_id


def lane_id() -> Value:
    return active_bctx().lane_id


def block_idx(axis: str) -> Value:
    """Cached block-index Value on the active KernelContext."""
    return active_kctx().block_idx(axis)


def block_base(axis: str, size: int) -> Value:
    """``block_idx(axis) * size`` on the active KernelContext."""
    return active_kctx().block_base(axis, size)


__all__ = [
    "_ACTIVE_BCTX",
    "_ACTIVE_KCTX",
    "Accumulators",
    "Block",
    "BlockContext",
    "C",
    "Carry",
    "KernelContext",
    "SetupBlock",
    "SmemTile",
    "SmemTileSpec",
    "SmemVector",
    "Stage",
    "TensorDecl",
    "_to_value",
    "active_bctx",
    "active_kctx",
    "barrier",
    "block_base",
    "block_idx",
    "c",
    "gid",
    "lane_id",
    "tid",
    "tig",
    "warp_id",
]
