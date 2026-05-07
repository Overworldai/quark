"""quark.lang — kernel authoring surface.

EXEMPT FROM 500-LINE RULE: this module is the single ``import quark.lang
as qk`` namespace kernel authors hit; every free function here is a
1-3 line passthrough to ``Builder`` / ``BlockContext``. Splitting forces
authors to remember which submodule each helper lives in for no benefit.

Free functions that forward to the active `Builder`. The builder is
held in a `ContextVar` stack so nested kernel emission is safe.

Usage:
    import quark.lang as qk

    bld = Builder("my_kernel")
    with qk.kernel_scope(bld):
        x = qk.load(ptr, idx)
        y = qk.exp2(x)
        qk.store(out, idx, y)

        with qk.for_range(qk.const(DType.S32, 0),
                           qk.const(DType.S32, K),
                           qk.const(DType.S32, 1)) as (k, _):
            ...

Module-setup operations (`begin_function`, `param`, `register_shape`,
`module` property) stay on the Builder — they run in `Kernel.emit()`,
not inside the authoring surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from quark.ir.builder import Builder
from quark.ir.value import _ACTIVE_BUILDER

__all__ = [  # noqa: RUF022 — grouped by category; see comments below
    "kernel_scope",
    "current_builder",
    # arithmetic
    "const",
    "add",
    "sub",
    "mul",
    "mul_hi",
    "div",
    "rem",
    "min",
    "max",
    "sum",
    "shl",
    "shr",
    "and_",
    "or_",
    "xor",
    "neg",
    "abs_",
    "fma",
    "fma_bf16x2",
    "cvt_rn_bf16x2_f32",
    # compare / select
    "cmp",
    "select",
    # conversion
    "convert",
    "packed_convert",
    "unpacked_convert",
    "bitcast",
    # math approximations
    "rcp_approx",
    "rsqrt_approx",
    "ex2_approx",
    "sqrt",
    "sqrt_approx",
    "exp2",
    "log2",
    "log2_approx",
    "tanh",
    # vectors
    "vec_build",
    "vec_extract",
    "vec_build_packed_b32",
    "packed_extract_b32",
    "split_b32",
    "merge_b32",
    # memory
    "load",
    "store",
    "vec_load",
    "vec_store",
    "async_copy",
    "async_commit",
    "async_wait",
    "atomic_rmw",
    "smem_alloc",
    # subgroup / lane
    "shuffle",
    "subgroup_reduce",
    "subgroup_broadcast",
    # indexing
    "thread_idx",
    "block_idx",
    "block_dim",
    "grid_dim",
    "lane_id",
    "subgroup_id",
    "group_id",
    "thread_id_in_group",
    # sync
    "barrier",
    # control flow
    "for_range",
    "if_",
    "yield_",
    # matmul / fragments
    "load_matrix",
    "store_matrix",
    "mma",
    "frag_apply",
    "frag_for_each",
    "frag_convert",
    "frag_reduce",
    # introspection
    "last_results",
    # high-level epilogue helpers
    "silu",
    "cast",
    "store_acc",
    "atomic_store_acc",
    # memory helpers
    "work_list_load",
    "index_cache",
    "q_register_load",
]


# ── High-level epilogue helpers (re-exports from .epilogue) ──
from quark.lang.epilogue import (
    atomic_store_acc,
    cast,
    silu,
    store_acc,
)

# ── Kernel-authoring memory helpers (re-exports from .memory) ──
from quark.lang.memory import (
    index_cache,
    q_register_load,
    work_list_load,
)


def current_builder() -> Builder:
    """Return the active Builder.

    Reads ``quark.ir.value._ACTIVE_BUILDER`` which is published by
    ``Builder.begin_function()`` and reset by ``end_function()``. Inside
    a kernel's ``emit()`` body this is always set. Raises if called
    with no active builder (e.g. at module scope or in a test that
    forgot to open a function).
    """
    bld = _ACTIVE_BUILDER.get(None)
    if bld is None:
        raise RuntimeError(
            "quark.lang: no active Builder. Call this from inside a "
            "kernel emit body (between begin_function / end_function), "
            "or wrap a bare block in `with quark.lang.kernel_scope(bld):`."
        )
    return bld


@contextmanager
def kernel_scope(bld: Builder) -> Iterator[Builder]:
    """Publish ``bld`` as the active Builder for the enclosed block.

    Normally not needed — ``Builder.begin_function()`` already publishes
    the active Builder via ``_ACTIVE_BUILDER``, and kernels enter that
    path through ``KernelContext.__init__``. This context manager is
    for tests and ad-hoc scripts that want to emit ops against a
    freshly-constructed Builder without opening a function.

    Idempotent against the underlying ``ContextVar``: resets via token
    on exit (including on exception), so nested uses are safe.
    """
    token = _ACTIVE_BUILDER.set(bld)
    try:
        yield bld
    finally:
        _ACTIVE_BUILDER.reset(token)


# ---------------------------------------------------------------
# Forwarders.
#
# Each forwarder is a thin wrapper. We keep them as explicit `def`s
# (not a loop + setattr) so IDE signature help / static type checkers
# see a real function per name.
# ---------------------------------------------------------------


def const(*args, **kwargs):
    return current_builder().const(*args, **kwargs)


# arithmetic
def add(*args, **kwargs):
    return current_builder().add(*args, **kwargs)


def sub(*args, **kwargs):
    return current_builder().sub(*args, **kwargs)


def mul(*args, **kwargs):
    return current_builder().mul(*args, **kwargs)


def mul_hi(*args, **kwargs):
    """High 32 bits of a u32×u32→u64 multiply. See ``Builder.mul_hi``."""
    return current_builder().mul_hi(*args, **kwargs)


def div(*args, **kwargs):
    return current_builder().div(*args, **kwargs)


def rem(*args, **kwargs):
    return current_builder().rem(*args, **kwargs)


def min(*args, **kwargs):
    """``qk.min(a, b)`` — elementwise min.

    ``qk.min(frag, dim=1)`` — cross-lane min reduction along the MMA
    frag's column axis (per-row reduction). Dispatches to
    ``frag_reduce(kind="min", axis="row")`` using the active
    BlockContext's mma_cfg. ``dim=0`` is the column reduction."""
    if "dim" in kwargs:
        return _frag_reduce_dim("min", args[0], kwargs["dim"])
    return current_builder().min(*args, **kwargs)


def max(*args, **kwargs):
    """``qk.max(a, b)`` — elementwise max.

    ``qk.max(frag, dim=1)`` — per-row max reduction across the MMA
    frag. See ``min`` docstring."""
    if "dim" in kwargs:
        return _frag_reduce_dim("max", args[0], kwargs["dim"])
    return current_builder().max(*args, **kwargs)


def sum(*args, **kwargs):
    """``qk.sum(frag, dim=1)`` — per-row sum reduction across the
    MMA frag. No ``qk.sum(a, b)`` form; use ``a + b`` for addition."""
    if "dim" in kwargs:
        return _frag_reduce_dim("add", args[0], kwargs["dim"])
    raise TypeError(
        "qk.sum: requires a `dim=` kwarg (cross-lane reduction). Use ``a + b`` for scalar addition."
    )


def _frag_reduce_dim(kind: str, value, dim: int):
    """Shared dim→axis router for qk.{min,max,sum}(x, dim=).

    ``dim=1`` → ``axis="row"`` (reduce across columns, one scalar per
    row class). ``dim=0`` → ``axis="col"`` (reduce across rows). Reads
    ``cd_offsets`` and ``shape_id`` from the active ``BlockContext``'s
    ``mma_cfg`` — the caller supplies just the frag Value.
    """
    from quark.blocks.dsl import active_bctx

    if dim not in (0, 1):
        raise ValueError(f"qk.{kind}: dim must be 0 or 1, got {dim}")
    axis = "row" if dim == 1 else "col"
    bctx = active_bctx()
    cfg = bctx.mma_cfg
    if cfg is None:
        raise RuntimeError(
            f"qk.{kind}(x, dim={dim}): no active MMA shape on bctx. "
            "Frag reductions require a bctx constructed with mma_cfg."
        )
    return current_builder().frag_reduce(
        cfg.shape_id,
        value,
        kind=kind,
        axis=axis,
        cd_offsets=cfg.cd_offsets,
    )


def shl(*args, **kwargs):
    return current_builder().shl(*args, **kwargs)


def shr(*args, **kwargs):
    return current_builder().shr(*args, **kwargs)


def and_(*args, **kwargs):
    return current_builder().and_(*args, **kwargs)


def or_(*args, **kwargs):
    return current_builder().or_(*args, **kwargs)


def xor(*args, **kwargs):
    return current_builder().xor(*args, **kwargs)


def neg(*args, **kwargs):
    return current_builder().neg(*args, **kwargs)


def abs_(*args, **kwargs):
    return current_builder().abs(*args, **kwargs)


def fma(*args, **kwargs):
    return current_builder().fma(*args, **kwargs)


def fma_bf16x2(*args, **kwargs):
    return current_builder().fma_bf16x2(*args, **kwargs)


def cvt_rn_bf16x2_f32(*args, **kwargs):
    return current_builder().cvt_rn_bf16x2_f32(*args, **kwargs)


# compare / select
def cmp(*args, **kwargs):
    return current_builder().cmp(*args, **kwargs)


def select(*args, **kwargs):
    return current_builder().select(*args, **kwargs)


# conversion
def convert(*args, **kwargs):
    return current_builder().convert(*args, **kwargs)


def packed_convert(*args, **kwargs):
    return current_builder().packed_convert(*args, **kwargs)


def unpacked_convert(*args, **kwargs):
    return current_builder().unpacked_convert(*args, **kwargs)


def bitcast(*args, **kwargs):
    return current_builder().bitcast(*args, **kwargs)


# math approximations
def rcp_approx(*args, **kwargs):
    return current_builder().rcp_approx(*args, **kwargs)


def rsqrt_approx(*args, **kwargs):
    return current_builder().rsqrt_approx(*args, **kwargs)


def ex2_approx(*args, **kwargs):
    return current_builder().ex2_approx(*args, **kwargs)


def exp_approx(*args, **kwargs):
    return current_builder().exp_approx(*args, **kwargs)


def sqrt(*args, **kwargs):
    return current_builder().sqrt(*args, **kwargs)


def sqrt_approx(*args, **kwargs):
    return current_builder().sqrt_approx(*args, **kwargs)


def exp2(*args, **kwargs):
    return current_builder().exp2(*args, **kwargs)


def log2(*args, **kwargs):
    return current_builder().log2(*args, **kwargs)


def log2_approx(*args, **kwargs):
    return current_builder().log2_approx(*args, **kwargs)


def sin(*args, **kwargs):
    return current_builder().sin(*args, **kwargs)


def cos(*args, **kwargs):
    return current_builder().cos(*args, **kwargs)


def tanh(*args, **kwargs):
    return current_builder().tanh(*args, **kwargs)


# vectors
def vec_build(*args, **kwargs):
    return current_builder().vec_build(*args, **kwargs)


def vec_extract(*args, **kwargs):
    return current_builder().vec_extract(*args, **kwargs)


def vec_build_packed_b32(*args, **kwargs):
    return current_builder().vec_build_packed_b32(*args, **kwargs)


def packed_extract_b32(*args, **kwargs):
    return current_builder().packed_extract_b32(*args, **kwargs)


def split_b32(*args, **kwargs):
    return current_builder().split_b32(*args, **kwargs)


def merge_b32(*args, **kwargs):
    return current_builder().merge_b32(*args, **kwargs)


# memory
def load(*args, **kwargs):
    return current_builder().load(*args, **kwargs)


def store(*args, **kwargs):
    return current_builder().store(*args, **kwargs)


def vec_load(*args, **kwargs):
    return current_builder().vec_load(*args, **kwargs)


def vec_store(*args, **kwargs):
    return current_builder().vec_store(*args, **kwargs)


def async_copy(*args, **kwargs):
    return current_builder().async_copy(*args, **kwargs)


def async_commit(*args, **kwargs):
    return current_builder().async_commit(*args, **kwargs)


def async_wait(*args, **kwargs):
    return current_builder().async_wait(*args, **kwargs)


def atomic_rmw(*args, **kwargs):
    return current_builder().atomic_rmw(*args, **kwargs)


def smem_alloc(*args, **kwargs):
    return current_builder().smem_alloc(*args, **kwargs)


# subgroup / lane
def shuffle(*args, **kwargs):
    return current_builder().shuffle(*args, **kwargs)


def subgroup_reduce(*args, **kwargs):
    return current_builder().subgroup_reduce(*args, **kwargs)


def subgroup_broadcast(*args, **kwargs):
    return current_builder().subgroup_broadcast(*args, **kwargs)


# indexing / launch identity
def thread_idx(*args, **kwargs):
    return current_builder().thread_idx(*args, **kwargs)


def block_idx(*args, **kwargs):
    return current_builder().block_idx(*args, **kwargs)


def block_dim(*args, **kwargs):
    return current_builder().block_dim(*args, **kwargs)


def grid_dim(*args, **kwargs):
    return current_builder().grid_dim(*args, **kwargs)


def lane_id(*args, **kwargs):
    return current_builder().lane_id(*args, **kwargs)


def subgroup_id(*args, **kwargs):
    return current_builder().subgroup_id(*args, **kwargs)


def group_id(*args, **kwargs):
    return current_builder().group_id(*args, **kwargs)


def thread_id_in_group(*args, **kwargs):
    return current_builder().thread_id_in_group(*args, **kwargs)


# sync
def barrier(*args, **kwargs):
    return current_builder().barrier(*args, **kwargs)


# control flow — these already return context managers on Builder;
# `with qk.for_range(...)` / `with qk.if_(...)` just forward.
def for_range(*args, **kwargs):
    """Python-friendly wrapper over ``Builder.for_loop``.

    Lifts Python ``int`` bounds (``lo`` / ``hi`` / ``step``) to
    :class:`~quark.ir.Value` constants using U32 as the default dtype
    so kernel authors can write::

        with qk.for_range(0, n_chunks, 1, iv_name="k", carried=carry):

    instead of sprinkling ``bctx.const(DType.U32, 0)`` everywhere. The
    ``carried=`` kwarg also accepts any iterable (tuple / Carry.flatten()
    output / generator) — it's normalized to a tuple on the way through.
    """
    from quark.ir import DType

    bld = current_builder()

    def _as_value(x: Any) -> Any:
        if isinstance(x, bool):
            return bld.const(DType.PRED, x)
        if isinstance(x, int):
            return bld.const(DType.U32, x)
        return x

    # Normalize leading positional args (lo / hi / step).
    args = tuple(_as_value(a) for a in args)
    if "lo" in kwargs:
        kwargs["lo"] = _as_value(kwargs["lo"])
    if "hi" in kwargs:
        kwargs["hi"] = _as_value(kwargs["hi"])
    if "step" in kwargs:
        kwargs["step"] = _as_value(kwargs["step"])
    if "carried" in kwargs and kwargs["carried"] is not None:
        kwargs["carried"] = tuple(kwargs["carried"])
    return bld.for_loop(*args, **kwargs)


def if_(*args, **kwargs):
    return current_builder().if_(*args, **kwargs)


def yield_(*args, **kwargs):
    # Auto-flatten a Carry argument: ``qk.yield_(carry)`` is equivalent
    # to ``qk.yield_(*carry.flatten())`` — the loop ``carried=`` API
    # still wants a flat tuple of Values under the hood.
    from quark.blocks.dsl import Carry

    if len(args) == 1 and isinstance(args[0], Carry):
        args = tuple(args[0].flatten())
    return current_builder().yield_(*args, **kwargs)


# matmul / fragments
def load_matrix(*args, **kwargs):
    return current_builder().load_matrix(*args, **kwargs)


def store_matrix(*args, **kwargs):
    return current_builder().store_matrix(*args, **kwargs)


def mma(*args, **kwargs):
    return current_builder().mma(*args, **kwargs)


def frag_apply(*args, **kwargs):
    return current_builder().frag_apply(*args, **kwargs)


def frag_for_each(*args, **kwargs):
    return current_builder().frag_for_each(*args, **kwargs)


def frag_convert(*args, **kwargs):
    return current_builder().frag_convert(*args, **kwargs)


def frag_reduce(*args, **kwargs):
    return current_builder().frag_reduce(*args, **kwargs)


# introspection
def last_results():
    return current_builder().last_results
