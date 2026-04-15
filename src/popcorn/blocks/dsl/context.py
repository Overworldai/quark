"""Active-context ContextVars for the kernel authoring DSL.

When a kernel's ``build()`` runs, the framework publishes the active
:class:`KernelContext` and :class:`BlockContext` here so every
downstream block, free helper, and Value operator can find them
without an explicit argument. The plumbing is analogous to
``_ACTIVE_BUILDER`` in ``popcorn.ir.value`` — one pattern, three handles.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popcorn.blocks.dsl.block_context import BlockContext
    from popcorn.blocks.dsl.kernel_context import KernelContext


# `default=None` is the runtime behavior; ty infers ContextVar[None]
# from that default, so we rely on the decorator/make_ctx setting
# the var before any reader and catch LookupError in the accessors.
_ACTIVE_KCTX: ContextVar[KernelContext | None] = ContextVar("popcorn_active_kctx")
_ACTIVE_BCTX: ContextVar[BlockContext | None] = ContextVar("popcorn_active_bctx")


def active_kctx() -> KernelContext:
    try:
        kctx = _ACTIVE_KCTX.get()
    except LookupError:
        kctx = None
    if kctx is None:
        raise RuntimeError(
            "active_kctx(): no KernelContext published. Call from inside "
            "a @kernel-decorated build() or manually publish via "
            "KernelContext.activate()."
        )
    return kctx


def active_bctx() -> BlockContext:
    try:
        bctx = _ACTIVE_BCTX.get()
    except LookupError:
        bctx = None
    if bctx is None:
        raise RuntimeError(
            "active_bctx(): no BlockContext published. Call "
            "KernelContext.make_ctx(mma_cfg) first (it auto-publishes)."
        )
    return bctx
