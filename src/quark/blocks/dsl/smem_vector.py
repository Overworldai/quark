"""SmemVector — declarative rank-1 smem allocation with a bundled
``.load_from`` primitive for cooperative gmem→smem vector loads.

Companion to :class:`SmemTile` at rank 1. Used for scale/bias/gate
vectors that live for the whole kernel (not pipelined), where the
gmem source is usually a single row of a ``[G, D]`` global tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import quark.lang as qk
from quark.blocks.dsl.context import active_bctx
from quark.ir import Builder, DType, SharedRegion, Value


@dataclass
class SmemVector:
    """Declarative rank-1 smem allocation.

    Auto-allocates via the active :class:`Builder` in ``__post_init__``.
    ``.smem`` is the underlying ``SharedRegion`` — hand it to
    ``qk.vec_load`` / subscript reads just like the output of
    ``qk.smem_alloc(name, dtype, (length,), pad=...)``.
    """

    name: str
    dtype: DType
    length: int
    pad: int = 0

    _smem: SharedRegion | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from quark.ir.value import _ACTIVE_BUILDER

        bld = _ACTIVE_BUILDER.get(None)
        if bld is not None and self._smem is None:
            self._allocate(bld)

    def _allocate(self, bld: Builder) -> None:
        self._smem = qk.smem_alloc(self.name, self.dtype, (self.length,), pad=self.pad)

    @property
    def smem(self) -> SharedRegion:
        assert self._smem is not None, f"SmemVector {self.name!r} not emitted yet"
        return self._smem

    @smem.setter
    def smem(self, v: SharedRegion) -> None:
        self._smem = v

    def load_from(
        self,
        gmem: Any,
        *,
        row: int | Value | None = None,
        cast: DType | None = None,
        use_async: bool = True,
    ) -> None:
        """Cooperative gmem→smem vector load into this SmemVector.

        Source shapes:
          * rank-1 ``GlobalTensor``: direct flat load.
          * rank-2 ``GlobalTensor``: pass ``row=`` (int or ``Value``) to
            select a single row. Mirrors the norm kernels' scale/bias
            vector pattern where ``gmem.shape == (G, D)`` and the active
            group's row is loaded.

        ``cast`` forces the scalar path and casts elements during the
        load. ``use_async`` toggles cp.async (auto-disabled under
        ``cast``).
        """
        from quark.blocks.l0.vector_loader import emit_vector_load
        from quark.ir import GlobalTensor

        if not isinstance(gmem, GlobalTensor):
            raise TypeError(
                f"SmemVector.load_from: gmem must be a GlobalTensor, got {type(gmem).__name__}"
            )

        bctx = active_bctx()
        if gmem.rank == 2:
            if row is None:
                raise ValueError("SmemVector.load_from: gmem is rank-2; pass row= to select a row")
            gmem_row = row if isinstance(row, Value) else bctx.c(row, dtype=DType.U32)
        elif gmem.rank == 1:
            if row is not None:
                raise ValueError(
                    f"SmemVector.load_from: gmem is rank-1; row= must be None (got {row!r})"
                )
            gmem_row = None
        else:
            raise ValueError(
                f"SmemVector.load_from: gmem must be rank-1 or rank-2 (got shape {gmem.shape})"
            )

        is_async = use_async and cast is None
        emit_vector_load(
            bctx.bld,
            dst_smem=self.smem,
            src_gmem=gmem,
            length=self.length,
            tid=bctx.tid,
            n_threads=bctx.n_threads,
            gmem_row=gmem_row,
            use_async=is_async,
            cast=cast,
        )
