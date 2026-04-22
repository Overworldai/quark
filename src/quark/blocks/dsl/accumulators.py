"""Accumulators — declarative MMA accumulator grid spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import quark.lang as qk
from quark.ir import DType, Value


@dataclass
class Accumulators:
    """Declarative accumulator grid spec.

    Describes MT×NT width-``width`` vectors of a given dtype, initialized
    to ``fill_value``. The IR Values are created by :meth:`init` under
    the active Builder.

    After ``run_pipeline`` returns, per-tile result Values are stashed
    on ``self.results`` so epilogue helpers (``qk.store_acc``,
    ``qk.atomic_store_acc``) can take just the ``Accumulators``
    instance and pull both the layout (MT/NT) and the values from it.
    Outside a ``run_pipeline`` flow ``self.results`` is None.
    """

    MT: int
    NT: int
    width: int = 4
    dtype: DType = DType.F32
    fill_value: float = 0.0
    # Populated by run_pipeline after the loop returns when this
    # Accumulators was passed as ``PipelineBody.carry``.
    results: list[Value] | None = field(default=None, repr=False)

    @classmethod
    def for_mma(cls, mma_cfg: Any, BM: int, BN: int) -> Accumulators:
        """Legacy: hardcoded to m16n8 layout. Kept for pre-A9 callers.
        New code should use :meth:`from_mma` which derives everything
        from the MmaConfig + per-warp N split."""
        return cls(MT=BM // 16, NT=BN // 8, width=4, dtype=DType.F32)

    @classmethod
    def from_mma(
        cls,
        mma_cfg: Any,
        *,
        BM: int,
        BN: int,
        n_warps: int = 1,
    ) -> Accumulators:
        """Derive the accumulator grid from an MmaConfig + block tiles.

        ``MT = BM / shape.m``, ``NT = (BN / shape.n) / n_warps`` (per-warp
        split), ``width = shape.c_regs``. Accumulator dtype is always
        the shape's acc_dtype (F32 for every MMA we currently support).

        Collapses the four locals (``m_tile``, ``n_tile``, ``NT_per_warp``,
        ``acc_width``) every GEMM-shaped kernel computes by hand into
        one call.
        """
        return cls(
            MT=BM // mma_cfg.shape.m,
            NT=(BN // mma_cfg.shape.n) // n_warps,
            width=mma_cfg.shape.c_regs,
            dtype=mma_cfg.shape.acc_dtype,
        )

    def init(self) -> list[Value]:
        """Emit MT*NT vec accumulators seeded with ``self.fill_value``.

        The Builder is looked up from the active kernel scope — kernels
        never need to pass it explicitly.
        """
        zero = qk.const(self.dtype, self.fill_value)
        return [qk.vec_build([zero] * self.width) for _ in range(self.MT * self.NT)]

    def count(self) -> int:
        return self.MT * self.NT
