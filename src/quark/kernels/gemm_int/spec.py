"""GemmIntSpec — int8 GEMM with per-row scales.

``C[M, N] = (A_s8[M, K] @ B_s8[K, N]) * A_scales[M] * B_scales[N]``

The s8/s8 MMA accumulates into s32; the epilogue converts s32→f32
with the combined per-row × per-col scale, then casts to ``out_dtype``.

Validated path: ``compute_dtype`` and ``acc_dtype`` are fixed to
``S8`` and ``S32`` respectively, matching the Intel Xe2
``m8n16k32_intel_s8_s32`` cooperative_matrix shape. The caller is
responsible for quantizing A/B to int8 + emitting per-row scale
vectors before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_OUT = frozenset({DType.BF16, DType.F16, DType.F32})


@dataclass(frozen=True)
class GemmIntSpec(KernelSpec):
    """int8 GEMM. ``a_dtype`` and ``b_dtype`` are fixed to S8."""

    M: int
    N: int
    K: int
    out_dtype: DType = DType.BF16
    # Per-row scale dtype — always F32 (the only sensible choice for
    # the accumulator-scale multiply path).
    scale_dtype: DType = DType.F32

    @property
    def a_dtype(self) -> DType:
        return DType.S8

    @property
    def b_dtype(self) -> DType:
        return DType.S8

    @property
    def acc_dtype(self) -> DType:
        return DType.S32

    @property
    def compute_dtype_resolved(self) -> DType:
        return DType.S8

    def __post_init__(self):
        if self.out_dtype not in _VALID_OUT:
            raise ValueError(
                f"GemmIntSpec: out_dtype {self.out_dtype!r} not in {_VALID_OUT}"
            )
        if self.M <= 0 or self.N <= 0 or self.K <= 0:
            raise ValueError(f"GemmIntSpec: M/N/K must be positive (got M={self.M} N={self.N} K={self.K})")
