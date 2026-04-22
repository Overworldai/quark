"""RandnSpec — configuration for the Philox-based randn kernel.

One output buffer ``Out[N]`` plus a 1-element ``counter_offset`` input
that the host bumps per launch to get a fresh draw. ``seed0`` / ``seed1``
are compile-time u32 constants that mix into the Philox key — different
seeds give uncorrelated sequences even when ``counter_offset`` collides.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class RandnSpec(KernelSpec):
    N: int
    dtype: DType = DType.BF16
    # Compile-time key bits. Different (seed0, seed1) pairs give
    # independent streams; the host picks these once at model init and
    # varies ``counter_offset`` per frame to draw without reseeding.
    seed0: int = 0xDEADBEEF
    seed1: int = 0xBADC0FFE

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in (DType.F32, DType.F16, DType.BF16):
            raise ValueError(f"RandnSpec: dtype must be F32/F16/BF16, got {self.dtype!r}")
        if self.N <= 0:
            raise ValueError(f"RandnSpec: N must be > 0, got {self.N}")
