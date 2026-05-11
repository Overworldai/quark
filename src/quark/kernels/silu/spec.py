"""SiLUSpec — element-wise ``y = x * sigmoid(x)``."""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class SiLUSpec(KernelSpec):
    N: int
    dtype: DType = DType.BF16

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in _VALID:
            raise ValueError(f"SiLUSpec: dtype {self.dtype!r} not in {_VALID}")
        if self.N <= 0:
            raise ValueError(f"SiLUSpec: N must be positive; got {self.N}")
