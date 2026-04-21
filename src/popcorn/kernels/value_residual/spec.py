"""ValueResidualSpec — ``out = v + lamb * (v1 - v)`` element-wise.

Waypoint-1.5 has ``value_residual: true``: after the V projection,
each layer blends the current V with the first-layer V by a learned
scalar ``lamb`` (per-layer parameter). This runs once per layer per
frame, right before ``kv_cache_update``.

``N`` is the flattened total element count — the kernel is rank-1 so
the caller can reshape ``[B, Hk, tpf, Dh]`` or anything else to
``[N]`` without kernel changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec

_VALID_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32})


@dataclass(frozen=True)
class ValueResidualSpec(KernelSpec):
    N: int
    dtype: DType = DType.BF16

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"ValueResidualSpec: dtype {self.dtype!r} not in {_VALID_DTYPES}")
        if self.N <= 0:
            raise ValueError(f"ValueResidualSpec: N must be positive; got {self.N}")
