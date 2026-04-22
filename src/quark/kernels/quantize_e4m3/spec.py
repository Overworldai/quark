"""QuantizeE4M3Spec — bf16/f16 → e4m3 quantization."""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_SRC = frozenset({DType.BF16, DType.F16, DType.F32})


@dataclass(frozen=True)
class QuantizeE4M3Spec(KernelSpec):
    """Quantize N elements from src_dtype to e4m3.

    N must be even (packed_convert operates on pairs).
    """

    N: int
    src_dtype: DType = DType.BF16

    def __post_init__(self):
        if isinstance(self.src_dtype, str) and not isinstance(self.src_dtype, DType):
            object.__setattr__(self, "src_dtype", DType(self.src_dtype))
        if self.src_dtype not in _VALID_SRC:
            raise ValueError(f"QuantizeE4M3Spec: src_dtype {self.src_dtype!r} not in {_VALID_SRC}")
        if self.N <= 0 or self.N % 2 != 0:
            raise ValueError(f"QuantizeE4M3Spec: N must be positive and even; got {self.N}")
