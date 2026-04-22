"""ElementwiseSpec — parametric spec for all element-wise ops.

One spec, one kernel, many ops. The ``op`` field selects which
computation happens in the inner loop. Unary ops (neg, abs, exp,
sin, cos, sqrt) use only ``X``; binary ops (add, sub, mul, div)
use both ``X`` and ``Y``.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_DTYPES = frozenset(
    {DType.BF16, DType.F16, DType.F32, DType.S32, DType.U32, DType.E4M3, DType.E5M2}
)

UNARY_OPS = frozenset({"neg", "abs", "exp", "sin", "cos", "sqrt"})
BINARY_OPS = frozenset({"add", "sub", "mul", "div"})
CAST_OPS = frozenset({"cast"})
ALL_OPS = UNARY_OPS | BINARY_OPS | CAST_OPS


def arity_of(op: str) -> int:
    if op in BINARY_OPS:
        return 2
    return 1


@dataclass(frozen=True)
class ElementwiseSpec(KernelSpec):
    """Flat 1-D elementwise op over ``N`` elements.

    ``op``: one of ``add``, ``sub``, ``mul``, ``div``, ``neg``,
    ``abs``, ``exp``, ``sin``, ``cos``, ``sqrt``, ``cast``.

    ``out_dtype``: output dtype (defaults to ``dtype`` when ``None``).
    Used by ``cast`` to specify the target dtype.
    """

    N: int
    dtype: DType = DType.BF16
    op: str = "add"
    out_dtype: DType | None = None

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.out_dtype is not None and isinstance(self.out_dtype, str):
            object.__setattr__(self, "out_dtype", DType(self.out_dtype))
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"ElementwiseSpec: dtype {self.dtype!r} not in {_VALID_DTYPES}")
        if self.op not in ALL_OPS:
            raise ValueError(f"ElementwiseSpec: unknown op {self.op!r}")
        if self.N <= 0:
            raise ValueError(f"ElementwiseSpec: N must be positive; got {self.N}")

    @property
    def arity(self) -> int:
        return arity_of(self.op)

    @property
    def effective_out_dtype(self) -> DType:
        return self.out_dtype if self.out_dtype is not None else self.dtype
