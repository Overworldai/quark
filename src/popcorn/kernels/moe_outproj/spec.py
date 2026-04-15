"""MoeOutprojSpec — problem definition for the MoE out-projection."""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec

_VALID_AB = frozenset({DType.BF16, DType.F16, DType.E4M3, DType.E5M2})
_VALID_OUT = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class MoeOutprojSpec(KernelSpec):
    M: int  # output tokens
    D: int  # output feature dim (N of GEMM)
    H: int  # hidden dim (K of GEMM)
    n_experts: int
    top_k: int = 2
    a_dtype: DType = DType.BF16
    b_dtype: DType = DType.BF16
    out_dtype: DType = DType.BF16
    # Optional compute dtype — A/B get cast to this during the tile
    # load, so smem and mma fragments run in compute_dtype. Defaults
    # to a_dtype (no cast) for back-compat.
    compute_dtype: DType | None = None

    def __post_init__(self):
        for field in ("a_dtype", "b_dtype", "out_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))
        if (
            self.compute_dtype is not None
            and isinstance(self.compute_dtype, str)
            and not isinstance(self.compute_dtype, DType)
        ):
            object.__setattr__(self, "compute_dtype", DType(self.compute_dtype))
        if self.a_dtype not in _VALID_AB:
            raise ValueError(f"MoeOutprojSpec: a_dtype {self.a_dtype!r} not in {_VALID_AB}")
        if self.b_dtype not in _VALID_AB:
            raise ValueError(f"MoeOutprojSpec: b_dtype {self.b_dtype!r} not in {_VALID_AB}")
        if self.out_dtype not in _VALID_OUT:
            raise ValueError(f"MoeOutprojSpec: out_dtype {self.out_dtype!r} not in {_VALID_OUT}")
        if self.compute_dtype is not None and self.compute_dtype not in _VALID_AB:
            raise ValueError(
                f"MoeOutprojSpec: compute_dtype {self.compute_dtype!r} not in {_VALID_AB}"
            )

    @property
    def total_slots(self) -> int:
        return self.M * self.top_k

    @property
    def compute_dtype_resolved(self) -> DType:
        return self.compute_dtype if self.compute_dtype is not None else self.a_dtype
