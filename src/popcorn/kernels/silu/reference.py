"""SiLU reference."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def silu_reference(X, *, dtype: DType | str = DType.BF16):
    out_dt = DType(dtype) if isinstance(dtype, str) else dtype
    X_f = PT.astype(X, PT.float32)
    sig = 1.0 / (1.0 + PT.exp(-X_f))
    Y = X_f * sig
    return PT.astype(Y, out_dt.backend)


def silu_reference_for_spec(kernel, X):
    return silu_reference(X, dtype=kernel.spec.dtype)
