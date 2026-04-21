"""ValueResidual reference — ``out = v + lamb * (v1 - v)``."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def value_residual_reference(V, V1, lamb, *, dtype: DType | str = DType.BF16):
    out_dt = DType(dtype) if isinstance(dtype, str) else dtype
    V_f = PT.astype(V, PT.float32)
    V1_f = PT.astype(V1, PT.float32)
    # lamb is a 1-element device tensor.
    lamb_f = PT.astype(lamb, PT.float32)
    # Broadcast the 1-elem scalar.
    out = V_f + lamb_f * (V1_f - V_f)
    return PT.astype(out, out_dt.backend)


def value_residual_reference_for_spec(kernel, V, V1, lamb):
    s = kernel.spec
    return value_residual_reference(V, V1, lamb, dtype=s.dtype)
