"""ValueResidualPackedSpec — lerp V columns of packed QKV tensor.

    QKV_out = copy of QKV_curr, except V columns are:
      V_out = V_curr + lamb * (V_first - V_curr)

V columns are at [v_col_offset, v_col_offset + n_kv_heads * Dh).
Q and K columns are copied unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class ValueResidualPackedSpec(KernelSpec):
    M: int  # tokens
    D_full: int  # qkv_dim (4096 for wp1.5)
    v_col_offset: int  # start of V columns
    v_width: int  # number of V columns (n_kv_heads * Dh)
    dtype: DType = DType.BF16

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
