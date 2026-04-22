"""UnpatchifySpec — GEMM with scatter epilogue to [B, C, H, W].

    X  [M, d_model]  ×  W [C*ph*pw, d_model]^T  +  bias [C*ph*pw]
      →  Out [B, C*H*W]   (flat image layout)

The epilogue maps each output element (token, patch_elem) to the
spatial position (b, c, h, w) and writes directly — no host-side
reshape or permute.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class UnpatchifySpec(KernelSpec):
    B: int
    C: int
    H: int  # full spatial height (= Hp * ph)
    W: int  # full spatial width (= Wp * pw)
    ph: int = 2
    pw: int = 2
    d_model: int = 2048
    dtype: DType = DType.BF16
    compute_dtype: DType | None = None
    has_bias: bool = True

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))

    @property
    def compute_dtype_resolved(self) -> DType:
        return self.compute_dtype if self.compute_dtype is not None else self.dtype

    @property
    def Hp(self) -> int:
        return self.H // self.ph

    @property
    def Wp(self) -> int:
        return self.W // self.pw

    @property
    def M(self) -> int:
        return self.B * self.Hp * self.Wp

    @property
    def K(self) -> int:
        return self.d_model

    @property
    def N(self) -> int:
        return self.C * self.ph * self.pw
