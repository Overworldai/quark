"""PatchifySpec — Conv2d(kernel=stride=patch) as a GEMM with strided input.

    X  [B, C, H, W]  →  Out [B*Hp*Wp, d_model]

Where Hp=H/ph, Wp=W/pw, K=C*ph*pw. The ``permute`` that interleaves
the patch pixels into the K dimension happens inside the kernel's A-tile
loader — no host-side reshape or transpose needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec


@dataclass(frozen=True)
class PatchifySpec(KernelSpec):
    B: int  # batch (always 1 for inference)
    C: int  # input channels (32 for wp1.5)
    H: int  # input height (pre-patchify)
    W: int  # input width (pre-patchify)
    ph: int = 2  # patch height
    pw: int = 2  # patch width
    d_model: int = 2048
    dtype: DType = DType.BF16
    compute_dtype: DType | None = None

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.H % self.ph != 0 or self.W % self.pw != 0:
            raise ValueError(
                f"PatchifySpec: H={self.H} W={self.W} not divisible by patch ({self.ph},{self.pw})"
            )

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
        """Output rows = tokens."""
        return self.B * self.Hp * self.Wp

    @property
    def K(self) -> int:
        """GEMM K dimension = C * ph * pw."""
        return self.C * self.ph * self.pw

    @property
    def N(self) -> int:
        """GEMM N dimension = d_model."""
        return self.d_model
