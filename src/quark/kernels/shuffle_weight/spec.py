"""ShuffleWeightSpec — GPU-side weight pre-shuffle for vectorized frag loads.

Permutes bytes within [8, bstride] element tiles of a [N, K] weight matrix
so that each MMA lane's fragment data is contiguous in shared memory.
The AE internally resizes 16:9 → 2:1 before encoding.

When ``bpad > 0``, the output K dimension is wider than the input:
``K_out = (K / K_CHUNK) * (K_CHUNK + bpad)``. The extra columns are
zero-padded for bank conflict avoidance.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class ShuffleWeightSpec(KernelSpec):
    """Shuffle [N, K] weight tensor in-tile.

    ``N``: number of rows (must be multiple of 8).
    ``K``: input K dimension in elements.
    ``dtype``: weight element type (bf16, f16, e4m3, etc.).
    ``K_CHUNK``: tile width in elements (e.g. 64).
    ``mma_k``: MMA K dimension (16 or 32).
    ``bpad``: pad elements per K tile (0 = no padding).
    """

    N: int
    K: int
    dtype: DType = DType.BF16
    K_CHUNK: int = 64
    mma_k: int = 16
    bpad: int = 0

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.N <= 0 or self.N % 8 != 0:
            raise ValueError(
                f"ShuffleWeightSpec: N must be positive and multiple of 8; got {self.N}"
            )
        if self.K <= 0:
            raise ValueError(f"ShuffleWeightSpec: K must be positive; got {self.K}")
        if self.K % self.K_CHUNK != 0:
            raise ValueError(
                f"ShuffleWeightSpec: K ({self.K}) must be divisible by K_CHUNK ({self.K_CHUNK})"
            )
        if self.K_CHUNK % self.mma_k != 0:
            raise ValueError(
                f"ShuffleWeightSpec: K_CHUNK ({self.K_CHUNK}) must be divisible by mma_k ({self.mma_k})"
            )

    @property
    def dtype_bytes(self) -> int:
        return self.dtype.bytes

    @property
    def K_CHUNK_BYTES(self) -> int:
        return self.K_CHUNK * self.dtype_bytes

    @property
    def K_bytes(self) -> int:
        return self.K * self.dtype_bytes

    @property
    def bstride(self) -> int:
        """Elements per padded K tile (= K_CHUNK + bpad)."""
        return self.K_CHUNK + self.bpad

    @property
    def bstride_bytes(self) -> int:
        return self.bstride * self.dtype_bytes

    @property
    def n_k_tiles(self) -> int:
        return self.K // self.K_CHUNK

    @property
    def K_out(self) -> int:
        """Output K dimension in elements."""
        return self.n_k_tiles * self.bstride

    @property
    def K_out_bytes(self) -> int:
        return self.K_out * self.dtype_bytes

    @property
    def tile_bytes(self) -> int:
        """Bytes per output tile (8 rows × bstride_bytes cols)."""
        return 8 * self.bstride_bytes

    @property
    def total_out_bytes(self) -> int:
        return (self.N // 8) * self.n_k_tiles * self.tile_bytes

    @property
    def FRAG_REGS(self) -> int:
        return self.mma_k * self.dtype_bytes // 16

    @property
    def FRAG_BYTES(self) -> int:
        return self.FRAG_REGS * 4

    @property
    def K_STEPS(self) -> int:
        return self.K_CHUNK // self.mma_k
