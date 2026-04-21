"""CopyStridedSpec — gather from a strided source into contiguous dst.

Used by PopcornTensor.contiguous() when the tensor is a non-contiguous
view (from slicing or permute). The kernel reads from
``src_base + linearize(flat_idx, shape, src_strides) * elem_bytes``
and writes to ``dst[flat_idx]``.

We support up to 4 dimensions. Higher-rank tensors are flattened to
4D by merging leading dims.
"""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec

_VALID_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32, DType.S32, DType.U8, DType.S8})


@dataclass(frozen=True)
class CopyStridedSpec(KernelSpec):
    """Copy N elements from a strided source to a contiguous dest.

    The source layout is described by up to 4 (shape, stride) pairs.
    ``ndim`` is the actual number of dimensions (1-4).
    ``offset`` is the element offset into the source buffer.
    """

    N: int
    dtype: DType = DType.BF16
    ndim: int = 2
    shape0: int = 1
    shape1: int = 1
    shape2: int = 1
    shape3: int = 1
    stride0: int = 0
    stride1: int = 1
    stride2: int = 1
    stride3: int = 1
    offset: int = 0

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.N <= 0:
            raise ValueError(f"CopyStridedSpec: N must be positive; got {self.N}")
