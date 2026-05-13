"""CopyStrided — gather from strided source into contiguous output.

Flat 1D layout: N total elements, each thread computes the strided
source index from its flat output index and copies one element.
Handles up to 4 dimensions via stride parameters passed as scalars.

The index computation for flat index ``i`` into an N-D strided tensor:
    i3 = i % shape3;  i = i / shape3
    i2 = i % shape2;  i = i / shape2
    i1 = i % shape1;  i = i / shape1
    i0 = i
    src_idx = offset + i0*stride0 + i1*stride1 + i2*stride2 + i3*stride3
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.copy_strided.config import CopyStridedConfig
from quark.kernels.copy_strided.spec import CopyStridedSpec
from quark.kernels.decorator import kernel


@kernel(
    "copy_strided",
    spec=CopyStridedSpec,
    config=CopyStridedConfig,
    output_idx=-1,
    problems=lambda: [],
    baselines=lambda: [],
    reference=lambda spec, *, Src, Dst=None: Src,  # identity (numpy returns input unchanged)
)
class CopyStridedKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        # Src is the strided input — we declare it as flat [N] for the
        # param spec, but the kernel indexes into it with computed offsets.
        # The actual buffer may be larger than N elements (the view is a
        # subset of a larger allocation).
        TensorDecl("Src", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("Dst", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,), role="out"),
    ]

    spec: CopyStridedSpec
    config: CopyStridedConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        n_threads = c.n_warps * self._sgs
        return (
            s.N % c.elems_per_block == 0
            and c.elems_per_block % n_threads == 0
            and 1 <= c.n_warps <= 32
        )

    def grid(self) -> tuple[int, int, int]:
        return (1, self.spec.N // self.config.elems_per_block, 1)

    def flops(self) -> int:
        return 0  # pure memory op

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4], "elems_per_block": [128, 256, 512, 1024]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = CopyStridedSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "Src": astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.dtype),
            "Dst": zeros_for_dtype((spec.N,), spec.dtype),
        }

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * self._sgs
        epb = c.elems_per_block
        epl = epb // n_threads

        block = qk.block_idx("y")
        base = block * bctx.c(epb, dtype=DType.U32)
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)

        # Stride constants (element strides, not byte strides).
        c_shape3 = bctx.c(s.shape3, dtype=DType.U32)
        c_shape2 = bctx.c(s.shape2, dtype=DType.U32)
        c_shape1 = bctx.c(s.shape1, dtype=DType.U32)
        c_stride0 = bctx.c(s.stride0, dtype=DType.U32)
        c_stride1 = bctx.c(s.stride1, dtype=DType.U32)
        c_stride2 = bctx.c(s.stride2, dtype=DType.U32)
        c_stride3 = bctx.c(s.stride3, dtype=DType.U32)
        c_offset = bctx.c(s.offset, dtype=DType.U32)

        for i in range(epl):
            flat = base + bctx.c(i) * n_threads_c + bctx.tid

            # Decompose flat index into N-D coordinates.
            # Work from innermost (dim 3) outward.
            rem = flat
            if s.ndim >= 4:
                i3 = rem % c_shape3
                rem = rem // c_shape3
            else:
                i3 = bctx.c(0, dtype=DType.U32)
            if s.ndim >= 3:
                i2 = rem % c_shape2
                rem = rem // c_shape2
            else:
                i2 = bctx.c(0, dtype=DType.U32)
            if s.ndim >= 2:
                i1 = rem % c_shape1
                rem = rem // c_shape1
            else:
                i1 = bctx.c(0, dtype=DType.U32)
            i0 = rem

            # Compute strided source index.
            src_idx = c_offset + i0 * c_stride0 + i1 * c_stride1 + i2 * c_stride2 + i3 * c_stride3

            # Copy element.
            val = g.Src[src_idx]
            g.Dst[flat] = val
