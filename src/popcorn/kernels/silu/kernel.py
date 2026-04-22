"""SiLU — element-wise ``y = x * sigmoid(x)`` via rcp + ex2.

Flat 1D layout: ``N`` total elements, ``elems_per_block / n_threads``
scalars per thread. Same rcp+ex2 fast path as the gemm epilogue.
"""

from __future__ import annotations

import math
from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import TensorDecl
from popcorn.ir import DType
from popcorn.kernels.base import Kernel
from popcorn.kernels.decorator import kernel
from popcorn.kernels.silu.baselines import silu_baselines
from popcorn.kernels.silu.config import SiLUConfig
from popcorn.kernels.silu.problems import silu_problems
from popcorn.kernels.silu.reference import silu_reference_numpy
from popcorn.kernels.silu.spec import SiLUSpec

_LOG2E = math.log2(math.e)


@kernel(
    "silu",
    spec=SiLUSpec,
    config=SiLUConfig,
    output_idx=-1,
    problems=silu_problems,
    baselines=silu_baselines,
    reference=silu_reference_numpy,
)
class SiLUKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("Out", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,), role="out"),
    ]

    spec: SiLUSpec
    config: SiLUConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        n_threads = c.n_warps * 32
        return (
            s.N % c.elems_per_block == 0
            and c.elems_per_block % n_threads == 0
            and 1 <= c.n_warps <= 32
        )

    def grid(self) -> tuple[int, int, int]:
        return (1, self.spec.N // self.config.elems_per_block, 1)

    def flops(self) -> int:
        return 5 * self.spec.N

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8], "elems_per_block": [128, 256, 512, 1024, 2048]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from popcorn.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = SiLUSpec(**problem)
        rng = np.random.default_rng(seed)
        x_f32 = rng.standard_normal(spec.N).astype(np.float32)
        return {
            "X": astype_numpy(x_f32, spec.dtype),
            "Out": zeros_for_dtype((spec.N,), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(cls, X) -> SiLUSpec:
        N = 1
        for d in X.shape:
            N *= int(d)
        return SiLUSpec(N=N, dtype=DType.from_backend(X.dtype))

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * 32
        epb = c.elems_per_block
        epl = epb // n_threads
        dtype = s.dtype

        block = pop.block_idx("y")
        base = block * bctx.c(epb, dtype=DType.U32)
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)

        log2e = bctx.c(_LOG2E, dtype=DType.F32)
        one_f = bctx.c(1.0, dtype=DType.F32)

        for i in range(epl):
            off = base + bctx.c(i) * n_threads_c + bctx.tid
            x_f = pop.convert(g.X[off], DType.F32)
            sig = pop.rcp_approx(one_f + pop.ex2_approx(pop.neg(x_f * log2e)))
            y = x_f * sig
            g.Out[off] = pop.convert(y, dtype) if dtype is not DType.F32 else y
