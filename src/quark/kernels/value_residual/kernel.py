"""ValueResidual — fused ``out = v + lamb * (v1 - v)`` element-wise.

Rank-1 layout: every tensor is viewed as ``[N]``. Grid is 1D over
``N // elems_per_block`` blocks, each block has ``n_warps*32`` threads
and each thread handles ``elems_per_block / n_threads`` scalars.

``lamb`` is a 1-element device tensor (loaded once per block into a
register shared across the inner loop).
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.value_residual.baselines import value_residual_baselines
from quark.kernels.value_residual.config import ValueResidualConfig
from quark.kernels.value_residual.problems import value_residual_problems
from quark.kernels.value_residual.reference import value_residual_reference_numpy
from quark.kernels.value_residual.spec import ValueResidualSpec


@kernel(
    "value_residual",
    spec=ValueResidualSpec,
    config=ValueResidualConfig,
    output_idx=-1,
    problems=value_residual_problems,
    baselines=value_residual_baselines,
    reference=value_residual_reference_numpy,
)
class ValueResidualKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("V", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("V1", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("lamb", dtype=DType.F32, shape=lambda s, c: (1,)),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.N,),
            role="out",
        ),
    ]

    spec: ValueResidualSpec
    config: ValueResidualConfig

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
        return 3 * self.spec.N  # sub + mul + add

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            "n_warps": [1, 2, 4, 8],
            "elems_per_block": [256, 512, 1024, 2048],
        }

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = ValueResidualSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "V": astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.dtype),
            "V1": astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.dtype),
            "lamb": np.array([0.35], dtype=np.float32),  # arbitrary per-layer mix
            "Out": zeros_for_dtype((spec.N,), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(cls, V, V1, lamb) -> ValueResidualSpec:
        if V.shape != V1.shape:
            raise ValueError(f"value_residual: V {V.shape} != V1 {V1.shape}")
        if int(lamb.shape[0]) != 1:
            raise ValueError(f"value_residual: lamb must be [1]; got {lamb.shape}")
        N = 1
        for d in V.shape:
            N *= int(d)
        dt = DType.from_backend(V)
        return ValueResidualSpec(N=N, dtype=dt)

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_warps = c.n_warps
        n_threads = n_warps * 32
        epb = c.elems_per_block
        epl = epb // n_threads
        dtype = s.dtype

        # Block-wide base offset into the flat tensor.
        block = qk.block_idx("y")
        base = block * bctx.c(epb, dtype=DType.U32)

        # Load lamb once per block (broadcast across all threads).
        lamb_f = qk.load(g.lamb, bctx.c(0, dtype=DType.U32))
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)

        for i in range(epl):
            off = base + bctx.c(i) * n_threads_c + bctx.tid
            v = qk.convert(g.V[off], DType.F32)
            v1 = qk.convert(g.V1[off], DType.F32)
            # out = v + lamb * (v1 - v)
            out = qk.fma(lamb_f, v1 - v, v)
            g.Out[off] = qk.convert(out, dtype) if dtype is not DType.F32 else out
