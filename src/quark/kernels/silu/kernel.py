"""SiLU — element-wise ``y = x * sigmoid(x)`` via rcp + ex2.

Flat 1D layout: ``N`` total elements, ``elems_per_block / n_threads``
scalars per thread. Same rcp+ex2 fast path as the gemm epilogue.
"""

from __future__ import annotations

import math
from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.silu.baselines import silu_baselines
from quark.kernels.silu.config import SiLUConfig
from quark.kernels.silu.problems import silu_problems
from quark.kernels.silu.reference import silu_reference_numpy
from quark.kernels.silu.spec import SiLUSpec

_LOG2E = math.log2(math.e)  # kept for backward compat; silu now uses exp_approx


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
        return (self.spec.N // self.config.elems_per_block, 1, 1)

    def flops(self) -> int:
        return 5 * self.spec.N

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8], "elems_per_block": [128, 256, 512, 1024, 2048]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

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
        return SiLUSpec(N=N, dtype=DType.from_backend(X))

    def _silu_body(self, *, use_exp: bool) -> None:
        """Shared body for ``build`` and ``build_metal``. When
        ``use_exp=True`` emits ``metal::fast::exp(-x)`` (one op);
        otherwise ``metal::fast::exp2(-x * log2e)`` (two ops, needed
        on CUDA where exp is not a single SFU instruction)."""
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * 32
        epb = c.elems_per_block
        epl = epb // n_threads
        dtype = s.dtype

        if epl % 4 == 0:
            vec_w = 4
        elif epl % 2 == 0:
            vec_w = 2
        else:
            vec_w = 1
        vecs_per_lane = epl // vec_w

        block = qk.block_idx("x")
        base = block * bctx.c(epb, dtype=DType.U32)
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)
        vec_w_c = bctx.c(vec_w, dtype=DType.U32)

        one_f = bctx.c(1.0, dtype=DType.F32)
        log2e = bctx.c(_LOG2E, dtype=DType.F32) if not use_exp else None

        def _sigmoid(x_f):
            if use_exp:
                # metal::fast::exp(-x) → one op; skip the mul by log2e.
                return qk.rcp_approx(one_f + qk.exp_approx(qk.neg(x_f)))
            # CUDA: exp2(-x * log2e) — PTX has no native exp, needs 2 ops.
            return qk.rcp_approx(one_f + qk.ex2_approx(qk.neg(x_f * log2e)))

        if vec_w == 1:
            for i in range(epl):
                off = base + bctx.c(i) * n_threads_c + bctx.tid
                x_f = qk.convert(g.X[off], DType.F32)
                y = x_f * _sigmoid(x_f)
                g.Out[off] = qk.convert(y, dtype) if dtype is not DType.F32 else y
            return

        for v in range(vecs_per_lane):
            v_off = base + (bctx.c(v) * n_threads_c + bctx.tid) * vec_w_c
            x_vec = qk.vec_load(g.X, v_off, width=vec_w, dtype=dtype)
            out_elems = []
            for j in range(vec_w):
                x_f = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                y = x_f * _sigmoid(x_f)
                out_elems.append(qk.convert(y, dtype) if dtype is not DType.F32 else y)
            qk.vec_store(g.Out, qk.vec_build(out_elems), v_off)

    def build(self) -> None:
        """CUDA / fallback path: ``exp2(-x * log2e)``."""
        self._silu_body(use_exp=False)

    def build_metal(self) -> None:
        """Metal path: ``metal::fast::exp(-x)`` — one op vs two."""
        self._silu_body(use_exp=True)
