"""Euler step — ``x = x + dsig * v`` for the denoise ODE.

Fused elementwise over N total elements:
    y[i] = cast(f32(x[i]) + dsig * f32(v[i]), x.dtype)

``dsig`` is a runtime scalar (1-element f32 tensor) so the denoise
loop can keep stepping without recompiling per sigma_idx. Keeps the
step on-device (no numpy round-trip in generate.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import TensorDecl
from popcorn.ir import DType
from popcorn.kernels.base import Kernel, KernelConfig, KernelSpec
from popcorn.kernels.decorator import kernel

_VALID_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32})


@dataclass(frozen=True)
class EulerStepSpec(KernelSpec):
    N: int
    dtype: DType = DType.BF16

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"EulerStepSpec: dtype {self.dtype!r} not in {_VALID_DTYPES}")
        if self.N <= 0:
            raise ValueError(f"EulerStepSpec: N must be positive; got {self.N}")


@dataclass(frozen=True)
class EulerStepConfig(KernelConfig):
    n_warps: int = 4
    elems_per_block: int = 1024

    @classmethod
    def default_for(cls, spec) -> EulerStepConfig:
        for epb in (1024, 512, 256, 128, 64, 32):
            if spec.N % epb == 0 and epb % 32 == 0:
                n_warps = min(4, epb // 32)
                if epb % (n_warps * 32) == 0:
                    return cls(n_warps=n_warps, elems_per_block=epb)
        return cls(n_warps=4, elems_per_block=1024)


def _reference(kernel, X, V, Dsig):
    from popcorn.backend import PT

    x_f = PT.astype(X, PT.float32)
    v_f = PT.astype(V, PT.float32)
    dsig_scalar = Dsig[0] if hasattr(Dsig, "__getitem__") else Dsig
    out = x_f + dsig_scalar * v_f
    return PT.astype(out, X.dtype)


@kernel(
    "euler_step",
    spec=EulerStepSpec,
    config=EulerStepConfig,
    output_idx=-1,
    problems=lambda: [],
    baselines=lambda: [],
    reference=_reference,
)
class EulerStepKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("V", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("Dsig", dtype=DType.F32, shape=lambda s, c: (1,)),
        TensorDecl("Out", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,), role="out"),
    ]

    spec: EulerStepSpec
    config: EulerStepConfig

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
        return 2 * self.spec.N  # mul + add per element

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8], "elems_per_block": [128, 256, 512, 1024, 2048]}

    @classmethod
    def make_tensors(cls, problem: dict) -> dict:
        from popcorn.backend import PT

        spec = EulerStepSpec(**problem)
        dt = spec.dtype.backend
        return {
            "X": PT.astype(PT.randn(spec.N), dt),
            "V": PT.astype(PT.randn(spec.N), dt),
            "Dsig": PT.tensor([0.1], dtype=PT.float32),
            "Out": PT.zeros(spec.N, dtype=dt),
        }

    @classmethod
    def spec_from_tensors(cls, X, V, Dsig) -> EulerStepSpec:
        N = 1
        for d in X.shape:
            N *= int(d)
        return EulerStepSpec(N=N, dtype=DType.from_backend(X.dtype))

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * 32
        epb = c.elems_per_block
        dtype = s.dtype

        block = pop.block_idx("y")
        base = block * bctx.c(epb, dtype=DType.U32)

        # Broadcast the scalar dsig (1-elem f32 tensor) to every thread.
        dsig = pop.load(g.Dsig, bctx.c(0, dtype=DType.U32))

        # Vectorized path — one 16 B transaction per load/store. Matches
        # the cast kernel's fast path in ``elementwise``. ``is_valid``
        # guarantees ``epb % n_threads == 0`` and ``spec.N % epb == 0``,
        # so ``epb`` is a clean multiple of the 16 B vector width as long
        # as ``epb >= n_threads * vec_elems`` (the small-epb fallback
        # branches to the scalar path below).
        vec_elems = 16 // dtype.bytes if dtype.bytes in (1, 2, 4, 8) else 0
        n_vecs_per_block = epb // vec_elems if vec_elems else 0
        if vec_elems > 0 and n_vecs_per_block > 0 and n_vecs_per_block % n_threads == 0:
            iters_per_thread = n_vecs_per_block // n_threads
            vec_elems_c = bctx.c(vec_elems, dtype=DType.U32)
            for i in range(iters_per_thread):
                chunk_off = bctx.c(i * n_threads * vec_elems, dtype=DType.U32)
                off = base + chunk_off + bctx.tid * vec_elems_c
                x_vec = pop.vec_load(g.X, off, width=vec_elems, dtype=dtype)
                v_vec = pop.vec_load(g.V, off, width=vec_elems, dtype=dtype)
                out_elems = []
                for j in range(vec_elems):
                    x_f = pop.convert(pop.vec_extract(x_vec, j), DType.F32)
                    v_f = pop.convert(pop.vec_extract(v_vec, j), DType.F32)
                    y_f = x_f + dsig * v_f
                    out_elems.append(pop.convert(y_f, dtype) if dtype is not DType.F32 else y_f)
                pop.vec_store(g.Out, pop.vec_build(out_elems), off)
            return

        # Scalar fallback for tiny ``epb`` that doesn't tile-align with
        # the 16 B vector width at this thread count.
        epl = epb // n_threads
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)
        for i in range(epl):
            off = base + bctx.c(i) * n_threads_c + bctx.tid
            x_f = pop.convert(g.X[off], DType.F32)
            v_f = pop.convert(g.V[off], DType.F32)
            y_f = x_f + dsig * v_f
            g.Out[off] = pop.convert(y_f, dtype) if dtype is not DType.F32 else y_f
