"""Elementwise kernel — one parametric kernel for all element-wise ops.

Flat 1D layout matching the SiLU kernel pattern: ``N`` total elements,
``elems_per_block / n_threads`` scalars per thread, grid-stride-loop
over blocks. Reduction in f32; output cast back to the input dtype.

Binary ops (add, sub, mul, div): two inputs ``X``, ``Y``, one output.
Unary ops (neg, abs, exp, sin, cos, sqrt): one input ``X``, one output.
Cast: one input ``X``, one output at a different dtype.

For unary ops, ``Y`` is declared as a dummy 1-element tensor and
ignored in ``build()``. This matches the ``has_bias=False`` pattern
used by GEMM — the dummy buffer is auto-allocated and cached on the
CompiledKernel so its data_ptr stays stable.
"""

from __future__ import annotations

import math
from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import TensorDecl
from popcorn.ir import DType
from popcorn.kernels.base import Kernel
from popcorn.kernels.decorator import kernel
from popcorn.kernels.elementwise.config import ElementwiseConfig
from popcorn.kernels.elementwise.problems import elementwise_problems
from popcorn.kernels.elementwise.reference import elementwise_reference_for_spec
from popcorn.kernels.elementwise.spec import BINARY_OPS, ElementwiseSpec

_LOG2E = math.log2(math.e)


def _y_shape(s, c):
    """Y is full-size for binary ops, dummy (1,) for unary/cast."""
    if s.op in BINARY_OPS:
        return (s.N,)
    return (1,)


@kernel(
    "elementwise",
    spec=ElementwiseSpec,
    config=ElementwiseConfig,
    output_idx=-1,
    problems=elementwise_problems,
    baselines=lambda: [],
    reference=elementwise_reference_for_spec,
)
class ElementwiseKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("Y", dtype=lambda s, c: s.dtype, shape=_y_shape),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.effective_out_dtype,
            shape=lambda s, c: (s.N,),
            role="out",
        ),
    ]

    spec: ElementwiseSpec
    config: ElementwiseConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if c.n_warps < 1 or c.n_warps > 32:
            return False
        # Only real constraint: N must be a whole number of 16 B vectors
        # for the fast-path ``vec_load`` / ``vec_store`` to be aligned.
        # ``vec_elems`` is capped so a width-``vec_elems`` vec_store of
        # the *output* dtype also fits in one 16 B transaction — for
        # casts where src and dst byte widths differ we use the wider
        # one to size the chunk.
        max_bytes = max(
            s.dtype.bytes,
            s.effective_out_dtype.bytes if s.op == "cast" else s.dtype.bytes,
        )
        vec_elems = 16 // max_bytes if max_bytes in (1, 2, 4, 8) else 1
        if s.N % vec_elems != 0:
            return False
        return s.N > 0

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        # ceil(N / epb). Last block handles the partial tail via
        # bounds-checked predication inside ``build``.
        n_blocks_y = (s.N + c.elems_per_block - 1) // c.elems_per_block
        return (1, n_blocks_y, 1)

    def flops(self) -> int:
        return self.spec.N * (2 if self.spec.arity == 2 else 1)

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8], "elems_per_block": [128, 256, 512, 1024, 2048]}

    @classmethod
    def make_tensors(cls, problem: dict) -> dict:
        from popcorn.backend import PT

        spec = ElementwiseSpec(**problem)
        dt = spec.dtype.backend
        tensors = {
            "X": PT.astype(PT.randn(spec.N), dt),
            "Out": PT.zeros(spec.N, dtype=spec.effective_out_dtype.backend),
        }
        if spec.arity == 2:
            tensors["Y"] = PT.astype(PT.randn(spec.N), dt)
        else:
            tensors["Y"] = PT.zeros(1, dtype=dt)
        return tensors

    @classmethod
    def spec_from_tensors(cls, X, Y=None, *, op: str = "add", out_dtype=None) -> ElementwiseSpec:
        N = 1
        for d in X.shape:
            N *= int(d)
        dt = DType.from_backend(X.dtype)
        out_dt = DType.from_backend(out_dtype) if out_dtype is not None else None
        return ElementwiseSpec(N=N, dtype=dt, op=op, out_dtype=out_dt)

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * 32
        epb = c.elems_per_block
        out_dtype = s.effective_out_dtype
        op = s.op

        block = pop.block_idx("y")
        base = block * bctx.c(epb, dtype=DType.U32)
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)
        N_c = bctx.c(s.N, dtype=DType.U32)

        is_int = s.dtype in (DType.S32, DType.U32)

        # Single compute step applied to one scalar (or a pair, for
        # binary ops). Works on both native-dtype ints and f32 floats;
        # the caller picks which representation to hand it.
        def _compute(x_scalar, y_scalar):
            if is_int:
                if op in BINARY_OPS:
                    if op == "add":
                        return x_scalar + y_scalar
                    if op == "sub":
                        return x_scalar - y_scalar
                    if op == "mul":
                        return x_scalar * y_scalar
                    raise ValueError(f"integer op {op} not supported")
                if op == "neg":
                    return pop.neg(x_scalar)
                raise ValueError(f"integer op {op} not supported")
            x_f = pop.convert(x_scalar, DType.F32)
            if op == "cast":
                return x_f
            if op in BINARY_OPS:
                y_f = pop.convert(y_scalar, DType.F32)
                if op == "add":
                    return x_f + y_f
                if op == "sub":
                    return x_f - y_f
                if op == "mul":
                    return x_f * y_f
                if op == "div":
                    return x_f * pop.rcp_approx(y_f)
                raise ValueError(f"unknown binary op: {op}")
            if op == "neg":
                return pop.neg(x_f)
            if op == "abs":
                return pop.abs_(x_f)
            if op == "exp":
                log2e = bctx.c(_LOG2E, dtype=DType.F32)
                return pop.ex2_approx(x_f * log2e)
            if op == "sin":
                return pop.sin(x_f)
            if op == "cos":
                return pop.cos(x_f)
            if op == "sqrt":
                return pop.sqrt(x_f)
            raise ValueError(f"unknown op: {op}")

        def _cast_out(r):
            """Cast the compute-domain result back to out_dtype for store."""
            if is_int:
                return r  # compute stays in native int dtype
            return pop.convert(r, out_dtype) if out_dtype is not DType.F32 else r

        # ── Vectorized fast path ───────────────────────────────────
        # One 16 B load/store per thread. Covers every op (cast +
        # binary + unary) when the per-element dtype is 1/2/4/8 B.
        # Bounds-check the chunk start against N so tail blocks
        # (``N % epb != 0``) mask their overrun threads.
        src_b = s.dtype.bytes
        dst_b = out_dtype.bytes
        max_b = max(src_b, dst_b)
        vec_elems = 16 // max_b if max_b in (1, 2, 4, 8) else 0
        if vec_elems > 0:
            n_vecs_per_block = epb // vec_elems
            iters_per_thread = max(1, (n_vecs_per_block + n_threads - 1) // n_threads)
            vec_elems_c = bctx.c(vec_elems, dtype=DType.U32)
            is_binary = op in BINARY_OPS
            for i in range(iters_per_thread):
                chunk_off = bctx.c(i * n_threads * vec_elems, dtype=DType.U32)
                off = base + chunk_off + bctx.tid * vec_elems_c
                # Vec chunks are vec_elems-aligned and N is too (per
                # is_valid), so ``off < N`` is a sufficient bounds check.
                pred = pop.cmp("lt", off, N_c)
                x_vec = pop.vec_load(g.X, off, width=vec_elems, dtype=s.dtype, pred=pred)
                y_vec = (
                    pop.vec_load(g.Y, off, width=vec_elems, dtype=s.dtype, pred=pred)
                    if is_binary
                    else None
                )
                out_elems = [
                    _cast_out(
                        _compute(
                            pop.vec_extract(x_vec, j),
                            pop.vec_extract(y_vec, j) if is_binary else None,
                        )
                    )
                    for j in range(vec_elems)
                ]
                pop.vec_store(g.Out, pop.vec_build(out_elems), off, pred=pred)
            return

        # ── Scalar fallback (tiny N or exotic dtype widths) ────────
        epl_iters = max(1, (epb + n_threads - 1) // n_threads)
        for i in range(epl_iters):
            off = base + bctx.c(i) * n_threads_c + bctx.tid
            pred = pop.cmp("lt", off, N_c)
            x = pop.load(g.X, off, pred=pred)
            y = pop.load(g.Y, off, pred=pred) if op in BINARY_OPS else None
            r = _compute(x, y)
            pop.store(g.Out, _cast_out(r), off, pred=pred)
