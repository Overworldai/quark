"""ValueResidualPacked — lerp V columns of packed QKV, copy Q/K.

Warp-per-row, chunked over D_full with double-buffered pipeline.
Per chunk: cp.async curr/first into per-warp smem (overlapped with
previous chunk's compute) → vec_load → compute → vec_build +
vec_store to gmem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import PipelineBody, TensorDecl
from popcorn.blocks.l2.run_pipeline import IterCtx
from popcorn.ir import DType
from popcorn.kernels.base import Kernel
from popcorn.kernels.decorator import kernel
from popcorn.kernels.value_residual_packed.baselines import value_residual_packed_baselines
from popcorn.kernels.value_residual_packed.config import ValueResidualPackedConfig
from popcorn.kernels.value_residual_packed.problems import value_residual_packed_problems
from popcorn.kernels.value_residual_packed.reference import value_residual_packed_reference_for_spec
from popcorn.kernels.value_residual_packed.spec import ValueResidualPackedSpec

_WARP = 32
_CP_BYTES = 16


@dataclass
class _VResStage:
    """Per-stage smem for curr/first activation chunks."""

    curr: object  # SharedRegion
    first: object  # SharedRegion


@kernel(
    "value_residual_packed",
    spec=ValueResidualPackedSpec,
    config=ValueResidualPackedConfig,
    output_idx=-1,
    problems=value_residual_packed_problems,
    baselines=value_residual_packed_baselines,
    reference=value_residual_packed_reference_for_spec,
)
class ValueResidualPackedKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("QKV_curr", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.D_full)),
        TensorDecl("QKV_first", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.D_full)),
        TensorDecl("lamb", dtype=DType.F32, shape=lambda s, c: (1,)),
        TensorDecl(
            "Out", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.D_full), role="out"
        ),
    ]

    spec: ValueResidualPackedSpec
    config: ValueResidualPackedConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if not (1 <= c.n_warps <= 32):
            return False
        if s.D_full % _WARP != 0:
            return False
        if s.M % c.n_warps != 0:
            return False
        vec_elems = _CP_BYTES // s.dtype.bytes
        if s.D_full % vec_elems != 0:
            return False
        chunk_D = c.chunk_D
        if chunk_D <= 0 or s.D_full % chunk_D != 0:
            return False
        if chunk_D % _WARP != 0:
            return False
        min_chunk = _WARP * vec_elems
        if chunk_D < min_chunk:
            return False
        if (chunk_D // vec_elems) // _WARP < 1:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        return (1, self.spec.M // self.config.n_warps, 1)

    def flops(self) -> int:
        return 3 * self.spec.M * self.spec.v_width + self.spec.M * (
            self.spec.D_full - self.spec.v_width
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8], "chunk_D": [256, 512, 1024, 2048, 4096]}

    @classmethod
    def make_tensors(cls, problem: dict) -> dict:
        from popcorn.backend import PT

        spec = ValueResidualPackedSpec(**problem)
        dt = spec.dtype.backend
        return {
            "QKV_curr": PT.astype(PT.randn(spec.M, spec.D_full), dt),
            "QKV_first": PT.astype(PT.randn(spec.M, spec.D_full), dt),
            "lamb": PT.tensor([0.5], dtype=PT.float32),
            "Out": PT.zeros(spec.M, spec.D_full, dtype=dt),
        }

    @classmethod
    def spec_from_tensors(cls, QKV_curr, QKV_first, lamb, *, v_col_offset: int, v_width: int):
        M = int(QKV_curr.shape[0])
        D_full = int(QKV_curr.shape[1])
        dt = DType.from_backend(QKV_curr.dtype)
        return ValueResidualPackedSpec(
            M=M, D_full=D_full, v_col_offset=v_col_offset, v_width=v_width, dtype=dt
        )

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_warps = c.n_warps
        dtype = s.dtype
        D_full = s.D_full
        v_start = s.v_col_offset
        v_end = v_start + s.v_width

        vec_elems = _CP_BYTES // dtype.bytes
        chunk_D = c.chunk_D
        n_chunks = D_full // chunk_D
        loads_per_lane = (chunk_D // vec_elems) // _WARP
        vecs_per_lane = loads_per_lane

        block_base = pop.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row = block_base + bctx.warp_id
        lane = bctx.lane_id
        v_start_c = bctx.c(v_start, dtype=DType.U32)
        v_end_c = bctx.c(v_end, dtype=DType.U32)

        lamb_f = pop.load(g.lamb, bctx.c(0, dtype=DType.U32))

        # ── Smem: double-buffered stages ──
        n_stages = min(2, n_chunks)
        stages = [
            _VResStage(
                curr=pop.smem_alloc(f"C_s{i}", dtype, (n_warps, chunk_D), pad=0),
                first=pop.smem_alloc(f"F_s{i}", dtype, (n_warps, chunk_D), pad=0),
            )
            for i in range(n_stages)
        ]

        # ── Double-buffered pipeline ──
        def produce(ictx: IterCtx) -> None:
            stage = ictx.stage
            col_off = ictx.iter_idx * chunk_D
            for v in range(loads_per_lane):
                local_col = (lane * loads_per_lane + v) * vec_elems
                pop.async_copy(
                    dst=stage.curr,
                    src=g.QKV_curr,
                    dst_idx=[bctx.warp_id, local_col],
                    src_idx=[my_row, col_off + local_col],
                    count=_CP_BYTES,
                )
                pop.async_copy(
                    dst=stage.first,
                    src=g.QKV_first,
                    dst_idx=[bctx.warp_id, local_col],
                    src_idx=[my_row, col_off + local_col],
                    count=_CP_BYTES,
                )

        def consume(ictx: IterCtx) -> tuple:
            stage = ictx.stage
            col_off = ictx.iter_idx * chunk_D
            for v in range(vecs_per_lane):
                vc = (lane * vecs_per_lane + v) * vec_elems
                c_vec = pop.vec_load(stage.curr, bctx.warp_id, vc, width=vec_elems, dtype=dtype)
                f_vec = pop.vec_load(stage.first, bctx.warp_id, vc, width=vec_elems, dtype=dtype)

                abs_col_base = col_off + vc
                out_elems = []
                for j in range(vec_elems):
                    abs_col = abs_col_base + j
                    curr = pop.convert(pop.vec_extract(c_vec, j), DType.F32)
                    in_v = pop.and_(
                        pop.cmp("ge", abs_col, v_start_c), pop.cmp("lt", abs_col, v_end_c)
                    )
                    first = pop.convert(pop.vec_extract(f_vec, j), DType.F32)
                    lerped = pop.fma(lamb_f, first - curr, curr)
                    y = pop.select(in_v, lerped, curr)
                    out_elems.append(pop.convert(y, dtype))

                pop.vec_store(g.Out, pop.vec_build(out_elems), my_row, col_off + vc)
            return ()

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=consume,
        ).run(n_iters=n_chunks, n_stages=n_stages)
