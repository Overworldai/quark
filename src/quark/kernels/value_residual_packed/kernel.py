"""ValueResidualPacked — lerp V columns of packed QKV, copy Q/K.

Warp-per-row, chunked over D_full with double-buffered pipeline.
Per chunk: cp.async curr/first into per-warp smem (overlapped with
previous chunk's compute) → vec_load → compute → vec_build +
vec_store to gmem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import quark.lang as qk
from quark.blocks import PipelineBody, TensorDecl
from quark.blocks.l2.run_pipeline import IterCtx
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.value_residual_packed.baselines import value_residual_packed_baselines
from quark.kernels.value_residual_packed.config import ValueResidualPackedConfig
from quark.kernels.value_residual_packed.problems import value_residual_packed_problems
from quark.kernels.value_residual_packed.reference import value_residual_packed_reference_numpy
from quark.kernels.value_residual_packed.spec import ValueResidualPackedSpec

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
    reference=value_residual_packed_reference_numpy,
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
        if s.D_full % self._sgs != 0:
            return False
        if s.M % c.n_warps != 0:
            return False
        vec_elems = _CP_BYTES // s.dtype.bytes
        if s.D_full % vec_elems != 0:
            return False
        chunk_D = c.chunk_D
        if chunk_D <= 0 or s.D_full % chunk_D != 0:
            return False
        if chunk_D % self._sgs != 0:
            return False
        min_chunk = self._sgs * vec_elems
        if chunk_D < min_chunk:
            return False
        if (chunk_D // vec_elems) // self._sgs < 1:
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
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = ValueResidualPackedSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "QKV_curr": astype_numpy(
                rng.standard_normal((spec.M, spec.D_full)).astype(np.float32), spec.dtype
            ),
            "QKV_first": astype_numpy(
                rng.standard_normal((spec.M, spec.D_full)).astype(np.float32), spec.dtype
            ),
            "lamb": np.array([0.5], dtype=np.float32),
            "Out": zeros_for_dtype((spec.M, spec.D_full), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(cls, QKV_curr, QKV_first, lamb, *, v_col_offset: int, v_width: int):
        M = int(QKV_curr.shape[0])
        D_full = int(QKV_curr.shape[1])
        dt = DType.from_backend(QKV_curr)
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
        loads_per_lane = (chunk_D // vec_elems) // self._sgs
        vecs_per_lane = loads_per_lane

        block_base = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row = block_base + bctx.warp_id
        lane = bctx.lane_id
        v_start_c = bctx.c(v_start, dtype=DType.U32)
        v_end_c = bctx.c(v_end, dtype=DType.U32)

        lamb_f = qk.load(g.lamb, bctx.c(0, dtype=DType.U32))

        # ── Smem: double-buffered stages ──
        n_stages = min(2, n_chunks)
        stages = [
            _VResStage(
                curr=qk.smem_alloc(f"C_s{i}", dtype, (n_warps, chunk_D), pad=0),
                first=qk.smem_alloc(f"F_s{i}", dtype, (n_warps, chunk_D), pad=0),
            )
            for i in range(n_stages)
        ]

        # ── Double-buffered pipeline ──
        def produce(ictx: IterCtx) -> None:
            stage = ictx.stage
            col_off = ictx.iter_idx * chunk_D
            for v in range(loads_per_lane):
                local_col = (lane * loads_per_lane + v) * vec_elems
                qk.async_copy(
                    dst=stage.curr,
                    src=g.QKV_curr,
                    dst_idx=[bctx.warp_id, local_col],
                    src_idx=[my_row, col_off + local_col],
                    count=_CP_BYTES,
                )
                qk.async_copy(
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
                c_vec = qk.vec_load(stage.curr, bctx.warp_id, vc, width=vec_elems, dtype=dtype)
                f_vec = qk.vec_load(stage.first, bctx.warp_id, vc, width=vec_elems, dtype=dtype)

                abs_col_base = col_off + vc
                out_elems = []
                for j in range(vec_elems):
                    abs_col = abs_col_base + j
                    curr = qk.convert(qk.vec_extract(c_vec, j), DType.F32)
                    in_v = qk.and_(qk.cmp("ge", abs_col, v_start_c), qk.cmp("lt", abs_col, v_end_c))
                    first = qk.convert(qk.vec_extract(f_vec, j), DType.F32)
                    lerped = qk.fma(lamb_f, first - curr, curr)
                    y = qk.select(in_v, lerped, curr)
                    out_elems.append(qk.convert(y, dtype))

                qk.vec_store(g.Out, qk.vec_build(out_elems), my_row, col_off + vc)
            return ()

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=consume,
        ).run(n_iters=n_chunks, n_stages=n_stages)
