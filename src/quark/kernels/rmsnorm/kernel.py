"""RMSNorm — per-row ``y = x * rsqrt(mean(x²) + eps)``.

Two modes based on D size:

Large D (D >= warp * vec_elems): warp-per-row, chunked. Each lane
issues cp.async from its warp's gmem row. Standard pattern.

Small D (D < warp * vec_elems, e.g. Dh=64): all threads cooperate
to load many rows at once via cp.async into flat smem. Each warp
processes rows sequentially with full 32-lane reduction.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.rmsnorm.baselines import rmsnorm_baselines
from quark.kernels.rmsnorm.config import RMSNormConfig
from quark.kernels.rmsnorm.problems import rmsnorm_problems
from quark.kernels.rmsnorm.reference import rmsnorm_reference_numpy
from quark.kernels.rmsnorm.spec import RMSNormSpec

_WARP = 32
_CP_BYTES = 16


@kernel(
    "rmsnorm",
    spec=RMSNormSpec,
    config=RMSNormConfig,
    output_idx=-1,
    problems=rmsnorm_problems,
    baselines=rmsnorm_baselines,
    reference=rmsnorm_reference_numpy,
)
class RMSNormKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.B, s.D)),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.B, s.D),
            role="out",
        ),
    ]

    spec: RMSNormSpec
    config: RMSNormConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if c.n_warps < 1 or c.n_warps > 32:
            return False
        if s.D % _WARP != 0:
            return False
        vec_elems = _CP_BYTES // s.dtype.bytes
        if s.D % vec_elems != 0:
            return False
        n_threads = c.n_warps * _WARP
        # Large-D mode: warp-per-row.
        if _WARP * vec_elems <= s.D:
            return s.B % c.n_warps == 0
        # Small-D mode: cooperative multi-row load.
        # rows_per_load = n_threads * vec_elems / D — must be integer.
        total_elems_per_load = n_threads * vec_elems
        if total_elems_per_load % s.D != 0:
            return False
        rows_per_load = total_elems_per_load // s.D
        # rows_per_load must be divisible by n_warps (each warp gets equal rows).
        if rows_per_load % c.n_warps != 0:
            return False
        # B must be divisible by rows_per_load.
        if s.B % rows_per_load != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        vec_elems = _CP_BYTES // s.dtype.bytes
        if _WARP * vec_elems <= s.D:
            return (1, s.B // c.n_warps, 1)
        n_threads = c.n_warps * _WARP
        rows_per_load = (n_threads * vec_elems) // s.D
        return (1, s.B // rows_per_load, 1)

    def flops(self) -> int:
        return 4 * self.spec.B * self.spec.D

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = RMSNormSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "X": astype_numpy(rng.standard_normal((spec.B, spec.D)).astype(np.float32), spec.dtype),
            "Out": zeros_for_dtype((spec.B, spec.D), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(cls, X, *, eps: float = 1.1920929e-07) -> RMSNormSpec:
        if X.ndim < 2:
            raise ValueError(f"rmsnorm: X must be rank≥2; got {X.shape}")
        B = 1
        for d in X.shape[:-1]:
            B *= int(d)
        D = int(X.shape[-1])
        dt = DType.from_backend(X.dtype)
        return RMSNormSpec(B=B, D=D, dtype=dt, eps=eps)

    def build(self) -> None:
        s, c = self.spec, self.config
        bctx = self.bctx

        D = s.D
        n_warps = c.n_warps
        n_threads = n_warps * _WARP
        dtype = s.dtype
        epl = D // _WARP
        vec_elems = _CP_BYTES // dtype.bytes
        min_chunk = _WARP * vec_elems  # 256 for bf16

        lane = bctx.lane_id
        warp_c = bctx.c(_WARP, dtype=DType.U32)

        if min_chunk <= D:
            self._build_large_d(
                D, n_warps, n_threads, dtype, epl, vec_elems, min_chunk, lane, warp_c
            )
        else:
            self._build_small_d(D, n_warps, n_threads, dtype, epl, vec_elems, lane, warp_c)

    def _build_large_d(self, D, n_warps, n_threads, dtype, epl, vec_elems, min_chunk, lane, warp_c):
        """Warp-per-row, double-buffered pipeline. No register stash —
        X is read twice from gmem (L2-hot on second pass)."""
        from dataclasses import dataclass

        from quark.blocks import PipelineBody
        from quark.blocks.l2.run_pipeline import IterCtx

        g = self.g
        bctx = self.bctx

        n_chunks = max(1, D // min_chunk)
        while n_chunks > 1 and D % n_chunks != 0:
            n_chunks -= 1
        chunk_D = D // n_chunks
        loads_per_lane = (chunk_D // vec_elems) // _WARP
        vecs_per_lane = loads_per_lane

        block_base = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row = block_base + bctx.warp_id

        n_stages = min(2, n_chunks)

        @dataclass
        class _Stage:
            X: object

        x_stages = [
            _Stage(X=qk.smem_alloc(f"X_s{i}", dtype, (n_warps, chunk_D), pad=0))
            for i in range(n_stages)
        ]

        def produce_x(ictx: IterCtx) -> None:
            stage = ictx.stage
            col_off = ictx.iter_idx * chunk_D
            for v in range(loads_per_lane):
                local_col = (lane * loads_per_lane + v) * vec_elems
                qk.async_copy(
                    dst=stage.X,
                    src=g.X,
                    dst_idx=[bctx.warp_id, local_col],
                    src_idx=[my_row, col_off + local_col],
                    count=_CP_BYTES,
                )

        # ── Pass 1: pipelined reduction ──
        init_sum = bctx.c(0.0, dtype=DType.F32)

        def consume_p1(ictx: IterCtx) -> tuple:
            stage = ictx.stage
            (local_sum,) = ictx.carry
            for v in range(vecs_per_lane):
                vc = (lane * vecs_per_lane + v) * vec_elems
                x_vec = qk.vec_load(stage.X, bctx.warp_id, vc, width=vec_elems, dtype=dtype)
                for j in range(vec_elems):
                    x = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                    local_sum = qk.fma(x, x, local_sum)
            return (local_sum,)

        p1_result = PipelineBody(
            stages=x_stages,
            produce=produce_x,
            consume=consume_p1,
            carry=(init_sum,),
        ).run(n_iters=n_chunks, n_stages=n_stages)

        (final_sum,) = p1_result
        total = qk.subgroup_reduce("sum", final_sum)
        rms_inv = qk.rsqrt_approx(
            total * bctx.c(1.0 / D, dtype=DType.F32) + bctx.c(self.spec.eps, dtype=DType.F32)
        )

        # ── Pass 2: pipelined rescale (X re-read, L2-hot) ──
        def consume_p2(ictx: IterCtx) -> tuple:
            stage = ictx.stage
            col_off = ictx.iter_idx * chunk_D
            for v in range(vecs_per_lane):
                vc = (lane * vecs_per_lane + v) * vec_elems
                x_vec = qk.vec_load(stage.X, bctx.warp_id, vc, width=vec_elems, dtype=dtype)
                out_elems = []
                for j in range(vec_elems):
                    x = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                    y = x * rms_inv
                    out_elems.append(qk.convert(y, dtype))
                qk.vec_store(g.Out, qk.vec_build(out_elems), my_row, col_off + vc)
            return ()

        PipelineBody(
            stages=x_stages,
            produce=produce_x,
            consume=consume_p2,
        ).run(n_iters=n_chunks, n_stages=n_stages)

    def _build_small_d(self, D, n_warps, n_threads, dtype, epl, vec_elems, lane, warp_c):
        """Small D: all threads cooperate to load many rows via cp.async.

        128 threads × 8 bf16 = 1024 elems per load pass = 16 rows of D=64.
        Each warp processes rows_per_warp rows sequentially with full
        32-lane reduction (epl=2 for D=64).
        """
        g = self.g
        bctx = self.bctx

        # How many rows we load per cooperative pass.
        rows_per_load = (n_threads * vec_elems) // D
        rows_per_warp = rows_per_load // n_warps

        # Flat smem: (rows_per_load, D). All threads load 1 vec each.
        flat_smem = qk.smem_alloc("X_flat", dtype, (rows_per_load, D), pad=0)

        block_row_base = qk.block_idx("y") * bctx.c(rows_per_load, dtype=DType.U32)

        # Each thread loads exactly 1 cp.async 16B.
        # Thread tid maps to: flat element index = tid * vec_elems.
        # That element is at row = (tid * vec_elems) // D, col = (tid * vec_elems) % D.
        flat_elem = bctx.tid * vec_elems
        D_c = bctx.c(D, dtype=DType.U32)
        load_row = flat_elem // D_c
        load_col = flat_elem % D_c
        gmem_row = block_row_base + load_row

        qk.async_copy(
            dst=flat_smem,
            src=g.X,
            dst_idx=[load_row, load_col],
            src_idx=[gmem_row, load_col],
            count=_CP_BYTES,
        )
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")

        # Each warp processes rows_per_warp rows sequentially.
        rows_per_warp_c = bctx.c(rows_per_warp, dtype=DType.U32)
        warp_row_start = bctx.warp_id * rows_per_warp_c

        for ri in range(rows_per_warp):
            smem_row = warp_row_start + ri
            gmem_out_row = block_row_base + smem_row

            # Reduce: each lane reads epl elements, accumulates.
            local_sum_sq = bctx.c(0.0, dtype=DType.F32)
            x_regs: list = []
            for e in range(epl):
                col = bctx.c(e) * warp_c + lane
                x = qk.convert(flat_smem[smem_row, col], DType.F32)
                local_sum_sq = qk.fma(x, x, local_sum_sq)
                x_regs.append(x)

            total = qk.subgroup_reduce("sum", local_sum_sq)
            rms_inv = qk.rsqrt_approx(
                total * bctx.c(1.0 / D, dtype=DType.F32) + bctx.c(self.spec.eps, dtype=DType.F32)
            )

            # Write output: scalar stores (D is small, not worth vec_store staging).
            for e in range(epl):
                col = bctx.c(e) * warp_c + lane
                y = x_regs[e] * rms_inv
                g.Out[gmem_out_row, col] = qk.convert(y, dtype) if dtype is not DType.F32 else y
