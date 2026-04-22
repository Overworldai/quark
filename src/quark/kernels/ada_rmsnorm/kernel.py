"""AdaRMSNorm — fused RMSNorm + ``(1 + scale) * y + bias`` epilogue.

    X:     [G*M, D]
    scale: [G, D]    (broadcast M-wise into X's row axis)
    bias:  [G, D]
    Out:   [G*M, D]

Warp-per-row, double-buffered pipeline via PipelineBody.
Pass 1: pipeline X chunks → accumulate sum(x²) via carry.
Pass 2: pipeline X chunks again (L2-hot re-read) → normalize
with scale/bias from smem → vec_store.

No register stash — trades one extra gmem read (L2-cached) for
pipeline overlap in both passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import quark.lang as qk
from quark.blocks import PipelineBody, TensorDecl
from quark.blocks.l2.run_pipeline import IterCtx
from quark.ir import DType
from quark.kernels.ada_rmsnorm.baselines import ada_rmsnorm_baselines
from quark.kernels.ada_rmsnorm.config import AdaRMSNormConfig
from quark.kernels.ada_rmsnorm.problems import ada_rmsnorm_problems
from quark.kernels.ada_rmsnorm.reference import ada_rmsnorm_reference_numpy
from quark.kernels.ada_rmsnorm.spec import AdaRMSNormSpec
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel

_WARP = 32
_CP_BYTES = 16


@dataclass
class _XStage:
    """Per-stage smem for one X activation chunk."""

    X: object  # SharedRegion


@kernel(
    "ada_rmsnorm",
    spec=AdaRMSNormSpec,
    config=AdaRMSNormConfig,
    output_idx=-1,
    problems=ada_rmsnorm_problems,
    baselines=ada_rmsnorm_baselines,
    reference=ada_rmsnorm_reference_numpy,
)
class AdaRMSNormKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.B, s.D)),
        TensorDecl("scale", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.G, s.D)),
        TensorDecl("bias", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.G, s.D)),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.B, s.D),
            role="out",
        ),
    ]

    spec: AdaRMSNormSpec
    config: AdaRMSNormConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if c.n_warps < 1 or c.n_warps > 32:
            return False
        if s.D % _WARP != 0:
            return False
        if s.B % c.n_warps != 0:
            return False
        n_threads = c.n_warps * _WARP
        vec_elems = _CP_BYTES // s.dtype.bytes
        if s.D % vec_elems != 0:
            return False
        if (s.D // vec_elems) % n_threads != 0:
            return False
        chunk_D = c.chunk_D
        if chunk_D <= 0 or s.D % chunk_D != 0:
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
        return (1, self.spec.B // self.config.n_warps, 1)

    def flops(self) -> int:
        return 6 * self.spec.B * self.spec.D

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8], "chunk_D": [256, 512, 1024, 2048]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = AdaRMSNormSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "X": astype_numpy(rng.standard_normal((spec.B, spec.D)).astype(np.float32), spec.dtype),
            "scale": astype_numpy(
                (rng.standard_normal((spec.G, spec.D)) * 0.1).astype(np.float32), spec.dtype
            ),
            "bias": astype_numpy(
                (rng.standard_normal((spec.G, spec.D)) * 0.1).astype(np.float32), spec.dtype
            ),
            "Out": zeros_for_dtype((spec.B, spec.D), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(cls, X, scale, bias, *, eps: float = 1.1920929e-07) -> AdaRMSNormSpec:
        if X.ndim != 2:
            raise ValueError(f"ada_rmsnorm: X must be rank-2 [B, D]; got {X.shape}")
        if scale.shape != bias.shape:
            raise ValueError(f"ada_rmsnorm: scale {scale.shape} and bias {bias.shape} must match")
        B, D = int(X.shape[0]), int(X.shape[1])
        G = int(scale.shape[0])
        if int(scale.shape[1]) != D:
            raise ValueError(f"ada_rmsnorm: scale D ({scale.shape[1]}) != X D ({D})")
        if B % G != 0:
            raise ValueError(f"ada_rmsnorm: X rows ({B}) not divisible by G ({G})")
        M = B // G
        dt = DType.from_backend(X.dtype)
        return AdaRMSNormSpec(G=G, M=M, D=D, dtype=dt, eps=eps)

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        D = s.D
        M = s.M
        n_warps = c.n_warps
        n_threads = n_warps * _WARP
        dtype = s.dtype

        vec_elems = _CP_BYTES // dtype.bytes
        total_vecs = D // vec_elems
        vecs_per_thread = total_vecs // n_threads

        chunk_D = c.chunk_D
        n_chunks = D // chunk_D
        loads_per_lane = (chunk_D // vec_elems) // _WARP
        vecs_per_lane = loads_per_lane

        block_base = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row = block_base + bctx.warp_id
        M_c = bctx.c(M, dtype=DType.U32)
        group_for_load = block_base // M_c

        lane = bctx.lane_id

        # ── Smem ──
        S_smem = qk.smem_alloc("S_smem", dtype, (D,), pad=0)
        B_smem = qk.smem_alloc("B_smem", dtype, (D,), pad=0)
        n_stages = min(2, n_chunks)
        x_stages = [
            _XStage(X=qk.smem_alloc(f"X_s{i}", dtype, (n_warps, chunk_D), pad=0))
            for i in range(n_stages)
        ]

        # ── Load scale/bias via cooperative cp.async ──
        for v in range(vecs_per_thread):
            elem = (bctx.tid * vecs_per_thread + v) * vec_elems
            qk.async_copy(
                dst=S_smem,
                src=g.scale,
                dst_idx=[elem],
                src_idx=[group_for_load, elem],
                count=_CP_BYTES,
            )
            qk.async_copy(
                dst=B_smem,
                src=g.bias,
                dst_idx=[elem],
                src_idx=[group_for_load, elem],
                count=_CP_BYTES,
            )
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")

        # ── Pass 1: pipelined X chunks → accumulate sum(x²) ──
        init_sum = bctx.c(0.0, dtype=DType.F32)

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
        one_f = bctx.c(1.0, dtype=DType.F32)
        rms_inv = qk.rsqrt_approx(
            total * bctx.c(1.0 / D, dtype=DType.F32) + bctx.c(s.eps, dtype=DType.F32)
        )

        # ── Pass 2: pipelined X re-read (L2-hot) + scale/bias → output ──
        do_silu = s.activation == "silu"
        log2e = bctx.c(math.log2(math.e), dtype=DType.F32) if do_silu else None

        def consume_p2(ictx: IterCtx) -> tuple:
            stage = ictx.stage
            col_off = ictx.iter_idx * chunk_D
            for v in range(vecs_per_lane):
                vc = (lane * vecs_per_lane + v) * vec_elems
                x_vec = qk.vec_load(stage.X, bctx.warp_id, vc, width=vec_elems, dtype=dtype)
                s_vec = qk.vec_load(S_smem, col_off + vc, width=vec_elems, dtype=dtype)
                b_vec = qk.vec_load(B_smem, col_off + vc, width=vec_elems, dtype=dtype)

                out_elems = []
                for j in range(vec_elems):
                    x = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                    sc = qk.convert(qk.vec_extract(s_vec, j), DType.F32)
                    bi = qk.convert(qk.vec_extract(b_vec, j), DType.F32)
                    y = qk.fma(x * rms_inv, one_f + sc, bi)
                    if do_silu:
                        assert log2e is not None
                        sig = qk.rcp_approx(one_f + qk.ex2_approx(qk.neg(y * log2e)))
                        y = y * sig
                    out_elems.append(qk.convert(y, dtype))

                qk.vec_store(g.Out, qk.vec_build(out_elems), my_row, col_off + vc)
            return ()

        PipelineBody(
            stages=x_stages,
            produce=produce_x,
            consume=consume_p2,
        ).run(n_iters=n_chunks, n_stages=n_stages)
