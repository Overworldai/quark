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
from quark.blocks import PipelineBody, SmemVector, TensorDecl
from quark.blocks.l2.run_pipeline import IterCtx
from quark.device import DEFAULT_SUBGROUP_WIDTH as _WARP  # see device.py:DEFAULT_SUBGROUP_WIDTH
from quark.device import DeviceFamily
from quark.ir import DType
from quark.kernels.ada_rmsnorm.baselines import ada_rmsnorm_baselines
from quark.kernels.ada_rmsnorm.config import AdaRMSNormConfig
from quark.kernels.ada_rmsnorm.problems import ada_rmsnorm_problems
from quark.kernels.ada_rmsnorm.reference import ada_rmsnorm_reference_numpy
from quark.kernels.ada_rmsnorm.spec import AdaRMSNormSpec
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel

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

    def _use_metal_multi_sg(self) -> bool:
        """Mirror of ``RMSNormKernel._use_metal_multi_sg``: pick the
        multi-simdgroup-per-row variant when the row's load chain
        would serialize >4 vec chunks per lane on a single warp."""
        s, c = self.spec, self.config
        if getattr(self, "caps", None) is None:
            return False
        if self.caps.family is not DeviceFamily.METAL:
            return False
        if c.n_warps <= 1:
            return False
        vec_elems = _CP_BYTES // s.dtype.bytes
        if _WARP * vec_elems > s.D or s.D % (_WARP * vec_elems) != 0:
            return False
        epl = s.D // _WARP
        if epl % vec_elems != 0:
            return False
        return (epl // vec_elems) > 4

    def grid(self) -> tuple[int, int, int]:
        if self._use_metal_multi_sg():
            return (1, self.spec.B, 1)
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
        dt = DType.from_backend(X)
        return AdaRMSNormSpec(G=G, M=M, D=D, dtype=dt, eps=eps)

    def build_metal(self) -> None:
        """Metal-flavored ada_rmsnorm: warp-per-row register-stash.

        Mirrors ``RMSNormKernel.build_metal`` and adds the
        ``(1 + scale) * y + bias`` epilogue (plus optional silu).
        scale / bias are read inline from gmem at the same offsets
        as X (L2-hot since every row in a group hits the same
        ``[group_row, col]`` slot), so smem is not needed.

        Falls back to the default ``build()`` (pipelined cp.async +
        smem; benefits from packed vec_load/async_copy lowering even
        on Metal) when the row layout doesn't divide cleanly into
        warp-sized vec chunks.
        """

        s, c = self.spec, self.config

        D = s.D
        M = s.M
        n_warps = c.n_warps
        dtype = s.dtype
        vec_elems = _CP_BYTES // dtype.bytes
        min_chunk = _WARP * vec_elems
        if min_chunk > D or D % min_chunk != 0:
            self.build()
            return
        epl = D // _WARP
        if epl % vec_elems != 0:
            self.build()
            return

        if self._use_metal_multi_sg():
            self._build_metal_multi_sg(D, M, n_warps, dtype, vec_elems)
            return

        self._build_metal_warp_per_row(D, M, n_warps, dtype, epl, vec_elems)

    def _build_metal_warp_per_row(self, D, M, n_warps, dtype, epl, vec_elems) -> None:
        import math as _math

        s = self.spec
        g = self.g
        bctx = self.bctx
        lane = bctx.lane_id
        vec_w = vec_elems
        vec_w_c = bctx.c(vec_w, dtype=DType.U32)
        vecs_per_lane = epl // vec_w

        block_base = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row = block_base + bctx.warp_id
        M_c = bctx.c(M, dtype=DType.U32)
        my_group = my_row // M_c

        sum_sq = bctx.c(0.0, dtype=DType.F32)
        regs: list[list] = []
        for v in range(vecs_per_lane):
            col = (lane + bctx.c(v * _WARP, dtype=DType.U32)) * vec_w_c
            x_vec = qk.vec_load(g.X, my_row, col, width=vec_w, dtype=dtype)
            v_regs = []
            for j in range(vec_w):
                x_f = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                sum_sq = qk.fma(x_f, x_f, sum_sq)
                v_regs.append(x_f)
            regs.append(v_regs)

        total = qk.subgroup_reduce("sum", sum_sq)
        one_f = bctx.c(1.0, dtype=DType.F32)
        rms_inv = qk.rsqrt_approx(
            total * bctx.c(1.0 / D, dtype=DType.F32) + bctx.c(s.eps, dtype=DType.F32)
        )

        do_silu = s.activation == "silu"
        log2e = bctx.c(_math.log2(_math.e), dtype=DType.F32) if do_silu else None

        for v in range(vecs_per_lane):
            col = (lane + bctx.c(v * _WARP, dtype=DType.U32)) * vec_w_c
            s_vec = qk.vec_load(g.scale, my_group, col, width=vec_w, dtype=dtype)
            b_vec = qk.vec_load(g.bias, my_group, col, width=vec_w, dtype=dtype)
            out_elems = []
            for j in range(vec_w):
                sc = qk.convert(qk.vec_extract(s_vec, j), DType.F32)
                bi = qk.convert(qk.vec_extract(b_vec, j), DType.F32)
                y = qk.fma(regs[v][j] * rms_inv, one_f + sc, bi)
                if do_silu:
                    assert log2e is not None
                    sig = qk.rcp_approx(one_f + qk.ex2_approx(qk.neg(y * log2e)))
                    y = y * sig
                out_elems.append(qk.convert(y, dtype))
            qk.vec_store(g.Out, qk.vec_build(out_elems), my_row, col)

    def _build_metal_multi_sg(self, D, M, n_warps, dtype, vec_elems) -> None:
        """Multi-simdgroup-per-row variant. Mirrors
        ``RMSNormKernel._build_metal_multi_sg`` plus the fused
        ``(1 + scale) * y + bias`` (+ optional silu) epilogue. Block
        is one row; ``n_warps`` simdgroups split the columns.
        Cross-warp reduction via ``tg_sums[n_warps]`` smem buffer.
        """
        import math as _math

        s = self.spec
        g = self.g
        bctx = self.bctx
        lane = bctx.lane_id
        sg_id = bctx.warp_id
        vec_w = vec_elems
        vec_w_c = bctx.c(vec_w, dtype=DType.U32)

        n_threads = n_warps * _WARP
        epl_per_thread = D // n_threads
        vecs_per_lane = epl_per_thread // vec_w
        sg_c = bctx.c(_WARP, dtype=DType.U32)
        tid = sg_id * sg_c + lane

        my_row = qk.block_idx("y")
        my_group = my_row // bctx.c(M, dtype=DType.U32)

        sum_sq = bctx.c(0.0, dtype=DType.F32)
        regs: list[list] = []
        for v in range(vecs_per_lane):
            col = (tid + bctx.c(v * n_threads, dtype=DType.U32)) * vec_w_c
            x_vec = qk.vec_load(g.X, my_row, col, width=vec_w, dtype=dtype)
            v_regs = []
            for j in range(vec_w):
                x_f = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                sum_sq = qk.fma(x_f, x_f, sum_sq)
                v_regs.append(x_f)
            regs.append(v_regs)

        sg_sum = qk.subgroup_reduce("sum", sum_sq)

        tg_sums = qk.smem_alloc("tg_sums", DType.F32, (n_warps,), pad=0)
        zero_u = bctx.c(0, dtype=DType.U32)
        is_lane_zero = qk.cmp("eq", lane, zero_u)
        qk.store(tg_sums, sg_sum, sg_id, pred=is_lane_zero)
        qk.barrier("block")

        n_warps_c = bctx.c(n_warps, dtype=DType.U32)
        is_active = qk.cmp("lt", lane, n_warps_c)
        partial = qk.load(tg_sums, lane, pred=is_active)
        total = qk.subgroup_reduce("sum", partial)

        one_f = bctx.c(1.0, dtype=DType.F32)
        rms_inv = qk.rsqrt_approx(
            total * bctx.c(1.0 / D, dtype=DType.F32) + bctx.c(s.eps, dtype=DType.F32)
        )

        do_silu = s.activation == "silu"
        log2e = bctx.c(_math.log2(_math.e), dtype=DType.F32) if do_silu else None

        for v in range(vecs_per_lane):
            col = (tid + bctx.c(v * n_threads, dtype=DType.U32)) * vec_w_c
            s_vec = qk.vec_load(g.scale, my_group, col, width=vec_w, dtype=dtype)
            b_vec = qk.vec_load(g.bias, my_group, col, width=vec_w, dtype=dtype)
            out_elems = []
            for j in range(vec_w):
                sc = qk.convert(qk.vec_extract(s_vec, j), DType.F32)
                bi = qk.convert(qk.vec_extract(b_vec, j), DType.F32)
                y = qk.fma(regs[v][j] * rms_inv, one_f + sc, bi)
                if do_silu:
                    assert log2e is not None
                    sig = qk.rcp_approx(one_f + qk.ex2_approx(qk.neg(y * log2e)))
                    y = y * sig
                out_elems.append(qk.convert(y, dtype))
            qk.vec_store(g.Out, qk.vec_build(out_elems), my_row, col)

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        D = s.D
        M = s.M
        n_warps = c.n_warps
        dtype = s.dtype

        vec_elems = _CP_BYTES // dtype.bytes

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
        scale_vec = SmemVector("S_smem", dtype, D)
        bias_vec = SmemVector("B_smem", dtype, D)
        n_stages = min(2, n_chunks)
        x_stages = [
            _XStage(X=qk.smem_alloc(f"X_s{i}", dtype, (n_warps, chunk_D), pad=0))
            for i in range(n_stages)
        ]

        # ── Load scale/bias via cooperative cp.async ──
        scale_vec.load_from(g.scale, row=group_for_load)
        bias_vec.load_from(g.bias, row=group_for_load)
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")
        S_smem = scale_vec.smem
        B_smem = bias_vec.smem

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
