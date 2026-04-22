"""AdaGateResidual — fused ``out = x + gate_bmcast * y``.

Warp-per-row, chunked over D with double-buffered pipeline.
Gate vector in 1D smem via cp.async. Per chunk: cp.async X/Y
into per-warp smem (overlapped with previous chunk's compute)
→ vec_load → compute → vec_build + vec_store to gmem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import quark.lang as qk
from quark.blocks import PipelineBody, TensorDecl
from quark.blocks.l2.run_pipeline import IterCtx
from quark.ir import DType
from quark.kernels.ada_gate_residual.baselines import ada_gate_residual_baselines
from quark.kernels.ada_gate_residual.config import AdaGateResidualConfig
from quark.kernels.ada_gate_residual.problems import ada_gate_residual_problems
from quark.kernels.ada_gate_residual.reference import ada_gate_residual_reference_numpy
from quark.kernels.ada_gate_residual.spec import AdaGateResidualSpec
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel

_WARP = 32
_CP_BYTES = 16


@dataclass
class _GateStage:
    """Per-stage smem for X/Y activation chunks."""

    X: object  # SharedRegion
    Y: object  # SharedRegion


@kernel(
    "ada_gate_residual",
    spec=AdaGateResidualSpec,
    config=AdaGateResidualConfig,
    output_idx=-1,
    problems=ada_gate_residual_problems,
    baselines=ada_gate_residual_baselines,
    reference=ada_gate_residual_reference_numpy,
)
class AdaGateResidualKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.B, s.D)),
        TensorDecl("Y", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.B, s.D)),
        TensorDecl("gate", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.G, s.D)),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.B, s.D),
            role="out",
        ),
    ]

    spec: AdaGateResidualSpec
    config: AdaGateResidualConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if not (1 <= c.n_warps <= 32):
            return False
        if s.D % _WARP != 0:
            return False
        if s.B % c.n_warps != 0:
            return False
        vec_elems = _CP_BYTES // s.dtype.bytes
        if s.D % vec_elems != 0:
            return False
        n_threads = c.n_warps * _WARP
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
        return 5 * self.spec.B * self.spec.D

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8], "chunk_D": [256, 512, 1024, 2048]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = AdaGateResidualSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "X": astype_numpy(rng.standard_normal((spec.B, spec.D)).astype(np.float32), spec.dtype),
            "Y": astype_numpy(rng.standard_normal((spec.B, spec.D)).astype(np.float32), spec.dtype),
            "gate": astype_numpy(
                (rng.standard_normal((spec.G, spec.D)) * 0.5).astype(np.float32), spec.dtype
            ),
            "Out": zeros_for_dtype((spec.B, spec.D), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(cls, X, Y, gate) -> AdaGateResidualSpec:
        if X.ndim != 2 or Y.ndim != 2 or gate.ndim != 2:
            raise ValueError("ada_gate_residual: all tensors must be rank-2")
        if X.shape != Y.shape:
            raise ValueError(f"ada_gate_residual: X {X.shape} != Y {Y.shape}")
        B, D = int(X.shape[0]), int(X.shape[1])
        G = int(gate.shape[0])
        if int(gate.shape[1]) != D:
            raise ValueError(f"ada_gate_residual: gate D mismatch: {gate.shape} vs {D}")
        if B % G != 0:
            raise ValueError(f"ada_gate_residual: X rows ({B}) not divisible by G ({G})")
        M = B // G
        dt = DType.from_backend(X.dtype)
        return AdaGateResidualSpec(G=G, M=M, D=D, dtype=dt)

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
        G_smem = qk.smem_alloc("G_smem", dtype, (D,), pad=0)
        n_stages = min(2, n_chunks)
        stages = [
            _GateStage(
                X=qk.smem_alloc(f"X_s{i}", dtype, (n_warps, chunk_D), pad=0),
                Y=qk.smem_alloc(f"Y_s{i}", dtype, (n_warps, chunk_D), pad=0),
            )
            for i in range(n_stages)
        ]

        # ── Load gate via cooperative cp.async ──
        for v in range(vecs_per_thread):
            elem = (bctx.tid * vecs_per_thread + v) * vec_elems
            qk.async_copy(
                dst=G_smem,
                src=g.gate,
                dst_idx=[elem],
                src_idx=[group_for_load, elem],
                count=_CP_BYTES,
            )
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")

        # ── Double-buffered pipeline over D chunks ──
        def produce(ictx: IterCtx) -> None:
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
                qk.async_copy(
                    dst=stage.Y,
                    src=g.Y,
                    dst_idx=[bctx.warp_id, local_col],
                    src_idx=[my_row, col_off + local_col],
                    count=_CP_BYTES,
                )

        def consume(ictx: IterCtx) -> tuple:
            stage = ictx.stage
            col_off = ictx.iter_idx * chunk_D
            for v in range(vecs_per_lane):
                vc = (lane * vecs_per_lane + v) * vec_elems
                g_vec = qk.vec_load(G_smem, col_off + vc, width=vec_elems, dtype=dtype)
                x_vec = qk.vec_load(stage.X, bctx.warp_id, vc, width=vec_elems, dtype=dtype)
                y_vec = qk.vec_load(stage.Y, bctx.warp_id, vc, width=vec_elems, dtype=dtype)

                out_elems = []
                for j in range(vec_elems):
                    gf = qk.convert(qk.vec_extract(g_vec, j), DType.F32)
                    xf = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                    yf = qk.convert(qk.vec_extract(y_vec, j), DType.F32)
                    out_elems.append(qk.convert(qk.fma(gf, yf, xf), dtype))

                qk.vec_store(g.Out, qk.vec_build(out_elems), my_row, col_off + vc)
            return ()

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=consume,
        ).run(n_iters=n_chunks, n_stages=n_stages)
