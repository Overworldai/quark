"""AdaGateResidual — fused ``out = x + gate_bmcast * y``.

Two build paths, selected by AdaGateResidualConfig.direct:

smem pipeline (direct=False):
  Warp-per-row, chunked over D with double-buffered pipeline.
  Gate vector staged to 1D smem via cp.async. Per chunk:
  cp.async X/Y into per-warp smem → vec_load → fma → vec_store.
  Best for large G where different blocks load different gate rows.

direct (direct=True):
  Warp-per-row, zero smem. Gate/X/Y read straight from gmem via
  ld.global.v4.b32. No async_wait barriers. All loads unrolled so
  ptxas can schedule them to hide L2 latency. Best for small G
  (e.g. G=1) where the gate row is L2-resident and smem staging
  overhead exceeds the load latency savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import quark.lang as qk
from quark.blocks import PipelineBody, SmemVector, TensorDecl
from quark.blocks.l2.run_pipeline import IterCtx
from quark.ir import DType
from quark.kernels.ada_gate_residual.baselines import ada_gate_residual_baselines
from quark.kernels.ada_gate_residual.config import AdaGateResidualConfig
from quark.kernels.ada_gate_residual.problems import ada_gate_residual_problems
from quark.kernels.ada_gate_residual.reference import ada_gate_residual_reference_numpy
from quark.kernels.ada_gate_residual.spec import AdaGateResidualSpec
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel

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
        if s.D % self._sgs != 0:
            return False
        if s.M % c.n_warps != 0:
            return False
        vec_elems = _CP_BYTES // s.dtype.bytes
        if s.D % vec_elems != 0:
            return False
        n_threads = c.n_warps * self._sgs
        if (s.D // vec_elems) % n_threads != 0:
            return False
        chunk_D = c.chunk_D
        if chunk_D <= 0 or s.D % chunk_D != 0:
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
        s, c = self.spec, self.config
        if c.direct:
            # 3D grid: x=D chunks, y=M//n_warps row-groups within each gate
            # group, z=G gate groups. Using z for gate groups eliminates the
            # integer division my_group = my_row // M at runtime.
            return (s.D // c.chunk_D, s.M // c.n_warps, s.G)
        return (1, s.M // c.n_warps, s.G)

    def flops(self) -> int:
        return 5 * self.spec.B * self.spec.D

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            "n_warps": [1, 2, 4, 8, 16],
            "chunk_D": [256, 512, 1024, 2048],
            "direct": [False, True],
        }

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
        dt = DType.from_backend(X)
        return AdaGateResidualSpec(G=G, M=M, D=D, dtype=dt)

    def build(self) -> None:
        if self.config.direct:
            self._build_direct()
        else:
            self._build_smem_pipeline()

    def build_metal(self) -> None:
        """Metal-flavored ada_gate_residual: warp-per-row, no smem, no
        D-chunk grid.

        ``out = x + gate * y`` is purely elementwise; there's no
        reduction so no register stash is needed — vec_load X / Y /
        gate, fma, vec_store, all in flight together. Avoids the
        ``_build_smem_pipeline`` cp.async detour (Metal stub) and the
        ``_build_direct`` D-chunked grid (which spawns one block per
        D-chunk; redundant once we use the wide vec_load path below).

        Grid: (1, M//n_warps, G) — same shape as the default smem path
        so ``grid()`` requires no override.
        """
        s, c = self.spec, self.config

        D = s.D
        M = s.M
        n_warps = c.n_warps
        dtype = s.dtype
        vec_elems = _CP_BYTES // dtype.bytes
        min_chunk = self._sgs * vec_elems
        if min_chunk > D or D % min_chunk != 0:
            self.build()
            return
        epl = D // self._sgs
        if epl % vec_elems != 0:
            self.build()
            return

        g = self.g
        bctx = self.bctx
        lane = bctx.lane_id
        vec_w = vec_elems
        vec_w_c = bctx.c(vec_w, dtype=DType.U32)
        vecs_per_lane = epl // vec_w

        my_group = qk.block_idx("z")
        block_base_in_group = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row_in_group = block_base_in_group + bctx.warp_id
        my_row = my_group * bctx.c(M, dtype=DType.U32) + my_row_in_group

        for v in range(vecs_per_lane):
            col = (lane + bctx.c(v * self._sgs, dtype=DType.U32)) * vec_w_c
            g_vec = qk.vec_load(g.gate, my_group, col, width=vec_w, dtype=dtype)
            x_vec = qk.vec_load(g.X, my_row, col, width=vec_w, dtype=dtype)
            y_vec = qk.vec_load(g.Y, my_row, col, width=vec_w, dtype=dtype)

            # Plain f32 fma — Metal has no fma.rn.bf16x2; the packed-b32
            # path is a CUDA-only optimization and would also block the
            # packed vec_store reinterpret_cast fast path.
            out_elems = []
            for j in range(vec_w):
                gf = qk.convert(qk.vec_extract(g_vec, j), DType.F32)
                xf = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                yf = qk.convert(qk.vec_extract(y_vec, j), DType.F32)
                out_elems.append(qk.convert(qk.fma(gf, yf, xf), dtype))
            qk.vec_store(g.Out, qk.vec_build(out_elems), my_row, col)

    # ── Direct gmem path ─────────────────────────────────────────────────

    def _build_direct(self) -> None:
        """Zero-smem path: all loads are ld.global, ptxas schedules them
        to hide L2 latency. No async_wait barriers.

        Uses a 3D grid: x=D-chunks, y=within-group row-groups, z=gate-groups.
        blockIdx.z encodes the gate group directly, eliminating the integer
        division my_group = my_row // M at runtime."""
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        M = s.M
        n_warps = c.n_warps
        dtype = s.dtype
        vec_elems = _CP_BYTES // dtype.bytes

        chunk_D = c.chunk_D
        vecs_per_lane = (chunk_D // vec_elems) // self._sgs

        # blockIdx.z = gate group index (no division needed).
        my_group = qk.block_idx("z")
        # blockIdx.y = within-group warp-row-group index.
        block_base_in_group = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row_in_group = block_base_in_group + bctx.warp_id
        my_row = my_group * bctx.c(M, dtype=DType.U32) + my_row_in_group
        lane = bctx.lane_id

        # blockIdx.x selects the D-chunk this block owns.
        col_off = qk.block_idx("x") * bctx.c(chunk_D, dtype=DType.U32)

        for v in range(vecs_per_lane):
            vc = (lane * vecs_per_lane + v) * vec_elems
            col = col_off + vc

            g_vec = qk.vec_load(g.gate, my_group, col, width=vec_elems, dtype=dtype)
            x_vec = qk.vec_load(g.X, my_row, col, width=vec_elems, dtype=dtype)
            y_vec = qk.vec_load(g.Y, my_row, col, width=vec_elems, dtype=dtype)

            n_pairs = vec_elems // 2
            if dtype == DType.BF16 and n_pairs * 2 == vec_elems:
                # Native bf16x2 FMA: avoids 3× convert chains, uses packed b32 ops.
                # fma.rn.bf16x2 d, gate, y, x  =>  d = gate * y + x
                out_b32 = [
                    qk.fma_bf16x2(
                        qk.packed_extract_b32(g_vec, k),
                        qk.packed_extract_b32(y_vec, k),
                        qk.packed_extract_b32(x_vec, k),
                    )
                    for k in range(n_pairs)
                ]
                out_vec = qk.vec_build_packed_b32(out_b32, elem_dtype=dtype, width=vec_elems)
            else:
                out_elems = []
                for j in range(vec_elems):
                    gf = qk.convert(qk.vec_extract(g_vec, j), DType.F32)
                    xf = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                    yf = qk.convert(qk.vec_extract(y_vec, j), DType.F32)
                    out_elems.append(qk.convert(qk.fma(gf, yf, xf), dtype))
                out_vec = qk.vec_build(out_elems)

            qk.vec_store(g.Out, out_vec, my_row, col)

    # ── Smem pipeline path ───────────────────────────────────────────────

    def _build_smem_pipeline(self) -> None:
        """Warp-per-row, double-buffered cp.async pipeline.
        Gate staged to 1D smem; X/Y staged per-warp per chunk."""
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
        loads_per_lane = (chunk_D // vec_elems) // self._sgs
        vecs_per_lane = loads_per_lane

        # 3D grid: z=gate-group, y=within-group row-group, x=always 1 (no D chunking).
        my_group = qk.block_idx("z")
        block_base_in_group = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row_in_group = block_base_in_group + bctx.warp_id
        my_row = my_group * bctx.c(M, dtype=DType.U32) + my_row_in_group
        group_for_load = my_group

        lane = bctx.lane_id

        # ── Smem ──
        gate_vec = SmemVector("G_smem", dtype, D)
        n_stages = min(2, n_chunks)
        stages = [
            _GateStage(
                X=qk.smem_alloc(f"X_s{i}", dtype, (n_warps, chunk_D), pad=0),
                Y=qk.smem_alloc(f"Y_s{i}", dtype, (n_warps, chunk_D), pad=0),
            )
            for i in range(n_stages)
        ]

        # ── Load gate via cooperative cp.async ──
        # Do NOT commit here. The uncommitted gate copies get bundled
        # with the prologue's first X/Y async_commit (auto-emitted by
        # PipelineBody). The pipeline's async_wait(1) before the first
        # consume covers gate + X/Y chunk 0 together, hiding the gate
        # load latency behind the prologue prefetch.
        gate_vec.load_from(g.gate, row=group_for_load)
        G_smem = gate_vec.smem

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

                n_pairs = vec_elems // 2
                if dtype == DType.BF16 and n_pairs * 2 == vec_elems:
                    out_b32 = [
                        qk.fma_bf16x2(
                            qk.packed_extract_b32(g_vec, k),
                            qk.packed_extract_b32(y_vec, k),
                            qk.packed_extract_b32(x_vec, k),
                        )
                        for k in range(n_pairs)
                    ]
                    out_vec = qk.vec_build_packed_b32(out_b32, elem_dtype=dtype, width=vec_elems)
                else:
                    out_elems = []
                    for j in range(vec_elems):
                        gf = qk.convert(qk.vec_extract(g_vec, j), DType.F32)
                        xf = qk.convert(qk.vec_extract(x_vec, j), DType.F32)
                        yf = qk.convert(qk.vec_extract(y_vec, j), DType.F32)
                        out_elems.append(qk.convert(qk.fma(gf, yf, xf), dtype))
                    out_vec = qk.vec_build(out_elems)

                qk.vec_store(g.Out, out_vec, my_row, col_off + vc)
            return ()

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=consume,
        ).run(n_iters=n_chunks, n_stages=n_stages)
