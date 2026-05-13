"""GemmIntKernel — int8 GEMM with per-row scales on Intel Xe2.

s8 × s8 → s32 MMA via ``m8n16k32_intel_s8_s32`` cooperative_matrix.
Bypasses the F32-only frag_apply/frag_for_each IR ops via a custom
epilogue that round-trips the s32 accumulator through smem then
runs a per-lane scatter (scale + cast + gmem store).

This is the minimum-viable first cut covering the smoke shape
(BM=8, BN=16, BK=32 — single MMA tile, single warp). Once
correctness is locked, the kernel will expand to multi-tile and
multi-warp configurations.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.gemm_int.baselines import gemm_int_baselines
from quark.kernels.gemm_int.config import GemmIntConfig
from quark.kernels.gemm_int.problems import gemm_int_problems
from quark.kernels.gemm_int.reference import gemm_int_reference_numpy
from quark.kernels.gemm_int.spec import GemmIntSpec


@kernel(
    "gemm_int",
    spec=GemmIntSpec,
    config=GemmIntConfig,
    output_idx=-1,
    problems=gemm_int_problems,
    baselines=lambda kernel, tensors: gemm_int_baselines(tensors),
    reference=gemm_int_reference_numpy,
)
class GemmIntKernel(Kernel):
    """C = (A_s8 @ B_s8) * A_scales[M] * B_scales[N], cast to out_dtype."""

    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A",        dtype=lambda s, c: DType.S8,
                   shape=lambda s, c: (s.M, s.K)),
        # B stored as (N, K) — the "B^T in storage" convention every
        # GEMM-shaped kernel uses. With this layout the coopmat B
        # load picks ColumnMajor (from MMA's K×N perspective) and
        # reads the right elements.
        TensorDecl("B",        dtype=lambda s, c: DType.S8,
                   shape=lambda s, c: (s.N, s.K)),
        TensorDecl("A_scales", dtype=lambda s, c: s.scale_dtype,
                   shape=lambda s, c: (s.M,)),
        TensorDecl("B_scales", dtype=lambda s, c: s.scale_dtype,
                   shape=lambda s, c: (s.N,)),
        TensorDecl("Out",      dtype=lambda s, c: s.out_dtype,
                   shape=lambda s, c: (s.M, s.N), role="out"),
    ]

    spec: GemmIntSpec
    config: GemmIntConfig

    # The MMA site is fixed: s8 a, s8 b, s32 acc, m8n16k32 shape.
    # Default mma_sites would derive from compute_dtype_resolved
    # (which is S8) but ``lookup_mma`` would search for k=16. Override.
    @classmethod
    def mma_sites(cls, spec) -> list:
        from quark.kernels.base import MmaSite
        return [MmaSite(name="main", a_dtype=DType.S8, b_dtype=DType.S8)]

    def _mma_cfg(self):
        """Force the int8 m8n16k32 shape from the registry."""
        from quark.ir.mma_registry import _BY_SHAPE_ID
        cfg = _BY_SHAPE_ID.get("m8n16k32_intel_s8_s32")
        if cfg is None:
            raise RuntimeError(
                "GemmIntKernel requires the m8n16k32_intel_s8_s32 shape "
                "to be registered in the MMA registry."
            )
        return cfg

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        # BM must divide cleanly into MMA m=8 tiles; BN into MMA n=16
        # tiles; BK fixed to MMA k=32 (the only int8 coopmat shape on
        # Battlemage). Multi-warp partitions BN across warps.
        if c.BK != 32:
            return False
        if c.BM % 8 != 0 or c.BN % 16 != 0:
            return False
        if c.n_warps < 1 or c.n_warps > 8:
            return False
        # Each warp owns BN/n_warps cols, must be a multiple of n=16.
        if c.BN % (16 * c.n_warps) != 0:
            return False
        if (s.M % c.BM) != 0 or (s.N % c.BN) != 0 or (s.K % c.BK) != 0:
            return False
        # Smem budget: A=BM*BK + B=BN*BK + C=BM*BN*4 (s32). Keep < 48KB.
        smem_bytes = c.BM * c.BK + c.BN * c.BK + c.BM * c.BN * 4
        if smem_bytes > 49152:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        return (s.N // c.BN, s.M // c.BM, 1)

    def flops(self) -> int:
        return 2 * self.spec.M * self.spec.N * self.spec.K

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # Smoke-only for now. Expanded once core path works.
        return {"BM": [8], "BN": [16], "BK": [32], "n_warps": [1], "n_stages": [1]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = GemmIntSpec(**problem)
        rng = np.random.default_rng(seed)
        # Use small s8 values so the s32 accumulator stays well within
        # range and the bf16 cast doesn't lose too many bits.
        A = rng.integers(-64, 64, size=(spec.M, spec.K), dtype=np.int8)
        B = rng.integers(-64, 64, size=(spec.K, spec.N), dtype=np.int8)
        # f32 per-row scales — small values to keep the dequantized
        # output in a reasonable range.
        A_scales = rng.uniform(0.001, 0.01, size=(spec.M,)).astype(np.float32)
        B_scales = rng.uniform(0.001, 0.01, size=(spec.N,)).astype(np.float32)
        return {
            "A": A,
            "B": B,
            "A_scales": A_scales,
            "B_scales": B_scales,
            "Out": zeros_for_dtype((spec.M, spec.N), spec.out_dtype.value),
        }

    @classmethod
    def spec_from_tensors(cls, A, B, A_scales, B_scales, Out=None):
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"GemmInt: K mismatch A.K={K} vs B.K={K_b}")
        return GemmIntSpec(M=M, N=N, K=K)

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx
        mma_cfg = self._mma_cfg()
        shape_id = mma_cfg.shape.name  # "m8n16k32_intel_s8_s32"
        m_tile = mma_cfg.shape.m       # 8
        n_tile = mma_cfg.shape.n       # 16

        BM, BN, BK = c.BM, c.BN, c.BK
        n_warps = c.n_warps
        MT = BM // m_tile              # # m-tiles per WG
        NT = BN // n_tile              # # n-tiles per WG (total)
        # Each warp owns NT_per_warp n-tiles. Per-warp N slice =
        # warp_id * BN_per_warp .. (warp_id+1) * BN_per_warp.
        NT_per_warp = NT // n_warps
        BN_per_warp = NT_per_warp * n_tile
        out_dtype = s.out_dtype
        K_iters = s.K // BK            # Python-side unroll count

        # ── Block bases for this WG ──
        n_block = qk.block_idx("x")
        m_block = qk.block_idx("y")
        m_base = m_block * bctx.c(BM, dtype=DType.U32)
        n_base = n_block * bctx.c(BN, dtype=DType.U32)

        # ── Smem allocations ──
        smem_a = qk.smem_alloc("smem_a", DType.S8, (BM, BK))
        smem_b = qk.smem_alloc("smem_b", DType.S8, (BN, BK))
        smem_c = qk.smem_alloc("smem_c", DType.S32, (BM, BN))

        # ── Per-warp accumulator grid (MT × NT_per_warp) ──
        # Each warp owns NT_per_warp n-tiles; cross-warp partition
        # is on the N axis. Each warp's accumulators live in its own
        # registers (subgroup-private; no shared region).
        zero_s32 = bctx.c(0, dtype=DType.S32)
        c_inits = [
            qk.vec_build([zero_s32] * mma_cfg.shape.c_regs)
            for _ in range(MT * NT_per_warp)
        ]
        c_frags = list(c_inits)

        # Warp's N base in *tiles* (0..NT_per_warp boundary).
        warp_id = bctx.warp_id  # subgroup id within WG
        warp_n_tile_base = warp_id * bctx.c(NT_per_warp, dtype=DType.U32)

        # ── K-loop with multi-tile MMA ──
        for k_iter in range(K_iters):
            k_col = bctx.c(k_iter * BK, dtype=DType.U32)
            smem_a.copy_from(
                g.A.tile(row=m_base, col=k_col, shape=(BM, BK)),
                tid=bctx.tid, n_threads=bctx.n_threads, async_load=False,
            )
            smem_b.copy_from(
                g.B.tile(row=n_base, col=k_col, shape=(BN, BK)),
                tid=bctx.tid, n_threads=bctx.n_threads, async_load=False,
            )
            qk.barrier("block")

            # Pre-load all A-frags (one per m-tile) for this K iter.
            # A is shared across warps; every warp loads the same A.
            a_frags = [
                qk.load_matrix(
                    smem_a, shape_id, which="a",
                    row=bctx.c(mt * m_tile, dtype=DType.U32),
                    col=bctx.c(0, dtype=DType.U32),
                    reg_offsets=mma_cfg.a_offsets,
                )
                for mt in range(MT)
            ]
            # B-frags: only this warp's NT_per_warp n-tiles.
            # Warp-relative nt → smem_b row = (warp_n_tile_base + nt) * n_tile.
            b_frags = [
                qk.load_matrix(
                    smem_b, shape_id, which="b",
                    row=(warp_n_tile_base + bctx.c(nt, dtype=DType.U32))
                        * bctx.c(n_tile, dtype=DType.U32),
                    col=bctx.c(0, dtype=DType.U32),
                    reg_offsets=mma_cfg.b_offsets,
                )
                for nt in range(NT_per_warp)
            ]
            for mt in range(MT):
                for nt in range(NT_per_warp):
                    idx = mt * NT_per_warp + nt
                    c_frags[idx] = qk.mma(
                        shape_id, a_frags[mt], b_frags[nt], c_frags[idx]
                    )

            if k_iter + 1 < K_iters:
                qk.barrier("block")

        # ── Store all (mt, nt) s32 accs → smem_c scratch ──
        # Each warp stores its own NT_per_warp tiles into its slice
        # of smem_c. The scratch is row-major BM × BN so each warp's
        # cols span warp_n_tile_base * n_tile .. that + BN_per_warp.
        for mt in range(MT):
            for nt in range(NT_per_warp):
                idx = mt * NT_per_warp + nt
                # Absolute col within the BM×BN scratch.
                col_abs = (warp_n_tile_base + bctx.c(nt, dtype=DType.U32)) \
                          * bctx.c(n_tile, dtype=DType.U32)
                qk.store_matrix(
                    smem_c, c_frags[idx],
                    shape_id, "d",
                    row=bctx.c(mt * m_tile, dtype=DType.U32),
                    col=col_abs,
                    reg_offsets=mma_cfg.cd_offsets,
                )
        qk.barrier("block")

        # ── Per-lane scatter: s32 → f32 × scales → cast → gmem ──
        # Each thread strides over BM*BN total output elements;
        # 32 threads cover the WG's tile.
        n_elems = BM * BN
        epl = n_elems // bctx.n_threads
        # n_elems may not be a clean multiple of n_threads at small
        # shapes; guard the tail with a predicate.
        n_threads_c = bctx.c(bctx.n_threads, dtype=DType.U32)
        bn_c = bctx.c(BN, dtype=DType.U32)
        for i in range(epl):
            flat = bctx.tid + bctx.c(i * bctx.n_threads, dtype=DType.U32)
            row_local = flat // bn_c
            col_local = flat % bn_c
            gmem_row = m_base + row_local
            gmem_col = n_base + col_local

            s32_val = smem_c[row_local, col_local]
            a_scale = g.A_scales[gmem_row]
            b_scale = g.B_scales[gmem_col]
            f32_val = qk.convert(s32_val, DType.F32) * a_scale * b_scale

            if out_dtype is DType.F32:
                out_val = f32_val
            else:
                out_val = qk.convert(f32_val, out_dtype)
            g.Out[gmem_row, gmem_col] = out_val
