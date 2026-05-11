"""Patchify — Conv2d(kernel=stride=patch) fused into a single GEMM.

    X  [B, C, H, W]  ×  W [d_model, C*ph*pw]^T  →  Out [M, d_model]

where M = B * (H/ph) * (W/pw) = number of output tokens.

The A-tile loader computes strided gmem addresses that read the input
in (h_tok, w_tok, c, ph_off, pw_off) order — the "permute" that
would normally rearrange [B,C,H,W] → [B,Hp,Wp,C,ph,pw] happens
entirely in the address arithmetic, with zero host-side data movement.

For each output token at grid position (h_tok, w_tok):
  A[token, k] reads from X[b, c, h_tok*ph + ph_off, w_tok*pw + pw_off]
  where k = c * (ph*pw) + ph_off * pw + pw_off

The B-tile (weight) is a standard contiguous [d_model, K] row-major
load, identical to the base gemm kernel.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import (
    Accumulators,
    IterCtx,
    MmaBody,
    PipelineBody,
    SmemPlan,
    TensorDecl,
)
from quark.ir import DType
from quark.kernels.base import Kernel, MmaSite
from quark.kernels.decorator import kernel
from quark.kernels.patchify.baselines import patchify_baselines
from quark.kernels.patchify.config import PatchifyConfig
from quark.kernels.patchify.problems import patchify_problems
from quark.kernels.patchify.reference import patchify_reference_numpy
from quark.kernels.patchify.spec import PatchifySpec


@kernel(
    "patchify",
    spec=PatchifySpec,
    config=PatchifyConfig,
    output_idx=-1,
    problems=patchify_problems,
    baselines=patchify_baselines,
    reference=patchify_reference_numpy,
)
class PatchifyKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.B, s.C * s.H * s.W)),
        TensorDecl("W", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.d_model, s.K)),
        TensorDecl("Out", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.N), role="out"),
    ]

    spec: PatchifySpec
    config: PatchifyConfig

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if s.M % c.BM != 0 or s.N % c.BN != 0 or s.K % c.BK != 0:
            return False
        try:
            mma = self._mma_cfg()
        except KeyError:
            return False
        if not self._validate_gemm_tile(mma):
            return False
        elem_b = s.dtype.bytes
        if c.a_pad and ((c.BK + c.a_pad) * elem_b) % 16 != 0:
            return False
        if c.b_pad and ((c.BK + c.b_pad) * elem_b) % 16 != 0:
            return False
        if c.n_stages == 2 and (s.K // c.BK) % 2 != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        return (s.N // c.BN, s.M // c.BM, 1)

    def flops(self) -> int:
        s = self.spec
        return 2 * s.M * s.N * s.K

    @classmethod
    def mma_sites(cls, spec) -> list[MmaSite]:
        if spec is None:
            return []
        return [MmaSite(name="main", a_dtype=spec.dtype, b_dtype=spec.dtype)]

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            "BM": [16, 32, 64, 128],
            "BN": [16, 32, 64, 128, 256],
            "BK": [16, 32, 64],
            "n_warps": [2, 4, 8],
            "n_stages": [1, 2],
            "a_pad": [0, 8],
            "b_pad": [0, 8],
        }

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = PatchifySpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            # X stored flat [B, C*H*W]; kernel indexes with strided arithmetic.
            "X": astype_numpy(
                rng.standard_normal((spec.B, spec.C * spec.H * spec.W)).astype(np.float32),
                spec.dtype,
            ),
            "W": astype_numpy(
                rng.standard_normal((spec.d_model, spec.K)).astype(np.float32), spec.dtype
            ),
            "Out": zeros_for_dtype((spec.M, spec.N), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(cls, X, W, *, C: int, H: int, W_spatial: int, ph: int = 2, pw: int = 2):
        B = int(X.shape[0])
        d_model = int(W.shape[0])
        dt = DType.from_backend(X)
        return PatchifySpec(B=B, C=C, H=H, W=W_spatial, ph=ph, pw=pw, d_model=d_model, dtype=dt)

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx, m_base, n_base = self.bctx, self.m_base, self.n_base
        compute_ir = s.dtype
        mma_cfg = self._mma_cfg()

        stages = SmemPlan.staged_pairs(
            compute_ir,
            a_shape=(c.BM, c.BK),
            b_shape=(c.BN, c.BK),
            a_pad=c.a_pad,
            b_pad=c.b_pad,
            mma_cfg=mma_cfg,
            n_warps=c.n_warps,
            b_shuffled=False,
            n_stages=c.n_stages,
        )
        acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
        mma = MmaBody(acc=acc)
        BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
        K_outer = s.K // c.BK

        # Precompute address constants for the strided A-tile load.
        # For output token m, the input patch is at:
        #   b = m // (Hp * Wp)
        #   h_tok = (m % (Hp * Wp)) // Wp
        #   w_tok = (m % (Hp * Wp)) % Wp
        # For K-dim element k:
        #   c_ch = k // (ph * pw)
        #   ph_off = (k % (ph * pw)) // pw
        #   pw_off = (k % (ph * pw)) % pw
        # Flat gmem index into X [B, C*H*W]:
        #   x_row = b
        #   x_col = c_ch * (H * W) + (h_tok * ph + ph_off) * W + (w_tok * pw + pw_off)
        Hp = s.Hp
        Wp_val = s.Wp
        HW = s.H * s.W
        ph, pw = s.ph, s.pw
        pp = ph * pw
        HpWp = Hp * Wp_val

        Wp_c = bctx.c(Wp_val)
        HpWp_c = bctx.c(HpWp)
        HW_c = bctx.c(HW)
        W_c = bctx.c(s.W)
        pp_c = bctx.c(pp)
        pw_c = bctx.c(pw)
        ph_c = bctx.c(ph)

        def produce(ictx: IterCtx) -> None:
            plan = ictx.stage
            k_col = ictx.iter_idx * c.BK

            # B-tile (weight): standard contiguous load.
            plan.b.load_from(g.W, row=n_base, col=k_col, cast=None)

            # A-tile: strided load from X [B, C*H*W].
            # We need to fill plan.a's smem tile [BM, BK] with elements
            # where row = output token (m_base + local_row) and
            # col = K-dim index (k_col + local_col).
            #
            # Since SmemTile.load_from expects a GlobalTensor address, we
            # compute the gmem (row, col) for each element in the tile.
            # The tile loader iterates over (row, col) pairs and we map:
            #   token_idx = m_base + row
            #   k_idx = k_col + col
            # to flat X address.
            #
            # We use the scalar-store path: write each element individually
            # to the A smem tile. This is less efficient than cp.async but
            # correct for non-contiguous access patterns.
            n_threads = c.n_warps * 32
            tid = bctx.tid
            total_elems = c.BM * c.BK
            elems_per_thread = total_elems // n_threads

            n_threads_c = bctx.c(n_threads)
            BK_c = bctx.c(c.BK)

            for i in range(elems_per_thread):
                flat = bctx.c(i) * n_threads_c + tid
                local_row = flat // BK_c
                local_col = flat % BK_c

                token_idx = m_base + local_row
                k_idx = k_col + local_col

                # Decompose token_idx → (b, h_tok, w_tok).
                b = token_idx // HpWp_c
                token_in_frame = token_idx % HpWp_c
                h_tok = token_in_frame // Wp_c
                w_tok = token_in_frame % Wp_c

                # Decompose k_idx → (c_ch, ph_off, pw_off).
                c_ch = k_idx // pp_c
                k_rem = k_idx % pp_c
                ph_off = k_rem // pw_c
                pw_off = k_rem % pw_c

                # Flat gmem address in X [B, C*H*W].
                x_row = b
                x_col = c_ch * HW_c + (h_tok * ph_c + ph_off) * W_c + (w_tok * pw_c + pw_off)

                val = g.X[x_row, x_col]
                plan.a.smem[local_row, local_col] = val

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=mma,
            carry=acc,
        ).run(n_iters=K_outer, n_stages=c.n_stages)

        qk.store_acc(
            g.Out,
            acc,
            row=m_base,
            col=n_base + bctx.warp_id * BN_per_warp,
            cast=s.dtype,
        )
