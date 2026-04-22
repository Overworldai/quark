"""Universal GEMM kernel — pipelined K, multi-dtype.

C[M, N] = A[M, K] @ B^T[N, K]^T

Architecture:
  - 2D grid: (N//BN, M//BM, 1); @kernel auto-publishes
    ``self.m_base = block_base("y", BM)`` / ``self.n_base =
    block_base("x", BN)`` before build() runs.
  - Pipelined K via ``run_pipeline``: n_stages=1 (synchronous)
    or n_stages=2 (double buffer).
  - ``SmemPlan.staged_pairs`` allocates n_stages of paired A/B
    smem with padding + b_shuffle-aware per-lane B views.
  - ``MmaBody(acc=, BK=)`` is callable as ``mma(ictx)`` for
    the K consume loop.
  - ``pop.store_acc`` epilogue, dtype-generic.
"""

from __future__ import annotations

from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import (
    Accumulators,
    IterCtx,
    MmaBody,
    PipelineBody,
    SmemPlan,
    TensorDecl,
)
from popcorn.ir import DType
from popcorn.kernels.base import Kernel
from popcorn.kernels.decorator import kernel
from popcorn.kernels.gemm.baselines import gemm_baselines
from popcorn.kernels.gemm.config import GemmConfig
from popcorn.kernels.gemm.problems import gemm_problems
from popcorn.kernels.gemm.reference import gemm_reference_numpy
from popcorn.kernels.gemm.spec import GemmSpec


@kernel(
    "gemm",
    spec=GemmSpec,
    config=GemmConfig,
    problems=gemm_problems,
    baselines=lambda kernel, tensors: gemm_baselines(tensors),
    reference=gemm_reference_numpy,
)
class GemmKernel(Kernel):
    # Parameter manifest — single source of truth for kernel tensor
    # shapes and dtypes (see popcorn.blocks.TensorDecl).
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A", dtype=lambda s, c: s.a_dtype, shape=lambda s, c: (s.M, s.K)),
        TensorDecl(
            "B",
            dtype=lambda s, c: s.b_dtype,
            shape=lambda s, c: (
                s.N,
                (s.K // c.BK) * (c.BK + c.b_pad) if s.b_shuffle and c.b_pad > 0 else s.K,
            ),
        ),
        TensorDecl(
            "Bias",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.N,) if s.has_bias else (1,),
        ),
        TensorDecl(
            "Out", dtype=lambda s, c: s.out_dtype, shape=lambda s, c: (s.M, s.N), role="out"
        ),
    ]

    spec: GemmSpec
    config: GemmConfig

    SHUFFLE_TENSOR = "B"

    # ── Validity ──

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if s.M % c.BM != 0 or s.N % c.BN != 0 or s.K % c.BK != 0:
            return False
        try:
            mma = self._mma_cfg()
        except KeyError:
            return False
        # The kernel's build() casts A/B on load to ``compute_dtype_resolved``
        # and lays out smem as that dtype. The MMA fragment loads then
        # read smem with the layout of ``mma_cfg.shape``. If the config
        # picks a ``main_shape`` whose dtype axes disagree with the spec's
        # compute dtype, the frag load uses the wrong stride / element
        # size on smem laid out for a different dtype — classic misaligned
        # address at launch. Reject the mismatch.
        compute = s.compute_dtype_resolved
        if mma.shape.a_dtype is not compute or mma.shape.b_dtype is not compute:
            return False
        if not self._validate_gemm_tile(mma):
            return False
        # K must split cleanly into BK-sized chunks. Without this the
        # kernel silently drops ``K % BK`` elements off the tail of
        # every row (cos_sim degrades and, because the tail read
        # straddles the real A/B boundary, some dtype / BK combos end
        # up dereferencing misaligned addresses at launch time).
        if s.K % c.BK != 0:
            return False
        # cp.async requires 16-byte aligned smem rows.
        compute_dt = s.compute_dtype_resolved
        elem_b = compute_dt.bytes
        if c.a_pad and ((c.BK + c.a_pad) * elem_b) % 16 != 0:
            return False
        if c.b_pad and ((c.BK + c.b_pad) * elem_b) % 16 != 0:
            return False
        # n_stages=2 uses a 2× unrolled loop → K-iteration count must be even.
        if c.n_stages == 2 and (s.K // c.BK) % 2 != 0:
            return False
        # split_k > 1 requires atomic add on the output dtype.
        # Dtype support is validated by is_valid_for(caps) which checks
        # AtomicRmwOp dtypes against device.atomic_add_dtypes.
        if c.split_k > 1:
            k_iters = s.K // c.BK
            if k_iters % c.split_k != 0:
                return False
            # Fused epilogue runs per-block BEFORE the atomic add, so
            # with split_k > 1 the bias gets added ``split_k`` times
            # (``out = sum_z (partial_z + bias)``) and the activation
            # is applied to each partial instead of the final sum
            # (``sum_z silu(partial_z) != silu(sum_z partial_z)`` —
            # silu is nonlinear). Reject the combo; the full-precision
            # fused path is only correct at split_k=1. A post-split
            # epilogue kernel would re-enable it but isn't wired up.
            if s.has_bias or s.activation is not None:
                return False
        return True

    def grid(self) -> tuple[int, int, int]:
        # Convention: x=N/BN (col_base), y=M/BM (row_base), z=split_k.
        s, c = self.spec, self.config
        return (s.N // c.BN, s.M // c.BM, c.split_k)

    def flops(self) -> int:
        return 2 * self.spec.M * self.spec.N * self.spec.K

    # ── Registry classmethods ──

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from popcorn.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = GemmSpec(**problem)
        rng = np.random.default_rng(seed)
        if spec.has_bias:
            bias_np = astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.out_dtype)
        else:
            bias_np = zeros_for_dtype((1,), spec.out_dtype)
        return {
            "A": astype_numpy(
                rng.standard_normal((spec.M, spec.K)).astype(np.float32), spec.a_dtype
            ),
            "B": astype_numpy(
                rng.standard_normal((spec.N, spec.K)).astype(np.float32), spec.b_dtype
            ),
            "Bias": bias_np,
            "Out": zeros_for_dtype((spec.M, spec.N), spec.out_dtype),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        A,
        B,
        *,
        out_dtype: DType | str | None = None,
        compute_dtype: DType | str | None = None,
        b_shuffled: bool = False,
        activation: str | None = None,
        has_bias: bool = False,
    ) -> GemmSpec:
        """Derive a GemmSpec from live A, B tensors + scalar kwargs.

        A: [M, K] in ``a_dtype``. B: [N, K] in ``b_dtype`` (or preshuffled
        when ``b_shuffled=True`` — B is still [N, K*pad] in that case,
        but the functional layer accepts the raw preshuffled buffer).
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"pcf.gemm: A, B must both be rank-2; got A={A.shape}, B={B.shape}")
        M, K = int(A.shape[0]), int(A.shape[1])
        N, K_b = int(B.shape[0]), int(B.shape[1])
        if not b_shuffled and K_b != K:
            raise ValueError(f"pcf.gemm: A.shape[1] ({K}) != B.shape[1] ({K_b})")
        a_dt_str = DType.from_backend(A.dtype)
        b_dt_str = DType.from_backend(B.dtype)
        return GemmSpec(
            M=M,
            N=N,
            K=K,
            a_dtype=a_dt_str,
            b_dtype=b_dt_str,
            out_dtype=DType.coerce(out_dtype) or a_dt_str,
            compute_dtype=DType.coerce(compute_dtype),
            activation=activation,
            has_bias=has_bias,
            b_shuffle=b_shuffled,
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # a_pad / b_pad includes both 8 (2-byte dtype granule) and 16
        # (1-byte dtype granule); is_valid() filters to the right one
        # per compute dtype. Backend-specific pruning (Metal unroll cap,
        # feature-flag gates like b_shuffle) happens in `is_valid_for`
        # via `DeviceCaps`, so this one search space serves every
        # backend — the autotuner just iterates the cartesian product
        # and drops invalid configs.
        return {
            "BM": [16, 32, 64, 128],
            "BN": [16, 32, 64, 128, 256],
            "BK": [16, 32, 64],
            "n_warps": [2, 4, 8],
            "n_stages": [1, 2],
            "a_pad": [0, 8, 16],
            "b_pad": [0, 8, 16],
            "split_k": [1, 2, 4, 8],
        }

    # ── build() ──

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx, m_base, n_base = self.bctx, self.m_base, self.n_base
        compute_ir = s.compute_dtype_resolved
        mma_cfg = self._mma_cfg()
        a_cast = compute_ir if s.a_dtype is not compute_ir else None
        b_cast = compute_ir if s.b_dtype is not compute_ir else None

        stages = SmemPlan.staged_pairs(
            compute_ir,
            a_shape=(c.BM, c.BK),
            b_shape=(c.BN, c.BK),
            a_pad=c.a_pad,
            b_pad=c.b_pad,
            mma_cfg=mma_cfg,
            n_warps=c.n_warps,
            b_shuffled=s.b_shuffle,
            n_stages=c.n_stages,
        )
        acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
        mma = MmaBody(acc=acc, b_shuffled=s.b_shuffle)
        BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
        bk_stride_a = c.BK
        bk_stride_b = (c.BK + c.b_pad) if (s.b_shuffle and c.b_pad > 0) else c.BK
        K_outer = s.K // c.BK

        # Split-K: each block processes a slice of the K dimension.
        split_k = c.split_k
        if split_k > 1:
            k_iters_per_split = K_outer // split_k
            k_split_idx = pop.block_idx("z")
            k_start = k_split_idx * bctx.c(k_iters_per_split, dtype=DType.U32)
        else:
            k_iters_per_split = K_outer
            k_start = bctx.c(0, dtype=DType.U32)

        def produce(ictx: IterCtx) -> None:
            plan = ictx.stage
            # Offset the K iteration by the split-K start.
            k_iter = k_start + ictx.iter_idx
            k_col_a = k_iter * bk_stride_a
            k_col_b = k_col_a if bk_stride_b == bk_stride_a else k_iter * bk_stride_b
            plan.a.load_from(g.A, row=m_base, col=k_col_a, cast=a_cast)
            plan.b.load_from(g.B, row=n_base, col=k_col_b, cast=b_cast)

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=mma,
            carry=acc,
        ).run(n_iters=k_iters_per_split, n_stages=c.n_stages)

        pop.store_acc(
            g.Out,
            acc,
            row=m_base,
            col=n_base + bctx.warp_id * BN_per_warp,
            cast=s.out_dtype,
            activation=s.activation,
            bias=g.Bias if s.has_bias else None,
            atomic=split_k > 1,
        )
