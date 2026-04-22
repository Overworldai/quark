"""MoE out-projection kernel on the new IR Builder.

output[tok_id, :] += weight * (h[slot, :] @ W_out[expert, :, :].T)

Architecture (same as inproj but different A loader + epilogue):
  - Grid: (D // BN, n_work_items, 1)
  - Each block reads (grp_start, expert) from work_list.
  - Caches token_ids AND slot_weights into smem.
  - A tile: CONTIGUOUS from h[grp_start..+BM, :] (hidden activations)
  - B tile: contiguous from W_out[expert*D + n_tile*BN, :]
  - K-pipelined GEMM body.
  - Epilogue: scale by weight, atomic scatter-add into output[tok_id, col].
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
    barrier,
    block_idx,
)
from popcorn.ir import DType
from popcorn.kernels.base import Kernel
from popcorn.kernels.decorator import kernel
from popcorn.kernels.moe_outproj.baselines import moe_outproj_baselines
from popcorn.kernels.moe_outproj.config import MoeOutprojConfig
from popcorn.kernels.moe_outproj.problems import moe_outproj_problems
from popcorn.kernels.moe_outproj.reference import moe_outproj_reference_numpy
from popcorn.kernels.moe_outproj.spec import MoeOutprojSpec


@kernel(
    "moe_outproj",
    spec=MoeOutprojSpec,
    config=MoeOutprojConfig,
    output_idx=2,
    problems=moe_outproj_problems,
    baselines=moe_outproj_baselines,
    reference=moe_outproj_reference_numpy,
)
class MoeOutprojKernel(Kernel):
    # Atomic scatter-add accumulates in non-deterministic order, so
    # the output drifts ~1e-3 from the reference (CPU sequential
    # scatter) even when the kernel is computing correctly. Relax to
    # 0.99 so bench doesn't reject tuned configs on fourth-decimal
    # numerical noise.
    CORRECTNESS_THRESHOLD = 0.99

    # Parameter manifest — single source of truth for shapes/dtypes of
    # every kernel parameter. emit() walks this via ctx.declare_tensors
    # instead of open-coding the six ctx.tensor(...) calls.
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("h_in", dtype=lambda s, c: s.a_dtype, shape=lambda s, c: (s.total_slots, s.H)),
        TensorDecl(
            "W_out",
            dtype=lambda s, c: s.b_dtype,
            shape=lambda s, c: (
                s.n_experts * s.D,
                (s.H // c.BK) * (c.BK + c.b_pad) if c.b_shuffle and c.b_pad > 0 else s.H,
            ),
        ),
        # Output is always f32: atomic scatter-add only lowers cleanly
        # at 4-byte granularity on every backend we target (sm_89 has
        # no bf16 atomic; Metal has no sub-4-byte atomic). Callers cast
        # scatter outputs to their preferred dtype in the surrounding
        # runtime code — a single f32→bf16 pass amortizes across the
        # real pipeline.
        TensorDecl("output", dtype=DType.F32, shape=lambda s, c: (s.M, s.D), role="out"),
        TensorDecl("token_ids", dtype=DType.S32, shape=lambda s, c: (s.total_slots,)),
        TensorDecl("slot_weights", dtype=DType.F32, shape=lambda s, c: (s.total_slots,)),
        TensorDecl("work_list", dtype=DType.S32, shape=lambda s, c: (s.total_slots // c.BM * 2,)),
    ]

    spec: MoeOutprojSpec
    config: MoeOutprojConfig

    SHUFFLE_TENSOR = "W_out"

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if s.D % c.BN != 0 or s.H % c.BK != 0:
            return False
        try:
            mma_cfg = self._mma_cfg()
        except KeyError:
            return False
        if not self._validate_gemm_tile(mma_cfg):
            return False
        compute_dt = s.compute_dtype_resolved
        elem_b = compute_dt.bytes
        if c.a_pad and ((c.BK + c.a_pad) * elem_b) % 16 != 0:
            return False
        if c.b_pad and ((c.BK + c.b_pad) * elem_b) % 16 != 0:
            return False
        if c.n_stages == 2 and (s.H // c.BK) % 2 != 0:
            return False
        slots_per_expert = s.total_slots // s.n_experts
        if slots_per_expert % c.BM != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        return (s.D // c.BN, s.total_slots // c.BM, 1)

    def flops(self) -> int:
        return 2 * self.spec.total_slots * self.spec.D * self.spec.H

    # ── Registry classmethods ──

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from popcorn.runtime.npconv import astype_numpy

        spec = MoeOutprojSpec(**problem)
        M, D, H, n_e, top_k = spec.M, spec.D, spec.H, spec.n_experts, spec.top_k
        total = M * top_k
        slots_per_expert = total // n_e

        rng = np.random.default_rng(seed)
        token_ids = (np.arange(total) % M).astype(np.int32)
        slot_weights = np.full(total, 0.5, dtype=np.float32)
        # Full-coverage work_list: one entry per BM=32 chunk, each
        # labelled with the expert that owns the chunk.
        wl = [(grp_start, grp_start // slots_per_expert) for grp_start in range(0, total, 32)]
        work_list = np.array(wl, dtype=np.int32).reshape(-1)
        return {
            "h_in": astype_numpy(rng.standard_normal((total, H)).astype(np.float32), spec.a_dtype),
            "W_out": astype_numpy(
                rng.standard_normal((n_e * D, H)).astype(np.float32), spec.b_dtype
            ),
            # Output always f32 — atomic scatter-add target.
            "output": np.zeros((M, D), dtype=np.float32),
            "token_ids": token_ids,
            "slot_weights": slot_weights,
            "work_list": work_list,
        }

    @classmethod
    def spec_from_tensors(
        cls,
        h_in,
        W_out,
        token_ids,
        slot_weights,
        work_list,
        *,
        M: int,
        n_experts: int,
        top_k: int = 2,
        out_dtype: DType | str = DType.BF16,
        compute_dtype: DType | str | None = None,
    ) -> MoeOutprojSpec:
        """Derive a ``MoeOutprojSpec`` from h_in + W_out + routing.

        h_in: ``[M * top_k, H]``; W_out: ``[n_experts * D, H]``.
        ``M`` can't be recovered from h_in's leading dim without
        ``top_k``; caller passes both.
        """

        if h_in.ndim != 2 or W_out.ndim != 2:
            raise ValueError(
                f"pcf.moe_outproj: h_in, W_out must be rank-2; "
                f"got h_in={h_in.shape}, W_out={W_out.shape}"
            )
        total_slots, H = int(h_in.shape[0]), int(h_in.shape[1])
        if total_slots != M * top_k:
            raise ValueError(
                f"pcf.moe_outproj: h_in.shape[0] ({total_slots}) != M*top_k ({M * top_k})"
            )
        nW, H_w = int(W_out.shape[0]), int(W_out.shape[1])
        if H_w != H:
            raise ValueError(f"pcf.moe_outproj: h_in.shape[1] ({H}) != W_out.shape[1] ({H_w})")
        if nW % n_experts != 0:
            raise ValueError(
                f"pcf.moe_outproj: W_out.shape[0] ({nW}) not divisible by n_experts ({n_experts})"
            )
        D = nW // n_experts
        if int(token_ids.shape[0]) != total_slots:
            raise ValueError(
                f"pcf.moe_outproj: token_ids shape {tuple(token_ids.shape)} != ({total_slots},)"
            )
        return MoeOutprojSpec(
            M=M,
            D=D,
            H=H,
            n_experts=n_experts,
            top_k=top_k,
            a_dtype=DType.from_backend(h_in.dtype),
            b_dtype=DType.from_backend(W_out.dtype),
            out_dtype=DType.coerce(out_dtype) or DType.BF16,
            compute_dtype=DType.coerce(compute_dtype),
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            "BM": [32, 64],
            "BN": [64, 128, 256],
            "BK": [16, 32, 64, 128],
            "n_warps": [4, 8],
            "n_stages": [1, 2],
            # Pad granule depends on compute dtype (fp8 → 16, bf16/fp16 → 8).
            "a_pad": [0, 8, 16],
            "b_pad": [0, 8, 16],
            "b_shuffle": [False, True],
            # mma_k dropped — shape injected as ``main_shape`` knob by
            # tune_space_resolved from device caps (MMA_SHAPES M3).
        }

    # ── build() ──

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        compute_ir = s.compute_dtype_resolved
        mma_cfg = self._mma_cfg()
        a_cast = compute_ir if s.a_dtype is not compute_ir else None
        b_cast = compute_ir if s.b_dtype is not compute_ir else None

        bctx, n_base = self.bctx, self.n_base
        grp_start, expert = pop.work_list_load(g.work_list, block_idx("y"))
        b_row_base = expert * s.D + n_base

        toks = pop.index_cache("toks_cache", g.token_ids, count=c.BM, base=grp_start)
        weights = pop.index_cache(
            "weights_cache", g.slot_weights, count=c.BM, base=grp_start, dtype=DType.F32
        )
        barrier("block")

        stages = SmemPlan.staged_pairs(
            compute_ir,
            a_shape=(c.BM, c.BK),
            b_shape=(c.BN, c.BK),
            a_pad=c.a_pad,
            b_pad=c.b_pad,
            mma_cfg=mma_cfg,
            n_warps=c.n_warps,
            b_shuffled=c.b_shuffle,
            n_stages=c.n_stages,
        )
        acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
        mma = MmaBody(shape=mma_cfg, acc=acc, BK=c.BK, b_shuffled=c.b_shuffle)
        BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
        bk_stride_a = c.BK
        bk_stride_b = (c.BK + c.b_pad) if (c.b_shuffle and c.b_pad > 0) else c.BK
        K_outer = s.H // c.BK

        def produce(ictx: IterCtx) -> None:
            plan = ictx.stage
            k_col_a = ictx.iter_idx * bk_stride_a
            k_col_b = k_col_a if bk_stride_b == bk_stride_a else ictx.iter_idx * bk_stride_b
            plan.a.load_from(g.h_in, row=grp_start, col=k_col_a, cast=a_cast)
            plan.b.load_from(g.W_out, row=b_row_base, col=k_col_b, cast=b_cast)

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=mma,
            carry=acc,
        ).run(n_iters=K_outer, n_stages=c.n_stages)

        pop.atomic_store_acc(
            g.output,
            acc,
            col=n_base + bctx.warp_id * BN_per_warp,
            index=toks,
            weight=weights,
        )
