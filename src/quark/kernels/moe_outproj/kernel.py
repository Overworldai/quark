"""MoE out-projection kernel — per-slot partial output.

partials[slot, :] = h[slot, :] @ W_out[expert, :, :].T

Architecture (mirrors moe_inproj, no SiLU, contiguous A load):
  - Grid: (D // BN, n_work_items, 1)
  - Each block reads (grp_start, expert) from work_list.
  - A tile: CONTIGUOUS from h[grp_start..+BM, :] (hidden activations)
  - B tile: contiguous from W_out[expert*D + n_tile*BN, :]
  - K-pipelined GEMM body.
  - Epilogue: contiguous store into ``partials[grp_start.., col]`` at
    ``s.out_dtype`` (typically f32 — see ``nn.MoE`` for why).

The per-token reduction (sum over top_k slots, scaled by slot_weight)
is split into a separate ``moe_reduce`` kernel that gathers via the
router's ``token_slot_table`` output. This avoids the
non-deterministic atomic scatter-add the previous design needed and
keeps the bf16 quantization confined to ``moe_reduce``'s final cast
(``moe_outproj`` writes f32 partials, ``moe_reduce`` accumulates in
f32 registers and casts to bf16 once on the final store).
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
    block_idx,
)
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.moe_outproj.baselines import moe_outproj_baselines
from quark.kernels.moe_outproj.config import MoeOutprojConfig
from quark.kernels.moe_outproj.problems import moe_outproj_problems
from quark.kernels.moe_outproj.reference import moe_outproj_reference_numpy
from quark.kernels.moe_outproj.spec import MoeOutprojSpec


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
    # Plain bf16 stores — but the kernel still supports fp8 compute
    # (config_overrides=c_e4m3 problems), so keep the same loose 0.99
    # threshold the inproj kernel uses. The atomic-scatter
    # non-determinism that drove the previous 0.99 threshold is gone.
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
        # Per-slot partial output: ``partials[slot, :] = h[slot] @ W_out[expert].T``.
        # The downstream ``moe_reduce`` kernel folds in slot_weights and
        # gathers per-token sums via ``token_slot_table``.
        TensorDecl(
            "partials",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.total_slots, s.D),
            role="out",
        ),
        TensorDecl("work_list", dtype=DType.S32, shape=lambda s, c: (s.total_slots // c.BM * 2,)),
    ]

    spec: MoeOutprojSpec
    config: MoeOutprojConfig

    SHUFFLE_TENSOR = "W_out"

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        # Hard-reject BM != 32 — same reason as ``moe_inproj.is_valid``:
        # ``moe_router_correct`` emits work_list at a 32-slot step and
        # this kernel reads one ``(grp_start, expert)`` pair per block,
        # so BM must match. Catches stale cached configs from before
        # the tune_space pin.
        if c.BM != 32:
            return False
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

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = MoeOutprojSpec(**problem)
        D, H, n_e = spec.D, spec.H, spec.n_experts
        total = spec.total_slots
        slots_per_expert = spec.capacity

        rng = np.random.default_rng(seed)
        # Full-coverage work_list: one entry per BM=32 chunk, each
        # labelled with the expert that owns the chunk.
        assert slots_per_expert is not None
        wl = [(grp_start, grp_start // slots_per_expert) for grp_start in range(0, total, 32)]
        work_list = np.array(wl, dtype=np.int32).reshape(-1)
        return {
            "h_in": astype_numpy(rng.standard_normal((total, H)).astype(np.float32), spec.a_dtype),
            "W_out": astype_numpy(
                rng.standard_normal((n_e * D, H)).astype(np.float32), spec.b_dtype
            ),
            "partials": zeros_for_dtype((total, D), spec.out_dtype),
            "work_list": work_list,
        }

    @classmethod
    def spec_from_tensors(
        cls,
        h_in,
        W_out,
        work_list,
        *,
        M: int,
        n_experts: int,
        top_k: int = 2,
        out_dtype: DType | str = DType.BF16,
        compute_dtype: DType | str | None = None,
    ) -> MoeOutprojSpec:
        """Derive a ``MoeOutprojSpec`` from h_in + W_out + routing.

        h_in: ``[total_slots, H]``; W_out: ``[n_experts * D, H]``;
        work_list: ``[2 * total_slots / BM]``. ``M`` (per-token output
        rows) and ``top_k`` aren't recoverable from the slot buffers
        alone; callers pass them. The kernel itself doesn't read ``M``
        — it's spec metadata used by ``moe_reduce`` downstream.
        """
        del work_list  # shape captured implicitly via h_in's slot count
        if h_in.ndim != 2 or W_out.ndim != 2:
            raise ValueError(
                f"pcf.moe_outproj: h_in, W_out must be rank-2; "
                f"got h_in={h_in.shape}, W_out={W_out.shape}"
            )
        total_slots, H = int(h_in.shape[0]), int(h_in.shape[1])
        if total_slots % n_experts != 0:
            raise ValueError(
                f"pcf.moe_outproj: h_in.shape[0] ({total_slots}) not divisible by "
                f"n_experts ({n_experts})"
            )
        capacity = total_slots // n_experts
        if total_slots < M * top_k:
            raise ValueError(
                f"pcf.moe_outproj: h_in slot budget ({total_slots}) too small for "
                f"M*top_k ({M * top_k}); need capacity*n_experts >= M*top_k"
            )
        nW, H_w = int(W_out.shape[0]), int(W_out.shape[1])
        if H_w != H:
            raise ValueError(f"pcf.moe_outproj: h_in.shape[1] ({H}) != W_out.shape[1] ({H_w})")
        if nW % n_experts != 0:
            raise ValueError(
                f"pcf.moe_outproj: W_out.shape[0] ({nW}) not divisible by n_experts ({n_experts})"
            )
        D = nW // n_experts
        return MoeOutprojSpec(
            M=M,
            D=D,
            H=H,
            n_experts=n_experts,
            top_k=top_k,
            a_dtype=DType.from_backend(h_in),
            b_dtype=DType.from_backend(W_out),
            out_dtype=DType.coerce(out_dtype) or DType.BF16,
            compute_dtype=DType.coerce(compute_dtype),
            capacity=capacity,
        )

    def autotune_input_key(self) -> tuple:
        # ``work_list`` step is ``config.BM`` — the autotune harness
        # rebuilds the test fixture per BM so configs sharing the
        # same BM reuse one (inputs, reference) pair.
        return (int(self.config.BM),)

    def rebuild_autotune_inputs(self, base_inputs_np: dict) -> dict:
        """Rebuild ``work_list`` at ``config.BM`` step. Other inputs
        (h_in, W_out, token_ids, slot_weights) are unchanged.
        """
        import numpy as np

        bm = int(self.config.BM)
        spec = self.spec
        total = spec.total_slots
        slots_per_expert = spec.capacity
        if total % bm != 0 or slots_per_expert is None or slots_per_expert % bm != 0:
            return base_inputs_np
        wl_entries = [
            (grp_start, grp_start // slots_per_expert) for grp_start in range(0, total, bm)
        ]
        work_list = np.array(wl_entries, dtype=np.int32).reshape(-1)
        return {**base_inputs_np, "work_list": work_list}

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # b_shuffle pinned to False — runtime weight shuffling is gated
        # off (see Linear.prepare). Problems that explicitly need the
        # shuffled path can pin via config_overrides.
        return {
            # BM pinned to 32 to match ``moe_router_correct``'s
            # hardcoded ``_BM=32`` work_list step. See the same note
            # in ``moe_inproj/kernel.py::tune_space``.
            "BM": [32],
            "BN": [64, 128, 256],
            "BK": [16, 32, 64, 128],
            "n_warps": [4, 8],
            "n_stages": [1, 2],
            # Pad granule depends on compute dtype (fp8 → 16, bf16/fp16 → 8).
            "a_pad": [0, 8, 16],
            "b_pad": [0, 8, 16],
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
        grp_start, expert = qk.work_list_load(g.work_list, block_idx("y"))

        # Sentinel-skip for ``moe_router_correct`` filler chunks. See
        # the matching note in moe_inproj — bitcast U32→S32 so the -1
        # sentinel compares as signed. Inactive chunks just leave
        # ``partials[grp_start..+BM, :]`` at whatever the consumer
        # wrote previously; ``moe_reduce`` only reads slots indexed by
        # ``token_slot_table``, which never points at sentinel chunks.
        is_valid = qk.cmp("ge", qk.bitcast(expert, DType.S32), bctx.c(0, dtype=DType.S32))
        with qk.if_(is_valid, carried=[]) as (_, _, arms):
            with arms.then_():
                b_row_base = expert * s.D + n_base

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

                # Plain bf16 store — ``[total_slots, D]`` partials at
                # ``row=grp_start``. No scaling, no scatter; the
                # downstream ``moe_reduce`` kernel folds in
                # ``slot_weights`` and gathers per-token sums via
                # ``token_slot_table``.
                warp_col_base = n_base + bctx.warp_id * BN_per_warp
                qk.store_acc(
                    g.partials,
                    acc,
                    row=grp_start,
                    col=warp_col_base,
                    cast=s.out_dtype,
                )
                qk.yield_()
            with arms.else_():
                qk.yield_()
