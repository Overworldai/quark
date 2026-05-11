"""MoE in-projection kernel.

h[total_slots, H] = SiLU( X[token_ids, :] @ W_in[expert, :, :].T )

Architecture:
  - Grid: (H // BN, n_work_items, 1)
    - blockIdx.x → n_tile (column tile along H); self.n_base auto-bound
    - blockIdx.y → work_idx (which (grp_start, expert) pair)
  - Each block:
    1. Load (grp_start, expert) from work_list[work_idx]
    2. Cache token_ids[grp_start..+BM] into smem (IndexCache)
    3. K-pipelined GEMM body (``run_pipeline`` with carry=acc):
       - A tile: gathered from X via the index cache (smem.gather_from)
       - B tile: contiguous from W_in[expert*H + n_base, :]
    4. Epilogue: ``qk.store_acc(..., activation="silu", cast=)``
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
    barrier,
    block_idx,
)
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.moe_inproj.baselines import moe_inproj_baselines
from quark.kernels.moe_inproj.config import MoeInprojConfig
from quark.kernels.moe_inproj.problems import moe_inproj_problems
from quark.kernels.moe_inproj.reference import moe_inproj_reference_numpy
from quark.kernels.moe_inproj.spec import MoeInprojSpec


@kernel(
    "moe_inproj",
    spec=MoeInprojSpec,
    config=MoeInprojConfig,
    output_idx=2,
    problems=moe_inproj_problems,
    baselines=moe_inproj_baselines,
    reference=moe_inproj_reference_numpy,
)
class MoeInprojKernel(Kernel):
    # Relax the default ~0.9999 cos-sim threshold. bf16 MMA +
    # SiLU-in-f32 epilogue accumulates tiny per-lane order differences
    # (~1e-3) that the f32 CPU reference doesn't reproduce, so valid
    # tuned configs routinely land at 0.999–0.9998.
    CORRECTNESS_THRESHOLD = 0.99

    # Parameter manifest — single source of truth for kernel tensor
    # shapes and dtypes (see quark.blocks.TensorDecl).
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.a_dtype, shape=lambda s, c: (s.M, s.D)),
        TensorDecl(
            "W_in",
            dtype=lambda s, c: s.b_dtype,
            shape=lambda s, c: (
                s.n_experts * s.H,
                (s.D // c.BK) * (c.BK + c.b_pad) if c.b_shuffle and c.b_pad > 0 else s.D,
            ),
        ),
        TensorDecl(
            "H_out",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.total_slots, s.H),
            role="out",
        ),
        TensorDecl("token_ids", dtype=DType.S32, shape=lambda s, c: (s.total_slots,)),
        TensorDecl("work_list", dtype=DType.S32, shape=lambda s, c: (s.total_slots // c.BM * 2,)),
    ]

    spec: MoeInprojSpec
    config: MoeInprojConfig

    SHUFFLE_TENSOR = "W_in"

    # ── Validity ──

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        # Hard-rejct BM != 32: ``moe_router_correct`` emits work_list
        # at a hardcoded 32-slot step and the inproj kernel reads one
        # ``(grp_start, expert)`` pair per block. Any other BM produces
        # silent miscompute (slots beyond the first 32 use the wrong
        # expert). Catches cached on-disk configs from before the
        # tune_space pin, which would otherwise survive a tune_space
        # change.
        if c.BM != 32:
            return False
        if s.H % c.BN != 0 or s.D % c.BK != 0:
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
        slots_per_expert = s.total_slots // s.n_experts
        if slots_per_expert % c.BM != 0:
            return False
        if c.n_stages == 2 and (s.D // c.BK) % 2 != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        n_h_tiles = s.H // c.BN
        n_work_items = s.total_slots // c.BM
        return (n_h_tiles, n_work_items, 1)

    def flops(self) -> int:
        s = self.spec
        return 2 * s.total_slots * s.H * s.D

    # ── Registry classmethods ──

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = MoeInprojSpec(**problem)
        M, D, H, n_experts, _ = spec.M, spec.D, spec.H, spec.n_experts, spec.top_k
        total = spec.total_slots
        slots_per_expert = spec.capacity

        rng = np.random.default_rng(seed)
        token_ids = (np.arange(total) % M).astype(np.int32)
        # One work_list entry per BM=32 slot chunk, labeled with the
        # expert that owns the chunk. Full coverage: every output slot
        # gets written (matters on Metal where the driver allocates
        # outputs uninitialized and uncovered slots diverge from the
        # reference).
        assert slots_per_expert is not None
        wl_entries = [
            (grp_start, grp_start // slots_per_expert) for grp_start in range(0, total, 32)
        ]
        work_list = np.array(wl_entries, dtype=np.int32).reshape(-1)
        return {
            "X": astype_numpy(rng.standard_normal((M, D)).astype(np.float32), spec.a_dtype),
            "W_in": astype_numpy(
                rng.standard_normal((n_experts * H, D)).astype(np.float32), spec.b_dtype
            ),
            "H_out": zeros_for_dtype((total, H), spec.out_dtype),
            "token_ids": token_ids,
            "work_list": work_list,
        }

    @classmethod
    def spec_from_tensors(
        cls,
        X,
        W_in,
        token_ids,
        work_list,
        *,
        n_experts: int,
        top_k: int = 2,
        out_dtype: DType | str | None = None,
        compute_dtype: DType | str | None = None,
    ) -> MoeInprojSpec:
        """Derive a ``MoeInprojSpec`` from X, W_in + expert routing.

        X: ``[M, D]``; W_in: ``[n_experts * H, D]``; token_ids:
        ``[M * top_k]``. ``M`` / ``D`` from X; ``H`` from W_in's leading
        dim divided by ``n_experts``.
        """

        if X.ndim != 2 or W_in.ndim != 2:
            raise ValueError(
                f"qf.moe_inproj: X, W_in must be rank-2; got X={X.shape}, W_in={W_in.shape}"
            )
        M, D = int(X.shape[0]), int(X.shape[1])
        nW, D_w = int(W_in.shape[0]), int(W_in.shape[1])
        if D_w != D:
            raise ValueError(f"qf.moe_inproj: X.shape[1] ({D}) != W_in.shape[1] ({D_w})")
        if nW % n_experts != 0:
            raise ValueError(
                f"qf.moe_inproj: W_in.shape[0] ({nW}) not divisible by n_experts ({n_experts})"
            )
        H = nW // n_experts
        ts = int(token_ids.shape[0])
        if ts % n_experts != 0:
            raise ValueError(
                f"qf.moe_inproj: token_ids shape {tuple(token_ids.shape)} not "
                f"divisible by n_experts ({n_experts})"
            )
        capacity = ts // n_experts
        if ts < M * top_k:
            raise ValueError(
                f"qf.moe_inproj: token_ids buffer ({ts}) too small for M*top_k "
                f"({M * top_k}); need capacity*n_experts >= M*top_k"
            )
        a_dt = DType.from_backend(X)
        return MoeInprojSpec(
            M=M,
            D=D,
            H=H,
            n_experts=n_experts,
            top_k=top_k,
            a_dtype=a_dt,
            b_dtype=DType.from_backend(W_in),
            out_dtype=DType.coerce(out_dtype) or a_dt,
            compute_dtype=DType.coerce(compute_dtype),
            capacity=capacity,
        )

    def autotune_input_key(self) -> tuple:
        # ``work_list`` step is ``config.BM`` — the autotune harness
        # rebuilds the test fixture per BM so configs sharing the
        # same BM reuse one (inputs, reference) pair.
        return (int(self.config.BM),)

    def rebuild_autotune_inputs(self, base_inputs_np: dict) -> dict:
        """Rebuild ``work_list`` at ``config.BM`` step. Other tensors
        (X, W_in, token_ids) are unchanged — only the work_list shape
        depends on BM.
        """
        import numpy as np

        bm = int(self.config.BM)
        spec = self.spec
        total = spec.total_slots
        slots_per_expert = spec.capacity
        if total % bm != 0 or slots_per_expert is None or slots_per_expert % bm != 0:
            # Spec/config mismatch — fall through to base; ``is_valid``
            # catches it before launch anyway.
            return base_inputs_np
        wl_entries = [
            (grp_start, grp_start // slots_per_expert) for grp_start in range(0, total, bm)
        ]
        work_list = np.array(wl_entries, dtype=np.int32).reshape(-1)
        return {**base_inputs_np, "work_list": work_list}

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # Single backend-agnostic search space. `is_valid_for` prunes
        # configs that opt into capabilities the device lacks
        # (vec_epilogue, b_shuffle, large unroll counts on Metal).
        # Shape selection (mma_k's job pre-MMA_SHAPES) is now injected
        # as a ``main_shape`` knob by ``tune_space_resolved`` from the
        # device's legal shape set — keep ``mma_k`` off the tune space
        # so the autotuner can't produce a (mma_k=32, main_shape=k16)
        # mismatch where config fields disagree.
        # b_shuffle pinned to False — runtime weight shuffling is gated
        # off (see Linear.prepare). Problems that explicitly need the
        # shuffled path can pin via config_overrides.
        return {
            # BM pinned to 32 to match ``moe_router_correct``'s
            # hardcoded ``_BM=32`` work_list step. Each kernel block
            # reads one ``(grp_start, expert)`` pair and processes
            # ``BM`` slots; with BM != 32 the routing for slots beyond
            # the first 32 of the block comes from the wrong work_list
            # entry and the kernel silently uses the wrong expert.
            # Lifting to BM=64 would require threading BM through the
            # router spec so all three kernels (router/inproj/outproj)
            # agree on the same step.
            "BM": [32],
            "BN": [64, 128, 256],
            "BK": [16, 32, 64, 128],
            "n_warps": [4, 8],
            "n_stages": [1, 2],
            "a_pad": [0, 8, 16],
            "b_pad": [0, 8, 16],
            "vec_epilogue": [False, True],
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

        # Sentinel-skip for ``moe_router_correct`` filler chunks. The
        # work_list expert is U32 by API convention (it's used below as
        # a row offset), so re-cast to S32 here to make the ``>= 0``
        # compare detect the -1 sentinel as signed. ``expert`` is
        # uniform across the block (single gmem load at block entry),
        # so this is a clean uniform branch.
        is_valid = qk.cmp("ge", qk.bitcast(expert, DType.S32), bctx.c(0, dtype=DType.S32))
        with qk.if_(is_valid, carried=[]) as (_, _, arms):
            with arms.then_():
                b_row_base = expert * s.H + n_base

                toks = qk.index_cache("toks_cache", g.token_ids, count=c.BM, base=grp_start)
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
                K_outer = s.D // c.BK

                def produce(ictx: IterCtx) -> None:
                    plan = ictx.stage
                    k_col_a = ictx.iter_idx * bk_stride_a
                    k_col_b = k_col_a if bk_stride_b == bk_stride_a else ictx.iter_idx * bk_stride_b
                    plan.a.gather_from(g.X, index=toks, col=k_col_a, cast=a_cast)
                    plan.b.load_from(g.W_in, row=b_row_base, col=k_col_b, cast=b_cast)

                PipelineBody(
                    stages=stages,
                    produce=produce,
                    consume=mma,
                    carry=acc,
                ).run(n_iters=K_outer, n_stages=c.n_stages)

                warp_col_base = n_base + bctx.warp_id * BN_per_warp

                # SiLU + cast + store. ``activation="silu"`` fuses silu inside
                # the per-element store body — required on MSL until the
                # lowerer can compose frag-derived values (memory:
                # project_lowerer_frag_compose). Vectorized through staging
                # smem when c.vec_epilogue, else direct scalar.
                if c.vec_epilogue:
                    silu_stage = qk.smem_alloc("silu_stage", s.out_dtype, (c.BM, c.BN))
                    qk.store_acc(
                        g.H_out,
                        acc,
                        row=grp_start,
                        col=warp_col_base,
                        cast=s.out_dtype,
                        activation="silu",
                        stage_in_smem=True,
                        staging_smem=silu_stage,
                        smem_col_offset=bctx.warp_id * BN_per_warp,
                        gmem_col_base_full=n_base,
                    )
                else:
                    qk.store_acc(
                        g.H_out,
                        acc,
                        row=grp_start,
                        col=warp_col_base,
                        cast=s.out_dtype,
                        activation="silu",
                    )
                qk.yield_()
            with arms.else_():
                qk.yield_()
