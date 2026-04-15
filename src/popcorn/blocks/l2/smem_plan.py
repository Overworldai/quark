"""L2: SmemPlan — allocate A/B smem tiles from SmemTile specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from popcorn.blocks.dsl import BlockContext, SetupBlock, SmemTile
from popcorn.ir import DType, SharedRegion

if TYPE_CHECKING:
    from popcorn.ir.mma_registry import MmaConfig


@dataclass
class SmemPlan(SetupBlock):
    """Declarative smem plan for paired A/B tiles.

    Constructed from SmemTile specs. Auto-emits under an active
    BlockContext (see ``SetupBlock``) — after construction, the plan
    exposes ``a_stages`` / ``b_stages`` (SharedRegion) and
    ``a_lane`` / ``b_lane`` (per-lane views).
    """

    a: SmemTile
    b: SmemTile
    n_stages: int = 1
    lane_col_step: int = 2

    # Populated after emit
    a_stages: list[SharedRegion] = field(default_factory=list, repr=False)
    b_stages: list[SharedRegion] = field(default_factory=list, repr=False)
    a_lane: list[SharedRegion] = field(default_factory=list, repr=False)
    b_lane: list[SharedRegion] = field(default_factory=list, repr=False)

    def emit(self, ctx: BlockContext) -> None:
        """Allocate smem and compute per-lane views.

        Threads the cached ``gid`` / ``tig`` from the BlockContext into
        ``emit_smem_base`` so repeated calls (e.g. per-stage plans on
        the same kernel) share one pair of ``GroupIdOp`` /
        ``ThreadIdInGroupOp`` ops rather than emitting fresh ones each
        time.
        """
        from popcorn.blocks.l0.smem_base import emit_smem_base

        b = ctx.bld
        a_dt, a_shape, a_pad = self.a.dtype, self.a.shape, self.a.pad
        b_dt, b_shape, b_pad = self.b.dtype, self.b.shape, self.b.pad
        BM, BK_A = a_shape
        # B may declare its own K width distinct from A's — e.g. shuffled B
        # bakes the bank-conflict padding into its data, so b_shape is
        # (BN, BK + b_pad) with SmemPlan's b_pad=0. A's K stays BK.
        BN, BK_B = b_shape

        a_smem = b.smem_alloc("A_smem", a_dt, (self.n_stages * BM, BK_A), pad=a_pad)
        b_smem = b.smem_alloc("B_smem", b_dt, (self.n_stages * BN, BK_B), pad=b_pad)

        a_stage_elems = BM * (BK_A + a_pad)
        b_stage_elems = BN * (BK_B + b_pad)

        self.a_stages = [
            a_smem.view(static_offset_add=st * a_stage_elems, shape=(BM, BK_A), name=f"A_s{st}")
            for st in range(self.n_stages)
        ]
        self.b_stages = [
            b_smem.view(static_offset_add=st * b_stage_elems, shape=(BN, BK_B), name=f"B_s{st}")
            for st in range(self.n_stages)
        ]

        step = self.a.lane_col_step or self.lane_col_step
        self.a_lane = [emit_smem_base(b, s, step, gid=ctx.gid, tig=ctx.tig) for s in self.a_stages]
        step_b = self.b.lane_col_step or self.lane_col_step
        self.b_lane = [
            emit_smem_base(b, s, step_b, gid=ctx.gid, tig=ctx.tig) for s in self.b_stages
        ]

        # Wire SmemTile specs to first stage so TileLoad can find them.
        self.a.smem = self.a_stages[0]
        self.a.lane = self.a_lane[0]
        self.b.smem = self.b_stages[0]
        self.b.lane = self.b_lane[0]

    @classmethod
    def paired(
        cls,
        name: str,
        dtype: DType,
        *,
        a_shape: tuple[int, int],
        b_shape: tuple[int, int],
        a_pad: int = 0,
        b_pad: int = 0,
        mma_cfg: MmaConfig,
        n_warps: int,
        b_shuffled: bool = False,
    ) -> SmemPlan:
        """Allocate one paired A/B stage with MMA-aware per-lane B view.

        Absorbs the stage-loop + ``warp_lane_view`` fixup every GEMM-
        shaped kernel replicates. Handles the two orthogonal concerns:

        * **pad placement** — when ``b_shuffled`` + ``b_pad > 0``, the
          pad is baked into the shuffled B's K dimension (physical
          ``BK + b_pad``) and ``SmemTile.pad`` is 0. Unshuffled B
          carries the pad as a plain ``SmemTile.pad``.
        * **per-lane B view** — shuffled B uses ``lane_id * frag_elems``
          (each lane holds a contiguous fragment block); unshuffled
          uses the standard ``gid * row_stride + tig * lane_col_step``.

        ``a_shape`` / ``b_shape`` are the *logical* tile shapes (BM, BK)
        / (BN, BK); the classmethod adjusts ``b_shape[1]`` internally
        when the shuffled+pad path applies.

        Call site reduces from ~30 lines to one construction per stage:

            stages = [
                SmemPlan.paired(f"stage{i}", compute,
                                a_shape=(c.BM, c.BK), b_shape=(c.BN, c.BK),
                                a_pad=c.a_pad, b_pad=c.b_pad,
                                mma_cfg=mma_cfg, n_warps=c.n_warps,
                                b_shuffled=s.b_shuffle)
                for i in range(c.n_stages)
            ]
        """
        from popcorn.blocks.dsl import _ACTIVE_BCTX

        BM, BK = a_shape
        BN, BK_b_logical = b_shape
        shuffle_with_pad = b_shuffled and b_pad > 0
        b_cols = BK_b_logical + b_pad if shuffle_with_pad else BK_b_logical
        b_smem_pad = 0 if shuffle_with_pad else b_pad

        plan = cls(
            a=SmemTile(
                f"A_{name}", dtype, (BM, BK), pad=a_pad, lane_col_step=mma_cfg.lane_col_step
            ),
            b=SmemTile(
                f"B_{name}",
                dtype,
                (BN, b_cols),
                pad=b_smem_pad,
                lane_col_step=mma_cfg.lane_col_step,
            ),
        )
        # SetupBlock.__post_init__ has already run plan.emit(active_bctx())
        # at ``cls(...)`` construction above. Now stamp the per-lane B
        # view with the b_shuffled-aware lane_offset.
        bctx = _ACTIVE_BCTX.get()
        if bctx is None:
            return plan  # outside a kernel build — no per-lane override to apply

        n_tile = mma_cfg.shape.n
        BN_per_warp = (BN // n_tile // n_warps) * n_tile
        if b_shuffled:
            frag_bytes = mma_cfg.shape.b_regs * 4
            lane_frag_elems = frag_bytes // dtype.bytes
            lane_offset = bctx.lane_id * lane_frag_elems
            step = None
        else:
            lane_offset = None
            step = mma_cfg.lane_col_step

        plan.b_lane = [
            plan.b.smem.warp_lane_view(
                rows=BN_per_warp,
                warp_id=bctx.warp_id,
                lane_col_step=step,
                lane_offset=lane_offset,
            )
        ]
        return plan

    @classmethod
    def staged_pairs(
        cls,
        dtype: DType,
        *,
        a_shape: tuple[int, int],
        b_shape: tuple[int, int],
        a_pad: int = 0,
        b_pad: int = 0,
        mma_cfg: MmaConfig,
        n_warps: int,
        b_shuffled: bool = False,
        n_stages: int,
    ) -> list[SmemPlan]:
        """``n_stages`` paired A/B SmemPlans in one call — the
        GEMM-shaped default. Each stage is constructed via
        ``SmemPlan.paired(f"s{i}", ...)`` so the stage index lands in
        the allocated smem's debug name.

        Collapses the ``[SmemPlan.paired(...) for i in range(n_stages)]``
        list comprehension to one line at the kernel call site:

            stages = SmemPlan.staged_pairs(
                compute, a_shape=..., b_shape=..., ..., n_stages=c.n_stages
            )
        """
        return [
            cls.paired(
                f"s{i}",
                dtype,
                a_shape=a_shape,
                b_shape=b_shape,
                a_pad=a_pad,
                b_pad=b_pad,
                mma_cfg=mma_cfg,
                n_warps=n_warps,
                b_shuffled=b_shuffled,
            )
            for i in range(n_stages)
        ]
