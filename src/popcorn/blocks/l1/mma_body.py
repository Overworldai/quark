"""L1: MmaBody — triple-nested MMA tile loop.

Supports per-warp N partitioning: when n_offset > 0, the body computes
N-tiles starting at n_offset instead of 0. This lets different warps
handle different output column ranges from the same shared B smem tile.

Supports two B-fragment load layouts via ``b_shuffled``:

* Default (``b_shuffled=False``) — standard row-major B smem with per-lane
  gid/tig offsets; loads each fragment register with a scalar
  ``ld.shared.b32`` via ``load_matrix``.
* ``b_shuffled=True`` — B smem is pre-permuted offline so each lane's
  entire fragment is ``FRAG_BYTES`` contiguous bytes; emits one
  ``ld.shared.v{N}.b32`` per fragment. A fragments are still scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import popcorn.lang as pop
from popcorn.blocks.dsl import Accumulators, Block, BlockContext, active_bctx
from popcorn.blocks.l0.mma_tile import emit_mma_tile
from popcorn.blocks.l0.shuffled_bfrag_load import emit_shuffled_bfrag_load
from popcorn.ir import SharedRegion, Value


def _resolve_grid(
    mma: MmaBody,  # type: ignore[name-defined]
    cfg: Any,
    *,
    ki_hint: int | None = None,
) -> tuple[int, int, int]:
    """Derive (MT, NT, K_inner) for the tile loop, priority:
    instance override > ``self.acc``-derived > ``self.BK`` > ``ki_hint``
    (inferred from the B operand's K-dim at call time).
    """
    mt = mma.MT if mma.MT is not None else (mma.acc.MT if mma.acc else 0)
    nt = mma.NT if mma.NT is not None else (mma.acc.NT if mma.acc else 0)
    if mma.K_inner is not None:
        ki = mma.K_inner
    elif mma.BK is not None:
        ki = mma.BK // cfg.mma_k
    elif ki_hint is not None:
        ki = ki_hint
    else:
        ki = 0
    if mt == 0 or nt == 0 or ki == 0:
        raise ValueError(
            f"MmaBody: grid underspecified — MT={mt}, NT={nt}, K_inner={ki}. "
            f"Pass MT/NT/K_inner (or an Accumulators + BK)."
        )
    return mt, nt, ki


def _infer_k_inner_from_b(b: Any, mma_k: int) -> int | None:
    """Infer ``K_inner`` from a B operand when possible.

    Convention: for ``SmemTile`` B tiles, the K (reduction) dimension is
    ``shape[1]``. Holds for GEMM (``b_shape=(BN, BK)``) and for
    attention's GEMM1 (K tile ``(KvTile, Dh)`` → K=Dh) / GEMM2 (V^T tile
    ``(Dh, KvTile)`` → K=KvTile).

    Returns None when B is a pre-loaded register tile (``list[list[Value]]``)
    or a raw SharedRegion — callers must provide ``K_inner`` explicitly.
    """
    from popcorn.blocks.dsl import SmemTile

    if isinstance(b, SmemTile):
        k_dim = b.shape[1]
        return k_dim // mma_k
    return None


@dataclass
class MmaBody(Block):
    """Emit the triple-nested (K_inner, MT, NT) MMA tile loop.

    Constructor takes the MMA shape plus a paired ``Accumulators`` and
    ``BK`` so ``MT`` / ``NT`` / ``K_inner`` derive automatically. Call
    the instance like ``mma(stage, carry)`` inside the K-loop:

        mma = MmaBody(shape=mma_cfg, acc=acc, BK=c.BK, b_shuffled=...)
        def consume(ictx):
            return mma(ictx.stage, ictx.carry)

    Kernels whose inner-K doesn't match ``BK // mma_k`` (attention's
    GEMM2 runs ``KvTile // mma_k``) pass an explicit ``K_inner=``.

    ``n_offset`` — number of N-tiles to skip (per-warp N partitioning).
    When n_offset > 0, B fragments are loaded from
    ``n_tile = n_offset + local_n``; accumulator indexing stays local.

    ``b_shuffled`` — True for vectorized shuffled B-fragment loads. The
    B smem lane view must have its dyn_offset encoded for the shuffled
    layout (see emit_shuffled_bfrag_load docstring).
    """

    shape: Any = None  # MmaConfig — uses ctx.mma_cfg if None
    acc: Accumulators | None = None  # when set, MT/NT/K_inner derive
    BK: int | None = None  # inner-K loop bound source (K_inner = BK // mma_k)
    MT: int | None = None  # override (attn uses this explicitly)
    NT: int | None = None  # override
    K_inner: int | None = None  # override (attn GEMM2 uses explicit K_inner)
    n_offset: int = 0  # per-warp N-tile offset
    b_shuffled: bool = False

    def __call__(self, *args, a=None, b=None, acc=None) -> Any:
        """Three call shapes — pick whichever fits the kernel body:

        * ``mma(ictx)`` — GEMM-style: reads the SmemPlan on
          ``ictx.stage`` + ``ictx.carry``, returns the updated carry.
          The consume callback:

              def consume(ictx):
                  return mma(ictx)

        * ``mma(stage, carry)`` — explicit SmemPlan form when carry
          isn't ``ictx.carry`` directly.

        * ``mma(a=, b=, acc=)`` — Triton-style ``tl.dot`` form:
          pass the A and B operand views directly. Each may be:

            - :class:`SmemTile` — B fragments (and A if requested)
              load via ``load_matrix`` on ``.lane`` in the inner loop.
            - Pre-loaded register fragments as ``list[list[Value]]``
              indexed by ``[m_tile][inner_k]`` — each entry is a
              width-N Value fed straight into ``pop.mma`` (no smem
              load). Used by attention's GEMM1 (pre-loaded Q) and
              GEMM2 (softmax-produced P fragments).

          ``acc`` is the list of accumulator Values to feed as the C
          input to every ``pop.mma`` in the tile loop; return value
          is the updated list. ``ctx`` / ``bctx`` is implicit via
          ``active_bctx()``.
        """
        from popcorn.blocks.l2.run_pipeline import IterCtx
        from popcorn.blocks.l2.smem_plan import SmemPlan

        # Triton-style kwargs form — dispatch on operand types.
        if a is not None or b is not None:
            if acc is None:
                raise TypeError("MmaBody(...): acc= is required with a=/b= call form")
            return self._dot(a=a, b=b, acc=list(acc))

        if len(args) == 1:
            ictx = args[0]
            if not isinstance(ictx, IterCtx):
                raise TypeError(
                    f"MmaBody(ictx): expected IterCtx, got {type(ictx).__name__}. "
                    f"Use mma(ictx) inside a PipelineBody.consume callback, "
                    f"mma(stage, carry) for the explicit SmemPlan form, or "
                    f"mma(a=, b=, acc=) for the Triton-style form."
                )
            stage, carry = ictx.stage, ictx.carry
        elif len(args) == 2:
            stage, carry = args
        else:
            raise TypeError(
                f"MmaBody(...): expected 1 (ictx), 2 (stage, carry), or "
                f"keyword-only (a=, b=, acc=) call form; got {len(args)} positional args"
            )

        if not isinstance(stage, SmemPlan):
            raise TypeError(
                f"MmaBody(stage, carry): stage must be a SmemPlan (got {type(stage).__name__})."
            )
        if not stage.a_lane or not stage.b_lane:
            raise ValueError(
                "MmaBody(stage, carry): stage.a_lane / stage.b_lane are empty — "
                "the SmemPlan was constructed but never emitted (no active BlockContext?)."
            )
        if not isinstance(carry, (list, tuple)):
            raise TypeError(
                f"MmaBody(stage, carry): carry must be list[Value] or tuple[Value, ...]; "
                f"got {type(carry).__name__}."
            )

        # Infer K_inner from the SmemPlan's B SmemTile spec (axis 1) so
        # GEMM/MoE callers don't need to pass ``BK=`` explicitly.
        ctx = active_bctx()
        cfg = self.shape if self.shape is not None else ctx.mma_cfg
        ki_hint = (
            stage.b.shape[1] // cfg.mma_k if self.K_inner is None and self.BK is None else None
        )
        return tuple(
            self.emit_with(
                ctx,
                a_lane=stage.a_lane[0],
                b_lane=stage.b_lane[0],
                acc=list(carry),
                K_inner=ki_hint,
            )
        )

    def _dot(self, *, a: Any, b: Any, acc: list[Value]) -> list[Value]:
        """Triton-style dispatch: A/B can be smem (SmemTile / SharedRegion)
        or pre-loaded register fragments (list[list[Value]] indexed by
        [m_tile][inner_k]). Loops the triple-nested (inner_k, m, n) grid
        and returns the updated accumulator list.

        ``K_inner`` is inferred from ``b.shape[1]`` when ``b`` is a
        SmemTile (convention: the reduction dim of the B tile is
        axis 1) and ``self.K_inner`` / ``self.BK`` weren't set. Callers
        whose B layout breaks the convention fall back to the
        explicit ``MmaBody(K_inner=...)`` kwarg.
        """
        from popcorn.blocks.dsl import SmemTile

        # Resolve ctx and mma config.
        ctx = active_bctx()
        cfg = self.shape if self.shape is not None else ctx.mma_cfg
        mma_k = cfg.mma_k
        ki_inferred = _infer_k_inner_from_b(b, mma_k)
        mt_count, nt_count, ki = _resolve_grid(self, cfg, ki_hint=ki_inferred)
        n_off = self.n_offset
        m_stride = cfg.shape.m
        n_stride = cfg.shape.n

        def _smem_lane(operand: Any, side: str) -> SharedRegion | None:
            if operand is None:
                return None
            if isinstance(operand, SmemTile):
                return operand.lane
            if isinstance(operand, SharedRegion):
                return operand
            if isinstance(operand, list):
                return None  # register-tile path
            raise TypeError(
                f"MmaBody({side}=): expected SmemTile, SharedRegion, or "
                f"list[list[Value]]; got {type(operand).__name__}"
            )

        a_smem = _smem_lane(a, "a")
        b_smem = _smem_lane(b, "b")

        for inner_k in range(ki):
            kk = inner_k * mma_k
            for m in range(mt_count):
                for n in range(nt_count):
                    i = m * nt_count + n
                    # A fragment: register-tile vs smem load.
                    if a_smem is None:
                        a_frag = a[m][inner_k]  # type: ignore[index]
                    else:
                        a_frag = pop.load_matrix(
                            a_smem,
                            cfg.shape_id,
                            which="a",
                            row=m * m_stride,
                            col=kk,
                            reg_offsets=cfg.a_offsets,
                        )
                    # B fragment: register vs smem (shuffled or flat).
                    if b_smem is None:
                        b_frag = b[n][inner_k]  # type: ignore[index]
                    elif self.b_shuffled:
                        b_frag = emit_shuffled_bfrag_load(
                            ctx,
                            b_smem,
                            frag_regs=cfg.shape.b_regs,
                            n_tile=n_off + n,
                            k_step=inner_k,
                        )
                    else:
                        b_frag = pop.load_matrix(
                            b_smem,
                            cfg.shape_id,
                            which="b",
                            row=(n_off + n) * n_stride,
                            col=kk,
                            reg_offsets=cfg.b_offsets,
                        )
                    acc[i] = pop.mma(cfg.shape_id, a_frag, b_frag, acc[i])
        return acc

    def emit_with(
        self,
        ctx: BlockContext,
        a_lane: SharedRegion | None,
        b_lane: SharedRegion,
        acc: list[Value],
        K_inner: int | None = None,
        MT: int | None = None,
        NT: int | None = None,
        n_offset: int | None = None,
        a_source: Any | None = None,
    ) -> list[Value]:
        """Lower-level emit interface. Prefer ``mma(stage, carry)`` at
        the kernel call site; this method remains the extension point
        for blocks that need per-call MT/NT/K_inner overrides.

        ``a_source`` (optional) — callable ``(m_tile, inner_k) -> Value``
        returning a pre-loaded A fragment. When set, the A-smem load is
        skipped (``a_lane`` can be ``None``). Used by attention's
        GEMM1 (Q pre-loaded to registers via ``QRegisterLoad``) and
        GEMM2 (A is the P fragment from online softmax). B still loads
        from ``b_lane`` smem via ``load_matrix``.
        """
        cfg = self.shape if self.shape is not None else ctx.mma_cfg
        mma_k = cfg.mma_k

        # Resolve MT / NT / K_inner in priority order:
        #   call-site kwarg  >  instance override  >  derived from acc/BK
        mt = (
            MT
            if MT is not None
            else (self.MT if self.MT is not None else (self.acc.MT if self.acc else 0))
        )
        nt = (
            NT
            if NT is not None
            else (self.NT if self.NT is not None else (self.acc.NT if self.acc else 0))
        )
        ki = (
            K_inner
            if K_inner is not None
            else (
                self.K_inner
                if self.K_inner is not None
                else (self.BK // mma_k if self.BK is not None else 0)
            )
        )
        n_off = n_offset if n_offset is not None else self.n_offset

        m_stride = cfg.shape.m
        n_stride = cfg.shape.n
        for inner_k in range(ki):
            kk = inner_k * mma_k
            for m in range(mt):
                for n in range(nt):
                    i = m * nt + n  # local accumulator index
                    if a_source is not None:
                        a_frag = a_source(m, inner_k)
                        b_frag = pop.load_matrix(
                            b_lane,
                            cfg.shape_id,
                            which="b",
                            row=(n_off + n) * n_stride,
                            col=kk,
                            reg_offsets=cfg.b_offsets,
                        )
                        acc[i] = pop.mma(cfg.shape_id, a_frag, b_frag, acc[i])
                    elif self.b_shuffled:
                        a_frag = pop.load_matrix(
                            a_lane,
                            cfg.shape_id,
                            which="a",
                            row=m * m_stride,
                            col=kk,
                            reg_offsets=cfg.a_offsets,
                        )
                        b_frag = emit_shuffled_bfrag_load(
                            ctx,
                            b_lane,
                            frag_regs=cfg.shape.b_regs,
                            n_tile=n_off + n,
                            k_step=inner_k,
                        )
                        acc[i] = pop.mma(cfg.shape_id, a_frag, b_frag, acc[i])
                    else:
                        assert a_lane is not None, (
                            "MmaBody: a_lane required unless a_source= is provided"
                        )
                        acc[i] = emit_mma_tile(
                            ctx.bld,
                            a_smem_lane=a_lane,
                            b_smem_lane=b_lane,
                            shape_id=cfg.shape_id,
                            a_offsets=cfg.a_offsets,
                            b_offsets=cfg.b_offsets,
                            cd_offsets=cfg.cd_offsets,
                            m_tile=m,
                            n_tile=n_off + n,
                            kk=kk,
                            acc_in=acc[i],
                            m_stride=m_stride,
                            n_stride=n_stride,
                        )
        return acc
