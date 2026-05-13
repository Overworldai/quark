"""Kernel-authoring memory helpers — work-list + index-cache loads.

Free-function replacements for the former ``WorkListLoad`` /
``IndexCache`` L1/L2 dataclass wrappers. Each consumes a
:class:`GlobalTensor` and returns a Value / tuple / populated
:class:`SharedRegion` — no intermediate dataclass, no ``.emit(ctx)``
step. ``bctx`` is resolved via :func:`active_bctx`.
"""

from __future__ import annotations

import quark.lang as qk
from quark.blocks.dsl import active_bctx
from quark.ir import DType, GlobalTensor, SharedRegion, Value


def work_list_load(work_list: GlobalTensor, work_idx: Value) -> tuple[Value, Value]:
    """Read a ``(grp_start, expert)`` pair from a flat ``int32`` work list.

    ``work_list`` is ``[n_items * 2]`` int32, packed as ``(grp_start,
    expert)`` pairs. ``work_idx`` is the block index selecting which
    pair to read::

        grp_start, expert = qk.work_list_load(g.work_list, block_idx("y"))

    Both values are returned as ``U32`` (downstream code uses them as
    row offsets / smem indices). Routers that emit a sentinel-skip
    marker (``expert == -1`` for filler chunks) must bitcast back to
    S32 before comparing — see the inproj/outproj kernels for the
    pattern. The bitcast here is cosmetic for grp_start but required
    for expert to keep the row-arithmetic dtype consistent.
    """
    bctx = active_bctx()
    wl_base = qk.mul(work_idx, bctx.c(2))
    grp_start = qk.bitcast(qk.load(work_list, wl_base), DType.U32)
    expert = qk.bitcast(qk.load(work_list, qk.add(wl_base, bctx.c(1))), DType.U32)
    return grp_start, expert


def q_register_load(
    g_q: GlobalTensor,
    *,
    smem: SharedRegion,
    q_row: Value,
    warp_id: Value,
    MT: int,
    Dh: int,
    kv_pad: int = 0,
) -> list[list[Value]]:
    """cp.async a warp's Q rows → smem → register A-fragments.

    Each warp loads its own Q rows into a per-warp view of ``smem``,
    then every lane reads its MMA A-fragment positions from smem via
    ``load_matrix``. Returns ``q_frags[mt][kk_step]`` — a nested list
    of width-N b32 Values ready to feed ``qk.mma`` (or the Triton-
    style ``MmaBody(a=q_frags, ...)`` form).

    The smem allocation is owned by the caller so it can be reused
    as epilogue output staging after the KV loop — Q is dead by the
    time the epilogue runs.
    """
    bctx = active_bctx()
    cfg = bctx.mma_cfg
    mma_k = cfg.mma_k
    KK_STEPS = Dh // mma_k
    m_stride = cfg.shape.m
    warp_rows = MT * m_stride
    stride_elems = Dh + kv_pad

    # Per-warp smem view: offset by warp_id * warp_rows * stride.
    warp_dyn = warp_id * (warp_rows * stride_elems)
    warp_smem = smem.view(
        dyn_offset=warp_dyn,
        shape=(warp_rows, Dh),
        name="Q_warp",
    )

    # cp.async Q tile (each warp loads its own rows, subgroup-width
    # threads). ``bctx.subgroup_size`` is 32 on the historical path
    # and 16 when ``KernelConfig.subgroup_size = 16``.
    sgs = bctx.subgroup_size
    lane_id = bctx.tid % bctx.c(sgs)
    warp_smem.copy_from(
        g_q.tile(row=q_row, col=0, shape=(warp_rows, Dh)),
        tid=lane_id,
        n_threads=sgs,
        async_load=True,
    )
    qk.async_commit()
    qk.async_wait(0)
    qk.barrier("block")

    # Per-warp lane view: fold warp offset + lane offset.
    lane_off = bctx.gid * stride_elems + bctx.tig * cfg.lane_col_step
    warp_lane = smem.view(
        dyn_offset=warp_dyn + lane_off,
        shape=(warp_rows, Dh),
        name="Q_warp_lane",
    )

    q_frags: list[list[Value]] = []
    for mt in range(MT):
        mt_frags: list[Value] = []
        for kk_step in range(KK_STEPS):
            kk = kk_step * mma_k
            frag = qk.load_matrix(
                warp_lane,
                cfg.shape_id,
                which="a",
                row=mt * m_stride,
                col=kk,
                reg_offsets=cfg.a_offsets,
            )
            mt_frags.append(frag)
        q_frags.append(mt_frags)
    return q_frags


def index_cache(
    name: str,
    gmem: GlobalTensor,
    *,
    count: int,
    base: Value | None = None,
    dtype: DType | None = None,
) -> SharedRegion:
    """Cache a 1D gmem array in smem — returns the populated region.

    Allocates ``smem[0..count-1]`` and cooperatively loads
    ``gmem[base..base+count-1]`` into it. Used by the MoE kernels to
    cache ``token_ids[grp_start..+BM]`` / ``slot_weights[...]`` once
    per work item so the gather loader + scatter epilogue can read
    per-row indices from smem rather than gmem.

    Callers should issue ``qk.barrier("block")`` before reading the
    cached values (the cooperative load stores to smem but the
    downstream readers may be on different threads).
    """
    from quark.blocks.l0.index_cache import emit_index_cache

    bctx = active_bctx()
    dt = dtype or gmem.dtype
    smem = qk.smem_alloc(name, dt, (count,))
    gmem_base = base if base is not None else bctx.c(0)
    emit_index_cache(
        bctx.bld,
        gmem=gmem,
        smem=smem,
        count=count,
        gmem_base_idx=gmem_base,
        tid=bctx.tid,
        n_threads=bctx.n_threads,
    )
    return smem
