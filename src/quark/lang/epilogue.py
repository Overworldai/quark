"""High-level epilogue helpers — frag/accumulator-tile transforms +
gmem/atomic stores. The user-facing surface that replaces the
ScatterStore / SiluCastStore / AtomicScatterAdd L1 Block trio.

EXEMPT FROM 500-LINE RULE: ``store_acc`` / ``atomic_store_acc`` share
the same frag_for_each scatter body + per-slot scale / staged smem /
cooperative vec_store plumbing; splitting keeps the staged and direct
paths in lockstep and avoids duplicating the row-class / slot-selector
math.

The Block classes survived only because there was no free-function
form; the per-element scatter math, smem staging, and atomic_rmw
emission stayed coupled to a Block subclass each. These helpers lift
the same emission paths out:

    pop.silu(values)            # element-wise silu on a vec4 acc list
    pop.cast(values, dtype)     # frag-layout dtype conversion
    pop.store_acc(dst, values, *, row, col, ...)            # vec_store epilogue
    pop.atomic_store_acc(dst, values, *, row, col, ...)     # atomic-add scatter

The implementation delegates to the existing ``emit_*_epilogue``
functions in ``quark.blocks.epilogue`` so IR emission stays
byte-for-byte identical with the legacy Block path.

"""

from __future__ import annotations

import math
from typing import Any

from quark.blocks.dsl import Accumulators, BlockContext, active_bctx
from quark.ir import DType, GlobalTensor, SharedRegion, Value

_LOG2E = math.log2(math.e)


# ---------------------------------------------------------------
# Per-element vec4 transforms (operate on accumulator-tile result lists).
# ---------------------------------------------------------------


def silu(values: list[Value]) -> list[Value]:
    """SiLU(x) = x * sigmoid(x) on a list of vec accumulator Values.

    Returns a new list of the same shape (one Value per accumulator
    tile). Stays in registers — emits per-vec ops without smem.

    Implementation uses the identity ``sigmoid(x) = rcp(1 + ex2(-x*log2e))``
    so we can lower through the fast approximate path on both PTX and
    MSL. Constants are broadcast to each Value's vector width via
    ``vec_build`` so f32x2 / f32x4 accumulators all work.
    """
    if not values:
        return []
    from quark.lang import const, ex2_approx, neg, rcp_approx, vec_build

    bctx = active_bctx()
    width = values[0].width
    log2e_scalar = bctx.c(_LOG2E)
    one_scalar = bctx.c(1.0)
    log2e = vec_build([log2e_scalar] * width) if width > 1 else log2e_scalar
    one = vec_build([one_scalar] * width) if width > 1 else one_scalar
    _ = const  # quiet ruff if unused
    out: list[Value] = []
    for v in values:
        sig = rcp_approx(one + ex2_approx(neg(v * log2e)))
        out.append(v * sig)
    return out


def cast(values: list[Value], dtype: DType) -> list[Value]:
    """Per-element dtype conversion on a list of vec4 accumulator Values.

    Returns a new list of the same shape, with each Value converted
    to ``dtype``. No-op when ``dtype`` matches the source dtype.
    """
    if not values:
        return []
    if values[0].dtype == dtype:
        return list(values)
    from quark.lang import convert

    return [convert(v, dtype) for v in values]


# ---------------------------------------------------------------
# Accumulator-tile gmem stores.
# ---------------------------------------------------------------


def store_acc(
    dst: GlobalTensor,
    acc: Accumulators,
    *,
    row: Value | int,
    col: Value | int,
    cast: DType | None = None,
    activation: str | None = None,
    bias: GlobalTensor | None = None,
    stage_in_smem: bool = False,
    staging_smem: SharedRegion | None = None,
    smem_col_offset: Value | None = None,
    gmem_col_base_full: Value | int | None = None,
    row_scale: list[Value] | None = None,
    per_warp: bool = False,
    warp_id: Value | None = None,
    kv_pad: int = 0,
    use_staging_smem: bool = True,
    atomic: bool = False,
) -> None:
    """Store an accumulator tile result to ``dst`` at ``(row, col)``.

    Pulls the per-tile values from ``acc.results`` (populated by
    ``run_pipeline`` when ``acc`` was passed as ``PipelineBody.carry``).
    For accumulator state that didn't originate from ``run_pipeline``
    (rare — attn-style kernels), set ``acc.results`` manually before
    calling.

    * ``stage_in_smem=True`` (with ``staging_smem=`` supplied) routes
      through smem and emits a cooperative 16B vec_store per row tile.
      The vectorized-epilogue fast path used by moe_inproj.
    * ``stage_in_smem=False`` (default) emits per-lane scalar stores
      to gmem — works for any shape, doesn't saturate gmem bandwidth.

    Attention-style normalize epilogue (replaces ``NormalizeAndStore``):

    * ``row_scale=`` — per-(mt, rc) Values whose **reciprocal** scales
      each accumulator element before cast. Length must equal
      ``acc.MT * n_rc`` where ``n_rc`` is the number of row classes
      (distinct ``dr`` in ``mma_cfg.cd_offsets`` — 2 for m16n8, 1 for
      m8n8k8). The classic ``O /= l`` pattern: pass ``l_vals`` and the
      epilogue rcps + multiplies per slot.
    * ``per_warp=True`` (with ``warp_id=``) stages into a per-warp
      slice of ``staging_smem`` and runs the cooperative store with
      the warp's 32 lanes (``tid % 32``) instead of the full block.
      Required when each warp owns a disjoint output row range.
    * ``kv_pad=`` — extra padding columns in the per-warp staging
      stride (``Dh + KvPad`` elements per row); only used to compute
      the per-warp ``dyn_offset``.

    ``cast=`` casts each element on the way out.
    """
    if acc.results is None:
        raise ValueError(
            "pop.store_acc: acc.results is None. Pass the Accumulators as "
            "PipelineBody.carry so run_pipeline populates the loop-final values, "
            "or set acc.results manually before calling."
        )
    bctx = active_bctx()

    if activation is not None and activation != "silu":
        raise ValueError(
            f"pop.store_acc: unsupported activation {activation!r}; only 'silu' or None"
        )
    if per_warp and warp_id is None:
        # Default to the active BlockContext's warp_id — the only sensible
        # value for per-warp staging (one warp → one output row range).
        # Override is still allowed for kernels that need a permuted
        # warp-to-rows mapping.
        warp_id = bctx.warp_id
    if row_scale is not None and not stage_in_smem and not per_warp:
        # Direct (non-staged) path with scaling — only supported for the
        # per-warp attention pattern today; lift if a non-attn kernel needs it.
        pass

    if stage_in_smem or per_warp:
        if staging_smem is None:
            if not use_staging_smem:
                raise ValueError(
                    "pop.store_acc(stage_in_smem=True or per_warp=True) "
                    "with use_staging_smem=False: staging_smem= is required."
                )
            staging_smem = _auto_alloc_staging(
                bctx,
                acc=acc,
                cast=cast or DType.BF16,
                per_warp=per_warp,
                kv_pad=kv_pad,
            )
        _emit_staged_store(
            bctx,
            dst=dst,
            acc_results=acc.results,
            acc=acc,
            row_base=row,
            col_base=col,
            cast=cast or DType.BF16,
            activation=activation,
            staging_smem=staging_smem,
            smem_col_offset=smem_col_offset,
            gmem_col_base_full=gmem_col_base_full if gmem_col_base_full is not None else col,
            row_scale=row_scale,
            per_warp=per_warp,
            warp_id=warp_id,
            kv_pad=kv_pad,
        )
        return

    # Paired bf16x2 atomic scatter — halves the atomic op count vs scalar
    # bf16 atomics. Used when: atomic=True, output is BF16, and the MMA
    # shape's cd_offsets walk in adjacent-column pairs (true for all
    # m16n8k16_bf16 / m16n8k8_f16 shapes). Device-cap gating of
    # (BF16, 2) in caps.atomic_add_vector happens in is_valid_for, so
    # configs that emit this on unsupported hardware get autotune-filtered.
    _cast = cast or DType.F32
    if atomic and _cast == DType.BF16 and _cd_offsets_pairable(bctx.mma_cfg.cd_offsets):
        _emit_scatter_store_paired_bf16x2(
            bctx,
            dst=dst,
            acc_results=acc.results,
            acc=acc,
            row_base=row,
            col_base=col,
            activation=activation,
            bias=bias,
            row_scale=row_scale,
        )
        return

    _emit_scatter_store(
        bctx,
        dst=dst,
        acc_results=acc.results,
        acc=acc,
        row_base=row,
        col_base=col,
        cast=_cast,
        activation=activation,
        bias=bias,
        row_scale=row_scale,
        atomic=atomic,
    )


def atomic_store_acc(
    dst: GlobalTensor,
    acc: Accumulators,
    *,
    col: Value | int,
    index: SharedRegion,
    weight: SharedRegion | None = None,
    op: str = "add",
) -> None:
    """Atomic scatter to ``dst`` keyed by per-row indices.

    For each (mt, nt, row, col) accumulator slot, looks up the gmem
    row via ``index[row]`` (token id) and atomically reduces the
    value into ``dst[token_id, col]`` with ``op`` (``"add"`` only for
    now — every backend we target lowers ``atom.global.add.f32``).

    ``weight`` (optional) scales the value by ``weight[row]`` before
    the atomic — the ``slot_weights`` path moe_outproj uses.

    No ``stage_in_smem`` knob: atomics need scalar granularity, so
    staging through smem doesn't apply.
    """
    if op != "add":
        raise ValueError(f"pop.atomic_store_acc: unsupported op {op!r}; only 'add' lowers cleanly")
    if acc.results is None:
        raise ValueError(
            "pop.atomic_store_acc: acc.results is None. Pass the Accumulators as "
            "PipelineBody.carry so run_pipeline populates the loop-final values."
        )
    from quark.blocks.l0.epilogue import emit_atomic_scatter_epilogue

    bctx = active_bctx()
    cfg = bctx.mma_cfg
    _col = bctx.c(col) if isinstance(col, int) else col
    emit_atomic_scatter_epilogue(
        bctx.bld,
        acc_results=acc.results,
        g_out=dst,
        index_smem=index,
        weight_smem=weight,
        col_base=_col,
        MT=acc.MT,
        NT=acc.NT,
        cd_offsets=cfg.cd_offsets,
        shape_id=cfg.shape_id,
        m_stride=cfg.shape.m,
        n_stride=cfg.shape.n,
    )


# ---------------------------------------------------------------
# Internal: staged-smem store body (lifted from SiluCastStore._emit_staged
# minus the SiLU body).
# ---------------------------------------------------------------


def _emit_scatter_store(
    bctx: BlockContext,
    *,
    dst: GlobalTensor,
    acc_results: list[Value],
    acc: Accumulators,
    row_base: Any,
    col_base: Any,
    cast: DType,
    activation: str | None = None,
    bias: GlobalTensor | None = None,
    row_scale: list[Value] | None = None,
    atomic: bool = False,
) -> None:
    """Direct per-lane scalar scatter to gmem. Mirrors the legacy
    ``ScatterStore`` body byte-for-byte; no smem staging.

    ``activation="silu"`` fuses ``elem = silu(elem)`` inside the per-
    element body so silu happens at the same MMA-frag-source the MSL
    lowerer requires (no separate FragForEach pass on derived values).

    ``bias`` if provided: a 1-D ``[N]`` GlobalTensor whose element at
    the output column is added to each accumulator element before
    activation and cast. Fuses ``Out = matmul + bias`` into the
    epilogue, eliminating a separate element-wise add pass.

    ``row_scale`` if provided: per-(mt, rc) Values whose reciprocal is
    multiplied into each element pre-cast (the attn ``O /= l`` epilogue).
    """
    from quark.lang import convert, frag_for_each, load, rcp_approx
    from quark.lang import store as _scalar_store

    _use_atomic = atomic

    cfg = bctx.mma_cfg
    shape_id = cfg.shape_id
    m_stride = cfg.shape.m
    n_stride = cfg.shape.n
    _row = bctx.c(row_base) if isinstance(row_base, int) else row_base
    _col = bctx.c(col_base) if isinstance(col_base, int) else col_base
    log2e = bctx.c(_LOG2E) if activation == "silu" else None
    one = bctx.c(1.0) if activation == "silu" else None
    _bias = bias

    rc_info = _rc_info(cfg) if row_scale is not None else None
    rcps = [rcp_approx(v) for v in row_scale] if row_scale is not None else None

    for mt in range(acc.MT):
        for nt in range(acc.NT):
            idx = mt * acc.NT + nt
            acc_v = acc_results[idx]
            mt_row_base = bctx.c(mt * m_stride)
            nt_col_base = bctx.c(nt * n_stride)

            def fn(elem, row, col, *sels, _mt=mt_row_base, _nt=nt_col_base):
                if sels:
                    elem = elem * sels[0]
                gmem_col = _col + (_nt + col)
                if _bias is not None:
                    b_val = load(_bias, gmem_col)
                    b_f32 = convert(b_val, DType.F32) if b_val.dtype != DType.F32 else b_val
                    elem = elem + b_f32
                if activation == "silu":
                    assert log2e is not None and one is not None
                    elem = _silu_scalar(elem, log2e, one)
                if cast != DType.F32:
                    elem = convert(elem, cast)
                gmem_row = _row + (_mt + row)
                if _use_atomic:
                    from quark.lang import atomic_rmw

                    atomic_rmw(dst, "add", elem, gmem_row, gmem_col)
                else:
                    _scalar_store(dst, elem, gmem_row, gmem_col)

            if rc_info is not None and rcps is not None:
                n_rc, slot_to_selector_idx = rc_info
                scales_rc = tuple(rcps[mt * n_rc + rc] for rc in range(n_rc))
                frag_for_each(
                    shape_id,
                    acc_v,
                    fn,
                    cfg.cd_offsets,
                    selectors=scales_rc,
                    slot_to_selector_idx=slot_to_selector_idx,
                )
            else:
                frag_for_each(shape_id, acc_v, fn, cfg.cd_offsets)


def _auto_alloc_staging(
    bctx: BlockContext,
    *,
    acc: Accumulators,
    cast: DType,
    per_warp: bool,
    kv_pad: int,
) -> SharedRegion:
    """Allocate a staging ``SharedRegion`` sized for one ``pop.store_acc``
    call, when the caller didn't pass ``staging_smem=``.

    Shape derivation — the staged epilogue writes ``(MT * m_stride)`` rows
    by ``(NT * n_stride)`` cols per warp (``per_warp=False``) or per CTA
    (``per_warp=True`` stacks one per-warp slice per warp in the block).

    The region gets the default ``Lifetime.auto()`` so the smem layout
    pass aliases it over dead regions (Q/K staging that finished before
    the epilogue runs — the main payoff: no manual ``smem_alloc`` + no
    over-allocation dance at the kernel level).
    """
    from quark.lang import smem_alloc

    cfg = bctx.mma_cfg
    m_stride = cfg.shape.m
    n_stride = cfg.shape.n
    warp_rows = acc.MT * m_stride
    n_warps = bctx.n_threads // 32
    Dh_or_BN = acc.NT * n_stride
    rows = n_warps * warp_rows if per_warp else warp_rows
    return smem_alloc(
        "O_stage",
        cast,
        (rows, Dh_or_BN),
        pad=kv_pad if per_warp else 0,
    )


def _rc_info(cfg: Any) -> tuple[int, tuple[int, ...]]:
    """Return (n_rc, slot_to_selector_idx) for per-row-class scaling.

    ``n_rc`` is the count of distinct ``dr`` values in ``cd_offsets``
    (1 for m8n8k8, 2 for m16n8). ``slot_to_selector_idx`` maps each
    ``cd_offsets`` slot to its row-class index — used as the
    ``selectors`` index when ``frag_for_each`` runs the per-element
    body, so the right scalar lands per fragment slot.
    """
    dr_vals = sorted({dr for dr, _ in cfg.cd_offsets})
    n_rc = len(dr_vals)
    slot_to_selector_idx = tuple(dr_vals.index(dr) for dr, _ in cfg.cd_offsets)
    return n_rc, slot_to_selector_idx


def _silu_scalar(elem: Value, log2e: Value, one: Value) -> Value:
    """Fused per-element silu via the rcp/ex2 fast path."""
    from quark.lang import ex2_approx, neg, rcp_approx

    return elem * rcp_approx(one + ex2_approx(neg(elem * log2e)))


def _cd_offsets_pairable(cd_offsets: tuple[tuple[int, int], ...]) -> bool:
    """True if cd_offsets slot-list walks in adjacent-column pairs with
    even left-column. E.g. ``((0,0),(0,1),(8,0),(8,1))`` → True."""
    if len(cd_offsets) % 2 != 0:
        return False
    for i in range(0, len(cd_offsets), 2):
        dr_a, dc_a = cd_offsets[i]
        dr_b, dc_b = cd_offsets[i + 1]
        if dr_a != dr_b or dc_b != dc_a + 1 or dc_a % 2 != 0:
            return False
    return True


def _emit_scatter_store_paired_bf16x2(
    bctx: BlockContext,
    *,
    dst: GlobalTensor,
    acc_results: list[Value],
    acc: Accumulators,
    row_base: Any,
    col_base: Any,
    activation: str | None = None,
    bias: GlobalTensor | None = None,
    row_scale: list[Value] | None = None,
) -> None:
    """Paired bf16x2 atomic scatter-add.

    Walks the accumulator fragment in column-adjacent pairs. For each
    pair (slot_a, slot_b) with ``dr_a==dr_b`` and ``dc_b==dc_a+1`` and
    even ``dc_a``, packs two F32 accumulator values into a single B32
    (``cvt.rn.bf16x2.f32``) and emits one ``atom.global.add.noftz.bf16x2``
    — half the atomic op count vs the scalar bf16 path.

    Bias/activation/row_scale are applied per-element in F32 before the
    final pack. Row is shared by the pair; bias reads two adjacent cols.
    """
    from popcorn.lang import (
        and_,
        atomic_rmw,
        convert,
        cvt_rn_bf16x2_f32,
        load,
        rcp_approx,
        shl,
        shr,
        vec_extract,
    )

    cfg = bctx.mma_cfg
    cd_offsets = cfg.cd_offsets
    if not _cd_offsets_pairable(cd_offsets):
        raise ValueError(
            f"_emit_scatter_store_paired_bf16x2: cd_offsets not pair-compatible: {cd_offsets}"
        )

    # Lane-local base: gid = laneid >> 2, tig_x2 = (laneid & 3) << 1
    lane = bctx.lane_id
    gid = shr(lane, bctx.c(2, dtype=DType.U32))
    tig = and_(lane, bctx.c(3, dtype=DType.U32))
    tig_x2 = shl(tig, bctx.c(1, dtype=DType.U32))

    _row = bctx.c(row_base) if isinstance(row_base, int) else row_base
    _col = bctx.c(col_base) if isinstance(col_base, int) else col_base
    log2e = bctx.c(_LOG2E) if activation == "silu" else None
    one = bctx.c(1.0) if activation == "silu" else None

    rc_info = _rc_info(cfg) if row_scale is not None else None
    rcps = [rcp_approx(v) for v in row_scale] if row_scale is not None else None

    m_stride = cfg.shape.m
    n_stride = cfg.shape.n

    for mt in range(acc.MT):
        for nt in range(acc.NT):
            idx = mt * acc.NT + nt
            acc_v = acc_results[idx]
            mt_row_base = bctx.c(mt * m_stride)
            nt_col_base = bctx.c(nt * n_stride)

            for pair_i in range(0, len(cd_offsets), 2):
                slot_a, slot_b = pair_i, pair_i + 1
                dr, dc_a = cd_offsets[slot_a]

                # F32 per-slot values via vec_extract on the fragment.
                elem_a = vec_extract(acc_v, slot_a)
                elem_b = vec_extract(acc_v, slot_b)

                # row_scale: same selector for both slots in the pair.
                if rc_info is not None and rcps is not None:
                    n_rc, slot_to_selector = rc_info
                    sel = rcps[mt * n_rc + slot_to_selector[slot_a]]
                    elem_a = elem_a * sel
                    elem_b = elem_b * sel

                # Row (shared by pair).
                row_in_tile = gid if dr == 0 else (gid + bctx.c(dr, dtype=DType.U32))
                gmem_row = _row + (mt_row_base + row_in_tile)

                # Col of left element of pair.
                col_in_tile_a = tig_x2 if dc_a == 0 else (tig_x2 + bctx.c(dc_a, dtype=DType.U32))
                gmem_col_a = _col + (nt_col_base + col_in_tile_a)

                # Bias: read two adjacent cols in F32 space.
                if bias is not None:
                    gmem_col_b = gmem_col_a + bctx.c(1, dtype=DType.U32)
                    b_a = load(bias, gmem_col_a)
                    b_b = load(bias, gmem_col_b)
                    b_a_f32 = convert(b_a, DType.F32) if b_a.dtype != DType.F32 else b_a
                    b_b_f32 = convert(b_b, DType.F32) if b_b.dtype != DType.F32 else b_b
                    elem_a = elem_a + b_a_f32
                    elem_b = elem_b + b_b_f32

                if activation == "silu":
                    assert log2e is not None and one is not None
                    elem_a = _silu_scalar(elem_a, log2e, one)
                    elem_b = _silu_scalar(elem_b, log2e, one)

                # Pack F32 pair → B32 (packed bf16x2).
                packed = cvt_rn_bf16x2_f32(elem_a, elem_b)

                # One atomic per pair — half the instruction count.
                atomic_rmw(
                    dst,
                    "add",
                    packed,
                    gmem_row,
                    gmem_col_a,
                    atomic_type="bf16x2",
                )


def _emit_staged_store(
    bctx: BlockContext,
    *,
    dst: GlobalTensor,
    acc_results: list[Value],
    acc: Accumulators,
    row_base: Any,
    col_base: Any,
    cast: DType,
    activation: str | None = None,
    staging_smem: SharedRegion,
    smem_col_offset: Value | None,
    gmem_col_base_full: Any,
    row_scale: list[Value] | None = None,
    per_warp: bool = False,
    warp_id: Value | None = None,
    kv_pad: int = 0,
) -> None:
    from quark.lang import (
        barrier,
        convert,
        frag_for_each,
        rcp_approx,
        vec_load,
        vec_store,
    )
    from quark.lang import (
        store as _scalar_store,
    )

    cfg = bctx.mma_cfg
    MT, NT = acc.MT, acc.NT
    shape_id = cfg.shape_id
    m_stride = cfg.shape.m
    n_stride = cfg.shape.n
    log2e = bctx.c(_LOG2E) if activation == "silu" else None
    one = bctx.c(1.0) if activation == "silu" else None

    # Per-warp staging view: each warp owns ``warp_rows`` rows of the
    # staging region, indexed by warp_id * (warp_rows * stride_elems).
    # This mirrors the legacy NormalizeAndStore staging layout.
    if per_warp:
        assert warp_id is not None
        Dh_local = staging_smem.shape[1]
        warp_rows = MT * m_stride
        stride_elems = Dh_local + kv_pad
        warp_dyn = warp_id * (warp_rows * stride_elems)
        stage_view = staging_smem.view(
            dyn_offset=warp_dyn,
            shape=(warp_rows, Dh_local),
            name="O_stage",
        )
    else:
        stage_view = staging_smem

    rc_info = _rc_info(cfg) if row_scale is not None else None
    rcps = [rcp_approx(v) for v in row_scale] if row_scale is not None else None

    # Step 1: (scale +) (silu +) cast + scatter to staging smem.
    for mt in range(MT):
        for nt in range(NT):
            acc_v = acc_results[mt * NT + nt]
            mt_row_base = bctx.c(mt * m_stride)
            nt_col_base = bctx.c(nt * n_stride)

            def fn(elem, row, col, *sels, _mt=mt_row_base, _nt=nt_col_base):
                if sels:
                    elem = elem * sels[0]
                if activation == "silu":
                    assert log2e is not None and one is not None
                    elem = _silu_scalar(elem, log2e, one)
                if cast != DType.F32:
                    elem = convert(elem, cast)
                smem_row = _mt + row
                local_col = _nt + col
                smem_col = smem_col_offset + local_col if smem_col_offset is not None else local_col
                _scalar_store(stage_view, elem, smem_row, smem_col)

            if rc_info is not None and rcps is not None:
                n_rc, slot_to_selector_idx = rc_info
                scales_rc = tuple(rcps[mt * n_rc + rc] for rc in range(n_rc))
                frag_for_each(
                    shape_id,
                    acc_v,
                    fn,
                    cfg.cd_offsets,
                    selectors=scales_rc,
                    slot_to_selector_idx=slot_to_selector_idx,
                )
            else:
                frag_for_each(shape_id, acc_v, fn, cfg.cd_offsets)

    barrier("block")

    # Step 2: cooperative v4.b32 vectorized store from smem → gmem.
    # Block-wide path uses bctx.tid / n_threads. Per-warp path uses
    # the warp's 32 lanes — each warp writes its own warp_rows slice
    # to a per-warp gmem row range supplied via row_base.
    BM = MT * m_stride
    smem_cols = stage_view.shape[1]
    b32_per_row = smem_cols // 2
    total_b32 = BM * b32_per_row
    if per_warp:
        n_lanes = 32
        thread_idx = bctx.tid % 32
    else:
        n_lanes = bctx.n_threads
        thread_idx = bctx.tid
    vecs_per_thread = total_b32 // (n_lanes * 4)

    _row = bctx.c(row_base) if isinstance(row_base, int) else row_base
    _gmem_col = (
        bctx.c(gmem_col_base_full) if isinstance(gmem_col_base_full, int) else gmem_col_base_full
    )

    for i in range(vecs_per_thread):
        vec_id = thread_idx * vecs_per_thread + i
        flat_b32 = vec_id * 4
        row = flat_b32 // b32_per_row
        col_b32 = flat_b32 % b32_per_row
        col_bf16 = col_b32 * 2

        v = vec_load(stage_view, row, col_bf16, width=4, dtype=DType.B32)
        gmem_row = _row + row
        gmem_col = _gmem_col + col_bf16
        vec_store(dst, v, gmem_row, gmem_col)
