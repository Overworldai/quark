"""L1: OnlineSoftmax — scale S, update m/l/O, build P fragments.

The full flash-attention online-softmax step: scale S, find row max,
rescale O/l, compute P = exp2((S - m_new) * log2e), accumulate the
new l, and build the P A-fragments for GEMM2. Operates in-place on
the MMA C/D fragment layout.

Per-row reductions use:
  * Metal path (``mma_shape_id`` set) — :meth:`Builder.frag_reduce`
    on the fragment directly; the lowerer emits ``thread_elements()``
    + two ``simd_shuffle_xor`` ops.
  * PTX path (``mma_shape_id`` None) — extract each register, compute,
    ``merge_b32`` back, and finish with a butterfly shuffle across
    the 4 tidIG threads that share each row.

All values stay in the f32 domain; ``exp()`` is expressed as
``exp2(x * log2(e))`` via ``ex2_approx`` for speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from quark.blocks.dsl import Accumulators, Block, active_bctx
from quark.ir import Builder, DType, SharedRegion, Value

_FP8_DTYPES = (DType.E4M3, DType.E5M2)

_LOG2E = math.log2(math.e)


def _shuffle_reduce(b: Builder, val: Value, op: str) -> Value:
    """Butterfly reduce ``val`` across tidIG (4 threads within a group).

    After two XOR shuffles (distance 1 and 2), all 4 threads in each
    group of 4 hold the same reduced value. PTX-path fallback only —
    the Metal path uses ``frag_reduce`` which lowers to two simd
    shuffles via the MSL visitor.
    """
    other1 = b.shuffle("xor", val, 1)
    if op == "max":
        val = b.max(val, other1)
    else:
        val = b.add(val, other1)
    other2 = b.shuffle("xor", val, 2)
    if op == "max":
        val = b.max(val, other2)
    else:
        val = b.add(val, other2)
    return val


@dataclass
class OnlineSoftmax(Block):
    """Online softmax update on S accumulators.

    Callable directly — no ``emit_*`` method on the kernel surface::

        o_vals, m_vals, l_vals, p_frags = softmax(
            s_acc_vals=s_vals,
            o_vals=carry.o,
            m_vals=carry.m,
            l_vals=carry.l,
        )

    ``ctx`` / ``bctx`` is resolved via :func:`active_bctx`. After the
    call, ``self.o_vals`` / ``.m_vals`` / ``.l_vals`` / ``.p_frags``
    hold the same result for post-hoc inspection.
    """

    s_acc: Accumulators  # S accumulator spec (MT × NK)
    o_acc: Accumulators  # O accumulator spec (MT × N_DH)
    scale: float
    # fp8 smem-round-trip path for GEMM2's A-fragment. When the active
    # MMA's a_dtype is e4m3 / e5m2 the in-register acc→a_frag convert
    # (f32→fp8) doesn't line up same-lane with the 16-bit accumulator
    # layout — the fp8 A-fragment wants 4 consecutive k-cols per lane
    # while the bf16 C-fragment hands us 2 cols from each of 2 adjacent
    # lanes. Caller allocates ``p_smem`` sized ``(n_warps * MT * m, K)``
    # in the compute dtype; softmax writes its f32 P through
    # ``packed_convert`` into this warp's slice at ``p_row_base`` and
    # returns ``p_frags=None``. The caller then issues a block barrier
    # and rebuilds ``p_frags`` via ``load_matrix`` from ``p_smem`` into
    # the fp8 A-frag layout. When ``p_smem is None`` the legacy
    # in-register ``frag_convert`` path runs (only correct for bf16).
    p_smem: SharedRegion | None = None
    p_row_base: Any = 0  # runtime Value or Python int

    # Populated on call.
    o_vals: list[Value] | None = None
    m_vals: list[Value] | None = None
    l_vals: list[Value] | None = None
    p_frags: list[list[Value]] | None = None

    def __call__(
        self,
        *,
        s_acc_vals: list[Value],
        o_vals: list[Value],
        m_vals: list[Value],
        l_vals: list[Value],
    ) -> tuple[list[Value], list[Value], list[Value], list[list[Value]] | None]:
        ctx = active_bctx()
        b = ctx.bld
        cfg = ctx.mma_cfg

        MT = self.s_acc.MT
        NK = self.s_acc.NT
        N_DH = self.o_acc.NT
        cd_offsets = cfg.cd_offsets
        mma_shape_id = cfg.shape_id
        shape_k = cfg.shape.k
        shape_n = cfg.shape.n
        c_regs = cfg.shape.c_regs

        s_acc = list(s_acc_vals)
        log2e = b.const(DType.F32, _LOG2E)
        scale_c = b.const(DType.F32, self.scale)
        neg_inf = b.const(DType.F32, -1e30)

        # ``is_intel``: Intel cooperative_matrix
        # (``cl_intel_subgroup_matrix_multiply_accumulate``) has an
        # implementation-private lane↔(row, col) mapping, so per-c_reg
        # ``cd_offsets`` can't express the ``rows``-many row classes
        # online softmax needs. The OCL lowerer recovers the row from
        # the smem layout instead — kernel asks for ``shape.m``-many
        # ``frag_reduce`` results via ``n_classes_override`` and
        # ``frag_apply`` selectors via dynamic-row-dispatch
        # (``slot_to_selector_idx=None``). Other backends (PTX/Apple)
        # keep the per-c_reg path that works for their public lane
        # layouts.
        is_intel = (
            mma_shape_id is not None and "_intel_" in mma_shape_id
        )
        # Number of row classes:
        #   - PTX/Apple: distinct dr values in cd_offsets
        #   - Intel: full row count (shape.m)
        if is_intel:
            n_rc = cfg.shape.m
            dr_vals = list(range(n_rc))
        else:
            dr_vals = sorted({dr for dr, _ in cd_offsets})
            n_rc = len(dr_vals)
        row_class = {i: dr_vals.index(dr) for i, (dr, _) in enumerate(cd_offsets)} if not is_intel else {}
        # GEMM2 P-fragment packs ``nk_per_kstep`` acc tiles per k-step.
        # For m16n8k16 this is 2 (k=16 covers 2 nk-tiles of 8 cols each).
        # For m8n8k8 it's 1 (k=8 matches one nk-tile).
        nk_per_kstep = max(shape_k // shape_n, 1)

        # ── Step 1: Scale S + per-nk row-max ──
        # Metal path uses frag_apply for the scale (stays in register
        # space — no smem round-trip) and frag_reduce for the per-nk
        # per-row max (thread_elements() + 2 simd_shuffle_xor's).
        # Combining across nk tiles is a scalar max chain; the final
        # ``row_max[mt][rc]`` is already cross-lane reduced.
        #
        # PTX path still uses the extract/mul/build pattern — c_regs are
        # already scalar there, so nothing is lost.
        # Using Any-typed nested lists avoids paying for ty narrowing
        # around the init-None-then-populate pattern (every slot ends
        # up Value-typed before any read).
        row_max: list[list[Any]] = [[None] * n_rc for _ in range(MT)]

        if mma_shape_id is not None:
            reduce_kwargs = (
                {"n_classes_override": n_rc} if is_intel else {}
            )
            for mt in range(MT):
                for nk in range(NK):
                    idx = mt * NK + nk
                    s_acc[idx] = b.frag_apply(
                        mma_shape_id,
                        s_acc[idx],
                        lambda x: b.mul(x, scale_c),
                    )
                    nk_max = b.frag_reduce(
                        mma_shape_id,
                        s_acc[idx],
                        kind="max",
                        axis="row",
                        cd_offsets=cd_offsets,
                        **reduce_kwargs,
                    )
                    for rc in range(n_rc):
                        if row_max[mt][rc] is None:
                            row_max[mt][rc] = nk_max[rc]
                        else:
                            row_max[mt][rc] = b.max(row_max[mt][rc], nk_max[rc])
        else:
            # Legacy PTX / fallback path.
            local_max = [[neg_inf for _ in range(n_rc)] for _ in range(MT)]
            for mt in range(MT):
                for nk in range(NK):
                    idx = mt * NK + nk
                    s_vec = s_acc[idx]
                    new_elems = []
                    for ri in range(c_regs):
                        elem = b.vec_extract(s_vec, ri)
                        scaled = b.mul(elem, scale_c)
                        new_elems.append(scaled)
                        rc = row_class[ri]
                        local_max[mt][rc] = b.max(local_max[mt][rc], scaled)
                    s_acc[idx] = b.vec_build(new_elems)
            for mt in range(MT):
                for rc in range(n_rc):
                    row_max[mt][rc] = _shuffle_reduce(b, local_max[mt][rc], "max")

        # ── Step 2: New max = max(old_m, row_max) ──
        new_m: list[list[Any]] = [[None] * n_rc for _ in range(MT)]
        for mt in range(MT):
            for rc in range(n_rc):
                new_m[mt][rc] = b.max(m_vals[mt * n_rc + rc], row_max[mt][rc])

        # ── Step 3: Rescale O and l ──
        # rescale = exp2((m_old - m_new) * log2e)
        new_o_acc = list(o_vals)
        new_l_vals = list(l_vals)

        rescale: list[list[Any]] = [[None] * n_rc for _ in range(MT)]
        for mt in range(MT):
            for rc in range(n_rc):
                diff = b.sub(m_vals[mt * n_rc + rc], new_m[mt][rc])
                rescale[mt][rc] = b.ex2_approx(b.mul(diff, log2e))

        # Rescale O accumulators. With ``mma_shape_id`` set, use the
        # fused FragApplyOp with per-row-class selectors — on MSL this
        # mutates the simdgroup_matrix in place via ``thread_elements()``
        # and skips the extract→store→barrier→per-lane-read→mul→
        # per-lane-write→barrier→load round-trip the naive
        # extract+vec_build pattern produces. PTX lowers to per-reg
        # ``mul.f32``, same count as before.
        if mma_shape_id is not None:
            # Intel path: pass selectors with ``slot_to_selector_idx
            # =None`` to opt into dynamic-row-dispatch (lowerer derives
            # row from smem layout). PTX/Apple: per-c_reg static map.
            slot_to_selector_idx = (
                None if is_intel
                else tuple(dr_vals.index(dr) for dr, _ in cd_offsets)
            )
            for mt in range(MT):
                scales_rc = tuple(rescale[mt])
                assert all(s is not None for s in scales_rc)
                for nd in range(N_DH):
                    oi = mt * N_DH + nd
                    new_o_acc[oi] = b.frag_apply(
                        mma_shape_id,
                        new_o_acc[oi],
                        lambda x, s: b.mul(x, s),
                        selectors=scales_rc,
                        slot_to_selector_idx=slot_to_selector_idx,
                    )
        else:
            for mt in range(MT):
                for nd in range(N_DH):
                    oi = mt * N_DH + nd
                    o_vec = new_o_acc[oi]
                    new_elems = []
                    for ri in range(c_regs):
                        elem = b.vec_extract(o_vec, ri)
                        rc = row_class[ri]
                        new_elems.append(b.mul(elem, rescale[mt][rc]))
                    new_o_acc[oi] = b.vec_build(new_elems)

        # Rescale l
        for mt in range(MT):
            for rc in range(n_rc):
                new_l_vals[mt * n_rc + rc] = b.mul(new_l_vals[mt * n_rc + rc], rescale[mt][rc])

        # ── Step 4: Compute P = exp2((S - m_new) * log2e), accumulate l ──
        # Metal path:
        #   1. frag_apply with row-class selectors to compute
        #      p_acc = exp2((S - m_new[rc]) * log2e) in-place (no smem
        #      round-trip). Yields f32 ACC fragments.
        #   2. frag_reduce(add) on each p_acc for per-row local psum.
        #   3. frag_convert(acc→a_frag, f32→bf16) combining nk_per_kstep
        #      p_acc tiles → 1 A-fragment.
        # PTX fallback keeps the legacy extract + pack pattern.
        GEMM2_K_STEPS = NK // nk_per_kstep

        if mma_shape_id is not None:
            slot_to_selector_idx = (
                None if is_intel
                else tuple(dr_vals.index(dr) for dr, _ in cd_offsets)
            )
            reduce_kwargs = (
                {"n_classes_override": n_rc} if is_intel else {}
            )

            # Step 4a: compute P f32 ACC fragments via frag_apply.
            p_acc: list[list[Value]] = [[None for _ in range(NK)] for _ in range(MT)]  # type: ignore
            for mt in range(MT):
                new_m_tup = tuple(new_m[mt])
                for nk in range(NK):
                    s_vec = s_acc[mt * NK + nk]
                    p_acc[mt][nk] = b.frag_apply(
                        mma_shape_id,
                        s_vec,
                        lambda x, m_rc: b.ex2_approx(b.mul(b.sub(x, m_rc), log2e)),
                        selectors=new_m_tup,
                        slot_to_selector_idx=slot_to_selector_idx,
                    )

            # Step 4b: per-(mt, nk) row-class psum via frag_reduce(add).
            # Combine across nk tiles by scalar add (per-lane, no extra
            # shuffles — frag_reduce already produced the full-row
            # reduction).
            local_psum_final: list[list[Value]] = [[None] * n_rc for _ in range(MT)]  # type: ignore
            for mt in range(MT):
                for nk in range(NK):
                    nk_sums = b.frag_reduce(
                        mma_shape_id,
                        p_acc[mt][nk],
                        kind="add",
                        axis="row",
                        cd_offsets=cd_offsets,
                        **reduce_kwargs,
                    )
                    for rc in range(n_rc):
                        if local_psum_final[mt][rc] is None:
                            local_psum_final[mt][rc] = nk_sums[rc]
                        else:
                            local_psum_final[mt][rc] = b.add(local_psum_final[mt][rc], nk_sums[rc])
                for rc in range(n_rc):
                    new_l_vals[mt * n_rc + rc] = b.add(
                        new_l_vals[mt * n_rc + rc], local_psum_final[mt][rc]
                    )

            # Step 4c: materialize the P A-fragment for GEMM2.
            # Two paths selected by the active MMA's A-dtype:
            #  - 16-bit (bf16/f16): in-register ``frag_convert`` packs
            #    ``nk_per_kstep`` f32 acc tiles into one b32-wide A-frag.
            #    Same-lane layout, no shuffles.
            #  - 8-bit (e4m3/e5m2): smem round-trip. Per-lane f32 P values
            #    go through ``packed_convert`` (f32×2 → fp8×2 in one b16)
            #    and scalar-store to ``p_smem`` at their true (row, col).
            #    The caller issues a block barrier and reloads with
            #    ``load_matrix`` into the fp8 A-frag layout — those reads
            #    match the PTX ISA fp8 m16n8k16/k32 fragment map.
            a_dtype = cfg.shape.a_dtype
            if a_dtype in _FP8_DTYPES:
                assert self.p_smem is not None, (
                    f"OnlineSoftmax: fp8 compute ({a_dtype}) requires p_smem "
                    f"for smem-round-trip P; pass p_smem= in the block init."
                )
                # cd_offsets for every m16n8 MMA in the registry is
                # ((0,0),(0,1),(8,0),(8,1)). We use this ordering directly
                # — pair slots {0,1} at row dr=0 and {2,3} at row dr=8.
                if tuple(cd_offsets) != ((0, 0), (0, 1), (8, 0), (8, 1)):
                    raise NotImplementedError(
                        f"OnlineSoftmax fp8 path expects m16n8 cd_offsets, got {cd_offsets}"
                    )
                shape_m = cfg.shape.m
                shape_n = cfg.shape.n
                # Per-lane offsets into the warp's p_smem slice:
                #   row = p_row_base + mt*m + gid            (top half)
                #   row = p_row_base + mt*m + (m//2) + gid   (bottom half)
                #   col = nk*n + 2*tig                        (packed pair)
                gid = ctx.gid
                # Emit ``tig * 2`` locally instead of going through the
                # bctx.tig_x2 lazy property. The property caches its Value
                # on first access, which creates an SSA scope violation
                # when ``consume()`` is called from multiple IR regions
                # (for_range body + epilogue tail): the first call caches
                # the MulOp inside the for_range body region, and the
                # epilogue's re-use references a Value that doesn't
                # dominate it. Emitting fresh each call lets the builder's
                # region-scoped CSE pick the right Value per caller region.
                tig_x2 = b.mul(ctx.tig, ctx.c(2))
                p_base = self.p_row_base
                p_base_v = p_base if isinstance(p_base, Value) else ctx.c(p_base)
                for mt in range(MT):
                    row_top = b.add(b.add(p_base_v, ctx.c(mt * shape_m)), gid)
                    row_bot = b.add(row_top, ctx.c(shape_m // 2))
                    for nk in range(NK):
                        col = b.add(ctx.c(nk * shape_n), tig_x2)
                        p_vec = p_acc[mt][nk]
                        c0 = b.vec_extract(p_vec, 0)
                        c1 = b.vec_extract(p_vec, 1)
                        c2 = b.vec_extract(p_vec, 2)
                        c3 = b.vec_extract(p_vec, 3)
                        p01 = b.packed_convert(c0, c1, a_dtype)
                        p23 = b.packed_convert(c2, c3, a_dtype)
                        b.store(self.p_smem, p01, row_top, col)
                        b.store(self.p_smem, p23, row_bot, col)
                p_frags: list[list[Value]] | None = None
            else:
                # m16n8k16: packs 2 nk-tiles per k-step; m8n8k8: 1 nk-tile.
                p_frags = [
                    [
                        b.frag_convert(
                            mma_shape_id,
                            src_frags=tuple(
                                p_acc[mt][k_step * nk_per_kstep + i] for i in range(nk_per_kstep)
                            ),
                            src_layout="acc",
                            dst_layout="a_frag",
                            src_dtype=DType.F32,
                            dst_dtype=DType.BF16,
                            cd_offsets=cd_offsets,
                        )
                        for k_step in range(GEMM2_K_STEPS)
                    ]
                    for mt in range(MT)
                ]
        else:
            # Legacy PTX fallback: extract → compute → merge_b32 pack.
            local_psum = [[b.const(DType.F32, 0.0) for _ in range(n_rc)] for _ in range(MT)]
            p_frags_legacy: list[list[list[Value]]] = [
                [[] for _ in range(GEMM2_K_STEPS)] for _ in range(MT)
            ]
            for mt in range(MT):
                for k_step in range(GEMM2_K_STEPS):
                    nk_lo = k_step * 2
                    nk_hi = k_step * 2 + 1
                    a_regs = [None] * 4
                    for half, nk in enumerate([nk_lo, nk_hi]):
                        si = mt * NK + nk
                        s_vec = s_acc[si]
                        p_f32 = []
                        for ri in range(c_regs):
                            elem = b.vec_extract(s_vec, ri)
                            rc = row_class[ri]
                            diff = b.sub(elem, new_m[mt][rc])
                            exp_arg = b.mul(diff, log2e)
                            p_val = b.ex2_approx(exp_arg)
                            local_psum[mt][rc] = b.add(local_psum[mt][rc], p_val)
                            p_f32.append(p_val)
                        p_bf16_0 = b.convert(p_f32[0], DType.BF16)
                        p_bf16_1 = b.convert(p_f32[1], DType.BF16)
                        p_bf16_2 = b.convert(p_f32[2], DType.BF16)
                        p_bf16_3 = b.convert(p_f32[3], DType.BF16)
                        lo_b16_0 = b.bitcast(p_bf16_0, DType.B16)
                        hi_b16_0 = b.bitcast(p_bf16_1, DType.B16)
                        lo_b16_1 = b.bitcast(p_bf16_2, DType.B16)
                        hi_b16_1 = b.bitcast(p_bf16_3, DType.B16)
                        a_regs[0 + half * 2] = b.merge_b32(lo_b16_0, hi_b16_0)
                        a_regs[1 + half * 2] = b.merge_b32(lo_b16_1, hi_b16_1)
                    p_frags_legacy[mt][k_step] = a_regs
            for mt in range(MT):
                for rc in range(n_rc):
                    full_sum = _shuffle_reduce(b, local_psum[mt][rc], "add")
                    new_l_vals[mt * n_rc + rc] = b.add(new_l_vals[mt * n_rc + rc], full_sum)
            # Flatten per-register b32 list into one width-N Value so
            # GEMM2 sees the same shape as the MSL frag_convert path.
            p_frags = [
                [b.vec_build(p_frags_legacy[mt][k_step]) for k_step in range(GEMM2_K_STEPS)]
                for mt in range(MT)
            ]

        # Flatten new_m.
        new_m_flat: list[Value] = []
        for mt in range(MT):
            for rc in range(n_rc):
                new_m_flat.append(new_m[mt][rc])

        self.o_vals, self.m_vals, self.l_vals, self.p_frags = (
            new_o_acc,
            new_m_flat,
            new_l_vals,
            p_frags,
        )
        return new_o_acc, new_m_flat, new_l_vals, p_frags
