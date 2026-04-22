"""MMA (simdgroup_matrix) visitor methods for the MSL lowerer.

EXEMPT FROM 500-LINE RULE: LoadMatrixOp / MmaOp / StoreMatrixOp and
the fragment→scalar extraction pipeline share the 2x2x2 Apple
bit-swizzle tables and per-dtype register-map generators; splitting
forces each visitor to re-import or re-derive them.

Handles LoadMatrixOp, MmaOp, StoreMatrixOp, and the fragment→scalar
extraction pipeline for kernel epilogues.
"""

from __future__ import annotations

from quark.ir import (
    FragApplyOp,
    FragConvertOp,
    FragForEachOp,
    FragReduceOp,
    LoadMatrixOp,
    MmaOp,
    StoreMatrixOp,
    YieldOp,
)

from .lower import _MslCtx, _tensor_buf_name
from .types import msl_type


def _parse_msl_tiling(msl_str: str) -> tuple[str, int, int, int]:
    """Parse the MmaShape.msl field: 'frag_dtype:m_frags:n_frags:k_frags'."""
    parts = msl_str.split(":")
    return parts[0], int(parts[1]), int(parts[2]), int(parts[3])


def _is_zero_vec_build(value) -> bool:
    """Return True if ``value`` is the result of a ``VecBuildOp`` whose
    components are all ``ConstOp`` with scalar value 0.

    Used by ``visit_mma`` to recognize ``Accumulators.emit_init``'s
    zero-broadcast pattern and skip ``_pack_scalars_to_frag_array`` in
    favor of a direct zero-init frag array.
    """
    from quark.ir.op import ConstOp, VecBuildOp

    producer = getattr(value, "producer", None)
    if not isinstance(producer, VecBuildOp):
        return False
    for comp in producer.operands:
        comp_producer = getattr(comp, "producer", None)
        if not isinstance(comp_producer, ConstOp):
            return False
        if comp_producer.attrs.get("value") != 0:
            return False
    return True


def _is_nonzero_vec_build(value) -> bool:
    """True iff ``value`` is a VecBuildOp whose components include at
    least one non-zero ConstOp or any non-Const producer. Distinguishes
    the deprecated "extract scalars → rebuild vec" round-trip (always
    a VecBuildOp) from legitimate for-loop body carries (which are
    Values synthesized by ForLoopOp, not VecBuild producers).
    """
    from quark.ir.op import VecBuildOp

    return isinstance(getattr(value, "producer", None), VecBuildOp) and not _is_zero_vec_build(
        value
    )


def extract_frag_scalars(value, ctx: _MslCtx) -> tuple[str, ...]:
    """REMOVED path. Was: extract per-lane scalars from a simdgroup_matrix
    fragment via a smem round-trip.

    Every former caller (online_softmax S-scale/row-max, O-rescale,
    P-fragment generation; normalize_store; silu_cast_store;
    scatter_store; atomic_scatter_add; epilogue.py) has been migrated
    to FragApplyOp / FragReduceOp / FragConvertOp / FragForEachOp —
    none of which need a smem round-trip.

    Raises loudly so any future caller sees a clear path forward
    instead of silently re-introducing the pool.
    """
    raise NotImplementedError(
        "extract_frag_scalars is REMOVED. VecExtractOp on a fragment "
        "Value is no longer supported — use one of the typed fragment "
        "primitives: FragApplyOp (.map / .map_per_row_class) for "
        "element-wise transforms, FragReduceOp (.reduce_along_cols) "
        "for cross-lane reductions, FragForEachOp (.for_each) for "
        "side-effect epilogues, or FragConvertOp (.convert) for "
        "layout / dtype changes."
    )


def _frag_name(value, ctx: _MslCtx) -> str:
    """Get the simdgroup_matrix array base name for a fragment Value."""
    if ctx.names.has(value):
        return ctx.names.components(value)[0]
    name = ctx.names.fresh("frag")
    ctx.names.bind(value, (name,))
    return name


def _frag_name_or_pack(value, frag_dtype: str, n_frags: int, ctx: _MslCtx) -> str:
    """Return a simdgroup_matrix-array name for ``value``.

    If ``value`` is already registered as a fragment array (produced by
    LoadMatrixOp or a prior MmaOp), return its name. Otherwise treat
    ``value`` as a PTX-style b32-packed fragment and unpack it into
    a fresh simdgroup_matrix array via ``_pack_b32_to_frag_array``.
    """
    if value.id in ctx.frag_values:
        return ctx.frag_values[value.id][0]
    # Cache packed conversions so the same packed source isn't unpacked
    # multiple times when a single Value feeds several MmaOps.
    if value.id in ctx.frag_packs:
        return ctx.frag_packs[value.id]
    arr = _pack_b32_to_frag_array(value, frag_dtype, n_frags, ctx)
    ctx.frag_packs[value.id] = arr
    return arr


def _pack_b32_to_frag_array(value, frag_dtype: str, n_frags: int, ctx: _MslCtx) -> str:
    """Convert a width=N b32 Value (PTX packed-fragment layout) into an
    array of ``N`` ``simdgroup_matrix<frag_dtype, 8, 8>``.

    Used for the inputs to ``simdgroup_multiply_accumulate`` when the
    caller produced the operand the PTX way — packed bf16×2 (or fp16×2)
    into a scalar b32 per fragment lane. The owl_attn online-softmax
    P-fragment path does this. Metal needs the full simdgroup_matrix.

    Apple's ``simdgroup_matrix<T, 8, 8>`` per-lane mapping is a 2×2×2
    swizzle — not PTX's ``lane L → (L/4, (L%4)*2 + 0..1)``. Empirically
    (see ``tests/lower/msl/test_frag_layout.py``):

        row  = ((L >> 4) & 1) * 4 + ((L >> 1) & 3)
        col0 = ((L >> 3) & 1) * 4 + (L & 1) * 2
        col1 = col0 + 1
        thread_elements()[0] is at (row, col0); [1] is at (row, col1).

    The PTX-format b32 regs the softmax produces hold the VALUE at
    (row = L_ptx / 4, col = (L_ptx % 4)*2 + 0..1). So to populate
    Apple tile ``arr[fi]`` via ``thread_elements()`` we need each
    Apple lane to receive the b32 that was computed on the PTX lane
    whose (row, col) matches THIS Apple lane's position. That source
    PTX lane is ``row * 4 + col0 / 2`` — same formula for every tile
    (``fi`` only picks which tile's reg). One ``simd_shuffle`` per
    tile per lane replaces the old ``simdgroup_store → barrier →
    per-lane scatter → simdgroup_load`` round-trip entirely: no
    threadgroup traffic, no barriers.
    """
    comps = ctx.names.components(value)
    assert len(comps) == n_frags, (
        f"_pack_b32_to_frag_array: value has {len(comps)} components, expected {n_frags}"
    )

    # Per-lane PTX-source-lane formula. Apple ``lane`` varies per SIMD
    # lane but the bit-math is uniform across all lanes, so this folds
    # into register ops.
    lane = ctx.names.fresh("lane")
    ap_r = ctx.names.fresh("ar")
    ap_c = ctx.names.fresh("ac")
    src = ctx.names.fresh("src")
    ctx.emit(f"uint {lane} = thread_index_in_simdgroup;")
    ctx.emit(f"uint {ap_r} = (({lane} >> 4u) & 1u) * 4u + (({lane} >> 1u) & 3u);")
    ctx.emit(f"uint {ap_c} = (({lane} >> 3u) & 1u) * 4u + ({lane} & 1u) * 2u;")
    ctx.emit(f"uint {src} = {ap_r} * 4u + ({ap_c} >> 1u);")

    arr = ctx.names.fresh("frag")
    ctx.emit(f"simdgroup_matrix<{frag_dtype}, 8, 8> {arr}[{n_frags}];")
    for fi, s in enumerate(comps):
        shuf = ctx.names.fresh("sh")
        pk = ctx.names.fresh("pk")
        e_ref = ctx.names.fresh("e")
        ctx.emit(f"uint {shuf} = simd_shuffle({s}, {src});")
        ctx.emit(f"ushort2 {pk} = as_type<ushort2>({shuf});")
        ctx.emit(f"thread auto& {e_ref} = {arr}[{fi}].thread_elements();")
        ctx.emit(f"{e_ref}[0] = as_type<{frag_dtype}>({pk}.x);")
        ctx.emit(f"{e_ref}[1] = as_type<{frag_dtype}>({pk}.y);")
    return arr


def visit_mma(self, op: MmaOp, ctx: _MslCtx) -> None:
    """Lower MmaOp to simdgroup_multiply_accumulate calls."""
    shape_id = op.attrs["shape_id"]
    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(f"MmaOp: shape {shape_id!r} not in module.kernel_shapes")
    shape = module.kernel_shapes[shape_id]
    if not shape.msl:
        raise NotImplementedError(
            f"MmaOp: MmaShape {shape_id!r} has no `msl` field — "
            f"Metal simdgroup lowering not available for this shape"
        )
    ctx.uses_simdgroup_matrix = True
    _, mf, nf, kf = _parse_msl_tiling(shape.msl)

    a, b_frag, c = op.operands
    (d,) = op.results
    frag_dtype_str, _, _, _ = _parse_msl_tiling(shape.msl)
    # If an operand arrived PTX-style (b32-packed bf16×2 per lane,
    # typically from the online-softmax P-fragment path in owl_attn),
    # unpack it into a real simdgroup_matrix array. `frag_values`
    # tracks the fragment-ARRAY form; anything else with the packed
    # signature (multiple b32 components) needs conversion.
    a_name = _frag_name_or_pack(a, frag_dtype_str, mf * kf, ctx)
    b_name = _frag_name_or_pack(b_frag, frag_dtype_str, kf * nf, ctx)
    acc_ty = msl_type(shape.acc_dtype)
    n_cd = mf * nf

    # Handle C input.
    #   - already a frag (prior MmaOp / LoadMatrix C/D)? reuse.
    #   - width == 2*n_cd scalar vec? this is the online-softmax round
    #     trip: O is extracted to per-lane scalars, rescaled by the
    #     rolling exp-diff factor, then VecBuild'd back. The rebuilt
    #     Value has scalar components (no frag registration). Without
    #     repack, the else branch zero-inits and silently throws away
    #     every rescale — the chief cause of cos≈0 in owl_attn on Metal.
    #     `_pack_scalars_to_frag_array` reverses extract_frag_scalars'
    #     2-scalars-per-tile layout. Cache via frag_values so the same
    #     rebuilt vec feeding multiple MMAs only repacks once.
    #   - otherwise: zero-init a fresh accumulator.
    if c.id in ctx.frag_values:
        c_name = _frag_name(c, ctx)
    elif _is_zero_vec_build(c):
        # Fast path: VecBuildOp of all-zero ConstOp values → emit a
        # zero-init frag array directly. Common for
        # ``Accumulators.emit_init`` (S accumulator re-inited each
        # kv_chunk, O init before the outer for_loop, etc.).
        c_name = ctx.names.fresh("frag")
        ctx.emit(f"simdgroup_matrix<{acc_ty}, 8, 8> {c_name}[{n_cd}];")
        for i in range(n_cd):
            ctx.emit(f"{c_name}[{i}] = simdgroup_matrix<{acc_ty}, 8, 8>(0);")
        ctx.frag_values[c.id] = (c_name, acc_ty, mf, nf)
    elif c.width == 2 * n_cd and ctx.names.has(c) and _is_nonzero_vec_build(c):
        # Deprecated round-trip (non-zero VecBuild from scalar extract).
        # For-loop body carries don't hit this — they aren't VecBuildOps.
        raise NotImplementedError(
            f"visit_mma: MmaOp {shape_id!r} got a non-zero scalar-vec C "
            f"(width={c.width}). Migrate the producer to a Frag* op."
        )
    else:
        c_name = ctx.names.fresh("frag")
        ctx.emit(f"simdgroup_matrix<{acc_ty}, 8, 8> {c_name}[{n_cd}];")
        for i in range(n_cd):
            ctx.emit(f"{c_name}[{i}] = simdgroup_matrix<{acc_ty}, 8, 8>(0);")
        ctx.frag_values[c.id] = (c_name, acc_ty, mf, nf)

    # Allocate D (output accumulator).
    d_name = ctx.names.fresh("frag")
    ctx.names.bind(d, (d_name,))
    ctx.frag_values[d.id] = (d_name, acc_ty, mf, nf)

    ctx.emit(f"simdgroup_matrix<{acc_ty}, 8, 8> {d_name}[{n_cd}];")
    for i in range(n_cd):
        ctx.emit(f"{d_name}[{i}] = {c_name}[{i}];")

    for m in range(mf):
        for n in range(nf):
            for k in range(kf):
                # A fragments are in PTX M-major order: frag[k * mf + m]
                a_idx = k * mf + m
                b_idx = k * nf + n
                d_idx = m * nf + n
                ctx.emit(
                    f"simdgroup_multiply_accumulate("
                    f"{d_name}[{d_idx}], {a_name}[{a_idx}], "
                    f"{b_name}[{b_idx}], {d_name}[{d_idx}]);"
                )


def _collect_body_local_value_ids(ops) -> list[int]:
    """Every Value.id produced by ops in this region (incl. nested).
    Used by the per-slot re-walker to wipe stale name bindings so each
    walk allocates fresh locals. See PTX ``_collect_body_local_value_ids``
    for the matching pattern on that backend.
    """
    ids: list[int] = []
    for op in ops:
        for v in op.results:
            ids.append(v.id)
        for r in op.regions:
            ids.extend(_collect_body_local_value_ids(r.ops))
    return ids


def visit_frag_apply(self, op: FragApplyOp, ctx: _MslCtx) -> None:
    """Lower FragApplyOp by re-walking the body region once per
    (tile, thread_elements-slot). Keeps the transform entirely in
    register-level simdgroup_matrix state — no threadgroup round-trip.

    For a ``(mf, nf)`` accumulator with ``mf·nf`` tiles, emits:

        simdgroup_matrix<T, 8, 8> out[mf*nf];
        for each tile fi:
          out[fi] = in[fi];
          thread auto& e_fi = out[fi].thread_elements();
          for slot ∈ {0, 1}:
            T slot_elem_i = e_fi[slot];
            <body subgraph with body_input_var bound to slot_elem_i>
            e_fi[slot] = yielded;

    The body may reference free variables (scales, consts) from the
    enclosing scope; those aren't in ``body_local_ids`` so their names
    survive across walks. Only body-local Values (the scalar ops
    ``fn`` built) get freshly re-allocated per slot — avoiding the
    "walk #2 reuses walk #1's names" trap.
    """
    shape_id = op.attrs["shape_id"]
    in_frag = op.in_frag
    selectors = op.selectors
    slot_to_sel = op.attrs.get("slot_to_selector_idx")
    (out,) = op.results

    if in_frag.id not in ctx.frag_values:
        raise RuntimeError(
            f"FragApplyOp (shape {shape_id!r}): operand frag isn't "
            "registered in frag_values — must be sourced from an MmaOp "
            "or LoadMatrixOp on the MSL path."
        )
    src_name, acc_ty, mf, nf = ctx.frag_values[in_frag.id]
    n_tiles = mf * nf

    input_var = op.body_input_var
    assert input_var is not None
    sel_var = op.body_selector_var
    sel_names = [ctx.names.name_for(s) for s in selectors] if selectors else []
    body_local_ids = _collect_body_local_value_ids(op.body.ops)
    # Drop input_var and sel_var from the wipe list — we bind them
    # explicitly per slot.
    skip_ids = {input_var.id}
    if sel_var is not None:
        skip_ids.add(sel_var.id)
    body_local_ids = [vid for vid in body_local_ids if vid not in skip_ids]

    # Map MSL (tile, slot) → logical slot index. For m16n8 with mf=2,nf=1
    # and c_regs=4 with cd_offsets ((0,0),(0,1),(8,0),(8,1)): the c_reg
    # ordering groups by (dr, dc) where dr=0,0,8,8 and dc=0,1,0,1. Tile 0
    # covers m-rows 0-7 (dr=0) and tile 1 covers m-rows 8-15 (dr=8). Each
    # tile's thread_elements[0,1] covers dc=0,1. So:
    #   (tile fi=0, slot=0) → c_reg (dr=0, dc=0) → logical slot 0
    #   (tile fi=0, slot=1) → c_reg (dr=0, dc=1) → logical slot 1
    #   (tile fi=1, slot=0) → c_reg (dr=8, dc=0) → logical slot 2
    #   (tile fi=1, slot=1) → c_reg (dr=8, dc=1) → logical slot 3
    # → logical_slot = fi * 2 + slot.

    out_name = ctx.names.fresh("frag")
    ctx.emit(f"simdgroup_matrix<{acc_ty}, 8, 8> {out_name}[{n_tiles}];")
    for fi in range(n_tiles):
        ctx.emit(f"{out_name}[{fi}] = {src_name}[{fi}];")
        e_ref = ctx.names.fresh("e")
        ctx.emit(f"thread auto& {e_ref} = {out_name}[{fi}].thread_elements();")
        for slot in (0, 1):
            slot_elem = ctx.names.fresh("apply_elem")
            ctx.emit(f"{acc_ty} {slot_elem} = {e_ref}[{slot}];")
            # Rebind input var to this slot's scalar (force past any
            # prior binding from a previous fi/slot pair).
            ctx.names.bind(input_var, (slot_elem,), force=True)
            if sel_var is not None and slot_to_sel is not None:
                logical_slot = fi * 2 + slot
                sel_name = sel_names[slot_to_sel[logical_slot]]
                ctx.names.bind(sel_var, (sel_name,), force=True)
            # Wipe body-local names so the walk re-allocates fresh.
            for vid in body_local_ids:
                ctx.names._names.pop(vid, None)
            # Walk body ops; capture yielded name when we hit the
            # terminator and write it back to the slot.
            for bop in op.body.ops:
                if isinstance(bop, YieldOp):
                    yielded = bop.operands[0]
                    ctx.emit(f"{e_ref}[{slot}] = {ctx.names.name_for(yielded)};")
                    break
                self._visit(bop, ctx)

    ctx.names.bind(out, (out_name,))
    ctx.frag_values[out.id] = (out_name, acc_ty, mf, nf)


def visit_frag_convert(self, op: FragConvertOp, ctx: _MslCtx) -> None:
    """Lower FragConvertOp for ACC f32 → A_FRAG bf16 on MSL.

    ONLY supports ``(acc, f32) → (a_frag, bf16)`` today. Arbitrary
    register-tile conversions (acc↔b_frag, dtype promotion / demotion
    outside f32↔bf16, mixed layouts like acc→store-layout, etc.) are
    bigger scope: they need per-(src, dst) mapping tables (positional
    slot remapping plus cross-lane shuffles when positions diverge
    between Apple and the destination's lane convention) and have no
    caller in the codebase yet. Those paths raise
    ``NotImplementedError`` so we don't silently emit wrong code if a
    future caller uses them.

    Current implementation: allocates a
    ``simdgroup_matrix<bfloat16_t, 8, 8>[mf*kf]`` output array. For each
    source tile, for each thread_elements slot, applies the optional
    body (with per-slot selector binding), converts f32 → bfloat16_t,
    and writes to the destination tile's thread_elements at the SAME
    Apple-layout position — Apple's per-lane layout is dtype-agnostic,
    so no shuffle is needed.

    Registers the output in ``frag_values`` so downstream MmaOps consume
    the simdgroup_matrix array directly without routing through
    ``_pack_b32_to_frag_array``.
    """
    src_layout = op.attrs["src_layout"]
    dst_layout = op.attrs["dst_layout"]
    src_dtype = op.attrs["src_dtype"]
    dst_dtype = op.attrs["dst_dtype"]
    if (src_layout, dst_layout) != ("acc", "a_frag"):
        raise NotImplementedError(
            f"FragConvertOp MSL: only acc→a_frag implemented "
            f"(got {src_layout}→{dst_layout}). Add a per-layout-pair "
            f"lowering template + slot-remap tables when a caller needs it."
        )
    from quark.ir.types import DType

    if (src_dtype, dst_dtype) != (DType.F32, DType.BF16):
        raise NotImplementedError(
            f"FragConvertOp MSL: only f32→bf16 implemented for acc→a_frag "
            f"(got {src_dtype}→{dst_dtype})."
        )
    shape_id = op.attrs["shape_id"]
    src_frags = op.src_frags
    selectors = op.selectors
    slot_to_sel = op.attrs.get("slot_to_selector_idx")
    sel_names = [ctx.names.name_for(s) for s in selectors] if selectors else []
    (out,) = op.results

    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(f"FragConvertOp: shape {shape_id!r} not in module.kernel_shapes")
    shape = module.kernel_shapes[shape_id]
    if not shape.msl:
        raise NotImplementedError(f"FragConvertOp: MmaShape {shape_id!r} has no `msl` field")
    ctx.uses_simdgroup_matrix = True
    dst_frag_dtype_str, mf, nf, kf_shape = _parse_msl_tiling(shape.msl)
    kf = len(src_frags)
    # Sanity check: each source ACC tile covers 8 K-cols; full A-frag
    # covers kf * 8 K-cols which must match shape's kf.
    if kf != kf_shape:
        # Partial A-frag — allowed but warn.
        pass

    # Verify each source is in frag_values (must come from MmaOp / LoadMatrix).
    for sf in src_frags:
        if sf.id not in ctx.frag_values:
            raise RuntimeError(
                "FragConvertOp: src frag isn't registered in frag_values "
                "— must be sourced from an MmaOp or LoadMatrixOp on MSL."
            )

    elem_var = op.body_input_var
    sel_var = op.body_selector_var
    body = op.body
    body_local_ids = _collect_body_local_value_ids(body.ops) if body else []
    skip_ids: set[int] = set()
    if elem_var is not None:
        skip_ids.add(elem_var.id)
    if sel_var is not None:
        skip_ids.add(sel_var.id)
    body_local_ids = [vid for vid in body_local_ids if vid not in skip_ids]

    # Allocate destination simdgroup_matrix<bfloat16_t, 8, 8>[mf*kf].
    # Tile ordering: K-major, matching PTX a_offsets layout where a_regs
    # for different k-slices are interleaved [k=0 m=0, k=0 m=1, k=1 m=0, k=1 m=1].
    # For m16n8k16: a_offsets = ((0,0),(8,0),(0,8),(8,8)) → out[0] = (dr=0,k=0),
    # out[1] = (dr=8,k=0), out[2] = (dr=0,k=8), out[3] = (dr=8,k=8).
    # Array layout: out_arr[ki*mf + mi] for (ki-th k-slice, mi-th m-tile).
    dst_name = ctx.names.fresh("frag")
    n_dst_tiles = mf * kf
    ctx.emit(f"simdgroup_matrix<{dst_frag_dtype_str}, 8, 8> {dst_name}[{n_dst_tiles}];")

    for ki, src_frag in enumerate(src_frags):
        src_name, src_acc_ty, src_mf, src_nf = ctx.frag_values[src_frag.id]
        for mi in range(src_mf):
            src_tile_idx = mi  # mf=2, nf=1 → tile index = m_idx
            dst_tile_idx = ki * mf + mi
            src_e_ref = ctx.names.fresh("conv_se")
            dst_e_ref = ctx.names.fresh("conv_de")
            ctx.emit(f"thread auto& {src_e_ref} = {src_name}[{src_tile_idx}].thread_elements();")
            ctx.emit(f"thread auto& {dst_e_ref} = {dst_name}[{dst_tile_idx}].thread_elements();")
            for slot in (0, 1):
                # Logical slot index into cd_offsets: mi * 2 + slot for
                # standard m16n8 bf16 (dr=0 for mi=0, dr=8 for mi=1;
                # slot 0 = dc=0, slot 1 = dc=1).
                logical_slot = mi * 2 + slot
                src_elem_name = ctx.names.fresh("conv_elem")
                ctx.emit(f"{src_acc_ty} {src_elem_name} = {src_e_ref}[{slot}];")
                # Bind body vars.
                if elem_var is not None:
                    ctx.names.bind(elem_var, (src_elem_name,), force=True)
                if sel_var is not None and slot_to_sel is not None:
                    sel_name = sel_names[slot_to_sel[logical_slot]]
                    ctx.names.bind(sel_var, (sel_name,), force=True)
                for vid in body_local_ids:
                    ctx.names._names.pop(vid, None)
                # Walk body (if present) and capture yielded value name.
                if body is not None:
                    yielded_name: str | None = None
                    for bop in body.ops:
                        if isinstance(bop, YieldOp):
                            yielded_name = ctx.names.name_for(bop.operands[0])
                            break
                        self._visit(bop, ctx)
                    assert yielded_name is not None
                    to_write = yielded_name
                else:
                    to_write = src_elem_name
                # Cast f32 → bfloat16_t and write.
                ctx.emit(f"{dst_e_ref}[{slot}] = ({dst_frag_dtype_str}){to_write};")

    ctx.names.bind(out, (dst_name,))
    ctx.frag_values[out.id] = (dst_name, dst_frag_dtype_str, mf, kf)


def visit_frag_for_each(self, op: FragForEachOp, ctx: _MslCtx) -> None:
    """Lower FragForEachOp on MSL: iterate (tile, slot), read
    thread_elements(), bind body_input_var/row_var/col_var, walk body.

    Position vars see the lane-dependent Apple positions — the body
    consumers (store_matrix, atomic_rmw, plain store) just read the
    row/col Values like any other IR U32.
    """
    shape_id = op.attrs["shape_id"]
    in_frag = op.in_frag

    if in_frag.id not in ctx.frag_values:
        raise RuntimeError(
            f"FragForEachOp (shape {shape_id!r}): operand frag isn't "
            "registered in frag_values — must be sourced from an MmaOp "
            "or LoadMatrixOp on the MSL path."
        )
    src_name, acc_ty, mf, nf = ctx.frag_values[in_frag.id]
    if nf != 1:
        raise NotImplementedError(
            f"FragForEachOp: multi-n-tile acc (nf={nf}) not yet supported on MSL"
        )
    n_tiles = mf * nf

    elem_var = op.body_input_var
    row_var = op.body_row_var
    col_var = op.body_col_var
    sel_var = op.body_selector_var
    selectors = op.selectors
    slot_to_sel = op.attrs.get("slot_to_selector_idx")
    sel_names = [ctx.names.name_for(s) for s in selectors] if selectors else []
    assert elem_var is not None and row_var is not None and col_var is not None

    # Compute Apple per-lane row/col bases ONCE at op entry.
    lane = ctx.names.fresh("lane")
    ap_row = ctx.names.fresh("ap_row")
    ap_col = ctx.names.fresh("ap_col")
    ctx.emit(f"uint {lane} = thread_index_in_simdgroup;")
    ctx.emit(f"uint {ap_row} = (({lane} >> 4u) & 1u) * 4u + (({lane} >> 1u) & 3u);")
    ctx.emit(f"uint {ap_col} = (({lane} >> 3u) & 1u) * 4u + ({lane} & 1u) * 2u;")

    body_local_ids = _collect_body_local_value_ids(op.body.ops)
    skip_ids = {elem_var.id, row_var.id, col_var.id}
    if sel_var is not None:
        skip_ids.add(sel_var.id)
    body_local_ids = [vid for vid in body_local_ids if vid not in skip_ids]

    for fi in range(n_tiles):
        e_ref = ctx.names.fresh("e")
        ctx.emit(f"thread auto& {e_ref} = {src_name}[{fi}].thread_elements();")
        for slot in (0, 1):
            # Element
            slot_elem = ctx.names.fresh("foreach_elem")
            ctx.emit(f"{acc_ty} {slot_elem} = {e_ref}[{slot}];")
            # Row / col for this (tile, slot). tile fi covers rows fi*8..fi*8+7.
            # Apple col0 + slot_idx covers col0 and col0+1.
            row_name = ctx.names.fresh("foreach_row")
            col_name = ctx.names.fresh("foreach_col")
            if fi == 0:
                ctx.emit(f"uint {row_name} = {ap_row};")
            else:
                ctx.emit(f"uint {row_name} = {ap_row} + {fi * 8}u;")
            if slot == 0:
                ctx.emit(f"uint {col_name} = {ap_col};")
            else:
                ctx.emit(f"uint {col_name} = {ap_col} + {slot}u;")
            # Bind body vars.
            ctx.names.bind(elem_var, (slot_elem,), force=True)
            ctx.names.bind(row_var, (row_name,), force=True)
            ctx.names.bind(col_var, (col_name,), force=True)
            if sel_var is not None and slot_to_sel is not None:
                logical_slot = fi * 2 + slot
                sel_name = sel_names[slot_to_sel[logical_slot]]
                ctx.names.bind(sel_var, (sel_name,), force=True)
            # Wipe body-local names.
            for vid in body_local_ids:
                ctx.names._names.pop(vid, None)
            # Walk body (skip terminator).
            for bop in op.body.ops:
                if isinstance(bop, YieldOp):
                    break
                self._visit(bop, ctx)


def visit_frag_reduce(self, op: FragReduceOp, ctx: _MslCtx) -> None:
    """Lower FragReduceOp on an accumulator to: local fold of
    thread_elements() slots within the tile(s) for each class, then
    butterfly shuffle across the 4 lanes that share the row.

    Apple's ``simdgroup_matrix<T, 8, 8>`` per-lane layout groups 4 lanes
    per row, differing in bits 0 and 3. So the butterfly is ``[1, 8]``:
    XOR bit 0 (flips col-pair within a col-block), then XOR bit 3
    (flips col-block). After 2 shuffles every lane in the row holds
    the full 8-col reduction.

    Tile mapping (m16n8 mf=2, nf=1):
      * rc 0 (dr=0) → tile 0, covers rows 0-7
      * rc 1 (dr=8) → tile 1, covers rows 8-15
    Each tile's thread_elements()[0,1] are in the SAME row — so the
    local fold is just the kind-op of the 2 slots.
    """
    from quark.ir.frag_tile import APPLE_ACC_ROW_REDUCE_BUTTERFLY

    kind = op.attrs["kind"]
    axis = op.attrs["axis"]
    cd_offsets = op.attrs["cd_offsets"]
    in_frag = op.in_frag

    if in_frag.id not in ctx.frag_values:
        raise RuntimeError(
            "FragReduceOp: operand frag must be registered in frag_values "
            "(i.e. sourced from MmaOp / LoadMatrixOp on MSL)."
        )
    src_name, acc_ty, mf, nf = ctx.frag_values[in_frag.id]

    if axis != "row":
        raise NotImplementedError(f"FragReduceOp axis={axis!r} not implemented on MSL yet")
    if nf != 1:
        raise NotImplementedError(f"FragReduceOp: multi-n-tile acc (nf={nf}) not yet supported")

    classes = sorted({dr for dr, _ in cd_offsets})
    # For m16n8 mf=2 nf=1, tile index == row class index.
    # For larger mf, map class → list of tile indices covering that band.
    # Simple case first: class i maps to tile i (mf == n_classes).
    if mf != len(classes):
        raise NotImplementedError(
            f"FragReduceOp: mf={mf} ≠ #row_classes={len(classes)} not supported yet"
        )

    _msl_kind = {"max": "max", "min": "min", "add": "+", "mul": "*"}[kind]
    kind_is_fn = kind in ("max", "min")

    for class_idx, result_val in enumerate(op.results):
        res_name = ctx.names.name_for(result_val)
        tile_idx = class_idx  # 1:1 for mf == n_classes
        # Read thread_elements()[0] and [1] — both live on the same row
        # (tile 8x8, row = per-lane from the Apple formula).
        te_ref = ctx.names.fresh("red_te")
        e0 = ctx.names.fresh("red_e0")
        e1 = ctx.names.fresh("red_e1")
        ctx.emit(f"thread auto& {te_ref} = {src_name}[{tile_idx}].thread_elements();")
        ctx.emit(f"{acc_ty} {e0} = {te_ref}[0];")
        ctx.emit(f"{acc_ty} {e1} = {te_ref}[1];")
        # Local fold of the 2 slots into res.
        if kind_is_fn:
            ctx.emit(f"{acc_ty} {res_name} = metal::{_msl_kind}({e0}, {e1});")
        else:
            ctx.emit(f"{acc_ty} {res_name} = {e0} {_msl_kind} {e1};")
        # Butterfly shuffle across 4 lanes in the row.
        for dist in APPLE_ACC_ROW_REDUCE_BUTTERFLY:
            tmp = ctx.names.fresh("red_sh")
            ctx.emit(f"{acc_ty} {tmp} = simd_shuffle_xor({res_name}, {dist}u);")
            if kind_is_fn:
                ctx.emit(f"{res_name} = metal::{_msl_kind}({res_name}, {tmp});")
            else:
                ctx.emit(f"{res_name} = {res_name} {_msl_kind} {tmp};")


def visit_load_matrix(self, op: LoadMatrixOp, ctx: _MslCtx) -> None:
    """Lower LoadMatrixOp to simdgroup_load calls."""
    tensor = op.attrs["src_tensor"]
    shape_id = op.attrs["shape_id"]
    which = op.attrs["which"]
    reg_offsets = op.attrs.get("reg_offsets")
    (out,) = op.results

    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(f"LoadMatrixOp: shape {shape_id!r} not in module.kernel_shapes")
    shape = module.kernel_shapes[shape_id]
    if not shape.msl:
        raise NotImplementedError(f"LoadMatrixOp: MmaShape {shape_id!r} has no `msl` field")
    ctx.uses_simdgroup_matrix = True
    frag_dtype_str, mf, nf, kf = _parse_msl_tiling(shape.msl)

    if which == "a":
        n_frags = mf * kf
    elif which == "b":
        n_frags = kf * nf
    else:
        n_frags = mf * nf

    frag_ty = msl_type(shape.acc_dtype) if which in ("c", "d") else frag_dtype_str

    row_val, col_val = op.operands
    row = ctx.names.name_for(row_val)
    col = ctx.names.name_for(col_val)
    row_stride = tensor.stride[0] if tensor.rank >= 2 else 1

    # Use the smem base name without the per-lane dyn_offset. On Metal,
    # simdgroup_load is a cooperative warp operation — it takes the tile
    # base address, not a per-lane address. The dyn_offset from
    # emit_smem_base (groupID * stride + tidIG * step) is PTX-specific.
    from quark.ir import SharedRegion as _SharedRegion

    if isinstance(tensor, _SharedRegion):
        alloc_id = tensor.alloc.id
        if alloc_id in ctx.smem_allocs:
            buf = ctx.smem_allocs[alloc_id][0]
        else:
            buf = f"smem_{alloc_id}"
        # Static offset (pipeline stage selection).
        static_off = f" + {tensor.static_offset}u" if tensor.static_offset else ""
        # Warp-level dynamic offset (uniform across simdgroup). Used for
        # per-warp N-partition in GEMM B tiles. The per-lane dyn_offset
        # is ignored — simdgroup_load distributes elements cooperatively.
        if tensor.warp_dyn_offset is not None:
            warp_off = ctx.names.name_for(tensor.warp_dyn_offset)
            static_off += f" + {warp_off}"
    else:
        buf = _tensor_buf_name(tensor, ctx)
        static_off = ""

    dst = ctx.names.fresh("frag")
    ctx.names.bind(out, (dst,))
    if which in ("c", "d"):
        ctx.frag_values[out.id] = (dst, msl_type(shape.acc_dtype), mf, nf)
    else:
        ctx.frag_values[out.id] = (
            dst,
            frag_ty,
            mf if which == "a" else kf,
            kf if which == "a" else nf,
        )
    ctx.emit(f"simdgroup_matrix<{frag_ty}, 8, 8> {dst}[{n_frags}];")

    # B is stored [N, K] row-major in smem. simdgroup_multiply_accumulate
    # does D = A × B (standard matmul). For C = A @ B^T, we need B in
    # [K, N] layout. The transpose flag on simdgroup_load transposes the
    # loaded 8x8 tile from [N, K] to [K, N].
    transpose = ", ulong2(0, 0), true" if which == "b" else ""

    if reg_offsets:
        for i, (dr, dc) in enumerate(reg_offsets):
            off = f"({row} + {dr}u) * {row_stride}u + ({col} + {dc}u){static_off}"
            ctx.emit(f"simdgroup_load({dst}[{i}], &{buf}[{off}], {row_stride}u{transpose});")
    else:
        for i in range(n_frags):
            mr = (i // (kf if which == "a" else nf)) * 8
            mc = (i % (kf if which == "a" else nf)) * 8
            off = f"({row} + {mr}u) * {row_stride}u + ({col} + {mc}u){static_off}"
            ctx.emit(f"simdgroup_load({dst}[{i}], &{buf}[{off}], {row_stride}u{transpose});")


def visit_store_matrix(self, op: StoreMatrixOp, ctx: _MslCtx) -> None:
    """Lower StoreMatrixOp to simdgroup_store calls."""
    tensor = op.attrs["dst_tensor"]
    shape_id = op.attrs["shape_id"]
    reg_offsets = op.attrs.get("reg_offsets")
    frag_val = op.operands[0]
    row_val, col_val = op.operands[1], op.operands[2]

    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(f"StoreMatrixOp: shape {shape_id!r} not in module.kernel_shapes")
    shape = module.kernel_shapes[shape_id]
    if not shape.msl:
        raise NotImplementedError(f"StoreMatrixOp: MmaShape {shape_id!r} has no `msl` field")
    ctx.uses_simdgroup_matrix = True
    _, mf, nf, _ = _parse_msl_tiling(shape.msl)
    n_frags = mf * nf

    frag = _frag_name(frag_val, ctx)
    row = ctx.names.name_for(row_val)
    col = ctx.names.name_for(col_val)
    buf = _tensor_buf_name(tensor, ctx)
    row_stride = tensor.stride[0] if tensor.rank >= 2 else 1

    if reg_offsets:
        for i, (dr, dc) in enumerate(reg_offsets):
            off = f"({row} + {dr}u) * {row_stride}u + ({col} + {dc}u)"
            ctx.emit(f"simdgroup_store({frag}[{i}], &{buf}[{off}], {row_stride}u);")
    else:
        for i in range(n_frags):
            mr = (i // nf) * 8
            mc = (i % nf) * 8
            off = f"({row} + {mr}u) * {row_stride}u + ({col} + {mc}u)"
            ctx.emit(f"simdgroup_store({frag}[{i}], &{buf}[{off}], {row_stride}u);")
