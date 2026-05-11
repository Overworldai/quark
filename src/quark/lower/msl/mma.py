"""MMA (simdgroup_matrix) visitor methods for the MSL lowerer.

EXEMPT FROM 500-LINE RULE: LoadMatrixOp / MmaOp / StoreMatrixOp and
the fragment→scalar extraction pipeline share the 2x2x2 Apple
bit-swizzle tables and per-dtype register-map generators; splitting
forces each visitor to re-import or re-derive them.

Handles LoadMatrixOp, MmaOp, StoreMatrixOp, and the fragment→scalar
extraction pipeline for kernel epilogues.
"""

from __future__ import annotations

from quark.device import DeviceFamily as _DeviceFamily
from quark.ir import (
    DType,
    FragApplyOp,
    FragConvertOp,
    FragForEachOp,
    FragReduceOp,
    LoadMatrixOp,
    MmaOp,
    StoreMatrixOp,
    YieldOp,
)
from quark.ir.mma_registry import payload_for as _payload_for

from .lower import _MslCtx, _tensor_buf_name
from .types import msl_type


def _msl_tiling_for(shape) -> str | None:
    return _payload_for(shape.name, _DeviceFamily.METAL)


def _parse_msl_tiling(msl_str: str | None) -> tuple[str, int, int, int]:
    """Parse an MSL tiling spec: 'frag_dtype:m_frags:n_frags:k_frags'."""
    if msl_str is None:
        raise ValueError("_parse_msl_tiling: no MSL tiling registered for shape")
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
    """Lower MmaOp to simdgroup_multiply_accumulate calls (or to MPP
    matmul2d when the shape's MSL payload starts with ``nax:``)."""
    shape_id = op.attrs["shape_id"]
    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(f"MmaOp: shape {shape_id!r} not in module.kernel_shapes")
    shape = module.kernel_shapes[shape_id]
    payload = _msl_tiling_for(shape)
    if not payload:
        raise NotImplementedError(
            f"MmaOp: MmaShape {shape_id!r} has no `msl` field — "
            f"Metal simdgroup lowering not available for this shape"
        )
    if payload.startswith("nax:"):
        return _visit_mma_nax(self, op, ctx, shape, payload)
    ctx.uses_simdgroup_matrix = True
    _, mf, nf, kf = _parse_msl_tiling(payload)

    a, b_frag, c = op.operands
    (d,) = op.results
    frag_dtype_str, _, _, _ = _parse_msl_tiling(_msl_tiling_for(shape))
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


# ---------------------------------------------------------------------------
# NAX (MPP matmul2d) lowering
# ---------------------------------------------------------------------------
#
# Apple's MetalPerformancePrimitives expose a per-simdgroup hardware
# matmul accelerator on M5+ ("NAX"). One ``mpp::tensor_ops::matmul2d``
# call computes a 16×32×16 BF16×BF16→F32 multiply-accumulate using
# overload 1 (1 left-input fragment + 2 right-input fragments → a
# 16×32 destination). Per-lane fragments are ``vec<bf16, 8>`` for
# A, ``vec<bf16, 16>`` for B (= 2 stitched 8-vecs), ``vec<float, 16>``
# for C/D (also 2 stitched 8-vecs).
#
# The IR's ``MmaOp`` triple (a, b, c) → d models one full call: ``a``
# is a width-8 BF16 Value, ``b`` and ``c`` are width-16 (BF16/F32);
# d is width-16 F32. The MSL emission pattern mirrors MLX's
# ``steel_gemm_fused_nax``: allocate three cooperative tensors, copy
# register fragments in, run the op, copy the destination back out
# into per-lane vec8 chunks the rest of the kernel can store.


def _emit_nax_preamble(ctx: _MslCtx) -> None:
    """Emit the matmul2d_descriptor + gemm_op handle plus the per-lane
    NAX coordinate (``_nax_fm``, ``_nax_fn``) once per kernel function.
    Subsequent NAX MmaOps and Load/Store ops reuse these vars.

    The descriptor is hard-coded to 16×32×16 BF16→F32 multiply-
    accumulate (overload 1: 1A + 2B → C[16×32]) — the only NAX shape
    Quark currently registers. When we add BF16×F16 / F16×F16 / etc.,
    the descriptor parameters move into the payload string.

    Lane→(fm, fn) layout matches MLX's ``BaseNAXFrag``:

        qid = lane >> 2
        fm  = (qid & 4) | ((lane >> 1) & 3)   # row in 0..7
        fn  = ((qid & 2) | (lane & 1)) * 4    # col {0, 4}

    Each lane reads/writes its own 2-row × 4-col sliver of every
    16×16 NAX fragment. Combined with the per-fragment offsets in
    the IR-level ``LoadMatrixOp`` / ``StoreMatrixOp`` visitors below
    this gives the full per-lane element addresses.
    """
    if ctx.nax_preamble_emitted:
        return
    ctx.nax_preamble_emitted = True
    # Per-lane NAX coordinate (fm, fn) for the BaseNAXFrag layout.
    ctx.emit("uint _nax_lane = thread_index_in_simdgroup;")
    ctx.emit("short _nax_qid = (short)(_nax_lane >> 2u);")
    ctx.emit("short _nax_fm = (short)((_nax_qid & 4) | ((_nax_lane >> 1) & 3));")
    ctx.emit("short _nax_fn = (short)(((_nax_qid & 2) | (_nax_lane & 1)) * 4);")
    # matmul2d descriptor + op handle are emitted PER MMA call below
    # (in _visit_mma_nax) — different MmaOps may have different
    # transpose_b settings so they need their own constexpr descriptor.
    # The preamble emits only the per-lane coordinate vars, which are
    # shared across all NAX ops (LoadMatrix / MmaOp / StoreMatrix).


# Per-lane within-fragment scalar layout: 2 rows × 4 cols per lane,
# rows separated by 8 (so a 16×16 fragment spans rows [fm, fm+8] and
# cols [fn, fn+3]). Same for A, B, C, D.
_NAX_FRAG_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (r * 8, c) for r in range(2) for c in range(4)
)


def _nax_frag_layout(which: str, shape=None) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Per-which (a/b/c/d) fragment count and (off_r, off_c) of each
    16×16 NAX fragment within the LoadMatrix/StoreMatrix tile.

    For the m=16 base shape (m16n32k16):
      A is 1 fragment (16×16), spanning a 16×16 tile.
      B is 2 fragments tiling the N-direction (16×32 effective). With
        ``transpose_b=true`` in the matmul2d_descriptor, B is stored as
        ``[N, K]`` row-major — the N-dimension is the outer (row) axis,
        so fragments split along ROW: offsets ``(0, 0)`` and ``(16, 0)``.
      C/D are 2 fragments tiling the N-direction in their natural
        ``[M, N]`` row-major storage — the N-dimension is the inner
        (col) axis here, so fragments split along COL: ``(0, 0)`` and
        ``(0, 16)``.

    For wider M-fragment shapes (e.g. m32n32k16), Apple's compiler
    decomposes the matmul2d_descriptor into multiple m16 fragments
    internally; the cooperative tensor's per-lane layout matches
    "stacked m16 fragments" — confirmed via probe in the world_engine
    MLX path (`M32NAXFrag` in `nax_m32.h`). So m32 = 2 stacked m16
    fragments along the M direction; the per-lane vec<8> array contains
    the first 8 elements for the top half (rows 0-15), the next 8 for
    the bottom half (rows 16-31). Same composition rule for C/D — m32
    output gets 2 vertically-stacked × 2 N-tiled = 4 sub-fragments.

    The cooperative-tensor index convention in ``visit_mma_nax`` —
    ``ct_b[0..7]`` = first frag, ``ct_b[8..15]`` = second frag — is
    parallel for B and C/D regardless of the per-frag direction; the
    layout difference only shows up in how the per-lane gmem addresses
    are computed.

    Returns (n_frags, [(off_r_per_frag, off_c_per_frag), ...]).
    """
    # Per-fragment dims (per-MMA): NAX m=16, n=16-tile of 32, k=16.
    M_FRAG = 16
    N_FRAG = 16

    m = M_FRAG if shape is None else int(shape.m)
    n = N_FRAG * 2 if shape is None else int(shape.n)
    n_m_frags = m // M_FRAG
    n_n_frags = n // N_FRAG

    if which == "a":
        # A: stacked m_frags × 1 K-frag. Each frag is 16×16 in (M, K).
        offsets = tuple((mi * M_FRAG, 0) for mi in range(n_m_frags))
        return n_m_frags, offsets
    if which == "b":
        # B is [N, K] under transpose_b — split along N (row).
        # Each frag is 16×16 in (N, K).
        offsets = tuple((ni * N_FRAG, 0) for ni in range(n_n_frags))
        return n_n_frags, offsets
    # C / D — natural [M, N] row-major. Layout convention: M outer (stacked)
    # then N inner (col-split). Order: (m0,n0), (m0,n1), (m1,n0), (m1,n1).
    offsets = tuple(
        (mi * M_FRAG, ni * N_FRAG) for mi in range(n_m_frags) for ni in range(n_n_frags)
    )
    return n_m_frags * n_n_frags, offsets


def _split_nax_components(comps: tuple[str, ...], expected_total: int) -> list[str]:
    """Validate and return the per-lane scalar names for a NAX fragment.

    NAX MmaOp operands carry their per-lane elements as ``expected_total``
    scalar components (8 for A, 16 for B/C/D). Anything else is a
    producer/consumer mismatch — the kernel author handed an MMA an
    operand whose width doesn't match the registered shape's a/b/c_regs.
    """
    if len(comps) != expected_total:
        raise NotImplementedError(
            f"NAX MmaOp: expected {expected_total} per-lane components, "
            f"got {len(comps)}. Check that the producer (LoadMatrix / "
            f"prior MmaOp / VecBuild) was sized to the shape's a/b/c_regs."
        )
    return list(comps)


def _emit_nax_mma_helper(
    *,
    helper_name: str,
    shape,
    transpose_b: bool,
    accumulate: bool,
    cast_a_from: DType | None,
    a_n_frags: int,
    b_n_frags: int,
    c_n_frags: int,
) -> str:
    """Emit a per-(shape, tb, acc, cast_a) MSL inline helper. The
    helper takes pointers to ``vec<T, 8>`` arrays — caller passes
    ``&fragment[fi]`` so a slice can pass a sub-range of a parent
    array. Cooperative_tensor allocation + the matmul2d descriptor
    live inside the helper, scoped to the call. After Apple inlines
    the helper, the cooperative_tensor's live range is the helper
    body — clean lifetime hint vs scattered per-call allocations.

    Helper signature (using accumulate=true, cast_a=None as example):

        inline void nax_mma_m16n32k16_tbT_accT(
            thread vec<float, 8>* C,
            const thread vec<bfloat, 8>* A,
            const thread vec<bfloat, 8>* B
        );
    """
    a_ty = msl_type(shape.a_dtype)
    b_ty = msl_type(shape.b_dtype)
    acc_ty = msl_type(shape.acc_dtype)
    a_param_ty = msl_type(cast_a_from) if cast_a_from is not None else a_ty
    tb_str = "true" if transpose_b else "false"
    acc_str = "true" if accumulate else "false"
    mode = "multiply_accumulate" if accumulate else "multiply"

    lines: list[str] = []
    lines.append(f"inline void {helper_name}(")
    lines.append(f"    thread vec<{acc_ty}, 8>* C,")
    lines.append(f"    const thread vec<{a_param_ty}, 8>* A,")
    lines.append(f"    const thread vec<{b_ty}, 8>* B")
    lines.append(") {")
    lines.append(
        f"    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor("
        f"{shape.m}, {shape.n}, {shape.k}, false, {tb_str}, {acc_str}, "
        f"mpp::tensor_ops::matmul2d_descriptor::mode::{mode});"
    )
    lines.append("    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;")
    lines.append(
        f"    auto ct_a = op.template "
        f"get_left_input_cooperative_tensor<{a_ty}, {b_ty}, {acc_ty}>();"
    )
    lines.append(
        f"    auto ct_b = op.template "
        f"get_right_input_cooperative_tensor<{a_ty}, {b_ty}, {acc_ty}>();"
    )
    lines.append(
        f"    auto ct_c = op.template "
        f"get_destination_cooperative_tensor<"
        f"decltype(ct_a), decltype(ct_b), {acc_ty}>();"
    )
    # Copy A → ct_a. With cast_a, narrow each element from
    # cast_a_from → a_dtype at the assign site (matches what HW's
    # nax_mma_nt does for the f32→bf16 case).
    if cast_a_from is not None:
        lines.append("    #pragma clang loop unroll(full)")
        lines.append(f"    for (short fi = 0; fi < {a_n_frags}; fi++) {{")
        lines.append("        #pragma clang loop unroll(full)")
        lines.append("        for (short si = 0; si < 8; si++) {")
        lines.append(f"            ct_a[fi*8 + si] = static_cast<{a_ty}>(A[fi][si]);")
        lines.append("        }")
        lines.append("    }")
    else:
        lines.append("    #pragma clang loop unroll(full)")
        lines.append(f"    for (short fi = 0; fi < {a_n_frags}; fi++) {{")
        lines.append("        #pragma clang loop unroll(full)")
        lines.append("        for (short si = 0; si < 8; si++) {")
        lines.append("            ct_a[fi*8 + si] = A[fi][si];")
        lines.append("        }")
        lines.append("    }")
    # Copy B → ct_b.
    lines.append("    #pragma clang loop unroll(full)")
    lines.append(f"    for (short fi = 0; fi < {b_n_frags}; fi++) {{")
    lines.append("        #pragma clang loop unroll(full)")
    lines.append("        for (short si = 0; si < 8; si++) {")
    lines.append("            ct_b[fi*8 + si] = B[fi][si];")
    lines.append("        }")
    lines.append("    }")
    # Copy C → ct_c (only when accumulating; ``multiply`` mode skips
    # the read).
    if accumulate:
        lines.append("    #pragma clang loop unroll(full)")
        lines.append(f"    for (short fi = 0; fi < {c_n_frags}; fi++) {{")
        lines.append("        #pragma clang loop unroll(full)")
        lines.append("        for (short si = 0; si < 8; si++) {")
        lines.append("            ct_c[fi*8 + si] = C[fi][si];")
        lines.append("        }")
        lines.append("    }")
    lines.append("    op.run(ct_a, ct_b, ct_c);")
    # Drain ct_c → C.
    lines.append("    #pragma clang loop unroll(full)")
    lines.append(f"    for (short fi = 0; fi < {c_n_frags}; fi++) {{")
    lines.append("        #pragma clang loop unroll(full)")
    lines.append("        for (short si = 0; si < 8; si++) {")
    lines.append("            C[fi][si] = ct_c[fi*8 + si];")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def _visit_mma_nax(self, op: MmaOp, ctx: _MslCtx, shape, payload: str) -> None:
    """Lower a NAX MmaOp to ``_nax_op.run(ct_a, ct_b, ct_c)``.

    Operands:
      * a: width-8 BF16 fragment (one 16×16 left tile; per-lane vec<bf16,8>)
      * b: width-16 BF16 fragment (two 16×16 right tiles stitched per lane)
      * c: width-16 F32 accumulator (two 16×16 destination tiles per lane)

    Result d carries the 16 per-lane F32 components of the new
    accumulator. Downstream consumers (next MmaOp, FragApply, store)
    read those components scalar-by-scalar — so the visitor binds d
    to a fresh array of 16 ``float`` lane locals, not a single
    cooperative-tensor handle.

    The cooperative tensors are scoped to this MMA call: allocated
    fresh, populated from registers, ``run``, drained back to
    registers, then go out of scope. Avoids carrying CT objects across
    loop iterations (the cooperative_tensor type is a template handle;
    carrying it confuses the Apple compiler's register allocator).
    """
    ctx.uses_nax = True
    _emit_nax_preamble(ctx)

    a, b, c = op.operands
    (d,) = op.results
    # Per-lane component counts come from the registered shape's
    # a_regs / b_regs / c_regs — the m=16 base shape uses 8/16/16; m=32
    # scales A and C linearly with the M-fragment count.
    a_comps = _split_nax_components(ctx.names.components(a), int(shape.a_regs))
    b_comps = _split_nax_components(ctx.names.components(b), int(shape.b_regs))
    c_comps = _split_nax_components(ctx.names.components(c), int(shape.c_regs))

    acc_ty = msl_type(shape.acc_dtype)  # "float"

    transpose_b = op.attrs.get("transpose_b", True)
    accumulate = op.attrs.get("accumulate", True)

    # B32 is a packing-carrier marker: load_matrix's a/b lane locals
    # are already declared at the shape's element type, so a B32 IR
    # dtype is "already matched" — no cast needed.
    def _needs_cast(src_dtype, dst_dtype) -> bool:
        if src_dtype == dst_dtype:
            return False
        if src_dtype is DType.B32:
            return False
        return True

    needs_cast_a = _needs_cast(a.dtype, shape.a_dtype)
    needs_cast_b = _needs_cast(b.dtype, shape.b_dtype)
    needs_cast_c = _needs_cast(c.dtype, shape.acc_dtype)
    cast_a_from = a.dtype if needs_cast_a else None

    # Helper-emission path: when all three operands have NAX-array
    # backing storage AND only A may need a cast (B/C casts are rare
    # and not in our attn kernel's hot path), invoke an inlined
    # ``nax_mma_*`` helper instead of spelling out the 50-line
    # cooperative_tensor + run + drain block per MMA. The helper is
    # defined once in the kernel header per (shape, tb, acc, cast_a)
    # combination; subsequent MmaOp call sites just emit a function
    # call. Apple inlines the helper, but the explicit scope cleans
    # up the cooperative_tensor live ranges → matches HW's source
    # structure (``nax_mma()`` / ``nax_mma_nt()`` helpers).
    a_arr = ctx.nax_frag_arrays.get(a.id)
    b_arr = ctx.nax_frag_arrays.get(b.id)
    c_arr = ctx.nax_frag_arrays.get(c.id)
    can_use_helper = (
        a_arr is not None
        and b_arr is not None
        and c_arr is not None
        and not needs_cast_b
        and not needs_cast_c
    )

    if can_use_helper:
        assert a_arr is not None and b_arr is not None and c_arr is not None
        a_n_frags = a_arr[1]
        b_n_frags = b_arr[1]
        c_n_frags = c_arr[1]
        helper_sig = (
            shape.name,
            transpose_b,
            accumulate,
            cast_a_from.name if cast_a_from else None,
        )
        tb_tag = "tbT" if transpose_b else "tbF"
        acc_tag = "accT" if accumulate else "accF"
        cast_tag = f"_cast{cast_a_from.name}" if cast_a_from is not None else ""
        helper_name = f"nax_mma_{shape.name}_{tb_tag}_{acc_tag}{cast_tag}"
        if helper_sig not in ctx.nax_mma_helpers:
            ctx.nax_mma_helpers.add(helper_sig)
            ctx.nax_mma_helper_defs.append(
                _emit_nax_mma_helper(
                    helper_name=helper_name,
                    shape=shape,
                    transpose_b=transpose_b,
                    accumulate=accumulate,
                    cast_a_from=cast_a_from,
                    a_n_frags=a_n_frags,
                    b_n_frags=b_n_frags,
                    c_n_frags=c_n_frags,
                )
            )

        # Helper expects ``thread vec<T, 8>*`` arguments. Pass each
        # operand as ``&array[start]``. For full arrays start=0 (
        # ``&array[0]``), for sub-frag slices start is the sub-frag
        # offset recorded by ``_visit_frag_slice``.
        def _helper_arg(arr_entry):
            arr_name, _n = arr_entry
            # The slice path stores e.g. ``parent[1]`` as the name; full
            # arrays store just ``parent``. Both are valid lvalues; we
            # simply prefix with ``&`` to take a pointer.
            return f"&{arr_name}[0]" if "[" not in arr_name else f"&{arr_name}"

        ctx.emit(
            f"{helper_name}({_helper_arg(c_arr)}, {_helper_arg(a_arr)}, {_helper_arg(b_arr)});"
        )
        # D shares C's storage in-place (drain happened inside helper).
        ctx.names.bind(d, tuple(c_comps), force=True)
        ctx.nax_frag_ids.add(c.id)
        ctx.nax_frag_ids.add(d.id)
        ctx.nax_frag_arrays[d.id] = c_arr
        return

    # Fallback: original inline emission, used when one or more
    # operands aren't array-backed (e.g. the first GEMM1 MMA whose C
    # is a scalar zero_frag). The helper preconditions don't hold;
    # spell out the cooperative_tensor block as before.
    a_ty = msl_type(shape.a_dtype)
    b_ty = msl_type(shape.b_dtype)
    tb_str = "true" if transpose_b else "false"
    acc_str = "true" if accumulate else "false"
    mode = "multiply_accumulate" if accumulate else "multiply"
    desc_var = ctx.names.fresh("nax_desc")
    op_var = ctx.names.fresh("nax_op")
    ctx.emit(
        f"constexpr auto {desc_var} = "
        f"mpp::tensor_ops::matmul2d_descriptor("
        f"{shape.m}, {shape.n}, {shape.k}, false, {tb_str}, {acc_str}, "
        f"mpp::tensor_ops::matmul2d_descriptor::mode::{mode});"
    )
    ctx.emit(f"mpp::tensor_ops::matmul2d<{desc_var}, metal::execution_simdgroup> {op_var};")
    ct_a = ctx.names.fresh("ct_a")
    ct_b = ctx.names.fresh("ct_b")
    ct_c = ctx.names.fresh("ct_c")
    ctx.emit(
        f"auto {ct_a} = {op_var}.template "
        f"get_left_input_cooperative_tensor<{a_ty}, {b_ty}, {acc_ty}>();"
    )
    ctx.emit(
        f"auto {ct_b} = {op_var}.template "
        f"get_right_input_cooperative_tensor<{a_ty}, {b_ty}, {acc_ty}>();"
    )
    ctx.emit(
        f"auto {ct_c} = {op_var}.template "
        f"get_destination_cooperative_tensor<"
        f"decltype({ct_a}), decltype({ct_b}), {acc_ty}>();"
    )

    def _assign(ct: str, names: list[str], src_dtype, dst_dtype, dst_ty: str) -> None:
        if _needs_cast(src_dtype, dst_dtype):
            for i, nm in enumerate(names):
                ctx.emit(f"{ct}[{i}] = static_cast<{dst_ty}>({nm});")
        else:
            for i, nm in enumerate(names):
                ctx.emit(f"{ct}[{i}] = {nm};")

    _assign(ct_a, a_comps, a.dtype, shape.a_dtype, a_ty)
    _assign(ct_b, b_comps, b.dtype, shape.b_dtype, b_ty)
    if accumulate:
        _assign(ct_c, c_comps, c.dtype, shape.acc_dtype, acc_ty)

    ctx.emit(f"{op_var}.run({ct_a}, {ct_b}, {ct_c});")

    # When C is scalar-form (no nax_frag_arrays entry — typically the
    # first MMA in a chain whose C came from ``_zero_frag``), allocate
    # a fresh ``vec<acc_ty, 8> D_arr[c_n_frags];`` for D's output and
    # drain into it. This lets the NEXT MMA in the chain see C
    # (= this D) as array-backed, so the helper-call path can fire
    # for the rest of the chain. Costs one extra register decl per
    # chain entry, saves N inline MMA blocks.
    c_has_array = c.id in ctx.nax_frag_arrays
    if c_has_array:
        # In-place chain: D shares C's array storage. Drain writes
        # back into the existing array slots.
        if needs_cast_c:
            c_ty = msl_type(c.dtype)
            for i, name in enumerate(c_comps):
                ctx.emit(f"{name} = static_cast<{c_ty}>({ct_c}[{i}]);")
        else:
            for i, name in enumerate(c_comps):
                ctx.emit(f"{name} = {ct_c}[{i}];")
        ctx.names.bind(d, tuple(c_comps), force=True)
        ctx.nax_frag_arrays[d.id] = ctx.nax_frag_arrays[c.id]
    else:
        # Allocate fresh array storage for D and drain into it.
        c_n_frags = len(c_comps) // _NAX_SUB_WIDTH
        d_arr = ctx.names.fresh("nax_d")
        d_comps = _emit_nax_frag_decl(ctx, d_arr, c_n_frags, acc_ty)
        for i in range(len(c_comps)):
            fi, si = divmod(i, _NAX_SUB_WIDTH)
            if needs_cast_c:
                c_ty = msl_type(c.dtype)
                ctx.emit(f"{d_arr}[{fi}][{si}] = static_cast<{c_ty}>({ct_c}[{i}]);")
            else:
                ctx.emit(f"{d_arr}[{fi}][{si}] = {ct_c}[{i}];")
        ctx.names.bind(d, d_comps, force=True)
        _record_nax_frag_array(ctx, d, d_arr, c_n_frags)
    ctx.nax_frag_ids.add(c.id)
    ctx.nax_frag_ids.add(d.id)


def _fold_view_row(row_expr: str, tensor, ctx: _MslCtx) -> str:
    """Fold a GlobalTensor.view's static_row_offset + dyn_row_offset
    into the row base expression. Mirrors what ``_compute_tensor_offset``
    does for plain load/store sites — the NAX load_matrix path needs
    the same treatment so ``view(row=base)`` actually starts at base."""
    from quark.ir.tensor import GlobalTensor

    if not isinstance(tensor, GlobalTensor):
        return row_expr
    parts = [row_expr]
    if tensor.static_row_offset:
        parts.append(f"{tensor.static_row_offset}u")
    if tensor.dyn_row_offset is not None:
        parts.append(ctx.names.name_for(tensor.dyn_row_offset))
    if len(parts) == 1:
        return row_expr
    return "(" + " + ".join(parts) + ")"


def _fold_view_col(col_expr: str, tensor, ctx: _MslCtx) -> str:
    """Same as _fold_view_row but for column offsets."""
    from quark.ir.tensor import GlobalTensor

    if not isinstance(tensor, GlobalTensor):
        return col_expr
    parts = [col_expr]
    if tensor.static_col_offset:
        parts.append(f"{tensor.static_col_offset}u")
    if tensor.dyn_col_offset is not None:
        parts.append(ctx.names.name_for(tensor.dyn_col_offset))
    if len(parts) == 1:
        return col_expr
    return "(" + " + ".join(parts) + ")"


def _fold_smem_flat_offset(tensor, ctx: _MslCtx) -> str:
    """Build a flat-element-offset suffix for a SharedRegion view.

    Returns ``""`` when the tensor isn't a SharedRegion or carries no
    offsets, otherwise ``" + <expr>"`` so callers can append directly
    to an address expression. Folds ``static_offset``, ``dyn_offset``,
    and ``warp_dyn_offset`` (all in element units, by SharedRegion's
    convention) into a single additive term.

    NAX load_matrix / store_matrix on a SharedRegion view (e.g. the
    inline-Q-RoPE smem region in ``owl_attn.nax``, where each
    simdgroup's 16-row band lives at ``sg_q_off * Dh`` element offset)
    needs this — without it, the per-simdgroup offset is silently
    dropped and every simdgroup reads the same band, producing
    cross-simdgroup output duplication.
    """
    from quark.ir.tensor import SharedRegion

    if not isinstance(tensor, SharedRegion):
        return ""
    parts: list[str] = []
    if tensor.static_offset:
        parts.append(f"{tensor.static_offset}u")
    if tensor.dyn_offset is not None:
        parts.append(ctx.names.name_for(tensor.dyn_offset))
    if tensor.warp_dyn_offset is not None:
        parts.append(ctx.names.name_for(tensor.warp_dyn_offset))
    if not parts:
        return ""
    return " + " + " + ".join(parts)


# ---------------------------------------------------------------------------
# NAX-fragment vec<T,8> array storage helpers.
#
# A NAX fragment of width N (= n_frags × 8) is backed by a single
# ``vec<dtype, 8> name[n_frags];`` declaration. Per-slot accesses use
# ``name[fi][si]`` indexing strings as the Value's lane components.
# Apple's compiler treats the array as 2 (or N) contiguous SIMD8
# register groups — the same shape the hand-written kernel uses for
# its ``vec<float,8> S_frags[2]`` declarations. Compared to the old
# per-slot scalar pattern (16 separate ``float fragN;`` decls), this
# cuts the SSA-value count Apple's optimizer has to track and gives
# tighter register allocation.
# ---------------------------------------------------------------------------

_NAX_SUB_WIDTH = 8


def _nax_array_components(name: str, n_frags: int) -> tuple[str, ...]:
    """The N×8 component-name strings for an array-backed NAX fragment."""
    return tuple(f"{name}[{fi}][{si}]" for fi in range(n_frags) for si in range(_NAX_SUB_WIDTH))


def _emit_nax_frag_decl(ctx: _MslCtx, name: str, n_frags: int, msl_dtype: str) -> tuple[str, ...]:
    """Emit ``vec<msl_dtype, 8> name[n_frags];`` and return the
    ``name[fi][si]`` component strings."""
    if n_frags == 1:
        # Single sub-frag: still wrap in a 1-element array for uniform
        # indexing. Apple's compiler tolerates ``vec<T,8> name[1]`` with
        # no overhead vs a bare ``vec<T,8> name`` decl.
        ctx.emit(f"vec<{msl_dtype}, {_NAX_SUB_WIDTH}> {name}[1];")
    else:
        ctx.emit(f"vec<{msl_dtype}, {_NAX_SUB_WIDTH}> {name}[{n_frags}];")
    return _nax_array_components(name, n_frags)


def _record_nax_frag_array(ctx: _MslCtx, value, array_name: str, n_frags: int) -> None:
    """Track that ``value`` is backed by ``array_name[n_frags]`` so the
    MMA helper-emission path can pass the array reference to its
    inlined ``nax_mma_*`` helper instead of re-deriving from the
    component strings."""
    ctx.nax_frag_arrays[value.id] = (array_name, n_frags)


def _visit_load_matrix_nax(self, op, ctx: _MslCtx, shape, tensor, which: str) -> None:
    """Lower a NAX-shape LoadMatrixOp to coalesced vec4 reads.

    The NAX BaseNAXFrag layout is 2 rows × 4 contiguous cols per lane.
    Each 4-col group maps to a single ``bfloat4`` / ``float4``
    reinterpret_cast load — explicitly contiguous for the compiler to
    vectorize. Verified to produce identical perf to MLX's templated
    ``BaseNAXFrag::load`` with ``Int<1>{}`` stride specialization.

    Fragment count and dtype depend on ``which``:

      * ``"a"`` → 1 fragment × 8 elements per lane = width-8 BF16
      * ``"b"`` → 2 fragments × 8 = width-16 BF16 (16×32 N-tile)
      * ``"c"`` / ``"d"`` → 2 fragments × 8 = width-16 F32
    """
    from .visitors import _msl_addr_space

    ctx.uses_nax = True
    _emit_nax_preamble(ctx)

    (out,) = op.results
    row_val, col_val = op.operands
    row = ctx.names.name_for(row_val)
    col = ctx.names.name_for(col_val)
    buf = _tensor_buf_name(tensor, ctx)
    row_stride = tensor.stride[0] if tensor.rank >= 2 else 1
    addr_space = _msl_addr_space(tensor)

    # Fold the GlobalTensor's view() row/col offsets into the row/col
    # base expressions. Without this, ``Vt_cache.view(row=vt_row_base,
    # col=kv_off)`` would silently read at row=0/col=0 — the view's
    # offsets must propagate into the address math the same way they
    # do for ``_compute_tensor_offset`` on regular load/store.
    row = _fold_view_row(row, tensor, ctx)
    col = _fold_view_col(col, tensor, ctx)

    if which == "a":
        comp_dtype = msl_type(shape.a_dtype)
    elif which == "b":
        comp_dtype = msl_type(shape.b_dtype)
    else:
        comp_dtype = msl_type(shape.acc_dtype)

    # NAX frag layout: 2 rows (offsets 0, 8) × 4 contiguous cols per
    # lane. Each row's 4 cols emit one vec4 reinterpret_cast load.
    _ROW_OFFSETS = (0, 8)  # kElemRowsJump = 8

    n_frags, frag_offsets = _nax_frag_layout(which, shape)
    # Single ``vec<T, 8> arr[n_frags];`` decl backs all per-lane slots;
    # arr[fi][si] are the Value's component names. The 4-col vec4
    # reads write directly into ``arr[fi][ri_idx*4 + j]``.
    arr = ctx.names.fresh(f"l{which}_frag")
    comps = _emit_nax_frag_decl(ctx, arr, n_frags, comp_dtype)
    smem_off = _fold_smem_flat_offset(tensor, ctx)
    for fi, (off_r, off_c) in enumerate(frag_offsets):
        for ri_idx, ri in enumerate(_ROW_OFFSETS):
            r_off = f"({row} + (uint)_nax_fm + {off_r + ri}u)"
            c_off = f"({col} + (uint)_nax_fn + {off_c}u)"
            addr = f"{r_off} * {row_stride}u + {c_off}{smem_off}"
            tmp = ctx.names.fresh(f"v{which}")
            ctx.emit(
                f"{comp_dtype}4 {tmp} = "
                f"*reinterpret_cast<const {addr_space} {comp_dtype}4*>"
                f"(&{buf}[{addr}]);"
            )
            for j in range(4):
                ctx.emit(f"{arr}[{fi}][{ri_idx * 4 + j}] = {tmp}[{j}];")
    ctx.names.bind(out, comps)
    # Tag this fragment as NAX-storage so Frag* visitors route correctly.
    ctx.nax_frag_ids.add(out.id)
    _record_nax_frag_array(ctx, out, arr, n_frags)


def _visit_store_matrix_nax(
    self, op, ctx: _MslCtx, shape, tensor, frag_val, row_val, col_val
) -> None:
    """Lower a NAX-shape StoreMatrixOp to per-lane scalar writes.

    Mirrors ``_visit_load_matrix_nax``: each lane writes its sliver
    of every 16×16 destination fragment. Supports the standard "d"
    accumulator output (16 F32 components) — the only ``which`` a
    typical NAX GEMM emits a store for.
    """
    ctx.uses_nax = True
    _emit_nax_preamble(ctx)

    row = ctx.names.name_for(row_val)
    col = ctx.names.name_for(col_val)
    buf = _tensor_buf_name(tensor, ctx)
    row_stride = tensor.stride[0] if tensor.rank >= 2 else 1

    # Same view-offset fold as _visit_load_matrix_nax — needed for the
    # output store too, otherwise output.view(row=q_row_base) silently
    # writes at row=0.
    row = _fold_view_row(row, tensor, ctx)
    col = _fold_view_col(col, tensor, ctx)

    comps = ctx.names.components(frag_val)
    expected_comps = int(shape.c_regs)
    if len(comps) != expected_comps:
        raise NotImplementedError(
            f"NAX StoreMatrixOp: expected {expected_comps} per-lane components "
            f"(c_regs for shape {shape.name}), got {len(comps)}. Frag must be "
            f"the destination of a NAX MmaOp or a same-width FragApply/Convert."
        )

    from .visitors import _msl_addr_space

    dst_dtype = msl_type(tensor.dtype)
    addr_space = _msl_addr_space(tensor)
    _ROW_OFFSETS = (0, 8)

    _n_frags, frag_offsets = _nax_frag_layout("d", shape)
    smem_off = _fold_smem_flat_offset(tensor, ctx)
    idx = 0
    for off_r, off_c in frag_offsets:
        for ri in _ROW_OFFSETS:
            r_off = f"({row} + (uint)_nax_fm + {off_r + ri}u)"
            c_off = f"({col} + (uint)_nax_fn + {off_c}u)"
            addr = f"{r_off} * {row_stride}u + {c_off}{smem_off}"
            elems = ", ".join(f"static_cast<{dst_dtype}>({comps[idx + j]})" for j in range(4))
            ctx.emit(
                f"*reinterpret_cast<{addr_space} {dst_dtype}4*>"
                f"(&{buf}[{addr}]) = {dst_dtype}4({elems});"
            )
            idx += 4


def _visit_store_matrix_gate_residual_nax(
    self, op, ctx: _MslCtx, shape, frag_val, row_val, col_val
) -> None:
    """Lower a NAX-shape StoreMatrixGateResidualOp.

    Per-lane fused-store: for each of the 16 fragment slots, read the
    matching residual + gate elements from gmem (bf16), upcast to F32,
    compute ``elem = residual + gate * frag_elem``, downcast to the
    destination dtype, and store. Walks the same ``BaseNAXFrag``
    layout as ``_visit_store_matrix_nax`` so the store is correct
    regardless of which (am, tn) tile the NAX inner loop produced.

    The residual + gate are read with vec4 loads (4-wide) matching
    the existing NAX store's vec4 write — saves 12 scalar loads per
    lane. Gate broadcast: ``gate_row = (lane_row) // m_per_group``;
    when m_per_group == M (the spec G==1 case) the divide collapses
    to a constant 0 the Metal compiler folds out.
    """
    ctx.uses_nax = True
    _emit_nax_preamble(ctx)

    dst_tensor = op.attrs["dst_tensor"]
    residual_tensor = op.attrs["residual_tensor"]
    gate_tensor = op.attrs["gate_tensor"]
    m_per_group = int(op.attrs["m_per_group"])

    row = ctx.names.name_for(row_val)
    col = ctx.names.name_for(col_val)
    row = _fold_view_row(row, dst_tensor, ctx)
    col = _fold_view_col(col, dst_tensor, ctx)

    dst_buf = _tensor_buf_name(dst_tensor, ctx)
    res_buf = _tensor_buf_name(residual_tensor, ctx)
    gate_buf = _tensor_buf_name(gate_tensor, ctx)

    dst_stride = dst_tensor.stride[0] if dst_tensor.rank >= 2 else 1
    res_stride = residual_tensor.stride[0] if residual_tensor.rank >= 2 else 1
    gate_stride = gate_tensor.stride[0] if gate_tensor.rank >= 2 else 1

    comps = ctx.names.components(frag_val)
    expected_comps = int(shape.c_regs)
    if len(comps) != expected_comps:
        raise NotImplementedError(
            f"NAX StoreMatrixGateResidualOp: expected {expected_comps} per-lane "
            f"components (c_regs for shape {shape.name}), got {len(comps)}. "
            f"Frag must be the destination of a NAX MmaOp."
        )

    from .visitors import _msl_addr_space

    dst_dtype = msl_type(dst_tensor.dtype)
    res_dtype = msl_type(residual_tensor.dtype)
    gate_dtype = msl_type(gate_tensor.dtype)
    dst_addr_space = _msl_addr_space(dst_tensor)
    res_addr_space = _msl_addr_space(residual_tensor)
    gate_addr_space = _msl_addr_space(gate_tensor)

    _ROW_OFFSETS = (0, 8)
    _, frag_offsets = _nax_frag_layout("d", shape)

    idx = 0
    for off_r, off_c in frag_offsets:
        for ri in _ROW_OFFSETS:
            r_off = f"({row} + (uint)_nax_fm + {off_r + ri}u)"
            c_off = f"({col} + (uint)_nax_fn + {off_c}u)"
            # Per-tile gmem addresses for residual + dst (same row/col),
            # gate (row collapsed via m_per_group divide).
            dst_addr = f"{r_off} * {dst_stride}u + {c_off}"
            res_addr = f"{r_off} * {res_stride}u + {c_off}"
            # G == 1 path collapses to ``0u * gate_stride`` which the
            # compiler optimises out; G > 1 emits one integer divide
            # per lane per fragment slice (4 slices × 16 lanes = 64
            # divides per simdgroup per tile — negligible on M5+).
            gate_row_expr = f"(({r_off}) / {m_per_group}u)"
            gate_addr = f"{gate_row_expr} * {gate_stride}u + {c_off}"

            # Vec-4 load of residual + gate, upcast to F32 lane-by-lane.
            res_var = ctx.names.fresh("nax_res4")
            gate_var = ctx.names.fresh("nax_gate4")
            ctx.emit(
                f"{res_dtype}4 {res_var} = "
                f"*reinterpret_cast<{res_addr_space} const {res_dtype}4*>"
                f"(&{res_buf}[{res_addr}]);"
            )
            ctx.emit(
                f"{gate_dtype}4 {gate_var} = "
                f"*reinterpret_cast<{gate_addr_space} const {gate_dtype}4*>"
                f"(&{gate_buf}[{gate_addr}]);"
            )

            # Combine in F32 (accumulator precision), then cast to dst.
            elems = ", ".join(
                f"static_cast<{dst_dtype}>("
                f"static_cast<float>({res_var}[{j}]) + "
                f"static_cast<float>({gate_var}[{j}]) * {comps[idx + j]}"
                f")"
                for j in range(4)
            )
            ctx.emit(
                f"*reinterpret_cast<{dst_addr_space} {dst_dtype}4*>"
                f"(&{dst_buf}[{dst_addr}]) = {dst_dtype}4({elems});"
            )
            idx += 4


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


def _visit_frag_apply_nax(self, op: FragApplyOp, ctx: _MslCtx) -> None:
    """Lower FragApplyOp for NAX per-lane ``vec<T, 8>[n_frags]`` storage.

    NAX BaseNAXFrag layout: 8 elements per lane (kElemRows=2,
    kElemCols=4). For c_regs=16 (TN=2, two stacked sub-tiles) the
    lane holds 16 scalar names, treated here as 2 fragments of 8.

    Slot → row-class mapping (per BaseNAXFrag's ``get_coord``):

        slot 0,1,2,3 → row class 0 (rows 0..7 group)
        slot 4,5,6,7 → row class 1 (rows 8..15 group)
        Same pattern repeats per stacked fragment.

    Emission:

        acc_ty {out}[N];                     // N = c_regs scalars total
        for slot in 0..N:
            acc_ty elem = {src}[slot];
            <body — input_var bound to elem, sel_var to selectors[slot_to_sel[slot]]>
            {out}[slot] = yielded;

    The NAX MmaOp / LoadMatrix produces fragments as flat tuples of
    scalar names (one per lane element). We iterate them directly —
    no ``thread_elements()`` indirection needed.
    """
    in_frag = op.in_frag
    selectors = op.selectors
    slot_to_sel = op.attrs.get("slot_to_selector_idx")
    (out,) = op.results

    # NAX fragments are stored as flat per-lane scalar tuples in
    # ctx.names. Each name holds one of the c_regs slot values.
    src_comps = ctx.names.components(in_frag)
    n_slots = len(src_comps)

    input_var = op.body_input_var
    assert input_var is not None
    sel_var = op.body_selector_var
    sel_names = [ctx.names.name_for(s) for s in selectors] if selectors else []

    # Resolve element MSL type from the input fragment's IR dtype.
    # NAX C accumulators use the shape's acc_dtype (typically f32);
    # NAX A inputs use a_dtype (bf16). Either way, the IR Value carries
    # the right scalar dtype through.
    acc_ty = msl_type(in_frag.dtype)

    body_local_ids = _collect_body_local_value_ids(op.body.ops)
    skip_ids = {input_var.id}
    if sel_var is not None:
        skip_ids.add(sel_var.id)
    body_local_ids = [vid for vid in body_local_ids if vid not in skip_ids]

    # Output goes into a single ``vec<acc_ty, 8> out_arr[n_frags];``
    # array. Each per-slot body walk emits the body's ops as fresh
    # scalars; the last (yielded) value gets written through to
    # ``out_arr[fi][si]``. The array form replaces the previous N
    # independent scalar lane-locals — Apple's compiler keeps the
    # array as n_frags contiguous SIMD8 register groups, matching the
    # hand-written kernel's ``vec<float, 8> S_frags[N]`` storage and
    # avoiding per-slot SSA fragmentation.
    n_frags = n_slots // _NAX_SUB_WIDTH
    out_arr = ctx.names.fresh("frag_out")
    out_comps = _emit_nax_frag_decl(ctx, out_arr, n_frags, acc_ty)

    for slot in range(n_slots):
        # Bind input_var directly to the source component (no
        # ``float apply_elemN = src;`` alias). The body's first op
        # reads input_var.name = src_comps[slot].
        ctx.names.bind(input_var, (src_comps[slot],), force=True)
        if sel_var is not None and slot_to_sel is not None:
            sel_name = sel_names[slot_to_sel[slot]]
            ctx.names.bind(sel_var, (sel_name,), force=True)
        # Wipe body-local names so the walk re-allocates fresh.
        for vid in body_local_ids:
            ctx.names._names.pop(vid, None)
        # Walk body ops; on the terminator, write the yielded scalar
        # into the array slot.
        fi, si = divmod(slot, _NAX_SUB_WIDTH)
        for bop in op.body.ops:
            if isinstance(bop, YieldOp):
                yielded = bop.operands[0]
                ctx.emit(f"{out_arr}[{fi}][{si}] = {ctx.names.name_for(yielded)};")
                break
            self._visit(bop, ctx)

    ctx.names.bind(out, out_comps)
    # Output is also NAX-stored — propagate the discriminator.
    ctx.nax_frag_ids.add(out.id)
    _record_nax_frag_array(ctx, out, out_arr, n_frags)


def visit_frag_apply(self, op: FragApplyOp, ctx: _MslCtx) -> None:
    """Lower FragApplyOp.

    Dispatches on the input fragment's storage kind:

      * NAX-stored fragment (``in_frag.id in ctx.nax_frag_ids``) →
        ``_visit_frag_apply_nax`` — operates on per-lane
        ``vec<T, 8>[n_frags]`` arrays, slot count is 8 per fragment
        with row-class layout (slots 0..3 → row class 0, 4..7 → row
        class 1, repeats per stacked fragment for c_regs > 8).

      * simdgroup_matrix-stored fragment (default) → emits the body
        once per (tile, thread_elements-slot). For a ``(mf, nf)``
        accumulator with ``mf·nf`` tiles:

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
    if op.in_frag.id in ctx.nax_frag_ids:
        return _visit_frag_apply_nax(self, op, ctx)
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


def _visit_frag_convert_nax(self, op: FragConvertOp, ctx: _MslCtx) -> None:
    """Lower FragConvertOp for NAX per-lane storage.

    NAX C-frag and A-frag share the same per-lane layout (8 elements,
    kElemRows=2, kElemCols=4) — the only difference is dtype. The
    convert is a pure per-element ``static_cast`` over the lane's
    scalar components. No shuffles, no slot remapping.

    Supported case (sufficient for IR-emitted attention's S → A path):
      * num_src_frags == 1: one source fragment in, one destination
        fragment out, same per-lane width.
      * any (src_dtype, dst_dtype) where MSL accepts a static_cast.
      * any (src_layout, dst_layout): for NAX the per-lane storage is
        identical regardless of layout name (acc/a_frag/c/d all use
        kElemRows=2, kElemCols=4); the layout label is purely a
        semantic hint for downstream MMA consumption.
      * Optional body applied per-element before the cast (matches the
        simdgroup path semantics for online-softmax exp).

    Width-changing converts (one C-frag → multiple A-frags, used in
    the simdgroup_matrix path's PTX-shaped K-packing) are NOT
    supported here. NAX doesn't need them — its C-frag and A-frag
    share the same per-lane width. If a future kernel wants
    "split one width-16 NAX frag into two width-8 frags", that's a
    separate split op rather than a mode of FragConvertOp.
    """
    src_dtype = op.attrs["src_dtype"]
    dst_dtype = op.attrs["dst_dtype"]
    num_src_frags = int(op.attrs["num_src_frags"])

    if num_src_frags != 1:
        raise NotImplementedError(
            f"FragConvertOp NAX: num_src_frags={num_src_frags} not supported (only 1 — same-width)."
        )

    src_frag = op.operands[0]
    selectors = op.operands[1:]
    slot_to_sel = op.attrs.get("slot_to_selector_idx")
    (out,) = op.results

    src_comps = ctx.names.components(src_frag)
    n_slots = len(src_comps)
    src_ty = msl_type(src_dtype)
    dst_ty = msl_type(dst_dtype)

    input_var = op.body_input_var
    sel_var = op.body_selector_var
    sel_names = [ctx.names.name_for(s) for s in selectors] if selectors else []
    has_body = input_var is not None and len(op.body.ops) > 0
    body_local_ids: list = []
    if has_body:
        assert input_var is not None  # narrowed by has_body, made explicit for ty
        body_local_ids = _collect_body_local_value_ids(op.body.ops)
        skip_ids = {input_var.id}
        if sel_var is not None:
            skip_ids.add(sel_var.id)
        body_local_ids = [vid for vid in body_local_ids if vid not in skip_ids]

    # Output array: ``vec<dst_ty, 8> out_arr[n_frags];`` — one
    # contiguous register tile rather than n_slots independent scalars.
    n_frags = n_slots // _NAX_SUB_WIDTH
    out_arr = ctx.names.fresh("frag_cvt")
    out_comps = _emit_nax_frag_decl(ctx, out_arr, n_frags, dst_ty)

    for slot in range(n_slots):
        fi, si = divmod(slot, _NAX_SUB_WIDTH)
        if has_body:
            assert input_var is not None  # has_body implies it
            slot_elem = ctx.names.fresh("cvt_elem")
            ctx.emit(f"{src_ty} {slot_elem} = {src_comps[slot]};")
            ctx.names.bind(input_var, (slot_elem,), force=True)
            if sel_var is not None and slot_to_sel is not None:
                sel_name = sel_names[slot_to_sel[slot]]
                ctx.names.bind(sel_var, (sel_name,), force=True)
            for vid in body_local_ids:
                ctx.names._names.pop(vid, None)
            yielded_name = None
            for bop in op.body.ops:
                if isinstance(bop, YieldOp):
                    yielded_name = ctx.names.name_for(bop.operands[0])
                    break
                self._visit(bop, ctx)
            assert yielded_name is not None, (
                "FragConvertOp NAX: body did not produce a yielded value"
            )
            ctx.emit(f"{out_arr}[{fi}][{si}] = static_cast<{dst_ty}>({yielded_name});")
        else:
            ctx.emit(f"{out_arr}[{fi}][{si}] = static_cast<{dst_ty}>({src_comps[slot]});")

    ctx.names.bind(out, out_comps)
    # Output is also NAX-stored — propagate the discriminator.
    ctx.nax_frag_ids.add(out.id)
    _record_nax_frag_array(ctx, out, out_arr, n_frags)


def visit_frag_convert(self, op: FragConvertOp, ctx: _MslCtx) -> None:
    """Lower FragConvertOp for ACC f32 → A_FRAG bf16 on MSL.

    Dispatches on input fragment storage kind: NAX → `_visit_frag_convert_nax`,
    else the simdgroup_matrix conversion path below.

    ONLY supports ``(acc, f32) → (a_frag, bf16)`` today on the simdgroup
    path. Arbitrary register-tile conversions (acc↔b_frag, dtype promotion
    / demotion outside f32↔bf16, mixed layouts like acc→store-layout, etc.)
    are bigger scope: they need per-(src, dst) mapping tables (positional
    slot remapping plus cross-lane shuffles when positions diverge
    between Apple and the destination's lane convention) and have no
    caller in the codebase yet. Those paths raise
    ``NotImplementedError`` so we don't silently emit wrong code if a
    future caller uses them.

    Current implementation: allocates a
    ``simdgroup_matrix<bfloat, 8, 8>[mf*kf]`` output array. For each
    source tile, for each thread_elements slot, applies the optional
    body (with per-slot selector binding), converts f32 → bfloat,
    and writes to the destination tile's thread_elements at the SAME
    Apple-layout position — Apple's per-lane layout is dtype-agnostic,
    so no shuffle is needed.

    Registers the output in ``frag_values`` so downstream MmaOps consume
    the simdgroup_matrix array directly without routing through
    ``_pack_b32_to_frag_array``.
    """
    # FragConvertOp's source fragments are the first num_src_frags
    # operands (followed by selectors). Use operands[0] for the
    # storage-kind discriminator — all source frags must share the
    # same storage by construction.
    if op.operands and op.operands[0].id in ctx.nax_frag_ids:
        return _visit_frag_convert_nax(self, op, ctx)
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
    if not _msl_tiling_for(shape):
        raise NotImplementedError(f"FragConvertOp: MmaShape {shape_id!r} has no `msl` field")
    ctx.uses_simdgroup_matrix = True
    dst_frag_dtype_str, mf, nf, kf_shape = _parse_msl_tiling(_msl_tiling_for(shape))
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

    # Allocate destination simdgroup_matrix<bfloat, 8, 8>[mf*kf].
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
                # Cast f32 → bfloat and write.
                ctx.emit(f"{dst_e_ref}[{slot}] = ({dst_frag_dtype_str}){to_write};")

    ctx.names.bind(out, (dst_name,))
    ctx.frag_values[out.id] = (dst_name, dst_frag_dtype_str, mf, kf)


def _visit_frag_for_each_nax(self, op: FragForEachOp, ctx: _MslCtx) -> None:
    """Stub for FragForEachOp on NAX-stored fragments.

    NAX position bindings would compute (row, col) from the BaseNAXFrag
    coordinate ((fm, fn) per lane) plus the per-slot offsets:

        row = fm + (slot >> 2) * 8       // kElemRowsJump
        col = fn + (slot & 3)

    Then bind ``body_row_var`` / ``body_col_var`` per slot and walk
    the body. Land when an IR-emitted NAX kernel needs an epilogue
    primitive (atomic store, scatter store, etc.).
    """
    raise NotImplementedError(
        "FragForEachOp on NAX-stored fragments not yet emitted. "
        "Position computation: row = fm + (slot>>2)*8, col = fn + (slot&3)."
    )


def visit_frag_for_each(self, op: FragForEachOp, ctx: _MslCtx) -> None:
    """Lower FragForEachOp on MSL: iterate (tile, slot), read
    thread_elements(), bind body_input_var/row_var/col_var, walk body.

    Dispatches on input fragment storage kind: NAX → `_visit_frag_for_each_nax`,
    else the simdgroup_matrix path below.

    Position vars see the lane-dependent Apple positions — the body
    consumers (store_matrix, atomic_rmw, plain store) just read the
    row/col Values like any other IR U32.
    """
    if op.in_frag.id in ctx.nax_frag_ids:
        return _visit_frag_for_each_nax(self, op, ctx)
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


def _visit_frag_reduce_nax(self, op: FragReduceOp, ctx: _MslCtx) -> None:
    """Lower FragReduceOp for NAX per-lane storage.

    BaseNAXFrag layout per lane: 8 elements arranged as 2 rows × 4 cols
    (kElemRows=2, kElemCols=4). For ``c_regs > 8`` (TN > 1, fragments
    stacked horizontally to make a wider tile), each stacked fragment
    contributes to the same 2 row classes. The slot→row-class mapping
    for c_regs scalar components is::

        slot      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        row class 0  0  0  0  1  1  1  1  0  0  0  0  1  1  1  1
                     <-- frag 0 -->         <-- frag 1 -->

    Emission per row class:

        T thr = <init>;                     // -INF for max, 0 for sum
        for slot in slots-of-this-class:
            thr = Op(thr, src[slot]);       // within-thread fold
        T qgr = Op(thr, simd_shuffle_xor(thr, 1));   // quad ±1
        T sgr = Op(qgr, simd_shuffle_xor(qgr, 8));   // cross-quad ±8
        result[class] = Op(<old result>, sgr);

    The cross-lane reduction reaches all 32 simdgroup lanes via two
    XOR shuffles — see MLX's ``BaseNAXFrag::row_reduce`` reference.
    """
    kind = op.attrs["kind"]
    axis = op.attrs["axis"]
    cd_offsets = op.attrs["cd_offsets"]
    in_frag = op.in_frag

    if axis != "row":
        raise NotImplementedError(f"FragReduceOp NAX: axis={axis!r} not implemented (only 'row').")

    src_comps = ctx.names.components(in_frag)
    n_slots = len(src_comps)
    acc_ty = msl_type(in_frag.dtype)

    # Partition slot indices by row class. cd_offsets describes the
    # per-fragment slot pattern (8 entries with dr ∈ {0, 8}); for
    # stacked fragments (c_regs > 8), the same pattern repeats.
    n_classes = len({dr for dr, _ in cd_offsets})
    if n_classes != 2:
        raise NotImplementedError(
            f"FragReduceOp NAX: expected 2 row classes from cd_offsets, "
            f"got {n_classes}. Only kElemRows=2 BaseNAXFrag supported."
        )
    cd_per_frag = len(cd_offsets)
    if n_slots % cd_per_frag != 0:
        raise NotImplementedError(
            f"FragReduceOp NAX: n_slots={n_slots} not a multiple of "
            f"cd_offsets length {cd_per_frag}."
        )
    # Build slot→class for one fragment from cd_offsets, then repeat
    # across stacked fragments.
    dr_vals = sorted({dr for dr, _ in cd_offsets})
    per_frag_class = [dr_vals.index(dr) for dr, _ in cd_offsets]
    slot_class = [per_frag_class[s % cd_per_frag] for s in range(n_slots)]

    # Init per kind. For min, MLX uses metal::numeric_limits<T>::max();
    # we restrict to the two ops attention uses (max, add) for now.
    if kind == "max":
        op_fn = lambda a, b: f"metal::max({a}, {b})"
        init_val = "-INFINITY"
    elif kind == "add":
        op_fn = lambda a, b: f"({a} + {b})"
        init_val = "0"
    else:
        raise NotImplementedError(
            f"FragReduceOp NAX: kind={kind!r} not implemented (only max, add)."
        )

    for class_idx, result_val in enumerate(op.results):
        slots_for_class = [s for s in range(n_slots) if slot_class[s] == class_idx]
        # Within-thread fold.
        thr_var = ctx.names.fresh("red_thr")
        ctx.emit(f"{acc_ty} {thr_var} = {init_val};")
        for s in slots_for_class:
            ctx.emit(f"{thr_var} = {op_fn(thr_var, src_comps[s])};")
        # Quad shuffle (lanes ±1) — combines lane-pair within each quad.
        qgr_var = ctx.names.fresh("red_qgr")
        ctx.emit(f"{acc_ty} {qgr_var} = simd_shuffle_xor({thr_var}, ushort(1));")
        ctx.emit(f"{qgr_var} = {op_fn(thr_var, qgr_var)};")
        # Simdgroup shuffle (lanes ±8) — reaches the 16 lanes covering
        # the same row band; combined with the prior quad fold, each
        # lane now holds the full per-row reduction. Use sgr_var as the
        # final value: write directly to the IR result's allocated name
        # (skipping the trailing ``T res_name = sgr_var;`` write-through).
        sgr_var = ctx.names.fresh("red_sgr")
        ctx.emit(f"{acc_ty} {sgr_var} = simd_shuffle_xor({qgr_var}, ushort(8));")
        ctx.emit(f"{sgr_var} = {op_fn(qgr_var, sgr_var)};")
        # Bind the IR result Value to sgr_var instead of emitting a
        # final assignment — saves one register def per row class.
        ctx.names.bind(result_val, (sgr_var,), force=True)


def visit_frag_reduce(self, op: FragReduceOp, ctx: _MslCtx) -> None:
    """Lower FragReduceOp on an accumulator.

    Dispatches on input fragment storage kind: NAX → `_visit_frag_reduce_nax`,
    else the simdgroup_matrix path below.

    Simdgroup path: local fold of thread_elements() slots within the
    tile(s) for each class, then butterfly shuffle across the 4 lanes
    that share the row.

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
    if op.in_frag.id in ctx.nax_frag_ids:
        return _visit_frag_reduce_nax(self, op, ctx)
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
    """Lower LoadMatrixOp to simdgroup_load calls (or NAX per-lane
    scalar reads when the shape's payload starts with ``nax:``)."""
    tensor = op.attrs["src_tensor"]
    shape_id = op.attrs["shape_id"]
    which = op.attrs["which"]
    reg_offsets = op.attrs.get("reg_offsets")
    (out,) = op.results

    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(f"LoadMatrixOp: shape {shape_id!r} not in module.kernel_shapes")
    shape = module.kernel_shapes[shape_id]
    payload = _msl_tiling_for(shape)
    if not payload:
        raise NotImplementedError(f"LoadMatrixOp: MmaShape {shape_id!r} has no `msl` field")
    if payload.startswith("nax:"):
        return _visit_load_matrix_nax(self, op, ctx, shape, tensor, which)
    ctx.uses_simdgroup_matrix = True
    frag_dtype_str, mf, nf, kf = _parse_msl_tiling(payload)

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
    """Lower StoreMatrixOp to simdgroup_store calls (or NAX per-lane
    scalar writes when the shape's payload starts with ``nax:``)."""
    tensor = op.attrs["dst_tensor"]
    shape_id = op.attrs["shape_id"]
    reg_offsets = op.attrs.get("reg_offsets")
    frag_val = op.operands[0]
    row_val, col_val = op.operands[1], op.operands[2]

    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(f"StoreMatrixOp: shape {shape_id!r} not in module.kernel_shapes")
    shape = module.kernel_shapes[shape_id]
    payload = _msl_tiling_for(shape)
    if not payload:
        raise NotImplementedError(f"StoreMatrixOp: MmaShape {shape_id!r} has no `msl` field")
    if payload.startswith("nax:"):
        return _visit_store_matrix_nax(self, op, ctx, shape, tensor, frag_val, row_val, col_val)
    ctx.uses_simdgroup_matrix = True
    _, mf, nf, _ = _parse_msl_tiling(payload)
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


def visit_store_matrix_gate_residual(self, op, ctx: _MslCtx) -> None:
    """Dispatch StoreMatrixGateResidualOp. NAX-only — non-NAX shapes
    fall back to the unfused ``StoreMatrixOp`` + standalone
    ``AdaGateResidualKernel`` chain via ``GemmKernel.is_valid``."""
    shape_id = op.attrs["shape_id"]
    frag_val = op.operands[0]
    row_val, col_val = op.operands[1], op.operands[2]

    module = ctx.module
    if module is None or shape_id not in module.kernel_shapes:
        raise RuntimeError(
            f"StoreMatrixGateResidualOp: shape {shape_id!r} not in module.kernel_shapes"
        )
    shape = module.kernel_shapes[shape_id]
    payload = _msl_tiling_for(shape)
    if not payload or not payload.startswith("nax:"):
        raise NotImplementedError(
            f"StoreMatrixGateResidualOp: shape {shape_id!r} payload {payload!r} "
            "is not NAX. The fused gate-residual store is wired only for the "
            "NAX m16n32k16_nax_bf16 path; non-NAX shapes go through the unfused "
            "GemmKernel.build() epilogue (`store_acc(..., gate=, residual=)`)."
        )
    return _visit_store_matrix_gate_residual_nax(self, op, ctx, shape, frag_val, row_val, col_val)
