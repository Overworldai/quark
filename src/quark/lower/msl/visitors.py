"""Per-op MSL visitor methods and the dispatch table.

EXEMPT FROM 500-LINE RULE — this file is the single dispatch table for
40 IR ops. Each visitor is a thin 5-15 line function; splitting further
would scatter the 1:1 op→MSL mapping across multiple files with no
readability benefit. Already split from lower.py to stay under 800.
"""

from __future__ import annotations

from typing import Any

from quark.ir import (
    ArithOp,
    AsyncCopyCommitOp,
    AsyncCopyOp,
    AsyncCopyWaitOp,
    AtomicRmwOp,
    BarrierOp,
    BitcastOp,
    BlockDimOp,
    BlockIdxOp,
    CmpOp,
    ConstOp,
    ConvertOp,
    DType,
    ForLoopOp,
    FragApplyOp,
    FragConvertOp,
    FragForEachOp,
    FragReduceOp,
    FragSliceOp,
    GridDimOp,
    GroupIdOp,
    IfRegionOp,
    LaneIdOp,
    LoadMatrixOp,
    LoadOp,
    MathOp,
    MergeB32Op,
    MmaOp,
    Op,
    PackedConvertOp,
    SelectOp,
    ShuffleOp,
    SmemAllocOp,
    SplitB32Op,
    StoreMatrixGateResidualOp,
    StoreMatrixOp,
    StoreOp,
    SubgroupBroadcastOp,
    SubgroupIdOp,
    SubgroupReduceOp,
    ThreadIdInGroupOp,
    ThreadIdxOp,
    VecBuildOp,
    VecExtractOp,
    VecLoadOp,
    VecStoreOp,
    WhileLoopOp,
    YieldOp,
)

from .lower import (
    _ARITH_OP,
    _ATOMIC_FN,
    _CMP_OP,
    _MATH_FN,
    _REDUCE_FN,
    _SHUFFLE_FN,
    _align_up,
    _compute_tensor_offset,
    _format_literal,
    _MslCtx,
    _tensor_buf_name,
)
from .mma import (
    visit_frag_apply as _visit_frag_apply,
)
from .mma import (
    visit_frag_convert as _visit_frag_convert,
)
from .mma import (
    visit_frag_for_each as _visit_frag_for_each,
)
from .mma import (
    visit_frag_reduce as _visit_frag_reduce,
)
from .mma import (
    visit_load_matrix as _visit_load_matrix,
)
from .mma import (
    visit_mma as _visit_mma,
)
from .mma import (
    visit_store_matrix as _visit_store_matrix,
)
from .mma import (
    visit_store_matrix_gate_residual as _visit_store_matrix_gate_residual,
)
from .types import msl_type

# ---------------------------------------------------------------------------
# Arithmetic / math / compare / select / convert / bitcast
# ---------------------------------------------------------------------------


def _visit_const(self, op: ConstOp, ctx: _MslCtx) -> None:
    (out,) = op.results
    # Skip if already hoisted (the pre-pass emits all consts at function scope).
    if ctx.names.has(out):
        return
    name = ctx.names.name_for(out)
    dtype: DType = op.attrs["dtype"]
    value = op.attrs["value"]
    ty = msl_type(dtype)
    ctx.emit(f"{ty} {name} = {_format_literal(dtype, value)};")


def _emit_pointwise(
    ctx: _MslCtx,
    out,
    operand_values,
    template: str,
    *,
    declare_ty: str | None = None,
) -> None:
    """Emit one pointwise C statement per output component.

    For width-1 ``out`` this is a single scalar emit. For width-N
    vec values, walks the parallel ``components()`` of out + each
    operand and emits N statements, one per element. ``template`` is
    a Python format string with ``{dst}`` and ``{a0}``, ``{a1}``,
    ... placeholders for the per-element operand names.

    This is what the IR's ``vec * vec`` (and other pointwise arith /
    math) ops *must* produce on MSL — ``ctx.names.name_for(value)``
    returns only the *first* component for width-N values (visitors
    are expected to iterate ``components()`` themselves), so the
    naive ``ty {dst} = {a} OP {b};`` emit silently miscompiles for
    vec inputs (declares only ``_pc_x_0``, then a downstream consumer
    references ``_pc_x_1..N`` and the kernel fails to compile).
    """
    ty = declare_ty if declare_ty is not None else msl_type(out.dtype)
    width = out.width
    if width == 1:
        dst = ctx.names.name_for(out)
        operand_names = [ctx.names.name_for(v) for v in operand_values]
        kw = {f"a{i}": n for i, n in enumerate(operand_names)}
        ctx.emit(f"{ty} {dst} = {template.format(dst=dst, **kw)};")
        return
    out_comps = ctx.names.components(out)
    operand_comps = []
    for v in operand_values:
        comps = ctx.names.components(v) if v.width > 1 else (ctx.names.name_for(v),) * width
        if len(comps) != width:
            # Mismatched widths can't be element-wise broadcast here;
            # the IR layer should have inserted a broadcast / vec_build
            # to widen the smaller operand.
            raise NotImplementedError(
                f"_emit_pointwise: operand width {len(comps)} != out width "
                f"{width}; insert a vec_build to broadcast first."
            )
        operand_comps.append(comps)
    for i in range(width):
        kw = {f"a{j}": operand_comps[j][i] for j in range(len(operand_values))}
        ctx.emit(f"{ty} {out_comps[i]} = {template.format(dst=out_comps[i], **kw)};")


def _visit_arith(self, op: ArithOp, ctx: _MslCtx) -> None:
    kind = op.attrs["kind"]
    (out,) = op.results
    dst = ctx.names.name_for(out)
    ty = msl_type(out.dtype)

    if kind == "neg":
        _emit_pointwise(ctx, out, op.operands[:1], "-{a0}")
    elif kind == "abs":
        _emit_pointwise(ctx, out, op.operands[:1], "metal::abs({a0})")
    elif kind == "fma":
        _emit_pointwise(ctx, out, op.operands[:3], "metal::fma({a0}, {a1}, {a2})")
    elif kind == "fma_bf16x2":
        # PTX ``fma.rn.bf16x2`` operates on B32 regs holding two packed
        # bf16 values. MSL has no packed-pair fma. On MSL, bfloat is a
        # real 16-bit type and VecLoad→VecExtract aliasing means the
        # "B32" operands often resolve to individual bfloat scalars.
        a = ctx.names.name_for(op.operands[0])
        b = ctx.names.name_for(op.operands[1])
        c = ctx.names.name_for(op.operands[2])
        if _is_msl_16bit(a):
            # Operands are individual bfloat scalars; one scalar fma.
            ctx.emit(f"bfloat {dst} = metal::fma({a}, {b}, {c});")
        else:
            # Operands are real B32 (uint, packed bf16x2): unpack, fma×2, repack.
            ap = ctx.names.fresh("fma2a")
            bp = ctx.names.fresh("fma2b")
            cp = ctx.names.fresh("fma2c")
            ctx.emit(f"ushort2 {ap} = as_type<ushort2>({a});")
            ctx.emit(f"ushort2 {bp} = as_type<ushort2>({b});")
            ctx.emit(f"ushort2 {cp} = as_type<ushort2>({c});")
            rlo = ctx.names.fresh("fma2lo")
            rhi = ctx.names.fresh("fma2hi")
            ctx.emit(
                f"bfloat {rlo} = metal::fma(as_type<bfloat>({ap}.x), "
                f"as_type<bfloat>({bp}.x), as_type<bfloat>({cp}.x));"
            )
            ctx.emit(
                f"bfloat {rhi} = metal::fma(as_type<bfloat>({ap}.y), "
                f"as_type<bfloat>({bp}.y), as_type<bfloat>({cp}.y));"
            )
            ctx.emit(
                f"{ty} {dst} = as_type<uint>(ushort2("
                f"as_type<ushort>({rlo}), as_type<ushort>({rhi})));"
            )
    elif kind == "min":
        _emit_pointwise(ctx, out, op.operands[:2], "metal::min({a0}, {a1})")
    elif kind == "max":
        _emit_pointwise(ctx, out, op.operands[:2], "metal::max({a0}, {a1})")
    elif kind == "mul_hi":
        # MSL's metal::mulhi computes the high half of an unsigned
        # 32×32→64 multiply (or the matching width for u16/u64 ops).
        _emit_pointwise(ctx, out, op.operands[:2], "metal::mulhi({a0}, {a1})")
    else:
        sym = _ARITH_OP.get(kind)
        if sym is None:
            raise NotImplementedError(f"ArithOp kind {kind!r} not implemented for MSL")
        _emit_pointwise(ctx, out, op.operands[:2], "{a0} " + sym + " {a1}")


def _visit_math(self, op: MathOp, ctx: _MslCtx) -> None:
    kind = op.attrs["kind"]
    (out,) = op.results
    ty = msl_type(out.dtype)

    if kind in ("rcp", "rcp_approx"):
        _emit_pointwise(
            ctx,
            out,
            op.operands[:1],
            f"static_cast<{ty}>(1.0f) / {{a0}}",
        )
    else:
        fn = _MATH_FN.get(kind)
        if fn is None:
            raise NotImplementedError(f"MathOp kind {kind!r} not implemented for MSL")
        _emit_pointwise(ctx, out, op.operands[:1], f"{fn}({{a0}})")


def _visit_cmp(self, op: CmpOp, ctx: _MslCtx) -> None:
    kind = op.attrs["kind"]
    (out,) = op.results
    dst = ctx.names.name_for(out)
    a = ctx.names.name_for(op.operands[0])
    b = ctx.names.name_for(op.operands[1])
    sym = _CMP_OP[kind]
    ctx.emit(f"bool {dst} = {a} {sym} {b};")


def _visit_select(self, op: SelectOp, ctx: _MslCtx) -> None:
    (out,) = op.results
    dst = ctx.names.name_for(out)
    pred = ctx.names.name_for(op.operands[0])
    t = ctx.names.name_for(op.operands[1])
    f = ctx.names.name_for(op.operands[2])
    ty = msl_type(out.dtype)
    ctx.emit(f"{ty} {dst} = {pred} ? {t} : {f};")


def _visit_convert(self, op: ConvertOp, ctx: _MslCtx) -> None:
    (out,) = op.results
    dst = ctx.names.name_for(out)
    src = ctx.names.name_for(op.operands[0])
    dst_dtype: DType = op.attrs["dst_dtype"]
    ty = msl_type(dst_dtype)
    ctx.emit(f"{ty} {dst} = static_cast<{ty}>({src});")


def _visit_packed_convert(self, op: PackedConvertOp, ctx: _MslCtx) -> None:
    (out,) = op.results
    dst_dtype: DType = op.attrs["dst_dtype"]
    lo_v, hi_v = op.operands
    lo = ctx.names.name_for(lo_v)
    hi = ctx.names.name_for(hi_v)
    ty = msl_type(dst_dtype)
    lo_name = ctx.names.fresh("pc")
    hi_name = ctx.names.fresh("pc")
    ctx.emit(f"{ty} {lo_name} = static_cast<{ty}>({lo});")
    ctx.emit(f"{ty} {hi_name} = static_cast<{ty}>({hi});")
    dst = ctx.names.name_for(out)
    ctx.emit(
        f"ushort {dst} = "
        f"(as_type<uchar>({lo_name})) | "
        f"(static_cast<ushort>(as_type<uchar>({hi_name})) << 8);"
    )


def _visit_bitcast(self, op: BitcastOp, ctx: _MslCtx) -> None:
    (out,) = op.results
    dst = ctx.names.name_for(out)
    src = ctx.names.name_for(op.operands[0])
    ty = msl_type(out.dtype)
    ctx.emit(f"{ty} {dst} = as_type<{ty}>({src});")


# ---------------------------------------------------------------------------
# Vector / bit manipulation
# ---------------------------------------------------------------------------


def _visit_vec_build(self, op: VecBuildOp, ctx: _MslCtx) -> None:
    (out,) = op.results
    names: list[str] = []
    for operand in op.operands:
        (single,) = ctx.names.components(operand)
        names.append(single)
    ctx.names.bind(out, tuple(names))


def _visit_vec_extract(self, op: VecExtractOp, ctx: _MslCtx) -> None:
    (out,) = op.results
    src = op.operands[0]
    idx = op.attrs["index"]

    # VecExtractOp on a fragment Value is no longer supported — every
    # former caller has been migrated to a Frag* primitive that stays
    # in-register. Raise loudly if a new caller slips through.
    if src.id in ctx.frag_values:
        raise NotImplementedError(
            "VecExtractOp on a fragment Value is no longer supported on "
            "MSL — the smem round-trip path was retired. Use FragApplyOp "
            "(.map / .map_per_row_class) for element-wise transforms, "
            "FragReduceOp (.reduce_along_cols) for cross-lane reductions, "
            "FragForEachOp (.for_each) for side-effect epilogues, or "
            "FragConvertOp (.convert) for layout / dtype changes."
        )

    ctx.names.alias_component(out, src, idx)


def _is_msl_16bit(name: str) -> bool:
    """Detect if an MSL variable name was allocated for a 16-bit type.

    On MSL, VecLoad→VecExtract aliasing can assign BF16/F16 names to
    Values that the IR types as B32 (because PTX's "bf16 lives in b32"
    convention doesn't hold). Downstream ops that assume B32=32-bit
    need this check to avoid ``as_type`` size mismatches.
    """
    return "_pc_bf_" in name or "_pc_f16_" in name or "_pc_b16_" in name


def _visit_split_b32(self, op: SplitB32Op, ctx: _MslCtx) -> None:
    lo, hi = op.results
    src = op.operands[0]
    src_name = ctx.names.name_for(src)
    lo_name = ctx.names.name_for(lo)
    hi_name = ctx.names.name_for(hi)
    # On MSL, a B32 value is a ``uint`` (32-bit) and the split is a
    # proper 4B → 2×2B reinterpret. But VecLoad→VecExtract aliasing
    # can map B32 IR values to 16-bit MSL names (bfloat, half). In that
    # case the value IS already 16-bit — treat lo as the value, hi as 0.
    if _is_msl_16bit(src_name):
        ctx.emit(f"ushort {lo_name} = as_type<ushort>({src_name});")
        ctx.emit(f"ushort {hi_name} = 0u;")
    else:
        ctx.emit(f"ushort {lo_name} = as_type<ushort2>({src_name}).x;")
        ctx.emit(f"ushort {hi_name} = as_type<ushort2>({src_name}).y;")


def _visit_merge_b32(self, op: MergeB32Op, ctx: _MslCtx) -> None:
    (out,) = op.results
    lo, hi = op.operands
    dst = ctx.names.name_for(out)
    ctx.emit(
        f"uint {dst} = as_type<uint>(ushort2({ctx.names.name_for(lo)}, {ctx.names.name_for(hi)}));"
    )


# ---------------------------------------------------------------------------
# Thread / block / lane indexing
# ---------------------------------------------------------------------------


def _visit_thread_idx(self, op: Op, ctx: _MslCtx) -> None:
    dim = op.attrs["dim"]
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = thread_position_in_threadgroup.{dim};")


def _visit_block_idx(self, op: Op, ctx: _MslCtx) -> None:
    dim = op.attrs["dim"]
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = threadgroup_position_in_grid.{dim};")


def _visit_block_dim(self, op: Op, ctx: _MslCtx) -> None:
    dim = op.attrs["dim"]
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = threads_per_threadgroup.{dim};")


def _visit_grid_dim(self, op: Op, ctx: _MslCtx) -> None:
    dim = op.attrs["dim"]
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = threadgroups_per_grid.{dim};")


def _visit_lane_id(self, op: Op, ctx: _MslCtx) -> None:
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = thread_index_in_simdgroup;")


def _visit_subgroup_id(self, op: Op, ctx: _MslCtx) -> None:
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = simdgroup_index_in_threadgroup;")


def _visit_group_id(self, op: Op, ctx: _MslCtx) -> None:
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = (thread_index_in_simdgroup >> 2);")


def _visit_thread_id_in_group(self, op: Op, ctx: _MslCtx) -> None:
    dst = ctx.names.name_for(op.results[0])
    ctx.emit(f"uint {dst} = (thread_index_in_simdgroup & 3);")


# ---------------------------------------------------------------------------
# Barriers
# ---------------------------------------------------------------------------


def _visit_barrier(self, op: BarrierOp, ctx: _MslCtx) -> None:
    scope = op.attrs.get("scope", "block")
    if scope == "block":
        ctx.emit("threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
    elif scope == "subgroup":
        ctx.emit("simdgroup_barrier(metal::mem_flags::mem_none);")
    elif scope == "system":
        ctx.emit("threadgroup_barrier(metal::mem_flags::mem_device);")
    else:
        raise NotImplementedError(f"BarrierOp scope {scope!r}")


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


def _is_accumulator_value(value) -> bool:
    """Heuristic: even-width f32 Values in carried positions are MMA
    accumulators. Each 8x8 simdgroup_matrix<f32> tile holds 2 f32 per
    lane on Apple silicon, so n_frags = width // 2. m16n8k16 → width=4
    → 2 tiles; m8n8k8 → width=2 → 1 tile. Higher even widths (future
    MMA shapes with more c_regs) map the same way."""
    from quark.ir import DType

    return value.dtype is DType.F32 and value.width >= 2 and value.width % 2 == 0


def _visit_for_loop(self, op: ForLoopOp, ctx: _MslCtx) -> None:
    iv = op.induction_var
    assert iv is not None
    iv_name = ctx.names.name_for(iv)
    lo = ctx.names.name_for(op.lo)
    hi = ctx.names.name_for(op.hi)
    step = ctx.names.name_for(op.step)
    ty = msl_type(iv.dtype)

    for cin, res, cbv in zip(op.carried_in, op.results, op.carried_body_vars, strict=False):
        # Multi-component vec carry without simdgroup_matrix backing
        # (e.g. NAX width-16 F32 accumulator). Emit a single ARRAY
        # local ``float _accN[width]`` rather than N individual scalar
        # locals. This gives Apple's shader compiler one contiguous
        # register allocation to track instead of N independent live
        # ranges — crucial for NAX GEMMs where 32+ float accumulators
        # can exceed the simdgroup register budget if declared as
        # separate scalars (see METAL_BACKEND.md analysis).
        #
        # The array persists across loop iterations (declared before
        # the ``for``). The yield handler writes ``_accN[i] = dN;``
        # which is a cheap array-element store.
        if (
            res.width > 1
            and not (_is_accumulator_value(res) and ctx.uses_simdgroup_matrix)
            and not ctx.uses_simdgroup_matrix  # CUDA simdgroup_matrix flag must be off
        ):
            cin_comps = ctx.names.components(cin)
            res_ty = msl_type(res.dtype)

            # NAX-storage carry: width-16 f32 (or any source already in
            # nax_frag_ids) on a NAX-using kernel uses ``vec<float, 8>
            # X[2]`` array storage instead of N scalar locals. Apple's
            # compiler treats the array as 2 contiguous SIMD8 register
            # groups — same shape as the hand-written kernel's
            # ``vec<float, 8> O_frags[N]`` declarations. Cuts SSA
            # fragmentation; matches HW's register-tile usage.
            is_nax_carry = (
                ctx.uses_nax and res.dtype is DType.F32 and res.width % 8 == 0 and res.width >= 8
            ) or cin.id in ctx.nax_frag_ids
            n_subfrags = res.width // 8 if is_nax_carry else 0

            # Carry-alias optimization: when ``cin`` is itself a loop-
            # carry body var (its producer is another ForLoopOp), its
            # components are mutable lane locals declared at the outer
            # loop's scope. The inner loop body can update them in
            # place — saves N register copies on entry (``chunk_carry =
            # seg_carry``) and another N at the outer yield (``seg_carry
            # = chunk_carry`` becomes a self-assign that the yield
            # visitor skips).
            cin_producer = getattr(cin, "producer", None)
            can_alias_carry = cin_producer is not None and isinstance(cin_producer, ForLoopOp)
            if can_alias_carry:
                # Reuse cin's storage; cin_comps may be array-element
                # strings (NAX) or scalars (other) — either way,
                # binding directly preserves the layout.
                ctx.names.bind(res, cin_comps, force=True)
                ctx.names.alias(cbv, res)
                # Propagate the cin's array-name registration to the
                # alias so MMA helpers can pass-by-reference.
                if cin.id in ctx.nax_frag_arrays:
                    ctx.nax_frag_arrays[res.id] = ctx.nax_frag_arrays[cin.id]
                    ctx.nax_frag_arrays[cbv.id] = ctx.nax_frag_arrays[cin.id]
            elif is_nax_carry:
                # Allocate a fresh ``vec<float, 8> X[n];`` and seed it
                # from cin (which may itself be array-form or scalar).
                arr = ctx.names.fresh(f"{iv_name}_carry")
                ctx.emit(f"vec<{res_ty}, 8> {arr}[{n_subfrags}];")
                res_comps = tuple(
                    f"{arr}[{fi}][{si}]" for fi in range(n_subfrags) for si in range(8)
                )
                ctx.names.bind(res, res_comps, force=True)
                ctx.names.alias(cbv, res)
                for r_name, c_name in zip(res_comps, cin_comps, strict=True):
                    ctx.emit(f"{r_name} = {c_name};")
                # Record the array name so MMA helper-emission can pass
                # the carry by reference into ``nax_mma_*`` helpers.
                ctx.nax_frag_arrays[res.id] = (arr, n_subfrags)
                ctx.nax_frag_arrays[cbv.id] = (arr, n_subfrags)
            else:
                res_comps = tuple(ctx.names.fresh(f"{iv_name}_carry") for _ in range(res.width))
                ctx.names.bind(res, res_comps, force=True)
                ctx.names.alias(cbv, res)
                for r_name, c_name in zip(res_comps, cin_comps, strict=True):
                    ctx.emit(f"{res_ty} {r_name} = {c_name};")
            # Tag the result + body var as NAX-stored when the source
            # was a NAX fragment (or by construction width-16 f32 on a
            # NAX kernel) so downstream Frag* visitors dispatch right.
            if is_nax_carry:
                ctx.nax_frag_ids.add(res.id)
                ctx.nax_frag_ids.add(cbv.id)
            continue
        if _is_accumulator_value(res) and ctx.uses_simdgroup_matrix:
            # Pre-declare as simdgroup_matrix array for MMA accumulators.
            # Each 8x8 simdgroup_matrix tile stores 2 f32 per lane (Apple
            # layout), so n_frags = res.width // 2. The mf / nf split
            # defaults to mf=n_frags, nf=1 (m-stacked) which matches both
            # m16n8 shapes (c_regs=4 → mf=2, nf=1) and the Metal-native
            # m8n8k8 shape (c_regs=2 → mf=1, nf=1). MMA visitor agrees.
            n_frags = max(res.width // 2, 1)
            mf, nf = n_frags, 1
            res_name = ctx.names.fresh("frag")
            ctx.names.bind(res, (res_name,), force=True)
            ctx.emit(f"simdgroup_matrix<float, 8, 8> {res_name}[{n_frags}];")
            # Seed the new array from the carried-in state, not zero:
            # nested for_loops (e.g. owl_attn's runtime seg→kv_chunk
            # stack) carry accumulators from the outer loop's current
            # value, and zero-initialising here would throw that state
            # away at every inner entry. When the cin is itself already
            # a registered fragment array, element-copy it in; otherwise
            # fall back to zero-init (top-level K-loop entering for the
            # first time).
            if cin.id in ctx.frag_values:
                src_name = ctx.frag_values[cin.id][0]
                for fi in range(n_frags):
                    ctx.emit(f"{res_name}[{fi}] = {src_name}[{fi}];")
            else:
                for fi in range(n_frags):
                    ctx.emit(f"{res_name}[{fi}] = simdgroup_matrix<float, 8, 8>(0);")
            ctx.frag_values[res.id] = (res_name, "float", mf, nf)
            ctx.names.alias(cbv, res)
            # Propagate frag info to the body var so MMA finds it.
            ctx.frag_values[cbv.id] = (res_name, "float", mf, nf)
        else:
            res_name = ctx.names.name_for(res)
            res_ty = msl_type(res.dtype)
            ctx.emit(f"{res_ty} {res_name} = {ctx.names.name_for(cin)};")
            ctx.names.alias(cbv, res)

    ctx.emit(f"for ({ty} {iv_name} = {lo}; {iv_name} < {hi}; {iv_name} += {step}) {{")
    ctx.indent += 1
    ctx.op_stack.append(op)
    self._walk_region(op.body.ops, ctx)
    ctx.op_stack.pop()
    ctx.indent -= 1
    ctx.emit("}")


def _visit_if_region(self, op: IfRegionOp, ctx: _MslCtx) -> None:
    n_carried = op.attrs.get("n_carried", 0)
    pred_name = ctx.names.name_for(op.pred)
    carried_in = op.operands[1 : 1 + n_carried]

    for i, (res, tbv, ebv) in enumerate(
        zip(op.results, op.then_body_vars, op.else_body_vars, strict=False)
    ):
        res_name = ctx.names.name_for(res)
        res_ty = msl_type(res.dtype)
        ctx.emit(f"{res_ty} {res_name} = {ctx.names.name_for(carried_in[i])};")
        ctx.names.alias(tbv, res)
        ctx.names.alias(ebv, res)

    ctx.emit(f"if ({pred_name}) {{")
    ctx.indent += 1
    ctx.op_stack.append(op)
    self._walk_region(op.then_region.ops, ctx)
    ctx.indent -= 1
    ctx.emit("} else {")
    ctx.indent += 1
    self._walk_region(op.else_region.ops, ctx)
    ctx.op_stack.pop()
    ctx.indent -= 1
    ctx.emit("}")


def _visit_while_loop(self, op: WhileLoopOp, ctx: _MslCtx) -> None:
    ctx.emit("while (true) {")
    ctx.indent += 1
    ctx.op_stack.append(op)
    _cond_region, body_region = op.regions
    self._walk_region(_cond_region.ops, ctx)
    self._walk_region(body_region.ops, ctx)
    ctx.op_stack.pop()
    ctx.indent -= 1
    ctx.emit("}")


def _visit_yield(self, op: YieldOp, ctx: _MslCtx) -> None:
    if not ctx.op_stack:
        return
    parent = ctx.op_stack[-1]
    if isinstance(parent, (ForLoopOp, IfRegionOp)):
        for res, yielded in zip(parent.results, op.operands, strict=False):
            val_name = ctx.names.name_for(yielded)

            if yielded.id in ctx.frag_values:
                _, acc_ty, mf, nf = ctx.frag_values[yielded.id]
                n_frags = mf * nf

                if res.id not in ctx.frag_values:
                    # First yield of a fragment into a non-fragment result.
                    # The result was pre-declared as a scalar — emit a new
                    # simdgroup_matrix decl (the scalar decl becomes dead).
                    res_name = ctx.names.fresh("frag")
                    ctx.names.bind(res, (res_name,), force=True)
                    # Don't emit the decl here — it's inside the loop.
                    # Instead, mark as pending; _visit_for_loop will hoist.
                    ctx.frag_values[res.id] = (res_name, acc_ty, mf, nf)

                res_name = ctx.names.name_for(res)
                if res_name != val_name:
                    for i in range(n_frags):
                        ctx.emit(f"{res_name}[{i}] = {val_name}[{i}];")
            else:
                # Multi-component vec yield (NAX accumulator and similar):
                # update each carried component slot from the yielded
                # value's per-component names. Skips no-op self-assigns.
                res_comps = ctx.names.components(res)
                yielded_comps = ctx.names.components(yielded)
                if len(res_comps) > 1 and len(yielded_comps) == len(res_comps):
                    for r_name, y_name in zip(res_comps, yielded_comps, strict=True):
                        if r_name != y_name:
                            ctx.emit(f"{r_name} = {y_name};")
                    continue
                res_name = ctx.names.name_for(res)
                if res_name != val_name:
                    ctx.emit(f"{res_name} = {val_name};")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _visit_smem_alloc(self, op: SmemAllocOp, ctx: _MslCtx) -> None:
    if op.results[0].id not in ctx.smem_allocs:
        dtype: DType = op.attrs["dtype"]
        shape: tuple[int, ...] = tuple(op.attrs["shape"])
        pad: int = int(op.attrs.get("pad", 0))
        (backing,) = op.results
        if len(shape) == 2:
            elems = shape[0] * (shape[1] + pad)
        else:
            elems = 1
            for s in shape:
                elems *= s
        size_bytes = elems * dtype.bytes
        offset = _align_up(ctx.smem_bytes, 16)
        ctx.smem_bytes = offset + size_bytes
        var_name = f"smem_{backing.id}"
        ctx.emit(f"threadgroup {msl_type(dtype)} {var_name}[{elems}];")
        ctx.smem_allocs[backing.id] = (var_name, offset, size_bytes)
        ctx.names.bind(backing, (var_name,))


def _atomic_access_expr(buf: str, offset_expr: str, ty: str) -> str:
    """``&buf[offset]`` cast to ``device atomic_<ty>*`` for atomic API calls.

    Mirrors the cast used in ``_visit_atomic_rmw`` so plain load/store
    on a buffer that some other op targets via ``AtomicRmwOp`` (e.g.
    moe_router_correct: ``counts`` is initialised by ``qk.store`` and
    later atomic-added) reads/writes through the same atomic typed
    pointer the harness declares.
    """
    return f"reinterpret_cast<device atomic_{ty}*>(&{buf}[{offset_expr}])"


def _visit_load(self, op: LoadOp, ctx: _MslCtx) -> None:
    tensor = op.attrs["tensor"]
    (out,) = op.results
    dst = ctx.names.name_for(out)
    ty = msl_type(out.dtype)
    pred = op.attrs.get("pred")
    indices = op.operands[:-1] if pred is not None else op.operands
    offset_expr = _compute_tensor_offset(tensor, indices, ctx)
    buf = _tensor_buf_name(tensor, ctx)
    is_atomic = buf in ctx.atomic_output_names

    if is_atomic:
        load_expr = (
            f"atomic_load_explicit({_atomic_access_expr(buf, offset_expr, ty)},"
            f" memory_order_relaxed)"
        )
    else:
        load_expr = f"{buf}[{offset_expr}]"

    if pred is not None:
        pred_name = ctx.names.name_for(pred)
        ctx.emit(f"{ty} {dst};")
        ctx.emit(f"if ({pred_name}) {{ {dst} = {load_expr}; }}")
        ctx.emit(f"else {{ {dst} = static_cast<{ty}>(0); }}")
    else:
        ctx.emit(f"{ty} {dst} = {load_expr};")


def _visit_store(self, op: StoreOp, ctx: _MslCtx) -> None:
    tensor = op.attrs["tensor"]
    value = op.operands[0]
    indices = op.operands[1:]
    pred = op.attrs.get("pred")
    if pred is not None:
        indices = indices[:-1]
    offset_expr = _compute_tensor_offset(tensor, indices, ctx)
    buf = _tensor_buf_name(tensor, ctx)
    val_name = ctx.names.name_for(value)
    is_atomic = buf in ctx.atomic_output_names

    if is_atomic:
        ty = msl_type(value.dtype)
        store_stmt = (
            f"atomic_store_explicit({_atomic_access_expr(buf, offset_expr, ty)},"
            f" {val_name}, memory_order_relaxed)"
        )
    else:
        store_stmt = f"{buf}[{offset_expr}] = {val_name}"

    if pred is not None:
        pred_name = ctx.names.name_for(pred)
        ctx.emit(f"if ({pred_name}) {{ {store_stmt}; }}")
    else:
        ctx.emit(f"{store_stmt};")


_VEC_PACK_WIDTHS = (2, 3, 4)  # MSL vector widths supported on every dtype


def _msl_addr_space(tensor) -> str:
    """Address-space qualifier for the MSL pointer cast.

    GlobalTensor → ``device``; SharedRegion → ``threadgroup``. Other
    tensor kinds aren't expected as vec_load/store targets — they fall
    through to scalar lowering via the caller's None return.
    """
    from quark.ir import GlobalTensor, SharedRegion

    if isinstance(tensor, GlobalTensor):
        return "device"
    if isinstance(tensor, SharedRegion):
        return "threadgroup"
    return ""


def _packed_vec_chunks(n: int) -> list[int] | None:
    """Decompose ``n`` components into a sequence of MSL vector widths.

    Returns the list of chunk widths covering ``n`` (e.g. 8 → [4,4],
    16 → [4,4,4,4], 4 → [4], 2 → [2], 6 → [4,2]). Returns ``None`` for
    widths that don't decompose cleanly into 2/3/4-vectors (1, 5, 7,
    11, ...) so the caller falls back to scalar emission.
    """
    if n < 2:
        return None
    chunks: list[int] = []
    remaining = n
    for w in (4, 3, 2):
        while remaining >= w:
            chunks.append(w)
            remaining -= w
        if remaining == 0:
            return chunks
    if remaining != 0:
        return None
    return chunks if all(c in _VEC_PACK_WIDTHS for c in chunks) else None


def _emit_packed_vec_load(buf: str, ty: str, addr_space: str, offset_expr: str, comps, ctx) -> bool:
    """Emit packed ``T{2,3,4}`` reinterpret_cast loads when the layout
    permits it. Returns True on success; False if the width can't be
    packed (caller falls back to scalar). Pred-guarded loads always
    return False — the per-element if/else is hard to vectorize.
    """
    if not addr_space:
        return False
    chunks = _packed_vec_chunks(len(comps))
    if chunks is None:
        return False
    cursor = 0
    for w in chunks:
        bi = f"{offset_expr} + {cursor}u" if cursor > 0 else offset_expr
        tmp = ctx.names.fresh("v")
        ctx.emit(f"{ty}{w} {tmp} = *reinterpret_cast<const {addr_space} {ty}{w}*>(&{buf}[{bi}]);")
        for j in range(w):
            ctx.emit(f"{ty} {comps[cursor + j]} = {tmp}[{j}];")
        cursor += w
    return True


def _emit_packed_vec_store(
    buf: str, ty: str, addr_space: str, offset_expr: str, comps, ctx
) -> bool:
    """Mirror of ``_emit_packed_vec_load`` for stores."""
    if not addr_space:
        return False
    chunks = _packed_vec_chunks(len(comps))
    if chunks is None:
        return False
    cursor = 0
    for w in chunks:
        bi = f"{offset_expr} + {cursor}u" if cursor > 0 else offset_expr
        elems = ", ".join(comps[cursor + j] for j in range(w))
        ctx.emit(f"*reinterpret_cast<{addr_space} {ty}{w}*>(&{buf}[{bi}]) = {ty}{w}({elems});")
        cursor += w
    return True


def _visit_vec_load(self, op: VecLoadOp, ctx: _MslCtx) -> None:
    """Lower VecLoadOp to N consecutive scalar reads.

    When the result dtype is *wider* than the buffer's element dtype
    (``vec_dtype.bytes > buf_dtype.bytes`` — typically B32 reads from
    a bf16/f16 buffer in NormalizeAndStore's coalesced epilogue), the
    address must advance by ``vec_dtype.bytes / buf_dtype.bytes`` buf
    elements per vec element, AND each vec element must be reassembled
    via ``as_type`` so the read transfers the full vec_dtype.bytes
    payload — not the implicit-narrowing-to-uint pattern the original
    naive lowering produced (which only moved buf_dtype.bytes per
    vec element, dropping half the data on every cooperative store).

    Same-dtype path (vec_dtype == buf_dtype, no pred) emits packed
    ``T{2,3,4}`` reinterpret_cast loads — one wide read instead of N
    scalar reads. Required to match hand-written vec4 kernels on
    Apple GPUs; the Metal compiler does not auto-vectorize sequential
    indexed scalar loads through register-named offsets.
    """
    tensor = op.attrs["tensor"]
    (out,) = op.results
    ty = msl_type(out.dtype)
    pred = op.attrs.get("pred")
    indices = op.operands[:-1] if pred is not None else op.operands
    offset_expr = _compute_tensor_offset(tensor, indices, ctx)
    buf = _tensor_buf_name(tensor, ctx)
    comps = ctx.names.components(out)

    buf_bytes = tensor.dtype.bytes
    vec_bytes = out.dtype.bytes
    stride = vec_bytes // buf_bytes if vec_bytes >= buf_bytes else 1
    needs_pack = vec_bytes > buf_bytes

    if pred is None and not needs_pack and tensor.dtype is out.dtype:
        addr_space = _msl_addr_space(tensor)
        if _emit_packed_vec_load(buf, ty, addr_space, offset_expr, comps, ctx):
            return
    # Pick the unsigned-int type matching the buf's bit width. MSL
    # ``as_type<T>`` requires same byte width on source and dest, so
    # ``as_type<ushort>(int)`` fails (2B vs 4B). Mapping:
    #   1B buf → uchar, 2B → ushort, 4B → uint, 8B → ulong.
    _BUF_UINT_BY_BYTES = {1: "uchar", 2: "ushort", 4: "uint", 8: "ulong"}
    buf_uint = _BUF_UINT_BY_BYTES[buf_bytes]

    def _read_at(buf_idx_expr: str) -> str:
        """Return an MSL expression yielding one vec_dtype value at
        the given buf-element index. When the vec dtype is wider than
        the buf dtype, pack ``stride`` consecutive buf elements into a
        ``{buf_uint}{stride}`` vec, then ``as_type`` to the target
        dtype."""
        if not needs_pack:
            return f"{buf}[{buf_idx_expr}]"
        # Bitcast each buf element to the same-byte-width unsigned int,
        # pack into buf_uint{stride}, then reinterpret as the vec dtype.
        parts = ", ".join(
            f"as_type<{buf_uint}>({buf}[{buf_idx_expr} + {j}u])"
            if j > 0
            else f"as_type<{buf_uint}>({buf}[{buf_idx_expr}])"
            for j in range(stride)
        )
        return f"as_type<{ty}>({buf_uint}{stride}({parts}))"

    for i, comp in enumerate(comps):
        bi = f"{offset_expr} + {i * stride}u" if i > 0 else offset_expr
        rhs = _read_at(bi)
        if pred is not None:
            pred_name = ctx.names.name_for(pred)
            ctx.emit(f"{ty} {comp};")
            ctx.emit(f"if ({pred_name}) {{ {comp} = {rhs}; }}")
            ctx.emit(f"else {{ {comp} = static_cast<{ty}>(0); }}")
        else:
            ctx.emit(f"{ty} {comp} = {rhs};")


def _visit_vec_store(self, op: VecStoreOp, ctx: _MslCtx) -> None:
    """Lower VecStoreOp to N consecutive scalar writes. Mirror of
    ``_visit_vec_load``: when the source vec dtype is wider than the
    buffer dtype, unpack each vec element into ``stride`` buf elements
    via ``as_type<ushort{stride}>(value)`` and write them out — keeping
    the per-vec-elem byte count consistent with the requested dtype.

    Same-dtype path (vec.dtype == buf.dtype, no pred) emits packed
    ``T{2,3,4}`` reinterpret_cast stores — see ``_visit_vec_load`` for
    the rationale.
    """
    tensor = op.attrs["tensor"]
    vec = op.operands[0]
    rest = op.operands[1:]
    pred = op.attrs.get("pred")
    if pred is not None:
        rest = rest[:-1]
    offset_expr = _compute_tensor_offset(tensor, rest, ctx)
    buf = _tensor_buf_name(tensor, ctx)
    comps = ctx.names.components(vec)

    buf_bytes = tensor.dtype.bytes
    vec_bytes = vec.dtype.bytes
    stride = vec_bytes // buf_bytes if vec_bytes >= buf_bytes else 1
    needs_unpack = vec_bytes > buf_bytes
    buf_ty = msl_type(tensor.dtype)

    if pred is None and not needs_unpack and tensor.dtype is vec.dtype:
        addr_space = _msl_addr_space(tensor)
        if _emit_packed_vec_store(buf, buf_ty, addr_space, offset_expr, comps, ctx):
            return
    # Matched-width unsigned int for the pack/unpack. Same table as
    # vec_load; ``as_type`` on MSL requires same byte width both ways.
    _BUF_UINT_BY_BYTES = {1: "uchar", 2: "ushort", 4: "uint", 8: "ulong"}
    buf_uint = _BUF_UINT_BY_BYTES[buf_bytes]

    def _emit_write(buf_idx_expr: str, comp: str) -> None:
        if not needs_unpack:
            line = f"{buf}[{buf_idx_expr}] = {comp};"
            ctx.emit(f"if ({pred_name}) {{ {line} }}" if pred is not None else line)
            return
        # Unpack vec elem → stride buf elements via {buf_uint}{stride}.
        tmp = ctx.names.fresh("vu")
        ctx.emit(f"{buf_uint}{stride} {tmp} = as_type<{buf_uint}{stride}>({comp});")
        for j in range(stride):
            sub_idx = f"{buf_idx_expr} + {j}u" if j > 0 else buf_idx_expr
            line = f"{buf}[{sub_idx}] = as_type<{buf_ty}>({tmp}[{j}]);"
            ctx.emit(f"if ({pred_name}) {{ {line} }}" if pred is not None else line)

    pred_name = ctx.names.name_for(pred) if pred is not None else None
    for i, comp in enumerate(comps):
        bi = f"{offset_expr} + {i * stride}u" if i > 0 else offset_expr
        _emit_write(bi, comp)


# ---------------------------------------------------------------------------
# Async copy (synchronous fallback) / Atomics / Subgroup / MMA stubs
# ---------------------------------------------------------------------------


def _visit_async_copy(self, op: AsyncCopyOp, ctx: _MslCtx) -> None:
    """Synchronous fallback for ``cp.async`` on Metal.

    Metal has no async DMA primitive; the kernel waits for the copy
    to land before any consumer reads from the dst smem region. Same-
    dtype gmem→smem (or smem→gmem) copies of width 2/3/4 lower to a
    register-routed packed reinterpret_cast pair instead of N scalar
    elementwise copies — fewer dispatch ops and the optimizer keeps
    the load in registers across the store, which Metal otherwise
    can't see through indexed scalar reads.
    """
    dst_tensor, src_tensor = op.attrs["dst_tensor"], op.attrs["src_tensor"]
    count = int(op.attrs["count"])
    n_dst, n_src = int(op.attrs["n_dst_idx"]), int(op.attrs["n_src_idx"])
    pred = op.attrs.get("pred")
    dst_idxs = tuple(op.operands[:n_dst])
    src_idxs = tuple(op.operands[n_dst : n_dst + n_src])
    dst_off = _compute_tensor_offset(dst_tensor, dst_idxs, ctx)
    src_off = _compute_tensor_offset(src_tensor, src_idxs, ctx)
    dst_buf, src_buf = _tensor_buf_name(dst_tensor, ctx), _tensor_buf_name(src_tensor, ctx)
    n_elems = count // src_tensor.dtype.bytes

    if pred is not None:
        pred_name = ctx.names.name_for(pred)
        ctx.emit(f"if ({pred_name}) {{")
        ctx.indent += 1

    chunks = _packed_vec_chunks(n_elems) if src_tensor.dtype is dst_tensor.dtype else None
    src_space = _msl_addr_space(src_tensor)
    dst_space = _msl_addr_space(dst_tensor)
    if chunks is not None and src_space and dst_space:
        ty = msl_type(src_tensor.dtype)
        cursor = 0
        for w in chunks:
            s = f"{src_off} + {cursor}u" if cursor > 0 else src_off
            d = f"{dst_off} + {cursor}u" if cursor > 0 else dst_off
            tmp = ctx.names.fresh("ac")
            ctx.emit(
                f"{ty}{w} {tmp} = *reinterpret_cast<const {src_space} {ty}{w}*>(&{src_buf}[{s}]);"
            )
            ctx.emit(f"*reinterpret_cast<{dst_space} {ty}{w}*>(&{dst_buf}[{d}]) = {tmp};")
            cursor += w
    else:
        for i in range(n_elems):
            s = f"{src_off} + {i}u" if i > 0 else src_off
            d = f"{dst_off} + {i}u" if i > 0 else dst_off
            ctx.emit(f"{dst_buf}[{d}] = {src_buf}[{s}];")

    if pred is not None:
        ctx.indent -= 1
        ctx.emit("}")


def _visit_async_commit(self, op: AsyncCopyCommitOp, ctx: _MslCtx) -> None:
    ctx.emit("// async_commit — no-op on Metal (synchronous fallback)")


def _visit_async_wait(self, op: AsyncCopyWaitOp, ctx: _MslCtx) -> None:
    ctx.emit("threadgroup_barrier(metal::mem_flags::mem_threadgroup);")


def _visit_atomic_rmw(self, op: AtomicRmwOp, ctx: _MslCtx) -> None:
    tensor, atomic_op = op.attrs["tensor"], op.attrs["op"]
    (out,) = op.results
    value = op.operands[0]
    indices = op.operands[1:]
    pred = op.attrs.get("pred")
    if pred is not None:
        indices = indices[:-1]
    ctx.uses_atomics = True
    offset_expr = _compute_tensor_offset(tensor, indices, ctx)
    buf = _tensor_buf_name(tensor, ctx)
    # Track which output buffer this rmw targets — kernels with a mix of
    # atomic and plain-store outputs (moe_router_correct: counts is rmw,
    # token_ids / slot_weights / offsets get plain stores) need
    # per-output qualification, not a kernel-wide flag.
    ctx.atomic_output_names.add(buf)
    fn = _ATOMIC_FN.get(atomic_op)
    if fn is None:
        raise NotImplementedError(f"AtomicRmwOp: op {atomic_op!r} not supported on MSL")
    dst, ty, val = ctx.names.name_for(out), msl_type(out.dtype), ctx.names.name_for(value)
    expr = (
        f"{fn}(reinterpret_cast<device atomic_{ty}*>"
        f"(&{buf}[{offset_expr}]), {val}, memory_order_relaxed)"
    )
    if pred is not None:
        ctx.emit(f"{ty} {dst};")
        ctx.emit(f"if ({ctx.names.name_for(pred)}) {{ {dst} = {expr}; }}")
    else:
        ctx.emit(f"{ty} {dst} = {expr};")


def _visit_shuffle(self, op: ShuffleOp, ctx: _MslCtx) -> None:
    kind, param = op.attrs["kind"], int(op.attrs["param"])
    (out,) = op.results
    dst, src_name = ctx.names.name_for(out), ctx.names.name_for(op.operands[0])
    ty = msl_type(out.dtype)
    ctx.emit(f"{ty} {dst} = {_SHUFFLE_FN[kind]}({src_name}, {param}u);")


def _visit_subgroup_reduce(self, op: SubgroupReduceOp, ctx: _MslCtx) -> None:
    reduce_op = op.attrs["op"]
    (out,) = op.results
    dst, src_name = ctx.names.name_for(out), ctx.names.name_for(op.operands[0])
    ty = msl_type(out.dtype)
    fn = _REDUCE_FN.get(reduce_op)
    if fn is None:
        raise NotImplementedError(f"SubgroupReduceOp: op {reduce_op!r}")
    ctx.emit(f"{ty} {dst} = {fn}({src_name});")


def _visit_subgroup_broadcast(self, op: SubgroupBroadcastOp, ctx: _MslCtx) -> None:
    lane = int(op.attrs["lane"])
    (out,) = op.results
    dst, src_name = ctx.names.name_for(out), ctx.names.name_for(op.operands[0])
    ty = msl_type(out.dtype)
    ctx.emit(f"{ty} {dst} = simd_broadcast({src_name}, {lane}u);")


def _visit_frag_slice(self, op: FragSliceOp, ctx: _MslCtx) -> None:
    """Bind the result Value to a slice of the source's components.

    Pure rename — no MSL emitted. The sliced fragment shares lane
    storage with its source. Storage class (NAX vs simdgroup_matrix)
    is inherited so downstream Frag* visitors dispatch correctly.
    """
    (src,) = op.operands
    (out,) = op.results
    start = int(op.attrs["start"])
    length = int(op.attrs["length"])
    src_comps = ctx.names.components(src)
    ctx.names.bind(out, tuple(src_comps[start : start + length]))
    if src.id in ctx.nax_frag_ids:
        ctx.nax_frag_ids.add(out.id)
    if src.id in ctx.frag_values:
        # simdgroup_matrix path: keep the same registration so frag ops
        # on the slice see the source's tile metadata.
        ctx.frag_values[out.id] = ctx.frag_values[src.id]
    # NAX-array slice: when the slice is sub-frag-aligned, the
    # ``&parent[start//8]`` pointer covers the slice's elements.
    # That lets MMA helper-call emission pass an array reference for
    # the slice (e.g. ``&S_array[1]`` for a width-8 slice starting at
    # index 8 of a width-16 parent). Records the slice's "view" as a
    # special array entry so ``_visit_mma_nax`` can detect it.
    if src.id in ctx.nax_frag_arrays and start % 8 == 0 and length % 8 == 0:
        parent_arr, _parent_n_frags = ctx.nax_frag_arrays[src.id]
        slice_n_frags = length // 8
        # Record the slice's name as ``parent`` (start=0) or
        # ``parent[start//8]`` so the helper-call site can emit
        # ``&{name}[0]`` uniformly across both cases.
        slice_arr_name = parent_arr if start == 0 else f"{parent_arr}[{start // 8}]"
        ctx.nax_frag_arrays[out.id] = (slice_arr_name, slice_n_frags)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

DISPATCH: dict[type, Any] = {
    ConstOp: _visit_const,
    ArithOp: _visit_arith,
    MathOp: _visit_math,
    CmpOp: _visit_cmp,
    SelectOp: _visit_select,
    ConvertOp: _visit_convert,
    PackedConvertOp: _visit_packed_convert,
    BitcastOp: _visit_bitcast,
    VecBuildOp: _visit_vec_build,
    VecExtractOp: _visit_vec_extract,
    SplitB32Op: _visit_split_b32,
    MergeB32Op: _visit_merge_b32,
    ShuffleOp: _visit_shuffle,
    SubgroupReduceOp: _visit_subgroup_reduce,
    SubgroupBroadcastOp: _visit_subgroup_broadcast,
    ThreadIdxOp: _visit_thread_idx,
    BlockIdxOp: _visit_block_idx,
    BlockDimOp: _visit_block_dim,
    GridDimOp: _visit_grid_dim,
    LaneIdOp: _visit_lane_id,
    SubgroupIdOp: _visit_subgroup_id,
    GroupIdOp: _visit_group_id,
    ThreadIdInGroupOp: _visit_thread_id_in_group,
    BarrierOp: _visit_barrier,
    ForLoopOp: _visit_for_loop,
    IfRegionOp: _visit_if_region,
    WhileLoopOp: _visit_while_loop,
    YieldOp: _visit_yield,
    SmemAllocOp: _visit_smem_alloc,
    LoadOp: _visit_load,
    StoreOp: _visit_store,
    VecLoadOp: _visit_vec_load,
    VecStoreOp: _visit_vec_store,
    AsyncCopyOp: _visit_async_copy,
    AsyncCopyCommitOp: _visit_async_commit,
    AsyncCopyWaitOp: _visit_async_wait,
    AtomicRmwOp: _visit_atomic_rmw,
    MmaOp: _visit_mma,
    FragApplyOp: _visit_frag_apply,
    FragConvertOp: _visit_frag_convert,
    FragForEachOp: _visit_frag_for_each,
    FragReduceOp: _visit_frag_reduce,
    FragSliceOp: _visit_frag_slice,
    LoadMatrixOp: _visit_load_matrix,
    StoreMatrixOp: _visit_store_matrix,
    StoreMatrixGateResidualOp: _visit_store_matrix_gate_residual,
}
