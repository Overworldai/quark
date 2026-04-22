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


def _visit_arith(self, op: ArithOp, ctx: _MslCtx) -> None:
    kind = op.attrs["kind"]
    (out,) = op.results
    dst = ctx.names.name_for(out)
    ty = msl_type(out.dtype)

    if kind == "neg":
        src = ctx.names.name_for(op.operands[0])
        ctx.emit(f"{ty} {dst} = -{src};")
    elif kind == "abs":
        src = ctx.names.name_for(op.operands[0])
        ctx.emit(f"{ty} {dst} = metal::abs({src});")
    elif kind == "fma":
        a = ctx.names.name_for(op.operands[0])
        b = ctx.names.name_for(op.operands[1])
        c = ctx.names.name_for(op.operands[2])
        ctx.emit(f"{ty} {dst} = metal::fma({a}, {b}, {c});")
    elif kind == "min":
        a = ctx.names.name_for(op.operands[0])
        b = ctx.names.name_for(op.operands[1])
        ctx.emit(f"{ty} {dst} = metal::min({a}, {b});")
    elif kind == "max":
        a = ctx.names.name_for(op.operands[0])
        b = ctx.names.name_for(op.operands[1])
        ctx.emit(f"{ty} {dst} = metal::max({a}, {b});")
    elif kind == "mul_hi":
        a = ctx.names.name_for(op.operands[0])
        b = ctx.names.name_for(op.operands[1])
        # MSL's metal::mulhi computes the high half of an unsigned
        # 32×32→64 multiply (or the matching width for u16/u64 ops).
        ctx.emit(f"{ty} {dst} = metal::mulhi({a}, {b});")
    else:
        sym = _ARITH_OP.get(kind)
        if sym is None:
            raise NotImplementedError(f"ArithOp kind {kind!r} not implemented for MSL")
        a = ctx.names.name_for(op.operands[0])
        b = ctx.names.name_for(op.operands[1])
        ctx.emit(f"{ty} {dst} = {a} {sym} {b};")


def _visit_math(self, op: MathOp, ctx: _MslCtx) -> None:
    kind = op.attrs["kind"]
    (out,) = op.results
    dst = ctx.names.name_for(out)
    src = ctx.names.name_for(op.operands[0])
    ty = msl_type(out.dtype)

    if kind in ("rcp", "rcp_approx"):
        ctx.emit(f"{ty} {dst} = static_cast<{ty}>(1.0f) / {src};")
    else:
        fn = _MATH_FN.get(kind)
        if fn is None:
            raise NotImplementedError(f"MathOp kind {kind!r} not implemented for MSL")
        ctx.emit(f"{ty} {dst} = {fn}({src});")


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


def _visit_split_b32(self, op: SplitB32Op, ctx: _MslCtx) -> None:
    lo, hi = op.results
    src_name = ctx.names.name_for(op.operands[0])
    lo_name = ctx.names.name_for(lo)
    hi_name = ctx.names.name_for(hi)
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


def _visit_load(self, op: LoadOp, ctx: _MslCtx) -> None:
    tensor = op.attrs["tensor"]
    (out,) = op.results
    dst = ctx.names.name_for(out)
    ty = msl_type(out.dtype)
    pred = op.attrs.get("pred")
    indices = op.operands[:-1] if pred is not None else op.operands
    offset_expr = _compute_tensor_offset(tensor, indices, ctx)
    buf = _tensor_buf_name(tensor, ctx)

    if pred is not None:
        pred_name = ctx.names.name_for(pred)
        ctx.emit(f"{ty} {dst};")
        ctx.emit(f"if ({pred_name}) {{ {dst} = {buf}[{offset_expr}]; }}")
        ctx.emit(f"else {{ {dst} = static_cast<{ty}>(0); }}")
    else:
        ctx.emit(f"{ty} {dst} = {buf}[{offset_expr}];")


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

    if pred is not None:
        pred_name = ctx.names.name_for(pred)
        ctx.emit(f"if ({pred_name}) {{ {buf}[{offset_expr}] = {val_name}; }}")
    else:
        ctx.emit(f"{buf}[{offset_expr}] = {val_name};")


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
    the per-vec-elem byte count consistent with the requested dtype."""
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
    LoadMatrixOp: _visit_load_matrix,
    StoreMatrixOp: _visit_store_matrix,
}
