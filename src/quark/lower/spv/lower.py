"""SPIR-V compute-shader lowerer (Intel Arc / Xe iGPU).

PORTABILITY_PLAN §3.2 first cut. Lowers a quark IR ``Module`` to a
SPIR-V text-assembly source string that ``spirv-as`` consumes; the
caller (``drivers.spv.SpvDriver`` via the launcher) runs the text
→ binary step + dispatches.

Visitor coverage is intentionally tight — just enough to lower a
``vec_add``-style kernel (3 GlobalTensor params, ``thread_idx`` for
the index, scalar arith, scalar load/store). Extending to the full
~45-op surface is incremental: each new visitor adds an entry to
``_DISPATCH`` + a per-section emit through the ``SpvText`` helper.
The §3.1 dispatch path proves the whole pipeline; visitors fill in.

Non-goals for the v1 lowerer (tracked in PORTABILITY_PLAN §3.3 / §3.6):
  * Cooperative-matrix MMA — needs ``OpCooperativeMatrixMulAddKHR``
    + the lane-coordinate FragApply analysis.
  * ``FragForEachOp`` / ``FragReduceOp`` — rejected at
    ``is_valid_for(caps)``; flash-attention rides on top in v3.
  * Multi-function modules — only the entry function is honoured.

Conventions locked in this first cut:
  * Storage buffers at descriptor set 0, sequential bindings 0..n-1.
  * Push-constant block at offset 0.
  * Entry point ``"main"`` (regardless of the IR function name).
  * LocalSize taken from a ``local_size`` attr on the function (the
    builder helper sets it; defaults to ``(64, 1, 1)`` if missing).
  * Single workgroup dispatch — kernel uses ``LocalInvocationId.x``
    as the per-element index. Composing global ids from
    ``block_idx*block_dim + thread_idx`` lands in the next visitor
    bundle once the framework's existing kernels start exercising
    that path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quark.ir import DType
from quark.ir.module import Function, Module
from quark.ir.op import (
    ArithOp,
    AtomicRmwOp,
    BarrierOp,
    BitcastOp,
    BlockDimOp,
    BlockIdxOp,
    CmpOp,
    ConstOp,
    ConvertOp,
    ForLoopOp,
    FragApplyOp,
    FragConvertOp,
    FragForEachOp,
    FragReduceOp,
    GroupIdOp,
    IfRegionOp,
    LaneIdOp,
    LoadMatrixOp,
    LoadOp,
    MathOp,
    MergeB32Op,
    MmaOp,
    SelectOp,
    SmemAllocOp,
    SplitB32Op,
    StoreMatrixOp,
    StoreOp,
    SubgroupIdOp,
    SubgroupReduceOp,
    ShuffleOp,
    ThreadIdInGroupOp,
    ThreadIdxOp,
    VecBuildOp,
    VecExtractOp,
    VecLoadOp,
    VecStoreOp,
    YieldOp,
)
from quark.ir.tensor import GlobalTensor, SharedRegion

from quark.lower.spv.text import SpvText


@dataclass
class LoweredSpirVKernel:
    """The SPIR-V-side ``LoweredKernel`` peer.

    ``source`` is SPIR-V text (assembly format consumed by
    ``spirv-as``). ``binary`` is filled in by ``assemble()`` —
    callers that already have the SPIR-V-Tools Python bindings can
    skip this step; everyone else shells out via
    ``quark.lower.spv.assemble.text_to_binary``.
    """

    source: str
    entry_name: str = "main"
    n_buffers: int = 0
    push_constants_size: int = 0
    smem_bytes: int = 0
    local_size: tuple[int, int, int] = (1, 1, 1)

    @property
    def kernel_name(self) -> str:
        """Compatibility alias matching ``LoweredKernel.kernel_name``
        (CUDA) and ``LoweredMslKernel.kernel_name`` (Metal). The
        launcher reads this property when constructing
        ``CompiledKernel.entry`` — the SPIR-V backend uses the
        explicit ``entry_name`` field (defaults to ``"main"``)."""
        return self.entry_name


@dataclass
class _SpvCtx:
    """Per-function lowering state. Mirrors ``_MslCtx`` but the
    side-tables it needs are different (SPIR-V is SSA-typed at
    every op; MSL is a syntactic emit)."""

    text: SpvText = field(default_factory=SpvText)
    # Value.id → SPIR-V SSA id (the result of OpLoad / OpFAdd / ...)
    val_to_id: dict[int, str] = field(default_factory=dict)
    # GlobalTensor id() → SSA id of the OpVariable for its buffer.
    tensor_to_var: dict[int, str] = field(default_factory=dict)
    # GlobalTensor id() → element OpTypePointer (StorageBuffer + elem)
    tensor_to_elem_ptr: dict[int, str] = field(default_factory=dict)
    # GlobalTensor id() → element type id (OpTypeFloat32, etc.)
    tensor_to_elem_type: dict[int, str] = field(default_factory=dict)
    # Cached SSA ids for compute-shader builtins. Lazily declared on
    # first reference — kernel that never reads ``LocalInvocationId``
    # doesn't get a useless ``%LocalInvocationId`` variable.
    local_inv_id_var: str = ""
    local_inv_id_x_loaded: str = ""
    workgroup_id_var: str = ""
    workgroup_id_x_loaded: str = ""
    # Local size constants from ``OpExecutionMode LocalSize`` —
    # populated by the lowerer entry point so ``BlockDimOp``
    # visitors emit the right ``OpConstant uint``.
    local_size: tuple[int, int, int] = (1, 1, 1)
    # Cached push-constant block id, if any.
    push_block_var: str = ""
    push_block_size: int = 0
    # Tracks GlobalTensors actually used (load/store visited). Drives
    # the entry-point interface list — Vulkan 1.3 SPIR-V demands every
    # statically-used global variable be enumerated there.
    tensor_to_binding: dict[int, int] = field(default_factory=dict)
    # Capability tracking for dtype-gated decls. Set on first use of
    # an ``OpTypeFloat 16`` / BFloat16 type; ``_emit_dtype`` reads it
    # to avoid re-emitting the capability line.
    has_f16_cap: bool = False
    has_bf16_cap: bool = False
    has_int16_cap: bool = False
    # Smem allocations: SmemAllocOp result Value.id → (var_id,
    # element_type_id, elem_pointer_id, total_elements). Visitors
    # that load/store on a SharedRegion look up by the
    # SharedRegion.alloc.id (which equals the SmemAllocOp's
    # backing Value id).
    smem_allocs: dict[int, tuple[str, str, str, int]] = field(default_factory=dict)
    # Stack of pending loop-yield targets. Each entry is the list of
    # ``(target_id, type_id)`` slots a ``YieldOp`` inside the loop body
    # should ``OpCopyObject`` into so the surrounding ``ForLoopOp``'s
    # ``OpPhi`` at the header can reference a known SSA id (allocated
    # by ``_visit_for_loop`` upfront, defined via OpCopyObject when the
    # body's YieldOp fires). Stack-shaped to allow nested for-loops.
    loop_yield_stack: list[list[tuple[str, str]]] = field(default_factory=list)


# Map quark DType values → (SPIR-V type emitter method name, byte width).
# Anything not in this map causes a NotImplementedError so missing
# coverage surfaces fast rather than silently emitting wrong types.
_DTYPE_TO_SPIR: dict[DType, tuple[str, int]] = {
    DType.F32: ("type_float", 4),
    DType.U32: ("type_int_unsigned", 4),
    DType.S32: ("type_int_signed", 4),
}


def _emit_dtype(text: SpvText, dt: DType, ctx: "_SpvCtx | None" = None) -> str:
    """Map a quark ``DType`` to its SPIR-V type id.

    Half-precision dtypes (f16, bf16) require declaring the
    ``Float16`` / ``BFloat16TypeKHR`` capability at module scope —
    handled when ``ctx`` is provided so the cap lands exactly once.
    """
    if dt is DType.F32:
        return text.type_float(32)
    if dt is DType.U32:
        return text.type_int(32, signed=False)
    if dt is DType.S32:
        return text.type_int(32, signed=True)
    if dt is DType.F16:
        if ctx is not None and not ctx.has_f16_cap:
            text.add_capability("Float16")
            ctx.has_f16_cap = True
        return text.type_float(16)
    if dt is DType.BF16:
        if ctx is not None and not ctx.has_bf16_cap:
            text.add_capability("BFloat16TypeKHR")
            text.add_extension("SPV_KHR_bfloat16")
            ctx.has_bf16_cap = True
        # ``OpTypeFloat 16 BFloat16KHR`` — the proper bf16 type from
        # SPV_KHR_bfloat16. spirv-as accepts the suffix only on
        # ``vulkan1.4`` (or higher) so the assembler target was bumped
        # to 1.4 alongside this. The earlier f16 fallback caused
        # silent miscompiles: f16's exponent bias / range are wrong
        # for bf16 bit patterns, so loaded values were re-interpreted
        # as out-of-range f16 NaNs / denormals.
        return text.type_float(16, bfloat16=True)
    if dt is DType.PRED:
        # PRED → OpTypeBool. Vulkan compute doesn't allow OpTypeBool
        # in storage buffers — it lives only as an SSA value (cmp
        # results, conditional branches, OpSelect). Using PRED in a
        # buffer would need an explicit u8/u32 storage encoding at
        # the kernel boundary.
        return text.type_bool()
    if dt is DType.B32:
        # ``B32`` is a raw 32-bit bit pattern — used by the
        # ``fma_bf16x2`` legalization to carry a packed bf16×2 pair.
        # SPIR-V has no distinct "bits" type at width 32; OpTypeInt 32
        # is the canonical reinterpretation target (every bitcast
        # site goes through OpBitcast which doesn't care about
        # signedness).
        return text.type_int(32, signed=False)
    if dt is DType.B16:
        # ``B16`` is a raw 16-bit bit pattern — used by the
        # ``fma_bf16x2`` expansion's split/merge step. Needs the
        # ``Int16`` capability to land an ``OpTypeInt 16``; the
        # capability is gated on the device exposing it
        # (Battlemage does, behind ``shaderInt16``).
        if ctx is not None and not getattr(ctx, "has_int16_cap", False):
            text.add_capability("Int16")
            ctx.has_int16_cap = True  # type: ignore[attr-defined]
        return text.type_int(16, signed=False)
    raise NotImplementedError(
        f"SpirVLowerer: dtype {dt!r} not yet supported. See "
        "PORTABILITY_PLAN §3.2 — extend the visitor table to "
        "cover this dtype."
    )


def _dtype_byte_width(dt: DType) -> int:
    if dt in (DType.F32, DType.U32, DType.S32):
        return 4
    if dt in (DType.F16, DType.BF16, DType.U16, DType.S16):
        return 2
    if dt in (DType.U8, DType.S8):
        return 1
    raise NotImplementedError(
        f"SpirVLowerer: byte width for {dt!r} not known"
    )


# ─────────────────────────────────────────────────────────────────
# Visitor implementations.
# ─────────────────────────────────────────────────────────────────


def _visit_const(op: ConstOp, ctx: _SpvCtx) -> None:
    (out,) = op.results
    dt = out.dtype
    raw = op.attrs.get("value")
    if dt is DType.U32:
        cid = ctx.text.const_uint(int(raw))
    elif dt is DType.S32:
        # SPIR-V doesn't distinguish signed/unsigned constants in the
        # text format — emit as OpConstant on the signed type.
        s32 = ctx.text.type_int(32, signed=True)
        cid = ctx.text.alloc_id(f"s_{int(raw)}")
        ctx.text.add_type_line(f"{cid} = OpConstant {s32} {int(raw)}")
    elif dt is DType.F32:
        cid = ctx.text.const_float(float(raw))
    elif dt is DType.PRED:
        # Boolean literal — SPIR-V has dedicated opcodes
        # (no OpConstant for OpTypeBool). Cache by the constant
        # itself so ``True``/``False`` only emit once per kernel.
        bool_t = ctx.text.type_bool()
        kind = "true" if bool(raw) else "false"
        cid = ctx.text._cached_type(
            f"const_{kind}",
            f"OpConstant{kind.title()} {bool_t}",
        )
    else:
        raise NotImplementedError(
            f"_visit_const: dtype {dt!r} — extend SpirVLowerer per "
            "PORTABILITY_PLAN §3.2"
        )
    ctx.val_to_id[out.id] = cid


def _visit_arith(op: ArithOp, ctx: _SpvCtx) -> None:
    (out,) = op.results
    kind = op.attrs.get("kind", "")
    operands = [ctx.val_to_id[v.id] for v in op.operands]
    type_id = _emit_dtype(ctx.text, out.dtype, ctx)

    # Unary kinds need their own emit: SPIR-V doesn't have a single
    # binary "neg" opcode the way binary kinds do. ``OpFNegate`` /
    # ``OpSNegate`` for negation; ``GLSL.std.450 FAbs`` /
    # ``SAbs`` for abs.
    if kind == "neg":
        if _dtype_kind(out.dtype) == "float":
            spv_op = "OpFNegate"
        elif _dtype_kind(out.dtype) == "sint":
            spv_op = "OpSNegate"
        else:
            raise NotImplementedError(
                f"_visit_arith(neg): unsigned negation isn't expressible — "
                f"kernel emitted neg on {out.dtype!r}"
            )
        res_id = ctx.text.alloc_id("neg")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = {spv_op} {type_id} {operands[0]}")
        return
    if kind == "abs":
        glsl_id = ctx.text.import_ext_inst("GLSL.std.450")
        glsl_name = "FAbs" if _dtype_kind(out.dtype) == "float" else "SAbs"
        res_id = ctx.text.alloc_id("abs")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpExtInst {type_id} {glsl_id} {glsl_name} {operands[0]}"
        )
        return
    if kind == "fma":
        # ``GLSL.std.450 Fma`` = a * b + c. Mirrors PTX ``fma.rn``.
        glsl_id = ctx.text.import_ext_inst("GLSL.std.450")
        res_id = ctx.text.alloc_id("fma")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpExtInst {type_id} {glsl_id} Fma {operands[0]} {operands[1]} {operands[2]}"
        )
        return
    if kind == "mul_hi":
        # SPIR-V has no single ``OpUMulHi``. The standard idiom is
        # ``OpUMulExtended`` which returns a struct ``{low, high}``;
        # extract the high half. Used by hash-based RNG kernels
        # (xoshiro / philox) for state advance.
        if _dtype_kind(out.dtype) != "uint":
            raise NotImplementedError(
                f"_visit_arith(mul_hi): only unsigned int wired, got {out.dtype!r}"
            )
        u32 = ctx.text.type_int(32, signed=False)
        # Declare an ``OpTypeStruct {u32, u32}`` for the extended result.
        struct_type = ctx.text._cached_type(
            f"struct_u32u32",
            f"OpTypeStruct {u32} {u32}",
        )
        ext_id = ctx.text.alloc_id("mul_ext")
        ctx.text.emit_function(
            f"{ext_id} = OpUMulExtended {struct_type} {operands[0]} {operands[1]}"
        )
        res_id = ctx.text.alloc_id("mul_hi")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpCompositeExtract {u32} {ext_id} 1"
        )
        return
    if kind in ("min", "max"):
        glsl_id = ctx.text.import_ext_inst("GLSL.std.450")
        if _dtype_kind(out.dtype) == "float":
            glsl_name = "FMin" if kind == "min" else "FMax"
        elif _dtype_kind(out.dtype) == "sint":
            glsl_name = "SMin" if kind == "min" else "SMax"
        else:
            glsl_name = "UMin" if kind == "min" else "UMax"
        res_id = ctx.text.alloc_id(kind)
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpExtInst {type_id} {glsl_id} {glsl_name} {operands[0]} {operands[1]}"
        )
        return

    spv_op = _ARITH_KIND_TO_SPV.get((kind, out.dtype))
    if spv_op is None:
        raise NotImplementedError(
            f"_visit_arith: kind={kind!r} dtype={out.dtype!r} not yet "
            "wired. See PORTABILITY_PLAN §3.2."
        )
    res_id = ctx.text.alloc_id(kind)
    ctx.val_to_id[out.id] = res_id
    args = " ".join(operands)
    ctx.text.emit_function(f"{res_id} = {spv_op} {type_id} {args}")


# Mapping from (quark ArithOp.kind, result-dtype) → SPIR-V opcode.
# Add entries as new ops show up in the visitor coverage. Type-aware
# because ``add`` lowers to ``OpFAdd`` for f32 but ``OpIAdd`` for
# u32/s32 — same kind, different opcode.
#
# Bitwise / shift ops on int types use their integer SPIR-V opcodes
# directly; SPIR-V doesn't distinguish signed/unsigned for And/Or/
# Xor (the bit pattern is what matters).
_ARITH_KIND_TO_SPV: dict[tuple[str, DType], str] = {
    ("add", DType.F32): "OpFAdd",
    ("sub", DType.F32): "OpFSub",
    ("mul", DType.F32): "OpFMul",
    ("div", DType.F32): "OpFDiv",
    ("add", DType.U32): "OpIAdd",
    ("sub", DType.U32): "OpISub",
    ("mul", DType.U32): "OpIMul",
    ("div", DType.U32): "OpUDiv",
    ("rem", DType.U32): "OpUMod",
    ("add", DType.S32): "OpIAdd",
    ("sub", DType.S32): "OpISub",
    ("mul", DType.S32): "OpIMul",
    ("div", DType.S32): "OpSDiv",
    ("rem", DType.S32): "OpSMod",
    ("shl", DType.U32): "OpShiftLeftLogical",
    ("shr", DType.U32): "OpShiftRightLogical",
    ("shl", DType.S32): "OpShiftLeftLogical",
    ("shr", DType.S32): "OpShiftRightArithmetic",
    ("and", DType.U32): "OpBitwiseAnd",
    ("or", DType.U32): "OpBitwiseOr",
    ("xor", DType.U32): "OpBitwiseXor",
    ("and", DType.S32): "OpBitwiseAnd",
    ("or", DType.S32): "OpBitwiseOr",
    ("xor", DType.S32): "OpBitwiseXor",
    # Boolean (PRED) logic ops use the dedicated ``OpLogical*`` opcodes
    # — bitwise opcodes don't accept ``OpTypeBool`` operands in SPIR-V.
    ("and", DType.PRED): "OpLogicalAnd",
    ("or", DType.PRED): "OpLogicalOr",
    ("xor", DType.PRED): "OpLogicalNotEqual",
}


_DIM_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def _ensure_builtin_vec3_component(
    ctx: _SpvCtx,
    *,
    var_attr: str,
    cache_attr_prefix: str,
    builtin_name: str,
    dim: str,
) -> str:
    """Generic helper for builtin ``BuiltIn`` vec3 inputs:
    ``LocalInvocationId``, ``WorkgroupId``, etc.

    Lazily declares the ``OpVariable Input`` for the builtin (one per
    builtin per kernel), loads its vec3 once, then extracts the
    requested component. Component extracts are also cached so a
    kernel that reads ``thread_idx("x")`` twice doesn't emit two
    ``OpCompositeExtract`` ops.
    """
    if dim not in _DIM_TO_INDEX:
        raise ValueError(f"_ensure_builtin: bad dim={dim!r}")

    cached_attr = f"{cache_attr_prefix}_{dim}_loaded"
    cached = getattr(ctx, cached_attr, "")
    if cached:
        return cached

    var_id = getattr(ctx, var_attr, "")
    if not var_id:
        u32 = ctx.text.type_int(32, signed=False)
        v3u = ctx.text.type_vec(u32, 3)
        ptr = ctx.text.type_pointer("Input", v3u)
        var_id = ctx.text.alloc_id(builtin_name)
        ctx.text.add_type_line(f"{var_id} = OpVariable {ptr} Input")
        ctx.text.add_decoration(f"OpDecorate {var_id} BuiltIn {builtin_name}")
        setattr(ctx, var_attr, var_id)

    u32 = ctx.text.type_int(32, signed=False)
    v3u = ctx.text.type_vec(u32, 3)
    # Load the vec3 once per (kernel, dim-extract) pair. Cache the
    # vec on the ctx by stashing it under the ``_x_loaded`` slot when
    # dim==x is the first read; subsequent dims need a fresh load
    # only when neither ``y`` nor ``z`` was previously cached. Simple
    # path: each dim has its own cache slot, so emit one ``OpLoad``
    # per dim accessed (cheap — Apple/Intel's compiler folds them).
    loaded = ctx.text.alloc_id(f"{builtin_name}_{dim}_vec")
    ctx.text.emit_function(f"{loaded} = OpLoad {v3u} {var_id}")
    res_id = ctx.text.alloc_id(f"{builtin_name}_{dim}")
    ctx.text.emit_function(
        f"{res_id} = OpCompositeExtract {u32} {loaded} {_DIM_TO_INDEX[dim]}"
    )
    setattr(ctx, cached_attr, res_id)
    return res_id


def _ensure_local_invocation_id(ctx: _SpvCtx, dim: str = "x") -> str:
    return _ensure_builtin_vec3_component(
        ctx,
        var_attr="local_inv_id_var",
        cache_attr_prefix="local_inv_id",
        builtin_name="LocalInvocationId",
        dim=dim,
    )


def _visit_thread_idx(op: ThreadIdxOp, ctx: _SpvCtx) -> None:
    """``thread_idx(dim)`` → ``LocalInvocationId.<dim>``."""
    (out,) = op.results
    dim = op.attrs.get("dim", "x")
    ctx.val_to_id[out.id] = _ensure_local_invocation_id(ctx, dim)


def _ensure_workgroup_id(ctx: _SpvCtx, dim: str = "x") -> str:
    return _ensure_builtin_vec3_component(
        ctx,
        var_attr="workgroup_id_var",
        cache_attr_prefix="workgroup_id",
        builtin_name="WorkgroupId",
        dim=dim,
    )


def _visit_block_idx(op: BlockIdxOp, ctx: _SpvCtx) -> None:
    """``block_idx(dim)`` → ``WorkgroupId.<dim>``."""
    (out,) = op.results
    dim = op.attrs.get("dim", "x")
    ctx.val_to_id[out.id] = _ensure_workgroup_id(ctx, dim)


def _ensure_scalar_builtin(
    ctx: _SpvCtx,
    *,
    var_attr: str,
    cache_attr: str,
    builtin_name: str,
) -> str:
    """Lazily declare a scalar ``BuiltIn`` u32 input variable + load
    it. Used for ``SubgroupLocalInvocationId`` (lane id within
    subgroup) and ``SubgroupId`` (subgroup id within workgroup) —
    each is a single u32 the kernel reads directly, no vec3 dance.
    """
    cached = getattr(ctx, cache_attr, "")
    if cached:
        return cached
    var_id = getattr(ctx, var_attr, "")
    if not var_id:
        u32 = ctx.text.type_int(32, signed=False)
        ptr = ctx.text.type_pointer("Input", u32)
        var_id = ctx.text.alloc_id(builtin_name)
        ctx.text.add_type_line(f"{var_id} = OpVariable {ptr} Input")
        ctx.text.add_decoration(f"OpDecorate {var_id} BuiltIn {builtin_name}")
        setattr(ctx, var_attr, var_id)
    u32 = ctx.text.type_int(32, signed=False)
    res_id = ctx.text.alloc_id(builtin_name + "_v")
    ctx.text.emit_function(f"{res_id} = OpLoad {u32} {var_id}")
    setattr(ctx, cache_attr, res_id)
    return res_id


def _visit_lane_id(op: LaneIdOp, ctx: _SpvCtx) -> None:
    """``lane_id()`` → ``SubgroupLocalInvocationId``.

    SPIR-V's ``SubgroupLocalInvocationId`` is a u32 giving the
    invocation's index within its subgroup (0..subgroup_size-1).
    Adds the ``GroupNonUniform`` capability on first use — required
    for ``OpGroupNonUniform*`` and for the builtin to be readable.
    """
    (out,) = op.results
    ctx.text.add_capability("GroupNonUniform")
    ctx.val_to_id[out.id] = _ensure_scalar_builtin(
        ctx,
        var_attr="lane_id_var",
        cache_attr="lane_id_loaded",
        builtin_name="SubgroupLocalInvocationId",
    )


def _visit_subgroup_id(op: SubgroupIdOp, ctx: _SpvCtx) -> None:
    """``subgroup_id()`` → ``SubgroupId``.

    The index of this invocation's subgroup within the workgroup
    (0..n_subgroups_per_workgroup-1). Needed by the ``GroupId =
    laneid >> 2`` PTX fragment formula's quark-IR analogue."""
    (out,) = op.results
    ctx.text.add_capability("GroupNonUniform")
    ctx.val_to_id[out.id] = _ensure_scalar_builtin(
        ctx,
        var_attr="subgroup_id_var",
        cache_attr="subgroup_id_loaded",
        builtin_name="SubgroupId",
    )


def _visit_group_id(op: GroupIdOp, ctx: _SpvCtx) -> None:
    """``group_id()`` = ``laneid >> 2`` (PTX MMA fragment formula).

    Materialises a synthetic op rather than emitting raw shifts at
    every call site — the IR exposes it as a first-class builtin.
    SPIR-V has no equivalent builtin, so we lower as the explicit
    shift on ``SubgroupLocalInvocationId``.
    """
    (out,) = op.results
    lane = _ensure_scalar_builtin(
        ctx,
        var_attr="lane_id_var",
        cache_attr="lane_id_loaded",
        builtin_name="SubgroupLocalInvocationId",
    )
    ctx.text.add_capability("GroupNonUniform")
    u32 = ctx.text.type_int(32, signed=False)
    two = ctx.text.const_uint(2)
    res_id = ctx.text.alloc_id("group_id")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpShiftRightLogical {u32} {lane} {two}")


def _visit_thread_id_in_group(op: ThreadIdInGroupOp, ctx: _SpvCtx) -> None:
    """``thread_id_in_group()`` = ``laneid & 3``. Companion to
    ``group_id`` in the PTX MMA formula. Lowers as an explicit AND
    against the lane id."""
    (out,) = op.results
    lane = _ensure_scalar_builtin(
        ctx,
        var_attr="lane_id_var",
        cache_attr="lane_id_loaded",
        builtin_name="SubgroupLocalInvocationId",
    )
    ctx.text.add_capability("GroupNonUniform")
    u32 = ctx.text.type_int(32, signed=False)
    three = ctx.text.const_uint(3)
    res_id = ctx.text.alloc_id("tid_in_group")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpBitwiseAnd {u32} {lane} {three}")


def _visit_block_dim(op: BlockDimOp, ctx: _SpvCtx) -> None:
    """``block_dim(dim)`` → ``OpConstant uint`` from the kernel's
    pinned ``LocalSize``. SPIR-V's ``OpExecutionMode LocalSize``
    bakes the workgroup size at compile time; the constant the
    visitor emits matches what the entry point declared."""
    (out,) = op.results
    dim = op.attrs.get("dim", "x")
    axis = _DIM_TO_INDEX.get(dim)
    if axis is None:
        raise ValueError(f"_visit_block_dim: bad dim={dim!r}")
    cid = ctx.text.const_uint(int(ctx.local_size[axis]))
    ctx.val_to_id[out.id] = cid


# ─── Comparison + structured if/else ──────────────────────────────


# (kind, dtype) → SPIR-V opcode. ``ord`` (ordered) variants for fp
# match GLSL semantics; integer cmps split signed / unsigned via the
# dtype.
_CMP_KIND_TO_SPV: dict[tuple[str, DType], str] = {
    # f32
    ("eq", DType.F32): "OpFOrdEqual",
    ("ne", DType.F32): "OpFOrdNotEqual",
    ("lt", DType.F32): "OpFOrdLessThan",
    ("le", DType.F32): "OpFOrdLessThanEqual",
    ("gt", DType.F32): "OpFOrdGreaterThan",
    ("ge", DType.F32): "OpFOrdGreaterThanEqual",
    # u32
    ("eq", DType.U32): "OpIEqual",
    ("ne", DType.U32): "OpINotEqual",
    ("lt", DType.U32): "OpULessThan",
    ("le", DType.U32): "OpULessThanEqual",
    ("gt", DType.U32): "OpUGreaterThan",
    ("ge", DType.U32): "OpUGreaterThanEqual",
    # s32
    ("eq", DType.S32): "OpIEqual",
    ("ne", DType.S32): "OpINotEqual",
    ("lt", DType.S32): "OpSLessThan",
    ("le", DType.S32): "OpSLessThanEqual",
    ("gt", DType.S32): "OpSGreaterThan",
    ("ge", DType.S32): "OpSGreaterThanEqual",
}


def _visit_cmp(op: CmpOp, ctx: _SpvCtx) -> None:
    (out,) = op.results
    kind = op.attrs["kind"]
    a, b = op.operands
    spv_op = _CMP_KIND_TO_SPV.get((kind, a.dtype))
    if spv_op is None:
        raise NotImplementedError(
            f"_visit_cmp: kind={kind!r} dtype={a.dtype!r} not yet wired"
        )
    bool_t = ctx.text.type_bool()
    res_id = ctx.text.alloc_id(f"cmp_{kind}")
    ctx.val_to_id[out.id] = res_id
    a_id = ctx.val_to_id[a.id]
    b_id = ctx.val_to_id[b.id]
    ctx.text.emit_function(f"{res_id} = {spv_op} {bool_t} {a_id} {b_id}")


def _visit_if_region(op: IfRegionOp, ctx: _SpvCtx) -> None:
    """Structured if/else, with or without loop-carried values.

    SPIR-V structured control flow:

        OpSelectionMerge %merge None
        OpBranchConditional %pred %then %else

        %then = OpLabel
          ;; ... then body ops ...
          ;; (with carries) YieldOp materialises each value via
          ;; OpCopyObject into pre-allocated then-yield ids.
          OpBranch %then_tail
        %then_tail = OpLabel
          OpBranch %merge

        %else = OpLabel
          ;; mirror
          OpBranch %else_tail
        %else_tail = OpLabel
          OpBranch %merge

        %merge = OpLabel
          ;; (with carries) OpPhi per result picks from
          ;; %then_tail / %else_tail.

    The dedicated ``*_tail`` blocks give OpPhi a stable predecessor
    label even when the arm body emits its own internal control flow
    (nested if/for-loop). Without them, the OpPhi predecessor would
    have to be the *last* basic block of the arm at OpBranch time,
    which the visitors don't currently track.

    For the no-carries case (the bounds-check pattern), the tail
    blocks are still emitted but there's no OpPhi at merge — they
    cost ~3 lines of SPIR-V each, which spirv-as / the GPU optimise
    away during compile.

    Body inputs (``then_body_vars`` / ``else_body_vars``) are aliased
    to the corresponding entry in ``op.operands`` so loads inside the
    arm see the carried-in value's SSA id. With the simple
    ``carried`` API on the builder, the same operand sequence appears
    twice in ``op.operands`` (once for then, once for else); we map
    each arm separately.
    """
    pred_id = ctx.val_to_id[op.pred.id]
    n_carried = int(op.attrs.get("n_carried", 0))

    # Carried-in operand split: (pred, *then_in, *else_in)
    then_in_vals = op.operands[1 : 1 + n_carried]
    else_in_vals = op.operands[1 + n_carried : 1 + 2 * n_carried]

    # Body-visible carry-in values are aliased to their corresponding
    # input operand's SSA id — the body can read them as if they were
    # the operand directly.
    for body_var, src_val in zip(op.then_body_vars, then_in_vals, strict=False):
        ctx.val_to_id[body_var.id] = ctx.val_to_id[src_val.id]
    for body_var, src_val in zip(op.else_body_vars, else_in_vals, strict=False):
        ctx.val_to_id[body_var.id] = ctx.val_to_id[src_val.id]

    # Per-arm yield-target ids (forward-declared; defined when the
    # arm's terminating YieldOp fires via OpCopyObject).
    has_carries = bool(op.results)
    then_yield_ids: list[str] = []
    else_yield_ids: list[str] = []
    type_ids: list[str] = []
    for i, res in enumerate(op.results):
        type_id = _emit_dtype(ctx.text, res.dtype, ctx)
        type_ids.append(type_id)
        then_yield_ids.append(ctx.text.alloc_id(f"if_then{i}"))
        else_yield_ids.append(ctx.text.alloc_id(f"if_else{i}"))

    merge_label = ctx.text.alloc_id("if_merge")
    then_label = ctx.text.alloc_id("if_then_lbl")
    else_label = ctx.text.alloc_id("if_else_lbl")
    then_tail = ctx.text.alloc_id("if_then_tail")
    else_tail = ctx.text.alloc_id("if_else_tail")

    ctx.text.emit_function(f"OpSelectionMerge {merge_label} None")
    ctx.text.emit_function(
        f"OpBranchConditional {pred_id} {then_label} {else_label}"
    )

    # ── Then arm ────────────────────────────────────────────────────
    ctx.text.emit_function(f"{then_label} = OpLabel")
    if has_carries:
        ctx.loop_yield_stack.append(
            list(zip(then_yield_ids, type_ids, strict=False))
        )
    try:
        for body_op in op.then_region.ops:
            _walk_op(body_op, ctx)
    finally:
        if has_carries:
            ctx.loop_yield_stack.pop()
    ctx.text.emit_function(f"OpBranch {then_tail}")
    ctx.text.emit_function(f"{then_tail} = OpLabel")
    ctx.text.emit_function(f"OpBranch {merge_label}")

    # ── Else arm ────────────────────────────────────────────────────
    ctx.text.emit_function(f"{else_label} = OpLabel")
    if has_carries:
        ctx.loop_yield_stack.append(
            list(zip(else_yield_ids, type_ids, strict=False))
        )
    try:
        for body_op in op.else_region.ops:
            _walk_op(body_op, ctx)
    finally:
        if has_carries:
            ctx.loop_yield_stack.pop()
    ctx.text.emit_function(f"OpBranch {else_tail}")
    ctx.text.emit_function(f"{else_tail} = OpLabel")
    ctx.text.emit_function(f"OpBranch {merge_label}")

    # ── Merge block ─────────────────────────────────────────────────
    ctx.text.emit_function(f"{merge_label} = OpLabel")
    for i, res in enumerate(op.results):
        phi_id = ctx.text.alloc_id(f"if_out{i}")
        ctx.val_to_id[res.id] = phi_id
        ctx.text.emit_function(
            f"{phi_id} = OpPhi {type_ids[i]} "
            f"{then_yield_ids[i]} {then_tail} "
            f"{else_yield_ids[i]} {else_tail}"
        )


def _visit_yield(op: YieldOp, ctx: _SpvCtx) -> None:
    """``YieldOp`` inside if/for body.

    Without carries (the if-without-carries path), this is a pure
    no-op: the region's terminator is consumed by the surrounding
    visitor's ``OpBranch``.

    Inside a ``ForLoopOp`` body with carries, ``_visit_for_loop``
    pushes a list of ``(target_id, type_id)`` slots onto
    ``ctx.loop_yield_stack``. We materialise each yielded value into
    its pre-allocated target id with ``OpCopyObject`` so the loop
    header's ``OpPhi`` (already emitted, referencing those ids as a
    forward declaration) resolves cleanly through ``spirv-as``.
    """
    if not op.operands:
        return
    if not ctx.loop_yield_stack:
        raise NotImplementedError(
            "_visit_yield: yielding values from a region without a "
            "surrounding for/while loop (likely if/else with carries) "
            "needs OpPhi merge — not yet wired. See PORTABILITY_PLAN §3.2."
        )
    targets = ctx.loop_yield_stack[-1]
    if len(targets) != len(op.operands):
        raise RuntimeError(
            f"_visit_yield: yielded {len(op.operands)} values vs "
            f"{len(targets)} carries"
        )
    for (target_id, type_id), val in zip(targets, op.operands, strict=False):
        src_id = ctx.val_to_id[val.id]
        ctx.text.emit_function(
            f"{target_id} = OpCopyObject {type_id} {src_id}"
        )


def _visit_for_loop(op: ForLoopOp, ctx: _SpvCtx) -> None:
    """Counted for-loop with optional loop-carried values.

    SPIR-V structured loop pattern (Vulkan compute):

        ;; <current block>
        OpBranch %preheader
        %preheader = OpLabel
        OpBranch %header

        %header = OpLabel
        %iv     = OpPhi <iv_t> %lo %preheader %iv_next %continue
        %ck     = OpPhi <ck_t> %ck_init %preheader %ck_next %continue   ; per carry
        %cond   = OpULessThan/OpSLessThan %iv %hi
        OpLoopMerge %merge %continue None
        OpBranchConditional %cond %body %merge

        %body = OpLabel
        ;; body emits ops. The terminating YieldOp materialises each
        ;; ck_next via OpCopyObject into the header's pre-declared id.
        OpBranch %continue

        %continue = OpLabel
        %iv_next  = OpIAdd <iv_t> %iv %step
        OpBranch %header

        %merge = OpLabel
        ;; for_op.results[i] is mapped to the header's OpPhi result —
        ;; on exit the OpPhi is the value-on-cond-false, which is the
        ;; final yielded carry value (or %ck_init if zero iterations).

    The OpPhi at the header references ``%iv_next`` and ``%ck_next``
    as forward references; spirv-as resolves them on its second pass.
    The preheader exists purely to give OpPhi a concrete predecessor
    label without having to track the surrounding visitor's "current
    block" label.
    """
    iv_dtype = op.attrs["iv_dtype"]
    iv_kind = _dtype_kind(iv_dtype)
    if iv_kind not in ("uint", "sint"):
        raise NotImplementedError(
            f"_visit_for_loop: only integer induction supported, got {iv_dtype!r}"
        )
    iv_type = _emit_dtype(ctx.text, iv_dtype, ctx)

    lo_id = ctx.val_to_id[op.lo.id]
    hi_id = ctx.val_to_id[op.hi.id]
    step_id = ctx.val_to_id[op.step.id]
    carried_init_ids = [ctx.val_to_id[c.id] for c in op.carried_in]

    # Allocate labels.
    preheader = ctx.text.alloc_id("loop_pre")
    header = ctx.text.alloc_id("loop_hdr")
    body_label = ctx.text.alloc_id("loop_body")
    cont_label = ctx.text.alloc_id("loop_cont")
    merge_label = ctx.text.alloc_id("loop_mrg")

    # Pre-allocate ids for forward refs in the header's OpPhi.
    iv_next_id = ctx.text.alloc_id("iv_next")

    # Per-carry: phi result id (== body-visible carry-in == loop result),
    # next id (yielded value, materialised by YieldOp visitor), and the
    # element type id.
    carry_phi_ids: list[str] = []
    # Pre-scan the body's terminator to detect when a carry's yielded
    # value is the result of a coopmat-typed op (``MmaOp`` /
    # ``LoadMatrixOp``). Those carries need an opaque
    # ``OpTypeCooperativeMatrixKHR`` for the OpPhi at the loop
    # header — a plain vec / array type would cause an OpPhi-vs-
    # MMA-result type mismatch (the GEMM accumulator pattern).
    body_term = op.body.terminator
    yielded_vals = list(body_term.operands) if body_term is not None else []

    def _coopmat_type_for_carry_yield(v):
        if v is None:
            return None
        prod = v.producer
        # Producer is an ``MmaOp`` or ``LoadMatrixOp``: the result is
        # a coopmat. Resolve its coopmat type id from the registry.
        if not isinstance(prod, (MmaOp, LoadMatrixOp)):
            return None
        from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore
        cfg = _BY_SHAPE_ID.get(prod.attrs.get("shape_id"))
        if cfg is None:
            return None
        which = "c" if isinstance(prod, MmaOp) else prod.attrs.get("which", "c")
        rows, cols, dtype = _coop_dims_for(cfg.shape, which)
        elem_t = _emit_dtype(ctx.text, dtype, ctx)
        if dtype is DType.BF16:
            ctx.text.add_capability("BFloat16CooperativeMatrixKHR")
        use = _COOPMAT_USE[which]
        _ensure_coopmat_caps(ctx)
        return ctx.text.type_coop_matrix(
            elem_t, scope=3, rows=rows, cols=cols, use=use,
        )

    carry_next_ids: list[str] = []
    carry_type_ids: list[str] = []
    carry_is_coopmat: list[bool] = []
    for i, body_var in enumerate(op.carried_body_vars):
        # Coopmat-yielding carry: the OpPhi must use the coopmat
        # type, and the init value (typically a vec_build zero pattern)
        # is splat-converted to a coopmat in the preheader below.
        coop_t = (
            _coopmat_type_for_carry_yield(yielded_vals[i])
            if i < len(yielded_vals) else None
        )
        if coop_t is not None:
            c_type = coop_t
            carry_is_coopmat.append(True)
        else:
            elem_t = _emit_dtype(ctx.text, body_var.dtype, ctx)
            if body_var.width > 1:
                c_type = ctx.text.type_vec(elem_t, body_var.width)
            else:
                c_type = elem_t
            carry_is_coopmat.append(False)
        phi_id = ctx.text.alloc_id(f"carry{i}")
        next_id = ctx.text.alloc_id(f"carry{i}_next")
        carry_phi_ids.append(phi_id)
        carry_next_ids.append(next_id)
        carry_type_ids.append(c_type)
        # Body-visible carry-in == header phi result.
        ctx.val_to_id[body_var.id] = phi_id
        # Loop result (visible after merge) == header phi result.
        ctx.val_to_id[op.results[i].id] = phi_id

    # Map the induction variable to its phi id.
    ctx.val_to_id[op.induction_var.id] = ctx.text.alloc_id("iv")
    iv_phi_id = ctx.val_to_id[op.induction_var.id]

    # ── Branch from current block into preheader, then header ────────
    ctx.text.emit_function(f"OpBranch {preheader}")
    ctx.text.emit_function(f"{preheader} = OpLabel")

    # Coopmat-typed carries need their (vec/array) init splat-
    # converted to a coopmat of the matching type. ``OpComposite
    # Construct`` on a CoopMat type with a single scalar component
    # is the SPV_KHR_cooperative_matrix idiom for "fill the matrix
    # with this scalar". For the GEMM accumulator pattern the init
    # is a vec_build of zeros — we extract one element (which is
    # also zero) and splat it into a zero coopmat. Only handles
    # the all-same-element init case; non-uniform init would need
    # a smem load detour.
    for i, (init_id, type_id, is_coop) in enumerate(zip(
        carried_init_ids, carry_type_ids, carry_is_coopmat,
        strict=False,
    )):
        if not is_coop:
            continue
        body_var = op.carried_body_vars[i]
        elem_dt = body_var.dtype
        elem_t = _emit_dtype(ctx.text, elem_dt, ctx)
        # Pick a scalar from the init's first slot (works when init
        # is all-same; if not, the splat is an approximation that
        # downstream MMA iterations will overwrite anyway).
        if body_var.width > 1:
            scalar_id = ctx.text.alloc_id(f"coop_init_scalar_{i}")
            ctx.text.emit_function(
                f"{scalar_id} = OpCompositeExtract {elem_t} {init_id} 0"
            )
        else:
            scalar_id = init_id
        new_init_id = ctx.text.alloc_id(f"coop_init_{i}")
        ctx.text.emit_function(
            f"{new_init_id} = OpCompositeConstruct {type_id} {scalar_id}"
        )
        # Replace the init id used by the OpPhi below.
        carried_init_ids[i] = new_init_id

    ctx.text.emit_function(f"OpBranch {header}")

    # ── Header: phi nodes + loop merge + cond branch ────────────────
    ctx.text.emit_function(f"{header} = OpLabel")
    ctx.text.emit_function(
        f"{iv_phi_id} = OpPhi {iv_type} {lo_id} {preheader} "
        f"{iv_next_id} {cont_label}"
    )
    for phi_id, type_id, init_id, next_id in zip(
        carry_phi_ids, carry_type_ids, carried_init_ids, carry_next_ids,
        strict=False,
    ):
        ctx.text.emit_function(
            f"{phi_id} = OpPhi {type_id} {init_id} {preheader} "
            f"{next_id} {cont_label}"
        )

    bool_t = ctx.text.type_bool()
    cond_id = ctx.text.alloc_id("loop_cond")
    cmp_op = "OpULessThan" if iv_kind == "uint" else "OpSLessThan"
    ctx.text.emit_function(
        f"{cond_id} = {cmp_op} {bool_t} {iv_phi_id} {hi_id}"
    )
    ctx.text.emit_function(
        f"OpLoopMerge {merge_label} {cont_label} None"
    )
    ctx.text.emit_function(
        f"OpBranchConditional {cond_id} {body_label} {merge_label}"
    )

    # ── Body block ──────────────────────────────────────────────────
    ctx.text.emit_function(f"{body_label} = OpLabel")
    # Push the carry yield target stack frame so the body's YieldOp
    # materialises into the pre-allocated ids.
    ctx.loop_yield_stack.append(
        list(zip(carry_next_ids, carry_type_ids, strict=False))
    )
    try:
        for body_op in op.body.ops:
            _walk_op(body_op, ctx)
    finally:
        ctx.loop_yield_stack.pop()
    ctx.text.emit_function(f"OpBranch {cont_label}")

    # ── Continue block: increment iv ────────────────────────────────
    ctx.text.emit_function(f"{cont_label} = OpLabel")
    ctx.text.emit_function(
        f"{iv_next_id} = OpIAdd {iv_type} {iv_phi_id} {step_id}"
    )
    ctx.text.emit_function(f"OpBranch {header}")

    # ── Merge block ─────────────────────────────────────────────────
    ctx.text.emit_function(f"{merge_label} = OpLabel")


def _ensure_buffer_var(tensor: GlobalTensor, binding_index: int,
                       ctx: _SpvCtx) -> tuple[str, str]:
    """Declare the storage-buffer variable for ``tensor`` at
    ``binding_index`` if not already present. Returns
    ``(buffer_var_id, elem_pointer_type_id)`` for the load/store
    visitors to use."""
    # Key the per-buffer caches on the param's id, not the tensor
    # instance — every ``GlobalTensor.view()`` returns a fresh
    # tensor object that shares the parent's ``param``. Without this,
    # any tile-load through a sub-tensor falls off the cache and
    # blows up at the ``tensor_to_binding`` lookup.
    tid = id(tensor.param)
    if tid in ctx.tensor_to_var:
        return ctx.tensor_to_var[tid], ctx.tensor_to_elem_ptr[tid]

    elem_type = _emit_dtype(ctx.text, tensor.dtype, ctx)
    elem_bytes = _dtype_byte_width(tensor.dtype)
    rta = ctx.text.type_runtime_array(elem_type, stride_bytes=elem_bytes)
    struct = ctx.text.type_struct(rta)
    # Block decoration on the struct + Offset 0 on its single member —
    # both required for a storage buffer block.
    ctx.text.add_decoration(f"OpDecorate {struct} Block")
    ctx.text.add_decoration(f"OpMemberDecorate {struct} 0 Offset 0")
    ptr_struct = ctx.text.type_pointer("StorageBuffer", struct)
    var_id = ctx.text.alloc_id(f"buf_{binding_index}")
    ctx.text.add_type_line(f"{var_id} = OpVariable {ptr_struct} StorageBuffer")
    ctx.text.add_decoration(f"OpDecorate {var_id} DescriptorSet 0")
    ctx.text.add_decoration(f"OpDecorate {var_id} Binding {binding_index}")

    # Pointer to one element — used by AccessChain at every
    # load/store site.
    elem_ptr = ctx.text.type_pointer("StorageBuffer", elem_type)

    ctx.tensor_to_var[tid] = var_id
    ctx.tensor_to_elem_ptr[tid] = elem_ptr
    ctx.tensor_to_elem_type[tid] = elem_type
    return var_id, elem_ptr


def _flatten_global_index(
    indices: tuple, tensor: "GlobalTensor", ctx: _SpvCtx,
) -> str:
    """Compute a flat element offset into a ``GlobalTensor``, threading
    its ``view``/``tile`` offsets and the parent's per-axis
    ``stride`` through the address arithmetic.

    The IR's ``GlobalTensor.view(...)`` returns a sub-tensor whose
    ``shape`` reflects the carved tile but whose ``stride`` and base
    offsets are inherited from the parent. The previous helper
    (``_flatten_index``) computed strides from the tile's ``shape``
    and ignored the offsets — correct for top-level ``GlobalTensor``
    but wrong for any tile loaded inside a kernel that does
    ``g.A.view(row=m_base, col=k_col)``. The gemm cohort hits this
    on every gmem→smem produce.

    Address shape (for 2-D tensors, the common case):

        flat = (static_row + dyn_row + i_row) * stride[0]
             + (static_col + dyn_col + i_col) * stride[1]

    1-D tensors use the same formula with axis 0 only and no
    column offset.
    """
    u32 = ctx.text.type_int(32, signed=False)
    static_offsets = (
        getattr(tensor, "static_row_offset", 0),
        getattr(tensor, "static_col_offset", 0),
    )
    dyn_offsets = (
        getattr(tensor, "dyn_row_offset", None),
        getattr(tensor, "dyn_col_offset", None),
    )
    strides = tensor.stride

    flat: str | None = None
    for axis, idx_v in enumerate(indices):
        if axis >= len(strides):
            # Trailing axes without strides — treat as 1-element-wide
            # (collapse). Should not happen for well-formed tensors.
            continue
        stride_val = int(strides[axis])
        # Compute (static + dyn + idx) for this axis.
        idx_id = ctx.val_to_id[idx_v.id]
        # Add static offset (compile-time int) when non-zero.
        s_off = static_offsets[axis] if axis < len(static_offsets) else 0
        if s_off:
            const_id = ctx.text.const_uint(int(s_off))
            new_id = ctx.text.alloc_id(f"gidx_s{axis}")
            ctx.text.emit_function(
                f"{new_id} = OpIAdd {u32} {idx_id} {const_id}"
            )
            idx_id = new_id
        # Add dyn offset (runtime Value) when present.
        d_off = dyn_offsets[axis] if axis < len(dyn_offsets) else None
        if d_off is not None:
            d_id = ctx.val_to_id[d_off.id]
            new_id = ctx.text.alloc_id(f"gidx_d{axis}")
            ctx.text.emit_function(
                f"{new_id} = OpIAdd {u32} {idx_id} {d_id}"
            )
            idx_id = new_id
        # Multiply by axis stride. Stride 1 → no mul.
        if stride_val == 1:
            term = idx_id
        else:
            stride_const = ctx.text.const_uint(stride_val)
            term = ctx.text.alloc_id(f"gflat_mul{axis}")
            ctx.text.emit_function(
                f"{term} = OpIMul {u32} {idx_id} {stride_const}"
            )
        if flat is None:
            flat = term
        else:
            new_flat = ctx.text.alloc_id(f"gflat_add{axis}")
            ctx.text.emit_function(
                f"{new_flat} = OpIAdd {u32} {flat} {term}"
            )
            flat = new_flat
    assert flat is not None
    # Scalar ``dyn_offset`` (set by ``view(dyn_offset=…)`` /
    # ``warp_lane_view``) is a flat offset added to the final
    # address — used by per-warp staging smem in
    # ``store_acc(per_warp=True)``. Drop = silent miscompile (each
    # warp writes the same staging smem range and overwrites the
    # others). Same class as the warp_dyn_offset coopmat fix
    # (commit f0ade34) — different code path.
    scalar_dyn = getattr(tensor, "dyn_offset", None)
    if scalar_dyn is not None:
        d_id = ctx.val_to_id[scalar_dyn.id]
        new_flat = ctx.text.alloc_id("gflat_dyn")
        ctx.text.emit_function(
            f"{new_flat} = OpIAdd {u32} {flat} {d_id}"
        )
        flat = new_flat
    return flat


def _flatten_index(indices: tuple, shape: tuple, ctx: _SpvCtx) -> str:
    """Compute a row-major flat index from N-D indices + shape.

    Returns the SSA id of the flat-index ``OpIAdd``. For 1-D this
    is just the single index; for N-D, emits the ``i*S2 + j*S3 +
    k`` chain. ``shape`` is the IR's declared shape (post-pad).

    For ``GlobalTensor`` accesses that need offset/stride awareness
    (any kernel that uses ``view`` / ``tile`` to carve out
    sub-tensors), use ``_flatten_global_index`` instead.
    """
    if len(indices) == 1:
        return ctx.val_to_id[indices[0].id]
    u32 = ctx.text.type_int(32, signed=False)
    # Compute strides right-to-left.
    flat: str | None = None
    stride = 1
    rev_strides = []
    for s in reversed(shape):
        rev_strides.append(stride)
        stride *= s
    rev_strides.reverse()
    for idx_v, s in zip(indices, rev_strides, strict=False):
        idx_id = ctx.val_to_id[idx_v.id]
        if s == 1:
            term = idx_id
        else:
            stride_const = ctx.text.const_uint(s)
            term = ctx.text.alloc_id("flat_mul")
            ctx.text.emit_function(
                f"{term} = OpIMul {u32} {idx_id} {stride_const}"
            )
        if flat is None:
            flat = term
        else:
            new_flat = ctx.text.alloc_id("flat_add")
            ctx.text.emit_function(
                f"{new_flat} = OpIAdd {u32} {flat} {term}"
            )
            flat = new_flat
    assert flat is not None
    return flat


def _add_dyn_offset(flat: str, tensor, ctx: _SpvCtx) -> str:
    """Append ``tensor.warp_dyn_offset`` (or warp-uniform
    ``dyn_offset``) to a flat index when present.

    SharedRegion's ``dyn_offset`` field is overloaded across two
    use cases:

      * ``view(dyn_offset=warp_id * stride)`` — warp-uniform (e.g.
        ``store_acc(per_warp=True)`` staging).
      * ``warp_lane_view(...)`` — sets BOTH ``dyn_offset =
        warp_off + per_lane_offset`` AND ``warp_dyn_offset =
        warp_off`` (warp-only).

    For NON-coopmat smem accesses (regular ``OpStore`` / ``OpLoad``
    / vec_load / vec_store), we want the warp-uniform component
    only; per-lane PTX-style offsets are wrong here because they
    were intended for ldmatrix and they end up zeroing half the
    cols of attn's output (the ``group_id``-parity-looking pattern).
    Prefer ``warp_dyn_offset`` when set; else fall back to
    ``dyn_offset`` (warp-uniform by construction in non-warp-lane-
    view cases).
    """
    warp_only = getattr(tensor, "warp_dyn_offset", None)
    if warp_only is not None:
        u32 = ctx.text.type_int(32, signed=False)
        d_id = ctx.val_to_id[warp_only.id]
        new_flat = ctx.text.alloc_id("flat_warp_dyn")
        ctx.text.emit_function(
            f"{new_flat} = OpIAdd {u32} {flat} {d_id}"
        )
        return new_flat
    scalar_dyn = getattr(tensor, "dyn_offset", None)
    if scalar_dyn is None:
        return flat
    u32 = ctx.text.type_int(32, signed=False)
    d_id = ctx.val_to_id[scalar_dyn.id]
    new_flat = ctx.text.alloc_id("flat_dyn")
    ctx.text.emit_function(
        f"{new_flat} = OpIAdd {u32} {flat} {d_id}"
    )
    return new_flat


def _visit_load(op: LoadOp, ctx: _SpvCtx) -> None:
    (out,) = op.results
    tensor = op.attrs["tensor"]
    if isinstance(tensor, GlobalTensor):
        binding = ctx.tensor_to_binding[id(tensor.param)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        elem_type = ctx.tensor_to_elem_type[id(tensor.param)]
        zero = ctx.text.const_uint(0)
        idx_id = _flatten_global_index(tuple(op.operands), tensor, ctx)
        chain_id = ctx.text.alloc_id("chain")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {zero} {idx_id}"
        )
        res_id = ctx.text.alloc_id("ld")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = OpLoad {elem_type} {chain_id}")
        return

    if isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_load: SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp was visited"
            )
        var_id, elem_type, elem_ptr, _n = rec
        idx_id = _flatten_index(tuple(op.operands), tensor.shape, ctx)
        idx_id = _add_dyn_offset(idx_id, tensor, ctx)
        chain_id = ctx.text.alloc_id("smem_chain")
        # Workgroup-class arrays are ``OpVariable Workgroup
        # OpTypeArray T n`` — no enclosing struct — so the access
        # chain takes one fewer index than the StorageBuffer path
        # (no ``%uint_0`` for the struct-member dimension).
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {idx_id}"
        )
        res_id = ctx.text.alloc_id("smem_ld")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = OpLoad {elem_type} {chain_id}")
        return

    raise NotImplementedError(
        f"_visit_load: tensor type {type(tensor).__name__} not wired"
    )


def _visit_store(op: StoreOp, ctx: _SpvCtx) -> None:
    tensor = op.attrs["tensor"]
    if isinstance(tensor, GlobalTensor):
        binding = ctx.tensor_to_binding[id(tensor.param)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        value_id = ctx.val_to_id[op.operands[0].id]
        zero = ctx.text.const_uint(0)
        idx_id = _flatten_global_index(tuple(op.operands[1:]), tensor, ctx)
        chain_id = ctx.text.alloc_id("chain")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {zero} {idx_id}"
        )
        ctx.text.emit_function(f"OpStore {chain_id} {value_id}")
        return

    if isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_store: SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp was visited"
            )
        var_id, _elem_type, elem_ptr, _n = rec
        value_id = ctx.val_to_id[op.operands[0].id]
        idx_id = _flatten_index(tuple(op.operands[1:]), tensor.shape, ctx)
        idx_id = _add_dyn_offset(idx_id, tensor, ctx)
        chain_id = ctx.text.alloc_id("smem_chain")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {idx_id}"
        )
        ctx.text.emit_function(f"OpStore {chain_id} {value_id}")
        return

    raise NotImplementedError(
        f"_visit_store: tensor type {type(tensor).__name__} not wired"
    )


# ``AtomicRmwOp.attrs['op']`` → SPIR-V atomic opcode. Splits by signed
# vs unsigned for min/max — the IR carries the dtype on the value.
_ATOMIC_OP_BY_KIND: dict[str, str] = {
    "add": "OpAtomicIAdd",
    "and": "OpAtomicAnd",
    "or": "OpAtomicOr",
    "xor": "OpAtomicXor",
    "exch": "OpAtomicExchange",
}


def _visit_atomic_rmw(op: AtomicRmwOp, ctx: _SpvCtx) -> None:
    """Atomic read-modify-write on a ``GlobalTensor`` slot.

    SPIR-V opcode shape: ``%result = OpAtomic<Op> %T %ptr %scope
    %semantics %value`` — except ``OpAtomicLoad`` / ``OpAtomicStore``
    which we don't need here. Result is the *original* value at the
    slot, matching the IR's ``atomic_rmw`` contract.

    Memory scope: ``Device`` (1) — the typical compute-shader choice
    when the atomic is meant to be visible across workgroups (the
    moe_router pattern, where multiple WGs race for an expert slot).
    Memory semantics: ``Relaxed`` (0) — atomics emitted from quark IR
    don't carry an acquire/release barrier today; ordering is the
    caller's responsibility (a ``barrier()`` op pairs with the rmw
    when needed).
    """
    (out,) = op.results
    tensor = op.attrs["tensor"]
    kind = op.attrs["op"]

    if not isinstance(tensor, GlobalTensor):
        raise NotImplementedError(
            f"_visit_atomic_rmw: tensor type "
            f"{type(tensor).__name__} not wired (only GlobalTensor today)"
        )

    binding = ctx.tensor_to_binding[id(tensor.param)]
    var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)

    value_id = ctx.val_to_id[op.operands[0].id]
    zero = ctx.text.const_uint(0)
    idx_id = _flatten_global_index(tuple(op.operands[1:]), tensor, ctx)
    chain_id = ctx.text.alloc_id("atomic_chain")
    ctx.text.emit_function(
        f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {zero} {idx_id}"
    )

    type_id = _emit_dtype(ctx.text, out.dtype, ctx)
    res_id = ctx.text.alloc_id(f"atomic_{kind}")
    ctx.val_to_id[out.id] = res_id

    if kind == "min":
        spv_op = "OpAtomicSMin" if _dtype_kind(out.dtype) == "sint" else "OpAtomicUMin"
    elif kind == "max":
        spv_op = "OpAtomicSMax" if _dtype_kind(out.dtype) == "sint" else "OpAtomicUMax"
    else:
        spv_op = _ATOMIC_OP_BY_KIND.get(kind)
        if spv_op is None:
            raise NotImplementedError(
                f"_visit_atomic_rmw: op={kind!r} not yet wired"
            )

    # Scope = Device (1), Memory semantics = Relaxed (0). Both are
    # ``OpConstant uint``; cache them so repeated atomics don't
    # produce duplicate constants.
    scope_id = ctx.text.const_uint(1)
    semantics_id = ctx.text.const_uint(0)

    ctx.text.emit_function(
        f"{res_id} = {spv_op} {type_id} {chain_id} {scope_id} "
        f"{semantics_id} {value_id}"
    )


# ─── Convert / Math / Select ──────────────────────────────────────


def _visit_convert(op: ConvertOp, ctx: _SpvCtx) -> None:
    """Explicit dtype conversion. Picks the right SPIR-V op based
    on (src dtype kind, dst dtype kind):

      * float → float (different widths) → ``OpFConvert``
      * int → float → ``OpConvertSToF`` / ``OpConvertUToF``
      * float → int → ``OpConvertFToS`` / ``OpConvertFToU``
      * int → int (different widths or signedness) → ``OpSConvert``
        / ``OpUConvert`` / ``OpBitcast`` for same-width sign change

    Same-dtype conversions short-circuit (no-op) — the kernel author
    occasionally emits a redundant cast, no need to round-trip.

    Rounding mode: SPIR-V's GLCompute profile doesn't expose
    per-instruction rounding decorations the way PTX does. ``rn``
    (round-to-nearest-even) is the spec default and what most
    drivers emit; non-RN rounding requires the
    ``RoundingModeRTE/RTZ`` execution mode + per-instruction
    decoration (``OpDecorate ... FPRoundingMode RTE``). Today we
    accept any rounding attr but only emit the default — non-RN
    callers should expect drift; track follow-up if a kernel needs
    deterministic non-RN.
    """
    (out,) = op.results
    src_v = op.operands[0]
    src_dt = src_v.dtype
    dst_dt = out.dtype
    src_id = ctx.val_to_id[src_v.id]
    if src_dt is dst_dt:
        ctx.val_to_id[out.id] = src_id
        return
    dst_t = _emit_dtype(ctx.text, dst_dt, ctx)
    src_kind = _dtype_kind(src_dt)
    dst_kind = _dtype_kind(dst_dt)
    if src_kind == "float" and dst_kind == "float":
        spv_op = "OpFConvert"
    elif src_kind == "uint" and dst_kind == "float":
        spv_op = "OpConvertUToF"
    elif src_kind == "sint" and dst_kind == "float":
        spv_op = "OpConvertSToF"
    elif src_kind == "float" and dst_kind == "uint":
        spv_op = "OpConvertFToU"
    elif src_kind == "float" and dst_kind == "sint":
        spv_op = "OpConvertFToS"
    elif src_kind == "uint" and dst_kind == "uint":
        spv_op = "OpUConvert"
    elif src_kind == "sint" and dst_kind == "sint":
        spv_op = "OpSConvert"
    elif src_kind == "uint" and dst_kind == "sint" and _dtype_byte_width(src_dt) == _dtype_byte_width(dst_dt):
        spv_op = "OpBitcast"  # same-width sign change
    elif src_kind == "sint" and dst_kind == "uint" and _dtype_byte_width(src_dt) == _dtype_byte_width(dst_dt):
        spv_op = "OpBitcast"
    else:
        raise NotImplementedError(
            f"_visit_convert: {src_dt!r} → {dst_dt!r} not yet wired"
        )
    res_id = ctx.text.alloc_id(f"cvt_{src_dt.value}_{dst_dt.value}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = {spv_op} {dst_t} {src_id}")


def _visit_split_b32(op: SplitB32Op, ctx: _SpvCtx) -> None:
    """``SplitB32Op``: B32 → (B16 lo, B16 hi).

    SPIR-V doesn't have a "split into halves" opcode, so we emit:
        b32_u32 = OpBitcast u32 b32_in    ; raw bits as u32
        lo_u32  = OpBitwiseAnd u32 b32_u32 0xFFFF
        hi_u32  = OpShiftRightLogical u32 b32_u32 16
        lo_b16  = OpUConvert u16 lo_u32
        hi_b16  = OpUConvert u16 hi_u32

    The ``OpUConvert`` is the right narrowing op when both src and
    dst are unsigned ints (the IR carries B16 / B32 as raw bits;
    we model them as unsigned). Used by the ``fma_bf16x2``
    legalization expansion to peel a packed bf16×2 register.
    """
    lo, hi = op.results
    src_id = ctx.val_to_id[op.operands[0].id]
    u32 = ctx.text.type_int(32, signed=False)
    u16 = _emit_dtype(ctx.text, DType.B16, ctx)
    mask = ctx.text.const_uint(0xFFFF)
    sixteen = ctx.text.const_uint(16)

    # As-uint32 reinterpretation. B32 lowers to OpTypeInt 32 already,
    # so OpBitcast is a no-op (same type) — but we emit it to keep
    # the chain explicit and let spirv-as fold the redundancy.
    asu32 = ctx.text.alloc_id("b32_as_u32")
    ctx.text.emit_function(f"{asu32} = OpBitcast {u32} {src_id}")

    lo_u32 = ctx.text.alloc_id("split_lo_u32")
    ctx.text.emit_function(
        f"{lo_u32} = OpBitwiseAnd {u32} {asu32} {mask}"
    )
    hi_u32 = ctx.text.alloc_id("split_hi_u32")
    ctx.text.emit_function(
        f"{hi_u32} = OpShiftRightLogical {u32} {asu32} {sixteen}"
    )

    lo_id = ctx.text.alloc_id("split_lo")
    hi_id = ctx.text.alloc_id("split_hi")
    ctx.val_to_id[lo.id] = lo_id
    ctx.val_to_id[hi.id] = hi_id
    ctx.text.emit_function(f"{lo_id} = OpUConvert {u16} {lo_u32}")
    ctx.text.emit_function(f"{hi_id} = OpUConvert {u16} {hi_u32}")


def _visit_merge_b32(op: MergeB32Op, ctx: _SpvCtx) -> None:
    """``MergeB32Op``: (B16 lo, B16 hi) → B32.

    Inverse of ``SplitB32Op``. Widen each B16 to u32 first (so the
    shift on ``hi`` doesn't lose bits), or-merge, bitcast to B32.
    """
    (out,) = op.results
    lo_id = ctx.val_to_id[op.operands[0].id]
    hi_id = ctx.val_to_id[op.operands[1].id]
    u32 = ctx.text.type_int(32, signed=False)
    sixteen = ctx.text.const_uint(16)

    lo_u32 = ctx.text.alloc_id("merge_lo_u32")
    hi_u32 = ctx.text.alloc_id("merge_hi_u32")
    ctx.text.emit_function(f"{lo_u32} = OpUConvert {u32} {lo_id}")
    ctx.text.emit_function(f"{hi_u32} = OpUConvert {u32} {hi_id}")
    hi_shifted = ctx.text.alloc_id("merge_hi_shl")
    ctx.text.emit_function(
        f"{hi_shifted} = OpShiftLeftLogical {u32} {hi_u32} {sixteen}"
    )
    merged = ctx.text.alloc_id("merge_or")
    ctx.text.emit_function(
        f"{merged} = OpBitwiseOr {u32} {lo_u32} {hi_shifted}"
    )
    res_dt = _emit_dtype(ctx.text, out.dtype, ctx)
    res_id = ctx.text.alloc_id("merge_b32")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpBitcast {res_dt} {merged}")


def _visit_bitcast(op: BitcastOp, ctx: _SpvCtx) -> None:
    """Reinterpret-cast: ``OpBitcast`` to the destination type. SPIR-V
    requires src and dst to have the same total bit width; the IR's
    own ``BitcastOp.__post_init__`` enforces that. Used by RNG /
    layout-shuffle kernels (e.g. ``moe_router_correct`` flips between
    S32 and U32 for shifts)."""
    (out,) = op.results
    dst_dt = op.attrs["dst_dtype"]
    dst_t = _emit_dtype(ctx.text, dst_dt, ctx)
    src_id = ctx.val_to_id[op.operands[0].id]
    res_id = ctx.text.alloc_id(f"bitcast_{dst_dt.value}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpBitcast {dst_t} {src_id}")


def _dtype_kind(dt: DType) -> str:
    if dt in (DType.F16, DType.BF16, DType.F32, DType.F64):
        return "float"
    if dt in (DType.U8, DType.U16, DType.U32, DType.U64, DType.PRED):
        return "uint"
    if dt in (DType.S8, DType.S16, DType.S32, DType.S64):
        return "sint"
    raise NotImplementedError(f"_dtype_kind: unknown {dt!r}")


# ``MathOp.kind`` → GLSL.std.450 symbolic instruction name. The
# spirv-as assembler accepts the symbolic name (not the numeric
# instruction code) for OpExtInst on GLSL.std.450. Reference:
# https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
#
# Every kind here maps to a single ExtInst; the ``_approx`` variants
# share the same instruction (Vulkan's transcendentals are always
# approximate per spec — no separate precise/approximate path). The
# ULP-slack risk this introduces is tracked in PORTABILITY_PLAN §3.7
# v1 ("OpExtInst GLSL.std.450 ULP slack"); validate per-kernel
# cos_sim against the CUDA reference during rollout.
_MATH_KIND_TO_GLSL_INSTR: dict[str, str | None] = {
    "rcp": None,          # not in GLSL.std.450 — emit OpFDiv 1.0/x
    "rcp_approx": None,
    "rsqrt": "InverseSqrt",
    "rsqrt_approx": "InverseSqrt",
    "sqrt": "Sqrt",
    "sqrt_approx": "Sqrt",
    "exp": "Exp",
    "exp_approx": "Exp",
    "exp2": "Exp2",
    "ex2_approx": "Exp2",
    "log2": "Log2",
    "log2_approx": "Log2",
    "sin": "Sin",
    "cos": "Cos",
    "tanh": "Tanh",
}


def _visit_math(op: MathOp, ctx: _SpvCtx) -> None:
    """Transcendental / approximate-math ops via GLSL.std.450
    extended instructions.

    ``rcp`` has no GLSL.std.450 entry — emit ``OpFDiv 1.0 / x``
    instead. The hardware reciprocal is what the driver will pick
    when GLSL source uses ``1.0 / x``; same lowering, no perf loss.
    """
    (out,) = op.results
    kind = op.attrs["kind"]
    if kind not in _MATH_KIND_TO_GLSL_INSTR:
        raise NotImplementedError(
            f"_visit_math: kind={kind!r} not yet wired"
        )
    glsl_name = _MATH_KIND_TO_GLSL_INSTR[kind]
    src_id = ctx.val_to_id[op.operands[0].id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)

    if glsl_name is None:  # rcp — emit OpFDiv 1.0 / x
        if out.dtype is not DType.F32:
            raise NotImplementedError(
                f"_visit_math(rcp): only f32 wired today, got {out.dtype!r}"
            )
        one = ctx.text.const_float(1.0)
        res_id = ctx.text.alloc_id("rcp")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = OpFDiv {dst_t} {one} {src_id}")
        return

    glsl_id = ctx.text.import_ext_inst("GLSL.std.450")
    res_id = ctx.text.alloc_id(kind)
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpExtInst {dst_t} {glsl_id} {glsl_name} {src_id}"
    )


def _visit_select(op: SelectOp, ctx: _SpvCtx) -> None:
    """Ternary select. SPIR-V's ``OpSelect`` directly handles the
    ``pred ? t : f`` semantics — including PRED/Bool selectors,
    which Vulkan SPIR-V 1.4+ permits for arbitrary types."""
    (out,) = op.results
    pred, t, f = op.operands
    pred_id = ctx.val_to_id[pred.id]
    t_id = ctx.val_to_id[t.id]
    f_id = ctx.val_to_id[f.id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)
    res_id = ctx.text.alloc_id("sel")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpSelect {dst_t} {pred_id} {t_id} {f_id}")


# ─── Threadgroup memory + sync ────────────────────────────────────


def _visit_smem_alloc(op: SmemAllocOp, ctx: _SpvCtx) -> None:
    """Allocate an ``OpVariable Workgroup`` for the smem region.

    SPIR-V's ``Workgroup`` storage class is the analogue of CUDA's
    ``__shared__`` / MSL's ``threadgroup``. The variable must be
    declared at module scope (not inside the function body) — the
    type-line emit path lands the variable in the type section
    automatically.

    Layout: a flat ``OpTypeArray`` sized to the total element count.
    N-D shapes flatten to 1D for SPIR-V; the
    ``_visit_load`` / ``_visit_store`` paths compute the row-major
    flat index from N-D indices. Pad bytes from the ``pad`` attr
    are folded into the per-row stride at index time.

    Smem aliasing (the framework's ``smem_layout`` pass picks
    overlapping offsets for disjoint-lifetime regions to fit a
    bigger working set into the same byte budget) is deferred — for
    v1 each ``SmemAllocOp`` gets its own variable. The waste is
    real but small at the kernel sizes Battlemage targets.
    """
    (backing,) = op.results
    elem_type = _emit_dtype(ctx.text, op.dtype, ctx)
    n_elems = 1
    for s in op.shape:
        n_elems *= s
    n_const = ctx.text.const_uint(n_elems)
    arr_id = ctx.text.alloc_id(f"smem_{op.name}_arr")
    ctx.text.add_type_line(f"{arr_id} = OpTypeArray {elem_type} {n_const}")
    ptr_arr = ctx.text.type_pointer("Workgroup", arr_id)
    var_id = ctx.text.alloc_id(f"smem_{op.name}")
    ctx.text.add_type_line(f"{var_id} = OpVariable {ptr_arr} Workgroup")
    elem_ptr = ctx.text.type_pointer("Workgroup", elem_type)
    ctx.smem_allocs[backing.id] = (var_id, elem_type, elem_ptr, n_elems)


def _visit_barrier(op: BarrierOp, ctx: _SpvCtx) -> None:
    """``OpControlBarrier execution memory semantics``.

    Maps the IR scope to the corresponding SPIR-V scope id:
      * ``"block"`` → Workgroup (2) — full threadgroup barrier
      * ``"subgroup"`` → Subgroup (3)
      * ``"system"`` → Device (1)

    Memory semantics: ``AcquireRelease | WorkgroupMemory`` for block
    scope (matches CUDA's ``__syncthreads`` / Metal's
    ``threadgroup_barrier(mem_threadgroup)`` semantics — release prior
    smem writes, acquire subsequent smem reads). Add
    ``SubgroupMemory`` for subgroup scope.
    """
    scope = op.attrs.get("scope", "block")
    # SPIR-V scope constants. Use OpConstant uint values that
    # spirv-as recognises as named scopes.
    scope_const = {"block": 2, "subgroup": 3, "system": 1}.get(scope)
    if scope_const is None:
        raise NotImplementedError(f"_visit_barrier: scope={scope!r}")
    # Memory semantics: 0x8 = AcquireRelease (Vulkan-required), plus
    # 0x100 = WorkgroupMemory or 0x80 = SubgroupMemory depending on
    # which smem the barrier protects.
    if scope == "block":
        mem_sem = 0x8 | 0x100  # AcquireRelease | WorkgroupMemory
    elif scope == "subgroup":
        mem_sem = 0x8 | 0x80   # AcquireRelease | SubgroupMemory
    else:
        mem_sem = 0x8

    exec_id = ctx.text.const_uint(scope_const)
    mem_id = ctx.text.const_uint(scope_const)
    sem_id = ctx.text.const_uint(mem_sem)
    ctx.text.emit_function(f"OpControlBarrier {exec_id} {mem_id} {sem_id}")


# ─── Subgroup ops ─────────────────────────────────────────────────


# (op kind, dtype kind) → SPIR-V opcode name. ``Reduce`` operation
# is the one we want for kernel reductions (every-lane gets the
# scalar result of the full-subgroup reduction).
_SUBGROUP_REDUCE_TO_SPV: dict[tuple[str, str], str] = {
    ("sum", "float"): "OpGroupNonUniformFAdd",
    ("sum", "uint"): "OpGroupNonUniformIAdd",
    ("sum", "sint"): "OpGroupNonUniformIAdd",
    ("max", "float"): "OpGroupNonUniformFMax",
    ("max", "uint"): "OpGroupNonUniformUMax",
    ("max", "sint"): "OpGroupNonUniformSMax",
    ("min", "float"): "OpGroupNonUniformFMin",
    ("min", "uint"): "OpGroupNonUniformUMin",
    ("min", "sint"): "OpGroupNonUniformSMin",
    ("and", "uint"): "OpGroupNonUniformBitwiseAnd",
    ("and", "sint"): "OpGroupNonUniformBitwiseAnd",
    ("or", "uint"): "OpGroupNonUniformBitwiseOr",
    ("or", "sint"): "OpGroupNonUniformBitwiseOr",
}


def _visit_shuffle(op: ShuffleOp, ctx: _SpvCtx) -> None:
    """Cross-lane shuffle within a subgroup.

    Maps the IR's shuffle ``kind`` to the SPIR-V op:
      * ``"bfly"`` / ``"xor"`` → ``OpGroupNonUniformShuffleXor`` —
        butterfly reduction's per-step communication, lane i talks to
        lane (i ^ param).
      * ``"up"`` → ``OpGroupNonUniformShuffleUp`` — receive from
        lane (i - param), wrapping around to 0 for low-lane callers.
      * ``"down"`` → ``OpGroupNonUniformShuffleDown`` — receive from
        lane (i + param).
      * ``"idx"`` → ``OpGroupNonUniformShuffle`` — receive from
        lane = param (broadcast / pick a specific lane).

    Capability: ``GroupNonUniformShuffle``. Subgroup scope (3) for
    every kind — these ops are subgroup-local by definition.
    """
    (out,) = op.results
    kind = op.attrs["kind"]
    param = op.attrs["param"]
    src_id = ctx.val_to_id[op.operands[0].id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)

    spv_op = {
        "bfly": "OpGroupNonUniformShuffleXor",
        "xor": "OpGroupNonUniformShuffleXor",
        "up": "OpGroupNonUniformShuffleUp",
        "down": "OpGroupNonUniformShuffleDown",
        "idx": "OpGroupNonUniformShuffle",
    }.get(kind)
    if spv_op is None:
        raise NotImplementedError(f"_visit_shuffle: kind={kind!r}")
    ctx.text.add_capability("GroupNonUniformShuffle")

    sg_scope = ctx.text.const_uint(3)  # Subgroup scope
    # ``param`` may be a Python int (constant lane offset / xor mask)
    # or a runtime IR Value (computed shuffle target). Both forms
    # need a SPIR-V Value reference at the SPIR-V site.
    from quark.ir.value import Value as _Value
    if isinstance(param, _Value):
        param_id = ctx.val_to_id[param.id]
    else:
        param_id = ctx.text.const_uint(int(param))
    res_id = ctx.text.alloc_id(f"shuffle_{kind}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = {spv_op} {dst_t} {sg_scope} {src_id} {param_id}"
    )


def _visit_subgroup_reduce(op: SubgroupReduceOp, ctx: _SpvCtx) -> None:
    """Cross-lane reduction within a subgroup.

    Maps the ``op`` attr to the right ``OpGroupNonUniform*`` opcode
    based on the operand dtype (float/uint/sint). Always uses the
    ``Reduce`` variant — every lane in the subgroup receives the
    same scalar result.

    Capability needed: ``GroupNonUniformArithmetic`` (Vulkan 1.1+
    core; Battlemage advertises it per the §3.1 probe).
    """
    (out,) = op.results
    src_v = op.operands[0]
    kind = op.attrs["op"]
    dt_kind = _dtype_kind(src_v.dtype)
    spv_op = _SUBGROUP_REDUCE_TO_SPV.get((kind, dt_kind))
    if spv_op is None:
        raise NotImplementedError(
            f"_visit_subgroup_reduce: op={kind!r} dtype_kind={dt_kind!r}"
        )
    ctx.text.add_capability("GroupNonUniformArithmetic")
    src_id = ctx.val_to_id[src_v.id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)
    # Subgroup scope (3) for the execution scope of the reduction.
    sg_scope = ctx.text.const_uint(3)
    res_id = ctx.text.alloc_id(f"sgreduce_{kind}")
    ctx.val_to_id[out.id] = res_id
    # ``Reduce`` operation — same result on every lane.
    ctx.text.emit_function(
        f"{res_id} = {spv_op} {dst_t} {sg_scope} Reduce {src_id}"
    )


# ─── Vec ops ───────────────────────────────────────────────────────


def _visit_vec_load(op: VecLoadOp, ctx: _SpvCtx) -> None:
    """Vector load — emits N scalar loads + ``OpCompositeConstruct``.

    SPIR-V's ``OpLoad`` from a single AccessChain returns a scalar
    matching the pointee type. To produce a vector result we
    AccessChain to the base of the contiguous ``width`` elements and
    load each one, then assemble with ``OpCompositeConstruct``. The
    assembler / driver pattern-matches this back to a vector load on
    targets that support vector pointers; on targets that don't, it
    becomes the ``width`` scalar loads we emit.

    Aligned vector loads via ``OpLoad`` on a typed vector pointer
    (``%v4float`` etc.) is the steady-state perf path. Lands once
    a kernel demands it; today's silu / elementwise kernels are
    bandwidth-bound long before the load pattern matters.
    """
    (out,) = op.results
    tensor = op.attrs["tensor"]
    width = int(op.attrs["width"])
    indices = list(op.operands)
    if op.attrs.get("pred") is not None:
        indices = indices[:-1]

    if isinstance(tensor, GlobalTensor):
        binding = ctx.tensor_to_binding[id(tensor.param)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        elem_type = ctx.tensor_to_elem_type[id(tensor.param)]
        zero = ctx.text.const_uint(0)
        chain_prefix = (var_id, zero)
    elif isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_vec_load: SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp"
            )
        var_id, elem_type, elem_ptr, _n = rec
        chain_prefix = (var_id,)
    else:
        raise NotImplementedError(
            f"_visit_vec_load: tensor type {type(tensor).__name__}"
        )

    # Compute the base flat index, then each lane's index = base + i.
    if isinstance(tensor, GlobalTensor):
        base_id = _flatten_global_index(tuple(indices), tensor, ctx)
    else:
        base_id = _flatten_index(tuple(indices), tensor.shape, ctx)
        base_id = _add_dyn_offset(base_id, tensor, ctx)
    u32 = ctx.text.type_int(32, signed=False)

    loaded: list[str] = []
    for i in range(width):
        if i == 0:
            idx_id = base_id
        else:
            inc = ctx.text.const_uint(i)
            idx_id = ctx.text.alloc_id(f"vec_idx_{i}")
            ctx.text.emit_function(f"{idx_id} = OpIAdd {u32} {base_id} {inc}")
        chain_id = ctx.text.alloc_id(f"vec_chain_{i}")
        chain_args = " ".join(chain_prefix)
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {chain_args} {idx_id}"
        )
        ld_id = ctx.text.alloc_id(f"vec_ld_{i}")
        ctx.text.emit_function(f"{ld_id} = OpLoad {elem_type} {chain_id}")
        loaded.append(ld_id)

    # Assemble the vector. The result's vec type needs declaring.
    vec_type = ctx.text.type_vec(elem_type, width)
    res_id = ctx.text.alloc_id(f"vec_w{width}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {vec_type} {' '.join(loaded)}"
    )


def _visit_vec_store(op: VecStoreOp, ctx: _SpvCtx) -> None:
    """Vector store — ``OpCompositeExtract`` + N scalar stores.

    Mirror of ``_visit_vec_load``: extract each element of the input
    vector, AccessChain + OpStore at base+i. The driver-side
    coalescing turns this back into a vector store on platforms
    that support it.
    """
    tensor = op.attrs["tensor"]
    vec_v = op.operands[0]
    width = int(vec_v.width)
    indices = list(op.operands[1:])
    if op.attrs.get("pred") is not None:
        indices = indices[:-1]

    if isinstance(tensor, GlobalTensor):
        binding = ctx.tensor_to_binding[id(tensor.param)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        elem_type = ctx.tensor_to_elem_type[id(tensor.param)]
        zero = ctx.text.const_uint(0)
        chain_prefix = (var_id, zero)
    elif isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_vec_store: SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp"
            )
        var_id, elem_type, elem_ptr, _n = rec
        chain_prefix = (var_id,)
    else:
        raise NotImplementedError(
            f"_visit_vec_store: tensor type {type(tensor).__name__}"
        )

    if isinstance(tensor, GlobalTensor):
        base_id = _flatten_global_index(tuple(indices), tensor, ctx)
    else:
        base_id = _flatten_index(tuple(indices), tensor.shape, ctx)
        base_id = _add_dyn_offset(base_id, tensor, ctx)
    vec_id = ctx.val_to_id[vec_v.id]
    u32 = ctx.text.type_int(32, signed=False)

    for i in range(width):
        # Extract element i from the vector.
        elem_id = ctx.text.alloc_id(f"vec_elem_{i}")
        ctx.text.emit_function(
            f"{elem_id} = OpCompositeExtract {elem_type} {vec_id} {i}"
        )
        if i == 0:
            idx_id = base_id
        else:
            inc = ctx.text.const_uint(i)
            idx_id = ctx.text.alloc_id(f"vec_sidx_{i}")
            ctx.text.emit_function(f"{idx_id} = OpIAdd {u32} {base_id} {inc}")
        chain_id = ctx.text.alloc_id(f"vec_schain_{i}")
        chain_args = " ".join(chain_prefix)
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {chain_args} {idx_id}"
        )
        ctx.text.emit_function(f"OpStore {chain_id} {elem_id}")


def _visit_vec_build(op: VecBuildOp, ctx: _SpvCtx) -> None:
    """``OpCompositeConstruct`` from N scalar operands.

    With ``packed_b32=True`` (the ``qk.vec_build_packed_b32`` IR
    helper): inputs are B32 scalars, each holding 2 elements of the
    output's element dtype. Inverse of the packed_b32 ``vec_extract``
    path. Splits each B32 into a (lo_b16, hi_b16) pair, bitcasts each
    half to the element dtype (bf16/f16), then composes into an
    array<elem, 2*N>.
    """
    (out,) = op.results
    if op.attrs.get("packed_b32"):
        elem_dt = out.dtype
        elem_type = _emit_dtype(ctx.text, elem_dt, ctx)
        u16 = _emit_dtype(ctx.text, DType.B16, ctx)
        u32 = ctx.text.type_int(32, signed=False)
        mask = ctx.text.const_uint(0xFFFF)
        sixteen = ctx.text.const_uint(16)

        all_elems: list[str] = []
        for i, b32_v in enumerate(op.operands):
            b32_id = ctx.val_to_id[b32_v.id]
            # B32 already lowers to u32; the OpBitcast is for clarity
            # (and would be a real conversion if B32 ever splits from
            # u32 in a future visitor).
            asu32 = ctx.text.alloc_id(f"vb32_as_u32_{i}")
            ctx.text.emit_function(f"{asu32} = OpBitcast {u32} {b32_id}")
            lo_u32 = ctx.text.alloc_id(f"vb_lo_u32_{i}")
            hi_u32 = ctx.text.alloc_id(f"vb_hi_u32_{i}")
            ctx.text.emit_function(
                f"{lo_u32} = OpBitwiseAnd {u32} {asu32} {mask}"
            )
            ctx.text.emit_function(
                f"{hi_u32} = OpShiftRightLogical {u32} {asu32} {sixteen}"
            )
            lo_u16 = ctx.text.alloc_id(f"vb_lo_u16_{i}")
            hi_u16 = ctx.text.alloc_id(f"vb_hi_u16_{i}")
            ctx.text.emit_function(f"{lo_u16} = OpUConvert {u16} {lo_u32}")
            ctx.text.emit_function(f"{hi_u16} = OpUConvert {u16} {hi_u32}")
            lo_e = ctx.text.alloc_id(f"vb_lo_e_{i}")
            hi_e = ctx.text.alloc_id(f"vb_hi_e_{i}")
            ctx.text.emit_function(f"{lo_e} = OpBitcast {elem_type} {lo_u16}")
            ctx.text.emit_function(f"{hi_e} = OpBitcast {elem_type} {hi_u16}")
            all_elems.extend([lo_e, hi_e])

        vec_type = ctx.text.type_vec(elem_type, len(all_elems))
        res_id = ctx.text.alloc_id("vec_build_pb32")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpCompositeConstruct {vec_type} {' '.join(all_elems)}"
        )
        return

    elem_type = _emit_dtype(ctx.text, op.operands[0].dtype, ctx)
    vec_type = ctx.text.type_vec(elem_type, len(op.operands))
    res_id = ctx.text.alloc_id("vec_build")
    ctx.val_to_id[out.id] = res_id
    operand_ids = [ctx.val_to_id[v.id] for v in op.operands]
    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {vec_type} {' '.join(operand_ids)}"
    )


def _visit_vec_extract(op: VecExtractOp, ctx: _SpvCtx) -> None:
    """``OpCompositeExtract`` — pick one component from a vector.

    With ``packed_b32=True`` (the ``qk.packed_extract_b32`` IR helper):
    extracts the ``index``-th 32-bit word from a packed bf16/f16 vec
    (each B32 holds two consecutive 16-bit elements). Used by
    ``fma_bf16x2`` and friends. SPIR-V doesn't allow bitcasting a
    composite directly, so we extract the two element halves, bitcast
    each to u16, and merge u16×2 → u32 (mirrors the inverse of
    ``MergeB32Op``).
    """
    (out,) = op.results
    idx = int(op.attrs["index"])
    src_v = op.operands[0]
    src_id = ctx.val_to_id[src_v.id]

    if op.attrs.get("packed_b32"):
        elem_type = _emit_dtype(ctx.text, src_v.dtype, ctx)  # bf16 / f16
        u16 = _emit_dtype(ctx.text, DType.B16, ctx)
        u32 = ctx.text.type_int(32, signed=False)
        sixteen = ctx.text.const_uint(16)

        lo_idx = 2 * idx
        hi_idx = 2 * idx + 1
        elem_lo = ctx.text.alloc_id(f"px_e_lo_{idx}")
        elem_hi = ctx.text.alloc_id(f"px_e_hi_{idx}")
        ctx.text.emit_function(
            f"{elem_lo} = OpCompositeExtract {elem_type} {src_id} {lo_idx}"
        )
        ctx.text.emit_function(
            f"{elem_hi} = OpCompositeExtract {elem_type} {src_id} {hi_idx}"
        )
        u16_lo = ctx.text.alloc_id(f"px_u16_lo_{idx}")
        u16_hi = ctx.text.alloc_id(f"px_u16_hi_{idx}")
        ctx.text.emit_function(f"{u16_lo} = OpBitcast {u16} {elem_lo}")
        ctx.text.emit_function(f"{u16_hi} = OpBitcast {u16} {elem_hi}")
        u32_lo = ctx.text.alloc_id(f"px_u32_lo_{idx}")
        u32_hi = ctx.text.alloc_id(f"px_u32_hi_{idx}")
        ctx.text.emit_function(f"{u32_lo} = OpUConvert {u32} {u16_lo}")
        ctx.text.emit_function(f"{u32_hi} = OpUConvert {u32} {u16_hi}")
        u32_hi_shl = ctx.text.alloc_id(f"px_hi_shl_{idx}")
        ctx.text.emit_function(
            f"{u32_hi_shl} = OpShiftLeftLogical {u32} {u32_hi} {sixteen}"
        )
        res_id = ctx.text.alloc_id(f"px_b32_{idx}")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpBitwiseOr {u32} {u32_lo} {u32_hi_shl}"
        )
        return

    elem_type = _emit_dtype(ctx.text, out.dtype, ctx)
    res_id = ctx.text.alloc_id(f"vec_x{idx}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCompositeExtract {elem_type} {src_id} {idx}"
    )


# ─── Cooperative-matrix ops (MMA) — scaffolding for v2 ────────────
#
# Each visitor below is a stub that surfaces a clear NotImplementedError
# pointing at PORTABILITY_PLAN §3.3 (the full coopmat lowering work) so
# kernels in the gemm / attn cohort fail informatively rather than at
# the dispatch table miss. The full lowering needs:
#   - ``OpTypeCooperativeMatrixKHR`` for each (use, dtype, shape) triple
#     — emit helper landed in ``text.type_coop_matrix``.
#   - ``OpCooperativeMatrixLoadKHR Pointer MemoryLayout=RowMajor (0)
#     Stride MemoryOperand=None`` for ``LoadMatrixOp``.
#   - ``OpCooperativeMatrixStoreKHR`` mirror for ``StoreMatrixOp``.
#   - ``OpCooperativeMatrixMulAddKHR A B C Operands=AccumulationModeNone (0)``
#     for ``MmaOp``.
#   - Shape→(rows, cols, use) mapping from ``MmaShape`` (registry already
#     carries ``min_intel_gpu_gen=INTEL_XE2`` rows for the four
#     ``m8n16k16_intel_*`` shapes).
#   - Capability ``CooperativeMatrixKHR`` + extension
#     ``SPV_KHR_cooperative_matrix`` (gated on first use, similar to the
#     bf16 path).
#   - Scope = ``Subgroup`` (3) — Vulkan compute-shader scope for KHR
#     coopmat. Subgroup-level fragment storage means each lane holds a
#     small slice of the matrix; the IR's ``b32`` carrier maps to that
#     opaque lane-local storage automatically.


def _ensure_coopmat_caps(ctx: _SpvCtx) -> None:
    """Lazy-declare the ``CooperativeMatrixKHR`` capability +
    ``SPV_KHR_cooperative_matrix`` extension. Idempotent — repeated
    calls fold into the dedup'd capability list.

    The CooperativeMatrixKHR capability requires VulkanMemoryModel
    (Vulkan validation rejects the SPIR-V otherwise). The
    VulkanMemoryModel capability needs the matching
    ``OpMemoryModel Logical Vulkan`` declaration too — but the
    framework's text emitter pins ``OpMemoryModel Logical GLSL450``
    by default. We override the memory model lazily here.
    """
    ctx.text.add_capability("CooperativeMatrixKHR")
    ctx.text.add_capability("VulkanMemoryModel")
    ctx.text.add_extension("SPV_KHR_cooperative_matrix")
    ctx.text.set_memory_model("Logical Vulkan")


# IR ``which`` (a/b/c) → SPIR-V Use index. ``c`` and ``d`` both map to
# the Accumulator slot (the IR distinguishes load (c) from store (d)
# but coopmat doesn't — same register class).
_COOPMAT_USE = {"a": 0, "b": 1, "c": 2, "d": 2}


def _coop_dims_for(shape, which: str) -> tuple[int, int, "DType"]:
    """Return (rows, cols, dtype) for the coop-matrix tile of the
    given operand role within an ``MmaShape``."""
    if which == "a":
        return shape.m, shape.k, shape.a_dtype
    if which == "b":
        return shape.k, shape.n, shape.b_dtype
    return shape.m, shape.n, shape.acc_dtype


def _emit_matrix_pointer(
    tensor, row_id: str, col_id: str, ctx: _SpvCtx,
) -> tuple[str, str]:
    """Build an ``OpAccessChain`` pointer to the (row, col) tile origin
    in ``tensor`` and return ``(pointer_id, storage_class)``.

    Used by both ``LoadMatrixOp`` and ``StoreMatrixOp`` — the
    coopmat ops consume a Pointer to the start of the matrix data
    plus a stride, and produce / consume an opaque coopmat value.
    """
    # Build a flat row-major offset from (row, col) using the tensor's
    # declared shape. Both inputs are u32 SSA ids.
    u32 = ctx.text.type_int(32, signed=False)
    # row * stride_elems + col
    stride_const = ctx.text.const_uint(tensor.shape[-1])
    row_mul = ctx.text.alloc_id("coop_row_mul")
    ctx.text.emit_function(
        f"{row_mul} = OpIMul {u32} {row_id} {stride_const}"
    )
    flat = ctx.text.alloc_id("coop_flat")
    ctx.text.emit_function(f"{flat} = OpIAdd {u32} {row_mul} {col_id}")

    if isinstance(tensor, GlobalTensor):
        binding = ctx.tensor_to_binding[id(tensor.param)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        zero = ctx.text.const_uint(0)
        chain_id = ctx.text.alloc_id("coop_chain")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {zero} {flat}"
        )
        return chain_id, "StorageBuffer"
    if isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_emit_matrix_pointer: SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp"
            )
        var_id, _elem_type, elem_ptr, _n = rec
        # Apply ``warp_dyn_offset`` (the warp-uniform component of
        # ``warp_lane_view``) if set. Cooperative-matrix
        # ``OpCooperativeMatrixLoadKHR`` is a subgroup-collective op
        # — the base pointer must be uniform across the subgroup —
        # so we use ``warp_dyn_offset`` (= ``warp_id * rows *
        # row_stride``), NOT ``dyn_offset`` (which also includes the
        # PTX-style per-lane component used by ldmatrix). Without this
        # fix every warp loads the same smem position and multi-warp
        # GEMM produces warp 0's output replicated across rows.
        warp_off = getattr(tensor, "warp_dyn_offset", None)
        if warp_off is not None:
            w_id = ctx.val_to_id[warp_off.id]
            new_flat = ctx.text.alloc_id("coop_warp_off")
            ctx.text.emit_function(
                f"{new_flat} = OpIAdd {u32} {flat} {w_id}"
            )
            flat = new_flat
        chain_id = ctx.text.alloc_id("coop_smem_chain")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {flat}"
        )
        return chain_id, "Workgroup"
    raise NotImplementedError(
        f"_emit_matrix_pointer: tensor type {type(tensor).__name__} not wired"
    )


def _visit_load_matrix(op: LoadMatrixOp, ctx: _SpvCtx) -> None:
    """``LoadMatrixOp`` → ``OpCooperativeMatrixLoadKHR``.

    Emits the cooperative matrix type for the (which, shape) pair,
    builds an ``OpAccessChain`` pointer to the (row, col) tile origin,
    and loads the tile in row-major layout with element-unit stride
    equal to the tensor's last-dim size.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    _ensure_coopmat_caps(ctx)
    (out,) = op.results
    tensor = op.attrs["src_tensor"]
    shape_id = op.attrs["shape_id"]
    which = op.attrs["which"]

    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_load_matrix: unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    rows, cols, dtype = _coop_dims_for(shape, which)
    use = _COOPMAT_USE[which]

    elem_type = _emit_dtype(ctx.text, dtype, ctx)
    if dtype is DType.BF16:
        # bf16 coopmat element type needs the matching capability.
        ctx.text.add_capability("BFloat16CooperativeMatrixKHR")
    coop_t = ctx.text.type_coop_matrix(
        elem_type, scope=3, rows=rows, cols=cols, use=use,
    )

    row_id = ctx.val_to_id[op.operands[0].id]
    col_id = ctx.val_to_id[op.operands[1].id]
    ptr_id, _storage_class = _emit_matrix_pointer(tensor, row_id, col_id, ctx)

    # Memory layout: the framework's gemm cohort stores B as (N, K)
    # row-major (K axis contiguous — the "B^T in storage" convention
    # every gemm uses). From MMA's K×N point of view, that's a
    # column-major matrix, so ``b`` loads emit
    # ``MemoryLayout = ColumnMajor``. ``a`` and ``c`` use row-major
    # since their tensors are stored in M-contiguous form.
    #
    # Stride must be the ELEMENT stride between consecutive rows of
    # the OUTERMOST dim (``tensor.stride[0]``), NOT ``shape[-1]`` —
    # those differ when the SharedRegion was padded for bank-conflict
    # avoidance (e.g. attn's V smem has ``shape=(Dh, KvTile=32)`` but
    # ``stride[0] = KvTile + KvPad = 40``). Using shape[-1]=32 read
    # the wrong elements for k > 0 and zeroed cols 4-7 / 12-15 / …
    # of the attn output (the ``group_id``-parity-looking pattern).
    stride_val = int(tensor.stride[0]) if hasattr(tensor, "stride") and tensor.stride else int(tensor.shape[-1])
    layout_id = ctx.text.const_uint(1 if which == "b" else 0)
    stride_id = ctx.text.const_uint(stride_val)

    res_id = ctx.text.alloc_id(f"coop_load_{which}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCooperativeMatrixLoadKHR {coop_t} {ptr_id} "
        f"{layout_id} {stride_id}"
    )


def _visit_store_matrix(op: StoreMatrixOp, ctx: _SpvCtx) -> None:
    """``StoreMatrixOp`` → ``OpCooperativeMatrixStoreKHR``. Mirror of
    the load path; consumes a coopmat SSA and writes back."""
    _ensure_coopmat_caps(ctx)
    tensor = op.attrs["dst_tensor"]
    # operand layout: (frag, row, col)
    frag_v = op.operands[0]
    frag_id = ctx.val_to_id[frag_v.id]
    row_id = ctx.val_to_id[op.operands[1].id]
    col_id = ctx.val_to_id[op.operands[2].id]
    ptr_id, _storage_class = _emit_matrix_pointer(tensor, row_id, col_id, ctx)
    # Same padding-aware stride as the load path — see
    # ``_visit_load_matrix`` for the fix rationale.
    stride_val = int(tensor.stride[0]) if hasattr(tensor, "stride") and tensor.stride else int(tensor.shape[-1])
    stride_id = ctx.text.const_uint(stride_val)
    layout_id = ctx.text.const_uint(0)
    ctx.text.emit_function(
        f"OpCooperativeMatrixStoreKHR {ptr_id} {frag_id} "
        f"{layout_id} {stride_id}"
    )


def _visit_frag_for_each(op: FragForEachOp, ctx: _SpvCtx) -> None:
    """``FragForEachOp`` via smem roundtrip.

    The IR primitive iterates a body over each storage slot of a
    fragment. PTX/Metal can index the per-register / per-lane storage
    directly; SPIR-V cooperative_matrix doesn't expose that mapping
    (it's implementation-private). The portable workaround is to
    ``OpCooperativeMatrixStoreKHR`` the fragment to a smem scratch
    region in row-major layout — at that point the (row, col)
    addressing the body uses is well-defined — then walk through the
    elements one slot at a time per lane.

    Lane-slot partition: each lane owns ``rows*cols / subgroup_size``
    slots, indexed as ``lane_id + s * subgroup_size``. The
    ``rows*cols`` must divide the subgroup size (Battlemage's 32
    matches the four ``m8n16k16_intel_*`` shapes' 8×16 tile).

    The body's ``body_input_var`` / ``body_row_var`` / ``body_col_var``
    are bound per slot before walking the body ops; the visitor
    surface (load, store, arith, etc.) sees them as plain SSA ids.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    _ensure_coopmat_caps(ctx)
    shape_id = op.attrs["shape_id"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_for_each: unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    rows, cols, dtype = _coop_dims_for(shape, "c")
    n_elems = rows * cols
    subgroup_width = 32  # Battlemage. TODO: pull from caps when Xe-LPG lands.
    if n_elems % subgroup_width != 0:
        raise NotImplementedError(
            f"_visit_frag_for_each: tile {rows}×{cols}={n_elems} not "
            f"divisible by subgroup_width={subgroup_width}"
        )
    n_slots_per_lane = n_elems // subgroup_width

    # 1. Allocate a unique scratch smem region for this FragForEach.
    # Sized per-warp so multi-warp kernels don't race on the same
    # smem range — see ``_frag_scratch_warp_partition``.
    elem_type = _emit_dtype(ctx.text, dtype, ctx)
    total_elems, smem_base_ssa, mat_base_ssa, warp_off = (
        _frag_scratch_warp_partition(ctx, n_elems, name_hint="frag_each")
    )
    multi_warp = bool(warp_off)
    n_const = ctx.text.const_uint(total_elems)
    arr_id = ctx.text.alloc_id("frag_each_arr")
    ctx.text.add_type_line(f"{arr_id} = OpTypeArray {elem_type} {n_const}")
    ptr_arr = ctx.text.type_pointer("Workgroup", arr_id)
    var_id = ctx.text.alloc_id("frag_each_smem")
    ctx.text.add_type_line(f"{var_id} = OpVariable {ptr_arr} Workgroup")
    elem_ptr = ctx.text.type_pointer("Workgroup", elem_type)
    # Track the scratch in smem_allocs so the entry-point interface
    # walker picks it up. ``smem_allocs`` keys on a Value.id; use a
    # synthetic key (id() of this op) since there's no IR Value here.
    ctx.smem_allocs[id(op)] = (var_id, elem_type, elem_ptr, total_elems)

    # 2. Get a pointer to scratch[warp_off] for the coop store
    # (each subgroup writes its private slice). ``warp_off`` shares
    # the ssa from the partition helper — no re-emit.
    zero = ctx.text.const_uint(0)
    base_ptr = ctx.text.alloc_id("frag_each_base")
    ctx.text.emit_function(
        f"{base_ptr} = OpAccessChain {elem_ptr} {var_id} "
        f"{warp_off if multi_warp else zero}"
    )

    # 3. Store the coopmat to scratch in row-major order.
    coop_id = ctx.val_to_id[op.in_frag.id]
    cols_const = ctx.text.const_uint(cols)
    layout_id = ctx.text.const_uint(0)  # RowMajor
    ctx.text.emit_function(
        f"OpCooperativeMatrixStoreKHR {base_ptr} {coop_id} "
        f"{layout_id} {cols_const}"
    )

    # 4. Subgroup barrier so all lanes see the coopmat-store writes
    # before reading per-slot below. Subgroup scope is sufficient
    # because each warp's smem slice is private; AcquireRelease |
    # WorkgroupMemory matches the smem semantics other barriers use.
    sg_scope = ctx.text.const_uint(3)  # Subgroup
    sg_sem = ctx.text.const_uint(0x8 | 0x100)
    ctx.text.emit_function(
        f"OpControlBarrier {sg_scope} {sg_scope} {sg_sem}"
    )

    # 5. Iterate slots Python-side. Each slot has two indices: smem
    # index ``smem_base + s*32`` (= warp_base + lane_id + s*32, the
    # actual smem slot to access) and matrix index ``lane_id + s*32``
    # (used to compute the body's row/col bindings — those are
    # matrix-relative, not warp-relative).
    u32 = ctx.text.type_int(32, signed=False)
    cols_const_div = ctx.text.const_uint(cols)
    for s in range(n_slots_per_lane):
        smem_idx = _frag_scratch_slot_idx(
            ctx, base_ssa=smem_base_ssa, s=s,
            subgroup_width=subgroup_width, name_prefix="frag_each_smem",
        )
        mat_idx = _frag_scratch_slot_idx(
            ctx, base_ssa=mat_base_ssa, s=s,
            subgroup_width=subgroup_width, name_prefix="frag_each_mat",
        ) if multi_warp else smem_idx

        # row = mat_idx / cols, col = mat_idx % cols (matrix-relative,
        # warp-independent — every warp's coopmat has the same shape).
        row_id = ctx.text.alloc_id(f"frag_each_row_{s}")
        ctx.text.emit_function(
            f"{row_id} = OpUDiv {u32} {mat_idx} {cols_const_div}"
        )
        col_id = ctx.text.alloc_id(f"frag_each_col_{s}")
        ctx.text.emit_function(
            f"{col_id} = OpUMod {u32} {mat_idx} {cols_const_div}"
        )

        # Load element from scratch[smem_idx] (warp-private slice).
        chain_id = ctx.text.alloc_id(f"frag_each_load_chain_{s}")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {smem_idx}"
        )
        elem_id = ctx.text.alloc_id(f"frag_each_elem_{s}")
        ctx.text.emit_function(
            f"{elem_id} = OpLoad {elem_type} {chain_id}"
        )

        # Bind body vars and walk the body ops. Suppress any
        # surrounding ``loop_yield_stack`` so the body's terminating
        # void YieldOp isn't read as a for-loop carry yield.
        ctx.val_to_id[op.body_input_var.id] = elem_id
        ctx.val_to_id[op.body_row_var.id] = row_id
        ctx.val_to_id[op.body_col_var.id] = col_id
        if op.body_selector_var is not None:
            slot_to_sel_attr = op.attrs.get("slot_to_selector_idx")
            n_selectors = len(op.operands) - 1
            if slot_to_sel_attr is not None:
                slot_to_sel = slot_to_sel_attr
                sel_idx = slot_to_sel[s] if s < len(slot_to_sel) else 0
                sel_v = op.operands[1 + sel_idx]
                ctx.val_to_id[op.body_selector_var.id] = ctx.val_to_id[sel_v.id]
            else:
                # Dynamic-row-dispatch: row already computed as
                # ``row_id``. OpSelect-chain selectors against the row.
                bool_t = ctx.text.type_bool()
                sel_chain = ctx.val_to_id[op.operands[1 + n_selectors - 1].id]
                for i in range(n_selectors - 2, -1, -1):
                    cmp_id = ctx.text.alloc_id(f"frag_each_sel_cmp_{s}_{i}")
                    i_const = ctx.text.const_uint(i)
                    ctx.text.emit_function(
                        f"{cmp_id} = OpIEqual {bool_t} {row_id} {i_const}"
                    )
                    sel_lhs = ctx.val_to_id[op.operands[1 + i].id]
                    sel_dtype_id = _emit_dtype(
                        ctx.text, op.operands[1 + i].dtype, ctx,
                    )
                    sel_id_new = ctx.text.alloc_id(f"frag_each_sel_{s}_{i}")
                    ctx.text.emit_function(
                        f"{sel_id_new} = OpSelect {sel_dtype_id} {cmp_id} "
                        f"{sel_lhs} {sel_chain}"
                    )
                    sel_chain = sel_id_new
                ctx.val_to_id[op.body_selector_var.id] = sel_chain
        saved_stack = ctx.loop_yield_stack
        ctx.loop_yield_stack = []  # type: ignore[assignment]
        try:
            for body_op in op.body.ops:
                _walk_op(body_op, ctx)
        finally:
            ctx.loop_yield_stack = saved_stack


def _ensure_lane_id(ctx: _SpvCtx) -> str:
    """Return the SSA id for ``SubgroupLocalInvocationId``, declaring
    the input variable + load lazily (mirrors the lane_id_x_loaded
    cache the LaneIdOp visitor uses)."""
    cached = getattr(ctx, "lane_id_x_loaded", "")
    if cached:
        return cached
    var_id = getattr(ctx, "lane_id_var", "")
    if not var_id:
        u32 = ctx.text.type_int(32, signed=False)
        ptr = ctx.text.type_pointer("Input", u32)
        var_id = ctx.text.alloc_id("SubgroupLocalInvocationId")
        ctx.text.add_type_line(f"{var_id} = OpVariable {ptr} Input")
        ctx.text.add_decoration(
            f"OpDecorate {var_id} BuiltIn SubgroupLocalInvocationId"
        )
        ctx.lane_id_var = var_id
    u32 = ctx.text.type_int(32, signed=False)
    loaded = ctx.text.alloc_id("SubgroupLocalInvocationId_v")
    ctx.text.emit_function(f"{loaded} = OpLoad {u32} {var_id}")
    ctx.lane_id_x_loaded = loaded
    return loaded


def _ensure_subgroup_id_ssa(ctx: _SpvCtx) -> str:
    """Return the SSA id for ``SubgroupId`` (the warp index inside the
    workgroup, 0..n_warps-1). Used by frag visitors to partition
    workgroup-scope scratch by warp so multiple subgroups don't race
    on the same smem region."""
    return _ensure_scalar_builtin(
        ctx,
        var_attr="subgroup_id_var",
        cache_attr="subgroup_id_loaded",
        builtin_name="SubgroupId",
    )


def _frag_scratch_warp_partition(
    ctx: _SpvCtx, n_elems_per_warp: int, *, name_hint: str = "frag",
) -> tuple[int, str, str, str]:
    """Return ``(total_elems, smem_base_ssa, matrix_base_ssa,
    warp_off_ssa)`` for a Frag* visitor's smem scratch.

    Each subgroup (warp) gets a private ``n_elems_per_warp`` slot in a
    workgroup-scoped scratch sized ``n_warps * n_elems_per_warp``.

    Three address-space SSA ids:
    * ``smem_base_ssa = warp_off + lane_id`` — slot base for
      ``OpAccessChain``. Each warp's slice is disjoint so the
      cross-subgroup race that bit the original FragForEach pattern
      (every warp wrote ``scratch[0..127]``) is gone.
    * ``matrix_base_ssa = lane_id`` — matrix-local, used to compute
      the body's ``(row, col)`` bindings. Independent of warp; each
      warp's coopmat has the same matrix shape.
    * ``warp_off_ssa = subgroup_id * n_elems_per_warp`` — the warp's
      offset into the global scratch array, used by callers as the
      base for ``OpCooperativeMatrix{Load,Store}KHR`` (collective
      ops that need a single per-warp pointer, not per-lane).
      Sharing this ssa with ``smem_base_ssa`` (smem_base = warp_off
      + lane_id) avoids re-emitting the same ``OpIMul``.

    Without this partition the FragForEach / FragApply / FragReduce /
    FragConvert smem-roundtrip pattern corrupts any kernel with
    ``n_warps > 1`` (e.g. multi-warp GEMM epilogue).

    For ``n_warps == 1`` smem_base and matrix_base are the cached lane
    id, and warp_off is empty (caller skips the warp-aware AccessChain
    branch entirely).
    """
    n_warps = max(1, (
        ctx.local_size[0] * ctx.local_size[1] * ctx.local_size[2]
    ) // 32)
    lane_id_ssa = _ensure_lane_id(ctx)
    if n_warps == 1:
        return n_elems_per_warp, lane_id_ssa, lane_id_ssa, ""
    total_elems = n_elems_per_warp * n_warps
    u32 = ctx.text.type_int(32, signed=False)
    sgid = _ensure_subgroup_id_ssa(ctx)
    n_const = ctx.text.const_uint(n_elems_per_warp)
    warp_off = ctx.text.alloc_id(f"{name_hint}_warp_base")
    ctx.text.emit_function(f"{warp_off} = OpIMul {u32} {sgid} {n_const}")
    smem_base = ctx.text.alloc_id(f"{name_hint}_smem_base")
    ctx.text.emit_function(f"{smem_base} = OpIAdd {u32} {warp_off} {lane_id_ssa}")
    return total_elems, smem_base, lane_id_ssa, warp_off


def _frag_scratch_slot_idx(
    ctx: _SpvCtx,
    *,
    base_ssa: str,
    s: int,
    subgroup_width: int,
    name_prefix: str,
) -> str:
    """Compute ``base + s * subgroup_width`` as an SSA id. Used to
    derive both the smem-slot index (from ``smem_base``) and the
    matrix-element index (from ``matrix_base``). ``s == 0``
    short-circuits to ``base_ssa``."""
    if s == 0:
        return base_ssa
    u32 = ctx.text.type_int(32, signed=False)
    offset = ctx.text.const_uint(s * subgroup_width)
    idx_id = ctx.text.alloc_id(f"{name_prefix}_idx_{s}")
    ctx.text.emit_function(f"{idx_id} = OpIAdd {u32} {base_ssa} {offset}")
    return idx_id


_FRAG_REDUCE_TO_SUBGROUP_OP: dict[tuple[str, str], str] = {
    ("add", "float"): "OpGroupNonUniformFAdd",
    ("add", "uint"): "OpGroupNonUniformIAdd",
    ("add", "sint"): "OpGroupNonUniformIAdd",
    ("max", "float"): "OpGroupNonUniformFMax",
    ("max", "uint"): "OpGroupNonUniformUMax",
    ("max", "sint"): "OpGroupNonUniformSMax",
    ("min", "float"): "OpGroupNonUniformFMin",
    ("min", "uint"): "OpGroupNonUniformUMin",
    ("min", "sint"): "OpGroupNonUniformSMin",
    ("mul", "float"): "OpGroupNonUniformFMul",
    ("mul", "uint"): "OpGroupNonUniformIMul",
    ("mul", "sint"): "OpGroupNonUniformIMul",
}


def _visit_frag_reduce(op: FragReduceOp, ctx: _SpvCtx) -> None:
    """``FragReduceOp`` — reduce a coopmat fragment along an axis.

    Smem-roundtrip + subgroup reduce. Stores the input coopmat to a
    scratch region, each lane reads its slots and forms a partial
    reduction, then ``OpGroupNonUniform<kind>`` broadcasts the
    cross-lane reduction so every result lane sees the full scalar.

    Multi-class path (Intel coopmat, ``n_classes_override`` set):

      Intel ``cd_offsets`` is ``((0,0),) * c_regs`` because the
      lane↔(row,col) mapping inside an Intel cooperative matrix is
      implementation-private (Vulkan KHR coopmat does not expose it).
      With those placeholders, ``len({dr})`` is 1, so the cd_offsets-
      derived contract collapses to a single class.

      ``n_classes_override`` (set by the kernel via the SPV-aware
      ``frag_reduce(..., n_classes_override=shape.m)`` path) bypasses
      that derivation and produces ``rows`` results via
      ``ClusteredReduce(cluster_size=cols)`` + ``OpGroupNonUniform
      Broadcast`` from each row's leader lane. Single source of truth
      for the result count is the override attr.

      Single-class fallback below (``n_results == 1``) keeps the
      simple cross-lane reduce + broadcast for PTX/Apple paths whose
      cd_offsets actually produces useful row-class info.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    _ensure_coopmat_caps(ctx)
    shape_id = op.attrs["shape_id"]
    axis = op.attrs["axis"]
    kind = op.attrs["kind"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_reduce: unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    rows, cols, dtype = _coop_dims_for(shape, "c")
    cd_offsets = op.attrs["cd_offsets"]
    n_classes_override = op.attrs.get("n_classes_override")

    # Class partition. ``axis="row"`` reduces *cols* (one scalar per
    # row class); ``axis="col"`` reduces rows (one scalar per col
    # class). ``n_classes_override`` (set on the SPV/Intel path) asks
    # for ``rows`` results derived from the smem layout instead of
    # cd_offsets — single source of truth for the result count.
    if n_classes_override is not None:
        n_results = int(n_classes_override)
    else:
        if axis == "row":
            classes = sorted({dr for dr, _ in cd_offsets})
        else:
            classes = sorted({dc for _, dc in cd_offsets})
        n_results = len(classes)
    if n_results > 1 and axis != "row":
        raise NotImplementedError(
            f"_visit_frag_reduce: multi-class reduction only wired for "
            f"axis=row today (got axis={axis!r}, n_results={n_results})"
        )

    n_elems = rows * cols
    subgroup_width = 32
    if n_elems % subgroup_width != 0:
        raise NotImplementedError(
            f"_visit_frag_reduce: tile {rows}×{cols} not divisible "
            f"by {subgroup_width}"
        )
    n_slots_per_lane = n_elems // subgroup_width

    elem_type = _emit_dtype(ctx.text, dtype, ctx)
    if dtype is DType.BF16:
        ctx.text.add_capability("BFloat16CooperativeMatrixKHR")

    # Scratch smem for the input coopmat. Per-warp partition so
    # multi-warp kernels don't race on the same smem range.
    total_elems, smem_base_ssa, _mat_base_ssa, warp_off = (
        _frag_scratch_warp_partition(ctx, n_elems, name_hint="frag_red")
    )
    multi_warp = bool(warp_off)
    n_const = ctx.text.const_uint(total_elems)
    arr_id = ctx.text.alloc_id("frag_red_arr")
    ctx.text.add_type_line(f"{arr_id} = OpTypeArray {elem_type} {n_const}")
    ptr_arr = ctx.text.type_pointer("Workgroup", arr_id)
    var_id = ctx.text.alloc_id("frag_red_smem")
    ctx.text.add_type_line(f"{var_id} = OpVariable {ptr_arr} Workgroup")
    elem_ptr = ctx.text.type_pointer("Workgroup", elem_type)
    ctx.smem_allocs[id(op)] = (var_id, elem_type, elem_ptr, total_elems)

    zero = ctx.text.const_uint(0)
    base_ptr = ctx.text.alloc_id("frag_red_base")
    ctx.text.emit_function(
        f"{base_ptr} = OpAccessChain {elem_ptr} {var_id} "
        f"{warp_off if multi_warp else zero}"
    )

    # Store input coopmat to scratch (warp's private slice).
    in_id = ctx.val_to_id[op.operands[0].id]
    cols_const = ctx.text.const_uint(cols)
    layout_id = ctx.text.const_uint(0)  # RowMajor
    ctx.text.emit_function(
        f"OpCooperativeMatrixStoreKHR {base_ptr} {in_id} "
        f"{layout_id} {cols_const}"
    )

    sg_scope = ctx.text.const_uint(3)
    sg_sem = ctx.text.const_uint(0x8 | 0x100)
    ctx.text.emit_function(
        f"OpControlBarrier {sg_scope} {sg_scope} {sg_sem}"
    )

    u32 = ctx.text.type_int(32, signed=False)

    sub_op = _FRAG_REDUCE_TO_SUBGROUP_OP.get((kind, _dtype_kind(dtype)))
    if sub_op is None:
        raise NotImplementedError(
            f"_visit_frag_reduce: kind={kind!r} dtype={dtype!r} not wired"
        )
    # Capability for arithmetic subgroup reduces (FMax/FMin/FAdd/etc.)
    # — the boolean-only ``GroupNonUniform`` capability isn't enough.
    ctx.text.add_capability("GroupNonUniformArithmetic")

    if n_results == 1:
        # ── Single-class path: per-lane fold across slots, then a
        # cross-lane reduce broadcasting the result to every lane.
        accum: str | None = None
        for s in range(n_slots_per_lane):
            idx_id = _frag_scratch_slot_idx(
                ctx, base_ssa=smem_base_ssa, s=s,
                subgroup_width=subgroup_width, name_prefix="frag_red",
            )
            chain = ctx.text.alloc_id(f"frag_red_chain_{s}")
            ctx.text.emit_function(
                f"{chain} = OpAccessChain {elem_ptr} {var_id} {idx_id}"
            )
            elem = ctx.text.alloc_id(f"frag_red_elem_{s}")
            ctx.text.emit_function(
                f"{elem} = OpLoad {elem_type} {chain}"
            )
            if accum is None:
                accum = elem
                continue
            new_accum = ctx.text.alloc_id(f"frag_red_acc_{s}")
            if kind == "add":
                spv_op = "OpFAdd" if _dtype_kind(dtype) == "float" else "OpIAdd"
                ctx.text.emit_function(
                    f"{new_accum} = {spv_op} {elem_type} {accum} {elem}"
                )
            elif kind == "mul":
                spv_op = "OpFMul" if _dtype_kind(dtype) == "float" else "OpIMul"
                ctx.text.emit_function(
                    f"{new_accum} = {spv_op} {elem_type} {accum} {elem}"
                )
            elif kind in ("max", "min"):
                glsl = ctx.text.import_ext_inst("GLSL.std.450")
                instr = (
                    ("FMax" if kind == "max" else "FMin")
                    if _dtype_kind(dtype) == "float"
                    else (("SMax" if kind == "max" else "SMin")
                          if _dtype_kind(dtype) == "sint"
                          else ("UMax" if kind == "max" else "UMin"))
                )
                ctx.text.emit_function(
                    f"{new_accum} = OpExtInst {elem_type} {glsl} "
                    f"{instr} {accum} {elem}"
                )
            accum = new_accum

        res_id = ctx.text.alloc_id(f"frag_red_{kind}")
        ctx.val_to_id[op.results[0].id] = res_id
        ctx.text.emit_function(
            f"{res_id} = {sub_op} {elem_type} {sg_scope} Reduce {accum}"
        )
        return

    # ── Multi-class path (axis=row, n_results > 1).
    #
    # Smem layout after ``OpCooperativeMatrixStoreKHR`` row-major:
    # ``scratch[r * cols + c] = M[r, c]``. Per-lane reads use
    # ``smem_idx = warp_off + lane_id + s * 32``, so for each fixed
    # slot ``s`` and the standard ``cols == 16`` Intel layout:
    #
    #   lanes 0..15  cover M[2*s,     0..15]   → row class 2*s
    #   lanes 16..31 cover M[2*s + 1, 0..15]   → row class 2*s + 1
    #
    # ``ClusteredReduce`` with ``cluster_size = cols`` reduces the row
    # contained in each cluster. With cols=16 there are
    # ``subgroup_width / cluster_size == 2`` clusters per slot, giving
    # ``2 * n_slots_per_lane`` row-class reductions which must equal
    # ``n_results``. Each row's reduce is then broadcast to every lane
    # via ``OpGroupNonUniformBroadcast`` from the leader lane of its
    # cluster (lane 0 for cluster 0, lane 16 for cluster 1).
    if cols == 0 or subgroup_width % cols != 0:
        raise NotImplementedError(
            f"_visit_frag_reduce multi-class: cols={cols} doesn't "
            f"divide subgroup_width={subgroup_width}"
        )
    cluster_size = cols
    rows_per_slot = subgroup_width // cluster_size
    if rows_per_slot * n_slots_per_lane != n_results:
        raise NotImplementedError(
            f"_visit_frag_reduce multi-class: layout doesn't fit — "
            f"rows_per_slot={rows_per_slot} * n_slots={n_slots_per_lane} "
            f"!= n_results={n_results}"
        )
    ctx.text.add_capability("GroupNonUniformClustered")
    cluster_const = ctx.text.const_uint(cluster_size)

    slot_reduced: list[str] = []
    for s in range(n_slots_per_lane):
        idx_id = _frag_scratch_slot_idx(
            ctx, base_ssa=smem_base_ssa, s=s,
            subgroup_width=subgroup_width, name_prefix="frag_red",
        )
        chain = ctx.text.alloc_id(f"frag_red_chain_{s}")
        ctx.text.emit_function(
            f"{chain} = OpAccessChain {elem_ptr} {var_id} {idx_id}"
        )
        elem = ctx.text.alloc_id(f"frag_red_elem_{s}")
        ctx.text.emit_function(
            f"{elem} = OpLoad {elem_type} {chain}"
        )
        red_id = ctx.text.alloc_id(f"frag_red_clr_{s}")
        slot_reduced.append(red_id)
        ctx.text.emit_function(
            f"{red_id} = {sub_op} {elem_type} {sg_scope} ClusteredReduce "
            f"{elem} {cluster_const}"
        )

    # Per-row broadcast: row r → slot s = r // rows_per_slot, leader
    # lane = (r % rows_per_slot) * cluster_size.
    for row_idx in range(n_results):
        s = row_idx // rows_per_slot
        leader_lane = (row_idx % rows_per_slot) * cluster_size
        leader_const = ctx.text.const_uint(leader_lane)
        bcast_id = ctx.text.alloc_id(f"frag_red_row{row_idx}")
        ctx.val_to_id[op.results[row_idx].id] = bcast_id
        ctx.text.emit_function(
            f"{bcast_id} = OpGroupNonUniformBroadcast {elem_type} "
            f"{sg_scope} {slot_reduced[s]} {leader_const}"
        )


def _visit_frag_convert(op: FragConvertOp, ctx: _SpvCtx) -> None:
    """``FragConvertOp`` — multi-source coopmat → coopmat dtype +
    layout change. Smem roundtrip with per-slot body transform.

    Used by attention online-softmax: ``kf`` source ACC fragments
    (f32) merged into one A fragment (bf16) covering ``kf*src_K``
    cols, optional body applies the per-slot scale/exp transform.

    Layout: source ACCs are use=2 with dim ``M_src × N_src``;
    destination A frag is use=0 with dim ``M_dst × K_dst`` where
    ``K_dst = kf * N_src`` (the merge across kf source columns
    builds the destination K axis).
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    _ensure_coopmat_caps(ctx)
    (out,) = op.results
    shape_id = op.attrs["shape_id"]
    src_dtype = op.attrs["src_dtype"]
    dst_dtype = op.attrs["dst_dtype"]
    num_src = int(op.attrs["num_src_frags"])

    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_convert: unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape

    # Source frag dimensions (acc).
    src_rows, src_cols, _src_dt = _coop_dims_for(shape, "c")
    # Destination frag dimensions (A): rows=M_dst, cols=kf*src_cols.
    dst_rows = shape.m
    dst_cols = num_src * src_cols

    src_elem = _emit_dtype(ctx.text, src_dtype, ctx)
    dst_elem = _emit_dtype(ctx.text, dst_dtype, ctx)
    if src_dtype is DType.BF16 or dst_dtype is DType.BF16:
        ctx.text.add_capability("BFloat16CooperativeMatrixKHR")

    # Scratch for sources (one per source frag) + dst, partitioned
    # per-warp so multi-warp kernels don't race on the same range.
    src_n_elems = src_rows * src_cols
    dst_n_elems = dst_rows * dst_cols
    subgroup_width = 32
    n_warps = max(1, (
        ctx.local_size[0] * ctx.local_size[1] * ctx.local_size[2]
    ) // 32)
    src_total = src_n_elems * n_warps
    dst_total = dst_n_elems * n_warps
    multi_warp = n_warps > 1

    src_n_const = ctx.text.const_uint(src_total)
    dst_n_const = ctx.text.const_uint(dst_total)

    src_arr_id = ctx.text.alloc_id("frag_cvt_src_arr")
    ctx.text.add_type_line(
        f"{src_arr_id} = OpTypeArray {src_elem} {src_n_const}"
    )
    src_ptr_arr = ctx.text.type_pointer("Workgroup", src_arr_id)
    dst_arr_id = ctx.text.alloc_id("frag_cvt_dst_arr")
    ctx.text.add_type_line(
        f"{dst_arr_id} = OpTypeArray {dst_elem} {dst_n_const}"
    )
    dst_ptr_arr = ctx.text.type_pointer("Workgroup", dst_arr_id)

    # One scratch var per source frag, plus one for the destination.
    src_vars: list[str] = []
    src_elem_ptr = ctx.text.type_pointer("Workgroup", src_elem)
    dst_elem_ptr = ctx.text.type_pointer("Workgroup", dst_elem)
    for i in range(num_src):
        v = ctx.text.alloc_id(f"frag_cvt_src{i}")
        ctx.text.add_type_line(f"{v} = OpVariable {src_ptr_arr} Workgroup")
        src_vars.append(v)
        ctx.smem_allocs[(id(op), "src", i)] = (
            v, src_elem, src_elem_ptr, src_total,
        )
    dst_var = ctx.text.alloc_id("frag_cvt_dst")
    ctx.text.add_type_line(f"{dst_var} = OpVariable {dst_ptr_arr} Workgroup")
    ctx.smem_allocs[(id(op), "dst")] = (
        dst_var, dst_elem, dst_elem_ptr, dst_total,
    )

    zero = ctx.text.const_uint(0)
    src_cols_const = ctx.text.const_uint(src_cols)
    dst_cols_const = ctx.text.const_uint(dst_cols)
    layout_id = ctx.text.const_uint(0)  # RowMajor

    # Compute per-warp offsets for src and dst arrays.
    u32 = ctx.text.type_int(32, signed=False)
    lane_id_ssa = _ensure_lane_id(ctx)
    if multi_warp:
        sgid = _ensure_subgroup_id_ssa(ctx)
        src_n_per_warp = ctx.text.const_uint(src_n_elems)
        dst_n_per_warp = ctx.text.const_uint(dst_n_elems)
        src_warp_off = ctx.text.alloc_id("frag_cvt_src_warp_off")
        ctx.text.emit_function(
            f"{src_warp_off} = OpIMul {u32} {sgid} {src_n_per_warp}"
        )
        dst_warp_off = ctx.text.alloc_id("frag_cvt_dst_warp_off")
        ctx.text.emit_function(
            f"{dst_warp_off} = OpIMul {u32} {sgid} {dst_n_per_warp}"
        )
    else:
        src_warp_off = ""
        dst_warp_off = ""

    # Store each source coopmat to its smem at warp-private offset.
    for i in range(num_src):
        in_id = ctx.val_to_id[op.operands[i].id]
        base = ctx.text.alloc_id(f"frag_cvt_src{i}_base")
        offset_id = src_warp_off if multi_warp else zero
        ctx.text.emit_function(
            f"{base} = OpAccessChain {src_elem_ptr} {src_vars[i]} {offset_id}"
        )
        ctx.text.emit_function(
            f"OpCooperativeMatrixStoreKHR {base} {in_id} "
            f"{layout_id} {src_cols_const}"
        )

    sg_scope = ctx.text.const_uint(3)
    sg_sem = ctx.text.const_uint(0x8 | 0x100)
    ctx.text.emit_function(
        f"OpControlBarrier {sg_scope} {sg_scope} {sg_sem}"
    )

    # Per-lane: each slot owns one (row, k_col) position in the dst
    # M×K_dst tile. k_col = src_idx*src_cols + col_within_src.
    dst_n_slots_per_lane = dst_n_elems // subgroup_width
    src_n_slots_per_lane = src_n_elems // subgroup_width

    for s in range(dst_n_slots_per_lane):
        # Matrix-relative dst index (lane_id + s*32) — used to derive
        # row/col within the dst frag's logical M×K_dst tile.
        if s == 0:
            mat_idx_id = lane_id_ssa
        else:
            offset = ctx.text.const_uint(s * subgroup_width)
            mat_idx_id = ctx.text.alloc_id(f"frag_cvt_mat_idx_{s}")
            ctx.text.emit_function(
                f"{mat_idx_id} = OpIAdd {u32} {lane_id_ssa} {offset}"
            )
        # Smem-relative dst index (warp_base + lane_id + s*32) — used
        # for OpAccessChain into the warp's private dst slice.
        if multi_warp:
            idx_id = ctx.text.alloc_id(f"frag_cvt_smem_idx_{s}")
            ctx.text.emit_function(
                f"{idx_id} = OpIAdd {u32} {dst_warp_off} {mat_idx_id}"
            )
        else:
            idx_id = mat_idx_id
        # Determine which source frag this slot pulls from.
        # idx in dst space: row = idx / dst_cols, col = idx % dst_cols.
        # src_idx = col / src_cols, col_within_src = col % src_cols.
        # row/col within the dst frag — derived from the matrix-relative
        # index, NOT the warp-offset smem index (different warps share
        # the same matrix shape).
        dst_cols_div = ctx.text.const_uint(dst_cols)
        src_cols_div = ctx.text.const_uint(src_cols)
        row_id = ctx.text.alloc_id(f"frag_cvt_row_{s}")
        ctx.text.emit_function(
            f"{row_id} = OpUDiv {u32} {mat_idx_id} {dst_cols_div}"
        )
        col_id = ctx.text.alloc_id(f"frag_cvt_col_{s}")
        ctx.text.emit_function(
            f"{col_id} = OpUMod {u32} {mat_idx_id} {dst_cols_div}"
        )
        src_idx_id = ctx.text.alloc_id(f"frag_cvt_srcidx_{s}")
        ctx.text.emit_function(
            f"{src_idx_id} = OpUDiv {u32} {col_id} {src_cols_div}"
        )
        col_in_src_id = ctx.text.alloc_id(f"frag_cvt_colinsrc_{s}")
        ctx.text.emit_function(
            f"{col_in_src_id} = OpUMod {u32} {col_id} {src_cols_div}"
        )
        # src_addr = warp_off + row * src_cols + col_in_src (smem
        # offset within src array; warp_off=0 in single-warp mode).
        row_mul = ctx.text.alloc_id(f"frag_cvt_rowmul_{s}")
        ctx.text.emit_function(
            f"{row_mul} = OpIMul {u32} {row_id} {src_cols_div}"
        )
        local_addr = ctx.text.alloc_id(f"frag_cvt_locaddr_{s}")
        ctx.text.emit_function(
            f"{local_addr} = OpIAdd {u32} {row_mul} {col_in_src_id}"
        )
        if multi_warp:
            src_addr = ctx.text.alloc_id(f"frag_cvt_srcaddr_{s}")
            ctx.text.emit_function(
                f"{src_addr} = OpIAdd {u32} {src_warp_off} {local_addr}"
            )
        else:
            src_addr = local_addr

        # The source array to read from depends on ``src_idx_id`` at
        # runtime. We unconditionally load from each source scratch
        # (all live in workgroup smem so any address is well-defined,
        # just stale-data for the wrong source), then OpSelect a chain
        # on ``src_idx_id`` to pick the right value. This avoids
        # OpSelectionMerge / OpBranchConditional clutter in the body
        # path; spilled redundant loads are smem hits that the GPU
        # coalesces, and the lowerer keeps a single straight-line
        # body — much easier for the back-end to schedule.
        loads: list[str] = []
        for i in range(num_src):
            chain = ctx.text.alloc_id(f"frag_cvt_chain_{s}_{i}")
            ctx.text.emit_function(
                f"{chain} = OpAccessChain {src_elem_ptr} {src_vars[i]} {src_addr}"
            )
            elem_i = ctx.text.alloc_id(f"frag_cvt_in_{s}_{i}")
            ctx.text.emit_function(
                f"{elem_i} = OpLoad {src_elem} {chain}"
            )
            loads.append(elem_i)
        if num_src == 1:
            elem_in = loads[0]
        else:
            # OpSelect chain — fold from highest src index downward so
            # the default tail is loads[num_src-1] (matches the kernel
            # invariant that ``src_idx`` ∈ [0, num_src)).
            bool_t = ctx.text.type_bool()
            elem_in = loads[num_src - 1]
            for i in range(num_src - 2, -1, -1):
                cmp_id = ctx.text.alloc_id(f"frag_cvt_cmp_{s}_{i}")
                i_const = ctx.text.const_uint(i)
                ctx.text.emit_function(
                    f"{cmp_id} = OpIEqual {bool_t} {src_idx_id} {i_const}"
                )
                sel_id = ctx.text.alloc_id(f"frag_cvt_sel_{s}_{i}")
                ctx.text.emit_function(
                    f"{sel_id} = OpSelect {src_elem} {cmp_id} {loads[i]} {elem_in}"
                )
                elem_in = sel_id

        # Walk the body (optional). If no body, the transform is
        # identity — direct cast from src_dtype to dst_dtype.
        if op.body is not None:
            ctx.val_to_id[op.body_input_var.id] = elem_in
            if op.body_selector_var is not None:
                slot_to_sel = op.attrs.get("slot_to_selector_idx", ())
                if s < len(slot_to_sel):
                    sel_idx = slot_to_sel[s]
                    sel_v = op.operands[num_src + sel_idx]
                    ctx.val_to_id[op.body_selector_var.id] = ctx.val_to_id[sel_v.id]
            saved_stack = ctx.loop_yield_stack
            ctx.loop_yield_stack = []  # type: ignore[assignment]
            try:
                for body_op in op.body.ops:
                    if isinstance(body_op, YieldOp):
                        continue
                    _walk_op(body_op, ctx)
            finally:
                ctx.loop_yield_stack = saved_stack
            term = op.body.terminator
            transformed_id = ctx.val_to_id[term.operands[0].id]
        else:
            transformed_id = elem_in

        # Cast to dst_dtype if needed.
        if src_dtype is not dst_dtype:
            cast_id = ctx.text.alloc_id(f"frag_cvt_cast_{s}")
            # FConvert covers float→float across precisions; for
            # other combos we'd add OpConvert variants.
            if (_dtype_kind(src_dtype) == "float"
                    and _dtype_kind(dst_dtype) == "float"):
                ctx.text.emit_function(
                    f"{cast_id} = OpFConvert {dst_elem} {transformed_id}"
                )
            else:
                raise NotImplementedError(
                    f"_visit_frag_convert: cast {src_dtype} → {dst_dtype} "
                    "not yet wired"
                )
            transformed_id = cast_id

        # Store to dst scratch at the same idx.
        chain_out = ctx.text.alloc_id(f"frag_cvt_out_{s}")
        ctx.text.emit_function(
            f"{chain_out} = OpAccessChain {dst_elem_ptr} {dst_var} {idx_id}"
        )
        ctx.text.emit_function(
            f"OpStore {chain_out} {transformed_id}"
        )

    ctx.text.emit_function(
        f"OpControlBarrier {sg_scope} {sg_scope} {sg_sem}"
    )

    # Load dst scratch as the result coopmat (use=0, MatrixA).
    coop_t = ctx.text.type_coop_matrix(
        dst_elem, scope=3, rows=dst_rows, cols=dst_cols, use=0,
    )
    dst_base = ctx.text.alloc_id("frag_cvt_dst_base")
    ctx.text.emit_function(
        f"{dst_base} = OpAccessChain {dst_elem_ptr} {dst_var} {zero}"
    )
    res_id = ctx.text.alloc_id("frag_cvt_result")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCooperativeMatrixLoadKHR {coop_t} {dst_base} "
        f"{layout_id} {dst_cols_const}"
    )


def _visit_frag_apply(op: FragApplyOp, ctx: _SpvCtx) -> None:
    """``FragApplyOp`` via smem roundtrip — produces an output fragment.

    Same pattern as ``_visit_frag_for_each`` but writes the per-slot
    yielded values to an output scratch region and reloads them as
    a fresh cooperative matrix. Used by attention's online-softmax
    epilogue (``f32 → bf16`` cast with per-slot scale).

    The body's terminating ``YieldOp`` produces one Value per slot;
    we look up its SSA id after walking the body and store it to
    ``out_scratch[idx]``. After all slots, ``OpCooperativeMatrix
    LoadKHR`` from ``out_scratch`` gives us the result coopmat.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    _ensure_coopmat_caps(ctx)
    (out,) = op.results
    shape_id = op.attrs["shape_id"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_apply: unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    rows, cols, dtype = _coop_dims_for(shape, "c")
    n_elems = rows * cols
    subgroup_width = 32
    if n_elems % subgroup_width != 0:
        raise NotImplementedError(
            f"_visit_frag_apply: tile {rows}×{cols} not divisible by "
            f"{subgroup_width}"
        )
    n_slots_per_lane = n_elems // subgroup_width

    elem_type = _emit_dtype(ctx.text, dtype, ctx)
    if dtype is DType.BF16:
        ctx.text.add_capability("BFloat16CooperativeMatrixKHR")

    # Two scratch regions: one for the input fragment, one for the
    # output. Each is partitioned per-warp so multi-warp kernels don't
    # race on the same smem range — see ``_frag_scratch_warp_partition``.
    total_elems, smem_base_ssa, mat_base_ssa, warp_off = (
        _frag_scratch_warp_partition(ctx, n_elems, name_hint="frag_apply")
    )
    multi_warp = bool(warp_off)
    n_const = ctx.text.const_uint(total_elems)
    in_arr_id = ctx.text.alloc_id("frag_apply_in_arr")
    ctx.text.add_type_line(f"{in_arr_id} = OpTypeArray {elem_type} {n_const}")
    out_arr_id = ctx.text.alloc_id("frag_apply_out_arr")
    ctx.text.add_type_line(f"{out_arr_id} = OpTypeArray {elem_type} {n_const}")
    ptr_in_arr = ctx.text.type_pointer("Workgroup", in_arr_id)
    ptr_out_arr = ctx.text.type_pointer("Workgroup", out_arr_id)
    in_var = ctx.text.alloc_id("frag_apply_in")
    out_var = ctx.text.alloc_id("frag_apply_out")
    ctx.text.add_type_line(f"{in_var} = OpVariable {ptr_in_arr} Workgroup")
    ctx.text.add_type_line(f"{out_var} = OpVariable {ptr_out_arr} Workgroup")
    elem_ptr = ctx.text.type_pointer("Workgroup", elem_type)
    ctx.smem_allocs[id(op)] = (in_var, elem_type, elem_ptr, total_elems)
    ctx.smem_allocs[(id(op), "out")] = (out_var, elem_type, elem_ptr, total_elems)

    zero = ctx.text.const_uint(0)
    base_off = warp_off if multi_warp else zero
    base_ptr_in = ctx.text.alloc_id("frag_apply_base_in")
    ctx.text.emit_function(
        f"{base_ptr_in} = OpAccessChain {elem_ptr} {in_var} {base_off}"
    )
    base_ptr_out = ctx.text.alloc_id("frag_apply_base_out")
    ctx.text.emit_function(
        f"{base_ptr_out} = OpAccessChain {elem_ptr} {out_var} {base_off}"
    )

    cols_const = ctx.text.const_uint(cols)
    layout_id = ctx.text.const_uint(0)

    # Store input coopmat → in_scratch.
    in_id = ctx.val_to_id[op.operands[0].id]
    ctx.text.emit_function(
        f"OpCooperativeMatrixStoreKHR {base_ptr_in} {in_id} "
        f"{layout_id} {cols_const}"
    )

    sg_scope = ctx.text.const_uint(3)
    sg_sem = ctx.text.const_uint(0x8 | 0x100)
    ctx.text.emit_function(
        f"OpControlBarrier {sg_scope} {sg_scope} {sg_sem}"
    )

    u32 = ctx.text.type_int(32, signed=False)
    # ``slot_to_selector_idx`` absent + selectors present → dynamic
    # row dispatch. Compute the matrix-relative row from
    # ``mat_idx // cols`` per slot and OpSelect-chain the right
    # selector. Used by the SPV/Intel coopmat path where the
    # lane↔(row, col) mapping is implementation-private.
    slot_to_sel_attr = op.attrs.get("slot_to_selector_idx")
    n_selectors = len(op.operands) - 1
    dynamic_dispatch = (
        op.body_selector_var is not None
        and slot_to_sel_attr is None
        and n_selectors > 0
    )
    cols_const_div = ctx.text.const_uint(cols) if dynamic_dispatch else None
    bool_t = ctx.text.type_bool() if dynamic_dispatch else None

    for s in range(n_slots_per_lane):
        idx_id = _frag_scratch_slot_idx(
            ctx, base_ssa=smem_base_ssa, s=s,
            subgroup_width=subgroup_width, name_prefix="frag_apply",
        )

        chain_in = ctx.text.alloc_id(f"frag_apply_in_chain_{s}")
        ctx.text.emit_function(
            f"{chain_in} = OpAccessChain {elem_ptr} {in_var} {idx_id}"
        )
        elem_in = ctx.text.alloc_id(f"frag_apply_elem_{s}")
        ctx.text.emit_function(
            f"{elem_in} = OpLoad {elem_type} {chain_in}"
        )

        # Bind body input + walk body. Skip the terminating YieldOp
        # — its operand is the per-slot result, captured below.
        # Suppress any surrounding ``loop_yield_stack`` frame so a
        # nested if-with-carries inside the body still works
        # (its visitor pushes its own frame onto the empty stack).
        ctx.val_to_id[op.body_input_var.id] = elem_in
        if op.body_selector_var is not None:
            if dynamic_dispatch:
                # Compute matrix-relative idx + row at runtime, then
                # OpSelect-chain to pick the selector for this row.
                if multi_warp:
                    mat_idx_id = _frag_scratch_slot_idx(
                        ctx, base_ssa=mat_base_ssa, s=s,
                        subgroup_width=subgroup_width,
                        name_prefix=f"frag_apply_mat_{s}",
                    )
                else:
                    mat_idx_id = idx_id
                row_id = ctx.text.alloc_id(f"frag_apply_row_{s}")
                ctx.text.emit_function(
                    f"{row_id} = OpUDiv {u32} {mat_idx_id} {cols_const_div}"
                )
                # Selectors: operands[1 .. 1+n_selectors).
                sel_chain = ctx.val_to_id[op.operands[1 + n_selectors - 1].id]
                for i in range(n_selectors - 2, -1, -1):
                    cmp_id = ctx.text.alloc_id(f"frag_apply_cmp_{s}_{i}")
                    i_const = ctx.text.const_uint(i)
                    ctx.text.emit_function(
                        f"{cmp_id} = OpIEqual {bool_t} {row_id} {i_const}"
                    )
                    sel_lhs = ctx.val_to_id[op.operands[1 + i].id]
                    sel_id_new = ctx.text.alloc_id(f"frag_apply_sel_{s}_{i}")
                    sel_dtype_id = _emit_dtype(
                        ctx.text,
                        op.operands[1 + i].dtype,
                        ctx,
                    )
                    ctx.text.emit_function(
                        f"{sel_id_new} = OpSelect {sel_dtype_id} {cmp_id} "
                        f"{sel_lhs} {sel_chain}"
                    )
                    sel_chain = sel_id_new
                ctx.val_to_id[op.body_selector_var.id] = sel_chain
            else:
                # Static dispatch — slot_to_selector_idx maps each
                # slot to one of the ``selectors`` operands.
                slot_to_sel = slot_to_sel_attr or ()
                sel_idx = slot_to_sel[s] if s < len(slot_to_sel) else 0
                sel_v = op.operands[1 + sel_idx]
                ctx.val_to_id[op.body_selector_var.id] = ctx.val_to_id[sel_v.id]
        saved_stack = ctx.loop_yield_stack
        ctx.loop_yield_stack = []  # type: ignore[assignment]
        try:
            for body_op in op.body.ops:
                if isinstance(body_op, YieldOp):
                    continue  # captured via body.terminator below
                _walk_op(body_op, ctx)
        finally:
            ctx.loop_yield_stack = saved_stack
        # Capture the yielded value's SSA id from the body's
        # terminator and store to out_scratch[idx].
        body_term = op.body.terminator
        if body_term is None or not body_term.operands:
            raise RuntimeError(
                "_visit_frag_apply: body must yield exactly one value"
            )
        yielded_id = ctx.val_to_id[body_term.operands[0].id]
        chain_out = ctx.text.alloc_id(f"frag_apply_out_chain_{s}")
        ctx.text.emit_function(
            f"{chain_out} = OpAccessChain {elem_ptr} {out_var} {idx_id}"
        )
        ctx.text.emit_function(
            f"OpStore {chain_out} {yielded_id}"
        )

    # Subgroup barrier so all lanes' writes to out_scratch are
    # visible to the upcoming coop-matrix load.
    ctx.text.emit_function(
        f"OpControlBarrier {sg_scope} {sg_scope} {sg_sem}"
    )

    # Load out_scratch back as a coopmat — the result of FragApplyOp.
    coop_t = ctx.text.type_coop_matrix(
        elem_type, scope=3, rows=rows, cols=cols, use=2,
    )
    res_id = ctx.text.alloc_id("frag_apply_result")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCooperativeMatrixLoadKHR {coop_t} {base_ptr_out} "
        f"{layout_id} {cols_const}"
    )


def _visit_mma(op: MmaOp, ctx: _SpvCtx) -> None:
    """``MmaOp`` → ``OpCooperativeMatrixMulAddKHR``. ``D = A * B + C``."""
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    _ensure_coopmat_caps(ctx)
    (out,) = op.results
    shape_id = op.attrs["shape_id"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_mma: unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    # Result is the accumulator (use=2) shape with acc_dtype.
    rows, cols, dtype = _coop_dims_for(shape, "c")
    elem_type = _emit_dtype(ctx.text, dtype, ctx)
    if dtype is DType.BF16:
        ctx.text.add_capability("BFloat16CooperativeMatrixKHR")
    coop_t = ctx.text.type_coop_matrix(
        elem_type, scope=3, rows=rows, cols=cols, use=2,
    )

    # Operand layout: (a_frag, b_frag, c_frag)
    a_id = ctx.val_to_id[op.operands[0].id]
    b_id = ctx.val_to_id[op.operands[1].id]
    c_v = op.operands[2]
    c_id = ctx.val_to_id[c_v.id]
    # If C isn't already a coopmat (e.g. it's the kernel's
    # ``vec_build([zero_f] * N)`` init for a non-loop-carried MMA),
    # splat-convert it to the accumulator coopmat type. The
    # ``OpCompositeConstruct CoopMat scalar`` idiom fills the matrix
    # with the scalar — exactly what's needed for an all-same init.
    c_producer = c_v.producer
    if isinstance(c_producer, VecBuildOp):
        elem_dtype = c_v.dtype
        scalar_t = _emit_dtype(ctx.text, elem_dtype, ctx)
        scalar_id = ctx.text.alloc_id("coop_mma_c_splat")
        ctx.text.emit_function(
            f"{scalar_id} = OpCompositeExtract {scalar_t} {c_id} 0"
        )
        new_c = ctx.text.alloc_id("coop_mma_c_init")
        ctx.text.emit_function(
            f"{new_c} = OpCompositeConstruct {coop_t} {scalar_id}"
        )
        c_id = new_c

    res_id = ctx.text.alloc_id("coop_mma")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCooperativeMatrixMulAddKHR {coop_t} "
        f"{a_id} {b_id} {c_id}"
    )


_DISPATCH: dict[type, Any] = {
    ConstOp: _visit_const,
    ArithOp: _visit_arith,
    CmpOp: _visit_cmp,
    ConvertOp: _visit_convert,
    BitcastOp: _visit_bitcast,
    SplitB32Op: _visit_split_b32,
    MergeB32Op: _visit_merge_b32,
    MathOp: _visit_math,
    SelectOp: _visit_select,
    ThreadIdxOp: _visit_thread_idx,
    BlockIdxOp: _visit_block_idx,
    BlockDimOp: _visit_block_dim,
    LaneIdOp: _visit_lane_id,
    SubgroupIdOp: _visit_subgroup_id,
    GroupIdOp: _visit_group_id,
    ThreadIdInGroupOp: _visit_thread_id_in_group,
    LoadOp: _visit_load,
    StoreOp: _visit_store,
    AtomicRmwOp: _visit_atomic_rmw,
    LoadMatrixOp: _visit_load_matrix,
    StoreMatrixOp: _visit_store_matrix,
    MmaOp: _visit_mma,
    FragForEachOp: _visit_frag_for_each,
    FragApplyOp: _visit_frag_apply,
    FragConvertOp: _visit_frag_convert,
    FragReduceOp: _visit_frag_reduce,
    VecLoadOp: _visit_vec_load,
    VecStoreOp: _visit_vec_store,
    VecBuildOp: _visit_vec_build,
    VecExtractOp: _visit_vec_extract,
    IfRegionOp: _visit_if_region,
    ForLoopOp: _visit_for_loop,
    YieldOp: _visit_yield,
    SmemAllocOp: _visit_smem_alloc,
    BarrierOp: _visit_barrier,
    SubgroupReduceOp: _visit_subgroup_reduce,
    ShuffleOp: _visit_shuffle,
}


def _walk_op(op: Any, ctx: _SpvCtx) -> None:
    """Single-op dispatch — used by both the top-level body walker
    and recursive structured-control-flow visitors (``IfRegionOp``,
    eventually ``ForLoopOp`` / ``WhileLoopOp``)."""
    visitor = _DISPATCH.get(type(op))
    if visitor is None:
        raise NotImplementedError(
            f"SpirVLowerer: no visitor for {type(op).__name__}. "
            "See PORTABILITY_PLAN §3.2 — visitor coverage table."
        )
    visitor(op, ctx)


# ─────────────────────────────────────────────────────────────────
# Lowerer entry point.
# ─────────────────────────────────────────────────────────────────


class SpirVLowerer:
    """quark IR → SPIR-V text. PORTABILITY_PLAN §3.2 first cut."""

    def __init__(self, caps: Any = None, *, local_size: tuple[int, int, int] | None = None) -> None:
        self.caps = caps
        # Allow callers to override LocalSize for the entry point.
        # The IR doesn't carry it today, so the lowerer assumes
        # 1 workgroup and infers the per-workgroup thread count
        # from the caller (or defaults to 64).
        self._local_size = local_size

    def lower_module(self, module: Module) -> LoweredSpirVKernel:
        if not module.functions:
            raise ValueError("lower_module: module has no functions")
        if len(module.functions) > 1:
            raise NotImplementedError(
                "lower_module: multi-function modules not yet supported"
            )
        fn = module.functions[0]
        return self._lower_function(fn)

    def _lower_function(self, fn: Function) -> LoweredSpirVKernel:
        ctx = _SpvCtx()
        # Collect the GlobalTensor params in declaration order; the
        # SPIR-V binding index is the param's position in the
        # function's signature.
        ctx.tensor_to_binding = {}  # type: ignore[attr-defined]
        global_tensors = self._collect_global_tensors(fn, ctx)
        for binding_index, t in enumerate(global_tensors):
            ctx.tensor_to_binding[id(t.param)] = binding_index

        text = ctx.text
        # ── Boilerplate sections ─────────────────────────────────
        text.add_capability("Shader")
        # ── Function shell ──────────────────────────────────────
        void_t = text.type_void()
        fn_t = text.type_function(void_t)
        fn_id = text.alloc_id("main")
        text.emit_function(f"{fn_id} = OpFunction {void_t} None {fn_t}")
        entry_label = text.alloc_id("entry")
        text.emit_function(f"{entry_label} = OpLabel")

        # Stash the kernel's resolved local size so BlockDimOp visitors
        # can synthesise it as a constant.
        ctx.local_size = self._resolve_local_size(fn)

        # Walk the IR via the recursive ``_walk_op`` helper so the
        # if/while-region visitors can recurse into body ops without
        # duplicating the dispatch table. ``_ensure_buffer_var``
        # fires lazily off load/store; after the body walk, we
        # explicitly declare bindings for any unused params so the
        # final ``n_buffers`` matches the launcher's ``ParamSpec``
        # exactly (phantom storage buffers without OpAccessChain
        # references are valid Vulkan).
        for op in fn.body.ops:
            _walk_op(op, ctx)
        self._ensure_unused_bindings_declared(global_tensors, ctx)

        text.emit_function("OpReturn")
        text.emit_function("OpFunctionEnd")

        # Entry point + execution mode finalised once the body's
        # walked the builtin-input variables. SPIR-V 1.4+ (which the
        # ``vulkan1.3`` profile uses) requires the entry-point
        # interface list to reference every global variable the
        # function statically uses — Input, Output, AND
        # StorageBuffer / Uniform / PushConstant. Earlier specs
        # only required Input / Output, which is the rule the bare
        # GLSL→SPIR-V test fixture was built against; the framework
        # lowerer here goes through ``spirv-as`` at vulkan1.3 and
        # has to play by the stricter rule.
        local_size = ctx.local_size
        interface = []
        # Compute-shader builtin inputs the body uses. Vulkan 1.4
        # SPIR-V requires every statically-referenced global variable
        # in the entry-point interface — Input class for builtins,
        # StorageBuffer for buffers, Workgroup for smem.
        for var_attr in (
            "local_inv_id_var",
            "workgroup_id_var",
            "lane_id_var",
            "subgroup_id_var",
        ):
            v = getattr(ctx, var_attr, "")
            if v:
                interface.append(v)
        # Storage buffers in declaration order — matches binding
        # index, makes the disassembly readable.
        for t in global_tensors:
            var_id = ctx.tensor_to_var.get(id(t.param))
            if var_id is not None:
                interface.append(var_id)
        # Workgroup-class smem variables. Vulkan 1.4 SPIR-V requires
        # these in the interface alongside Input / Output / StorageBuffer
        # — earlier specs only required Input/Output, the broader rule
        # applies on the ``vulkan1.3`` profile we target.
        for _alloc_value_id, (var_id, *_rest) in ctx.smem_allocs.items():
            interface.append(var_id)
        text.add_entry_point(fn_id, "main", "GLCompute", interface)
        text.add_execution_mode(
            f"OpExecutionMode {fn_id} LocalSize {local_size[0]} {local_size[1]} {local_size[2]}"
        )

        return LoweredSpirVKernel(
            source=text.serialize(),
            entry_name="main",
            n_buffers=len(global_tensors),
            push_constants_size=0,
            smem_bytes=0,
            local_size=local_size,
        )

    def _collect_global_tensors(self, fn: Function, ctx: _SpvCtx) -> list[GlobalTensor]:
        """Collect GlobalTensors in **kernel-parameter order**.

        Binding indices feed into ``CompiledKernel.launch``'s per-
        buffer dispatch; the launcher passes buffers in
        ``ParamSpec.buffers`` order, which itself comes from the
        kernel function's parameter declaration order. Earlier this
        walked ops in encounter order (``LoadOp``s in the body), but
        kernels often reference parameters in a different order than
        they're declared (e.g. ``EulerStepKernel`` loads ``Dsig``
        first, then ``X``, ``V``, ``Out``). Encounter-order binding
        produced shader bindings out of phase with the framework's
        per-buffer launch list — kernel reads the wrong buffer.

        For every param the function declares, this returns a
        ``GlobalTensor`` (synthesising a phantom one for params the
        body never loads/stores from — e.g. ``ElementwiseKernel``'s
        ``Y`` slot for unary ops). The phantom binding is declared
        but never has an OpAccessChain referencing it, which Vulkan
        accepts as long as the descriptor-set binding is present in
        the pipeline layout. Phantom bindings keep the lowered
        ``n_buffers`` aligned with the launcher's ``ParamSpec`` so
        ``CompiledKernel.launch`` doesn't trip its buffer-count
        check on kernels with declared-but-unused params.
        """
        from quark.ir.types import BufferType

        used_param_ids: set[int] = set()
        param_to_tensor: dict[int, GlobalTensor] = {}

        def consider(t: GlobalTensor) -> None:
            param_to_tensor.setdefault(id(t.param), t)
            used_param_ids.add(id(t.param))

        def walk(ops):
            for op in ops:
                if isinstance(op, (LoadOp, StoreOp, VecLoadOp, VecStoreOp)):
                    t = op.attrs.get("tensor")
                    if isinstance(t, GlobalTensor):
                        consider(t)
                if isinstance(op, LoadMatrixOp):
                    t = op.attrs.get("src_tensor")
                    if isinstance(t, GlobalTensor):
                        consider(t)
                if isinstance(op, StoreMatrixOp):
                    t = op.attrs.get("dst_tensor")
                    if isinstance(t, GlobalTensor):
                        consider(t)
                if isinstance(op, AtomicRmwOp):
                    t = op.attrs.get("tensor")
                    if isinstance(t, GlobalTensor):
                        consider(t)
                for region in getattr(op, "regions", ()):
                    walk(region.ops)

        walk(fn.body.ops)

        # Order by the param's index in ``fn.params``. For params
        # without a load/store-discovered GlobalTensor, synthesise a
        # phantom shape=(1,) tensor so the binding gets declared and
        # the launcher's per-buffer dispatch lines up.
        ordered: list[GlobalTensor] = []
        for p in fn.params:
            if id(p) in param_to_tensor:
                ordered.append(param_to_tensor[id(p)])
                continue
            if isinstance(p.type, BufferType):
                phantom = GlobalTensor(
                    dtype=p.type.dtype,
                    shape=(1,),
                    stride=(1,),
                    name=f"{p.name}_unused",
                    param=p,
                )
                ordered.append(phantom)
        return ordered

    def _ensure_unused_bindings_declared(
        self, global_tensors: list[GlobalTensor], ctx: _SpvCtx,
    ) -> None:
        """Pre-declare a ``StorageBuffer`` ``OpVariable`` for every
        global tensor — even unused (phantom) ones — so the lowered
        kernel's ``n_buffers`` matches the launcher's ``ParamSpec``.

        Used tensors get their bindings lazily via ``_ensure_buffer_var``
        on first load/store; phantoms never see one of those visitors,
        so we proactively call ``_ensure_buffer_var`` here for every
        tensor that hasn't been emitted yet."""
        for t in global_tensors:
            if id(t.param) not in ctx.tensor_to_var:
                binding = ctx.tensor_to_binding[id(t.param)]
                _ensure_buffer_var(t, binding, ctx)

    def _resolve_local_size(self, fn: Function) -> tuple[int, int, int]:
        if self._local_size is not None:
            return self._local_size
        attr = getattr(fn, "attrs", None)
        if attr is not None:
            ls = getattr(attr, "local_size", None)
            if ls is not None:
                return tuple(ls)  # type: ignore[return-value]
        return (64, 1, 1)
