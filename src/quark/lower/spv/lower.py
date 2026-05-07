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
    BarrierOp,
    BlockDimOp,
    BlockIdxOp,
    CmpOp,
    ConstOp,
    ConvertOp,
    GroupIdOp,
    IfRegionOp,
    LaneIdOp,
    LoadOp,
    MathOp,
    SelectOp,
    SmemAllocOp,
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
    # Smem allocations: SmemAllocOp result Value.id → (var_id,
    # element_type_id, elem_pointer_id, total_elements). Visitors
    # that load/store on a SharedRegion look up by the
    # SharedRegion.alloc.id (which equals the SmemAllocOp's
    # backing Value id).
    smem_allocs: dict[int, tuple[str, str, str, int]] = field(default_factory=dict)


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
        # BFloat16 is an OpTypeFloat 16 with a Width=16 BFloat16
        # tag — but ``spirv-as`` accepts ``OpTypeFloat 16 BFloat16``
        # only on the ``vulkan1.4`` profile. For now we emit the same
        # ``OpTypeFloat 16`` and rely on the capability gating + the
        # KHR_shader_bfloat16 storage flag at the buffer / coopmat
        # site to keep it interpreted as bf16.
        # TODO: re-emit as ``OpTypeFloat 16 BFloat16KHR`` once the
        # framework standardises on Vulkan 1.4 (see PORTABILITY_PLAN
        # §3.2 sub-table for the version bump).
        return text.type_float(16)
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
    """Structured if/else with no carries (the bounds-check pattern).

    SPIR-V structured control flow:
      OpSelectionMerge merge None
      OpBranchConditional pred then_label else_label
      then_label = OpLabel
        ...then body ops...
        OpBranch merge
      else_label = OpLabel
        ...else body ops...   (empty if no else)
        OpBranch merge
      merge = OpLabel

    Carries (operands beyond the predicate, results from the op) need
    OpPhi at the merge block to thread per-arm-yielded values
    forward. Deferred — every in-tree kernel I've checked uses
    if-without-carries today; the carry path lands when a kernel
    needs it. ``YieldOp`` body terminators are simply walked inside
    each region; they emit their operand into the val map, no phi.
    """
    if op.results:
        raise NotImplementedError(
            "_visit_if_region: result-yielding (carried) if/else "
            "needs OpPhi merge — not yet wired. See PORTABILITY_PLAN "
            "§3.2."
        )
    pred_id = ctx.val_to_id[op.pred.id]

    merge_label = ctx.text.alloc_id("if_merge")
    then_label = ctx.text.alloc_id("if_then")
    else_label = ctx.text.alloc_id("if_else")

    ctx.text.emit_function(f"OpSelectionMerge {merge_label} None")
    ctx.text.emit_function(
        f"OpBranchConditional {pred_id} {then_label} {else_label}"
    )

    # Then arm
    ctx.text.emit_function(f"{then_label} = OpLabel")
    for body_op in op.then_region.ops:
        _walk_op(body_op, ctx)
    ctx.text.emit_function(f"OpBranch {merge_label}")

    # Else arm — may be empty.
    ctx.text.emit_function(f"{else_label} = OpLabel")
    for body_op in op.else_region.ops:
        _walk_op(body_op, ctx)
    ctx.text.emit_function(f"OpBranch {merge_label}")

    # Merge block. Subsequent ops in the parent region land here.
    ctx.text.emit_function(f"{merge_label} = OpLabel")


def _visit_yield(op: YieldOp, ctx: _SpvCtx) -> None:
    """``YieldOp`` inside if/for body. With no carries (the path
    we support today) it's a no-op: the region's terminator gets
    handled by the surrounding visitor's ``OpBranch`` emit.

    When carries land (PORTABILITY_PLAN §3.2 follow-up), this
    visitor will record the per-arm yielded values for the surrounding
    op's OpPhi emit at merge.
    """
    if op.operands:
        raise NotImplementedError(
            "_visit_yield: yielding values from a region requires "
            "OpPhi support (carry path). See PORTABILITY_PLAN §3.2."
        )


def _ensure_buffer_var(tensor: GlobalTensor, binding_index: int,
                       ctx: _SpvCtx) -> tuple[str, str]:
    """Declare the storage-buffer variable for ``tensor`` at
    ``binding_index`` if not already present. Returns
    ``(buffer_var_id, elem_pointer_type_id)`` for the load/store
    visitors to use."""
    tid = id(tensor)
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


def _flatten_index(indices: tuple, shape: tuple, ctx: _SpvCtx) -> str:
    """Compute a row-major flat index from N-D indices + shape.

    Returns the SSA id of the flat-index ``OpIAdd``. For 1-D this
    is just the single index; for N-D, emits the ``i*S2 + j*S3 +
    k`` chain. ``shape`` is the IR's declared shape (post-pad).
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


def _visit_load(op: LoadOp, ctx: _SpvCtx) -> None:
    (out,) = op.results
    tensor = op.attrs["tensor"]
    if isinstance(tensor, GlobalTensor):
        binding = ctx.tensor_to_binding[id(tensor)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        elem_type = ctx.tensor_to_elem_type[id(tensor)]
        zero = ctx.text.const_uint(0)
        idx_id = _flatten_index(tuple(op.operands), tensor.shape, ctx)
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
        binding = ctx.tensor_to_binding[id(tensor)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        value_id = ctx.val_to_id[op.operands[0].id]
        zero = ctx.text.const_uint(0)
        idx_id = _flatten_index(tuple(op.operands[1:]), tensor.shape, ctx)
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
        chain_id = ctx.text.alloc_id("smem_chain")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {idx_id}"
        )
        ctx.text.emit_function(f"OpStore {chain_id} {value_id}")
        return

    raise NotImplementedError(
        f"_visit_store: tensor type {type(tensor).__name__} not wired"
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
        binding = ctx.tensor_to_binding[id(tensor)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        elem_type = ctx.tensor_to_elem_type[id(tensor)]
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
    base_id = _flatten_index(tuple(indices), tensor.shape, ctx)
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
        binding = ctx.tensor_to_binding[id(tensor)]
        var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
        elem_type = ctx.tensor_to_elem_type[id(tensor)]
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

    base_id = _flatten_index(tuple(indices), tensor.shape, ctx)
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
    """``OpCompositeConstruct`` from N scalar operands."""
    (out,) = op.results
    elem_type = _emit_dtype(ctx.text, op.operands[0].dtype, ctx)
    vec_type = ctx.text.type_vec(elem_type, len(op.operands))
    res_id = ctx.text.alloc_id("vec_build")
    ctx.val_to_id[out.id] = res_id
    operand_ids = [ctx.val_to_id[v.id] for v in op.operands]
    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {vec_type} {' '.join(operand_ids)}"
    )


def _visit_vec_extract(op: VecExtractOp, ctx: _SpvCtx) -> None:
    """``OpCompositeExtract`` — pick one component from a vector."""
    (out,) = op.results
    idx = int(op.attrs["index"])
    elem_type = _emit_dtype(ctx.text, out.dtype, ctx)
    src_id = ctx.val_to_id[op.operands[0].id]
    res_id = ctx.text.alloc_id(f"vec_x{idx}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCompositeExtract {elem_type} {src_id} {idx}"
    )


_DISPATCH: dict[type, Any] = {
    ConstOp: _visit_const,
    ArithOp: _visit_arith,
    CmpOp: _visit_cmp,
    ConvertOp: _visit_convert,
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
    VecLoadOp: _visit_vec_load,
    VecStoreOp: _visit_vec_store,
    VecBuildOp: _visit_vec_build,
    VecExtractOp: _visit_vec_extract,
    IfRegionOp: _visit_if_region,
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
            ctx.tensor_to_binding[id(t)] = binding_index

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
        # fires lazily off load/store so an unused param doesn't
        # produce a SPIR-V validation error.
        for op in fn.body.ops:
            _walk_op(op, ctx)

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
            var_id = ctx.tensor_to_var.get(id(t))
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

        Walks IR ops to discover *which* params are actually used,
        then orders by ``fn.params`` index. Unused params don't get
        a binding (Vulkan rejects unreferenced descriptor-set
        bindings on some configurations); they take a slot in the
        launcher's buffer list anyway, but the lowerer skips them.
        """
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
                for region in getattr(op, "regions", ()):
                    walk(region.ops)

        walk(fn.body.ops)

        # Order by the param's index in ``fn.params``, so bindings 0
        # ..n-1 line up with the launcher's per-buffer launch list.
        ordered: list[GlobalTensor] = []
        for p in fn.params:
            if id(p) in used_param_ids and id(p) in param_to_tensor:
                ordered.append(param_to_tensor[id(p)])
        return ordered

    def _resolve_local_size(self, fn: Function) -> tuple[int, int, int]:
        if self._local_size is not None:
            return self._local_size
        attr = getattr(fn, "attrs", None)
        if attr is not None:
            ls = getattr(attr, "local_size", None)
            if ls is not None:
                return tuple(ls)  # type: ignore[return-value]
        return (64, 1, 1)
