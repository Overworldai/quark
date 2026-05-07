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
    BlockDimOp,
    BlockIdxOp,
    ConstOp,
    LoadOp,
    StoreOp,
    ThreadIdxOp,
)
from quark.ir.tensor import GlobalTensor

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
    # Cached SSA id for LocalInvocationId — the builtin variable + a
    # convenience accessor that loads its .x component.
    local_inv_id_var: str = ""
    local_inv_id_x_loaded: str = ""
    # Cached push-constant block id, if any.
    push_block_var: str = ""
    push_block_size: int = 0


# Map quark DType values → (SPIR-V type emitter method name, byte width).
# Anything not in this map causes a NotImplementedError so missing
# coverage surfaces fast rather than silently emitting wrong types.
_DTYPE_TO_SPIR: dict[DType, tuple[str, int]] = {
    DType.F32: ("type_float", 4),
    DType.U32: ("type_int_unsigned", 4),
    DType.S32: ("type_int_signed", 4),
}


def _emit_dtype(text: SpvText, dt: DType) -> str:
    if dt is DType.F32:
        return text.type_float(32)
    if dt is DType.U32:
        return text.type_int(32, signed=False)
    if dt is DType.S32:
        return text.type_int(32, signed=True)
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
    type_id = _emit_dtype(ctx.text, out.dtype)
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
_ARITH_KIND_TO_SPV: dict[tuple[str, DType], str] = {
    ("add", DType.F32): "OpFAdd",
    ("sub", DType.F32): "OpFSub",
    ("mul", DType.F32): "OpFMul",
    ("div", DType.F32): "OpFDiv",
    ("add", DType.U32): "OpIAdd",
    ("sub", DType.U32): "OpISub",
    ("mul", DType.U32): "OpIMul",
    ("add", DType.S32): "OpIAdd",
    ("sub", DType.S32): "OpISub",
    ("mul", DType.S32): "OpIMul",
}


def _ensure_local_invocation_id(ctx: _SpvCtx) -> str:
    """Return the SSA id of LocalInvocationId.x, declaring the
    builtin variable + a u32 load if not already present."""
    if ctx.local_inv_id_x_loaded:
        return ctx.local_inv_id_x_loaded
    if not ctx.local_inv_id_var:
        u32 = ctx.text.type_int(32, signed=False)
        v3u = ctx.text.type_vec(u32, 3)
        ptr = ctx.text.type_pointer("Input", v3u)
        var_id = ctx.text.alloc_id("LocalInvocationId")
        ctx.text.add_type_line(f"{var_id} = OpVariable {ptr} Input")
        ctx.text.add_decoration(f"OpDecorate {var_id} BuiltIn LocalInvocationId")
        ctx.local_inv_id_var = var_id
    # Load the full vec3, then extract .x.
    u32 = ctx.text.type_int(32, signed=False)
    v3u = ctx.text.type_vec(u32, 3)
    loaded = ctx.text.alloc_id("liid_vec")
    ctx.text.emit_function(f"{loaded} = OpLoad {v3u} {ctx.local_inv_id_var}")
    x_id = ctx.text.alloc_id("liid_x")
    ctx.text.emit_function(f"{x_id} = OpCompositeExtract {u32} {loaded} 0")
    ctx.local_inv_id_x_loaded = x_id
    return x_id


def _visit_thread_idx(op: ThreadIdxOp, ctx: _SpvCtx) -> None:
    """Today only handles the .x component (the only one vec_add
    needs). .y / .z extend trivially; defer until a kernel demands
    them."""
    (out,) = op.results
    dim = op.attrs.get("dim", "x")
    if dim != "x":
        raise NotImplementedError(
            f"_visit_thread_idx: dim={dim!r} not yet wired (only 'x'). "
            "See PORTABILITY_PLAN §3.2."
        )
    x_id = _ensure_local_invocation_id(ctx)
    ctx.val_to_id[out.id] = x_id


def _visit_block_idx(op: BlockIdxOp, ctx: _SpvCtx) -> None:
    raise NotImplementedError(
        "_visit_block_idx: WorkgroupId emit deferred to next visitor "
        "bundle. v1 vec_add uses LocalInvocationId only (single-"
        "workgroup dispatch). See PORTABILITY_PLAN §3.2."
    )


def _visit_block_dim(op: BlockDimOp, ctx: _SpvCtx) -> None:
    raise NotImplementedError(
        "_visit_block_dim: emit a constant from LocalSize. Deferred to "
        "the next visitor bundle. See PORTABILITY_PLAN §3.2."
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

    elem_type = _emit_dtype(ctx.text, tensor.dtype)
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


def _visit_load(op: LoadOp, ctx: _SpvCtx) -> None:
    (out,) = op.results
    tensor = op.attrs["tensor"]
    if not isinstance(tensor, GlobalTensor):
        raise NotImplementedError(
            "_visit_load: only GlobalTensor sources are wired today. "
            "SharedRegion loads land with the threadgroup-memory "
            "visitor bundle in PORTABILITY_PLAN §3.2."
        )
    if len(op.operands) != 1:
        raise NotImplementedError(
            f"_visit_load: only 1D loads wired (got {len(op.operands)} "
            "indices). N-D loads need stride math; defer until a kernel "
            "demands them."
        )

    # Resolve the binding index from the function's param ordering —
    # set by the lowerer below before walking the body.
    binding = ctx.tensor_to_binding[id(tensor)]
    var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
    elem_type = ctx.tensor_to_elem_type[id(tensor)]

    idx_id = ctx.val_to_id[op.operands[0].id]
    zero = ctx.text.const_uint(0)

    chain_id = ctx.text.alloc_id("chain")
    ctx.text.emit_function(
        f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {zero} {idx_id}"
    )
    res_id = ctx.text.alloc_id("ld")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpLoad {elem_type} {chain_id}")


def _visit_store(op: StoreOp, ctx: _SpvCtx) -> None:
    tensor = op.attrs["tensor"]
    if not isinstance(tensor, GlobalTensor):
        raise NotImplementedError(
            "_visit_store: only GlobalTensor sinks are wired today."
        )
    if len(op.operands) != 2:
        raise NotImplementedError(
            f"_visit_store: only 1D stores wired (got "
            f"{len(op.operands) - 1} indices)."
        )
    binding = ctx.tensor_to_binding[id(tensor)]
    var_id, elem_ptr = _ensure_buffer_var(tensor, binding, ctx)
    value_id = ctx.val_to_id[op.operands[0].id]
    idx_id = ctx.val_to_id[op.operands[1].id]
    zero = ctx.text.const_uint(0)
    chain_id = ctx.text.alloc_id("chain")
    ctx.text.emit_function(
        f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {zero} {idx_id}"
    )
    ctx.text.emit_function(f"OpStore {chain_id} {value_id}")


_DISPATCH: dict[type, Any] = {
    ConstOp: _visit_const,
    ArithOp: _visit_arith,
    ThreadIdxOp: _visit_thread_idx,
    BlockIdxOp: _visit_block_idx,
    BlockDimOp: _visit_block_dim,
    LoadOp: _visit_load,
    StoreOp: _visit_store,
}


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

        # Walk the IR. We drive _ensure_buffer_var lazily off
        # _visit_load / _visit_store so an unused param doesn't get
        # declared (a binding declared but not used is a SPIR-V
        # validation error in some configurations).
        for op in fn.body.ops:
            visitor = _DISPATCH.get(type(op))
            if visitor is None:
                raise NotImplementedError(
                    f"SpirVLowerer: no visitor for {type(op).__name__}. "
                    "See PORTABILITY_PLAN §3.2 — visitor coverage table."
                )
            visitor(op, ctx)

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
        local_size = self._resolve_local_size(fn)
        interface = []
        if ctx.local_inv_id_var:
            interface.append(ctx.local_inv_id_var)
        # Storage buffers in declaration order — matches binding
        # index, makes the disassembly readable.
        for binding_index, t in enumerate(global_tensors):
            var_id = ctx.tensor_to_var.get(id(t))
            if var_id is not None:
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
        """Walk the IR's load/store ops and collect the GlobalTensor
        objects in the order they're first referenced."""
        seen: dict[int, GlobalTensor] = {}
        order: list[GlobalTensor] = []

        def consider(t: GlobalTensor) -> None:
            if id(t) in seen:
                return
            seen[id(t)] = t
            order.append(t)

        for op in fn.body.ops:
            if isinstance(op, (LoadOp, StoreOp)):
                t = op.attrs.get("tensor")
                if isinstance(t, GlobalTensor):
                    consider(t)
        return order

    def _resolve_local_size(self, fn: Function) -> tuple[int, int, int]:
        if self._local_size is not None:
            return self._local_size
        attr = getattr(fn, "attrs", None)
        if attr is not None:
            ls = getattr(attr, "local_size", None)
            if ls is not None:
                return tuple(ls)  # type: ignore[return-value]
        return (64, 1, 1)
