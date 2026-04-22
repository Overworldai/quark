"""MSL lowerer for the quark IR.

Walks a `Module` and emits an MSL kernel body suitable for
`mx.fast.metal_kernel`. The output is a **body string** (not a full
function) — MLX wraps it with the `[[kernel]]` signature, parameter
declarations, and built-in bindings.

The visitor methods and dispatch table live in `visitors.py` to keep
this file under the 800-line cap. See visitors.py for the per-op
codegen.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from quark.ir import (
    DType,
    Function,
    GlobalTensor,
    Module,
    Op,
    ScalarType,
    SharedRegion,
    SmemAllocOp,
    Value,
)
from quark.ir.module import BufferType
from quark.ir.tensor import Tensor

from .names import NameAlloc
from .types import msl_type

# ---------------------------------------------------------------------------
# Lowered artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweredMslKernel:
    """MSL kernel body + metadata for MLX's metal_kernel API.

    `source` is the kernel body (not a full function). MLX wraps it
    with the `[[kernel]] void` prefix and parameter bindings.
    """

    source: str
    header: str
    kernel_name: str
    smem_bytes: int
    input_names: list[str]
    output_names: list[str]
    scalar_names: list[str]
    atomic_outputs: bool
    template_args: list[tuple[str, Any]]

    def __str__(self) -> str:
        return self.source


# ---------------------------------------------------------------------------
# Per-function lowering context
# ---------------------------------------------------------------------------


@dataclass
class _MslCtx:
    """State tracked while lowering one Function."""

    module: Module | None = None
    lines: list[str] = field(default_factory=list)
    names: NameAlloc = field(default_factory=NameAlloc)
    indent: int = 1
    # Shared memory bookkeeping
    smem_bytes: int = 0
    smem_allocs: dict[int, tuple[str, int, int]] = field(default_factory=dict)
    # Parameter tracking
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    scalar_names: list[str] = field(default_factory=list)
    uses_atomics: bool = False
    uses_simdgroup_matrix: bool = False
    # Control-flow stack (for yield resolution)
    op_stack: list[Op] = field(default_factory=list)
    # Fragment tracking: Value.id → (array_name, acc_dtype_str, m_frags, n_frags).
    # Populated by LoadMatrixOp / MmaOp / Frag* op visitors. Consumers
    # (MmaOp, FragApply/Reduce/Convert/ForEachOp) look up the source
    # array name here instead of re-deriving from the Value's reg names.
    frag_values: dict[int, tuple[str, str, int, int]] = field(default_factory=dict)
    # Cache of b32-packed-fragment → simdgroup_matrix-array conversions.
    # Value.id → array name. Lets a single packed source feed multiple
    # MmaOps without re-emitting the unpack / simdgroup_load sequence.
    frag_packs: dict[int, str] = field(default_factory=dict)
    # Counter used to mint unique names for scratch buffers.
    _scratch_counter: int = 0
    # Insertion point (index into ``lines``) for hoisting threadgroup
    # declarations to function scope. Metal requires *every*
    # ``threadgroup`` decl at kernel-entry scope — a decl emitted
    # inside a for-loop body isn't visible elsewhere — so all
    # ``alloc_smem`` calls splice their decls here, then bump the
    # cursor by one. Initialised by ``lower_function`` after the
    # kernel-declared SmemAllocOps land, pointing at the first line
    # of the actual op body.
    _smem_decl_insert_idx: int = 0

    # Back-compat alias — older code paths still reference this name.
    # Points at ``_scratch_counter`` via a property.
    @property
    def frag_tmp_counter(self) -> int:
        return self._scratch_counter

    @frag_tmp_counter.setter
    def frag_tmp_counter(self, value: int) -> None:
        self._scratch_counter = value

    def alloc_smem(
        self,
        dtype: str,
        elems: int,
        name: str | None = None,
    ) -> str:
        """Allocate a threadgroup buffer. Emits a hoisted
        ``threadgroup T[N];`` decl at function scope and returns the
        buffer name. All kernel-declared smem (via ``SmemAllocOp``)
        routes through here. The ``smem_layout`` pass assigns offsets
        and aliasing — this function only emits the physical decl.

        The old ``aliasable=True`` pool was retired when every lowerer-
        internal scratch caller (extract_frag_scalars,
        _pack_scalars_to_frag_array) got migrated to Frag* primitives
        that stay in register. The smem_layout pass now handles all
        aliasing declaratively via ``Lifetime``.
        """
        # MSL dtype → bytes-per-element for the byte counter.
        _bpe = {
            "float": 4,
            "half": 2,
            "bfloat16_t": 2,
            "int": 4,
            "uint": 4,
            "short": 2,
            "ushort": 2,
            "char": 1,
            "uchar": 1,
        }.get(dtype, 4)
        if name is None:
            name = f"_scratch_{self._scratch_counter}"
            self._scratch_counter += 1
        self._hoist_smem_decl(dtype, name, elems)
        self.smem_bytes += elems * _bpe
        return name

    def _hoist_smem_decl(self, dtype: str, name: str, elems: int) -> int:
        """Insert a ``threadgroup`` declaration at function scope and
        return its line index. All MSL ``threadgroup`` decls must live
        at kernel-entry scope (Metal rejects them inside nested
        scopes), so callers from within the op walk splice here
        instead of emitting at the current indent."""
        # Use the current function-body indent (same as ``emit``)
        # so the hoisted line lines up with its siblings.
        line = ("    " * self.indent) + f"threadgroup {dtype} {name}[{elems}];"
        idx = self._smem_decl_insert_idx
        self.lines.insert(idx, line)
        self._smem_decl_insert_idx = idx + 1
        return idx

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def emit_blank(self) -> None:
        self.lines.append("")


# ---------------------------------------------------------------------------
# Helpers (shared by lower.py and visitors.py)
# ---------------------------------------------------------------------------

_ARITH_OP: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "rem": "%",
    "shl": "<<",
    "shr": ">>",
    "and": "&",
    "or": "|",
    "xor": "^",
}

_MATH_FN: dict[str, str] = {
    "rcp": "1.0f /",
    "rsqrt": "metal::rsqrt",
    "sqrt": "metal::sqrt",
    "exp2": "metal::exp2",
    "log2": "metal::log2",
    "sin": "metal::sin",
    "cos": "metal::cos",
    "tanh": "metal::tanh",
    "ex2_approx": "metal::fast::exp2",
    "rcp_approx": "metal::fast::divide",
    "rsqrt_approx": "metal::fast::rsqrt",
    "log2_approx": "metal::fast::log2",
    "sqrt_approx": "metal::fast::sqrt",
}

_CMP_OP: dict[str, str] = {
    "lt": "<",
    "le": "<=",
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "ge": ">=",
}

_SHUFFLE_FN: dict[str, str] = {
    "idx": "simd_shuffle",
    "xor": "simd_shuffle_xor",
    "up": "simd_shuffle_up",
    "down": "simd_shuffle_down",
}

_REDUCE_FN: dict[str, str] = {
    "sum": "simd_sum",
    "max": "simd_max",
    "min": "simd_min",
    "and": "simd_and",
    "or": "simd_or",
}

_ATOMIC_FN: dict[str, str] = {
    "add": "atomic_fetch_add_explicit",
    "sub": "atomic_fetch_sub_explicit",
    "min": "atomic_fetch_min_explicit",
    "max": "atomic_fetch_max_explicit",
    "and": "atomic_fetch_and_explicit",
    "or": "atomic_fetch_or_explicit",
    "xor": "atomic_fetch_xor_explicit",
    "exch": "atomic_exchange_explicit",
}


def _format_literal(dtype: DType, value: Any) -> str:
    """Format a Python numeric literal as MSL source text."""
    if dtype is DType.PRED:
        return "true" if value else "false"
    if dtype is DType.F32:
        raw = struct.pack("=f", float(value))
        bits = struct.unpack("=I", raw)[0]
        return f"as_type<float>(0x{bits:08x}u)"
    if dtype is DType.F64:
        raw = struct.pack("=d", float(value))
        bits = struct.unpack("=Q", raw)[0]
        return f"as_type<double>(0x{bits:016x}ul)"
    if dtype is DType.F16:
        return f"static_cast<half>({float(value)}f)"
    if dtype is DType.BF16:
        return f"static_cast<bfloat16_t>({float(value)}f)"
    if dtype.is_int or dtype.is_bit:
        suffix = "u" if not dtype.is_signed_int else ""
        return f"{int(value)}{suffix}"
    return str(value)


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) & ~(align - 1)


def _compute_tensor_offset(tensor: Tensor, indices: tuple[Value, ...], ctx: _MslCtx) -> str:
    """Compute a flat element offset expression for a tensor access."""
    parts: list[str] = []
    for i, idx in enumerate(indices):
        stride = tensor.stride[i]
        idx_name = ctx.names.name_for(idx)
        if stride == 1:
            parts.append(idx_name)
        else:
            parts.append(f"({idx_name} * {stride}u)")

    if isinstance(tensor, GlobalTensor):
        if tensor.static_row_offset and tensor.stride[0]:
            parts.append(f"{tensor.static_row_offset * tensor.stride[0]}u")
        if tensor.static_col_offset and len(tensor.stride) > 1 and tensor.stride[1]:
            parts.append(f"{tensor.static_col_offset * tensor.stride[1]}u")
        if tensor.dyn_row_offset is not None:
            dyn = ctx.names.name_for(tensor.dyn_row_offset)
            parts.append(f"({dyn} * {tensor.stride[0]}u)")
        if tensor.dyn_col_offset is not None:
            dyn = ctx.names.name_for(tensor.dyn_col_offset)
            s = tensor.stride[1] if len(tensor.stride) > 1 else 1
            parts.append(f"({dyn} * {s}u)")
    elif isinstance(tensor, SharedRegion):
        if tensor.static_offset:
            parts.append(f"{tensor.static_offset}u")
        if tensor.dyn_offset is not None:
            parts.append(ctx.names.name_for(tensor.dyn_offset))

    if not parts:
        return "0u"
    return " + ".join(parts)


def _tensor_buf_name(tensor: Tensor, ctx: _MslCtx) -> str:
    """Return the MSL buffer variable name for a tensor."""
    if isinstance(tensor, GlobalTensor):
        return tensor.param.name
    if isinstance(tensor, SharedRegion):
        alloc_id = tensor.alloc.id
        if alloc_id in ctx.smem_allocs:
            return ctx.smem_allocs[alloc_id][0]
        return f"smem_{alloc_id}"
    raise TypeError(f"Unknown tensor type: {type(tensor).__name__}")


# ---------------------------------------------------------------------------
# MslLowerer
# ---------------------------------------------------------------------------


class MslLowerer:
    """IR -> MSL kernel body text.

    Mirrors `PtxLowerer` so the launcher's `_lower()` stays symmetric.
    Visitor methods live in `visitors.py`; they are wired into the class
    via the `_DISPATCH` table imported below.
    """

    def __init__(self, device_caps=None, *, smem_aliasing: bool = True) -> None:
        self.caps = device_caps
        # Aliasing of disjoint-lifetime smem regions. ON by default —
        # the layout pass uses AUTO lifetime inference (first_use →
        # last_use+1) which is conservative enough that mis-aliasing is
        # a bug in the lifetime-inference logic, not a correctness risk
        # at the user level. Kernel authors get tighter aliasing by
        # annotating ``Lifetime.in_region(...)`` explicitly.
        self._smem_aliasing_enabled = smem_aliasing

    def lower_module(self, module: Module) -> LoweredMslKernel:
        if not module.functions:
            raise ValueError("lower_module: module has no functions")
        if len(module.functions) > 1:
            raise NotImplementedError("lower_module: multi-function modules not yet supported")
        fn = module.functions[0]
        return self.lower_function(fn, module)

    def lower_function(self, fn: Function, module: Module | None = None) -> LoweredMslKernel:
        ctx = _MslCtx(module=module)
        # Pre-detect MMA usage so for-loop result declarations can
        # allocate simdgroup_matrix arrays for accumulator carries.
        if module and module.kernel_shapes:
            for shape in module.kernel_shapes.values():
                if shape.msl:
                    ctx.uses_simdgroup_matrix = True
                    break
        self._classify_params(fn, ctx)
        self._emit_smem_allocs(fn, ctx)
        # Freeze the "top of function body" line index now that every
        # kernel-declared smem alloc has landed. Subsequent
        # ``ctx.alloc_smem`` calls from op visitors splice their
        # threadgroup decls at this cursor (Metal requires every
        # threadgroup decl at kernel-entry scope — can't sit inside
        # a for-loop body or conditional).
        ctx._smem_decl_insert_idx = len(ctx.lines)
        # Hoist all ConstOps to the top of the function so their
        # declarations aren't trapped inside for-loop scopes. CSE in
        # the IR builder means a const created inside a loop body may
        # be reused in the epilogue — MSL's C-style scoping needs it
        # declared at function scope.
        self._hoist_consts(fn.body.ops, ctx)
        self._walk_region(fn.body.ops, ctx)
        header = self._build_header(ctx)
        source = "\n".join(ctx.lines)
        return LoweredMslKernel(
            source=source,
            header=header,
            kernel_name=f"quark_{fn.name}",
            smem_bytes=ctx.smem_bytes,
            input_names=ctx.input_names,
            output_names=ctx.output_names,
            scalar_names=ctx.scalar_names,
            atomic_outputs=ctx.uses_atomics,
            template_args=[],
        )

    def _classify_params(self, fn: Function, ctx: _MslCtx) -> None:
        for p in fn.params:
            if isinstance(p.type, BufferType):
                if p.attrs.readonly:
                    ctx.input_names.append(p.name)
                else:
                    ctx.output_names.append(p.name)
            elif isinstance(p.type, ScalarType):
                ctx.scalar_names.append(p.name)
                ctx.input_names.append(p.name)

    def _emit_smem_allocs(self, fn: Function, ctx: _MslCtx) -> None:
        """Emit every IR-level SmemAllocOp via the layout plan.

        Calls ``compute_smem_layout(fn)`` to get per-region slot
        assignments. Each unique slot gets exactly one
        ``threadgroup`` decl; multiple SmemAllocOps mapped to the same
        slot share that decl (aliased storage). Lowerer-internal scratch
        (MMA extract/pack) still goes through ``ctx.alloc_smem`` for
        backwards compatibility.
        """
        import os as _os

        from quark.lower.smem_layout import compute_smem_layout, dump_smem_layout

        plan = compute_smem_layout(fn, enable_aliasing=self._smem_aliasing_enabled)
        if _os.environ.get("QUARK_DUMP_SMEM_LAYOUT", "").lower() in ("1", "true", "yes", "on"):
            print(dump_smem_layout(fn, plan, label="(msl)"), flush=True)

        # Emit one decl per slot. Pick a representative SmemAllocOp from
        # each slot to derive the dtype + name; aliased members reuse
        # the slot's buffer name.
        slot_to_repr: dict[int, SmemAllocOp] = {}
        for op in fn.smem_allocs:
            (backing,) = op.results
            slot = plan.region_to_slot.get(backing.id)
            if slot is None:
                continue
            slot_to_repr.setdefault(slot, op)

        slot_buf_name: dict[int, str] = {}
        for slot, repr_op in slot_to_repr.items():
            dtype: DType = repr_op.attrs["dtype"]
            slot_size_bytes = plan.slot_sizes[slot]
            elems = slot_size_bytes // dtype.bytes
            (backing,) = repr_op.results
            var_name = ctx.alloc_smem(msl_type(dtype), elems, f"smem_{backing.id}")
            slot_buf_name[slot] = var_name

        # Bind every SmemAllocOp's backing Value to its slot's buffer.
        # Aliased ops share the same C-level buffer name; views handle
        # static_offset arithmetic from the SharedRegion side.
        for op in fn.smem_allocs:
            (backing,) = op.results
            slot = plan.region_to_slot.get(backing.id)
            if slot is None:
                continue
            var_name = slot_buf_name[slot]
            offset = plan.slot_offsets[slot]
            size_bytes = plan.slot_sizes[slot]
            ctx.smem_allocs[backing.id] = (var_name, offset, size_bytes)
            if not ctx.names.has(backing):
                ctx.names.bind(backing, (var_name,))

    def _hoist_consts(self, ops: list[Op], ctx: _MslCtx) -> None:
        """Pre-emit all ConstOps at function scope to avoid C scoping issues.

        The IR builder's CSE creates consts inside loop bodies that may be
        referenced in the epilogue. On PTX (flat register scope) this is
        fine; on MSL the for-loop's C braces hide them. We solve it by
        emitting every const declaration at the top, then _visit_const
        becomes a no-op for already-declared values.
        """
        from quark.ir import ConstOp as _ConstOp

        for op in ops:
            if isinstance(op, _ConstOp):
                # Emit at current (function-level) indent.
                from .visitors import _visit_const

                _visit_const(self, op, ctx)
            # Recurse into sub-regions (for-loop body, if arms, etc.)
            for region in op.regions:
                self._hoist_consts(region.ops, ctx)

    def _build_header(self, ctx: _MslCtx) -> str:
        parts: list[str] = []
        if ctx.uses_simdgroup_matrix:
            parts.append("#include <metal_simdgroup>")
            parts.append("#include <metal_simdgroup_matrix>")
        if parts:
            return "\n".join(parts) + "\n"
        return ""

    def _walk_region(self, ops: list[Op], ctx: _MslCtx) -> None:
        for op in ops:
            self._visit(op, ctx)

    def _visit(self, op: Op, ctx: _MslCtx) -> None:
        from .visitors import DISPATCH

        handler = DISPATCH.get(type(op))
        if handler is None:
            raise NotImplementedError(f"MslLowerer: no handler for {type(op).__name__}")
        handler(self, op, ctx)
