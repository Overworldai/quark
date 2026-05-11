"""MSL lowerer for the quark IR.

Walks a ``Module`` and emits an MSL kernel body. The output is a
**body string** (not a full function) — the harness
(``drivers/metal_harness.py``) wraps it with the ``[[kernel]]``
signature, parameter declarations, and built-in bindings.

The visitor methods and dispatch table live in ``visitors.py`` to keep
this file under the 800-line cap.

EXEMPT FROM 500-LINE RULE: this file owns the MSL lowering driver —
LowerCtx state, NameMap, smem layout planning, the recursive
``_emit_*`` helpers for control flow, and the public ``lower_module``
/ ``LoweredMslKernel`` API. Splitting further would fragment the
ctx/dispatch coupling; the visitor catalog already lives in
``visitors.py``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from quark.ir import (
    AtomicRmwOp,
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
    """MSL kernel body + metadata for the Metal driver.

    `source` is the kernel body (not a full function). The harness
    wraps it with the ``[[kernel]] void`` prefix and parameter bindings.
    """

    source: str
    header: str
    kernel_name: str
    smem_bytes: int
    input_names: list[str]
    output_names: list[str]
    scalar_names: list[str]
    # True if any AtomicRmwOp was emitted. Kept for backwards-compat;
    # callers picking the per-output qualifier should consult
    # ``atomic_output_names`` instead so kernels with a mix of atomic and
    # plain-store outputs (e.g. moe_router_correct) only flag the
    # buffers actually targeted by atomics.
    atomic_outputs: bool
    template_args: list[tuple[str, Any]]
    # Per-param MSL dtype strings (e.g. "float", "half", "bfloat").
    # Parallel to input_names / output_names / scalar_names respectively.
    # Used by the metal driver's harness to emit typed pointer params
    # (``device float*`` vs ``device half*``).
    input_dtypes: list[str] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)
    scalar_dtypes: list[str] = field(default_factory=list)
    # Subset of ``output_names`` that an AtomicRmwOp targets. The metal
    # harness uses this set (not ``atomic_outputs``) to decide which
    # outputs get the ``device atomic<T>*`` qualifier.
    atomic_output_names: frozenset[str] = field(default_factory=frozenset)

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
    input_dtypes: list[str] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)
    scalar_dtypes: list[str] = field(default_factory=list)
    uses_atomics: bool = False
    # Per-output-name set of buffers targeted by an AtomicRmwOp. Drives
    # which outputs get the ``device atomic<T>*`` vs ``device T*``
    # parameter qualifier in the harness. A kernel-wide flag would taint
    # every output (e.g. moe_router_correct: ``counts`` is atomic but
    # ``token_ids`` / ``slot_weights`` / ``offsets`` get plain stores —
    # tagging them all atomic breaks the plain-store paths).
    atomic_output_names: set[str] = field(default_factory=set)
    uses_simdgroup_matrix: bool = False
    # Set when any MmaOp this kernel emits resolves to a NAX (MPP
    # matmul2d) payload. Drives the ``#include
    # <MetalPerformancePrimitives/...>`` injection and forces MSL 4.0
    # at compile time. ``uses_simdgroup_matrix`` and this can both be
    # true in a single kernel: the simdgroup_matrix path still owns
    # m8n8k8/m16n8 shapes that NAX doesn't replace.
    uses_nax: bool = False
    # Per-kernel-function flag tracking whether the NAX preamble
    # (Coord struct, descriptor, gemm_op handle) has been emitted
    # yet. The visit_mma_nax visitor emits the preamble lazily on
    # the first NAX MmaOp it sees within a function body.
    nax_preamble_emitted: bool = False
    # Control-flow stack (for yield resolution)
    op_stack: list[Op] = field(default_factory=list)
    # Fragment tracking: Value.id → (array_name, acc_dtype_str, m_frags, n_frags).
    # Populated by LoadMatrixOp / MmaOp / Frag* op visitors. Consumers
    # (MmaOp, FragApply/Reduce/Convert/ForEachOp) look up the source
    # array name here instead of re-deriving from the Value's reg names.
    frag_values: dict[int, tuple[str, str, int, int]] = field(default_factory=dict)
    # Discriminator set: Value.id of fragments whose MSL storage is the
    # NAX per-lane vec<T, 8>[n_frags] form (rather than the default
    # simdgroup_matrix<T, 8, 8>[mf*nf] form). Frag* visitors check this
    # set to dispatch to NAX-specific emission. Populated by
    # ``_visit_mma_nax`` and the NAX LoadMatrix / FragConvert visitors.
    nax_frag_ids: set[int] = field(default_factory=set)
    # NAX fragment Value.id → ``(array_name, n_frags)`` lookup. Tells
    # the MMA helper-emission path which array to pass by reference
    # when invoking the inlined ``nax_mma_*`` helper. Populated by
    # ``_emit_nax_frag_decl`` and similar array-bind sites.
    nax_frag_arrays: dict[int, tuple[str, int]] = field(default_factory=dict)
    # NAX MMA helper variants used by the kernel (one definition per
    # unique ``(shape_id, transpose_b, accumulate, cast_a)`` tuple).
    # Definitions are emitted in the kernel header so each is declared
    # exactly once even if the helper is called many times. The set
    # tracks which we've seen; the list tracks emission order.
    nax_mma_helpers: set[tuple] = field(default_factory=set)
    nax_mma_helper_defs: list[str] = field(default_factory=list)
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
            "bfloat": 2,
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
    "exp": "metal::exp",
    "exp2": "metal::exp2",
    "log2": "metal::log2",
    "sin": "metal::sin",
    "cos": "metal::cos",
    "tanh": "metal::tanh",
    "exp_approx": "metal::fast::exp",
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
        return f"static_cast<bfloat>({float(value)}f)"
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
        # NAX shapes (payload starting with ``nax:``) use cooperative
        # tensors and per-lane scalar carries, NOT simdgroup_matrix —
        # don't flip the flag for them.
        if module and module.kernel_shapes:
            from quark.device import DeviceFamily
            from quark.ir.mma_registry import payload_for

            for shape in module.kernel_shapes.values():
                payload = payload_for(shape.name, DeviceFamily.METAL)
                if payload is None:
                    continue
                if payload.startswith("nax:"):
                    ctx.uses_nax = True
                else:
                    ctx.uses_simdgroup_matrix = True
                    break
        self._classify_params(fn, ctx)
        self._prescan_atomic_targets(fn, ctx)
        self._emit_smem_allocs(fn, ctx)
        # Freeze the "top of function body" line index now that every
        # kernel-declared smem alloc has landed. Subsequent
        # ``ctx.alloc_smem`` calls from op visitors splice their
        # threadgroup decls at this cursor (Metal requires every
        # threadgroup decl at kernel-entry scope — can't sit inside
        # a for-loop body or conditional).
        ctx._smem_decl_insert_idx = len(ctx.lines)
        # Hoist NAX preamble (Coord setup + descriptor + op handle) to
        # function-entry scope when the kernel will use NAX. Emitting
        # lazily at the first MmaOp/LoadMatrixOp would trap ``_nax_fm``
        # / ``_nax_fn`` inside a for-loop body and they'd be out of
        # scope at any post-loop StoreMatrixOp.
        if ctx.uses_nax:
            from .mma import _emit_nax_preamble

            _emit_nax_preamble(ctx)
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
            input_dtypes=ctx.input_dtypes,
            output_dtypes=ctx.output_dtypes,
            scalar_dtypes=ctx.scalar_dtypes,
            atomic_output_names=frozenset(ctx.atomic_output_names),
        )

    def _classify_params(self, fn: Function, ctx: _MslCtx) -> None:
        for p in fn.params:
            if isinstance(p.type, BufferType):
                dt = msl_type(p.type.dtype)
                if p.attrs.readonly:
                    ctx.input_names.append(p.name)
                    ctx.input_dtypes.append(dt)
                else:
                    ctx.output_names.append(p.name)
                    ctx.output_dtypes.append(dt)
            elif isinstance(p.type, ScalarType):
                dt = msl_type(p.type.dtype)
                ctx.scalar_names.append(p.name)
                ctx.scalar_dtypes.append(dt)
                ctx.input_names.append(p.name)
                ctx.input_dtypes.append(dt)

    def _prescan_atomic_targets(self, fn: Function, ctx: _MslCtx) -> None:
        """Walk the op graph once and record every buffer name targeted
        by an ``AtomicRmwOp``.

        Has to run before the main visitor walk: kernels with init →
        atomic → finalize phases (moe_router_correct: ``counts`` is
        ``qk.store``-zeroed in phase 0 and ``atomic_rmw``-added in phase
        1) emit the plain store *first*, so populating the set lazily
        from inside ``_visit_atomic_rmw`` would miss the earlier
        ``_visit_store`` and emit a non-atomic ``buf[i] = v`` against
        an ``atomic<T>*`` parameter.
        """

        def walk(ops: list[Op]) -> None:
            for op in ops:
                if isinstance(op, AtomicRmwOp):
                    tensor = op.attrs["tensor"]
                    ctx.atomic_output_names.add(_tensor_buf_name(tensor, ctx))
                for region in op.regions:
                    walk(region.ops)

        walk(fn.body.ops)

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
        """Pre-emit ConstOps at function scope ONLY when they're referenced
        outside their declaring region.

        The IR builder's CSE keeps consts within their declaring region's
        frame, but Python-level Value handles can carry a const out of
        the loop body to be used in an epilogue or sibling region. On
        MSL the for-loop's C braces hide locals, so we hoist the
        cross-region ones to function scope.

        Region-local consts (only used within their declaring region or
        nested children) STAY LOCAL — Apple's compiler then sees a
        narrower live range and can reuse the register slot once the
        loop body exits. For a kernel with N loop-local zero-init consts
        × M loop iterations, this saves N register slots persisting at
        function scope. Material on register-pressure-bound kernels;
        free on others.
        """
        from quark.ir import ConstOp as _ConstOp

        # First pass: find every ConstOp + its declaring region, plus
        # every reference (operand) site keyed by the operand's id.
        const_ops: dict[int, tuple[_ConstOp, int]] = {}  # value.id → (op, region_id)
        ref_regions: dict[int, set[int]] = {}  # value.id → {region_ids}

        def walk(region_ops: list[Op], region_id: int) -> None:
            for op in region_ops:
                if isinstance(op, _ConstOp):
                    for r in op.results:
                        const_ops[r.id] = (op, region_id)
                for operand in op.operands:
                    ref_regions.setdefault(operand.id, set()).add(region_id)
                for region in op.regions:
                    walk(region.ops, id(region))

        # Treat the top-level (function body) as region_id = 0.
        walk(ops, 0)

        # Hoist ConstOps whose only uses are in their declaring region.
        # Cross-region uses (or no uses at all — could be unused, but
        # hoist anyway for safety) get hoisted to function scope.
        from .visitors import _visit_const

        for vid, (op, decl_region) in const_ops.items():
            uses = ref_regions.get(vid, set())
            cross_region = any(rid != decl_region for rid in uses)
            if cross_region or not uses:
                _visit_const(self, op, ctx)
        # Region-local consts will be emitted naturally when the
        # walker visits their region — _visit_const checks if the
        # binding already exists and is a no-op for hoisted ones.

    def _build_header(self, ctx: _MslCtx) -> str:
        # ``metal_stdlib`` declares the simd_* lane ops, fast-math
        # builtins, and the typedef pulling ``namespace metal::`` symbols
        # into scope. Without ``using namespace metal;`` the generated
        # body would have to qualify every ``simd_sum``/``half2`` etc.
        # with ``metal::``; cheaper to just import the namespace once.
        parts: list[str] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
        ]
        if ctx.uses_simdgroup_matrix:
            parts.append("#include <metal_simdgroup>")
            parts.append("#include <metal_simdgroup_matrix>")
        if ctx.uses_nax:
            # MPP matmul2d (NAX hardware accelerator). Requires Metal 4
            # language version at compile time — the driver picks the
            # right ``MTLLanguageVersion`` based on
            # ``DeviceCaps.supports_metal4``.
            parts.append("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>")
        # NAX MMA helper functions emitted at the kernel-source level
        # so multiple MmaOp call sites share one definition. Inlined
        # by the Apple compiler at each call but the explicit helper
        # scope gives cleaner cooperative_tensor live-range tracking.
        if ctx.nax_mma_helper_defs:
            parts.append("")
            parts.extend(ctx.nax_mma_helper_defs)
        return "\n".join(parts) + "\n"

    def _walk_region(self, ops: list[Op], ctx: _MslCtx) -> None:
        for op in ops:
            self._visit(op, ctx)

    def _visit(self, op: Op, ctx: _MslCtx) -> None:
        from .visitors import DISPATCH

        handler = DISPATCH.get(type(op))
        if handler is None:
            raise NotImplementedError(f"MslLowerer: no handler for {type(op).__name__}")
        handler(self, op, ctx)
