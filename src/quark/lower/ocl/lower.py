"""OpenCL/IGC SPIR-V lowerer — Intel-flavor kernel emit.

Emits OpenCL-dialect SPIR-V that Intel's IGC compiler accepts:

  OpCapability Kernel
  OpMemoryModel Physical64 OpenCL
  OpEntryPoint Kernel
  OpExtInstImport "OpenCL.std"
  Buffers: OpFunctionParameter (CrossWorkgroup pointer) +
    OpInBoundsPtrAccessChain for indexing
  Builtin vec3 inputs: OpTypeVector u64 3 (NDRange ids are 64-bit
    in OCL)
  Cooperative-matrix MMA via ``OpSubgroupMatrixMultiplyAccumulate
    INTEL`` (opcode 6237) + ``SPV_INTEL_subgroup_matrix_multiply_
    accumulate`` extension.
  ``SmemAllocOp`` / ``BarrierOp`` via Workgroup-class storage +
    ``OpControlBarrier`` with OpenCL memory semantics.

One ``_visit_<op>`` per IR op; visitors mutate the ``_OclCtx`` to
build the SPIR-V text via the dialect-agnostic ``SpvText``
assembler in ``quark.lower._common.spirv_text``.

Multi-function modules: only the entry function is honoured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quark.ir import DType
from quark.ir.types import _DTYPE_BYTES
from quark.ir.module import Function, Module
from quark.ir.op import (  # noqa: F401  (some used only for isinstance checks)
    MergeB32Op,
    SplitB32Op,
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
    MmaOp,
    SelectOp,
    ShuffleOp,
    SmemAllocOp,
    StoreMatrixOp,
    StoreOp,
    SubgroupIdOp,
    SubgroupReduceOp,
    ThreadIdInGroupOp,
    ThreadIdxOp,
    VecBuildOp,
    VecExtractOp,
    VecLoadOp,
    VecStoreOp,
    YieldOp,
)
from quark.ir.tensor import GlobalTensor, SharedRegion

from quark.lower._common.spirv_text import SpvText


@dataclass
class LoweredOclSpirVKernel:
    """OCL/IGC peer of ``LoweredSpirVKernel``.

    ``source`` is SPIR-V text in the OpenCL dialect (``OpMemoryModel
    Physical64 OpenCL``, ``OpCapability Kernel``, etc.). ``spirv-as
    --target-env opencl2.0`` consumes this; the OCL driver then hands
    the binary to IGC via ``clCreateProgramWithIL``.
    """

    source: str
    entry_name: str = "main"
    n_buffers: int = 0
    smem_bytes: int = 0
    local_size: tuple[int, int, int] = (1, 1, 1)
    subgroup_size: int = 32

    @property
    def kernel_name(self) -> str:
        return self.entry_name


@dataclass
class _OclCtx:
    """Per-function lowering state.

    Mirrors ``_SpvCtx`` but the per-tensor side tables hold function-
    parameter SSA ids (not buffer ``OpVariable``s) — buffers in OCL
    SPIR-V are kernel pointer arguments, not module-scope storage
    variables.
    """

    text: SpvText = field(default_factory=SpvText)
    # Value.id → SPIR-V SSA id (the result of OpLoad / OpFAdd / ...).
    val_to_id: dict[int, str] = field(default_factory=dict)
    # IR ``Param`` id() → SSA id of the OpFunctionParameter for that buffer.
    param_to_arg: dict[int, str] = field(default_factory=dict)
    # IR ``Param`` id() → element OpTypePointer CrossWorkgroup id.
    param_to_elem_ptr: dict[int, str] = field(default_factory=dict)
    # IR ``Param`` id() → element type id (OpTypeFloat 32, etc.).
    param_to_elem_type: dict[int, str] = field(default_factory=dict)
    # Builtin vec3 inputs. OCL kernels use u64 (vs Vulkan's u32) per
    # the SPIR-V OpenCL environment spec — GlobalInvocationId &c are
    # ``OpTypeVector OpTypeInt 64 0 3``.
    local_inv_id_var: str = ""
    workgroup_id_var: str = ""
    # LocalSize constants (driver supplies the matching values at
    # clEnqueueNDRangeKernel time).
    local_size: tuple[int, int, int] = (1, 1, 1)
    # Capability tracking for dtype-gated decls.
    has_f16_cap: bool = False
    has_bf16_cap: bool = False
    has_int16_cap: bool = False
    has_int8_cap: bool = False
    has_int64_cap: bool = False
    # Workgroup-class smem allocations: SmemAllocOp result Value.id →
    # (var_id, elem_type, elem_pointer_id, total_elements). Visitors
    # that load/store on a ``SharedRegion`` look up by the SharedRegion's
    # backing ``alloc.id`` (the Value.id of the SmemAllocOp result).
    smem_allocs: dict[int, tuple[str, str, str, int]] = field(default_factory=dict)
    # Total ``Workgroup``-class bytes emitted for this kernel.
    # Surfaced as ``LoweredOclSpirVKernel.smem_bytes`` so the driver
    # can validate against ``CL_DEVICE_LOCAL_MEM_SIZE`` before
    # ``clCreateKernel``. Increments in ``_visit_smem_alloc``; the
    # row-pad attr is folded in for 2D regions (same convention as
    # the SPV side — see project_spv_smem_offset_bug.md).
    smem_total_bytes: int = 0
    # Cached lane-id SSA (``SubgroupLocalInvocationId``). Re-emitted on
    # every per-lane fragment load/store rather than cached on the ctx
    # for the same dominance reasons as the other builtins (see
    # ``_ensure_builtin_vec3_u32_component``'s docstring). The
    # variable declaration is cached at module scope.
    lane_id_var: str = ""
    # First-MmaOp gating: emit ``OpCapability SubgroupMatrixMultiply
    # AccumulateINTEL`` + ``OpExtension "SPV_INTEL_subgroup_matrix_
    # multiply_accumulate"`` exactly once per module.
    has_intel_mma_caps: bool = False
    # Cached SubgroupId builtin variable id (lazy decl).
    subgroup_id_var: str = ""
    # Stack of pending region-yield slots — one entry per surrounding
    # region (if-with-carries, for-loop-with-carries). Each entry is a
    # list of (target_id, type_id) pairs; a ``YieldOp`` inside that
    # region materialises each operand into the corresponding target
    # via ``OpCopyObject``. Same shape as the SPV lowerer's
    # ``loop_yield_stack``.
    loop_yield_stack: list[list[tuple[str, str]]] = field(default_factory=list)
    # SIMD width the kernel is compiled for. Battlemage / PTL default
    # to 32; the launcher pairs this with ``cl_intel_required_subgroup_
    # size`` at compile time (Phase 3 follow-up — kernel-level pinning
    # not yet wired).
    subgroup_width: int = 32


# ── Dtype emission ──────────────────────────────────────────────────


def _emit_dtype(text: SpvText, dt: DType, ctx: "_OclCtx | None" = None) -> str:
    """Map a quark ``DType`` to its SPIR-V type id.

    Same caps gating as the Vulkan lowerer — half-precision dtypes
    need their corresponding capability declared once at module scope.
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
            # IGC accepts the Khronos BFloat16TypeKHR capability +
            # SPV_KHR_bfloat16 extension as the canonical encoding.
            text.add_capability("BFloat16TypeKHR")
            text.add_extension("SPV_KHR_bfloat16")
            ctx.has_bf16_cap = True
        return text.type_float(16, bfloat16=True)
    if dt is DType.U16:
        if ctx is not None and not ctx.has_int16_cap:
            text.add_capability("Int16")
            ctx.has_int16_cap = True
        return text.type_int(16, signed=False)
    if dt is DType.S16:
        if ctx is not None and not ctx.has_int16_cap:
            text.add_capability("Int16")
            ctx.has_int16_cap = True
        return text.type_int(16, signed=True)
    if dt is DType.U8:
        if ctx is not None and not ctx.has_int8_cap:
            text.add_capability("Int8")
            ctx.has_int8_cap = True
        return text.type_int(8, signed=False)
    if dt is DType.S8:
        if ctx is not None and not ctx.has_int8_cap:
            text.add_capability("Int8")
            ctx.has_int8_cap = True
        return text.type_int(8, signed=True)
    if dt is DType.PRED:
        return text.type_bool()
    # Bit-typed variants: B16/B32/B64 are raw storage with no
    # numeric interpretation. Lower them to OpTypeInt<width> 0
    # (unsigned int carrier) — matches how the SPV side handled them.
    if dt is DType.B32:
        return text.type_int(32, signed=False)
    if dt is DType.B16:
        if ctx is not None and not ctx.has_int16_cap:
            text.add_capability("Int16")
            ctx.has_int16_cap = True
        return text.type_int(16, signed=False)
    if dt is DType.B64:
        if ctx is not None and not ctx.has_int64_cap:
            text.add_capability("Int64")
            ctx.has_int64_cap = True
        return text.type_int(64, signed=False)
    raise NotImplementedError(
        f"OclSpirVLowerer: dtype {dt!r} not yet supported in Phase 3 "
        "first cut. Extend ``_emit_dtype`` per the visitor coverage plan."
    )


# ── Arith opcode table (OpenCL-flavor — identical to Vulkan for the
# pure-arith subset). Ext-inst names use the OpenCL.std set instead
# of GLSL.std.450 (handled separately in MathOp once that visitor
# lands). ────────────────────────────────────────────────────────────
_ARITH_KIND_TO_OP: dict[tuple[str, DType], str] = {
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
    # Bitwise + shifts. ``shr`` on unsigned is logical (zero-fill);
    # on signed is arithmetic (sign-fill) per the standard semantics
    # the IR ArithOp inherits.
    ("shl", DType.U32): "OpShiftLeftLogical",
    ("shl", DType.S32): "OpShiftLeftLogical",
    ("shr", DType.U32): "OpShiftRightLogical",
    ("shr", DType.S32): "OpShiftRightArithmetic",
    ("and", DType.U32): "OpBitwiseAnd",
    ("and", DType.S32): "OpBitwiseAnd",
    ("or",  DType.U32): "OpBitwiseOr",
    ("or",  DType.S32): "OpBitwiseOr",
    ("xor", DType.U32): "OpBitwiseXor",
    ("xor", DType.S32): "OpBitwiseXor",
    # Boolean (PRED) ops — separate opcodes for logical AND/OR/XOR.
    ("and", DType.PRED): "OpLogicalAnd",
    ("or",  DType.PRED): "OpLogicalOr",
    ("xor", DType.PRED): "OpLogicalNotEqual",  # XOR == ≠ for booleans
}


# (kind, dtype) → SPIR-V comparison opcode. Identical to the Vulkan
# table — comparison opcodes are environment-agnostic; the only
# dialect-sensitive thing about ``CmpOp`` is the result type (``OpType
# Bool`` here just like Vulkan).
_CMP_KIND_TO_OP: dict[tuple[str, DType], str] = {
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


# ``MathOp.kind`` → OpenCL.std symbolic instruction name. The
# spirv-as assembler accepts the symbolic name (not the numeric
# instruction code) for OpExtInst on OpenCL.std. Names are
# lower-case (mirroring the OpenCL C library) — the GLSL.std.450
# set uses CamelCase, so this isn't a search-and-replace; the entire
# table differs in case.
# Reference: https://registry.khronos.org/SPIR-V/specs/unified1/OpenCL.ExtendedInstructionSet.100.html
_MATH_KIND_TO_OPENCL_INSTR: dict[str, str | None] = {
    "rcp": None,          # no direct OpenCL.std entry — emit OpFDiv 1.0 / x
    "rcp_approx": None,
    "rsqrt": "rsqrt",
    "rsqrt_approx": "rsqrt",
    "sqrt": "sqrt",
    "sqrt_approx": "sqrt",
    "exp": "exp",
    "exp_approx": "exp",
    "exp2": "exp2",
    "ex2_approx": "exp2",
    "log2": "log2",
    "log2_approx": "log2",
    "sin": "sin",
    "cos": "cos",
    "tanh": "tanh",
}


_DIM_TO_INDEX = {"x": 0, "y": 1, "z": 2}


# Intel SPV_INTEL_subgroup_matrix_multiply_accumulate per-lane layout
# descriptor. Keyed on the MmaShape tuple + subgroup_width;
# returns the SPIR-V per-lane vector types + K-Dim const value +
# MatrixOperands flag the corresponding ``OpSubgroupMatrixMultiply
# AccumulateINTEL`` invocation needs.
#
# These values were discovered empirically by probing IGC on the
# PTL devkit (2026-05-12) — see ``test_intel_mma_spv_compiles_through_
# igc`` in ``tests/drivers/test_ocl_compile_launch.py`` for the
# locked-in template. The per-lane *register footprint* (e.g.
# ``v8u32`` for B at SG=32 bf16) differs from the per-lane *element
# count* (8 bf16 per lane × 32 lanes = 256 = K*N for m8n16k16). IGC
# rejected the "natural" v4u32 with "Matrix B argument must have 8
# components for targeted HW. Actual: 4".
#
# Each entry's lane_types are emitted via the spec strings below
# (e.g. ``v4i16``) — the OCL emitter materializes them via SpvText's
# type_vec helper at first use.
#
# Layout keys: (a_dtype, b_dtype, acc_dtype, m, n, k).
# Layout values: (subgroup_size, a_lane_elem_dt, a_lane_width,
# b_lane_elem_dt, b_lane_width, c_lane_elem_dt, c_lane_width,
# k_dim_const, matrix_operands_flag_str).
#
# Subgroup width is a PROPERTY of the layout, not a key. The Intel
# MMA extension only supports SG ∈ {8, 16}, and each shape has a
# canonical SG (m8n16k16 bf16 → SG=16). The lowerer must use the
# subgroup width the layout demands, regardless of the device's
# default SG (32 on Battlemage).
_INTEL_MMA_LAYOUTS: dict[tuple, tuple] = {
    # bf16×bf16→f32, M=8 N=16 K=16 on SG=16. This is the canonical
    # form per the Khronos ``cl_intel_subgroup_matrix_multiply_
    # accumulate`` extension spec: the only supported subgroup sizes
    # for the builtin (and thus the SPV op) are 8 and 16. Battlemage's
    # native DPAS is 16-wide bf16 (1024-bit matrix engine), so SG=16
    # matches the hardware's native lane count. oneDNN's GEMM-micro
    # JIT (and OpenVINO via oneDNN) also pin SG=16 for the same
    # reason.
    #
    # Lane convention (from Khronos spec + oneDNN micro JIT):
    #   A (short8 per lane): lane L holds **column k=L** of A.
    #     Components s0..s7 = rows m=0..7 of A[:, k=L].
    #   B (int8 per lane): lane L holds **column n=L** of B. Each int
    #     packs 2 bf16 along K: s0 low = B[k=0, n=L], s0 high =
    #     B[k=1, n=L]; …; s7 high = B[k=15, n=L].
    #   C/D (float8 per lane): lane L holds **column n=L** of C/D.
    #     Components s0..s7 = rows m=0..7 of C[:, n=L].
    #
    # MatrixOperands flag = MatrixAPackedBFloat16INTEL |
    # MatrixBPackedBFloat16INTEL (0x1000 | 0x2000) tells IGC that the
    # integer-class registers carry bf16 components.
    #
    # An earlier SG=32 layout entry produced only 2/4 valid f32 slots
    # per lane (Battlemage's DPAS is 16-wide; at SG=32 IGC issued one
    # DPAS that filled lanes 0–15 only). The Khronos extension spec
    # explicitly rules out SG=32 — "the only supported subgroup sizes
    # are 8 and 16."  The SG=32 entry was removed to avoid silent
    # mis-issue.
    (DType.BF16, DType.BF16, DType.F32, 8, 16, 16): (
        16,             # subgroup_size required by this MMA form
        DType.U16, 8,   # A: v8 of i16 (bf16-pattern; column k=L)
        DType.U32, 8,   # B: v8 of u32 (bf16-pairs; column n=L)
        DType.F32, 8,   # C: v8 of f32 (column n=L, rows m=0..7)
        16,             # K-Dim operand
        "MatrixAPackedBFloat16INTEL|MatrixBPackedBFloat16INTEL",
    ),
    # s8×s8→s32, M=8 N=16 K=32 on SG=16. Per the Khronos
    # ``SPV_INTEL_subgroup_matrix_multiply_accumulate`` extension's
    # 8-bit form: 4 s8 components pack into one i32 along the K
    # direction. Per-lane element counts:
    #
    #   A: M=8 × K=32 of s8 = 256 bytes total. At SG=16, each lane
    #      holds K/SG = 2 K-cols × M=8 rows = 16 s8 = 4 i32. Lane L
    #      covers K-cols (L, L+16); within each i32, components pack
    #      4 K-rows for one M.
    #   B: K=32 × N=16 of s8 = 512 bytes total. At SG=16, each lane
    #      holds full K for one N-col = 32 s8 = 8 i32. Lane L holds
    #      column N=L; each i32 packs 4 K-rows.
    #   C/D: M=8 × N=16 of s32 = 128 elements total. At SG=16, each
    #      lane holds full M for one N-col = 8 s32. Same shape as
    #      the bf16/f32 form's C (just signed-int element type).
    #
    # MatrixOperands flag: 0x10 | 0x20 | 0x100 | 0x200 = 0x330.
    #   MatrixASignedComponentsINTEL (0x10) — A values are signed s8
    #   MatrixBSignedComponentsINTEL (0x20) — B values are signed s8
    #   MatrixAPackedInt8INTEL       (0x100) — A is i32-packed 4-quartets
    #   MatrixBPackedInt8INTEL       (0x200) — B is i32-packed 4-quartets
    #
    # Note: the "Packed" operands DON'T carry the ``Components``
    # suffix — only the signedness operands do. spirv-as rejects the
    # "PackedInt8ComponentsINTEL" form.
    #
    # Mirrors the bf16 entry's ``MatrixA/BPackedBFloat16INTEL`` shape
    # — the operands flag tells IGC the integer-carrier register lanes
    # hold 8-bit packed components, not raw 32-bit ints.
    (DType.S8, DType.S8, DType.S32, 8, 16, 32): (
        16,             # SG=16 per spec
        # Empirical IGC requirements for the s8 form (K=32, M=8,
        # Result=S32):
        #   - A per-lane width must match M (=8); ``size 8 to match
        #     M defined by Result type`` else IGC rejects.
        #   - A element type must be int16 (not int32) per the
        #     ``expected A element type to be int16_t for K Dim = 32``
        #     check. Packing: 2 s8 along K per u16; per lane = 16 s8
        #     = 8 u16; 16 lanes × 16 s8 = M*K = 256 ✓.
        #   - B element type stays int32 (each i32 packs 4 s8 along
        #     K, per the bf16-form analogy).
        DType.U16, 8,   # A: v8 of u16 (2 packed s8 along K per u16)
        DType.U32, 8,   # B: v8 of u32 (4 packed s8 along K per u32)
        DType.S32, 8,   # C: v8 of s32 (column n=L, rows m=0..7)
        32,             # K-Dim operand
        "MatrixASignedComponentsINTEL|MatrixBSignedComponentsINTEL"
        "|MatrixAPackedInt8INTEL|MatrixBPackedInt8INTEL",
    ),
}


def _intel_mma_layout(shape) -> tuple:
    """Resolve an ``MmaShape`` to its Intel MMA layout descriptor.

    The subgroup width is part of the layout (each shape has a
    canonical SG per the Intel MMA extension spec — SG=16 for the
    bf16 m8n16k16 form on Battlemage). Callers that need the SG
    width pull it from ``layout[0]``.
    """
    key = (
        shape.a_dtype, shape.b_dtype, shape.acc_dtype,
        shape.m, shape.n, shape.k,
    )
    layout = _INTEL_MMA_LAYOUTS.get(key)
    if layout is None:
        raise NotImplementedError(
            f"_intel_mma_layout: shape "
            f"a={shape.a_dtype} b={shape.b_dtype} acc={shape.acc_dtype} "
            f"m={shape.m} n={shape.n} k={shape.k} "
            f"not yet in ``_INTEL_MMA_LAYOUTS``. Read the Khronos "
            f"``cl_intel_subgroup_matrix_multiply_accumulate`` spec "
            f"+ oneDNN GEMM-micro JIT for the per-lane register "
            f"footprint."
        )
    return layout


def _ocl_type_vec(text: SpvText, elem_id: str, width: int) -> str:
    """Emit ``OpTypeVector`` with ``width`` components.

    Override of ``SpvText.type_vec``: under the OpenCL environment,
    OpTypeVector accepts component counts 2..16 (vs Vulkan compute's
    2/3/4 ceiling). ``SpvText`` falls back to ``OpTypeArray`` for
    width > 4 to satisfy Vulkan; IGC rejects the array form for
    Intel MMA operands (Matrix B with v8u32 is what the spec wants).
    """
    key = f"ocl_vec_{elem_id}_{width}"
    if key in text.type_cache:
        return text.type_cache[key]
    tid = text.alloc_id(key)
    text.type_cache[key] = tid
    text.add_type_line(f"{tid} = OpTypeVector {elem_id} {width}")
    return tid


def _ensure_intel_mma_caps(ctx: _OclCtx) -> None:
    if ctx.has_intel_mma_caps:
        return
    ctx.text.add_capability("SubgroupMatrixMultiplyAccumulateINTEL")
    ctx.text.add_extension("SPV_INTEL_subgroup_matrix_multiply_accumulate")
    ctx.has_intel_mma_caps = True


def _ensure_lane_id(ctx: _OclCtx) -> str:
    """Lazily declare ``SubgroupLocalInvocationId`` (u32) and emit
    a fresh OpLoad at every call site. Same dominance rule as the
    builtin vec3 helper — caching the loaded SSA breaks when used
    from multiple basic blocks."""
    var_id = ctx.lane_id_var
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
    res_id = ctx.text.alloc_id("lane_id")
    ctx.text.emit_function(f"{res_id} = OpLoad {u32} {var_id}")
    return res_id


# ── Visitors ────────────────────────────────────────────────────────


def _visit_const(op: ConstOp, ctx: _OclCtx) -> None:
    (out,) = op.results
    dt = out.dtype
    raw = op.attrs.get("value")
    if dt is DType.U32 or dt is DType.B32:
        cid = ctx.text.const_uint(int(raw) & 0xFFFFFFFF)
    elif dt is DType.S32:
        s32 = ctx.text.type_int(32, signed=True)
        cid = ctx.text.alloc_id(f"s_{int(raw)}")
        ctx.text.add_type_line(f"{cid} = OpConstant {s32} {int(raw)}")
    elif dt is DType.F32:
        cid = ctx.text.const_float(float(raw))
    elif dt is DType.B16:
        u16 = ctx.text.type_int(16, signed=False)
        cid = ctx.text.alloc_id(f"b16_{int(raw) & 0xFFFF}")
        ctx.text.add_type_line(f"{cid} = OpConstant {u16} {int(raw) & 0xFFFF}")
    elif dt is DType.B64:
        u64 = ctx.text.type_int(64, signed=False)
        cid = ctx.text.alloc_id(f"b64_{int(raw) & 0xFFFFFFFFFFFFFFFF}")
        ctx.text.add_type_line(
            f"{cid} = OpConstant {u64} {int(raw) & 0xFFFFFFFFFFFFFFFF}"
        )
    else:
        raise NotImplementedError(
            f"_visit_const(ocl): dtype {dt!r} not yet wired"
        )
    ctx.val_to_id[out.id] = cid


def _visit_cmp(op: CmpOp, ctx: _OclCtx) -> None:
    """``cmp(kind, a, b)`` → SPIR-V comparison op producing ``OpTypeBool``."""
    (out,) = op.results
    kind = op.attrs["kind"]
    a, b = op.operands
    spv_op = _CMP_KIND_TO_OP.get((kind, a.dtype))
    if spv_op is None:
        raise NotImplementedError(
            f"_visit_cmp(ocl): kind={kind!r} dtype={a.dtype!r} not yet wired"
        )
    bool_t = ctx.text.type_bool()
    res_id = ctx.text.alloc_id(f"cmp_{kind}")
    ctx.val_to_id[out.id] = res_id
    a_id = ctx.val_to_id[a.id]
    b_id = ctx.val_to_id[b.id]
    ctx.text.emit_function(f"{res_id} = {spv_op} {bool_t} {a_id} {b_id}")


def _visit_if_region(op: IfRegionOp, ctx: _OclCtx) -> None:
    """Structured if/else (with or without carries).

    SPIR-V structured control flow is dialect-agnostic — the emit
    shape is byte-for-byte the same as the Vulkan lowerer. Pattern:

        OpSelectionMerge %merge None
        OpBranchConditional %pred %then %else
        %then = OpLabel
          ;; body
          OpBranch %then_tail
        %then_tail = OpLabel
          OpBranch %merge
        %else = OpLabel
          ;; body
          OpBranch %else_tail
        %else_tail = OpLabel
          OpBranch %merge
        %merge = OpLabel
          ;; (with carries) OpPhi from %then_tail / %else_tail

    The dedicated ``*_tail`` blocks give OpPhi a stable predecessor
    label even when an arm body nests its own structured control
    flow. For the bounds-check pattern (no carries) the tail blocks
    cost ~3 lines that the GPU compiler folds during optimize.
    """
    pred_id = ctx.val_to_id[op.pred.id]
    n_carried = int(op.attrs.get("n_carried", 0))

    then_in_vals = op.operands[1 : 1 + n_carried]
    else_in_vals = op.operands[1 + n_carried : 1 + 2 * n_carried]

    # Carried-in body vars alias their corresponding operand's SSA id.
    for body_var, src_val in zip(op.then_body_vars, then_in_vals, strict=False):
        ctx.val_to_id[body_var.id] = ctx.val_to_id[src_val.id]
    for body_var, src_val in zip(op.else_body_vars, else_in_vals, strict=False):
        ctx.val_to_id[body_var.id] = ctx.val_to_id[src_val.id]

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

    ctx.text.emit_function(f"{merge_label} = OpLabel")
    for i, res in enumerate(op.results):
        phi_id = ctx.text.alloc_id(f"if_out{i}")
        ctx.val_to_id[res.id] = phi_id
        ctx.text.emit_function(
            f"{phi_id} = OpPhi {type_ids[i]} "
            f"{then_yield_ids[i]} {then_tail} "
            f"{else_yield_ids[i]} {else_tail}"
        )


def _visit_yield(op: YieldOp, ctx: _OclCtx) -> None:
    """``YieldOp`` inside an if-with-carries / for-loop body. Without
    carries (the if-without-carries bounds-check pattern) this is a
    pure no-op: the surrounding visitor's ``OpBranch`` consumes the
    region terminator. With carries, materialise each yielded value
    into its pre-allocated target id via ``OpCopyObject`` so the
    merge block's ``OpPhi`` (forward-declared, referencing those ids)
    resolves cleanly through ``spirv-as``."""
    if not op.operands:
        return
    if not ctx.loop_yield_stack:
        raise NotImplementedError(
            "_visit_yield(ocl): yielding values from a region without a "
            "surrounding for/while loop scope — not yet wired."
        )
    targets = ctx.loop_yield_stack[-1]
    if len(targets) != len(op.operands):
        raise RuntimeError(
            f"_visit_yield(ocl): yielded {len(op.operands)} values vs "
            f"{len(targets)} carries"
        )
    for (target_id, type_id), val in zip(targets, op.operands, strict=False):
        src_id = ctx.val_to_id[val.id]
        ctx.text.emit_function(f"{target_id} = OpCopyObject {type_id} {src_id}")


def _visit_arith(op: ArithOp, ctx: _OclCtx) -> None:
    (out,) = op.results
    kind = op.attrs.get("kind", "")
    operands = [ctx.val_to_id[v.id] for v in op.operands]
    type_id = _emit_dtype(ctx.text, out.dtype, ctx)

    # Unary ops dispatched separately (no entry in the binary table).
    if kind == "neg":
        spv_op = "OpFNegate" if _dtype_kind(out.dtype) == "float" else "OpSNegate"
        res_id = ctx.text.alloc_id("neg")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = {spv_op} {type_id} {operands[0]}")
        return
    if kind == "abs":
        cl_set = ctx.text.import_ext_inst("OpenCL.std")
        cl_name = "fabs" if _dtype_kind(out.dtype) == "float" else "s_abs"
        res_id = ctx.text.alloc_id("abs")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpExtInst {type_id} {cl_set} {cl_name} {operands[0]}"
        )
        return
    if kind == "fma":
        cl_set = ctx.text.import_ext_inst("OpenCL.std")
        res_id = ctx.text.alloc_id("fma")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpExtInst {type_id} {cl_set} fma "
            f"{operands[0]} {operands[1]} {operands[2]}"
        )
        return
    if kind in ("min", "max"):
        cl_set = ctx.text.import_ext_inst("OpenCL.std")
        if _dtype_kind(out.dtype) == "float":
            cl_name = "fmin" if kind == "min" else "fmax"
        elif _dtype_kind(out.dtype) == "sint":
            cl_name = "s_min" if kind == "min" else "s_max"
        else:
            cl_name = "u_min" if kind == "min" else "u_max"
        res_id = ctx.text.alloc_id(kind)
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(
            f"{res_id} = OpExtInst {type_id} {cl_set} {cl_name} {operands[0]} {operands[1]}"
        )
        return

    spv_op = _ARITH_KIND_TO_OP.get((kind, out.dtype))
    if spv_op is None:
        raise NotImplementedError(
            f"_visit_arith(ocl): kind={kind!r} dtype={out.dtype!r} not yet "
            "wired in Phase 3 first cut"
        )
    res_id = ctx.text.alloc_id(kind)
    ctx.val_to_id[out.id] = res_id
    args = " ".join(operands)
    ctx.text.emit_function(f"{res_id} = {spv_op} {type_id} {args}")


def _ensure_builtin_vec3_u32_component(
    ctx: _OclCtx,
    *,
    var_attr: str,
    builtin_name: str,
    dim: str,
) -> str:
    """Emit per-call ``OpLoad`` + ``OpCompositeExtract`` + ``OpUConvert``
    for an OCL builtin vec3 input. The OCL SPIR-V env defines these
    builtins as ``OpTypeVector OpTypeInt 64 0 3`` — wider than the
    Vulkan u32 form. The IR's index values are u32; convert the
    extracted u64 lane to u32 before returning.

    Lazily declares the OpVariable Input at module scope (cached on
    ``ctx.<var_attr>``). The load is *not* cached — caching the SSA
    id would break dominance if the helper fires from inside a region
    different from the one that first emitted it; the SPV lowerer
    learned this the hard way (see its docstring on
    ``_ensure_scalar_builtin``).
    """
    if dim not in _DIM_TO_INDEX:
        raise ValueError(f"_ensure_builtin: bad dim={dim!r}")

    var_id = getattr(ctx, var_attr, "")
    if not var_id:
        u64 = ctx.text.type_int(64, signed=False)
        v3u64 = ctx.text.type_vec(u64, 3)
        ptr = ctx.text.type_pointer("Input", v3u64)
        var_id = ctx.text.alloc_id(builtin_name)
        ctx.text.add_type_line(f"{var_id} = OpVariable {ptr} Input")
        ctx.text.add_decoration(f"OpDecorate {var_id} BuiltIn {builtin_name}")
        setattr(ctx, var_attr, var_id)

    u64 = ctx.text.type_int(64, signed=False)
    v3u64 = ctx.text.type_vec(u64, 3)
    loaded = ctx.text.alloc_id(f"{builtin_name}_{dim}_vec")
    ctx.text.emit_function(f"{loaded} = OpLoad {v3u64} {var_id}")
    lane64 = ctx.text.alloc_id(f"{builtin_name}_{dim}_u64")
    ctx.text.emit_function(
        f"{lane64} = OpCompositeExtract {u64} {loaded} {_DIM_TO_INDEX[dim]}"
    )
    # IR is u32 throughout — convert. IGC permits ``OpUConvert`` between
    # any pair of integer widths.
    u32 = ctx.text.type_int(32, signed=False)
    res_id = ctx.text.alloc_id(f"{builtin_name}_{dim}")
    ctx.text.emit_function(f"{res_id} = OpUConvert {u32} {lane64}")
    return res_id


def _visit_thread_idx(op: ThreadIdxOp, ctx: _OclCtx) -> None:
    """``thread_idx(dim)`` → ``LocalInvocationId.<dim>`` (within-WG index)."""
    (out,) = op.results
    dim = op.attrs.get("dim", "x")
    ctx.val_to_id[out.id] = _ensure_builtin_vec3_u32_component(
        ctx, var_attr="local_inv_id_var",
        builtin_name="LocalInvocationId", dim=dim,
    )


def _visit_block_idx(op: BlockIdxOp, ctx: _OclCtx) -> None:
    """``block_idx(dim)`` → ``WorkgroupId.<dim>``."""
    (out,) = op.results
    dim = op.attrs.get("dim", "x")
    ctx.val_to_id[out.id] = _ensure_builtin_vec3_u32_component(
        ctx, var_attr="workgroup_id_var",
        builtin_name="WorkgroupId", dim=dim,
    )


def _visit_block_dim(op: BlockDimOp, ctx: _OclCtx) -> None:
    """``block_dim(dim)`` → compile-time constant from ``ctx.local_size``.

    The driver guarantees the matching ``local_size`` at
    ``clEnqueueNDRangeKernel`` time. Matches the SPV lowerer's
    convention; OCL's ``get_local_size`` would also work but the
    constant form gives IGC strictly more room to optimize.
    """
    (out,) = op.results
    dim = op.attrs.get("dim", "x")
    axis = _DIM_TO_INDEX.get(dim)
    if axis is None:
        raise ValueError(f"_visit_block_dim(ocl): bad dim={dim!r}")
    cid = ctx.text.const_uint(int(ctx.local_size[axis]))
    ctx.val_to_id[out.id] = cid


def _visit_math(op: MathOp, ctx: _OclCtx) -> None:
    """Transcendental / approximate-math ops via OpenCL.std ext-inst.

    ``rcp`` has no direct OpenCL.std entry — emit ``OpFDiv 1.0 / x``
    (IGC pattern-matches this into the hardware reciprocal, same as
    the GLSL.std.450 fallback path on the Vulkan side).
    """
    (out,) = op.results
    kind = op.attrs["kind"]
    if kind not in _MATH_KIND_TO_OPENCL_INSTR:
        raise NotImplementedError(
            f"_visit_math(ocl): kind={kind!r} not yet wired"
        )
    cl_name = _MATH_KIND_TO_OPENCL_INSTR[kind]
    src_id = ctx.val_to_id[op.operands[0].id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)

    if cl_name is None:  # rcp — emit OpFDiv 1.0 / x
        if out.dtype is not DType.F32:
            raise NotImplementedError(
                f"_visit_math(ocl,rcp): only f32 wired today, got {out.dtype!r}"
            )
        one = ctx.text.const_float(1.0)
        res_id = ctx.text.alloc_id("rcp")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = OpFDiv {dst_t} {one} {src_id}")
        return

    cl_set_id = ctx.text.import_ext_inst("OpenCL.std")
    res_id = ctx.text.alloc_id(kind)
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpExtInst {dst_t} {cl_set_id} {cl_name} {src_id}"
    )


def _visit_lane_id(op: LaneIdOp, ctx: _OclCtx) -> None:
    """``lane_id()`` → ``SubgroupLocalInvocationId``. Adds
    ``GroupNonUniform`` capability on first use — required for IGC to
    accept the builtin.
    """
    (out,) = op.results
    ctx.text.add_capability("GroupNonUniform")
    ctx.val_to_id[out.id] = _ensure_lane_id(ctx)


def _visit_load_matrix(op: LoadMatrixOp, ctx: _OclCtx) -> None:
    """``LoadMatrixOp`` → per-lane gather of the Intel fragment from
    row-major smem.

    Intel's lane convention for SG=16 m8n16k16 (Khronos
    ``cl_intel_subgroup_matrix_multiply_accumulate`` spec):

      A (M×K bf16, short8 per lane): lane L holds column k=L of A;
        slots s0..s7 = rows m=0..7 of A[:, k=L].
        Per-slot gather: smem[(row+s)*row_stride + (col+L)].
      B (K×N bf16, int8 per lane): lane L holds column n=L of B;
        each i32 slot s packs two bf16 along K — low half =
        B[k=2s, n=L], high half = B[k=2s+1, n=L].
        Per-slot gather: pack u16 pair (smem[(row+2s)*row_stride +
        (col+L)], smem[(row+2s+1)*row_stride + (col+L)]) into i32.
      C/D (M×N f32 or bf16, float8/short8 per lane): same as A but
        the tensor's stride is N (not K). Lane L = column n=L,
        slots = rows m=0..7.

    The smem region is **row-major** (the layout the kernel cohort's
    smem layout pass produces). Each slot's flat offset is computed
    from the tile origin (op.operands[0]=row, op.operands[1]=col),
    the slot's row index, and the lane id.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    (out,) = op.results
    tensor = op.attrs["src_tensor"]
    if not isinstance(tensor, SharedRegion):
        raise NotImplementedError(
            "_visit_load_matrix(ocl): only SharedRegion sources wired"
        )
    shape_id = op.attrs["shape_id"]
    which = op.attrs["which"]

    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_load_matrix(ocl): unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    layout = _intel_mma_layout(shape)
    (
        _sg, a_elem_dt, a_width, b_elem_dt, b_width, c_elem_dt, c_width,
        _k_dim, _operands_flag,
    ) = layout

    if which == "a":
        elem_dt, width = a_elem_dt, a_width
    elif which == "b":
        elem_dt, width = b_elem_dt, b_width
    elif which == "c":
        elem_dt, width = c_elem_dt, c_width
    else:
        raise NotImplementedError(
            f"_visit_load_matrix(ocl): which={which!r}"
        )

    rec = ctx.smem_allocs.get(tensor.alloc.id)
    if rec is None:
        raise RuntimeError(
            f"_visit_load_matrix(ocl): SharedRegion {tensor.name!r} accessed "
            "before its SmemAllocOp was visited"
        )
    smem_var_id, smem_elem_type, _smem_elem_ptr, _n = rec

    lane_elem_type = _emit_dtype(ctx.text, elem_dt, ctx)
    lane_vec_type = _ocl_type_vec(ctx.text, lane_elem_type, width)

    u32 = ctx.text.type_int(32, signed=False)
    lane_id = _ensure_lane_id(ctx)

    # Tile origin (row, col) — from op.operands. For our simple
    # single-tile fixtures these are const 0, but for GemmKernel
    # they're runtime values.
    row_base = ctx.val_to_id[op.operands[0].id]
    col_base = ctx.val_to_id[op.operands[1].id]

    # Row stride in elements (``tensor.stride[-2]`` for 2D
    # SharedRegions; falls back to ``shape[-1]`` for 1D test
    # fixtures that happen to size smem as a flat array).
    if len(tensor.stride) >= 2:
        row_stride_val = int(tensor.stride[-2])
    else:
        row_stride_val = int(tensor.shape[-1]) if tensor.shape else 1
    row_stride = ctx.text.const_uint(row_stride_val)

    # col + lane_id — the column position this lane reads.
    col_lane = ctx.text.alloc_id(f"lm_{which}_col_lane")
    ctx.text.emit_function(f"{col_lane} = OpIAdd {u32} {col_base} {lane_id}")

    # ``warp_dyn_offset`` (when set on the SharedRegion) is the warp-
    # uniform flat element offset that shifts this warp's view into
    # the shared backing region — multi-warp kernels partition smem
    # by giving each warp its own slice. Add it to the flat index.
    warp_off = getattr(tensor, "warp_dyn_offset", None)
    warp_off_id = ctx.val_to_id[warp_off.id] if warp_off is not None else None

    smem_scalar_ptr = ctx.text.type_pointer("Workgroup", smem_elem_type)
    elem_loads: list[str] = []

    if which == "b":
        # B: each slot is a u32 packing 2 bf16. The kernel cohort
        # stores B as ``(N, K)`` row-major (the "B^T in storage"
        # convention every gemm uses — see SPV ``_visit_load_matrix``
        # docstring on the Vulkan side for the same observation).
        # So the address arithmetic SWAPS the (lane, slot) → (outer,
        # inner) roles vs A:
        #   * Lane L → outer (N coord): N row in storage
        #   * Slot s pair → inner (K coord): K col in storage
        # Formula: flat = (row_op + L) * row_stride + (col_op + 2s + k_off)
        # where ``row_op`` is the n-base operand and ``col_op`` is the
        # k-base operand (the kernel cohort passes them with this
        # convention, see ``Gemm.kernel.py`` ``plan.b.load_from(g.B,
        # row=n_base, col=k_col_b)``).
        u16 = ctx.text.type_int(16, signed=False)
        u16_ptr = ctx.text.type_pointer("Workgroup", u16)
        # outer = row_base + lane_id (n coord into B's (N, K) storage).
        outer = ctx.text.alloc_id("lm_b_outer")
        ctx.text.emit_function(f"{outer} = OpIAdd {u32} {row_base} {lane_id}")
        outer_mul = ctx.text.alloc_id("lm_b_outer_mul")
        ctx.text.emit_function(
            f"{outer_mul} = OpIMul {u32} {outer} {row_stride}"
        )
        for s in range(width):
            slot_id = ctx.text.alloc_id(f"lm_b_slot_{s}")
            for k_off in range(2):
                # inner = col_base + 2s + k_off (k coord into storage).
                k_off_const = ctx.text.const_uint(2 * s + k_off)
                inner = ctx.text.alloc_id(f"lm_b_inner_{s}_{k_off}")
                ctx.text.emit_function(
                    f"{inner} = OpIAdd {u32} {col_base} {k_off_const}"
                )
                flat = ctx.text.alloc_id(f"lm_b_flat_{s}_{k_off}")
                ctx.text.emit_function(
                    f"{flat} = OpIAdd {u32} {outer_mul} {inner}"
                )
                if warp_off_id is not None:
                    flat_w = ctx.text.alloc_id(f"lm_b_flat_warp_{s}_{k_off}")
                    ctx.text.emit_function(
                        f"{flat_w} = OpIAdd {u32} {flat} {warp_off_id}"
                    )
                    flat = flat_w
                chain = ctx.text.alloc_id(f"lm_b_chain_{s}_{k_off}")
                # If smem is U16/BF16 (2-byte elem), access directly.
                # If smem is U32 (legacy lane-major test fixtures),
                # we treat the buffer as u16 via pointer bitcast.
                ctx.text.emit_function(
                    f"{chain} = OpAccessChain {u16_ptr} {smem_var_id} {flat}"
                )
                val_u16 = ctx.text.alloc_id(f"lm_b_val_{s}_{k_off}")
                ctx.text.emit_function(
                    f"{val_u16} = OpLoad {u16} {chain}"
                )
                # Extend to u32 for shift+OR packing.
                ext = ctx.text.alloc_id(f"lm_b_ext_{s}_{k_off}")
                ctx.text.emit_function(
                    f"{ext} = OpUConvert {u32} {val_u16}"
                )
                if k_off == 0:
                    # Even K row (k=2s) goes in LOW half.
                    slot_lo = ext
                else:
                    # Odd K row (k=2s+1) goes in HIGH half — shift left 16.
                    shifted = ctx.text.alloc_id(f"lm_b_shift_{s}_{k_off}")
                    sixteen = ctx.text.const_uint(16)
                    ctx.text.emit_function(
                        f"{shifted} = OpShiftLeftLogical {u32} {ext} {sixteen}"
                    )
                    packed = ctx.text.alloc_id(f"lm_b_pack_{s}")
                    ctx.text.emit_function(
                        f"{packed} = OpBitwiseOr {u32} {slot_lo} {shifted}"
                    )
                    slot_id = packed
            elem_loads.append(slot_id)
    else:
        # A or C: each slot is one lane-element at
        # ``smem[(row+s)*stride + (col+L)]``.
        #
        # Packed-K case (s8/s32 form A): the smem element is narrower
        # than the lane element (s8 in smem, u16 per lane). Each slot
        # then reads ``pack_ratio = lane_bytes / smem_bytes`` smem
        # bytes at K-offsets ``L*pack_ratio + k_off`` for
        # ``k_off ∈ [0, pack_ratio)``, packed little-endian into the
        # lane element. For bf16/f32 / f32/f32, pack_ratio==1 and the
        # loop degenerates to the single-element read.
        smem_bytes = _dtype_bytes_for_lane(smem_elem_type, ctx)
        lane_bytes = _dtype_bytes_for_lane(lane_elem_type, ctx)
        if lane_bytes < smem_bytes:
            raise NotImplementedError(
                f"_visit_load_matrix(ocl): lane elem ({lane_elem_type}) "
                f"narrower than smem elem ({smem_elem_type}) — not wired"
            )
        pack_ratio = lane_bytes // smem_bytes
        # Per Khronos SPV_INTEL_subgroup_matrix_multiply_accumulate
        # spec: lower-numbered invocations carry lower-numbered K
        # columns (sequential). For SG=16, K=32, pack_ratio=2: lane
        # L holds K-cols (2L, 2L+1).
        #
        # NOTE: this packing gives "even M-row correct, odd M-row
        # zero" output on Battlemage with all-ones probe — see
        # `Known OCL blockers` for the unresolved per-lane convention
        # gap. Same result with M-pair packing or byte-order swap;
        # the pattern is structural. Needs Intel GPU ISA spec or
        # oneDNN dpas micro-JIT reference to fix.
        pack_k_stride = 1
        a_pack_mpair = False  # disabled — see comment above
        if which == "a" and pack_ratio > 1 and not a_pack_mpair:
            # lane L → K-col base = L*pack_ratio, k_off ∈ [0..pack_ratio)
            # gives lane L K-cols (2L, 2L+1) for s8 form.
            col_lane_packed = ctx.text.alloc_id("lm_a_col_lane_pack")
            pack_const = ctx.text.const_uint(pack_ratio)
            lane_off = ctx.text.alloc_id("lm_a_lane_off")
            ctx.text.emit_function(
                f"{lane_off} = OpIMul {u32} {lane_id} {pack_const}"
            )
            ctx.text.emit_function(
                f"{col_lane_packed} = OpIAdd {u32} {col_base} {lane_off}"
            )
            col_lane = col_lane_packed

        for s in range(width):
            if a_pack_mpair:
                # M-pair packing for Intel s8 DPAS: slot s → K-col =
                # (col_lane + (s%2)*SG), M-pair = (2*(s//2), 2*(s//2)+1)
                # Low byte = even-M, high byte = odd-M.
                pair_m_low = 2 * (s // 2)
                pair_m_high = pair_m_low + 1
                k_off_in_lane = (s % 2) * ctx.subgroup_width
                k_pos = ctx.text.alloc_id(f"lm_a_kpos_{s}")
                k_off_const = ctx.text.const_uint(k_off_in_lane)
                ctx.text.emit_function(
                    f"{k_pos} = OpIAdd {u32} {col_lane} {k_off_const}"
                )

                def _read_byte_at(m_idx, k_pos, tag):
                    m_const = ctx.text.const_uint(m_idx)
                    rb = ctx.text.alloc_id(f"lm_a_mrow_{tag}")
                    ctx.text.emit_function(
                        f"{rb} = OpIAdd {u32} {row_base} {m_const}"
                    )
                    rm = ctx.text.alloc_id(f"lm_a_mrmul_{tag}")
                    ctx.text.emit_function(
                        f"{rm} = OpIMul {u32} {rb} {row_stride}"
                    )
                    fl = ctx.text.alloc_id(f"lm_a_mflat_{tag}")
                    ctx.text.emit_function(
                        f"{fl} = OpIAdd {u32} {rm} {k_pos}"
                    )
                    if warp_off_id is not None:
                        flw = ctx.text.alloc_id(f"lm_a_mflatw_{tag}")
                        ctx.text.emit_function(
                            f"{flw} = OpIAdd {u32} {fl} {warp_off_id}"
                        )
                        fl = flw
                    ch = ctx.text.alloc_id(f"lm_a_mchain_{tag}")
                    ctx.text.emit_function(
                        f"{ch} = OpAccessChain {smem_scalar_ptr} {smem_var_id} {fl}"
                    )
                    vs = ctx.text.alloc_id(f"lm_a_mval_{tag}")
                    ctx.text.emit_function(
                        f"{vs} = OpLoad {smem_elem_type} {ch}"
                    )
                    su = ctx.text.alloc_id(f"lm_a_msmu_{tag}")
                    smem_unsigned_t = ctx.text.type_int(
                        smem_bytes * 8, signed=False,
                    )
                    ctx.text.emit_function(
                        f"{su} = OpBitcast {smem_unsigned_t} {vs}"
                    )
                    wid = ctx.text.alloc_id(f"lm_a_mwid_{tag}")
                    ctx.text.emit_function(
                        f"{wid} = OpUConvert {lane_elem_type} {su}"
                    )
                    return wid

                lo = _read_byte_at(pair_m_low, k_pos, f"{s}_lo")
                hi = _read_byte_at(pair_m_high, k_pos, f"{s}_hi")
                shift_amt = ctx.text.const_uint(smem_bytes * 8)
                # Trying byte-order: high byte = even-m, low byte =
                # odd-m. The other order gives even-m correct, odd-m
                # zero; this swap tests the opposite convention.
                lo_shifted = ctx.text.alloc_id(f"lm_a_loshift_{s}")
                ctx.text.emit_function(
                    f"{lo_shifted} = OpShiftLeftLogical {lane_elem_type} {lo} {shift_amt}"
                )
                packed = ctx.text.alloc_id(f"lm_a_mpacked_{s}")
                ctx.text.emit_function(
                    f"{packed} = OpBitwiseOr {lane_elem_type} {hi} {lo_shifted}"
                )
                elem_loads.append(packed)
                continue

            row_s = ctx.text.alloc_id(f"lm_{which}_row_{s}")
            s_const = ctx.text.const_uint(s)
            ctx.text.emit_function(
                f"{row_s} = OpIAdd {u32} {row_base} {s_const}"
            )
            row_mul = ctx.text.alloc_id(f"lm_{which}_rmul_{s}")
            ctx.text.emit_function(
                f"{row_mul} = OpIMul {u32} {row_s} {row_stride}"
            )

            if pack_ratio == 1:
                flat = ctx.text.alloc_id(f"lm_{which}_flat_{s}")
                ctx.text.emit_function(
                    f"{flat} = OpIAdd {u32} {row_mul} {col_lane}"
                )
                if warp_off_id is not None:
                    flat_w = ctx.text.alloc_id(f"lm_{which}_flat_warp_{s}")
                    ctx.text.emit_function(
                        f"{flat_w} = OpIAdd {u32} {flat} {warp_off_id}"
                    )
                    flat = flat_w
                chain = ctx.text.alloc_id(f"lm_{which}_chain_{s}")
                ctx.text.emit_function(
                    f"{chain} = OpAccessChain {smem_scalar_ptr} {smem_var_id} {flat}"
                )
                val = ctx.text.alloc_id(f"lm_{which}_val_{s}")
                ctx.text.emit_function(
                    f"{val} = OpLoad {smem_elem_type} {chain}"
                )
                # Bitcast if smem dtype doesn't match lane elem dtype
                # (BF16↔U16 same width; F32↔F32 same).
                if smem_elem_type != lane_elem_type:
                    cast = ctx.text.alloc_id(f"lm_{which}_cast_{s}")
                    ctx.text.emit_function(
                        f"{cast} = OpBitcast {lane_elem_type} {val}"
                    )
                    val = cast
                elem_loads.append(val)
            else:
                # Packed: read ``pack_ratio`` smem elements at
                # K-offsets [0, pack_k_stride, 2*pack_k_stride, …],
                # pack little-endian into one lane element via
                # shift+OR. For cl_intel s8 DPAS, pack_k_stride =
                # SG (=16), giving lane L → K-cols [L, L+16].
                packed = None
                for k_off in range(pack_ratio):
                    k_off_const = ctx.text.const_uint(k_off * pack_k_stride)
                    pos = ctx.text.alloc_id(f"lm_{which}_pos_{s}_{k_off}")
                    ctx.text.emit_function(
                        f"{pos} = OpIAdd {u32} {col_lane} {k_off_const}"
                    )
                    flat = ctx.text.alloc_id(f"lm_{which}_flat_{s}_{k_off}")
                    ctx.text.emit_function(
                        f"{flat} = OpIAdd {u32} {row_mul} {pos}"
                    )
                    if warp_off_id is not None:
                        flat_w = ctx.text.alloc_id(
                            f"lm_{which}_flat_warp_{s}_{k_off}"
                        )
                        ctx.text.emit_function(
                            f"{flat_w} = OpIAdd {u32} {flat} {warp_off_id}"
                        )
                        flat = flat_w
                    chain = ctx.text.alloc_id(
                        f"lm_{which}_chain_{s}_{k_off}"
                    )
                    ctx.text.emit_function(
                        f"{chain} = OpAccessChain {smem_scalar_ptr} "
                        f"{smem_var_id} {flat}"
                    )
                    val_smem = ctx.text.alloc_id(
                        f"lm_{which}_val_smem_{s}_{k_off}"
                    )
                    ctx.text.emit_function(
                        f"{val_smem} = OpLoad {smem_elem_type} {chain}"
                    )
                    # Widen smem element to the lane element type
                    # (e.g. s8 → u16). Use OpUConvert via an
                    # unsigned intermediate type so signed s8s don't
                    # get sign-extended into the high byte — DPAS
                    # interprets the int8 carrier byte-wise.
                    # smem s8 is signed; bitcast to unsigned then
                    # zero-extend.
                    smem_u = ctx.text.alloc_id(
                        f"lm_{which}_smem_u_{s}_{k_off}"
                    )
                    smem_unsigned_t = ctx.text.type_int(
                        smem_bytes * 8, signed=False,
                    )
                    ctx.text.emit_function(
                        f"{smem_u} = OpBitcast {smem_unsigned_t} {val_smem}"
                    )
                    widened = ctx.text.alloc_id(
                        f"lm_{which}_widened_{s}_{k_off}"
                    )
                    ctx.text.emit_function(
                        f"{widened} = OpUConvert {lane_elem_type} {smem_u}"
                    )
                    if k_off == 0:
                        packed = widened
                    else:
                        shift_amt = ctx.text.const_uint(k_off * smem_bytes * 8)
                        shifted = ctx.text.alloc_id(
                            f"lm_{which}_shifted_{s}_{k_off}"
                        )
                        ctx.text.emit_function(
                            f"{shifted} = OpShiftLeftLogical "
                            f"{lane_elem_type} {widened} {shift_amt}"
                        )
                        new_packed = ctx.text.alloc_id(
                            f"lm_{which}_packed_{s}_{k_off}"
                        )
                        ctx.text.emit_function(
                            f"{new_packed} = OpBitwiseOr "
                            f"{lane_elem_type} {packed} {shifted}"
                        )
                        packed = new_packed
                assert packed is not None
                elem_loads.append(packed)

    res_id = ctx.text.alloc_id(f"mma_load_{which}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {lane_vec_type} "
        f"{' '.join(elem_loads)}"
    )


def _visit_store_matrix(op: StoreMatrixOp, ctx: _OclCtx) -> None:
    """``StoreMatrixOp`` → per-lane scatter of the Intel fragment back
    to row-major smem. Mirror of ``_visit_load_matrix``.

    Lane L scatters its v8f32 (C/D layout) into rows m=0..7 of column
    n=L of the destination tile. Per-slot scatter:
    ``smem[(row+s)*row_stride + (col+L)] = OpCompositeExtract frag s``.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    tensor = op.attrs["dst_tensor"]
    if not isinstance(tensor, SharedRegion):
        raise NotImplementedError(
            "_visit_store_matrix(ocl): only SharedRegion destinations wired"
        )
    shape_id = op.attrs["shape_id"]
    which = op.attrs["which"]

    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_store_matrix(ocl): unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    layout = _intel_mma_layout(shape)
    (
        _sg, _a_dt, _a_w, _b_dt, _b_w, c_elem_dt, c_width,
        _k_dim, _operands_flag,
    ) = layout

    if which not in ("c", "d"):
        raise NotImplementedError(
            f"_visit_store_matrix(ocl): which={which!r} — only c/d wired"
        )

    frag_v = op.operands[0]
    frag_id = ctx.val_to_id[frag_v.id]
    row_base = ctx.val_to_id[op.operands[1].id]
    col_base = ctx.val_to_id[op.operands[2].id]

    rec = ctx.smem_allocs.get(tensor.alloc.id)
    if rec is None:
        raise RuntimeError(
            f"_visit_store_matrix(ocl): SharedRegion {tensor.name!r} accessed "
            "before its SmemAllocOp was visited"
        )
    smem_var_id, smem_elem_type, _smem_elem_ptr, _n = rec

    lane_elem_type = _emit_dtype(ctx.text, c_elem_dt, ctx)

    u32 = ctx.text.type_int(32, signed=False)
    lane_id = _ensure_lane_id(ctx)

    if len(tensor.stride) >= 2:
        row_stride_val = int(tensor.stride[-2])
    else:
        row_stride_val = int(tensor.shape[-1]) if tensor.shape else 1
    row_stride = ctx.text.const_uint(row_stride_val)

    col_lane = ctx.text.alloc_id("sm_col_lane")
    ctx.text.emit_function(f"{col_lane} = OpIAdd {u32} {col_base} {lane_id}")

    warp_off = getattr(tensor, "warp_dyn_offset", None)
    warp_off_id = ctx.val_to_id[warp_off.id] if warp_off is not None else None

    smem_scalar_ptr = ctx.text.type_pointer("Workgroup", smem_elem_type)
    for s in range(c_width):
        # Extract slot s from the fragment (row m=s value).
        elem = ctx.text.alloc_id(f"sm_elem_{s}")
        ctx.text.emit_function(
            f"{elem} = OpCompositeExtract {lane_elem_type} {frag_id} {s}"
        )
        # Bitcast to smem element dtype if they differ
        # (e.g. F32→F32 same, but BF16 acc stored to U16 buffer).
        if smem_elem_type != lane_elem_type:
            cast = ctx.text.alloc_id(f"sm_cast_{s}")
            ctx.text.emit_function(
                f"{cast} = OpBitcast {smem_elem_type} {elem}"
            )
            elem = cast
        row_s = ctx.text.alloc_id(f"sm_row_{s}")
        s_const = ctx.text.const_uint(s)
        ctx.text.emit_function(
            f"{row_s} = OpIAdd {u32} {row_base} {s_const}"
        )
        row_mul = ctx.text.alloc_id(f"sm_rmul_{s}")
        ctx.text.emit_function(
            f"{row_mul} = OpIMul {u32} {row_s} {row_stride}"
        )
        flat = ctx.text.alloc_id(f"sm_flat_{s}")
        ctx.text.emit_function(
            f"{flat} = OpIAdd {u32} {row_mul} {col_lane}"
        )
        if warp_off_id is not None:
            flat_w = ctx.text.alloc_id(f"sm_flat_warp_{s}")
            ctx.text.emit_function(
                f"{flat_w} = OpIAdd {u32} {flat} {warp_off_id}"
            )
            flat = flat_w
        chain = ctx.text.alloc_id(f"sm_chain_{s}")
        ctx.text.emit_function(
            f"{chain} = OpAccessChain {smem_scalar_ptr} {smem_var_id} {flat}"
        )
        ctx.text.emit_function(f"OpStore {chain} {elem}")


# (kind, dtype_kind) → SPIR-V ``OpGroupNonUniform*`` opcode. Same
# table as the Vulkan lowerer — these opcodes are dialect-agnostic
# and IGC accepts the KHR/standard subgroup arithmetic set.
_SUBGROUP_REDUCE_TO_OP: dict[tuple[str, str], str] = {
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


def _visit_subgroup_reduce(op: SubgroupReduceOp, ctx: _OclCtx) -> None:
    """Cross-lane reduction. ``OpGroupNonUniform*`` with Subgroup
    execution scope + ``Reduce`` group-operation. Every lane gets
    the same scalar result. Dialect-agnostic — same emit as Vulkan.
    """
    (out,) = op.results
    src_v = op.operands[0]
    kind = op.attrs["op"]
    dt_kind = _dtype_kind(src_v.dtype)
    spv_op = _SUBGROUP_REDUCE_TO_OP.get((kind, dt_kind))
    if spv_op is None:
        raise NotImplementedError(
            f"_visit_subgroup_reduce(ocl): op={kind!r} dtype_kind={dt_kind!r}"
        )
    ctx.text.add_capability("GroupNonUniformArithmetic")
    src_id = ctx.val_to_id[src_v.id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)
    sg_scope = ctx.text.const_uint(3)
    res_id = ctx.text.alloc_id(f"sgreduce_{kind}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = {spv_op} {dst_t} {sg_scope} Reduce {src_id}"
    )


def _visit_group_id(op: GroupIdOp, ctx: _OclCtx) -> None:
    """``group_id()`` = ``laneid >> 2`` (PTX MMA fragment formula).
    SPIR-V has no equivalent builtin; emit the explicit shift on
    SubgroupLocalInvocationId. Mirrors the SPV side."""
    (out,) = op.results
    lane = _ensure_lane_id(ctx)
    ctx.text.add_capability("GroupNonUniform")
    u32 = ctx.text.type_int(32, signed=False)
    two = ctx.text.const_uint(2)
    res_id = ctx.text.alloc_id("group_id")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpShiftRightLogical {u32} {lane} {two}")


def _visit_thread_id_in_group(op: ThreadIdInGroupOp, ctx: _OclCtx) -> None:
    """``thread_id_in_group()`` = ``laneid & 3``."""
    (out,) = op.results
    lane = _ensure_lane_id(ctx)
    ctx.text.add_capability("GroupNonUniform")
    u32 = ctx.text.type_int(32, signed=False)
    three = ctx.text.const_uint(3)
    res_id = ctx.text.alloc_id("tid_in_group")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpBitwiseAnd {u32} {lane} {three}")


def _visit_shuffle(op: ShuffleOp, ctx: _OclCtx) -> None:
    """Cross-lane shuffle within a subgroup. Maps ``kind`` to the
    matching ``OpGroupNonUniformShuffle*`` opcode. Dialect-agnostic —
    same as the Vulkan path."""
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
        raise NotImplementedError(f"_visit_shuffle(ocl): kind={kind!r}")
    ctx.text.add_capability("GroupNonUniformShuffle")

    sg_scope = ctx.text.const_uint(3)
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


# ``AtomicRmwOp.attrs['op']`` → SPIR-V atomic opcode. Same table as
# Vulkan side — these opcodes are dialect-agnostic; the only delta
# vs Vulkan is min/max get dispatched by signedness at emit time
# (mirrored in ``_visit_atomic_rmw`` below).
_ATOMIC_OP_BY_KIND: dict[str, str] = {
    "add": "OpAtomicIAdd",
    "and": "OpAtomicAnd",
    "or": "OpAtomicOr",
    "xor": "OpAtomicXor",
    "exch": "OpAtomicExchange",
}


def _visit_atomic_rmw(op: AtomicRmwOp, ctx: _OclCtx) -> None:
    """Atomic read-modify-write on a GlobalTensor slot.

    Memory scope = Device (1), semantics = Relaxed (0) — same constants
    as Vulkan. OCL spec accepts them; ordering across atomics is the
    caller's responsibility (a ``barrier()`` op pairs with the rmw
    when needed). Address computation differs from Vulkan: OCL uses
    ``OpInBoundsPtrAccessChain`` on the CrossWorkgroup kernel-param
    pointer instead of OpAccessChain through a Block-decorated struct.
    """
    (out,) = op.results
    tensor = op.attrs["tensor"]
    kind = op.attrs["op"]

    if not isinstance(tensor, GlobalTensor):
        raise NotImplementedError(
            f"_visit_atomic_rmw(ocl): tensor type "
            f"{type(tensor).__name__} not wired (only GlobalTensor today)"
        )

    arg_id = ctx.param_to_arg[id(tensor.param)]
    elem_ptr = ctx.param_to_elem_ptr[id(tensor.param)]

    value_id = ctx.val_to_id[op.operands[0].id]
    idx_id = _flatten_global_index(tuple(op.operands[1:]), tensor, ctx)
    chain_id = ctx.text.alloc_id("atomic_chain")
    ctx.text.emit_function(
        f"{chain_id} = OpInBoundsPtrAccessChain {elem_ptr} {arg_id} {idx_id}"
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
                f"_visit_atomic_rmw(ocl): op={kind!r} not yet wired"
            )

    scope_id = ctx.text.const_uint(1)      # Device
    semantics_id = ctx.text.const_uint(0)  # Relaxed
    ctx.text.emit_function(
        f"{res_id} = {spv_op} {type_id} {chain_id} {scope_id} "
        f"{semantics_id} {value_id}"
    )


def _visit_subgroup_id(op: SubgroupIdOp, ctx: _OclCtx) -> None:
    """``subgroup_id()`` → ``SubgroupId`` builtin (the subgroup's
    index within the workgroup). Lazy-decl pattern; capability
    ``GroupNonUniform`` declared on first use."""
    (out,) = op.results
    ctx.text.add_capability("GroupNonUniform")
    var_id = ctx.subgroup_id_var
    if not var_id:
        u32 = ctx.text.type_int(32, signed=False)
        ptr = ctx.text.type_pointer("Input", u32)
        var_id = ctx.text.alloc_id("SubgroupId")
        ctx.text.add_type_line(f"{var_id} = OpVariable {ptr} Input")
        ctx.text.add_decoration(f"OpDecorate {var_id} BuiltIn SubgroupId")
        ctx.subgroup_id_var = var_id
    u32 = ctx.text.type_int(32, signed=False)
    res_id = ctx.text.alloc_id("sg_id")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpLoad {u32} {var_id}")


def _visit_for_loop(op: ForLoopOp, ctx: _OclCtx) -> None:
    """Counted for-loop with optional loop-carried values.

    Structured loop pattern (same as Vulkan path — ``OpLoopMerge`` +
    ``OpBranchConditional`` + ``OpPhi`` at the header are dialect-
    agnostic):

        OpBranch %preheader
        %preheader = OpLabel
        OpBranch %header

        %header = OpLabel
          %iv = OpPhi <iv_t> %lo %preheader %iv_next %continue
          %ck = OpPhi <ck_t> %ck_init %preheader %ck_next %continue
          %cond = OpULessThan/OpSLessThan %iv %hi
          OpLoopMerge %merge %continue None
          OpBranchConditional %cond %body %merge

        %body = OpLabel
          ;; body ops; YieldOp materialises ck_next via OpCopyObject
          OpBranch %continue

        %continue = OpLabel
          %iv_next = OpIAdd <iv_t> %iv %step
          OpBranch %header

        %merge = OpLabel

    Coopmat-typed carries: NOT a concern for OCL since the Intel MMA
    surface uses plain per-lane vectors (no opaque coopmat type),
    unlike Vulkan KHR_cooperative_matrix. So this path is strictly
    simpler than ``quark.lower.spv.lower._visit_for_loop``.
    """
    iv_dtype = op.attrs["iv_dtype"]
    iv_kind = _dtype_kind(iv_dtype)
    if iv_kind not in ("uint", "sint"):
        raise NotImplementedError(
            f"_visit_for_loop(ocl): only integer induction supported, got {iv_dtype!r}"
        )
    iv_type = _emit_dtype(ctx.text, iv_dtype, ctx)

    lo_id = ctx.val_to_id[op.lo.id]
    hi_id = ctx.val_to_id[op.hi.id]
    step_id = ctx.val_to_id[op.step.id]
    carried_init_ids = [ctx.val_to_id[c.id] for c in op.carried_in]

    preheader = ctx.text.alloc_id("loop_pre")
    header = ctx.text.alloc_id("loop_hdr")
    body_label = ctx.text.alloc_id("loop_body")
    cont_label = ctx.text.alloc_id("loop_cont")
    merge_label = ctx.text.alloc_id("loop_mrg")

    iv_next_id = ctx.text.alloc_id("iv_next")

    carry_phi_ids: list[str] = []
    carry_next_ids: list[str] = []
    carry_type_ids: list[str] = []
    for i, body_var in enumerate(op.carried_body_vars):
        elem_t = _emit_dtype(ctx.text, body_var.dtype, ctx)
        if body_var.width > 1:
            c_type = _ocl_type_vec(ctx.text, elem_t, body_var.width)
        else:
            c_type = elem_t
        phi_id = ctx.text.alloc_id(f"carry{i}")
        next_id = ctx.text.alloc_id(f"carry{i}_next")
        carry_phi_ids.append(phi_id)
        carry_next_ids.append(next_id)
        carry_type_ids.append(c_type)
        ctx.val_to_id[body_var.id] = phi_id
        ctx.val_to_id[op.results[i].id] = phi_id

    ctx.val_to_id[op.induction_var.id] = ctx.text.alloc_id("iv")
    iv_phi_id = ctx.val_to_id[op.induction_var.id]

    ctx.text.emit_function(f"OpBranch {preheader}")
    ctx.text.emit_function(f"{preheader} = OpLabel")
    ctx.text.emit_function(f"OpBranch {header}")

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
    # ``unroll=True`` on the IR ForLoopOp → emit ``OpLoopMerge ...
    # Unroll`` (LoopControl::Unroll = 0x1). IGC honors this hint and
    # unrolls const-bound loops; falls back to runtime form for
    # truly-runtime bounds. The PTX/Metal paths Python-unroll at
    # lower time instead — see ``quark.ir.op.ForLoopOp`` docstring.
    loop_control = "Unroll" if op.attrs.get("unroll", False) else "None"
    ctx.text.emit_function(f"OpLoopMerge {merge_label} {cont_label} {loop_control}")
    ctx.text.emit_function(
        f"OpBranchConditional {cond_id} {body_label} {merge_label}"
    )

    ctx.text.emit_function(f"{body_label} = OpLabel")
    ctx.loop_yield_stack.append(
        list(zip(carry_next_ids, carry_type_ids, strict=False))
    )
    try:
        for body_op in op.body.ops:
            _walk_op(body_op, ctx)
    finally:
        ctx.loop_yield_stack.pop()
    ctx.text.emit_function(f"OpBranch {cont_label}")

    ctx.text.emit_function(f"{cont_label} = OpLabel")
    ctx.text.emit_function(
        f"{iv_next_id} = OpIAdd {iv_type} {iv_phi_id} {step_id}"
    )
    ctx.text.emit_function(f"OpBranch {header}")
    ctx.text.emit_function(f"{merge_label} = OpLabel")


def _visit_convert(op: ConvertOp, ctx: _OclCtx) -> None:
    """Type conversion — dialect-agnostic ``OpConvert*`` / ``OpFConvert``
    / ``OpUConvert`` / ``OpSConvert`` selection by src/dst dtype kind.
    """
    (out,) = op.results
    src = op.operands[0]
    src_id = ctx.val_to_id[src.id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)

    src_kind = _dtype_kind(src.dtype)
    dst_kind = _dtype_kind(out.dtype)
    if src_kind == "float" and dst_kind == "float":
        opcode = "OpFConvert"
    elif src_kind == "float" and dst_kind == "sint":
        opcode = "OpConvertFToS"
    elif src_kind == "float" and dst_kind == "uint":
        opcode = "OpConvertFToU"
    elif src_kind == "sint" and dst_kind == "float":
        opcode = "OpConvertSToF"
    elif src_kind == "uint" and dst_kind == "float":
        opcode = "OpConvertUToF"
    elif src_kind == "sint" and dst_kind == "sint":
        opcode = "OpSConvert"
    elif src_kind == "uint" and dst_kind == "uint":
        opcode = "OpUConvert"
    elif (src_kind, dst_kind) in (("sint", "uint"), ("uint", "sint")):
        # Same-width signedness flip → OpBitcast; different-width
        # needs an intermediate convert in the source signedness
        # before the bitcast. SPIR-V's spec disallows
        # bitcast across widths.
        if src.dtype.bytes == out.dtype.bytes:
            opcode = "OpBitcast"
        else:
            # Convert width first in source signedness, then bitcast
            # signedness. Emit two ops.
            intermediate_dtype = _matching_width_dtype(
                src_kind, out.dtype.bytes,
            )
            tmp_t = _emit_dtype(ctx.text, intermediate_dtype, ctx)
            tmp_id = ctx.text.alloc_id("cvt_w")
            width_op = "OpSConvert" if src_kind == "sint" else "OpUConvert"
            ctx.text.emit_function(f"{tmp_id} = {width_op} {tmp_t} {src_id}")
            src_id = tmp_id
            opcode = "OpBitcast"
    else:
        raise NotImplementedError(
            f"_visit_convert(ocl): {src.dtype!r} → {out.dtype!r} not wired"
        )
    res_id = ctx.text.alloc_id("cvt")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = {opcode} {dst_t} {src_id}")


def _matching_width_dtype(kind: str, bytes_: int) -> DType:
    """Find the DType of the given kind ('sint'/'uint') with the
    requested byte width. Used by _visit_convert when an sint↔uint
    flip also needs a width change."""
    table = {
        ("sint", 1): DType.S8, ("sint", 2): DType.S16, ("sint", 4): DType.S32,
        ("uint", 1): DType.U8, ("uint", 2): DType.U16, ("uint", 4): DType.U32,
    }
    if (kind, bytes_) not in table:
        raise NotImplementedError(
            f"_matching_width_dtype(ocl): no {kind} dtype with {bytes_} bytes"
        )
    return table[(kind, bytes_)]


def _dtype_bytes_for_lane(spv_type_id: str, ctx: _OclCtx) -> int:
    """Reverse-lookup a SPIR-V scalar type-id back to its byte width.

    Walks the SPIR-V text's type lines for the declared scalar.
    Cheaper than passing dtype around — visitors already have the
    type-id from ``_emit_dtype`` / ``smem_elem_type``.
    """
    # Look for the type-id's declaration in the assembler's emitted
    # type lines (e.g. ``%int_8_1 = OpTypeInt 8 1`` → 1 byte).
    for line in ctx.text.type_lines:
        if not line.startswith(spv_type_id + " = "):
            continue
        rest = line.split(" = ", 1)[1].strip()
        toks = rest.split()
        if toks[0] == "OpTypeInt":
            bits = int(toks[1])
            return max(1, bits // 8)
        if toks[0] == "OpTypeFloat":
            return max(1, int(toks[1]) // 8)
        # Anything else (OpTypeBool, OpTypeVector…) isn't a scalar
        # the lane-elem reverse lookup should hit.
        raise NotImplementedError(
            f"_dtype_bytes_for_lane: type-id {spv_type_id!r} resolves to "
            f"{toks[0]!r}; only OpTypeInt/OpTypeFloat scalars are handled."
        )
    raise RuntimeError(
        f"_dtype_bytes_for_lane: type-id {spv_type_id!r} not declared in "
        "ctx.text.type_lines"
    )


def _dtype_kind(dt: DType) -> str:
    if dt in (DType.F16, DType.BF16, DType.F32):
        return "float"
    if dt in (DType.U8, DType.U16, DType.U32, DType.PRED):
        return "uint"
    if dt in (DType.S8, DType.S16, DType.S32):
        return "sint"
    raise NotImplementedError(f"_dtype_kind(ocl): unknown {dt!r}")


def _visit_bitcast(op: BitcastOp, ctx: _OclCtx) -> None:
    """Reinterpret cast via ``OpBitcast`` — width must match
    (validated at the IR level)."""
    (out,) = op.results
    src_id = ctx.val_to_id[op.operands[0].id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)
    res_id = ctx.text.alloc_id("bcast")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(f"{res_id} = OpBitcast {dst_t} {src_id}")


def _visit_vec_build(op: VecBuildOp, ctx: _OclCtx) -> None:
    """Build a vector from scalar operands via ``OpCompositeConstruct``.

    With ``packed_b32=True`` the operands are B32 scalars each carrying
    two BF16/F16 elements (low half = lane 2k, high half = lane 2k+1).
    The result is a width=2N vector of the element dtype. Lower as a
    v<N> packed vector then ``OpBitcast`` to the bf16 vector type —
    same total bit-width so the cast is well-formed.
    """
    (out,) = op.results
    width = int(op.attrs.get("width", out.width or len(op.operands)))
    elem_type = _emit_dtype(ctx.text, out.dtype, ctx)
    vec_type = _ocl_type_vec(ctx.text, elem_type, width)
    operands = [ctx.val_to_id[v.id] for v in op.operands]
    res_id = ctx.text.alloc_id("vec_build")
    ctx.val_to_id[out.id] = res_id

    if op.attrs.get("packed_b32"):
        # Pack B32 operands into a uint32 vector, then bitcast to the
        # bf16/f16 vec type (twice the lane count, same bits).
        u32_t = ctx.text.type_int(32, signed=False)
        u32_vec_t = _ocl_type_vec(ctx.text, u32_t, len(operands))
        u32_vec_id = ctx.text.alloc_id("vec_b32_pack")
        ctx.text.emit_function(
            f"{u32_vec_id} = OpCompositeConstruct {u32_vec_t} {' '.join(operands)}"
        )
        ctx.text.emit_function(
            f"{res_id} = OpBitcast {vec_type} {u32_vec_id}"
        )
        return

    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {vec_type} {' '.join(operands)}"
    )


def _visit_vec_extract(op: VecExtractOp, ctx: _OclCtx) -> None:
    """Extract one element from a vector via ``OpCompositeExtract``.

    With ``packed_b32=True`` the source is a BF16/F16 vector and the
    desired result is a B32 holding the (2*idx, 2*idx+1) lane pair.
    Lower as ``OpBitcast`` of the source to a uint32 vector, then
    ``OpCompositeExtract`` at index ``pair_idx``.
    """
    (out,) = op.results
    src_id = ctx.val_to_id[op.operands[0].id]
    idx = int(op.attrs["index"])
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)
    res_id = ctx.text.alloc_id("vec_ext")
    ctx.val_to_id[out.id] = res_id

    if op.attrs.get("packed_b32"):
        src_vec = op.operands[0]
        src_width = src_vec.width or 0
        if src_width == 0 or src_width % 2 != 0:
            raise NotImplementedError(
                "_visit_vec_extract(ocl): packed_b32 requires even "
                f"source width, got {src_width!r}"
            )
        u32_t = ctx.text.type_int(32, signed=False)
        u32_vec_t = _ocl_type_vec(ctx.text, u32_t, src_width // 2)
        u32_vec_id = ctx.text.alloc_id("vec_b32_view")
        ctx.text.emit_function(
            f"{u32_vec_id} = OpBitcast {u32_vec_t} {src_id}"
        )
        ctx.text.emit_function(
            f"{res_id} = OpCompositeExtract {dst_t} {u32_vec_id} {idx}"
        )
        return

    ctx.text.emit_function(
        f"{res_id} = OpCompositeExtract {dst_t} {src_id} {idx}"
    )


def _visit_vec_load(op: VecLoadOp, ctx: _OclCtx) -> None:
    """Vector load — N scalar OpLoads followed by ``OpCompositeConstruct``.

    Minimal first cut: only supports ``out.dtype == tensor.dtype`` (no
    reinterpret-load packing). The SPV side's bf16-as-b32 packing path
    can be ported in a follow-up once a kernel demands it. Common
    case (Elementwise / SiLU vectorized loads on a matching dtype)
    works through this path.
    """
    (out,) = op.results
    tensor = op.attrs["tensor"]
    width = int(op.attrs["width"])
    indices = list(op.operands)
    if op.attrs.get("pred") is not None:
        indices = indices[:-1]

    # bf16-as-b32 packing path: source bf16 (or u16-class), output
    # b32 (or u32). Each output element packs 2 source elements
    # along the inner-most index (low half = idx 2i, high half =
    # idx 2i+1). Mirrors the packed-K path in _visit_load_matrix.
    pack_ratio = 1
    if out.dtype is not tensor.dtype:
        src_bytes = _DTYPE_BYTES.get(tensor.dtype, 0)
        dst_bytes = _DTYPE_BYTES.get(out.dtype, 0)
        if src_bytes == 0 or dst_bytes == 0 or dst_bytes % src_bytes != 0:
            raise NotImplementedError(
                f"_visit_vec_load(ocl): dtype mix ({tensor.dtype!r} → "
                f"{out.dtype!r}) not yet wired in OCL first cut"
            )
        pack_ratio = dst_bytes // src_bytes

    if isinstance(tensor, GlobalTensor):
        arg_id = ctx.param_to_arg[id(tensor.param)]
        elem_ptr = ctx.param_to_elem_ptr[id(tensor.param)]
        elem_type = ctx.param_to_elem_type[id(tensor.param)]
        base_id = _flatten_global_index(tuple(indices), tensor, ctx)
        chain_op = "OpInBoundsPtrAccessChain"
        chain_base = f"{arg_id}"
    elif isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_vec_load(ocl): SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp"
            )
        var_id, elem_type, elem_ptr, _n = rec
        base_id = _flatten_index(
            tuple(indices), tensor.shape, ctx, stride=tuple(tensor.stride)
        )
        base_id = _add_dyn_offset(base_id, tensor, ctx)
        chain_op = "OpAccessChain"
        chain_base = f"{var_id}"
    else:
        raise NotImplementedError(
            f"_visit_vec_load(ocl): tensor type {type(tensor).__name__}"
        )

    u32 = ctx.text.type_int(32, signed=False)
    out_elem_type = _emit_dtype(ctx.text, out.dtype, ctx)
    loaded: list[str] = []
    for i in range(width):
        if pack_ratio == 1:
            if i == 0:
                idx_id = base_id
            else:
                inc = ctx.text.const_uint(i)
                idx_id = ctx.text.alloc_id(f"vec_idx_{i}")
                ctx.text.emit_function(f"{idx_id} = OpIAdd {u32} {base_id} {inc}")
            chain_id = ctx.text.alloc_id(f"vec_chain_{i}")
            ctx.text.emit_function(
                f"{chain_id} = {chain_op} {elem_ptr} {chain_base} {idx_id}"
            )
            ld_id = ctx.text.alloc_id(f"vec_ld_{i}")
            ctx.text.emit_function(f"{ld_id} = OpLoad {elem_type} {chain_id}")
            loaded.append(ld_id)
            continue

        # Packed: read pack_ratio source elements at consecutive
        # offsets, widen + shift + OR into one out-elem.
        packed_id = None
        src_bits = _DTYPE_BYTES[tensor.dtype] * 8
        for k in range(pack_ratio):
            inc = ctx.text.const_uint(i * pack_ratio + k)
            idx_id = ctx.text.alloc_id(f"vec_idx_{i}_{k}")
            ctx.text.emit_function(f"{idx_id} = OpIAdd {u32} {base_id} {inc}")
            chain_id = ctx.text.alloc_id(f"vec_chain_{i}_{k}")
            ctx.text.emit_function(
                f"{chain_id} = {chain_op} {elem_ptr} {chain_base} {idx_id}"
            )
            ld_id = ctx.text.alloc_id(f"vec_ld_{i}_{k}")
            ctx.text.emit_function(f"{ld_id} = OpLoad {elem_type} {chain_id}")
            # Reinterpret the source element to its unsigned-int
            # carrier (bf16 → u16) so we can widen + shift cleanly.
            src_uint_t = ctx.text.type_int(src_bits, signed=False)
            carrier = ld_id
            if elem_type != src_uint_t:
                rb = ctx.text.alloc_id(f"vec_rcast_{i}_{k}")
                ctx.text.emit_function(
                    f"{rb} = OpBitcast {src_uint_t} {ld_id}"
                )
                carrier = rb
            widened = ctx.text.alloc_id(f"vec_wid_{i}_{k}")
            ctx.text.emit_function(
                f"{widened} = OpUConvert {out_elem_type} {carrier}"
            )
            if k == 0:
                packed_id = widened
            else:
                shift_amt = ctx.text.const_uint(k * src_bits)
                shifted = ctx.text.alloc_id(f"vec_shift_{i}_{k}")
                ctx.text.emit_function(
                    f"{shifted} = OpShiftLeftLogical {out_elem_type} "
                    f"{widened} {shift_amt}"
                )
                new_packed = ctx.text.alloc_id(f"vec_packed_{i}_{k}")
                ctx.text.emit_function(
                    f"{new_packed} = OpBitwiseOr {out_elem_type} "
                    f"{packed_id} {shifted}"
                )
                packed_id = new_packed
        assert packed_id is not None
        loaded.append(packed_id)

    vec_type = _ocl_type_vec(ctx.text, out_elem_type, width)
    res_id = ctx.text.alloc_id(f"vec_w{width}")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {vec_type} {' '.join(loaded)}"
    )


def _visit_vec_store(op: VecStoreOp, ctx: _OclCtx) -> None:
    """Vector store — N ``OpCompositeExtract``s + N scalar OpStores.

    Same minimal-coverage constraint as ``_visit_vec_load``: only the
    matching-dtype case is wired. Honors ``pred=`` via OpSelection
    Merge gating (same rule as scalar StoreOp — see project_spv_
    correctness_bug.md for the pred-drop bug class). ``width`` comes
    from the value operand's vector shape (not ``attrs``)."""
    tensor = op.attrs["tensor"]
    width = int(op.operands[0].width)
    pred_val = op.attrs.get("pred")
    raw_ops = list(op.operands)
    if pred_val is not None:
        index_ops = raw_ops[1:-1]
    else:
        index_ops = raw_ops[1:]
    value_v = raw_ops[0]
    # Packed-store path: source is wider (e.g. B32) than tensor
    # element (e.g. BF16). Each input element unpacks into pack_ratio
    # output elements at consecutive offsets (low bits = idx 2i,
    # high bits = idx 2i+1). Mirror of the packed-load path above.
    pack_ratio = 1
    if value_v.dtype is not tensor.dtype:
        src_bytes = _DTYPE_BYTES.get(value_v.dtype, 0)
        dst_bytes = _DTYPE_BYTES.get(tensor.dtype, 0)
        if src_bytes == 0 or dst_bytes == 0 or src_bytes % dst_bytes != 0:
            raise NotImplementedError(
                f"_visit_vec_store(ocl): dtype mix ({value_v.dtype!r} → "
                f"{tensor.dtype!r}) not yet wired"
            )
        pack_ratio = src_bytes // dst_bytes

    if isinstance(tensor, GlobalTensor):
        arg_id = ctx.param_to_arg[id(tensor.param)]
        elem_ptr = ctx.param_to_elem_ptr[id(tensor.param)]
        elem_type = ctx.param_to_elem_type[id(tensor.param)]
        base_id = _flatten_global_index(tuple(index_ops), tensor, ctx)
        chain_op = "OpInBoundsPtrAccessChain"
        chain_base = f"{arg_id}"
    elif isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_vec_store(ocl): SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp"
            )
        var_id, elem_type, elem_ptr, _n = rec
        base_id = _flatten_index(
            tuple(index_ops), tensor.shape, ctx, stride=tuple(tensor.stride)
        )
        base_id = _add_dyn_offset(base_id, tensor, ctx)
        chain_op = "OpAccessChain"
        chain_base = f"{var_id}"
    else:
        raise NotImplementedError(
            f"_visit_vec_store(ocl): tensor type {type(tensor).__name__}"
        )

    value_id = ctx.val_to_id[value_v.id]
    u32 = ctx.text.type_int(32, signed=False)

    src_elem_type = _emit_dtype(ctx.text, value_v.dtype, ctx)

    def _emit_one_store(i: int) -> None:
        if pack_ratio == 1:
            if i == 0:
                idx_id = base_id
            else:
                inc = ctx.text.const_uint(i)
                idx_id = ctx.text.alloc_id(f"vs_idx_{i}")
                ctx.text.emit_function(
                    f"{idx_id} = OpIAdd {u32} {base_id} {inc}"
                )
            chain_id = ctx.text.alloc_id(f"vs_chain_{i}")
            ctx.text.emit_function(
                f"{chain_id} = {chain_op} {elem_ptr} {chain_base} {idx_id}"
            )
            elem_id = ctx.text.alloc_id(f"vs_elem_{i}")
            ctx.text.emit_function(
                f"{elem_id} = OpCompositeExtract {elem_type} {value_id} {i}"
            )
            ctx.text.emit_function(f"OpStore {chain_id} {elem_id}")
            return

        # Packed: extract one wide src elem, unpack pack_ratio narrow
        # elems via shift + truncate, store each at the right offset.
        wide_id = ctx.text.alloc_id(f"vs_wide_{i}")
        ctx.text.emit_function(
            f"{wide_id} = OpCompositeExtract {src_elem_type} {value_id} {i}"
        )
        dst_bits = _DTYPE_BYTES[tensor.dtype] * 8
        # Unsigned narrow-int carrier for the truncate step (we go
        # tensor.dtype → narrow-uint → bitcast tensor.dtype to keep
        # bf16/f16 etc happy).
        narrow_uint_t = ctx.text.type_int(dst_bits, signed=False)
        for k in range(pack_ratio):
            inc = ctx.text.const_uint(i * pack_ratio + k)
            idx_id = ctx.text.alloc_id(f"vs_idx_{i}_{k}")
            ctx.text.emit_function(
                f"{idx_id} = OpIAdd {u32} {base_id} {inc}"
            )
            chain_id = ctx.text.alloc_id(f"vs_chain_{i}_{k}")
            ctx.text.emit_function(
                f"{chain_id} = {chain_op} {elem_ptr} {chain_base} {idx_id}"
            )
            shifted = wide_id
            if k > 0:
                shift_amt = ctx.text.const_uint(k * dst_bits)
                sh_id = ctx.text.alloc_id(f"vs_shift_{i}_{k}")
                ctx.text.emit_function(
                    f"{sh_id} = OpShiftRightLogical {src_elem_type} "
                    f"{wide_id} {shift_amt}"
                )
                shifted = sh_id
            trunc = ctx.text.alloc_id(f"vs_trunc_{i}_{k}")
            ctx.text.emit_function(
                f"{trunc} = OpUConvert {narrow_uint_t} {shifted}"
            )
            store_val = trunc
            if elem_type != narrow_uint_t:
                cast = ctx.text.alloc_id(f"vs_cast_{i}_{k}")
                ctx.text.emit_function(
                    f"{cast} = OpBitcast {elem_type} {trunc}"
                )
                store_val = cast
            ctx.text.emit_function(f"OpStore {chain_id} {store_val}")

    if pred_val is not None:
        pred_id = ctx.val_to_id[pred_val.id]
        merge_label = ctx.text.alloc_id("vs_merge")
        then_label = ctx.text.alloc_id("vs_then")
        ctx.text.emit_function(f"OpSelectionMerge {merge_label} None")
        ctx.text.emit_function(
            f"OpBranchConditional {pred_id} {then_label} {merge_label}"
        )
        ctx.text.emit_function(f"{then_label} = OpLabel")
        for i in range(width):
            _emit_one_store(i)
        ctx.text.emit_function(f"OpBranch {merge_label}")
        ctx.text.emit_function(f"{merge_label} = OpLabel")
    else:
        for i in range(width):
            _emit_one_store(i)


def _visit_frag_for_each(op: FragForEachOp, ctx: _OclCtx) -> None:
    """``FragForEachOp`` — iterate body over each storage slot of a
    fragment.

    Intel's lane layout for the C/D fragment (m8n16k16 bf16 SG=16):
      * Lane L holds column n=L; components s0..s7 = rows m=0..7.

    This makes the per-slot iteration trivial vs Vulkan:
      * No smem round-trip needed (Intel fragments are plain per-lane
        vectors, not opaque coopmat values).
      * Slot s directly corresponds to row m=s (compile-time constant).
      * Column n is the lane id (runtime SubgroupLocalInvocationId).

    So for each slot s ∈ [0, c_width):
      1. Extract the element via OpCompositeExtract from the input
         fragment vector at index s.
      2. Bind body_row_var = OpConstant(s), body_col_var = lane_id.
      3. Walk the body ops with these bindings.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    shape_id = op.attrs["shape_id"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_for_each(ocl): unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    layout = _intel_mma_layout(shape)
    (_sg, _a_dt, _a_w, _b_dt, _b_w, c_elem_dt, c_width,
     _k_dim, _operands_flag) = layout

    elem_type = _emit_dtype(ctx.text, c_elem_dt, ctx)
    frag_id = ctx.val_to_id[op.in_frag.id]
    lane_id = _ensure_lane_id(ctx)

    n_selectors = max(0, len(op.operands) - 1)

    for s in range(c_width):
        # Extract the per-slot element from the input fragment.
        elem_id = ctx.text.alloc_id(f"fe_elem_{s}")
        ctx.text.emit_function(
            f"{elem_id} = OpCompositeExtract {elem_type} {frag_id} {s}"
        )
        # body_row_var = constant s. body_col_var = lane_id.
        row_const = ctx.text.const_uint(s)

        ctx.val_to_id[op.body_input_var.id] = elem_id
        ctx.val_to_id[op.body_row_var.id] = row_const
        ctx.val_to_id[op.body_col_var.id] = lane_id

        # Optional selector binding — mirrors the SPV path.
        if op.body_selector_var is not None and n_selectors > 0:
            slot_to_sel_attr = op.attrs.get("slot_to_selector_idx")
            if slot_to_sel_attr is not None:
                sel_idx = slot_to_sel_attr[s] if s < len(slot_to_sel_attr) else 0
                sel_v = op.operands[1 + sel_idx]
                ctx.val_to_id[op.body_selector_var.id] = ctx.val_to_id[sel_v.id]
            else:
                # Dynamic-row dispatch: chain OpSelects against
                # row==i comparisons.
                bool_t = ctx.text.type_bool()
                sel_chain = ctx.val_to_id[op.operands[1 + n_selectors - 1].id]
                for i in range(n_selectors - 2, -1, -1):
                    cmp_id = ctx.text.alloc_id(f"fe_sel_cmp_{s}_{i}")
                    i_const = ctx.text.const_uint(i)
                    ctx.text.emit_function(
                        f"{cmp_id} = OpIEqual {bool_t} {row_const} {i_const}"
                    )
                    sel_lhs = ctx.val_to_id[op.operands[1 + i].id]
                    sel_dt = _emit_dtype(ctx.text, op.operands[1 + i].dtype, ctx)
                    sel_new = ctx.text.alloc_id(f"fe_sel_{s}_{i}")
                    ctx.text.emit_function(
                        f"{sel_new} = OpSelect {sel_dt} {cmp_id} "
                        f"{sel_lhs} {sel_chain}"
                    )
                    sel_chain = sel_new
                ctx.val_to_id[op.body_selector_var.id] = sel_chain

        # Walk the body. Suppress any surrounding loop_yield_stack so
        # the body's terminating void YieldOp isn't read as a carry yield.
        saved_stack = ctx.loop_yield_stack
        ctx.loop_yield_stack = []
        try:
            for body_op in op.body.ops:
                _walk_op(body_op, ctx)
        finally:
            ctx.loop_yield_stack = saved_stack


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


def _visit_frag_reduce(op: FragReduceOp, ctx: _OclCtx) -> None:
    """``FragReduceOp`` — reduce a fragment along one axis.

    Intel SG=16 m8n16k16 acc layout: lane L holds column n=L, slot
    s holds row m=s. So:

    * **axis="row"** (reduce across columns for each row m): for
      each row m ∈ [0, M), extract slot s=m from every lane (each
      lane's slot s=m is M[m, n=lane_id]) and cross-lane reduce
      across all SG lanes. One scalar per row, broadcast to all
      lanes via `OpGroupNonUniform<kind>` with `Reduce`
      group-operation.

    * **axis="col"** (reduce across rows for each column): per-lane
      fold across slots gives the column reduce in that lane's
      register, but the result IR Value must be readable on all
      lanes — requires N×OpGroupNonUniformBroadcast. Not yet wired
      (only attention's row-reduce path is in the kernel cohort).
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    shape_id = op.attrs["shape_id"]
    axis = op.attrs["axis"]
    kind = op.attrs["kind"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_reduce(ocl): unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    layout = _intel_mma_layout(shape)
    (_sg, _a_dt, _a_w, _b_dt, _b_w, c_elem_dt, c_width,
     _k_dim, _operands_flag) = layout

    elem_type = _emit_dtype(ctx.text, c_elem_dt, ctx)
    in_id = ctx.val_to_id[op.operands[0].id]
    n_results = len(op.results)

    if axis != "row":
        raise NotImplementedError(
            f"_visit_frag_reduce(ocl): axis={axis!r} not yet wired — only "
            "axis='row' (reduce across N) is implemented. Reducing across "
            "M (axis='col') needs N×OpGroupNonUniformBroadcast from each "
            "lane that owns its column; can be added when a kernel needs it."
        )
    if n_results != c_width:
        raise NotImplementedError(
            f"_visit_frag_reduce(ocl): expected n_results={c_width} (one "
            f"per row m), got {n_results}. Sub-row partitioning isn't "
            "wired."
        )

    sub_op = _FRAG_REDUCE_TO_SUBGROUP_OP.get((kind, _dtype_kind(c_elem_dt)))
    if sub_op is None:
        raise NotImplementedError(
            f"_visit_frag_reduce(ocl): kind={kind!r} dtype={c_elem_dt!r} "
            "not wired"
        )
    ctx.text.add_capability("GroupNonUniformArithmetic")
    sg_scope = ctx.text.const_uint(3)

    for s in range(c_width):
        # Extract row m=s from this lane's column slot.
        elem_id = ctx.text.alloc_id(f"fr_elem_{s}")
        ctx.text.emit_function(
            f"{elem_id} = OpCompositeExtract {elem_type} {in_id} {s}"
        )
        # Cross-lane reduce: every lane sees the same scalar = row-m
        # reduce across all columns.
        res_id = ctx.text.alloc_id(f"fr_{kind}_{s}")
        ctx.val_to_id[op.results[s].id] = res_id
        ctx.text.emit_function(
            f"{res_id} = {sub_op} {elem_type} {sg_scope} Reduce {elem_id}"
        )


def _visit_frag_convert(op: FragConvertOp, ctx: _OclCtx) -> None:
    """``FragConvertOp`` — convert N source ACC frags to one A frag
    with dtype change.

    For Intel ``m8n16k16_intel_bf16_*`` with ``num_src_frags=1`` (the
    online-softmax P-fragment path on Waypoint DiT), nk_per_kstep =
    shape_k / shape_n = 16/16 = 1 — no K-widening. Per-lane mapping:

      * Source ACC: ``v8 f32`` — lane L holds column n=L, rows m=0..7.
      * Dest A frag: ``v8 u16`` (bf16 carrier) — lane L holds column
        k=L, rows m=0..7. Identical layout (n_of_P == k_of_A; m == m).

    So we extract each of 8 ACC elements, optionally run the body
    (e.g. ``exp2((x - m_rc) * log2e)``), convert f32→bf16 via
    ``OpFConvert``, then ``OpCompositeConstruct`` the v8u16 vector.

    The IR registers the result Value as ``B32 width=a_regs`` (CUDA
    carrier convention), but the MMA visitor passes the SPIR-V ID
    verbatim — so what matters is the *actual emitted type* of the
    SSA the MMA consumes. We emit v<a_w>u16 (a_w = layout's per-lane
    A width), bound to ``ctx.val_to_id[out.id]``; the MMA reads it as
    the IGC-required ``v8u16`` regardless of the IR type tag.

    ``num_src_frags > 1`` (K-widening) is still unsupported on Intel
    — Intel m8n16k16 has K fixed at 16. That case needs an IR-level
    rewrite that splits FragConvert + downstream MMA into N per-K
    MMA calls. Not used by Waypoint DiT (nk_per_kstep=1 on this shape).
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    (out,) = op.results
    shape_id = op.attrs["shape_id"]
    src_dtype = op.attrs["src_dtype"]
    dst_dtype = op.attrs["dst_dtype"]
    num_src = int(op.attrs["num_src_frags"])

    if num_src != 1:
        raise NotImplementedError(
            "_visit_frag_convert(ocl): num_src_frags > 1 (K-widening) "
            f"not supported — Intel SG=16 m8n16k16 fixes K at 16. Got "
            f"num_src_frags={num_src}. Needs an IR-level rewrite that "
            "splits FragConvert + downstream MMA into N per-K-tile MMA "
            "calls."
        )
    if (src_dtype, dst_dtype) != (DType.F32, DType.BF16):
        raise NotImplementedError(
            f"_visit_frag_convert(ocl): only F32→BF16 supported "
            f"(got {src_dtype}→{dst_dtype})"
        )

    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_convert(ocl): unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    layout = _intel_mma_layout(shape)
    (_sg, a_elem_dt, a_w, _b_dt, _b_w, c_elem_dt, c_width,
     _k_dim, _operands_flag) = layout

    if c_width != a_w:
        raise NotImplementedError(
            f"_visit_frag_convert(ocl): src ACC width {c_width} != dst "
            f"A width {a_w}; only the same-width single-K-tile case is "
            "wired."
        )

    src_elem_type = _emit_dtype(ctx.text, c_elem_dt, ctx)
    dst_elem_type = _emit_dtype(ctx.text, a_elem_dt, ctx)
    out_vec_type = _ocl_type_vec(ctx.text, dst_elem_type, a_w)

    src_frag_id = ctx.val_to_id[op.operands[0].id]
    lane_id = _ensure_lane_id(ctx)

    slot_to_sel_attr = op.attrs.get("slot_to_selector_idx")
    n_selectors = max(0, len(op.operands) - num_src)
    dynamic_dispatch = (
        op.body_selector_var is not None
        and slot_to_sel_attr is None
        and n_selectors > 0
    )

    # bf16 in SPIR-V Kernel land = u16 carrier. OpFConvert from f32 to
    # bf16 isn't legal (bf16 isn't a SPIR-V Kernel-mode dtype); we have
    # to bit-pattern down via the same OpUConvert + OpShiftRight pattern
    # the f32→bf16 _visit_convert uses. _emit_f32_to_bf16 below mirrors
    # the b16 lane the OCL convert visitor produces.
    yielded_ids: list[str] = []
    for s in range(a_w):
        elem_id = ctx.text.alloc_id(f"fc_in_{s}")
        ctx.text.emit_function(
            f"{elem_id} = OpCompositeExtract {src_elem_type} {src_frag_id} {s}"
        )

        # Optional body: produces a transformed f32 from the per-slot f32.
        if op.body is not None and len(op.body.ops) > 0:
            row_const = ctx.text.const_uint(s)
            ctx.val_to_id[op.body_input_var.id] = elem_id

            if op.body_selector_var is not None and n_selectors > 0:
                if dynamic_dispatch:
                    bool_t = ctx.text.type_bool()
                    sel_chain = ctx.val_to_id[op.operands[num_src + n_selectors - 1].id]
                    for i in range(n_selectors - 2, -1, -1):
                        cmp_id = ctx.text.alloc_id(f"fc_cmp_{s}_{i}")
                        i_const = ctx.text.const_uint(i)
                        ctx.text.emit_function(
                            f"{cmp_id} = OpIEqual {bool_t} {row_const} {i_const}"
                        )
                        sel_lhs = ctx.val_to_id[op.operands[num_src + i].id]
                        sel_dt = _emit_dtype(
                            ctx.text, op.operands[num_src + i].dtype, ctx
                        )
                        sel_new = ctx.text.alloc_id(f"fc_sel_{s}_{i}")
                        ctx.text.emit_function(
                            f"{sel_new} = OpSelect {sel_dt} {cmp_id} "
                            f"{sel_lhs} {sel_chain}"
                        )
                        sel_chain = sel_new
                    ctx.val_to_id[op.body_selector_var.id] = sel_chain
                else:
                    slot_to_sel = slot_to_sel_attr or ()
                    sel_idx = slot_to_sel[s] if s < len(slot_to_sel) else 0
                    sel_v = op.operands[num_src + sel_idx]
                    ctx.val_to_id[op.body_selector_var.id] = ctx.val_to_id[sel_v.id]

            saved_stack = ctx.loop_yield_stack
            ctx.loop_yield_stack = []
            try:
                for body_op in op.body.ops:
                    if isinstance(body_op, YieldOp):
                        continue
                    _walk_op(body_op, ctx)
            finally:
                ctx.loop_yield_stack = saved_stack

            term = op.body.terminator
            if term is None or not term.operands:
                raise RuntimeError(
                    "_visit_frag_convert(ocl): body must yield 1 f32 value"
                )
            elem_id = ctx.val_to_id[term.operands[0].id]

        # Convert f32 → bf16 → u16. The OpFConvert + OpBitcast pair
        # mirrors the load_matrix A path (which loads as bfloat16_2
        # then bitcasts to u16). IGC's DPAS pattern matcher segfaults
        # if A is built from any other op chain (bitcast-and-shift
        # produces semantically-equivalent u16 but trips the matcher).
        bf16_type = _emit_dtype(ctx.text, DType.BF16, ctx)
        bf16_id = ctx.text.alloc_id(f"fc_bf_{s}")
        ctx.text.emit_function(
            f"{bf16_id} = OpFConvert {bf16_type} {elem_id}"
        )
        u16_id = ctx.text.alloc_id(f"fc_u16_{s}")
        ctx.text.emit_function(
            f"{u16_id} = OpBitcast {dst_elem_type} {bf16_id}"
        )
        yielded_ids.append(u16_id)

    _ = lane_id

    res_id = ctx.text.alloc_id("frag_convert_result")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {out_vec_type} {' '.join(yielded_ids)}"
    )


def _visit_frag_apply(op: FragApplyOp, ctx: _OclCtx) -> None:
    """``FragApplyOp`` — produce a new fragment by applying body to
    each storage slot. Intel-simplified: no smem round-trip, no
    output scratch — just OpCompositeExtract per slot, walk body,
    capture yielded SSA, then OpCompositeConstruct the result vector.

    Result shape must match input shape (the IR validator enforces
    this; FragConvertOp is the dtype-changing variant). For Intel
    SG=16 m8n16k16 acc: input/output are both v8f32; per slot we
    emit 1 OpCompositeExtract + body + 1 capture; finally 1
    OpCompositeConstruct over 8 captured values.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    (out,) = op.results
    shape_id = op.attrs["shape_id"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_frag_apply(ocl): unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    layout = _intel_mma_layout(shape)
    (_sg, _a_dt, _a_w, _b_dt, _b_w, c_elem_dt, c_width,
     _k_dim, _operands_flag) = layout

    elem_type = _emit_dtype(ctx.text, c_elem_dt, ctx)
    out_vec_type = _ocl_type_vec(ctx.text, elem_type, c_width)
    in_frag_id = ctx.val_to_id[op.operands[0].id]
    lane_id = _ensure_lane_id(ctx)

    slot_to_sel_attr = op.attrs.get("slot_to_selector_idx")
    n_selectors = max(0, len(op.operands) - 1)
    dynamic_dispatch = (
        op.body_selector_var is not None
        and slot_to_sel_attr is None
        and n_selectors > 0
    )

    yielded_ids: list[str] = []
    for s in range(c_width):
        elem_id = ctx.text.alloc_id(f"fa_in_{s}")
        ctx.text.emit_function(
            f"{elem_id} = OpCompositeExtract {elem_type} {in_frag_id} {s}"
        )
        row_const = ctx.text.const_uint(s)
        ctx.val_to_id[op.body_input_var.id] = elem_id

        if op.body_selector_var is not None and n_selectors > 0:
            if dynamic_dispatch:
                bool_t = ctx.text.type_bool()
                sel_chain = ctx.val_to_id[op.operands[1 + n_selectors - 1].id]
                for i in range(n_selectors - 2, -1, -1):
                    cmp_id = ctx.text.alloc_id(f"fa_cmp_{s}_{i}")
                    i_const = ctx.text.const_uint(i)
                    ctx.text.emit_function(
                        f"{cmp_id} = OpIEqual {bool_t} {row_const} {i_const}"
                    )
                    sel_lhs = ctx.val_to_id[op.operands[1 + i].id]
                    sel_dt = _emit_dtype(ctx.text, op.operands[1 + i].dtype, ctx)
                    sel_new = ctx.text.alloc_id(f"fa_sel_{s}_{i}")
                    ctx.text.emit_function(
                        f"{sel_new} = OpSelect {sel_dt} {cmp_id} "
                        f"{sel_lhs} {sel_chain}"
                    )
                    sel_chain = sel_new
                ctx.val_to_id[op.body_selector_var.id] = sel_chain
            else:
                slot_to_sel = slot_to_sel_attr or ()
                sel_idx = slot_to_sel[s] if s < len(slot_to_sel) else 0
                sel_v = op.operands[1 + sel_idx]
                ctx.val_to_id[op.body_selector_var.id] = ctx.val_to_id[sel_v.id]

        # Walk body, skip terminating YieldOp (captured below).
        saved_stack = ctx.loop_yield_stack
        ctx.loop_yield_stack = []
        try:
            for body_op in op.body.ops:
                if isinstance(body_op, YieldOp):
                    continue
                _walk_op(body_op, ctx)
        finally:
            ctx.loop_yield_stack = saved_stack

        term = op.body.terminator
        if term is None or not term.operands:
            raise RuntimeError(
                "_visit_frag_apply(ocl): body must yield exactly 1 value"
            )
        yielded_ids.append(ctx.val_to_id[term.operands[0].id])
        # Silence "unused" lint on lane_id when no selector path uses it.
        _ = lane_id

    res_id = ctx.text.alloc_id("frag_apply_result")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpCompositeConstruct {out_vec_type} {' '.join(yielded_ids)}"
    )


def _visit_mma(op: MmaOp, ctx: _OclCtx) -> None:
    """``MmaOp`` → ``OpSubgroupMatrixMultiplyAccumulateINTEL``.

    ``D = A * B + C`` where A/B/C/D are the per-lane vectors emitted
    by ``_visit_load_matrix`` (and the result feeds back into a
    ``_visit_store_matrix`` for the next consumer).

    The per-lane vector types + the K-Dim constant + the
    MatrixOperands flag are all keyed on the (shape, subgroup_width)
    tuple via ``_INTEL_MMA_LAYOUTS``. The capability + extension are
    emitted on first MmaOp visit by ``_ensure_intel_mma_caps``.

    Operand-flag is REQUIRED by IGC, not optional — see
    ``test_intel_mma_spv_compiles_through_igc`` for the empirical
    probe that locked this in.
    """
    from quark.ir.mma_registry import _BY_SHAPE_ID  # type: ignore

    _ensure_intel_mma_caps(ctx)
    (out,) = op.results
    shape_id = op.attrs["shape_id"]
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        raise NotImplementedError(
            f"_visit_mma(ocl): unknown shape_id={shape_id!r}"
        )
    shape = cfg.shape
    layout = _intel_mma_layout(shape)
    (
        sg, _a_elem_dt, _a_width, _b_elem_dt, _b_width,
        c_elem_dt, c_width, k_dim, operands_flag,
    ) = layout
    # The lowerer's ``subgroup_width`` must match the layout's
    # canonical SG — otherwise IGC won't use the right DPAS shape.
    if ctx.subgroup_width != sg:
        raise RuntimeError(
            f"_visit_mma(ocl): kernel subgroup_width={ctx.subgroup_width} "
            f"but shape {shape.name!r} requires SG={sg}. The lowerer's "
            f"subgroup_width parameter must match the MMA layout's "
            f"required SG; this is set per-kernel via "
            f"``OpenClSpirVLowerer(subgroup_width=...)`` and pinned at "
            f"compile time via OpExecutionMode SubgroupSize."
        )

    c_elem_type = _emit_dtype(ctx.text, c_elem_dt, ctx)
    c_vec_type = _ocl_type_vec(ctx.text, c_elem_type, c_width)
    k_dim_const = ctx.text.const_uint(k_dim)

    # Operand layout: (a_frag, b_frag, c_frag).
    a_id = ctx.val_to_id[op.operands[0].id]
    b_id = ctx.val_to_id[op.operands[1].id]
    c_v = op.operands[2]
    c_id = ctx.val_to_id[c_v.id]

    # If C is a vec_build of zeros (the common "init acc to 0"
    # idiom for non-loop-carried MMAs), splat-construct it as the
    # accumulator vector type. Mirror of the SPV path's same handling.
    c_producer = c_v.producer
    if isinstance(c_producer, VecBuildOp):
        elem_dtype = c_v.dtype
        scalar_t = _emit_dtype(ctx.text, elem_dtype, ctx)
        scalar_id = ctx.text.alloc_id("mma_c_splat")
        ctx.text.emit_function(
            f"{scalar_id} = OpCompositeExtract {scalar_t} {c_id} 0"
        )
        # OpCompositeConstruct needs N component operands (one per
        # vector lane), not a single scalar — IGC interprets a single
        # operand as a v1 vector type and rejects the MMA with
        # "Matrix C type: <1 x i32>". Repeat the scalar c_width times
        # to match the vector lane count.
        new_c = ctx.text.alloc_id("mma_c_init")
        repeated = " ".join([scalar_id] * c_width)
        ctx.text.emit_function(
            f"{new_c} = OpCompositeConstruct {c_vec_type} {repeated}"
        )
        c_id = new_c

    res_id = ctx.text.alloc_id("intel_mma")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpSubgroupMatrixMultiplyAccumulateINTEL {c_vec_type} "
        f"{k_dim_const} {a_id} {b_id} {c_id} {operands_flag}"
    )


def _visit_select(op: SelectOp, ctx: _OclCtx) -> None:
    """Ternary select via ``OpSelect``. Dialect-agnostic — the opcode
    + operand shape is identical to the Vulkan path."""
    (out,) = op.results
    pred_id = ctx.val_to_id[op.operands[0].id]
    t_id = ctx.val_to_id[op.operands[1].id]
    f_id = ctx.val_to_id[op.operands[2].id]
    dst_t = _emit_dtype(ctx.text, out.dtype, ctx)
    res_id = ctx.text.alloc_id("sel")
    ctx.val_to_id[out.id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpSelect {dst_t} {pred_id} {t_id} {f_id}"
    )


def _flatten_index(
    indices: tuple, shape: tuple, ctx: _OclCtx, *, stride: tuple | None = None,
) -> str:
    """Compute a row-major flat index from N-D indices + shape.

    Mirrors ``quark.lower.spv.lower._flatten_index`` byte-for-byte:
    same OpIAdd/OpIMul chain shape, same ``stride``-vs-``shape``
    preference (when ``stride`` is supplied, those per-axis
    multipliers are used directly — SharedRegion callers must pass
    ``tuple(tensor.stride)`` so the row-stride honors any
    bank-conflict pad).

    See project_spv_smem_offset_bug.md: forgetting to thread ``stride``
    through silently drops pad and silently miscompiles multi-warp
    coopmat GEMM. Same correctness rule applies on the OCL side."""
    if len(indices) == 1:
        return ctx.val_to_id[indices[0].id]
    u32 = ctx.text.type_int(32, signed=False)
    flat: str | None = None
    if stride is not None:
        rev_strides = list(stride)
    else:
        rev_strides = []
        s_acc = 1
        for s in reversed(shape):
            rev_strides.append(s_acc)
            s_acc *= s
        rev_strides.reverse()
    for idx_v, s in zip(indices, rev_strides, strict=False):
        idx_id = ctx.val_to_id[idx_v.id]
        if s == 1:
            term = idx_id
        else:
            stride_const = ctx.text.const_uint(s)
            term = ctx.text.alloc_id("flat_mul")
            ctx.text.emit_function(f"{term} = OpIMul {u32} {idx_id} {stride_const}")
        if flat is None:
            flat = term
        else:
            new_flat = ctx.text.alloc_id("flat_add")
            ctx.text.emit_function(f"{new_flat} = OpIAdd {u32} {flat} {term}")
            flat = new_flat
    assert flat is not None
    return flat


def _add_dyn_offset(flat: str, tensor, ctx: _OclCtx) -> str:
    """Append ``tensor.dyn_offset`` to a flat index when present.

    Used after ``_flatten_index`` for SharedRegion accesses. The
    scalar ``dyn_offset`` is what ``view(dyn_offset=…)`` /
    ``warp_lane_view`` set — each warp uses a private smem slice and
    the per-warp offset is what makes the slices disjoint. Drop = the
    NAX/SPV silent-miscompile pattern (project_nax_smem_offset_bug.md,
    project_spv_smem_offset_bug.md). Same rule on OCL."""
    scalar_dyn = getattr(tensor, "dyn_offset", None)
    if scalar_dyn is None:
        return flat
    u32 = ctx.text.type_int(32, signed=False)
    d_id = ctx.val_to_id[scalar_dyn.id]
    new_flat = ctx.text.alloc_id("flat_dyn")
    ctx.text.emit_function(f"{new_flat} = OpIAdd {u32} {flat} {d_id}")
    return new_flat


# Fallback byte-widths for dtypes the OCL emitter sees in smem but
# that aren't in the f32/u32/s32 trio. Used by ``_visit_smem_alloc``
# to track total Workgroup-class bytes for driver-side validation.
_SMEM_BYTES_FALLBACK = {
    DType.F16: 2, DType.BF16: 2, DType.U16: 2, DType.S16: 2,
    DType.U8: 1, DType.S8: 1, DType.F32: 4, DType.U32: 4,
    DType.S32: 4, DType.F64: 8, DType.U64: 8, DType.S64: 8,
}


def _visit_smem_alloc(op: SmemAllocOp, ctx: _OclCtx) -> None:
    """Allocate a ``Workgroup``-class array for the smem region.

    Layout is identical to the Vulkan path (``OpVariable Workgroup
    OpTypeArray T n``) — Workgroup storage class is dialect-agnostic.
    The only behavioural rule is the pad-handling for 2D regions:
    when ``op.attrs["pad"] > 0`` (bank-conflict avoidance), the
    backing-store size must be ``rows * (cols + pad)`` so the last
    row's last column doesn't walk past the buffer end. The SPV side
    learned this the hard way (project_spv_smem_offset_bug.md fixed
    2026-05-10); keeping the same rule here so the OCL path inherits
    the fix instead of rediscovering it.
    """
    (backing,) = op.results
    elem_type = _emit_dtype(ctx.text, op.dtype, ctx)
    pad = int(op.attrs.get("pad", 0)) if hasattr(op, "attrs") else 0
    if len(op.shape) == 2 and pad > 0:
        n_elems = op.shape[0] * (op.shape[1] + pad)
    else:
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

    elem_bytes = _SMEM_BYTES_FALLBACK.get(op.dtype, 0)
    if elem_bytes == 0:
        raise NotImplementedError(
            f"_visit_smem_alloc(ocl): dtype {op.dtype!r} has no known byte width"
        )
    ctx.smem_total_bytes += n_elems * elem_bytes


def _visit_barrier(op: BarrierOp, ctx: _OclCtx) -> None:
    """``OpControlBarrier execution memory semantics``.

    Scope mapping (identical to Vulkan side — these are environment-
    agnostic):
      * ``"block"``    → Workgroup (2)
      * ``"subgroup"`` → Subgroup (3)
      * ``"system"``   → Device (1)

    Memory semantics: ``AcquireRelease (0x8) | WorkgroupMemory (0x100)``
    for block-scope (smem synchronization). The OpenCL memory model
    accepts the same bitmask as Vulkan here — the AcquireRelease bit
    is required for any barrier that mixes loads and stores under the
    Physical64 OpenCL model just as it is under Logical GLSL450. For
    subgroup scope we widen to ``WorkgroupMemory`` instead of
    ``SubgroupMemory`` (0x80) because the subgroup-scope barriers are
    in practice protecting Workgroup-class scratch (per-warp tiles in
    a shared backing-store); the SubgroupMemory bit only matters if
    the smem is itself in the (rarely-used) Subgroup storage class,
    which quark doesn't emit.
    """
    scope = op.attrs.get("scope", "block")
    scope_const = {"block": 2, "subgroup": 3, "system": 1}.get(scope)
    if scope_const is None:
        raise NotImplementedError(f"_visit_barrier(ocl): scope={scope!r}")
    if scope == "block":
        mem_sem = 0x8 | 0x100  # AcquireRelease | WorkgroupMemory
    elif scope == "subgroup":
        mem_sem = 0x8 | 0x100  # see docstring — Workgroup-class smem
    else:
        mem_sem = 0x8

    exec_id = ctx.text.const_uint(scope_const)
    mem_id = ctx.text.const_uint(scope_const)
    sem_id = ctx.text.const_uint(mem_sem)
    ctx.text.emit_function(f"OpControlBarrier {exec_id} {mem_id} {sem_id}")


def _flatten_global_index(
    indices: tuple, tensor: "GlobalTensor", ctx: _OclCtx,
) -> str:
    """Compute a flat element index into a ``GlobalTensor``, honouring
    its ``view``/``tile`` offsets + per-axis strides — the OCL peer of
    ``quark.lower.spv.lower._flatten_global_index``.

    The address-arithmetic shape is identical to the Vulkan side
    (multi-axis ``(static + dyn + idx) * stride``) — the only delta
    is that for OCL the resulting flat index feeds
    ``OpInBoundsPtrAccessChain`` (one operand to advance the pointer)
    rather than ``OpAccessChain`` through a struct.
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
            continue
        stride_val = int(strides[axis])
        idx_id = ctx.val_to_id[idx_v.id]
        s_off = static_offsets[axis] if axis < len(static_offsets) else 0
        if s_off:
            const_id = ctx.text.const_uint(int(s_off))
            new_id = ctx.text.alloc_id(f"gidx_s{axis}")
            ctx.text.emit_function(f"{new_id} = OpIAdd {u32} {idx_id} {const_id}")
            idx_id = new_id
        d_off = dyn_offsets[axis] if axis < len(dyn_offsets) else None
        if d_off is not None:
            d_id = ctx.val_to_id[d_off.id]
            new_id = ctx.text.alloc_id(f"gidx_d{axis}")
            ctx.text.emit_function(f"{new_id} = OpIAdd {u32} {idx_id} {d_id}")
            idx_id = new_id
        if stride_val == 1:
            term = idx_id
        else:
            stride_const = ctx.text.const_uint(stride_val)
            term = ctx.text.alloc_id(f"gflat_mul{axis}")
            ctx.text.emit_function(f"{term} = OpIMul {u32} {idx_id} {stride_const}")
        if flat is None:
            flat = term
        else:
            new_flat = ctx.text.alloc_id(f"gflat_add{axis}")
            ctx.text.emit_function(f"{new_flat} = OpIAdd {u32} {flat} {term}")
            flat = new_flat
    assert flat is not None
    scalar_dyn = getattr(tensor, "dyn_offset", None)
    if scalar_dyn is not None:
        d_id = ctx.val_to_id[scalar_dyn.id]
        new_flat = ctx.text.alloc_id("gflat_dyn")
        ctx.text.emit_function(f"{new_flat} = OpIAdd {u32} {flat} {d_id}")
        flat = new_flat
    return flat


def _visit_load(op: LoadOp, ctx: _OclCtx) -> None:
    (out,) = op.results
    tensor = op.attrs["tensor"]
    if isinstance(tensor, GlobalTensor):
        arg_id = ctx.param_to_arg[id(tensor.param)]
        elem_ptr = ctx.param_to_elem_ptr[id(tensor.param)]
        elem_type = ctx.param_to_elem_type[id(tensor.param)]
        idx_id = _flatten_global_index(tuple(op.operands), tensor, ctx)
        # ``OpInBoundsPtrAccessChain`` advances a ``CrossWorkgroup``
        # pointer by a runtime offset — the OCL replacement for the
        # Vulkan path's nested ``OpAccessChain %ptr %buf %u32_0 %idx``
        # (the extra %u32_0 only exists because the Vulkan side wraps
        # the runtime-array in a Block-decorated struct).
        chain_id = ctx.text.alloc_id("chain")
        ctx.text.emit_function(
            f"{chain_id} = OpInBoundsPtrAccessChain {elem_ptr} {arg_id} {idx_id}"
        )
        res_id = ctx.text.alloc_id("ld")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = OpLoad {elem_type} {chain_id}")
        return

    if isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_load(ocl): SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp was visited"
            )
        var_id, elem_type, elem_ptr, _n = rec
        idx_id = _flatten_index(
            tuple(op.operands), tensor.shape, ctx, stride=tuple(tensor.stride)
        )
        idx_id = _add_dyn_offset(idx_id, tensor, ctx)
        # ``OpAccessChain`` for the Workgroup-class array — same shape
        # as the Vulkan smem path, no extra struct-member index.
        # (``OpInBoundsPtrAccessChain`` would also work under
        # Physical64 OpenCL, but ``OpAccessChain`` matches the Vulkan
        # smem emit byte-for-byte and keeps the two lowerers in
        # lockstep — easier to diff-debug.)
        chain_id = ctx.text.alloc_id("smem_chain")
        ctx.text.emit_function(
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {idx_id}"
        )
        res_id = ctx.text.alloc_id("smem_ld")
        ctx.val_to_id[out.id] = res_id
        ctx.text.emit_function(f"{res_id} = OpLoad {elem_type} {chain_id}")
        return

    raise NotImplementedError(
        f"_visit_load(ocl): tensor type {type(tensor).__name__} not wired"
    )


def _visit_store(op: StoreOp, ctx: _OclCtx) -> None:
    tensor = op.attrs["tensor"]
    pred_val = op.attrs.get("pred")
    raw_ops = list(op.operands)
    if pred_val is not None:
        index_ops = tuple(raw_ops[1:-1])
    else:
        index_ops = tuple(raw_ops[1:])

    if isinstance(tensor, GlobalTensor):
        arg_id = ctx.param_to_arg[id(tensor.param)]
        elem_ptr = ctx.param_to_elem_ptr[id(tensor.param)]
        value_id = ctx.val_to_id[op.operands[0].id]
        idx_id = _flatten_global_index(index_ops, tensor, ctx)
        chain_id = ctx.text.alloc_id("chain")
        chain_line = (
            f"{chain_id} = OpInBoundsPtrAccessChain {elem_ptr} {arg_id} {idx_id}"
        )
    elif isinstance(tensor, SharedRegion):
        rec = ctx.smem_allocs.get(tensor.alloc.id)
        if rec is None:
            raise RuntimeError(
                f"_visit_store(ocl): SharedRegion {tensor.name!r} accessed "
                "before its SmemAllocOp was visited"
            )
        var_id, _elem_type, elem_ptr, _n = rec
        value_id = ctx.val_to_id[op.operands[0].id]
        idx_id = _flatten_index(
            index_ops, tensor.shape, ctx, stride=tuple(tensor.stride)
        )
        idx_id = _add_dyn_offset(idx_id, tensor, ctx)
        chain_id = ctx.text.alloc_id("smem_chain")
        chain_line = (
            f"{chain_id} = OpAccessChain {elem_ptr} {var_id} {idx_id}"
        )
    else:
        raise NotImplementedError(
            f"_visit_store(ocl): tensor type {type(tensor).__name__} not wired"
        )

    if pred_val is not None:
        # ``pred=`` must guard the side-effecting OpStore. Same rule as
        # the Vulkan lowerer (see project_spv_correctness_bug.md and
        # feedback_spv_pred_codegen.md — pred= was historically dropped
        # in the SPV lowerer and produced silent miscompiles).
        pred_id = ctx.val_to_id[pred_val.id]
        merge_label = ctx.text.alloc_id("store_merge")
        then_label = ctx.text.alloc_id("store_then")
        ctx.text.emit_function(f"OpSelectionMerge {merge_label} None")
        ctx.text.emit_function(
            f"OpBranchConditional {pred_id} {then_label} {merge_label}"
        )
        ctx.text.emit_function(f"{then_label} = OpLabel")
        ctx.text.emit_function(chain_line)
        ctx.text.emit_function(f"OpStore {chain_id} {value_id}")
        ctx.text.emit_function(f"OpBranch {merge_label}")
        ctx.text.emit_function(f"{merge_label} = OpLabel")
    else:
        ctx.text.emit_function(chain_line)
        ctx.text.emit_function(f"OpStore {chain_id} {value_id}")


def _visit_split_b32(op: SplitB32Op, ctx: _OclCtx) -> None:
    """``SplitB32Op`` — split one B32 into (lo_b16, hi_b16).

    Emits OpUConvert (truncate to u16 for low half) and
    OpShiftRightLogical + OpUConvert (extract high half).
    Both results are u16 carriers (DType.B16 = unsigned int 16).
    """
    src = ctx.val_to_id[op.operands[0].id]
    u32 = ctx.text.type_int(32, signed=False)
    u16 = ctx.text.type_int(16, signed=False)
    # Low half
    lo_id = ctx.text.alloc_id("split_lo")
    ctx.val_to_id[op.results[0].id] = lo_id
    ctx.text.emit_function(f"{lo_id} = OpUConvert {u16} {src}")
    # High half
    sixteen = ctx.text.const_uint(16)
    shifted = ctx.text.alloc_id("split_shift")
    ctx.text.emit_function(
        f"{shifted} = OpShiftRightLogical {u32} {src} {sixteen}"
    )
    hi_id = ctx.text.alloc_id("split_hi")
    ctx.val_to_id[op.results[1].id] = hi_id
    ctx.text.emit_function(f"{hi_id} = OpUConvert {u16} {shifted}")


def _visit_merge_b32(op: MergeB32Op, ctx: _OclCtx) -> None:
    """``MergeB32Op`` — pack (lo_b16, hi_b16) into one B32.

    Inverse of SplitB32Op. OpUConvert each u16 to u32, shift hi by
    16, OR. Mirror of the packed-K load pattern.
    """
    lo = ctx.val_to_id[op.operands[0].id]
    hi = ctx.val_to_id[op.operands[1].id]
    u32 = ctx.text.type_int(32, signed=False)
    sixteen = ctx.text.const_uint(16)
    lo_ext = ctx.text.alloc_id("merge_lo_ext")
    ctx.text.emit_function(f"{lo_ext} = OpUConvert {u32} {lo}")
    hi_ext = ctx.text.alloc_id("merge_hi_ext")
    ctx.text.emit_function(f"{hi_ext} = OpUConvert {u32} {hi}")
    hi_shifted = ctx.text.alloc_id("merge_hi_shifted")
    ctx.text.emit_function(
        f"{hi_shifted} = OpShiftLeftLogical {u32} {hi_ext} {sixteen}"
    )
    res_id = ctx.text.alloc_id("merge_b32")
    ctx.val_to_id[op.results[0].id] = res_id
    ctx.text.emit_function(
        f"{res_id} = OpBitwiseOr {u32} {lo_ext} {hi_shifted}"
    )


_DISPATCH: dict[type, Any] = {
    ConstOp: _visit_const,
    SplitB32Op: _visit_split_b32,
    MergeB32Op: _visit_merge_b32,
    ArithOp: _visit_arith,
    CmpOp: _visit_cmp,
    ThreadIdxOp: _visit_thread_idx,
    BlockIdxOp: _visit_block_idx,
    BlockDimOp: _visit_block_dim,
    LoadOp: _visit_load,
    StoreOp: _visit_store,
    IfRegionOp: _visit_if_region,
    YieldOp: _visit_yield,
    MathOp: _visit_math,
    SelectOp: _visit_select,
    SmemAllocOp: _visit_smem_alloc,
    BarrierOp: _visit_barrier,
    LaneIdOp: _visit_lane_id,
    MmaOp: _visit_mma,
    LoadMatrixOp: _visit_load_matrix,
    StoreMatrixOp: _visit_store_matrix,
    ConvertOp: _visit_convert,
    BitcastOp: _visit_bitcast,
    VecBuildOp: _visit_vec_build,
    VecExtractOp: _visit_vec_extract,
    VecLoadOp: _visit_vec_load,
    VecStoreOp: _visit_vec_store,
    ForLoopOp: _visit_for_loop,
    SubgroupReduceOp: _visit_subgroup_reduce,
    SubgroupIdOp: _visit_subgroup_id,
    ShuffleOp: _visit_shuffle,
    AtomicRmwOp: _visit_atomic_rmw,
    GroupIdOp: _visit_group_id,
    ThreadIdInGroupOp: _visit_thread_id_in_group,
    FragForEachOp: _visit_frag_for_each,
    FragApplyOp: _visit_frag_apply,
    FragReduceOp: _visit_frag_reduce,
    FragConvertOp: _visit_frag_convert,
}


def _walk_op(op: Any, ctx: _OclCtx) -> None:
    visitor = _DISPATCH.get(type(op))
    if visitor is None:
        raise NotImplementedError(
            f"OclSpirVLowerer: no visitor for {type(op).__name__}. "
            "Phase 3 first cut covers the vec_add IR surface only — "
            "extend ``_DISPATCH`` to widen coverage."
        )
    visitor(op, ctx)


# ── Lowerer entry point ─────────────────────────────────────────────


# Subgroup widths the Intel stack accepts via cl_intel_required_subgroup_
# size. PTL / Battlemage commonly run with SIMD16 (better VE
# utilisation) or SIMD32 (back-compat). Keep the default at 32 so
# the OCL backend matches the SPV path's behaviour exactly on Phase 3
# turnup.
_SUBGROUP_WIDTH_DEFAULT = 32


class OpenClSpirVLowerer:
    """quark IR → OpenCL-flavor SPIR-V text (consumed by IGC).

    The visitor coverage is intentionally narrow in this first cut —
    enough to lower a ``vec_add``-class kernel (3 GlobalTensor params,
    ``thread_idx`` for the index, scalar arith, scalar load/store).
    """

    def __init__(
        self,
        caps: Any = None,
        *,
        local_size: tuple[int, int, int] | None = None,
        subgroup_width: int = _SUBGROUP_WIDTH_DEFAULT,
    ) -> None:
        self.caps = caps
        self._local_size = local_size
        if subgroup_width not in (8, 16, 32):
            raise ValueError(
                f"OpenClSpirVLowerer: subgroup_width={subgroup_width} "
                "not supported — must be 8, 16, or 32"
            )
        self._subgroup_width = subgroup_width

    def lower_module(self, module: Module) -> LoweredOclSpirVKernel:
        if not module.functions:
            raise ValueError("lower_module(ocl): module has no functions")
        if len(module.functions) > 1:
            raise NotImplementedError(
                "lower_module(ocl): multi-function modules not yet supported"
            )
        fn = module.functions[0]
        return self._lower_function(fn)

    def _lower_function(self, fn: Function) -> LoweredOclSpirVKernel:
        from quark.ir.types import BufferType

        # Auto-derive subgroup_width from any MMA shape the kernel
        # uses — the Intel MMA extension requires SG ∈ {8, 16}, so a
        # kernel with MMA at any shape must run at the layout's SG.
        # Falls back to the lowerer's explicit subgroup_width when
        # no MMA is present (e.g. ElementwiseKernel can run at SG=32).
        derived_sg = self._derive_mma_subgroup_size(fn)
        subgroup_width = derived_sg if derived_sg is not None else self._subgroup_width
        ctx = _OclCtx(subgroup_width=subgroup_width)
        text = ctx.text

        # Kernels report the correct SG-aware block size via
        # ``Kernel.resolve_subgroup_size()`` and ``Kernel.block()``, so
        # we no longer need to rescale local_size here. (Previously
        # the launcher handed us ``local_size = n_warps *
        # DEFAULT_KERNEL_SG=32`` regardless of the actual SG; now it
        # arrives as ``n_warps * subgroup_width`` already, so a rescale
        # would double-shrink.) Leaving _local_size_override unset
        # passes the launcher's value straight through.

        # ── Module-scope header (OpenCL flavor) ────────────────────
        # Order in serialize(): capabilities → extensions → ext-imports →
        # memory model → entry points → exec modes → debug → decorations →
        # types → functions. SpvText handles this for us; we just need to
        # populate the right fields.
        text.add_capability("Addresses")
        text.add_capability("Linkage")
        text.add_capability("Kernel")
        text.add_capability("Int64")
        # OpenCL.std ext-inst set — IGC rejects GLSL.std.450 outright
        # (probe error: "Expects OpenCL.std. Actual is GLSL.std.450").
        text.import_ext_inst("OpenCL.std")
        # Physical64 OpenCL memory model — pointer types resolve to
        # 64-bit on the device, kernel args are typed pointers (not
        # Block-decorated structs).
        text.set_memory_model("Physical64 OpenCL")

        # ── Walk params, allocate SSA ids + emit pointer types ────
        # Each BufferType param becomes an OpFunctionParameter with a
        # CrossWorkgroup pointer type. We collect ids ahead of OpFunction
        # so the OpEntryPoint interface list can reference them.
        buffer_params: list[tuple[Any, str, str, str]] = []  # (param, arg_id, ptr_t, elem_t)
        for p in fn.params:
            if not isinstance(p.type, BufferType):
                raise NotImplementedError(
                    "OpenClSpirVLowerer: non-buffer params not yet wired "
                    f"(param {p.name!r} type={type(p.type).__name__})"
                )
            elem_t = _emit_dtype(text, p.type.dtype, ctx)
            ptr_t = text.type_pointer("CrossWorkgroup", elem_t)
            arg_id = text.alloc_id(f"arg_{p.name}")
            ctx.param_to_arg[id(p)] = arg_id
            ctx.param_to_elem_ptr[id(p)] = ptr_t
            ctx.param_to_elem_type[id(p)] = elem_t
            buffer_params.append((p, arg_id, ptr_t, elem_t))

        # ── Function shell ────────────────────────────────────────
        void_t = text.type_void()
        fn_type = text.type_function(void_t, *(ptr_t for _, _, ptr_t, _ in buffer_params))
        fn_id = text.alloc_id("main")
        text.emit_function(f"{fn_id} = OpFunction {void_t} None {fn_type}")
        # Function parameters appear *between* OpFunction and the first
        # OpLabel — that's where IGC expects them in OCL SPIR-V.
        for p, arg_id, ptr_t, _elem_t in buffer_params:
            text.emit_function(f"{arg_id} = OpFunctionParameter {ptr_t}")
        entry_label = text.alloc_id("entry")
        text.emit_function(f"{entry_label} = OpLabel")

        ctx.local_size = self._resolve_local_size(fn)

        for op in fn.body.ops:
            _walk_op(op, ctx)

        text.emit_function("OpReturn")
        text.emit_function("OpFunctionEnd")

        # ── Entry point ──────────────────────────────────────────
        # OCL SPIR-V convention (mirrors the hand-written vec_add probe
        # at tests/drivers/test_ocl_compile_launch.py): the entry-point
        # interface list names the kernel argument SSA ids. Input-class
        # builtin variables (GlobalInvocationId / WorkgroupId / etc.)
        # are *not* listed — IGC's frontend resolves them via the
        # decoration on the OpVariable itself.
        interface_ids = [arg_id for _, arg_id, _, _ in buffer_params]
        text.add_entry_point(fn_id, "main", "Kernel", interface_ids)
        # OpExecutionMode LocalSize is *optional* in OCL — the launcher
        # supplies the local_size at clEnqueueNDRangeKernel time. We
        # skip it so the kernel stays dispatch-flexible (e.g. running
        # the same compiled program at different grid shapes); the
        # driver pins the matching value when CompiledKernel.launch
        # fires.
        #
        # Always pin SubgroupSize to the kernel's resolved SG. MMA
        # kernels MUST pin (``cl_intel_subgroup_matrix_multiply_
        # accumulate`` requires SG ∈ {8, 16}; without the pin
        # Battlemage defaults to SG=32 and the MMA fills only 2/4
        # slots — see project_openvino_taehv.md). Non-MMA kernels
        # don't strictly *require* the pin, but pinning makes the
        # kernel's compile-time work-distribution arithmetic
        # (``n_threads = NumWarps * self._sgs``, cooperative thread
        # splits, etc.) agree with IGC's actual SG instead of relying
        # on IGC's heuristic happening to pick the same value the
        # kernel assumed. Single source of truth: ``ctx.subgroup_width``
        # comes from ``Kernel.resolve_subgroup_size()`` via the
        # decorator and the launcher's lowerer wiring.
        # The SubgroupSize execution mode is part of OpenCL 2.x core
        # and IGC accepts it directly (no extra capability needed
        # under Kernel execution model).
        text.add_execution_mode(
            f"OpExecutionMode {fn_id} SubgroupSize {ctx.subgroup_width}"
        )

        return LoweredOclSpirVKernel(
            source=text.serialize(),
            entry_name="main",
            n_buffers=len(buffer_params),
            smem_bytes=ctx.smem_total_bytes,
            local_size=ctx.local_size,
            subgroup_size=ctx.subgroup_width,
        )

    def _derive_mma_subgroup_size(self, fn: Function) -> int | None:
        """Walk the function body and find any MMA shape; return its
        required subgroup_width per ``_INTEL_MMA_LAYOUTS``. Returns
        ``None`` if no MMA op is present (caller falls back to the
        lowerer's default subgroup_width)."""
        from quark.ir.mma_registry import _BY_SHAPE_ID

        def _walk(ops):
            for op in ops:
                if isinstance(op, MmaOp):
                    shape_id = op.attrs.get("shape_id")
                    cfg = _BY_SHAPE_ID.get(shape_id) if shape_id else None
                    if cfg is not None:
                        layout = _INTEL_MMA_LAYOUTS.get((
                            cfg.shape.a_dtype, cfg.shape.b_dtype,
                            cfg.shape.acc_dtype,
                            cfg.shape.m, cfg.shape.n, cfg.shape.k,
                        ))
                        if layout is not None:
                            return layout[0]
                for region in getattr(op, "regions", ()):
                    found = _walk(region.ops)
                    if found is not None:
                        return found
            return None

        return _walk(fn.body.ops)

    def _resolve_local_size(self, fn: Function) -> tuple[int, int, int]:
        # SG-override rescaling (computed in ``_lower_function``)
        # takes precedence over the kernel's declared local_size.
        override = getattr(self, "_local_size_override", None)
        if override is not None:
            return override
        if self._local_size is not None:
            return self._local_size
        attr = getattr(fn, "attrs", None)
        if attr is not None:
            ls = getattr(attr, "local_size", None)
            if ls is not None:
                return tuple(ls)  # type: ignore[return-value]
        return (64, 1, 1)


__all__ = [
    "LoweredOclSpirVKernel",
    "OpenClSpirVLowerer",
]
