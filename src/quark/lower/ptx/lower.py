"""PTX lowerer for the quark IR.

Walks a `Module` and emits a full PTX text artifact — the `.version` /
`.target` / `.address_size` header, the `.visible .entry` kernel
wrapper, the `.reg` declaration block, and the instruction stream.

V1 scope (this file):
  - ConstOp / ArithOp / MathOp / CmpOp / SelectOp / ConvertOp / BitcastOp
  - ThreadIdxOp / BlockIdxOp / BlockDimOp / GridDimOp / LaneIdOp / SubgroupIdOp
  - BarrierOp
  - ForLoopOp / IfRegionOp / YieldOp with loop-carried register coalescing
  - SmemAllocOp → a single shared pool + per-alloc byte offsets
  - Scalar LoadOp / StoreOp on GlobalTensor / SharedRegion

Deferred to a follow-up:
  - VecLoadOp / VecStoreOp / async copy family
  - Vec build/extract, SplitB32Op / MergeB32Op
  - Matmul (LoadMatrix/StoreMatrix/MmaOp)
  - Shuffles / SubgroupReduceOp / SubgroupBroadcastOp
  - AtomicRmwOp / predicated memory ops

The output is canonical PTX — not byte-identical to the current
`Program` emitter. When we later want byte-identity (for PTX-only
regression testing against the real kernel suite), that's a knob we'll
tune on top of this structure, not a design change.

See QUARK_IR_PROPOSAL.md §10.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    BufferType,
    CmpOp,
    ConstOp,
    ConvertOp,
    DType,
    ForLoopOp,
    FragApplyOp,
    FragConvertOp,
    FragForEachOp,
    FragReduceOp,
    Function,
    GlobalTensor,
    GridDimOp,
    GroupIdOp,
    IfRegionOp,
    LaneIdOp,
    LoadMatrixOp,
    LoadOp,
    MathOp,
    MergeB32Op,
    MmaOp,
    Module,
    Op,
    PackedConvertOp,
    ScalarType,
    SelectOp,
    SharedRegion,
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
    UnpackedConvertOp,
    Value,
    VecBuildOp,
    VecExtractOp,
    VecLoadOp,
    VecStoreOp,
    YieldOp,
)

from .regs import RegAllocator, arith_suffix, reg_class

# ---------------------------------------------------------------------------
# Per-function lowering context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweredKernel:
    """A compiled kernel artifact: PTX text plus launch-time metadata.

    Returned from `PtxLowerer.lower_module`. The `ptx` string is a
    full `.visible .entry` module ready to pass to ptxas. The
    `smem_bytes` field tells the launcher how much dynamic shared
    memory to request at kernel launch — this is the `.extern .shared`
    pool size the kernel declares.

    `__str__` returns the PTX text for callers that just want a
    string, so `str(lowered)` is equivalent to `lowered.ptx`.
    """

    ptx: str
    smem_bytes: int
    kernel_name: str

    def __str__(self) -> str:
        return self.ptx


@dataclass
class _FnCtx:
    """State tracked while lowering one Function.

    Owns the register allocator, the instruction stream, the smem pool,
    the label counter, and an explicit `op_stack` that tracks the
    enclosing structured-op chain during the walk so that a YieldOp
    can find the results of its enclosing ForLoop / IfRegion without
    needing back-pointers on the Op base class.
    """

    regs: RegAllocator = field(default_factory=RegAllocator)
    instructions: list[str] = field(default_factory=list)
    # Shared-memory pool bookkeeping. Each SmemAllocOp reserves a slice
    # of this pool and gets a u32 base-address register. The total
    # pool size is reported via `smem_size_bytes`.
    smem_base_name: str = "_smem_pool"
    smem_size_bytes: int = 0
    smem_align: int = 16
    # Per-alloc record: Value.id → (reg_with_u32_base, byte_offset, size_bytes)
    smem_allocs: dict[int, tuple[str, int, int]] = field(default_factory=dict)
    # Layout plan from the smem_layout pass — populated by lower_function
    # before walking the IR. ``_visit_smem_alloc`` consumes it for offsets.
    smem_plan: object | None = None
    # Label counter (for loop/if branches)
    label_counter: int = 0
    # Param Value.id → (param_name, ptx_type)
    param_info: dict[int, tuple[str, str]] = field(default_factory=dict)
    # Stack of currently-open structured ops (for yield resolution).
    op_stack: list[Op] = field(default_factory=list)
    # Back-reference to the owning module (for MmaShape lookups).
    module: Module | None = None

    def fresh_label(self, stem: str) -> str:
        self.label_counter += 1
        return f"${stem}_{self.label_counter}"

    def emit(self, line: str) -> None:
        self.instructions.append("    " + line)

    def emit_label(self, label: str) -> None:
        self.instructions.append(f"{label}:")


# ---------------------------------------------------------------------------
# PTX lowerer
# ---------------------------------------------------------------------------


class PtxLowerer:
    # Aliasing of disjoint-lifetime smem regions. Off by default during
    # bring-up; flip on once kernels annotate explicit lifetimes and
    # cos-sim is verified across every kernel.
    _smem_aliasing_enabled: bool = False

    """IR → PTX text.

    Construction takes target-specific knobs (PTX version, SM number).
    `lower_module(module)` returns a `LoweredKernel` containing the
    PTX text wrapped in a `.visible .entry` kernel plus launch
    metadata.

    `ptx_version=None` (the default) auto-picks based on `target_sm`:
    sm_120/100 (Blackwell) needs PTX 8.7+; sm_90 (Hopper) PTX 7.8+;
    sm_89 (Ada) PTX 8.4+ for FP8; sm_80/86 (Ampere) PTX 7.0+. The
    chosen version is the lowest one that supports the target SM and
    the dtypes Bundle 4's universal GEMM will need.
    """

    def __init__(
        self,
        ptx_version: str | None = None,
        target_sm: int = 89,
    ) -> None:
        self.target_sm = target_sm
        self.ptx_version = ptx_version or self._default_ptx_version(target_sm)

    @staticmethod
    def _default_ptx_version(target_sm: int) -> str:
        if target_sm >= 100:
            return "8.7"  # Blackwell
        if target_sm >= 90:
            return "8.0"  # Hopper
        if target_sm >= 89:
            return "8.4"  # Ada (fp8)
        return "7.0"  # Ampere

    # ---- top-level ----

    def lower_module(self, module: Module) -> LoweredKernel:
        """Lower a Module's first function. Returns a LoweredKernel
        with the PTX text and the smem byte count the launcher needs
        to pass at kernel launch time.

        Runs ``validate_module`` first — raises ValidationError on
        correctness issues (OOB, missing shape_ids, SSA dominance
        violations) and emits ``PerfWarning`` for perf patterns (bank
        conflicts, oversized smem, misaligned cp.async). Perf warnings
        are off by default; enable via ``QUARK_ENABLE_PERF_WARNINGS=1``
        (and filter further with ``warnings.simplefilter(...)`` on
        ``PerfWarning``).
        """
        from quark.ir.validator import validate_module as _validate_module

        _validate_module(module)
        if not module.functions:
            raise ValueError("lower_module: module has no functions")
        if len(module.functions) > 1:
            raise NotImplementedError("lower_module: multi-function modules not yet supported")
        fn = module.functions[0]
        return self.lower_function(fn, module)

    def lower_function(self, fn: Function, module: Module | None = None) -> LoweredKernel:
        import os as _os

        from quark.lower.smem_layout import compute_smem_layout, dump_smem_layout

        ctx = _FnCtx(module=module)
        # Compute the smem layout plan up front so per-op alloc visitors
        # consume offsets/aliasing rather than running their own counter.
        # Aliasing off by default during bring-up (passthrough mode).
        ctx.smem_plan = compute_smem_layout(
            fn, enable_aliasing=getattr(self, "_smem_aliasing_enabled", False)
        )
        if _os.environ.get("QUARK_DUMP_SMEM_LAYOUT", "").lower() in ("1", "true", "yes", "on"):
            print(dump_smem_layout(fn, ctx.smem_plan, label="(ptx)"), flush=True)
        # Preload param register names so operands referencing them
        # resolve cleanly during walk.
        param_decls = self._emit_param_decls(fn, ctx)
        # Walk the function body.
        self._walk_region(fn.body.ops, ctx)
        # Smem total comes from the plan, not the per-op counter.
        smem_total = ctx.smem_plan.total_bytes if ctx.smem_plan is not None else ctx.smem_size_bytes
        ctx.smem_size_bytes = smem_total
        ptx = self._wrap_kernel(fn, ctx, param_decls)
        return LoweredKernel(
            ptx=ptx,
            smem_bytes=smem_total,
            kernel_name=fn.name,
        )

    # ---- kernel wrapper ----

    def _emit_param_decls(self, fn: Function, ctx: _FnCtx) -> list[str]:
        """Build the `.param .u64 X` declaration list and record each
        param's ptx_type so load-from-param can be emitted on use."""
        decls: list[str] = []
        for p in fn.params:
            if isinstance(p.type, BufferType):
                ptx_type = "u64"
            elif isinstance(p.type, ScalarType):
                ptx_type = reg_class(p.type.dtype)
            else:
                raise TypeError(f"PtxLowerer: unsupported Param type {p.type!r}")
            decls.append(f"    .param .{ptx_type} {p.name}")
            assert p.value is not None
            # Allocate a register for the param's SSA Value + emit a
            # one-shot ld.param at function entry.
            reg = ctx.regs.name_for(p.value)
            ctx.instructions.append(f"    ld.param.{ptx_type} {reg}, [{p.name}];")
            ctx.param_info[p.value.id] = (p.name, ptx_type)
        return decls

    def _wrap_kernel(self, fn: Function, ctx: _FnCtx, param_decls: list[str]) -> str:
        """Assemble the final PTX text for one kernel."""
        header = [
            f".version {self.ptx_version}",
            f".target sm_{self.target_sm}",
            ".address_size 64",
            "",
        ]

        # Shared-memory pool declaration (if any allocs were made).
        #
        # `.extern .shared` — dynamic shared memory. The `[ ]` is
        # deliberately unsized; the actual byte count is passed at
        # kernel-launch time by the runtime (see `LoweredKernel
        # .smem_bytes` for the value to pass). The pool offsets
        # computed by `_visit_smem_alloc` remain valid because they're
        # bounded by `ctx.smem_size_bytes`, which the launcher honors.
        module_decls: list[str] = []
        if ctx.smem_size_bytes > 0:
            module_decls.append(
                f".extern .shared .align {ctx.smem_align} .b8 {ctx.smem_base_name}[];"
            )
            module_decls.append("")

        # Kernel entry line + params.
        params_str = ",\n".join(param_decls) if param_decls else ""
        entry_header = [
            f".visible .entry {fn.name}(",
            params_str,
            ")",
            "{",
        ]

        body_lines = ctx.regs.declarations()
        body_lines.append("")
        body_lines.extend(ctx.instructions)
        body_lines.append("    ret;")
        body_lines.append("}")

        return "\n".join(header + module_decls + entry_header + body_lines) + "\n"

    # ------------------------------------------------------------------
    # Region walker
    # ------------------------------------------------------------------

    def _walk_region(self, ops: list[Op], ctx: _FnCtx) -> None:
        for op in ops:
            self._visit(op, ctx)

    def _visit(self, op: Op, ctx: _FnCtx) -> None:
        # Dispatch on op class. We use a dict-driven table so that new
        # ops only need one entry.
        handler = _DISPATCH.get(type(op))
        if handler is None:
            raise NotImplementedError(f"PtxLowerer: no handler for {type(op).__name__}")
        handler(self, op, ctx)

    # ------------------------------------------------------------------
    # §5.1 Arithmetic / math
    # ------------------------------------------------------------------

    def _visit_const(self, op: ConstOp, ctx: _FnCtx) -> None:
        (out,) = op.results
        reg = ctx.regs.name_for(out)
        dtype: DType = op.attrs["dtype"]
        value = op.attrs["value"]
        ctx.emit(f"mov.{reg_class(dtype)} {reg}, {_format_literal(dtype, value)};")

    def _visit_arith(self, op: ArithOp, ctx: _FnCtx) -> None:
        kind = op.attrs["kind"]
        (out,) = op.results
        dst = ctx.regs.name_for(out)
        dtype = out.dtype
        suffix = arith_suffix(dtype)

        if kind == "fma_bf16x2":
            ops_str = ", ".join(ctx.regs.name_for(v) for v in op.operands)
            ctx.emit(f"fma.rn.bf16x2 {dst}, {ops_str};")
            return
        if kind == "cvt_rn_bf16x2_f32":
            a, b = op.operands
            ctx.emit(f"cvt.rn.bf16x2.f32 {dst}, {ctx.regs.name_for(b)}, {ctx.regs.name_for(a)};")
            return
        # Integer multiply needs .lo. Int div/rem use the same syntax.
        if kind == "mul" and (dtype.is_int or dtype.is_bit):
            instr = f"mul.lo.{suffix}"
        elif kind == "mul_hi":
            # Unsigned high half of 32×32→64 (used by Philox-family RNG).
            # Emitted with the unsigned suffix regardless of the Value's
            # declared int signedness — mul.hi is well-defined either way
            # and we only need the unsigned variant for counter-based RNG.
            hi_suf = {2: "u16", 4: "u32", 8: "u64"}.get(dtype.bytes)
            if hi_suf is None:
                raise NotImplementedError(f"mul_hi: unsupported dtype {dtype}")
            instr = f"mul.hi.{hi_suf}"
        elif kind == "fma" and dtype.is_float:
            instr = f"fma.rn.{suffix}"
        elif kind in ("and", "or", "xor") and dtype is DType.PRED:
            # Predicate-typed bitwise ops lower to `and.pred` / `or.pred` /
            # `xor.pred` — the PTX path for combining PREDs (selp.pred
            # doesn't exist, so this is how `bld.and_(p1, p2)` reaches the
            # back end).
            instr = f"{kind}.pred"
        elif kind in ("and", "or", "xor", "shl", "shr") and not dtype.is_bit:
            # Bitwise ops run in the bit-typed class even for int regs.
            # PTX spells them as `and.b32` / `or.b32` / etc.
            suffix = {2: "b16", 4: "b32", 8: "b64"}[dtype.bytes]
            instr = f"{kind}.{suffix}"
        else:
            instr = f"{_ARITH_MNEMONIC[kind]}.{suffix}"

        ops_str = ", ".join(ctx.regs.name_for(v) for v in op.operands)
        ctx.emit(f"{instr} {dst}, {ops_str};")

    def _visit_math(self, op: MathOp, ctx: _FnCtx) -> None:
        kind = op.attrs["kind"]
        (out,) = op.results
        dst = ctx.regs.name_for(out)
        src = ctx.regs.name_for(op.operands[0])
        suffix = arith_suffix(out.dtype)
        instr = _MATH_MNEMONIC[kind]
        ctx.emit(f"{instr}.{suffix} {dst}, {src};")

    def _visit_cmp(self, op: CmpOp, ctx: _FnCtx) -> None:
        kind = op.attrs["kind"]
        (out,) = op.results
        dst = ctx.regs.name_for(out)
        a, b = op.operands
        suffix = arith_suffix(a.dtype)
        ctx.emit(f"setp.{kind}.{suffix} {dst}, {ctx.regs.name_for(a)}, {ctx.regs.name_for(b)};")

    def _visit_select(self, op: SelectOp, ctx: _FnCtx) -> None:
        (out,) = op.results
        if out.dtype is DType.PRED:
            # PTX has no `selp.pred`. Combining two predicates by predicate
            # selection ((p1, p2, FALSE) → AND; (p1, TRUE, p2) → OR) must use
            # `and.pred` / `or.pred` — i.e. `bld.and_(p1, p2)` / `bld.or_(...)`.
            raise NotImplementedError(
                "SelectOp: PTX has no `selp.pred`. Combine PRED Values with "
                "`builder.and_(p1, p2)` / `builder.or_(p1, p2)` (which lower "
                "to `and.pred` / `or.pred`)."
            )
        dst = ctx.regs.name_for(out)
        pred, t, f = op.operands
        suffix = reg_class(out.dtype)
        ctx.emit(
            f"selp.{suffix} {dst}, "
            f"{ctx.regs.name_for(t)}, {ctx.regs.name_for(f)}, "
            f"{ctx.regs.name_for(pred)};"
        )

    def _visit_convert(self, op: ConvertOp, ctx: _FnCtx) -> None:
        (out,) = op.results
        dst = ctx.regs.name_for(out)
        src_v = op.operands[0]
        src = ctx.regs.name_for(src_v)
        src_dtype: DType = op.attrs["src_dtype"]
        dst_dtype: DType = op.attrs["dst_dtype"]
        rounding: str = op.attrs.get("rounding", "rn")
        if dst_dtype in (DType.E4M3, DType.E5M2):
            # PTX has no scalar fp8 cvt — only the packed
            # `cvt.<rnd>.satfinite.<fp8>x2.<src>x2` form. Callers that
            # want fp8 dst must go through `PackedConvertOp` (see
            # `_visit_packed_convert`), which consumes two source scalars
            # at once and writes a width-2 fp8 Value. A scalar-wide
            # ConvertOp to fp8 would have to fake a pair and discard half
            # the result, so reject it here.
            raise NotImplementedError(
                f"scalar cvt → {dst_dtype.value} is not representable in PTX "
                f"(only packed `{dst_dtype.value}x2` exists). Use "
                f"`builder.packed_convert(lo, hi, {dst_dtype})` to convert "
                f"two source values at once."
            )
        # Rounding mode: required when converting to a narrower float OR
        # when converting from integer to float (the integer value may
        # not be exactly representable).
        round_prefix = ""
        if dst_dtype.is_float and (dst_dtype.bytes < src_dtype.bytes or src_dtype.is_int):
            round_prefix = f".{rounding}"
        ctx.emit(
            f"cvt{round_prefix}.{_cvt_suffix(dst_dtype)}.{_cvt_suffix(src_dtype)} {dst}, {src};"
        )

    def _visit_packed_convert(self, op: PackedConvertOp, ctx: _FnCtx) -> None:
        """Packed fp8 cvt — the only cvt form PTX exposes for fp8 dst.

        Emits one of:
          cvt.<rnd>.satfinite.<fp8>x2.f32     d_b16, a_f32, b_f32;
          cvt.<rnd>.satfinite.<fp8>x2.<fp>x2  d_b16, packed_b32;

        The output is a width-1 B16 Value whose two bytes are the
        packed fp8 pair. Callers write the result to contiguous
        fp8 storage via a single b16 store.
        """
        (out,) = op.results
        dst = ctx.regs.name_for(out)
        src_dtype: DType = op.attrs["src_dtype"]
        dst_dtype: DType = op.attrs["dst_dtype"]
        rounding: str = op.attrs.get("rounding", "rn")
        fp8_suffix = dst_dtype.value

        # Byte ordering (per PTX ISA v9.2 §9.7.9.21 "cvt"):
        #   For `.e4m3x2/.e5m2x2` destination with `.f32` source:
        #     d[15:8] = convert(a)   ← upper byte
        #     d[7: 0] = convert(b)   ← lower byte
        #   For `.e4m3x2/.e5m2x2` destination with `.f16x2/.bf16x2` source:
        #     d[15:8] = convert(src[31:16])   ← upper byte from upper half
        #     d[7: 0] = convert(src[15: 0])   ← lower byte from lower half
        # The subsequent `st.b16` writes the b16 little-endian, so the
        # lower byte lands at the +0 byte offset. Callers pass (lo, hi)
        # meaning "lo gets the +0 byte, hi gets the +1 byte", so we
        # route lo → lower byte of d and hi → upper byte of d.
        lo_v, hi_v = op.operands
        lo = ctx.regs.name_for(lo_v)
        hi = ctx.regs.name_for(hi_v)
        if src_dtype is DType.F32:
            # a → upper byte (hi), b → lower byte (lo).
            ctx.emit(f"cvt.{rounding}.satfinite.{fp8_suffix}x2.f32 {dst}, {hi}, {lo};")
        elif src_dtype is DType.F16:
            # Pack 2 f16 into a b32 with lo in the low 16 bits, hi in
            # the upper 16 bits, then the packed f16x2 → fp8x2 cvt maps
            # lo → lower byte, hi → upper byte directly.
            b32_tmp = ctx.regs.declare("b32")
            ctx.emit(f"mov.b32 {b32_tmp}, {{{lo}, {hi}}};")
            ctx.emit(f"cvt.{rounding}.satfinite.{fp8_suffix}x2.f16x2 {dst}, {b32_tmp};")
        elif src_dtype is DType.BF16:
            # PTX 8.8 / CUDA 12.9 does not yet expose `cvt.<fp8>x2.bf16x2`
            # (the `fp16x2`-source form is f16x2-only in pre-9.2 ISA).
            # Go through f32 — PTX's `cvt.f32.bf16` is a single instruction
            # that bit-shifts a bf16 into the upper half of an f32 with
            # the lower 16 bits zero, so this isn't materially more work
            # than a hypothetical direct form.
            lo_f32 = ctx.regs.declare("f32")
            hi_f32 = ctx.regs.declare("f32")
            ctx.emit(f"cvt.f32.bf16 {lo_f32}, {lo};")
            ctx.emit(f"cvt.f32.bf16 {hi_f32}, {hi};")
            # Same a=hi / b=lo routing as the f32 path.
            ctx.emit(f"cvt.{rounding}.satfinite.{fp8_suffix}x2.f32 {dst}, {hi_f32}, {lo_f32};")
        else:
            raise NotImplementedError(f"packed cvt {src_dtype} → {dst_dtype} not implemented")

    def _visit_unpacked_convert(self, op: UnpackedConvertOp, ctx: _FnCtx) -> None:
        """Inverse of `_visit_packed_convert`: split a packed fp8x2 (b16) into
        two wider scalars (bf16 / f16 / f32). Emits one of:

          cvt.rn.bf16x2.<src>x2  d_b32, packed_b16;   // PTX 9.2+
          cvt.rn.f16x2.<src>x2   d_b32, packed_b16;   // PTX 7.8+

        The packed b32 result is split into the result Value's two
        components via `mov.b32 {lo, hi}, b32_packed`. Lo/hi follow the
        same byte ordering as packed_convert: the lower 8 bits of the
        b16 source land in `lo`, upper 8 bits land in `hi`.
        """
        (out,) = op.results
        src_dtype: DType = op.attrs["src_dtype"]
        dst_dtype: DType = op.attrs["dst_dtype"]
        rounding: str = op.attrs.get("rounding", "rn")
        fp8_suffix = src_dtype.value
        packed = ctx.regs.name_for(op.operands[0])
        comps = ctx.regs.components(out)
        if len(comps) != 2:
            raise RuntimeError(f"unpacked_convert: expected 2 result components, got {len(comps)}")
        lo_name, hi_name = comps[0], comps[1]
        if dst_dtype is DType.BF16:
            # `cvt.rn.bf16x2.e4m3x2` is PTX 9.2 / sm_120f — too narrow
            # a target. Direct `cvt.bf16.f16` also requires sm_90+ (an
            # earlier comment here was wrong — ptxas on sm_89 rejects
            # it). Route through f32: fp8x2 → f16x2 → split → f32 →
            # bf16. All the intermediate casts are sm_80+.
            packed_f16x2 = ctx.regs.declare("b32")
            lo_h = ctx.regs.declare("b16")
            hi_h = ctx.regs.declare("b16")
            lo_f = ctx.regs.declare("f32")
            hi_f = ctx.regs.declare("f32")
            ctx.emit(f"cvt.{rounding}.f16x2.{fp8_suffix}x2 {packed_f16x2}, {packed};")
            ctx.emit(f"mov.b32 {{{lo_h}, {hi_h}}}, {packed_f16x2};")
            ctx.emit(f"cvt.f32.f16 {lo_f}, {lo_h};")
            ctx.emit(f"cvt.f32.f16 {hi_f}, {hi_h};")
            ctx.emit(f"cvt.{rounding}.bf16.f32 {lo_name}, {lo_f};")
            ctx.emit(f"cvt.{rounding}.bf16.f32 {hi_name}, {hi_f};")
        elif dst_dtype is DType.F16:
            packed_b32 = ctx.regs.declare("b32")
            ctx.emit(f"cvt.{rounding}.f16x2.{fp8_suffix}x2 {packed_b32}, {packed};")
            ctx.emit(f"mov.b32 {{{lo_name}, {hi_name}}}, {packed_b32};")
        elif dst_dtype is DType.F32:
            # No direct cvt.f32x2.<fp8>x2; route through f16x2 then f16→f32.
            packed_b32 = ctx.regs.declare("b32")
            lo_h = ctx.regs.declare("b16")
            hi_h = ctx.regs.declare("b16")
            ctx.emit(f"cvt.{rounding}.f16x2.{fp8_suffix}x2 {packed_b32}, {packed};")
            ctx.emit(f"mov.b32 {{{lo_h}, {hi_h}}}, {packed_b32};")
            ctx.emit(f"cvt.f32.f16 {lo_name}, {lo_h};")
            ctx.emit(f"cvt.f32.f16 {hi_name}, {hi_h};")
        else:
            raise NotImplementedError(
                f"unpacked_convert: dst {dst_dtype} not implemented (have BF16, F16, F32)"
            )

    def _visit_bitcast(self, op: BitcastOp, ctx: _FnCtx) -> None:
        (out,) = op.results
        dst = ctx.regs.name_for(out)
        src = ctx.regs.name_for(op.operands[0])
        # PTX has no dedicated bitcast — mov.bN between registers in
        # the same bit-width is a bitcast.
        nbytes = out.shape.bytes
        if nbytes == 2:
            bc = "b16"
        elif nbytes == 4:
            bc = "b32"
        else:
            bc = "b64"
        ctx.emit(f"mov.{bc} {dst}, {src};")

    # ------------------------------------------------------------------
    # §5.2 Vector / bit manipulation
    # ------------------------------------------------------------------

    def _visit_vec_build(self, op: VecBuildOp, ctx: _FnCtx) -> None:
        """Pack N scalars into a width-N vec Value.

        When the element dtype is sub-register-width (e.g. BF16 scalars
        packed into B32 registers), pairs of elements are merged via
        ``mov.b32 dst, {lo, hi}`` to produce the physical register set.
        Otherwise zero-emit: bind the result to the concat of input regs.
        """
        (out,) = op.results

        if op.attrs.get("packed_b32"):
            # Operands are pre-packed B32 values; bind them directly.
            # Bypass bind()'s width check: N B32 regs back 2N BF16 logical elements.
            b32_names = tuple(ctx.regs.name_for(v) for v in op.operands)
            ctx.regs._components[out.id] = b32_names
            return

        n_logical = len(op.operands)
        elem_dt = op.operands[0].dtype
        form = _vec_phys_form(n_logical, elem_dt)

        if form is None or form[0] == n_logical:
            # 1:1 mapping — no packing needed.
            names: list[str] = []
            for operand in op.operands:
                (single,) = ctx.regs.components(operand)
                names.append(single)
            ctx.regs.bind(out, tuple(names))
            return

        v_width, reg_dt = form
        pack_factor = n_logical // v_width

        if pack_factor == 2:
            # Merge pairs of B16 → B32.
            packed: list[str] = []
            for i in range(0, n_logical, 2):
                (lo_r,) = ctx.regs.components(op.operands[i])
                (hi_r,) = ctx.regs.components(op.operands[i + 1])
                dst = ctx.regs.declare(reg_class(reg_dt))
                ctx.emit(f"mov.b32 {dst}, {{{lo_r}, {hi_r}}};")
                packed.append(dst)
            ctx.regs._components[out.id] = tuple(packed)
        else:
            raise NotImplementedError(f"VecBuildOp: pack_factor={pack_factor} not yet supported")

    def _visit_vec_extract(self, op: VecExtractOp, ctx: _FnCtx) -> None:
        """Expose one component of a vec Value as a scalar.

        When the logical element dtype is sub-register-width (e.g. BF16
        elements packed into B32 registers from a v4.b32 load), this
        emits sub-register extraction: identify the physical register
        containing the element and split it via ``mov.b32 {lo, hi}, reg``.
        """
        (out,) = op.results
        src = op.operands[0]
        idx = op.attrs["index"]
        phys_comps = ctx.regs.components(src)
        n_phys = len(phys_comps)
        n_logical = src.width

        if n_phys == n_logical:
            # 1:1 mapping — no packing.
            ctx.regs.alias_component(out, src, idx)
            return

        # Packed: n_logical > n_phys. Each physical register holds
        # pack_factor logical elements.
        pack_factor = n_logical // n_phys
        phys_idx = idx // pack_factor
        sub_idx = idx % pack_factor
        phys_reg = phys_comps[phys_idx]

        if pack_factor == 2 and op.attrs.get("packed_b32"):
            # idx is a pair index (B32 register index), not a logical element
            # index — use it directly as the physical register selector.
            ctx.regs.bind(out, (phys_comps[idx],))
        elif pack_factor == 2:
            # B32 → 2 × B16. Split via mov.b32 {lo, hi}, reg.
            lo = ctx.regs.declare("b16")
            hi = ctx.regs.declare("b16")
            ctx.emit(f"mov.b32 {{{lo}, {hi}}}, {phys_reg};")
            ctx.regs.bind(out, (lo if sub_idx == 0 else hi,))
        elif pack_factor == 4:
            # B32 → 4 × U8/E4M3. Extract via byte shifts.
            tmp = ctx.regs.declare("b32")
            shift = sub_idx * 8
            if shift > 0:
                ctx.emit(f"shr.u32 {tmp}, {phys_reg}, {shift};")
            else:
                ctx.emit(f"mov.b32 {tmp}, {phys_reg};")
            out_r = ctx.regs.declare("b16")
            ctx.emit(f"cvt.u16.u32 {out_r}, {tmp};")
            ctx.regs.bind(out, (out_r,))
        else:
            raise NotImplementedError(f"VecExtractOp: pack_factor={pack_factor} not supported")

    def _visit_split_b32(self, op: SplitB32Op, ctx: _FnCtx) -> None:
        """b32 → (lo_b16, hi_b16) via `mov.b32 {lo, hi}, src;`."""
        lo, hi = op.results
        (src,) = op.operands
        lo_name = ctx.regs.name_for(lo)
        hi_name = ctx.regs.name_for(hi)
        src_name = ctx.regs.name_for(src)
        ctx.emit(f"mov.b32 {{{lo_name}, {hi_name}}}, {src_name};")

    def _visit_merge_b32(self, op: MergeB32Op, ctx: _FnCtx) -> None:
        """(lo_b16, hi_b16) → b32 via `mov.b32 dst, {lo, hi};`."""
        (out,) = op.results
        lo, hi = op.operands
        ctx.emit(
            f"mov.b32 {ctx.regs.name_for(out)}, "
            f"{{{ctx.regs.name_for(lo)}, {ctx.regs.name_for(hi)}}};"
        )

    # ------------------------------------------------------------------
    # §5.5 Cross-lane / subgroup
    # ------------------------------------------------------------------

    def _visit_shuffle(self, op: ShuffleOp, ctx: _FnCtx) -> None:
        """Lower `ShuffleOp` to `shfl.sync.<mode>.b32`.

        PTX has four physical shuffle modes (up/down/bfly/idx); we map
        our IR `xor` kind to `bfly` (they're the same operation). The
        `clamp` parameter is 0 for `up` (lanes that would read out of
        range keep their own value) and 31 (warp-wrap) for everything
        else — matching the convention in `ops/reduce.py`. The
        member-mask is -1, meaning "all 32 lanes".
        """
        kind = op.attrs["kind"]
        param = int(op.attrs["param"])
        src = op.operands[0]
        (out,) = op.results
        # xor is the same operation as bfly in PTX.
        ptx_mode = "bfly" if kind == "xor" else kind
        clamp = 0 if ptx_mode == "up" else 31
        dst_reg = ctx.regs.name_for(out)
        src_reg = ctx.regs.name_for(src)
        ctx.emit(f"shfl.sync.{ptx_mode}.b32 {dst_reg}, {src_reg}, {param}, {clamp}, -1;")

    def _visit_subgroup_reduce(self, op: SubgroupReduceOp, ctx: _FnCtx) -> None:
        """Lower `SubgroupReduceOp` to a butterfly shuffle chain.

        PTX has no direct warp-reduction primitive on sm_89 — the
        canonical pattern is 5 `shfl.sync.bfly` + 5 combine ops over
        xor-offsets (16, 8, 4, 2, 1). This matches
        `ops/reduce.py::emit_warp_reduce_sum_f32` exactly.

        Integer/float sum/max/min use the dtype-native arith; bitwise
        and/or use the bit-typed suffix.
        """
        reduce_op = op.attrs["op"]
        src = op.operands[0]
        (out,) = op.results
        src_reg = ctx.regs.name_for(src)
        cls = reg_class(src.dtype)

        # Pick the combine instruction for each reduction op.
        if reduce_op in ("sum", "max", "min"):
            mnemonic = {"sum": "add", "max": "max", "min": "min"}[reduce_op]
            combine_suffix = arith_suffix(src.dtype)
        elif reduce_op in ("and", "or"):
            mnemonic = reduce_op
            combine_suffix = {2: "b16", 4: "b32", 8: "b64"}[src.dtype.bytes]
        else:
            raise NotImplementedError(f"SubgroupReduceOp: op {reduce_op!r}")

        cur = src_reg
        for offset in (16, 8, 4, 2, 1):
            other = ctx.regs.declare(cls)
            ctx.emit(f"shfl.sync.bfly.b32 {other}, {cur}, {offset}, 31, -1;")
            nxt = ctx.regs.declare(cls)
            ctx.emit(f"{mnemonic}.{combine_suffix} {nxt}, {cur}, {other};")
            cur = nxt

        # Bind the final accumulator register to the result Value so any
        # downstream reader sees it without an extra mov.
        ctx.regs.bind(out, (cur,))

    def _visit_subgroup_broadcast(self, op: SubgroupBroadcastOp, ctx: _FnCtx) -> None:
        """Lower `SubgroupBroadcastOp` to `shfl.sync.idx.b32`.

        Broadcast a scalar from one lane to every lane. Typically used
        after a one-lane reduction (e.g. lane 0 holds the block sum)
        to share the result with the whole warp.
        """
        lane = int(op.attrs["lane"])
        src = op.operands[0]
        (out,) = op.results
        ctx.emit(
            f"shfl.sync.idx.b32 {ctx.regs.name_for(out)}, {ctx.regs.name_for(src)}, {lane}, 31, -1;"
        )

    # ------------------------------------------------------------------
    # §5.6 Thread identity
    # ------------------------------------------------------------------

    def _visit_thread_idx(self, op: ThreadIdxOp, ctx: _FnCtx) -> None:
        self._emit_sreg_mov(op.results[0], f"%tid.{op.attrs['dim']}", ctx)

    def _visit_block_idx(self, op: BlockIdxOp, ctx: _FnCtx) -> None:
        self._emit_sreg_mov(op.results[0], f"%ctaid.{op.attrs['dim']}", ctx)

    def _visit_block_dim(self, op: BlockDimOp, ctx: _FnCtx) -> None:
        self._emit_sreg_mov(op.results[0], f"%ntid.{op.attrs['dim']}", ctx)

    def _visit_grid_dim(self, op: GridDimOp, ctx: _FnCtx) -> None:
        self._emit_sreg_mov(op.results[0], f"%nctaid.{op.attrs['dim']}", ctx)

    def _visit_lane_id(self, op: LaneIdOp, ctx: _FnCtx) -> None:
        self._emit_sreg_mov(op.results[0], "%laneid", ctx)

    def _visit_subgroup_id(self, op: SubgroupIdOp, ctx: _FnCtx) -> None:
        # warp_id = tid.x >> 5 (traditional PTX idiom).
        out = op.results[0]
        dst = ctx.regs.name_for(out)
        tmp = ctx.regs.declare("u32")
        ctx.emit(f"mov.u32 {tmp}, %tid.x;")
        ctx.emit(f"shr.u32 {dst}, {tmp}, 5;")

    def _visit_group_id(self, op: Op, ctx: _FnCtx) -> None:
        """groupID = %laneid >> 2."""
        out = op.results[0]
        dst = ctx.regs.name_for(out)
        tmp = ctx.regs.declare("u32")
        ctx.emit(f"mov.u32 {tmp}, %laneid;")
        ctx.emit(f"shr.b32 {dst}, {tmp}, 2;")

    def _visit_thread_id_in_group(self, op: Op, ctx: _FnCtx) -> None:
        """threadID_in_group = %laneid & 3."""
        out = op.results[0]
        dst = ctx.regs.name_for(out)
        tmp = ctx.regs.declare("u32")
        ctx.emit(f"mov.u32 {tmp}, %laneid;")
        ctx.emit(f"and.b32 {dst}, {tmp}, 3;")

    def _emit_sreg_mov(self, value: Value, sreg: str, ctx: _FnCtx) -> None:
        dst = ctx.regs.name_for(value)
        ctx.emit(f"mov.u32 {dst}, {sreg};")

    # ------------------------------------------------------------------
    # §5.7 Control flow
    # ------------------------------------------------------------------

    def _visit_barrier(self, op: BarrierOp, ctx: _FnCtx) -> None:
        scope = op.attrs.get("scope", "block")
        if scope == "block":
            ctx.emit("bar.sync 0;")
        elif scope == "subgroup":
            ctx.emit("bar.warp.sync 0xffffffff;")
        elif scope == "system":
            ctx.emit("membar.sys;")
        else:
            raise NotImplementedError(f"BarrierOp scope {scope!r}")

    def _visit_for_loop(self, op: ForLoopOp, ctx: _FnCtx) -> None:
        # Layout — while loop (guard before first iteration):
        #
        #     mov.u32 iv, lo
        #     (for each carried_in[i]) mov.<cls> result[i], carried_in[i]
        #     setp.lt.u32 p_guard, iv, hi
        #     @!p_guard bra end_label
        #   loop_label:
        #     (body ops — carried_body_vars[i] aliased to result[i])
        #     (yield) mov.<cls> result[i], yielded[i]      # coalesce
        #     add.u32 iv, iv, step
        #     setp.lt.u32 p, iv, hi
        #     @p bra loop_label
        #   end_label:
        #
        # The guard ensures zero-trip loops skip the body entirely.
        iv = op.induction_var
        assert iv is not None
        iv_reg = ctx.regs.name_for(iv)
        lo, hi, step = op.lo, op.hi, op.step

        # Register class for the induction variable (always int in V1).
        iv_cls = reg_class(iv.dtype)

        # Initialize iv = lo.
        ctx.emit(f"mov.{iv_cls} {iv_reg}, {ctx.regs.name_for(lo)};")

        # Allocate registers for each result (one per carried slot) and
        # alias carried_body_vars onto them. Pre-loop mov: result[i] =
        # carried_in[i].
        carried_in = op.carried_in
        for cin, res, cbv in zip(carried_in, op.results, op.carried_body_vars, strict=False):
            ctx.regs.alias(cbv, res)
            _emit_value_mov(ctx, res, cin)

        # Guard: skip body entirely when lo >= hi (zero-trip loop).
        end_label = ctx.fresh_label("LEND")
        guard_pred = ctx.regs.declare("pred")
        ctx.emit(f"setp.lt.{iv_cls} {guard_pred}, {iv_reg}, {ctx.regs.name_for(hi)};")
        ctx.emit(f"@!{guard_pred} bra {end_label};")

        loop_label = ctx.fresh_label("L")
        ctx.emit_label(loop_label)

        # Body — push self on op_stack so any yield inside resolves
        # to this loop's result regs.
        ctx.op_stack.append(op)
        self._walk_region(op.body.ops, ctx)
        ctx.op_stack.pop()

        # Backedge: iv += step; setp.lt iv, hi; @p bra loop_label.
        pred = ctx.regs.declare("pred")
        ctx.emit(f"add.{iv_cls} {iv_reg}, {iv_reg}, {ctx.regs.name_for(step)};")
        ctx.emit(f"setp.lt.{iv_cls} {pred}, {iv_reg}, {ctx.regs.name_for(hi)};")
        ctx.emit(f"@{pred} bra {loop_label};")
        ctx.emit_label(end_label)

    def _visit_if_region(self, op: IfRegionOp, ctx: _FnCtx) -> None:
        n_carried = op.attrs.get("n_carried", 0)
        pred = op.pred
        pred_reg = ctx.regs.name_for(pred)

        else_label = ctx.fresh_label("ELSE")
        end_label = ctx.fresh_label("ENDIF")

        # Pre-allocate result registers and alias body vars onto them so
        # both arms coalesce into the same physical slots.
        carried_in_operands = op.operands[1 : 1 + n_carried]
        # (The else carried list is at 1+n_carried:1+2*n_carried, but for
        # yielding we only need one result reg per slot — both arms write
        # to the same register. So we alias then_body_vars + else_body_vars
        # onto the same result slot.)
        for i, (res, tbv, ebv) in enumerate(
            zip(op.results, op.then_body_vars, op.else_body_vars, strict=False)
        ):
            ctx.regs.alias(tbv, res)
            ctx.regs.alias(ebv, res)
            _emit_value_mov(ctx, res, carried_in_operands[i])

        # Branch: if !pred go to else.
        ctx.emit(f"@!{pred_reg} bra {else_label};")

        # Then region — body will emit YieldOp which coalesces into result regs.
        ctx.op_stack.append(op)
        self._walk_region(op.then_region.ops, ctx)
        ctx.emit(f"bra {end_label};")

        # Else region
        ctx.emit_label(else_label)
        self._walk_region(op.else_region.ops, ctx)
        ctx.op_stack.pop()

        ctx.emit_label(end_label)

    def _visit_yield(self, op: YieldOp, ctx: _FnCtx) -> None:
        """Emit register moves from yielded values into the parent op's
        result regs. The parent op is the innermost structured op on
        `ctx.op_stack`; a top-level yield (no enclosing loop/if) is a
        no-op since there are no result regs to coalesce into."""
        if not ctx.op_stack:
            return
        parent_op = ctx.op_stack[-1]
        if isinstance(parent_op, (ForLoopOp, IfRegionOp)):
            results = parent_op.results
        else:
            raise NotImplementedError(f"YieldOp parent {type(parent_op).__name__} not supported")
        for res, yielded in zip(results, op.operands, strict=False):
            _emit_value_mov(ctx, res, yielded)

    # ------------------------------------------------------------------
    # §5.4 Shared-memory allocation
    # ------------------------------------------------------------------

    def _visit_smem_alloc(self, op: SmemAllocOp, ctx: _FnCtx) -> None:
        (backing,) = op.results
        dtype: DType = op.attrs["dtype"]
        shape: tuple[int, ...] = tuple(op.attrs["shape"])
        pad: int = int(op.attrs.get("pad", 0))
        elem_bytes = dtype.bytes
        # Element count honors the per-row pad (only on rank-2 tiles today).
        if len(shape) == 2:
            row_stride = shape[1] + pad
            elems = shape[0] * row_stride
        else:
            elems = 1
            for s in shape:
                elems *= s
        size = elems * elem_bytes
        # Offset comes from the smem_layout plan when present; aliased
        # allocs share offsets. Falls back to per-op counter for the
        # legacy / no-plan path.
        if ctx.smem_plan is not None and backing.id in ctx.smem_plan.region_to_slot:  # type: ignore
            offset = ctx.smem_plan.offset_for(backing.id)  # type: ignore
            slot_size = ctx.smem_plan.size_for_slot(backing.id)  # type: ignore
            # Track running max so legacy callers reading
            # ``ctx.smem_size_bytes`` mid-walk see the right total.
            end_byte = offset + slot_size
            if end_byte > ctx.smem_size_bytes:
                ctx.smem_size_bytes = end_byte
        else:
            offset = _align_up(ctx.smem_size_bytes, ctx.smem_align)
            ctx.smem_size_bytes = offset + size
        # Register holding the u32 shared-space base address for this alloc.
        base_reg = ctx.regs.name_for(backing)  # class: u64 per IR; we override
        # The backing Value is declared u64 in the IR (so it can be
        # passed around), but PTX shared addresses are u32. We compute
        # the u32 base directly via cvta.shared.u32 and then a sized
        # immediate add, and store it in a u32 temp register. The
        # "backing" Value's register is used at load/store sites; we
        # allocate a fresh u32 for that job.
        #
        # For simplicity, allocate a u32 reg aliased to this Value.
        u32_base = ctx.regs.declare("u32")
        ctx.emit(f"mov.u32 {u32_base}, {ctx.smem_base_name};")
        if offset != 0:
            ctx.emit(f"add.u32 {u32_base}, {u32_base}, {offset};")
        # Record the mapping so load/store can find this base.
        ctx.smem_allocs[backing.id] = (u32_base, offset, size)
        # Also emit a stub mov on the IR Value's own (u64) reg so that
        # any SSA consumer that reads `backing` still sees a defined
        # value. This is dead in V1 (nothing reads it directly) but
        # keeps the reg-alloc decl consistent.
        ctx.emit(f"cvt.u64.u32 {base_reg}, {u32_base};")

    # ------------------------------------------------------------------
    # §5.3 Memory (scalar load/store)
    # ------------------------------------------------------------------

    def _visit_load(self, op: LoadOp, ctx: _FnCtx) -> None:
        tensor = op.attrs["tensor"]
        (out,) = op.results
        dst = ctx.regs.name_for(out)
        indices = self._strip_pred(op)
        space, addr_expr = self._compute_tensor_addr(tensor, indices, ctx)
        cls = reg_class(out.dtype)
        prefix = self._pred_prefix(op, ctx)
        ctx.emit(f"{prefix}ld.{space}.{cls} {dst}, [{addr_expr}];")

    def _visit_store(self, op: StoreOp, ctx: _FnCtx) -> None:
        tensor = op.attrs["tensor"]
        value = op.operands[0]
        indices = op.operands[1:]
        if op.attrs.get("pred") is not None:
            indices = indices[:-1]
        space, addr_expr = self._compute_tensor_addr(tensor, indices, ctx)
        cls = reg_class(value.dtype)
        prefix = self._pred_prefix(op, ctx)
        ctx.emit(f"{prefix}st.{space}.{cls} [{addr_expr}], {ctx.regs.name_for(value)};")

    def _visit_vec_load(self, op: VecLoadOp, ctx: _FnCtx) -> None:
        tensor = op.attrs["tensor"]
        (out,) = op.results
        width = int(op.attrs["width"])
        # Pick the canonical PTX (vec_width, reg_class) for this transfer
        # — total bytes are preserved, but the reg class may be widened
        # (e.g. v8.b16 → v4.b32) so we always emit ONE instruction. The
        # IR Value's components are bound to physical regs of the chosen
        # class; downstream uses (vec_store) re-derive the same form via
        # `_vec_phys_form` and read the same components.
        form = _vec_phys_form(width, out.dtype)
        if form is None:
            raise NotImplementedError(
                f"VecLoadOp: width={width} dtype={out.dtype} (= "
                f"{width * out.dtype.bytes} B total) has no single-instruction "
                f"PTX lowering. Legal totals: 1/2/4/8/16 B."
            )
        v_width, reg_dt = form
        comps = _bind_phys_vec(ctx.regs, out, v_width, reg_dt)
        cls = reg_class(reg_dt)
        indices = self._strip_pred(op)
        space, addr_expr = self._compute_tensor_addr(tensor, indices, ctx)
        prefix = self._pred_prefix(op, ctx)
        if v_width == 1:
            ctx.emit(f"{prefix}ld.{space}.{cls} {comps[0]}, [{addr_expr}];")
        else:
            braced = "{" + ", ".join(comps) + "}"
            ctx.emit(f"{prefix}ld.{space}.v{v_width}.{cls} {braced}, [{addr_expr}];")

    def _visit_vec_store(self, op: VecStoreOp, ctx: _FnCtx) -> None:
        tensor = op.attrs["tensor"]
        vec = op.operands[0]
        width = vec.width
        rest = op.operands[1:]
        if op.attrs.get("pred") is not None:
            rest = rest[:-1]
        form = _vec_phys_form(width, vec.dtype)
        if form is None:
            raise NotImplementedError(
                f"VecStoreOp: width={width} dtype={vec.dtype} (= "
                f"{width * vec.dtype.bytes} B total) has no single-instruction "
                f"PTX lowering. Legal totals: 1/2/4/8/16 B."
            )
        v_width, reg_dt = form
        comps = _bind_phys_vec(ctx.regs, vec, v_width, reg_dt)
        cls = reg_class(reg_dt)
        space, addr_expr = self._compute_tensor_addr(tensor, rest, ctx)
        prefix = self._pred_prefix(op, ctx)
        if v_width == 1:
            ctx.emit(f"{prefix}st.{space}.{cls} [{addr_expr}], {comps[0]};")
        else:
            braced = "{" + ", ".join(comps) + "}"
            ctx.emit(f"{prefix}st.{space}.v{v_width}.{cls} [{addr_expr}], {braced};")

    # ------------------------------------------------------------------
    # §5.3 Async copy (cp.async)
    # ------------------------------------------------------------------

    def _visit_async_copy(self, op: AsyncCopyOp, ctx: _FnCtx) -> None:
        dst = op.attrs["dst_tensor"]
        src = op.attrs["src_tensor"]
        count = int(op.attrs["count"])
        n_dst = int(op.attrs["n_dst_idx"])
        n_src = int(op.attrs["n_src_idx"])
        pred = op.attrs.get("pred")
        if count not in (4, 8, 16):
            raise NotImplementedError(
                f"AsyncCopyOp: cp.async count must be 4/8/16 bytes, got {count}"
            )
        # Split operands into (dst_idxs, src_idxs, [pred]).
        dst_idxs = tuple(op.operands[:n_dst])
        src_idxs = tuple(op.operands[n_dst : n_dst + n_src])
        # Compute the two addresses.
        smem_space, smem_addr = self._compute_tensor_addr(dst, dst_idxs, ctx)
        gmem_space, gmem_addr = self._compute_tensor_addr(src, src_idxs, ctx)
        assert smem_space == "shared" and gmem_space == "global"
        prefix = ""
        if pred is not None:
            prefix = f"@{ctx.regs.name_for(pred)} "
        ctx.emit(
            f"{prefix}cp.async.ca.shared.global.L2::256B [{smem_addr}], [{gmem_addr}], {count};"
        )

    def _visit_async_commit(self, op: AsyncCopyCommitOp, ctx: _FnCtx) -> None:
        ctx.emit("cp.async.commit_group;")

    def _visit_async_wait(self, op: AsyncCopyWaitOp, ctx: _FnCtx) -> None:
        n = int(op.attrs["n"])
        ctx.emit(f"cp.async.wait_group {n};")

    # ------------------------------------------------------------------
    # §5.3 Atomic RMW
    # ------------------------------------------------------------------

    def _visit_atomic_rmw(self, op: AtomicRmwOp, ctx: _FnCtx) -> None:
        tensor = op.attrs["tensor"]
        atomic_op = op.attrs["op"]  # "add" / "min" / "max" / ...
        (out,) = op.results
        value = op.operands[0]
        indices = op.operands[1:]
        if op.attrs.get("pred") is not None:
            indices = indices[:-1]
        space, addr_expr = self._compute_tensor_addr(tensor, indices, ctx)
        assert space == "global", "AtomicRmwOp: only global atomics in V1"
        atomic_type = op.attrs.get("atomic_type")
        prefix = self._pred_prefix(op, ctx)
        if atomic_type:
            cls = atomic_type
        else:
            cls = arith_suffix(value.dtype)
        # f16/bf16/f16x2/bf16x2 atomic add requires .noftz qualifier.
        noftz = ""
        if atomic_op == "add" and (
            value.dtype in (DType.F16, DType.BF16) or atomic_type in ("bf16x2", "f16x2")
        ):
            noftz = ".noftz"
        # Vector atomic types (bf16x2, f16x2) are only exposed via `red.add`
        # on sm_80–sm_89 — `atom.add.bf16x2` requires sm_90+. `red` has no
        # result (no old-value write-back); the fresh output register we
        # declared is left unwritten and elided by ptxas as dead. Callers
        # never read it (scatter epilogue throws it away), so this is safe.
        if atomic_type in ("bf16x2", "f16x2"):
            ctx.emit(
                f"{prefix}red.global.{atomic_op}{noftz}.{cls} "
                f"[{addr_expr}], {ctx.regs.name_for(value)};"
            )
            return
        ctx.emit(
            f"{prefix}atom.global.{atomic_op}{noftz}.{cls} "
            f"{ctx.regs.name_for(out)}, [{addr_expr}], {ctx.regs.name_for(value)};"
        )

    # ------------------------------------------------------------------
    # §5.8 Matmul (MmaOp / LoadMatrixOp / StoreMatrixOp)
    # ------------------------------------------------------------------

    def _visit_frag_apply(self, op: FragApplyOp, ctx: _FnCtx) -> None:
        """Lower FragApplyOp by re-walking the body region once per c_reg.

        PTX accumulator regs are already scalar — each c_reg is one f32
        register per lane. The transform body is a scalar subgraph
        (one scalar input → one scalar output), so "apply per element"
        means "walk the body once per c_reg, binding the body input
        to that c_reg and fresh-naming every body-local Value so we
        don't reuse names across walks."

        Free variables (Values defined OUTSIDE the body region — scales,
        constants, reduced maxes) keep their bindings; they're read-only
        on each walk and their source ops were emitted before this op.
        """
        in_frag = op.in_frag
        (out,) = op.results

        in_comps = ctx.regs.components(in_frag)
        out_comps = ctx.regs.components(out)

        input_var = op.body_input_var
        assert input_var is not None
        sel_var = op.body_selector_var
        selectors = op.selectors
        slot_to_sel = op.attrs.get("slot_to_selector_idx")
        sel_comps = [ctx.regs.components(s)[0] for s in selectors] if selectors else []
        # Collect body-local Value ids (results of ops inside the body).
        # These must be re-allocated fresh on each slot-walk. The input_var
        # is handled separately — we bind it explicitly per slot.
        body_local_ids = _collect_body_local_value_ids(op.body.ops)

        for i, out_comp in enumerate(out_comps):
            # Rebind input var to the current c_reg. force-pop any prior
            # binding so successive walks see successive c_regs.
            ctx.regs._components.pop(input_var.id, None)
            ctx.regs._components[input_var.id] = (in_comps[i],)
            if sel_var is not None and slot_to_sel is not None:
                ctx.regs._components.pop(sel_var.id, None)
                ctx.regs._components[sel_var.id] = (sel_comps[slot_to_sel[i]],)
            # Wipe body-local bindings so the walk freshly allocates
            # names. Without this, walk #2 would reuse walk #1's regs
            # and emit `mov %f7, %f7; …` duplicates.
            for vid in body_local_ids:
                ctx.regs._components.pop(vid, None)

            for bop in op.body.ops:
                if isinstance(bop, YieldOp):
                    yielded = bop.operands[0]
                    # Emit the slot-write from the yielded body result
                    # into out[i]. `_emit_value_mov` picks the right
                    # reg class / suffix.
                    ctx.emit(
                        f"mov.{reg_class(yielded.dtype)} {out_comp}, {ctx.regs.name_for(yielded)};"
                    )
                    break
                self._visit(bop, ctx)

    def _visit_frag_convert(self, op: FragConvertOp, ctx: _FnCtx) -> None:
        """Lower FragConvertOp for ACC f32 → A_FRAG bf16.

        Arbitrary register-tile conversions are out of scope today —
        this lowering is specialized for standard m16n8 bf16 where
        cd_offsets groups c_regs as [0,1] (dr=0) and [2,3] (dr=8).
        Other source/destination layouts or dtypes raise
        ``NotImplementedError`` so we don't silently miscompile when a
        future caller takes the generic path.
        """
        src_layout = op.attrs["src_layout"]
        dst_layout = op.attrs["dst_layout"]
        src_dtype = op.attrs["src_dtype"]
        dst_dtype = op.attrs["dst_dtype"]
        if (src_layout, dst_layout) != ("acc", "a_frag"):
            raise NotImplementedError(
                f"FragConvertOp PTX: only acc→a_frag implemented (got {src_layout}→{dst_layout})."
            )
        if (src_dtype, dst_dtype) != (DType.F32, DType.BF16):
            raise NotImplementedError(
                f"FragConvertOp PTX: only f32→bf16 implemented for acc→a_frag "
                f"(got {src_dtype}→{dst_dtype})."
            )
        cd_offsets = op.attrs["cd_offsets"]
        src_frags = op.src_frags
        selectors = op.selectors
        slot_to_sel = op.attrs.get("slot_to_selector_idx")
        sel_comps = [ctx.regs.components(s)[0] for s in selectors] if selectors else []
        (out_val,) = op.results
        out_comps = ctx.regs.components(out_val)

        elem_var = op.body_input_var
        sel_var = op.body_selector_var
        body = op.body
        body_local_ids = _collect_body_local_value_ids(body.ops) if body else []

        # Row-class groupings: indices of c_regs per row class.
        dr_vals = sorted({dr for dr, _ in cd_offsets})
        class_of = [dr_vals.index(dr) for dr, _ in cd_offsets]
        rc_to_idxs: dict[int, list[int]] = {}
        for i, rc in enumerate(class_of):
            rc_to_idxs.setdefault(rc, []).append(i)
        # Expect exactly 2 c_regs per rc (for standard m16n8 bf16). Emit
        # pairs in rc order → output reg order matches a_offsets.

        out_idx = 0
        for src_frag in src_frags:
            src_comps = ctx.regs.components(src_frag)
            for rc in range(len(dr_vals)):
                pair = rc_to_idxs[rc]
                if len(pair) != 2:
                    raise NotImplementedError(
                        f"FragConvertOp: expected 2 c_regs per row class, "
                        f"got {len(pair)} for rc {rc}"
                    )
                bf16_regs: list[str] = []
                for slot_idx in pair:
                    # Rebind body vars.
                    if elem_var is not None:
                        ctx.regs._components.pop(elem_var.id, None)
                        ctx.regs._components[elem_var.id] = (src_comps[slot_idx],)
                    if sel_var is not None and slot_to_sel is not None:
                        ctx.regs._components.pop(sel_var.id, None)
                        ctx.regs._components[sel_var.id] = (sel_comps[slot_to_sel[slot_idx]],)
                    for vid in body_local_ids:
                        ctx.regs._components.pop(vid, None)

                    # Value to convert: yielded (if body), else direct src.
                    if body is not None:
                        for bop in body.ops:
                            if isinstance(bop, YieldOp):
                                to_convert = ctx.regs.name_for(bop.operands[0])
                                break
                            self._visit(bop, ctx)
                        else:
                            raise RuntimeError("FragConvertOp body missing YieldOp")
                    else:
                        to_convert = src_comps[slot_idx]

                    # cvt f32 → bf16
                    bf16 = ctx.regs.declare("b16")
                    ctx.emit(f"cvt.rn.bf16.f32 {bf16}, {to_convert};")
                    bf16_regs.append(bf16)

                # Pack 2 bf16 → 1 b32: mov.b32 %out, {%h_lo, %h_hi};
                ctx.emit(f"mov.b32 {out_comps[out_idx]}, {{{bf16_regs[0]}, {bf16_regs[1]}}};")
                out_idx += 1

    def _visit_frag_for_each(self, op: FragForEachOp, ctx: _FnCtx) -> None:
        """Lower FragForEachOp on PTX: walk body once per c_reg, binding
        body_input_var to the c_reg, body_row_var to ``gid + dr``, and
        body_col_var to ``tig*2 + dc``. No output fragment; body emits
        stores/atomics inline.
        """
        cd_offsets = op.attrs["cd_offsets"]
        in_frag = op.in_frag
        in_comps = ctx.regs.components(in_frag)

        elem_var = op.body_input_var
        row_var = op.body_row_var
        col_var = op.body_col_var
        sel_var = op.body_selector_var
        selectors = op.selectors
        slot_to_sel = op.attrs.get("slot_to_selector_idx")
        sel_comps = [ctx.regs.components(s)[0] for s in selectors] if selectors else []
        assert elem_var is not None and row_var is not None and col_var is not None

        # Compute the per-lane bases once at op entry.
        gid_reg = ctx.regs.declare("u32")
        tig_reg = ctx.regs.declare("u32")
        tig_x2_reg = ctx.regs.declare("u32")
        ctx.emit(f"mov.u32 {gid_reg}, %laneid;")
        ctx.emit(f"shr.u32 {gid_reg}, {gid_reg}, 2;")
        ctx.emit(f"mov.u32 {tig_reg}, %laneid;")
        ctx.emit(f"and.b32 {tig_reg}, {tig_reg}, 3;")
        ctx.emit(f"shl.b32 {tig_x2_reg}, {tig_reg}, 1;")

        body_local_ids = _collect_body_local_value_ids(op.body.ops)

        for i, (dr, dc) in enumerate(cd_offsets):
            # Allocate per-slot row/col regs and compute.
            row_reg = ctx.regs.declare("u32")
            col_reg = ctx.regs.declare("u32")
            if dr:
                ctx.emit(f"add.u32 {row_reg}, {gid_reg}, {dr};")
            else:
                ctx.emit(f"mov.u32 {row_reg}, {gid_reg};")
            if dc:
                ctx.emit(f"add.u32 {col_reg}, {tig_x2_reg}, {dc};")
            else:
                ctx.emit(f"mov.u32 {col_reg}, {tig_x2_reg};")

            # Rebind body position vars + elem var.
            ctx.regs._components.pop(elem_var.id, None)
            ctx.regs._components[elem_var.id] = (in_comps[i],)
            ctx.regs._components.pop(row_var.id, None)
            ctx.regs._components[row_var.id] = (row_reg,)
            ctx.regs._components.pop(col_var.id, None)
            ctx.regs._components[col_var.id] = (col_reg,)
            if sel_var is not None and slot_to_sel is not None:
                ctx.regs._components.pop(sel_var.id, None)
                ctx.regs._components[sel_var.id] = (sel_comps[slot_to_sel[i]],)

            # Wipe body-local Values so each walk allocates fresh regs.
            for vid in body_local_ids:
                ctx.regs._components.pop(vid, None)

            # Walk body. Skip terminator (void YieldOp).
            for bop in op.body.ops:
                if isinstance(bop, YieldOp):
                    break
                self._visit(bop, ctx)

    def _visit_frag_reduce(self, op: FragReduceOp, ctx: _FnCtx) -> None:
        """Lower FragReduceOp to per-class local scalar reduce + butterfly
        shuffle.

        For axis=row on m16n8 acc:
          * Group c_regs by row class (dr value).
          * For each class, fold local c_regs with the kind op (max/add…).
          * Butterfly-shuffle across tig via XOR {1, 2} so every lane in
            the row holds the full reduction.
        """
        from quark.ir.frag_tile import PTX_ACC_ROW_REDUCE_BUTTERFLY

        kind = op.attrs["kind"]
        axis = op.attrs["axis"]
        cd_offsets = op.attrs["cd_offsets"]
        in_frag = op.in_frag
        in_comps = ctx.regs.components(in_frag)

        # Partition c_regs by class.
        if axis == "row":
            classes = sorted({dr for dr, _ in cd_offsets})
            class_of = [classes.index(dr) for dr, _ in cd_offsets]
            butterfly = PTX_ACC_ROW_REDUCE_BUTTERFLY
        else:  # col
            classes = sorted({dc for _, dc in cd_offsets})
            class_of = [classes.index(dc) for _, dc in cd_offsets]
            # For col reduction we'd fold across groupID (gid) — not a
            # tig butterfly. Needs a different pattern; leave as TODO.
            raise NotImplementedError("FragReduceOp axis='col' not yet implemented on PTX")

        suffix = arith_suffix(in_frag.dtype)
        per_class_comps: list[list[str]] = [[] for _ in classes]
        for ci, comp in zip(class_of, in_comps, strict=False):
            per_class_comps[ci].append(comp)

        for class_idx, result_val in enumerate(op.results):
            comps = per_class_comps[class_idx]
            res_reg = ctx.regs.name_for(result_val)
            # Local fold: start with comp[0], fold each subsequent comp.
            if len(comps) == 1:
                ctx.emit(f"mov.{reg_class(in_frag.dtype)} {res_reg}, {comps[0]};")
            else:
                ctx.emit(f"{kind}.{suffix} {res_reg}, {comps[0]}, {comps[1]};")
                for extra in comps[2:]:
                    ctx.emit(f"{kind}.{suffix} {res_reg}, {res_reg}, {extra};")
            # Butterfly shuffle reduce.
            for dist in butterfly:
                tmp = ctx.regs.declare("f32")
                ctx.emit(f"shfl.sync.bfly.b32 {tmp}, {res_reg}, {dist}, 0x1f, 0xffffffff;")
                ctx.emit(f"{kind}.{suffix} {res_reg}, {res_reg}, {tmp};")

    def _visit_mma(self, op: MmaOp, ctx: _FnCtx) -> None:
        """Generic mma.sync emit: look up the MmaShape.ptx suffix and
        emit `mma.sync.aligned.<ptx> {d}, {a}, {b}, {c};`.

        The fragment Values are width-N b32 vecs (see Builder.load_matrix
        / Builder.mma); `name_for` returns the braced `{%b0, %b1, ...}`
        form directly.
        """
        shape_id = op.attrs["shape_id"]
        module = ctx.module
        if module is None or shape_id not in module.kernel_shapes:
            raise RuntimeError(f"MmaOp: shape {shape_id!r} not in module.kernel_shapes")
        shape = module.kernel_shapes[shape_id]
        if not shape.ptx:
            raise NotImplementedError(
                f"MmaOp: MmaShape {shape_id!r} has no `ptx` suffix registered "
                f"(needed for PTX lowering)"
            )
        a, b_frag, c = op.operands
        (d,) = op.results

        # `mma.sync` requires every operand to be a braced vector,
        # even the single-register B fragments in packed fp8 shapes.
        # `name_for` drops the braces for width-1 Values, so reach for
        # `components()` and brace them here.
        def _braced(v: Value) -> str:
            return "{" + ", ".join(ctx.regs.components(v)) + "}"

        ctx.emit(
            f"mma.sync.aligned.{shape.ptx} "
            f"{_braced(d)}, "
            f"{_braced(a)}, "
            f"{_braced(b_frag)}, "
            f"{_braced(c)};"
        )

    def _visit_load_matrix(self, op: LoadMatrixOp, ctx: _FnCtx) -> None:
        """Lower a fragment load.

        Two paths, selected by `attrs["layout_hint"]`:

        1. **Default (manual scalar loads)** — emit N `ld.shared.b32`
           instructions, one per fragment register, at
           `[base + (row+dr)*row_stride + (col+dc)*elem_bytes]` where
           `(dr, dc)` comes from `attrs["reg_offsets"][i]`. This mirrors
           the hoisted-base pattern in `mma/frag.py` and supports
           preshuffled smem layouts (since the offsets are fully
           explicit) and arbitrary mma shapes/dtypes (since the
           per-register byte stride drops out of the tensor's stride
           and element dtype automatically).

        2. **`layout_hint="ldmatrix"` (opt-in fast path)** — emit one
           `ldmatrix.sync.aligned.x<N>.m8n8.shared.b16` at the tile
           base address. Requires the smem layout to match ldmatrix's
           fixed m8n8 lane mapping; most kernels in this repo do NOT
           use this path because they exploit a preshuffled layout or
           a dtype cast around the load.

        SharedRegion sources only; GlobalTensor fragment loads should
        stage through smem first via `cp.async`.
        """
        src = op.attrs["src_tensor"]
        if not isinstance(src, SharedRegion):
            raise NotImplementedError(
                "LoadMatrixOp: PTX backend only supports SharedRegion sources; "
                "stage gmem → smem via cp.async first"
            )
        (out,) = op.results
        width = out.width
        layout = op.attrs.get("layout_hint", "manual")
        row, col = op.operands

        if layout == "ldmatrix":
            # Opt-in fast path. Width must be one of {1, 2, 4}.
            if width not in (1, 2, 4):
                raise NotImplementedError(
                    f"LoadMatrixOp: ldmatrix only supports x1/x2/x4, got width={width}"
                )
            space, addr = self._compute_tensor_addr(src, op.operands, ctx)
            dst = ctx.regs.name_for(out)
            ctx.emit(f"ldmatrix.sync.aligned.x{width}.m8n8.shared.b16 {dst}, [{addr}];")
            return

        # Default: manual scalar loads at per-register offsets.
        reg_offsets = op.attrs.get("reg_offsets")
        if reg_offsets is None:
            raise ValueError(
                "LoadMatrixOp: default manual lowering requires `reg_offsets` "
                "in attrs (one (row_elem, col_elem) pair per fragment "
                "register). Sourced from the PTX ISA fragment formulas — "
                "see tests/lower/ptx/test_matmul.py for worked tables. "
                "Pass `layout_hint='ldmatrix'` to use the ldmatrix fast path "
                "instead."
            )
        self._emit_manual_frag_loads(src, row, col, out, reg_offsets, ctx)

    def _emit_manual_frag_loads(
        self,
        src: SharedRegion,
        row: Value,
        col: Value,
        out: Value,
        reg_offsets: tuple[tuple[int, int], ...],
        ctx: _FnCtx,
    ) -> None:
        """Emit N scalar `ld.shared.b32` at per-register offsets.

        The per-lane offset (groupID/tidIG math) lives on the
        SharedRegion view's `dyn_offset`; the tile base `(row, col)`
        is a compile-time or runtime value; and the per-register
        `(dr, dc)` pair is a Python tuple of ints. All three collapse
        together via `_compute_tensor_addr_parts`, which hands us a
        `(space, addr_reg, static_bytes)` triple — we then fold each
        register's `(dr, dc)` into the static byte offset.
        """
        space, addr_reg, base_static_bytes = self._compute_tensor_addr_parts(src, (row, col), ctx)
        row_stride_bytes = src.stride[0] * src.dtype.bytes
        col_stride_bytes = src.dtype.bytes  # innermost element stride
        comps = ctx.regs.components(out)
        # Load instruction suffix matches the fragment's *carrier* dtype,
        # not the smem tensor's element dtype: a C/D accumulator
        # fragment with f32 carrier wants `ld.shared.f32`, while A/B
        # packed fragments use `ld.shared.b32`. The underlying bits are
        # what they are; the suffix just keeps ptxas type-checking
        # happy when the scalar register class is f32.
        ld_suffix = reg_class(out.dtype)
        for (dr, dc), comp in zip(reg_offsets, comps, strict=False):
            per_reg_bytes = dr * row_stride_bytes + dc * col_stride_bytes
            total_off = base_static_bytes + per_reg_bytes
            if total_off:
                addr_expr = f"{addr_reg} + {total_off}"
            else:
                addr_expr = addr_reg
            ctx.emit(f"ld.{space}.{ld_suffix} {comp}, [{addr_expr}];")

    def _visit_store_matrix(self, op: StoreMatrixOp, ctx: _FnCtx) -> None:
        """Lower a fragment store.

        Mirror image of `_visit_load_matrix`'s default path: emit N
        scalar `st.<space>.b32` at per-register byte offsets derived
        from `attrs["reg_offsets"]` and the destination tensor's
        stride / element size. Supports both SharedRegion and
        GlobalTensor destinations, matching epilogue patterns that
        store directly to gmem.
        """
        dst = op.attrs["dst_tensor"]
        frag, row, col = op.operands
        width = frag.width
        if width == 0:
            raise ValueError("StoreMatrixOp: fragment width is 0")

        reg_offsets = op.attrs.get("reg_offsets")
        if reg_offsets is None:
            raise ValueError(
                "StoreMatrixOp: default lowering requires `reg_offsets` in "
                "attrs (one (row_elem, col_elem) pair per fragment register). "
                "Sourced from the PTX ISA accumulator formulas — see "
                "tests/lower/ptx/test_matmul.py for worked tables."
            )

        space, addr_reg, base_static_bytes = self._compute_tensor_addr_parts(dst, (row, col), ctx)
        row_stride_bytes = dst.stride[0] * dst.dtype.bytes
        col_stride_bytes = dst.dtype.bytes
        comps = ctx.regs.components(frag)
        # Instruction suffix matches the fragment carrier (f32 for
        # F32-accumulator D/C, b32 for packed A/B / integer acc).
        st_suffix = reg_class(frag.dtype)
        for (dr, dc), comp in zip(reg_offsets, comps, strict=False):
            per_reg_bytes = dr * row_stride_bytes + dc * col_stride_bytes
            total_off = base_static_bytes + per_reg_bytes
            if total_off:
                addr_expr = f"{addr_reg} + {total_off}"
            else:
                addr_expr = addr_reg
            ctx.emit(f"st.{space}.{st_suffix} [{addr_expr}], {comp};")

    # ------------------------------------------------------------------
    # Helpers shared by memory / atomic ops
    # ------------------------------------------------------------------

    def _pred_prefix(self, op: Op, ctx: _FnCtx) -> str:
        """Return `@%pN ` for a predicated op, else empty."""
        pred = op.attrs.get("pred")
        if pred is None:
            return ""
        return f"@{ctx.regs.name_for(pred)} "

    def _strip_pred(self, op: Op) -> tuple[Value, ...]:
        """Return the op's operands minus the trailing pred operand (if any)."""
        if op.attrs.get("pred") is not None:
            return op.operands[:-1]
        return op.operands

    def _compute_tensor_addr_parts(
        self,
        tensor: Any,
        indices: tuple[Value, ...],
        ctx: _FnCtx,
    ) -> tuple[str, str, int]:
        """Return (space, addr_reg, static_bytes) for tensor+indices.

        This is the structured version. Callers that emit a single load
        at (addr+static) format `[addr + static]`; callers that emit a
        group of loads at (addr + static + i*step) fold the per-iteration
        step themselves (see LoadMatrix/StoreMatrix lowering).
        """
        elem_bytes = tensor.dtype.bytes

        if isinstance(tensor, SharedRegion):
            alloc_info = ctx.smem_allocs.get(tensor.alloc.id)
            if alloc_info is None:
                raise RuntimeError(
                    f"PtxLowerer: SharedRegion {tensor.name!r} backing "
                    f"(Value id={tensor.alloc.id}) has no alloc record"
                )
            base_reg, _alloc_off, _size = alloc_info
            static_byte_off = tensor.static_offset * elem_bytes
            static_byte_off += self._static_index_bytes(tensor, indices, elem_bytes)

            addr = base_reg
            dyn_accum = self._accumulate_dyn_byte_offset(tensor, indices, ctx, elem_bytes)
            if dyn_accum is not None:
                tmp = ctx.regs.declare("u32")
                ctx.emit(f"add.u32 {tmp}, {base_reg}, {dyn_accum};")
                addr = tmp
            return "shared", addr, static_byte_off

        if isinstance(tensor, GlobalTensor):
            param_reg = ctx.regs.name_for(tensor.param.value)
            static_bytes = (
                tensor.static_row_offset * tensor.stride[0] * elem_bytes
                + tensor.static_col_offset * elem_bytes
                + self._static_index_bytes(tensor, indices, elem_bytes)
            )
            dyn_accum = self._accumulate_gmem_dyn_bytes(tensor, indices, ctx, elem_bytes)
            addr = param_reg
            if dyn_accum is not None:
                tmp = ctx.regs.declare("u64")
                ctx.emit(f"add.u64 {tmp}, {param_reg}, {dyn_accum};")
                addr = tmp
            return "global", addr, static_bytes

        raise NotImplementedError(f"Unknown tensor type: {type(tensor).__name__}")

    def _compute_tensor_addr(
        self,
        tensor: Any,
        indices: tuple[Value, ...],
        ctx: _FnCtx,
    ) -> tuple[str, str]:
        """Convenience wrapper: returns a single PTX address expression
        formed from the parts returned by `_compute_tensor_addr_parts`."""
        space, addr_reg, static_bytes = self._compute_tensor_addr_parts(tensor, indices, ctx)
        if static_bytes:
            return space, f"{addr_reg} + {static_bytes}"
        return space, addr_reg

    def _static_index_bytes(self, tensor: Any, indices: tuple[Value, ...], elem_bytes: int) -> int:
        """Return the byte contribution of any indices whose producer is
        a ConstOp. Dynamic indices contribute 0 here and are handled by
        the dyn-byte accumulator."""
        total = 0
        strides = tensor.stride
        for i, idx in enumerate(indices):
            if _is_const_value(idx):
                total += _const_int(idx) * strides[i] * elem_bytes
        return total

    def _accumulate_dyn_byte_offset(
        self,
        tensor: SharedRegion,
        indices: tuple[Value, ...],
        ctx: _FnCtx,
        elem_bytes: int,
    ) -> str | None:
        """Shared-memory dynamic offset: sum of (non-const index * stride)
        plus the tensor's dyn_offset (if any), all in u32 bytes. Returns
        the name of a u32 register holding the byte offset, or None if
        there is no dynamic contribution."""
        terms: list[str] = []
        strides = tensor.stride
        for i, idx in enumerate(indices):
            if _is_const_value(idx):
                continue
            term_reg = ctx.regs.declare("u32")
            factor = strides[i] * elem_bytes
            ctx.emit(f"mul.lo.u32 {term_reg}, {ctx.regs.name_for(idx)}, {factor};")
            terms.append(term_reg)
        if tensor.dyn_offset is not None:
            # dyn_offset is in elements — convert to bytes.
            term_reg = ctx.regs.declare("u32")
            ctx.emit(
                f"mul.lo.u32 {term_reg}, {ctx.regs.name_for(tensor.dyn_offset)}, {elem_bytes};"
            )
            terms.append(term_reg)
        if not terms:
            return None
        acc = terms[0]
        for t in terms[1:]:
            nxt = ctx.regs.declare("u32")
            ctx.emit(f"add.u32 {nxt}, {acc}, {t};")
            acc = nxt
        return acc

    def _accumulate_gmem_dyn_bytes(
        self,
        tensor: GlobalTensor,
        indices: tuple[Value, ...],
        ctx: _FnCtx,
        elem_bytes: int,
    ) -> str | None:
        """Global-memory dynamic offset: widened to u64 for pointer math."""
        terms: list[str] = []
        strides = tensor.stride
        for i, idx in enumerate(indices):
            if _is_const_value(idx):
                continue
            # Widen the u32 index to u64, multiply by stride*elem_bytes.
            wide = ctx.regs.declare("u64")
            ctx.emit(f"cvt.u64.u32 {wide}, {ctx.regs.name_for(idx)};")
            factor = strides[i] * elem_bytes
            term = ctx.regs.declare("u64")
            ctx.emit(f"mul.lo.u64 {term}, {wide}, {factor};")
            terms.append(term)
        # dyn_row/col offsets
        if tensor.dyn_row_offset is not None:
            wide = ctx.regs.declare("u64")
            ctx.emit(f"cvt.u64.u32 {wide}, {ctx.regs.name_for(tensor.dyn_row_offset)};")
            term = ctx.regs.declare("u64")
            ctx.emit(f"mul.lo.u64 {term}, {wide}, {strides[0] * elem_bytes};")
            terms.append(term)
        if tensor.dyn_col_offset is not None:
            wide = ctx.regs.declare("u64")
            ctx.emit(f"cvt.u64.u32 {wide}, {ctx.regs.name_for(tensor.dyn_col_offset)};")
            term = ctx.regs.declare("u64")
            ctx.emit(f"mul.lo.u64 {term}, {wide}, {elem_bytes};")
            terms.append(term)
        if not terms:
            return None
        acc = terms[0]
        for t in terms[1:]:
            nxt = ctx.regs.declare("u64")
            ctx.emit(f"add.u64 {nxt}, {acc}, {t};")
            acc = nxt
        return acc


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH: dict[type, Any] = {
    ConstOp: PtxLowerer._visit_const,
    ArithOp: PtxLowerer._visit_arith,
    MathOp: PtxLowerer._visit_math,
    CmpOp: PtxLowerer._visit_cmp,
    SelectOp: PtxLowerer._visit_select,
    ConvertOp: PtxLowerer._visit_convert,
    PackedConvertOp: PtxLowerer._visit_packed_convert,
    UnpackedConvertOp: PtxLowerer._visit_unpacked_convert,
    BitcastOp: PtxLowerer._visit_bitcast,
    VecBuildOp: PtxLowerer._visit_vec_build,
    VecExtractOp: PtxLowerer._visit_vec_extract,
    SplitB32Op: PtxLowerer._visit_split_b32,
    MergeB32Op: PtxLowerer._visit_merge_b32,
    ShuffleOp: PtxLowerer._visit_shuffle,
    SubgroupReduceOp: PtxLowerer._visit_subgroup_reduce,
    SubgroupBroadcastOp: PtxLowerer._visit_subgroup_broadcast,
    ThreadIdxOp: PtxLowerer._visit_thread_idx,
    BlockIdxOp: PtxLowerer._visit_block_idx,
    BlockDimOp: PtxLowerer._visit_block_dim,
    GridDimOp: PtxLowerer._visit_grid_dim,
    LaneIdOp: PtxLowerer._visit_lane_id,
    SubgroupIdOp: PtxLowerer._visit_subgroup_id,
    GroupIdOp: PtxLowerer._visit_group_id,
    ThreadIdInGroupOp: PtxLowerer._visit_thread_id_in_group,
    BarrierOp: PtxLowerer._visit_barrier,
    ForLoopOp: PtxLowerer._visit_for_loop,
    IfRegionOp: PtxLowerer._visit_if_region,
    YieldOp: PtxLowerer._visit_yield,
    SmemAllocOp: PtxLowerer._visit_smem_alloc,
    LoadOp: PtxLowerer._visit_load,
    StoreOp: PtxLowerer._visit_store,
    VecLoadOp: PtxLowerer._visit_vec_load,
    VecStoreOp: PtxLowerer._visit_vec_store,
    AsyncCopyOp: PtxLowerer._visit_async_copy,
    AsyncCopyCommitOp: PtxLowerer._visit_async_commit,
    AsyncCopyWaitOp: PtxLowerer._visit_async_wait,
    AtomicRmwOp: PtxLowerer._visit_atomic_rmw,
    MmaOp: PtxLowerer._visit_mma,
    FragApplyOp: PtxLowerer._visit_frag_apply,
    FragConvertOp: PtxLowerer._visit_frag_convert,
    FragForEachOp: PtxLowerer._visit_frag_for_each,
    FragReduceOp: PtxLowerer._visit_frag_reduce,
    LoadMatrixOp: PtxLowerer._visit_load_matrix,
    StoreMatrixOp: PtxLowerer._visit_store_matrix,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ARITH_MNEMONIC: dict[str, str] = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "min": "min",
    "max": "max",
    "div": "div",
    "rem": "rem",
    "fma": "fma",
    "neg": "neg",
    "abs": "abs",
    # Bitwise / shifts handled separately in _visit_arith.
    "and": "and",
    "or": "or",
    "xor": "xor",
    "shl": "shl",
    "shr": "shr",
}

_MATH_MNEMONIC: dict[str, str] = {
    "rcp": "rcp",
    "rsqrt": "rsqrt",
    "sqrt": "sqrt",
    "exp2": "ex2",
    "log2": "lg2",
    "sin": "sin.approx",
    "cos": "cos.approx",
    "tanh": "tanh",
    "ex2_approx": "ex2.approx",
    "rcp_approx": "rcp.approx",
    "rsqrt_approx": "rsqrt.approx",
    "log2_approx": "lg2.approx",
    "sqrt_approx": "sqrt.approx",
}


def _emit_value_mov(ctx: _FnCtx, dst: Value, src: Value) -> None:
    """Emit mov(s) from `src` into `dst`, handling both scalar and
    vec Values correctly.

    PTX `mov.T` only supports scalar-to-scalar. For width-N vec
    Values we emit N per-component movs. Skips self-moves (same
    component reg name) which arise from aliased carried values.
    """
    dst_comps = ctx.regs.components(dst)
    src_comps = ctx.regs.components(src)
    if len(dst_comps) != len(src_comps):
        raise ValueError(
            f"_emit_value_mov: width mismatch dst={len(dst_comps)} vs src={len(src_comps)}"
        )
    cls = reg_class(dst.dtype)
    for d, s in zip(dst_comps, src_comps, strict=False):
        if d == s:
            continue  # aliased — no self-mov
        ctx.emit(f"mov.{cls} {d}, {s};")


def _collect_body_local_value_ids(ops: list[Op]) -> list[int]:
    """Return every Value.id produced by ops in this region (and nested
    regions). Used by the per-slot re-walker in ``_visit_frag_apply`` to
    wipe stale name bindings so each walk freshly allocates registers.
    Free variables defined OUTSIDE the region aren't in this list, so
    their bindings survive across walks.
    """
    ids: list[int] = []
    for op in ops:
        for v in op.results:
            ids.append(v.id)
        for r in op.regions:
            ids.extend(_collect_body_local_value_ids(r.ops))
    return ids


def _align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


# Pick the canonical PTX (vec_width, reg_dtype) form for a vec_load /
# vec_store of `width` elements of `dtype`. Bytes are bytes — for a
# 16-byte transfer requested as v8.b16 we lower to a single `ld.v4.b32`
# (4 b32 regs holding 8 packed b16 values), not two `ld.v4.b16`. The IR
# Value's components get bound to physical b32 regs at lowering time so
# downstream consumers (e.g. a paired vec_store) read the same regs.
#
# Returns (vec_width, reg_dtype) such that
#   vec_width * reg_dtype.bytes == width * dtype.bytes  (total bytes)
#   vec_width ∈ {1, 2, 4}                                (PTX legal)
# Falls through to None if the total bytes isn't a PTX legal size.
#
# Reg-dtype priority is `b32` first (widest universally-supported reg),
# then `b64` (for v2.b64 = 16B), then narrower. This keeps the common
# 8B/16B paths on a single instruction with no register-size surprises.
def _vec_phys_form(width: int, dtype: DType) -> tuple[int, DType] | None:
    if width <= 0:
        return None
    total = width * dtype.bytes
    if total not in (1, 2, 4, 8, 16):
        return None
    # If the requested form is ALREADY a legal single PTX vec instruction
    # (width ∈ {1, 2, 4}) AND the dtype has a real PTX reg class, keep it
    # as-is — preserves the caller's chosen dtype so reg classes / arith
    # uses downstream stay sane.
    #
    # Two reasons we *do* canonicalize otherwise:
    #   - width > 4: PTX has no v8 / v16 form. Pick the widest single-instr
    #     equivalent (e.g. v8.b16 → v4.b32 — bytes are bytes).
    #   - dtype has no reg class (E4M3 / E5M2): PTX `ld.global.<vw>.<class>`
    #     needs a registered reg class. Drop down to a byte-equivalent
    #     bit-typed class (E4M3x2 → B16, E4M3x4 → B32, etc.) so the load
    #     becomes a plain bit-width-only memory op.
    if width in (1, 2, 4) and dtype not in (DType.E4M3, DType.E5M2):
        return width, dtype
    # No `B8` here — single-byte transfers use U8 for the reg class.
    for reg_bytes, reg_dt in (
        (4, DType.B32),
        (8, DType.B64),
        (2, DType.B16),
        (1, DType.U8),
    ):
        if total % reg_bytes != 0:
            continue
        n = total // reg_bytes
        if n in (1, 2, 4):
            return n, reg_dt
    return None


def _bind_phys_vec(regs: Any, value: Any, v_width: int, reg_dt: DType) -> tuple[str, ...]:
    """Bind `value` to `v_width` freshly-allocated regs of `reg_dt`'s class.
    If already bound (paired vec_load → vec_store on the same Value), returns
    the existing components. Bypasses `RegAllocator.bind`'s width check —
    the IR Value's logical (width, dtype) may differ from the physical
    (v_width, reg_dt) used by PTX, since `bytes are bytes` for vec memory
    ops that don't introspect element values."""
    existing = regs._components.get(value.id)
    if existing is not None:
        return existing
    cls = reg_class(reg_dt)
    comps = tuple(regs.declare(cls) for _ in range(v_width))
    regs._components[value.id] = comps
    return comps


def _is_const_value(v: Value) -> bool:
    return isinstance(v.producer, ConstOp)


def _const_int(v: Value) -> int:
    assert isinstance(v.producer, ConstOp)
    return int(v.producer.attrs["value"])


def _cvt_suffix(dtype: DType) -> str:
    """PTX cvt instruction operand type string."""
    # cvt uses the dtype name directly for most types; bit-typed classes
    # use their native name (e.g. cvt.f32.b32 is legal where needed).
    return arith_suffix(dtype)


def _format_literal(dtype: DType, value: Any) -> str:
    """Format a Python literal for emission inside a `mov.<dtype>` immediate.

    F32 uses hex float encoding (0f...) for exact bit patterns; integers
    are emitted in decimal; PRED uses 0/1; bit-typed classes take
    integer-looking literals.
    """
    if dtype is DType.F32:
        import struct

        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return f"0f{bits:08x}"
    if dtype is DType.F64:
        import struct

        bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        return f"0d{bits:016x}"
    if dtype is DType.PRED:
        return "1" if bool(value) else "0"
    return str(int(value))
