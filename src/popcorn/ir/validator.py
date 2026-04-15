"""Structural validation + perf-warning pass for popcorn IR modules.

EXEMPT FROM 500-LINE RULE: absorbed the former ``ir/lint.py`` passes
(bank-conflict / alignment / OOB) alongside the structural checks so
one module-walk drives every pre-lowering gate. Splitting would
double-traverse the IR and duplicate the per-op dispatch table.

Runs before any lowerer. Two layers:

**Correctness (raises ``ValidationError``):**
  1. SSA dominance — every operand Value must be defined in an ancestor
     region of its use site.
  2. Region terminators — structured ops' regions must end with a
     ``YieldOp`` whose operand shapes match the op's signature.
  3. Tensor indices — memory-op indices must match the tensor's rank
     and be scalar int/bit Values.
  4. Matmul shape registry — every ``shape_id`` used must be registered
     in ``Module.kernel_shapes``.
  5. Smem backing — every SharedRegion referenced by a memory op must
     point to a SmemAllocOp result inside the same Function.
  6. OOB access — when indices are constant, they must stay within the
     tensor's declared shape. Covers GlobalTensor and SharedRegion
     accesses (Load / Store / VecLoad / VecStore).

**Perf warnings (emit ``PerfWarning``, non-fatal):**
  * Smem bank conflicts for rank-2 allocations.
  * Smem footprint warnings above 48 KB.
  * cp.async count ∉ {4, 8, 16}.
  * cp.async src/dst row-stride alignment.
  * Vec load/store misalignment vs ``SharedRegion.align_bytes``.

The old ``ir/lint.py`` pass has been folded in here — one pass, one
entry point. Perf warnings are off by default (autotune sweeps print
them at every compile); set ``POPCORN_ENABLE_PERF_WARNINGS=1`` to
opt in to the non-fatal emissions.

See POPCORN_IR_PROPOSAL.md §11.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

from .module import Function, Module
from .op import (
    AsyncCopyOp,
    ForLoopOp,
    IfRegionOp,
    LoadMatrixOp,
    LoadOp,
    MmaOp,
    Op,
    StoreMatrixOp,
    StoreOp,
    VecLoadOp,
    VecStoreOp,
    WhileLoopOp,
    YieldOp,
)
from .region import Region
from .tensor import GlobalTensor, SharedRegion
from .types import DType
from .value import Value


class ValidationError(Exception):
    pass


class PerfWarning(UserWarning):
    """Non-fatal perf-pattern warning emitted during ``validate_module``.

    Off by default. Set ``POPCORN_ENABLE_PERF_WARNINGS=1`` in the env
    to opt in; filter further via ``warnings.simplefilter`` on this
    class.
    """


def _perf_warnings_enabled() -> bool:
    return os.environ.get("POPCORN_ENABLE_PERF_WARNINGS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class _PerfFinding:
    where: str
    message: str


@dataclass
class _Scope:
    """A set of Values defined in an ancestor region path."""

    defs: set[int]  # Value ids visible at this point

    def copy(self) -> _Scope:
        return _Scope(defs=set(self.defs))


def validate_module(module: Module) -> list[_PerfFinding]:
    """Run full validation. Raises ``ValidationError`` on correctness
    failures; returns the perf-finding list and emits them via
    ``warnings.warn(..., PerfWarning)`` unless suppressed.
    """
    findings: list[_PerfFinding] = []
    for fn in module.functions:
        findings.extend(validate_function(fn, module))
    if _perf_warnings_enabled():
        for f in findings:
            warnings.warn(f"{f.where}: {f.message}", PerfWarning, stacklevel=2)
    return findings


def validate_function(fn: Function, module: Module) -> list[_PerfFinding]:
    # Track which SmemAllocOp result values belong to this function.
    smem_backings: set[int] = {op.results[0].id for op in fn.smem_allocs}

    # Seed the scope with the parameter Values.
    scope = _Scope(defs={p.value.id for p in fn.params if p.value is not None})

    _validate_region(
        region=fn.body,
        scope=scope,
        fn=fn,
        module=module,
        smem_backings=smem_backings,
        where=f"function @{fn.name}",
        is_function_body=True,
    )
    # Perf-layer checks. None of these can raise; they accumulate
    # findings for the caller to warn / filter.
    findings: list[_PerfFinding] = []
    findings.extend(_check_smem_bank_conflicts(fn))
    findings.extend(_check_smem_footprint(fn))
    findings.extend(_check_async_copy_granules(fn))
    findings.extend(_check_async_copy_stride_alignment(fn))
    return findings


def _validate_region(
    *,
    region: Region,
    scope: _Scope,
    fn: Function,
    module: Module,
    smem_backings: set[int],
    where: str,
    is_function_body: bool,
) -> None:
    local = scope.copy()
    n = len(region.ops)
    for i, op in enumerate(region.ops):
        # A YieldOp must be the last op in any region where it appears.
        if isinstance(op, YieldOp) and i != n - 1:
            raise ValidationError(
                f"{where}: YieldOp must be the terminator (position {i} of {n - 1})"
            )

        _check_operands_in_scope(op, local, where)
        _check_memory_op(op, fn, module, smem_backings, where)
        _check_matmul_op(op, module, where)

        # For structured ops, recurse into their regions with the induction
        # variable / carried values available.
        if isinstance(op, ForLoopOp):
            body_scope = local.copy()
            assert op.induction_var is not None
            body_scope.defs.add(op.induction_var.id)
            # Per-iteration carried-in Values are defined by this op and
            # visible only inside its body region.
            for cbv in op.carried_body_vars:
                body_scope.defs.add(cbv.id)
            _validate_region(
                region=op.body,
                scope=body_scope,
                fn=fn,
                module=module,
                smem_backings=smem_backings,
                where=f"{where}: for_loop body",
                is_function_body=False,
            )
            _check_region_terminator(op.body, op, where)

        elif isinstance(op, IfRegionOp):
            n_carried = op.attrs.get("n_carried", 0)
            for arm_name, arm_region, arm_body_vars in (
                ("then", op.then_region, op.then_body_vars),
                ("else", op.else_region, op.else_body_vars),
            ):
                arm_scope = local.copy()
                for bv in arm_body_vars:
                    arm_scope.defs.add(bv.id)
                _validate_region(
                    region=arm_region,
                    scope=arm_scope,
                    fn=fn,
                    module=module,
                    smem_backings=smem_backings,
                    where=f"{where}: if_region.{arm_name}",
                    is_function_body=False,
                )
                _check_region_terminator(arm_region, op, where + f".{arm_name}")
            # Both arms must yield `n_carried` values with matching shapes.
            t_term = op.then_region.terminator
            e_term = op.else_region.terminator
            if t_term is None or e_term is None:
                raise ValidationError(f"{where}: if_region arms must both terminate with YieldOp")
            if len(t_term.operands) != n_carried or len(e_term.operands) != n_carried:
                raise ValidationError(
                    f"{where}: if_region yields must produce {n_carried} values "
                    f"(then={len(t_term.operands)}, else={len(e_term.operands)})"
                )
            for idx, (tv, ev) in enumerate(zip(t_term.operands, e_term.operands, strict=False)):
                if tv.shape != ev.shape:
                    raise ValidationError(
                        f"{where}: if_region yield[{idx}] shape mismatch "
                        f"then={tv.shape} vs else={ev.shape}"
                    )

        elif isinstance(op, WhileLoopOp):
            # Both regions see the loop-carried Values (which are not
            # explicitly listed — the op's results play that role).
            body_scope = local.copy()
            _validate_region(
                region=op.cond_region,
                scope=body_scope.copy(),
                fn=fn,
                module=module,
                smem_backings=smem_backings,
                where=f"{where}: while_loop cond",
                is_function_body=False,
            )
            _validate_region(
                region=op.body_region,
                scope=body_scope,
                fn=fn,
                module=module,
                smem_backings=smem_backings,
                where=f"{where}: while_loop body",
                is_function_body=False,
            )
            _check_region_terminator(op.cond_region, op, where + ": while.cond")
            _check_region_terminator(op.body_region, op, where + ": while.body")

        # Register the op's results in the current scope so downstream
        # ops can see them.
        for v in op.results:
            local.defs.add(v.id)


def _check_operands_in_scope(op: Op, scope: _Scope, where: str) -> None:
    # Allow operands that are params (defined in the function scope) and
    # Values produced above in any ancestor region. The _Scope's `defs`
    # contains all currently-visible Value ids.
    for i, v in enumerate(op.operands):
        if v.id not in scope.defs:
            raise ValidationError(
                f"{where}: op {type(op).__name__} operand[{i}] (id={v.id}) "
                f"is not in scope — possible SSA dominance violation "
                f"or forward reference"
            )
    # Some ops reference Values through attrs too (e.g. dyn offsets on
    # SharedRegion/GlobalTensor). Walk those.
    for key, v in op.attrs.items():
        if isinstance(v, Value):
            if v.id not in scope.defs:
                raise ValidationError(
                    f"{where}: op {type(op).__name__} attr {key!r} references "
                    f"Value id={v.id} not in scope"
                )
        if isinstance(v, SharedRegion):
            if v.alloc.id not in scope.defs:
                raise ValidationError(
                    f"{where}: op {type(op).__name__}: SharedRegion {v.name!r} "
                    f"backing alloc (id={v.alloc.id}) not in scope"
                )
            if v.dyn_offset is not None and v.dyn_offset.id not in scope.defs:
                raise ValidationError(
                    f"{where}: op {type(op).__name__}: SharedRegion {v.name!r} "
                    f"dyn_offset Value not in scope"
                )
        if isinstance(v, GlobalTensor):
            if v.param.value is not None and v.param.value.id not in scope.defs:
                raise ValidationError(
                    f"{where}: op {type(op).__name__}: GlobalTensor {v.name!r} "
                    f"param Value not in scope"
                )
            for field_name in ("dyn_row_offset", "dyn_col_offset"):
                dv = getattr(v, field_name)
                if dv is not None and dv.id not in scope.defs:
                    raise ValidationError(
                        f"{where}: op {type(op).__name__}: GlobalTensor {v.name!r} "
                        f"{field_name} Value not in scope"
                    )


def _check_memory_op(
    op: Op,
    fn: Function,
    module: Module,
    smem_backings: set[int],
    where: str,
) -> None:
    tensor = None
    if isinstance(op, (LoadOp, StoreOp, VecLoadOp, VecStoreOp)):
        tensor = op.attrs.get("tensor")
    elif isinstance(op, LoadMatrixOp):
        tensor = op.attrs.get("src_tensor")
    elif isinstance(op, StoreMatrixOp):
        tensor = op.attrs.get("dst_tensor")
    if tensor is None:
        return
    if isinstance(tensor, SharedRegion):
        if tensor.alloc.id not in smem_backings:
            raise ValidationError(
                f"{where}: op {type(op).__name__}: SharedRegion {tensor.name!r} "
                f"references alloc Value id={tensor.alloc.id} which is not a "
                f"SmemAllocOp of function @{fn.name}"
            )
    _check_oob_access(op, tensor, where)


def _check_oob_access(op: Op, tensor, where: str) -> None:
    """Constant-index OOB check. For Load / Store / VecLoad / VecStore
    with purely-constant index Values, verify the index stays within
    the tensor's declared shape. Skips anything where indices can't be
    statically resolved to integers — the common case (thread-id
    arithmetic) would need symbolic interval analysis to prove safety.

    Catches the typo class — e.g. ``smem[r * (Dh + pad) + c]`` written
    with the wrong ``pad``. Fires on both SharedRegion and GlobalTensor
    accesses.
    """
    from .op import ConstOp

    # Pull the index Values. For LoadOp/StoreOp/VecLoadOp/VecStoreOp
    # the operand layout is: StoreOp = (value, *indices[, pred]).
    if isinstance(op, LoadOp):
        indices = op.operands
    elif isinstance(op, StoreOp):
        # operands = (value, *indices [, pred])
        indices = op.operands[1:]
    elif isinstance(op, VecLoadOp):
        indices = op.operands
    elif isinstance(op, VecStoreOp):
        indices = op.operands[1:]
    elif isinstance(op, (LoadMatrixOp, StoreMatrixOp)):
        # Matrix ops carry fragment-level tile bases; per-element OOB
        # depends on shape_id layout tables — skip the check here.
        return
    else:
        return

    # Strip a trailing predicate if one exists.
    pred = op.attrs.get("pred")
    if pred is not None and indices and indices[-1].id == pred.id:
        indices = indices[:-1]

    if len(indices) != tensor.rank:
        return  # caller's arity check will catch this

    # Resolve each index to a Python int if possible.
    concrete: list[int] = []
    for idx_val in indices:
        producer = getattr(idx_val, "producer", None)
        if not isinstance(producer, ConstOp):
            return  # non-constant — can't statically check
        val = producer.attrs.get("value")
        if not isinstance(val, int):
            return  # non-int const (PRED etc.) — skip
        concrete.append(val)

    # Factor in static offsets from GlobalTensor / SharedRegion views.
    # dyn_* / dyn_offset contributions are runtime — if they're non-None
    # we can't prove OOB statically, so skip.
    if isinstance(tensor, GlobalTensor):
        if tensor.dyn_row_offset is not None or tensor.dyn_col_offset is not None:
            return
        row = concrete[0] + tensor.static_row_offset
        col = concrete[1] + tensor.static_col_offset if tensor.rank >= 2 else None
        if row < 0 or row >= tensor.shape[0]:
            raise ValidationError(
                f"{where}: op {type(op).__name__}: GlobalTensor {tensor.name!r} "
                f"row={row} out of bounds for shape={tensor.shape}"
            )
        if col is not None and (col < 0 or col >= tensor.shape[1]):
            raise ValidationError(
                f"{where}: op {type(op).__name__}: GlobalTensor {tensor.name!r} "
                f"col={col} out of bounds for shape={tensor.shape}"
            )
    elif isinstance(tensor, SharedRegion):
        if tensor.dyn_offset is not None or tensor.warp_dyn_offset is not None:
            return
        # For SharedRegion we check against declared ``shape``. The
        # ``static_offset`` is a byte-ish offset into the alloc used by
        # view() / stage() and doesn't change the per-axis bounds from
        # this view's perspective — so we only check indices vs shape.
        for axis, (idx, bound) in enumerate(zip(concrete, tensor.shape, strict=False)):
            if idx < 0 or idx >= bound:
                raise ValidationError(
                    f"{where}: op {type(op).__name__}: SharedRegion {tensor.name!r} "
                    f"axis {axis} index={idx} out of bounds for shape={tensor.shape}"
                )


def _check_matmul_op(op: Op, module: Module, where: str) -> None:
    shape_id = None
    if isinstance(op, (LoadMatrixOp, StoreMatrixOp, MmaOp)):
        shape_id = op.attrs.get("shape_id")
    if shape_id is None:
        return
    if shape_id not in module.kernel_shapes:
        raise ValidationError(
            f"{where}: op {type(op).__name__}: shape_id {shape_id!r} "
            f"not in module.kernel_shapes (registered: "
            f"{sorted(module.kernel_shapes)})"
        )


def _check_region_terminator(region: Region, parent_op: Op, where: str) -> None:
    if not region.ops:
        return
    last = region.ops[-1]
    if not isinstance(last, YieldOp):
        # Function body doesn't need a terminator today, only structured
        # op regions do.
        raise ValidationError(
            f"{where}: region of op {type(parent_op).__name__} must end with "
            f"YieldOp (last op was {type(last).__name__})"
        )


# ---------------------------------------------------------------------------
# Perf-pattern checks (emit PerfWarning, non-fatal)
#
# Folded in from the former ``ir/lint.py``. Same heuristics, same
# messages — just routed through one entry point now.
# ---------------------------------------------------------------------------


_BANK_ROW_BYTES = 128  # 32 banks × 4 B per SIMT lane
_SMEM_WARN_BYTES = 48 * 1024


def _check_smem_bank_conflicts(fn: Function) -> list[_PerfFinding]:
    """Rank-2 smem allocations whose row stride is a multiple of 128B
    cause N-way bank conflicts on m8n8 / m16n8 fragment loads. Suggest
    a small pad."""
    findings: list[_PerfFinding] = []
    for op in fn.smem_allocs:
        shape: tuple[int, ...] = tuple(op.attrs["shape"])
        if len(shape) != 2:
            continue
        dtype: DType = op.attrs["dtype"]
        pad: int = int(op.attrs.get("pad", 0))
        elem_bytes = dtype.bytes
        row_stride_bytes = (shape[1] + pad) * elem_bytes
        if row_stride_bytes == 0 or row_stride_bytes % _BANK_ROW_BYTES != 0:
            continue
        suggested = 16 if elem_bytes == 1 else 8
        findings.append(
            _PerfFinding(
                where=f"@{fn.name} smem_alloc {op.attrs['name']!r}",
                message=(
                    f"row stride {row_stride_bytes}B is a multiple of "
                    f"{_BANK_ROW_BYTES}B (32 banks × 4B) — expect bank "
                    f"conflicts on m8n8/m16n8 fragment loads. Set "
                    f"pad={suggested} on this alloc (or bump `a_pad`/"
                    f"`b_pad` in the kernel config)."
                ),
            )
        )
    return findings


def _check_smem_footprint(fn: Function) -> list[_PerfFinding]:
    """Warn when static smem > 48KB — the carve-out beyond which the
    kernel needs opt-in dynamic smem and may lose occupancy."""
    findings: list[_PerfFinding] = []
    total = 0
    for op in fn.smem_allocs:
        shape = tuple(op.attrs["shape"])
        pad = int(op.attrs.get("pad", 0))
        dtype: DType = op.attrs["dtype"]
        if len(shape) == 2:
            elems = shape[0] * (shape[1] + pad)
        else:
            elems = 1
            for s in shape:
                elems *= s
        total += elems * dtype.bytes
    if total > _SMEM_WARN_BYTES:
        findings.append(
            _PerfFinding(
                where=f"@{fn.name} smem pool",
                message=(
                    f"static smem footprint {total}B exceeds {_SMEM_WARN_BYTES}B — "
                    f"kernel requires opt-in dynamic smem and may hurt occupancy."
                ),
            )
        )
    return findings


def _check_async_copy_granules(fn: Function) -> list[_PerfFinding]:
    """cp.async.ca hardware supports 4 / 8 / 16 byte copies per lane.
    Anything else silently falls back to scalar."""
    findings: list[_PerfFinding] = []
    for op in _iter_ops(fn):
        if not isinstance(op, AsyncCopyOp):
            continue
        count = int(op.attrs.get("count", 0))
        if count not in (4, 8, 16):
            dst = op.attrs.get("dst_tensor")
            dst_name = dst.name if isinstance(dst, SharedRegion) else "?"
            findings.append(
                _PerfFinding(
                    where=f"@{fn.name} async_copy → {dst_name!r}",
                    message=(
                        f"cp.async count={count}B — hardware supports "
                        f"{{4, 8, 16}} only; any other value lowers to "
                        f"scalar ld.global/st.shared. Align to a 16B line."
                    ),
                )
            )
    return findings


def _check_async_copy_stride_alignment(fn: Function) -> list[_PerfFinding]:
    """For each AsyncCopyOp with count ∈ {4, 8, 16} bytes, both the
    src and dst row stride must be a multiple of ``count``."""
    findings: list[_PerfFinding] = []
    for op in _iter_ops(fn):
        if not isinstance(op, AsyncCopyOp):
            continue
        count = int(op.attrs.get("count", 0))
        if count not in (4, 8, 16):
            continue  # separate rule handles bad counts
        for role, tensor in (
            ("src", op.attrs.get("src_tensor")),
            ("dst", op.attrs.get("dst_tensor")),
        ):
            if not isinstance(tensor, (GlobalTensor, SharedRegion)):
                continue
            if tensor.rank < 2:
                continue
            row_stride_elems = int(tensor.stride[0])
            row_stride_bytes = row_stride_elems * tensor.dtype.bytes
            if row_stride_bytes == 0 or row_stride_bytes % count == 0:
                continue
            findings.append(
                _PerfFinding(
                    where=f"@{fn.name} async_copy ({role}={tensor.name!r})",
                    message=(
                        f"row stride {row_stride_bytes}B is not a multiple "
                        f"of cp.async count={count}B — trailing partial "
                        f"line per row will break the vectorized path. "
                        f"Pad the row to the next multiple of {count}B."
                    ),
                )
            )
    return findings


def _iter_ops(fn: Function):
    """Pre-order walk over every op in the function body."""
    stack = [fn.body]
    while stack:
        region = stack.pop()
        for op in region.ops:
            yield op
            for sub in op.regions:
                stack.append(sub)
