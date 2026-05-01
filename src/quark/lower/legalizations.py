"""Concrete legalization rewrites registered against
:mod:`quark.lower.legalize`.

EXEMPT FROM 500-LINE RULE: each rewrite pairs a small
``_legalize_*`` dispatcher with a much larger ``_expand_*`` body,
and the two need to live together — splitting the registry from
the bodies forces every rewrite to carry its own module +
cross-import + its own test-registration order. The current file
reads top-to-bottom as "one rewrite per section," which is the
shape phase-2 work has converged on. Target size after every
body is real: ~700 lines (bf16x2 arith ~170, vector-atomic ~100,
async_copy ~90, subgroup_reduce butterfly ~180 when migrated).
The 800-line hard cap still applies; split before then.

Phase 2.2 scope (see PORTABILITY_PLAN.md §2.2): claim ownership of
every op type whose lowering is backend-conditional, with a correct
keep-path on every currently-wired backend and a surface-level
``NotImplementedError`` when a future backend lacking the primitive
actually reaches the expansion path.

The five rewrites:

  * :class:`ArithOp` ``kind="fma_bf16x2"``      — keep if ``caps.has_fma_bf16x2``.
  * :class:`ArithOp` ``kind="cvt_rn_bf16x2_f32"`` — keep if ``caps.has_fma_bf16x2``.
  * :class:`AtomicRmwOp` ``atomic_type in {"bf16x2","f16x2"}`` — keep
    if the matching ``(dtype, 2)`` tuple is in ``caps.atomic_add_vector``.
  * :class:`AsyncCopyOp` / commit / wait — keep if ``caps.supports_async_copy``.
  * :class:`SubgroupReduceOp` — currently always keep (PTX inlines the
    butterfly expansion in the lowerer; moving it here is a separate
    migration tracked in the plan).

Currently-wired backends (CUDA + Metal):

  * CUDA sm_80+: every rewrite's cap is ``True`` on sm_80+ (bf16x2
    compute / async_copy); the vector-atomic rewrite's cap is
    ``True`` on sm_90+. Kernels emitted on older CUDA never reach
    the expansion branch because ``Kernel.is_valid_for`` rejects
    them upstream.
  * Metal: the legacy kernels never emit ``AsyncCopyOp`` /
    ``fma_bf16x2`` / vector-atomic ``AtomicRmwOp`` — so these
    rewrites don't fire on Metal today. When a future kernel does
    emit one on Metal, that's the point at which the expansion
    body needs a real implementation; the ``NotImplementedError``
    raised here points the maintainer at this file.

Future SPIR-V (plan §3.1): all four compute/atomic caps are
``False``. The rewrites will fire; the expansion bodies need real
implementations by then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quark.ir.op import (
    ArithOp,
    AsyncCopyCommitOp,
    AsyncCopyOp,
    AsyncCopyWaitOp,
    AtomicRmwOp,
    SubgroupReduceOp,
)
from quark.lower.legalize import register_legalization

if TYPE_CHECKING:
    from quark.ir.op import Op


def _expansion_todo(op_desc: str, backend_hint: str) -> list[Op]:
    """Raise a ``NotImplementedError`` with a pointed message.

    The legalization driver calls this only when a rewrite's cap is
    ``False`` — i.e., we're on a backend that genuinely needs the
    expansion. Every wired backend today either has the cap set
    ``True`` on the generations its kernels run on, or never emits
    the op in the first place, so this function is unreached on
    trunk. When SPIR-V (or another new backend) lands and first
    trips one of these, the error message names the op and the
    file to edit.
    """
    raise NotImplementedError(
        f"legalize: expansion for {op_desc} is not implemented yet. "
        f"This fires on backends where {backend_hint}. "
        f"Add the rewrite body in quark/lower/legalizations.py."
    )


# -----------------------------------------------------------------------------
# bf16x2 compute ops — both gated on ``caps.has_fma_bf16x2`` (CUDA sm_80+).
# -----------------------------------------------------------------------------


@register_legalization(ArithOp)
def _legalize_bf16x2_arith(op: Op, caps: Any) -> list[Op] | None:
    """Keep the op on backends with native bf16x2 arithmetic; expand
    otherwise.

    Handles ``kind="fma_bf16x2"`` (packed B32×3 → B32 FMA) and
    ``kind="cvt_rn_bf16x2_f32"`` (F32×2 → packed B32). Every other
    ArithOp kind passes through — this rewrite only claims the
    bf16x2 family.
    """
    kind = op.attrs.get("kind")
    if kind not in ("fma_bf16x2", "cvt_rn_bf16x2_f32"):
        return None  # some other ArithOp kind — not ours
    if getattr(caps, "has_fma_bf16x2", False):
        return None  # backend emits the op natively
    if kind == "cvt_rn_bf16x2_f32":
        return _expand_cvt_rn_bf16x2_f32(op)
    if kind == "fma_bf16x2":
        return _expand_fma_bf16x2(op)
    # Every ArithOp kind this rewrite claims is handled above; reaching
    # here is a registry/kind-list mismatch bug.
    _expansion_todo(
        f"ArithOp(kind={kind!r})",
        "caps.has_fma_bf16x2 is False and no expansion body is wired up",
    )
    return None  # unreachable — for the type checker


def _expand_cvt_rn_bf16x2_f32(op: Op) -> list[Op]:
    """Expand ``cvt_rn_bf16x2_f32(f32_a, f32_b) -> b32`` into a
    Convert+Bitcast+Merge chain that preserves the original result
    Value (downstream consumers' operand tuples stay valid).

    Chain:
        bf16_a = convert(f32_a, BF16)       # ConvertOp
        bf16_b = convert(f32_b, BF16)
        b16_a  = bitcast(bf16_a, B16)       # BitcastOp — same bits
        b16_b  = bitcast(bf16_b, B16)
        out    = merge_b32(b16_a, b16_b)    # MergeB32Op
    """
    from quark.ir import DType
    from quark.ir.op import BitcastOp, ConvertOp, MergeB32Op
    from quark.ir.types import ValueShape
    from quark.lower.legalize import Rewriter

    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError(
            "_expand_cvt_rn_bf16x2_f32: expected 2 operands and 1 result, "
            f"got {len(op.operands)} / {len(op.results)}"
        )
    f32_a, f32_b = op.operands
    (out_b32,) = op.results  # reuse in the final MergeB32Op

    rw = Rewriter.for_op(op)
    bf16_a = rw.alloc(ValueShape(DType.BF16))
    bf16_b = rw.alloc(ValueShape(DType.BF16))
    b16_a = rw.alloc(ValueShape(DType.B16))
    b16_b = rw.alloc(ValueShape(DType.B16))

    convert_attrs = {"rounding": "rn"}
    return [
        ConvertOp(
            results=(bf16_a,),
            operands=(f32_a,),
            attrs={"src_dtype": DType.F32, "dst_dtype": DType.BF16, **convert_attrs},
        ),
        ConvertOp(
            results=(bf16_b,),
            operands=(f32_b,),
            attrs={"src_dtype": DType.F32, "dst_dtype": DType.BF16, **convert_attrs},
        ),
        BitcastOp(results=(b16_a,), operands=(bf16_a,), attrs={"dst_dtype": DType.B16}),
        BitcastOp(results=(b16_b,), operands=(bf16_b,), attrs={"dst_dtype": DType.B16}),
        MergeB32Op(results=(out_b32,), operands=(b16_a, b16_b)),
    ]


def _expand_fma_bf16x2(op: Op) -> list[Op]:
    """Expand ``fma_bf16x2(a, b, c) -> d`` (each operand a packed B32
    holding bf16×2) into a two-lane F32 FMA chain.

    For each lane (lo, hi):
        split each B32 → (b16_lo, b16_hi)
        bitcast each b16 → bf16
        convert each bf16 → f32
        f32_d_lane = fma(f32_a_lane, f32_b_lane, f32_c_lane)
        convert f32_d_lane → bf16
        bitcast bf16 → b16
    merge (d_lo_b16, d_hi_b16) → B32 output (reuses original result Value)

    22 ops replace 1. The ``MergeB32Op`` at the end reuses the original
    fma_bf16x2's result Value so downstream consumers stay linked.
    """
    from quark.ir import DType
    from quark.ir.op import ArithOp, BitcastOp, ConvertOp, MergeB32Op, SplitB32Op
    from quark.ir.types import ValueShape
    from quark.lower.legalize import Rewriter

    if len(op.operands) != 3 or len(op.results) != 1:
        raise ValueError(
            "_expand_fma_bf16x2: expected 3 operands and 1 result, "
            f"got {len(op.operands)} / {len(op.results)}"
        )
    a_b32, b_b32, c_b32 = op.operands
    (out_b32,) = op.results

    rw = Rewriter.for_op(op)

    # Step 1: split each B32 → (lo_b16, hi_b16). 3 splits, 6 intermediate Values.
    a_lo_b16 = rw.alloc(ValueShape(DType.B16))
    a_hi_b16 = rw.alloc(ValueShape(DType.B16))
    b_lo_b16 = rw.alloc(ValueShape(DType.B16))
    b_hi_b16 = rw.alloc(ValueShape(DType.B16))
    c_lo_b16 = rw.alloc(ValueShape(DType.B16))
    c_hi_b16 = rw.alloc(ValueShape(DType.B16))

    # Step 2: bitcast each B16 → BF16. 6 bitcasts.
    a_lo_bf = rw.alloc(ValueShape(DType.BF16))
    a_hi_bf = rw.alloc(ValueShape(DType.BF16))
    b_lo_bf = rw.alloc(ValueShape(DType.BF16))
    b_hi_bf = rw.alloc(ValueShape(DType.BF16))
    c_lo_bf = rw.alloc(ValueShape(DType.BF16))
    c_hi_bf = rw.alloc(ValueShape(DType.BF16))

    # Step 3: convert BF16 → F32. 6 converts.
    a_lo_f = rw.alloc(ValueShape(DType.F32))
    a_hi_f = rw.alloc(ValueShape(DType.F32))
    b_lo_f = rw.alloc(ValueShape(DType.F32))
    b_hi_f = rw.alloc(ValueShape(DType.F32))
    c_lo_f = rw.alloc(ValueShape(DType.F32))
    c_hi_f = rw.alloc(ValueShape(DType.F32))

    # Step 4: per-lane F32 FMAs. 2 arith ops.
    d_lo_f = rw.alloc(ValueShape(DType.F32))
    d_hi_f = rw.alloc(ValueShape(DType.F32))

    # Step 5: convert F32 results → BF16. 2 converts.
    d_lo_bf = rw.alloc(ValueShape(DType.BF16))
    d_hi_bf = rw.alloc(ValueShape(DType.BF16))

    # Step 6: bitcast BF16 → B16. 2 bitcasts.
    d_lo_b16 = rw.alloc(ValueShape(DType.B16))
    d_hi_b16 = rw.alloc(ValueShape(DType.B16))

    cvt_rn = {"rounding": "rn"}

    def _convert(out_v, in_v, src, dst):
        return ConvertOp(
            results=(out_v,),
            operands=(in_v,),
            attrs={"src_dtype": src, "dst_dtype": dst, **cvt_rn},
        )

    def _bitcast(out_v, in_v, dst_dtype):
        return BitcastOp(results=(out_v,), operands=(in_v,), attrs={"dst_dtype": dst_dtype})

    def _fma_f32(out_v, x, y, z):
        return ArithOp(results=(out_v,), operands=(x, y, z), attrs={"kind": "fma"})

    return [
        # Splits.
        SplitB32Op(results=(a_lo_b16, a_hi_b16), operands=(a_b32,)),
        SplitB32Op(results=(b_lo_b16, b_hi_b16), operands=(b_b32,)),
        SplitB32Op(results=(c_lo_b16, c_hi_b16), operands=(c_b32,)),
        # B16 → BF16 bitcasts.
        _bitcast(a_lo_bf, a_lo_b16, DType.BF16),
        _bitcast(a_hi_bf, a_hi_b16, DType.BF16),
        _bitcast(b_lo_bf, b_lo_b16, DType.BF16),
        _bitcast(b_hi_bf, b_hi_b16, DType.BF16),
        _bitcast(c_lo_bf, c_lo_b16, DType.BF16),
        _bitcast(c_hi_bf, c_hi_b16, DType.BF16),
        # BF16 → F32 converts.
        _convert(a_lo_f, a_lo_bf, DType.BF16, DType.F32),
        _convert(a_hi_f, a_hi_bf, DType.BF16, DType.F32),
        _convert(b_lo_f, b_lo_bf, DType.BF16, DType.F32),
        _convert(b_hi_f, b_hi_bf, DType.BF16, DType.F32),
        _convert(c_lo_f, c_lo_bf, DType.BF16, DType.F32),
        _convert(c_hi_f, c_hi_bf, DType.BF16, DType.F32),
        # Per-lane F32 FMAs.
        _fma_f32(d_lo_f, a_lo_f, b_lo_f, c_lo_f),
        _fma_f32(d_hi_f, a_hi_f, b_hi_f, c_hi_f),
        # F32 → BF16 converts.
        _convert(d_lo_bf, d_lo_f, DType.F32, DType.BF16),
        _convert(d_hi_bf, d_hi_f, DType.F32, DType.BF16),
        # BF16 → B16 bitcasts.
        _bitcast(d_lo_b16, d_lo_bf, DType.B16),
        _bitcast(d_hi_b16, d_hi_bf, DType.B16),
        # Merge → original result Value.
        MergeB32Op(results=(out_b32,), operands=(d_lo_b16, d_hi_b16)),
    ]


# -----------------------------------------------------------------------------
# Vector atomic add — gated on ``caps.atomic_add_vector`` membership.
# -----------------------------------------------------------------------------


@register_legalization(AtomicRmwOp)
def _legalize_vector_atomic(op: Op, caps: Any) -> list[Op] | None:
    """Keep the vector-atomic op when the hardware supports the
    packed form; expand to two scalar atomics on adjacent columns
    otherwise.

    Only claims AtomicRmwOps with ``atomic_type`` set to a vector
    string (``"bf16x2"`` / ``"f16x2"``). Scalar atomics pass through."""
    atomic_type = op.attrs.get("atomic_type")
    if atomic_type not in ("bf16x2", "f16x2"):
        return None
    # Map atomic_type → (DType, lanes) tuple used by the caps set.
    from quark.ir import DType

    elem = DType.BF16 if atomic_type == "bf16x2" else DType.F16
    caps_vec = getattr(caps, "atomic_add_vector", frozenset())
    if (elem, 2) in caps_vec:
        return None  # native packed atomic available
    return _expand_vector_atomic(op, elem)


def _expand_vector_atomic(op: Op, elem_dtype: Any) -> list[Op]:
    """Expand ``atomic_rmw(dst, "add", packed_b32, ..., col, atomic_type="bf16x2")``
    into two scalar atomics on adjacent columns.

    Chain (bf16x2 form; f16x2 identical with BF16 → F16):

        lo_b16, hi_b16 = split_b32(packed_b32)
        lo_elem = bitcast(lo_b16, elem_dtype)     # BF16 / F16
        hi_elem = bitcast(hi_b16, elem_dtype)
        one = const(1, U32)
        col_plus_1 = add(col, one)
        _ = atomic_rmw(dst, "add", lo_elem, ..., col)        # no atomic_type
        _ = atomic_rmw(dst, "add", hi_elem, ..., col_plus_1)

    The original vector atomic's result is unused (red.add-style atomics
    have no writeback — see the PTX lowerer comment on the packed
    path), so this expansion produces two scalar AtomicRmwOps without
    preserving a result Value. Downstream consumers of the vector
    atomic don't exist; the orphan result Value just gets GC'd.
    """
    from quark.ir import DType
    from quark.ir.op import ArithOp, BitcastOp, ConstOp, SplitB32Op
    from quark.ir.op import AtomicRmwOp as _Atomic
    from quark.ir.types import ValueShape
    from quark.lower.legalize import Rewriter

    if len(op.operands) < 2:
        raise ValueError(
            "_expand_vector_atomic: expected at least 2 operands (value + 1 index), "
            f"got {len(op.operands)}"
        )
    packed = op.operands[0]
    indices = op.operands[1:]
    col = indices[-1]  # last index is always the column axis for vector atomics

    tensor = op.attrs["tensor"]
    atomic_op_name = op.attrs["op"]  # "add"

    rw = Rewriter.for_op(op)
    lo_b16 = rw.alloc(ValueShape(DType.B16))
    hi_b16 = rw.alloc(ValueShape(DType.B16))
    lo_elem = rw.alloc(ValueShape(elem_dtype))
    hi_elem = rw.alloc(ValueShape(elem_dtype))
    one_v = rw.alloc(ValueShape(DType.U32))
    col_plus_1 = rw.alloc(ValueShape(DType.U32))
    # Unused scalar-atomic results — one per scalar atomic. Kept so the
    # AtomicRmwOp constructor's `results` slot is satisfied.
    old_lo = rw.alloc(ValueShape(elem_dtype))
    old_hi = rw.alloc(ValueShape(elem_dtype))

    scalar_attrs: dict = {"tensor": tensor, "op": atomic_op_name}
    return [
        SplitB32Op(results=(lo_b16, hi_b16), operands=(packed,)),
        BitcastOp(results=(lo_elem,), operands=(lo_b16,), attrs={"dst_dtype": elem_dtype}),
        BitcastOp(results=(hi_elem,), operands=(hi_b16,), attrs={"dst_dtype": elem_dtype}),
        ConstOp(results=(one_v,), operands=(), attrs={"value": 1, "dtype": DType.U32}),
        ArithOp(
            results=(col_plus_1,),
            operands=(col, one_v),
            attrs={"kind": "add"},
        ),
        _Atomic(
            results=(old_lo,),
            operands=(lo_elem, *indices),
            attrs=scalar_attrs,
        ),
        _Atomic(
            results=(old_hi,),
            operands=(hi_elem, *indices[:-1], col_plus_1),
            attrs=scalar_attrs,
        ),
    ]


# -----------------------------------------------------------------------------
# Async copy family — gated on ``caps.supports_async_copy``.
# -----------------------------------------------------------------------------


def _legalize_async(op: Op, caps: Any) -> list[Op] | None:
    """Shared keep-or-expand path for AsyncCopyOp / Commit / Wait.

    All three ops travel as a set — if the backend supports one, it
    supports them all. When ``supports_async_copy`` is False, every
    ``cp.async``-family op gets legalized to the synchronous
    equivalent in the same pass: ``AsyncCopyOp`` becomes a
    ``VecLoad``→``VecStore`` pair; ``AsyncCopyCommitOp`` and
    ``AsyncCopyWaitOp`` strip entirely (no sync state to track when
    every load is already synchronous)."""
    if getattr(caps, "supports_async_copy", False):
        return None  # backend has cp.async / equivalent
    if isinstance(op, (AsyncCopyCommitOp, AsyncCopyWaitOp)):
        return []  # no-op on synchronous backends
    # AsyncCopyOp expansion.
    return _expand_async_copy(op)


register_legalization(AsyncCopyOp)(_legalize_async)
register_legalization(AsyncCopyCommitOp)(_legalize_async)
register_legalization(AsyncCopyWaitOp)(_legalize_async)


def _expand_async_copy(op: Op) -> list[Op]:
    """Expand ``async_copy(dst_smem, src_gmem, dst_idx=..., src_idx=...,
    count=N)`` into a synchronous ``VecLoad``→``VecStore`` pair.

    Chain:
        vec = vec_load(src_gmem, *src_idx, width=count//elem_bytes)
        vec_store(dst_smem, vec, *dst_idx)

    The op carries ``n_dst_idx`` / ``n_src_idx`` attrs so the rewrite
    can recover the two index tuples from the flat ``operands`` list
    without re-parsing tensor ranks.

    When ``pred`` is present on the original op, it's the last operand;
    both the ``VecLoad`` and the ``VecStore`` inherit it.
    """
    from quark.ir.op import VecLoadOp, VecStoreOp
    from quark.ir.types import ValueShape
    from quark.lower.legalize import Rewriter

    dst_tensor = op.attrs["dst_tensor"]
    src_tensor = op.attrs["src_tensor"]
    count_bytes = op.attrs["count"]
    pred = op.attrs.get("pred")
    n_dst = op.attrs["n_dst_idx"]
    n_src = op.attrs["n_src_idx"]

    operands = op.operands
    # Operand layout: (*dst_idx, *src_idx, [pred])
    dst_idx = operands[:n_dst]
    src_idx = operands[n_dst : n_dst + n_src]

    elem_bytes = src_tensor.dtype.bytes
    if count_bytes % elem_bytes != 0:
        raise ValueError(
            f"_expand_async_copy: count {count_bytes} not a multiple of "
            f"src element size {elem_bytes}"
        )
    width = count_bytes // elem_bytes
    if width < 2:
        # Degenerate "vector of 1" — the vec_load/store ops require
        # width >= 2. Fall back to a scalar load + store would need
        # two more op types; rather than plumbing that, fail loudly.
        raise NotImplementedError(
            f"_expand_async_copy: count={count_bytes} gives width={width}; "
            "scalar-fallback path not wired up (add a LoadOp + StoreOp chain "
            "when a real target needs it)."
        )

    rw = Rewriter.for_op(op)
    vec = rw.alloc(ValueShape(src_tensor.dtype, width=width))

    vec_load_operands: tuple = tuple(src_idx)
    vec_store_operands: tuple = (vec, *dst_idx)
    if pred is not None:
        vec_load_operands = vec_load_operands + (pred,)
        vec_store_operands = vec_store_operands + (pred,)

    return [
        VecLoadOp(
            results=(vec,),
            operands=vec_load_operands,
            attrs={"tensor": src_tensor, "width": width, "pred": pred},
        ),
        VecStoreOp(
            results=(),
            operands=vec_store_operands,
            attrs={"tensor": dst_tensor, "pred": pred},
        ),
    ]


# -----------------------------------------------------------------------------
# Subgroup reduce — split by ``caps.has_native_subgroup_reduce``.
#
# Two valid lowering strategies exist:
#
#   * Native reduce (``has_native_subgroup_reduce=True``): the lowerer
#     emits one intrinsic — ``simd_sum`` on Metal,
#     ``OpGroupNonUniformAdd`` on SPIR-V. Legalization keeps the op
#     intact.
#   * Butterfly expansion (``has_native_subgroup_reduce=False``): a
#     chain of ``shfl.sync.bfly.b32`` + combine ops. PTX currently
#     implements this *inline* in ``_visit_subgroup_reduce``, so the
#     legalization pass is a no-op even on CUDA. A future migration
#     will lift the butterfly out of the PTX lowerer and into this
#     module so every no-native-reduce backend reads it from one
#     place — today that's tracked as a follow-up, not a blocker.
#
# Either way, the legalization pass never needs to *expand*
# SubgroupReduceOp right now: wired backends are either native
# (Metal) or handle the expansion in the lowerer (CUDA). The
# rewrite below is the decision matrix made explicit.
# -----------------------------------------------------------------------------


@register_legalization(SubgroupReduceOp)
def _legalize_subgroup_reduce(op: Op, caps: Any) -> list[Op] | None:
    """Keep the op on backends with a native single-op reduce; expand
    to a butterfly chain otherwise.

    Native-reduce backends (Metal ``simd_sum``, SPIR-V
    ``OpGroupNonUniformAdd``) keep the op for the lowerer to emit
    directly. Non-native backends (CUDA — only ``shfl.sync.bfly``)
    get the classic butterfly expansion here, which the
    downstream lowerer sees as a plain ``ShuffleOp`` + ``ArithOp``
    chain.

    Safety net: PTX's ``_visit_subgroup_reduce`` still inlines its
    own butterfly for code paths that bypass the legalization pass
    (direct-to-lowerer tests, ad-hoc IR construction). The expansion
    here is the preferred path for anything going through
    ``Launcher._lower``."""
    if getattr(caps, "has_native_subgroup_reduce", False):
        return None
    return _expand_subgroup_reduce_butterfly(op, caps)


def _expand_subgroup_reduce_butterfly(op: Op, caps: Any) -> list[Op]:
    """Butterfly-reduce a Value across a subgroup.

    For subgroup width ``W`` (2, 4, ..., 32), emit ``log2(W)``
    iterations, each one a ``ShuffleOp(bfly, offset)`` followed by
    an ``ArithOp`` combining the shuffled and current values with
    the reduction's combine op:

        cur = v
        for offset in (W//2, W//4, ..., 1):
            other = shuffle(bfly, cur, offset)
            cur   = <combine>(cur, other)

    The mapping from reduce op to arith kind:

        sum → add
        max → max
        min → min
        and → and
        or  → or

    The final arith op's result Value reuses the original
    ``SubgroupReduceOp``'s result Value so downstream consumers
    stay linked without a separate use-replacement pass.

    ``caps.subgroup_width`` determines ``W`` — defaults to 32 when
    not present. The PTX lowerer's inline butterfly uses the same
    derivation, so the emitted instruction sequence is byte-
    identical on CUDA where ``W=32``.
    """
    from quark.ir.op import ArithOp, ShuffleOp
    from quark.ir.types import ValueShape
    from quark.lower.legalize import Rewriter

    if len(op.operands) != 1 or len(op.results) != 1:
        raise ValueError(
            "_expand_subgroup_reduce_butterfly: expected 1 operand and 1 result, "
            f"got {len(op.operands)} / {len(op.results)}"
        )
    (src_val,) = op.operands
    (out_val,) = op.results

    reduce_op = op.attrs["op"]
    combine_kind = {
        "sum": "add",
        "max": "max",
        "min": "min",
        "and": "and",
        "or": "or",
    }.get(reduce_op)
    if combine_kind is None:
        raise ValueError(f"_expand_subgroup_reduce_butterfly: unknown reduce op {reduce_op!r}")

    W = int(getattr(caps, "subgroup_width", 32))
    if W < 2 or (W & (W - 1)) != 0:
        raise ValueError(
            f"_expand_subgroup_reduce_butterfly: subgroup_width {W} must be a power of two ≥2"
        )

    rw = Rewriter.for_op(op)
    ops: list[Op] = []

    # Chain carries the partial-reduction Value. Starts as the original
    # operand; each iteration produces a new partial; the final one
    # lands in the original op's result Value.
    cur = src_val
    offsets = []
    o = W // 2
    while o >= 1:
        offsets.append(o)
        o //= 2

    for i, offset in enumerate(offsets):
        other = rw.alloc(ValueShape(src_val.dtype, width=src_val.width))
        ops.append(
            ShuffleOp(
                results=(other,),
                operands=(cur,),
                attrs={"kind": "bfly", "param": offset},
            )
        )
        # Final iteration produces the original result Value.
        is_last = i == len(offsets) - 1
        nxt = out_val if is_last else rw.alloc(ValueShape(src_val.dtype, width=src_val.width))
        ops.append(
            ArithOp(
                results=(nxt,),
                operands=(cur, other),
                attrs={"kind": combine_kind},
            )
        )
        cur = nxt

    return ops
