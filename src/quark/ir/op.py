"""Op catalog for the quark IR.

Every op inherits from `Op` and carries:
- `results`: a tuple of Values this op defines
- `operands`: a tuple of Values this op consumes
- `attrs`: a dict of static attributes (kinds, shapes, constants, ...)
- `regions`: a tuple of Region objects for structured control flow

Op subclasses are dataclasses with fixed arity and an explicit set of
attribute keys. The Builder is the only thing that should construct
them directly; kernel code calls Builder methods.

See QUARK_IR_PROPOSAL.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from .region import Region
from .tensor import GlobalTensor, SharedRegion, Tensor
from .types import DType
from .value import Value

# ---------------------------------------------------------------------------
# Op base
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Op:
    """Base class for every IR op.

    Subclasses override `KIND` (a short tag) and often narrow `attrs`
    to a fixed schema. The base class keeps the unified dataclass
    layout so generic walkers (printer, validator, lowerer) can
    treat every op uniformly.
    """

    KIND: ClassVar[str] = "op"

    results: tuple[Value, ...] = ()
    operands: tuple[Value, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()

    def __post_init__(self) -> None:
        # Wire the back-reference on every result Value and attach any
        # regions to this op as their parent.
        for v in self.results:
            if v.producer is None:
                v.producer = self
        for r in self.regions:
            r.parent_op = self

    @property
    def result(self) -> Value:
        """Convenience accessor for single-result ops. Raises if 0 or >1."""
        if len(self.results) != 1:
            raise ValueError(f"{type(self).__name__}.result: op has {len(self.results)} results")
        return self.results[0]

    def __repr__(self) -> str:  # pragma: no cover — printer.py is the real one
        return f"<{type(self).__name__}>"


# ===========================================================================
# §5.1 Arithmetic and math
# ===========================================================================


_ARITH_KINDS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "mul_hi",
        "neg",
        "abs",
        "min",
        "max",
        "fma",
        "fma_bf16x2",
        "cvt_rn_bf16x2_f32",
        "div",
        "rem",
        "shl",
        "shr",
        "and",
        "or",
        "xor",
    }
)

_MATH_KINDS = frozenset(
    {
        "rcp",
        "rsqrt",
        "sqrt",
        "exp2",
        "log2",
        "sin",
        "cos",
        "tanh",
        "ex2_approx",
        "rcp_approx",
        "rsqrt_approx",
        "log2_approx",
        "sqrt_approx",
    }
)

_CMP_KINDS = frozenset({"lt", "le", "eq", "ne", "gt", "ge"})

_ROUND_MODES = frozenset({"rn", "rz", "rm", "rp", "satfinite"})


@dataclass(eq=False)
class ConstOp(Op):
    """A scalar literal. `attrs['value']` is a Python number; `attrs['dtype']`
    is the target DType. The result Value carries ValueShape(dtype)."""

    KIND: ClassVar[str] = "const"


@dataclass(eq=False)
class ArithOp(Op):
    """Integer and float arithmetic + bitwise. `attrs['kind']` picks the op.

    Unary ops (neg, abs) take one operand; binary ops take two; `fma`
    takes three (a, b, c → a*b + c).
    """

    KIND: ClassVar[str] = "arith"

    def __post_init__(self) -> None:
        super().__post_init__()
        kind = self.attrs.get("kind")
        if kind not in _ARITH_KINDS:
            raise ValueError(f"ArithOp: unknown kind {kind!r}")
        expected = {"neg": 1, "abs": 1, "fma": 3, "fma_bf16x2": 3, "cvt_rn_bf16x2_f32": 2}.get(
            kind, 2
        )
        if len(self.operands) != expected:
            raise ValueError(
                f"ArithOp({kind}): expected {expected} operands, got {len(self.operands)}"
            )


@dataclass(eq=False)
class MathOp(Op):
    """Transcendental / approximate math. Unary: one operand, one result."""

    KIND: ClassVar[str] = "math"

    def __post_init__(self) -> None:
        super().__post_init__()
        kind = self.attrs.get("kind")
        if kind not in _MATH_KINDS:
            raise ValueError(f"MathOp: unknown kind {kind!r}")
        if len(self.operands) != 1:
            raise ValueError(f"MathOp({kind}): expected 1 operand, got {len(self.operands)}")


@dataclass(eq=False)
class CmpOp(Op):
    """Comparison returning a PRED Value. Binary."""

    KIND: ClassVar[str] = "cmp"

    def __post_init__(self) -> None:
        super().__post_init__()
        kind = self.attrs.get("kind")
        if kind not in _CMP_KINDS:
            raise ValueError(f"CmpOp: unknown kind {kind!r}")
        if len(self.operands) != 2:
            raise ValueError(f"CmpOp({kind}): expected 2 operands, got {len(self.operands)}")


@dataclass(eq=False)
class SelectOp(Op):
    """Ternary: operands are (pred, true_val, false_val)."""

    KIND: ClassVar[str] = "select"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 3:
            raise ValueError(f"SelectOp: expected 3 operands, got {len(self.operands)}")
        pred, t, f = self.operands
        if pred.dtype is not DType.PRED:
            raise TypeError(f"SelectOp: pred operand must be PRED, got {pred.dtype}")
        if t.shape != f.shape:
            raise TypeError(
                f"SelectOp: true/false operands must have matching shape, "
                f"got {t.shape} vs {f.shape}"
            )


@dataclass(eq=False)
class ConvertOp(Op):
    """Explicit dtype conversion with a rounding mode."""

    KIND: ClassVar[str] = "convert"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 1:
            raise ValueError(f"ConvertOp: expected 1 operand, got {len(self.operands)}")
        if "src_dtype" not in self.attrs or "dst_dtype" not in self.attrs:
            raise ValueError("ConvertOp: missing src_dtype/dst_dtype attrs")
        rounding = self.attrs.get("rounding", "rn")
        if rounding not in _ROUND_MODES:
            raise ValueError(f"ConvertOp: unknown rounding {rounding!r}")


@dataclass(eq=False)
class PackedConvertOp(Op):
    """Packed 2-into-1 conversion — the PTX ISA's only cvt form for fp8
    destinations. Consumes two scalar source Values of the same dtype and
    produces one width-1 B16 Value whose two bytes are the packed
    ``<fp8>x2`` result, ready to be stored into contiguous fp8 memory
    via a single b16 store.

    Present as a first-class op (rather than a flag on ConvertOp) because
    it has a distinct signature: two operands, one result, different
    bit-width than either operand.
    """

    KIND: ClassVar[str] = "packed_convert"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 2:
            raise ValueError(f"PackedConvertOp: expected 2 operands, got {len(self.operands)}")
        if "src_dtype" not in self.attrs or "dst_dtype" not in self.attrs:
            raise ValueError("PackedConvertOp: missing src_dtype/dst_dtype attrs")
        if self.operands[0].dtype is not self.operands[1].dtype:
            raise TypeError(
                "PackedConvertOp: operands must share a dtype, got "
                f"{self.operands[0].dtype} vs {self.operands[1].dtype}"
            )
        rounding = self.attrs.get("rounding", "rn")
        if rounding not in _ROUND_MODES:
            raise ValueError(f"PackedConvertOp: unknown rounding {rounding!r}")


@dataclass(eq=False)
class UnpackedConvertOp(Op):
    """Packed 1-into-2 conversion — the symmetric inverse of PackedConvertOp.

    Consumes one width-1 B16 Value whose two bytes hold a packed
    ``<src_fp8>x2`` (or other 2x sub-byte format) value, and produces a
    width-2 Value of `dst_dtype` (typically BF16 or F16). Lowers to PTX's
    ``cvt.rn{.satfinite}{.relu}.<dst>x2.<src>x2`` instruction (PTX 9.2+ for
    bf16, PTX 7.8+ for f16).

    Required because PTX has NO scalar ``cvt.<wider>.e4m3`` instruction —
    the only cvt form for fp8 SOURCE is the packed one. This is the
    symmetric counterpart to PackedConvertOp (which handles fp8 dst).
    """

    KIND: ClassVar[str] = "unpacked_convert"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 1:
            raise ValueError(f"UnpackedConvertOp: expected 1 operand, got {len(self.operands)}")
        if "src_dtype" not in self.attrs or "dst_dtype" not in self.attrs:
            raise ValueError("UnpackedConvertOp: missing src_dtype/dst_dtype attrs")
        rounding = self.attrs.get("rounding", "rn")
        if rounding not in _ROUND_MODES:
            raise ValueError(f"UnpackedConvertOp: unknown rounding {rounding!r}")


@dataclass(eq=False)
class BitcastOp(Op):
    """Reinterpret bits as a different dtype of the same width."""

    KIND: ClassVar[str] = "bitcast"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 1:
            raise ValueError(f"BitcastOp: expected 1 operand, got {len(self.operands)}")
        dst = self.attrs.get("dst_dtype")
        if dst is None:
            raise ValueError("BitcastOp: missing dst_dtype attr")
        src_bytes = self.operands[0].shape.bytes
        dst_bytes = dst.bytes * self.operands[0].shape.width
        if src_bytes != dst_bytes:
            raise TypeError(f"BitcastOp: operand bits {src_bytes * 8} != dst bits {dst_bytes * 8}")


# ===========================================================================
# §5.2 Bit / lane manipulation
# ===========================================================================


@dataclass(eq=False)
class VecBuildOp(Op):
    """Pack N scalars into a width-N vector Value."""

    KIND: ClassVar[str] = "vec_build"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.results) != 1:
            raise ValueError("VecBuildOp: expected 1 result")
        width = self.results[0].width
        if not self.attrs.get("packed_b32") and len(self.operands) != width:
            raise ValueError(
                f"VecBuildOp: result width {width} must match operand count {len(self.operands)}"
            )


@dataclass(eq=False)
class VecExtractOp(Op):
    """Extract a single element from a vector Value. `attrs['index']` int."""

    KIND: ClassVar[str] = "vec_extract"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 1:
            raise ValueError("VecExtractOp: expected 1 operand")
        idx = self.attrs.get("index")
        if not isinstance(idx, int):
            raise ValueError("VecExtractOp: missing integer 'index' attr")
        if not (0 <= idx < self.operands[0].width):
            raise ValueError(
                f"VecExtractOp: index {idx} out of range for width {self.operands[0].width}"
            )


@dataclass(eq=False)
class SplitB32Op(Op):
    """Split a B32 Value into (lo_b16, hi_b16)."""

    KIND: ClassVar[str] = "split_b32"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 1:
            raise ValueError("SplitB32Op: expected 1 operand")
        if self.operands[0].dtype is not DType.B32 or self.operands[0].width != 1:
            raise TypeError("SplitB32Op: operand must be scalar B32")
        if len(self.results) != 2:
            raise ValueError("SplitB32Op: expected 2 results")


@dataclass(eq=False)
class MergeB32Op(Op):
    """Merge (lo_b16, hi_b16) into a B32."""

    KIND: ClassVar[str] = "merge_b32"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) != 2:
            raise ValueError("MergeB32Op: expected 2 operands")
        for v in self.operands:
            if v.dtype is not DType.B16 or v.width != 1:
                raise TypeError("MergeB32Op: operands must be scalar B16")


# ===========================================================================
# §5.3 Memory
# ===========================================================================


def _check_tensor_indices(tensor: Tensor, indices: tuple[Value, ...], op_name: str) -> None:
    if len(indices) != tensor.rank:
        raise ValueError(f"{op_name}: tensor rank {tensor.rank} but {len(indices)} indices given")
    for i, idx in enumerate(indices):
        if idx.width != 1 or not (idx.dtype.is_int or idx.dtype.is_bit):
            raise TypeError(
                f"{op_name}: index {i} must be a scalar int/bit Value, got {idx.shape!r}"
            )


@dataclass(eq=False)
class LoadOp(Op):
    """Scalar load from a GlobalTensor or SharedRegion.

    `attrs['tensor']` is the source Tensor; `operands` are the element
    indices; `attrs['pred']` is an optional predicate Value.
    """

    KIND: ClassVar[str] = "load"

    def __post_init__(self) -> None:
        super().__post_init__()
        t = self.attrs.get("tensor")
        if not isinstance(t, (GlobalTensor, SharedRegion)):
            raise TypeError("LoadOp: attrs['tensor'] must be GlobalTensor or SharedRegion")
        indices = self.operands
        pred = self.attrs.get("pred")
        if pred is not None:
            indices = indices[:-1]
            if self.operands[-1].dtype is not DType.PRED:
                raise TypeError("LoadOp: last operand is 'pred' but not PRED")
        _check_tensor_indices(t, indices, "LoadOp")
        if len(self.results) != 1:
            raise ValueError("LoadOp: expected 1 result")
        # Allow result-dtype reinterpret (bytes are bytes) — useful for
        # reading 2 contiguous fp8 elements as a single b16 to feed
        # `unpacked_convert`. The lowerer emits `ld.<space>.<cls(out)>` at
        # the element-byte address, so the caller is responsible for
        # picking an alignment that makes sense.
        if self.results[0].dtype is not t.dtype and self.results[0].dtype.bytes < t.dtype.bytes:
            raise TypeError(
                f"LoadOp: result dtype {self.results[0].dtype} narrower than "
                f"tensor dtype {t.dtype} — would read past the requested element."
            )


@dataclass(eq=False)
class StoreOp(Op):
    """Scalar store to a GlobalTensor or SharedRegion.

    `operands = (value, *indices [, pred])`.
    """

    KIND: ClassVar[str] = "store"

    def __post_init__(self) -> None:
        super().__post_init__()
        t = self.attrs.get("tensor")
        if not isinstance(t, (GlobalTensor, SharedRegion)):
            raise TypeError("StoreOp: attrs['tensor'] must be GlobalTensor or SharedRegion")
        if self.results:
            raise ValueError("StoreOp: must have no results")
        if not self.operands:
            raise ValueError("StoreOp: expected (value, *indices)")
        value = self.operands[0]
        rest = self.operands[1:]
        if self.attrs.get("pred") is not None:
            if not rest or rest[-1].dtype is not DType.PRED:
                raise TypeError("StoreOp: last operand must be PRED when pred attr is set")
            rest = rest[:-1]
        _check_tensor_indices(t, rest, "StoreOp")
        if value.dtype is not t.dtype:
            # Permit "packed" stores — a value whose total bit width is a
            # whole multiple of the tensor's element width (e.g. a B16
            # value written into an E4M3 tile lays down two fp8 bytes at
            # the index-computed byte address). The address arithmetic
            # downstream uses `tensor.dtype.bytes`, so the indices still
            # name the *first* narrow element that the store touches.
            v_bytes = value.shape.bytes
            t_elem_bytes = t.dtype.bytes
            if t_elem_bytes == 0 or v_bytes % t_elem_bytes != 0:
                raise TypeError(
                    f"StoreOp: value bits ({v_bytes * 8}) not a whole multiple of "
                    f"tensor element bits ({t_elem_bytes * 8}) — can't pack "
                    f"{value.dtype} into {t.dtype} storage"
                )


@dataclass(eq=False)
class VecLoadOp(Op):
    """Vector load — result is a width-N Value. `attrs['width']` required."""

    KIND: ClassVar[str] = "vec_load"

    def __post_init__(self) -> None:
        super().__post_init__()
        t = self.attrs.get("tensor")
        if not isinstance(t, (GlobalTensor, SharedRegion)):
            raise TypeError("VecLoadOp: attrs['tensor'] must be Global/SharedRegion")
        width = self.attrs.get("width")
        if not isinstance(width, int) or width < 2:
            raise ValueError(f"VecLoadOp: width attr must be int >=2, got {width!r}")
        indices = self.operands
        if self.attrs.get("pred") is not None:
            indices = indices[:-1]
        _check_tensor_indices(t, indices, "VecLoadOp")
        if len(self.results) != 1 or self.results[0].width != width:
            raise ValueError(f"VecLoadOp: must have 1 result of width={width}")


@dataclass(eq=False)
class VecStoreOp(Op):
    """Vector store — operand 0 is a width-N Value."""

    KIND: ClassVar[str] = "vec_store"

    def __post_init__(self) -> None:
        super().__post_init__()
        t = self.attrs.get("tensor")
        if not isinstance(t, (GlobalTensor, SharedRegion)):
            raise TypeError("VecStoreOp: attrs['tensor'] must be Global/SharedRegion")
        if self.results:
            raise ValueError("VecStoreOp: must have no results")
        if not self.operands:
            raise ValueError("VecStoreOp: expected (vec, *indices)")
        vec = self.operands[0]
        if vec.width < 2:
            raise TypeError("VecStoreOp: first operand must be a vector Value")
        rest = self.operands[1:]
        if self.attrs.get("pred") is not None:
            rest = rest[:-1]
        _check_tensor_indices(t, rest, "VecStoreOp")


@dataclass(eq=False)
class AsyncCopyOp(Op):
    """cp.async-style copy from gmem to smem. No results.

    `attrs`: dst_tensor, src_tensor, count (bytes), pred?
    `operands`: (*dst_idxs, *src_idxs [, pred])
    """

    KIND: ClassVar[str] = "async_copy"

    def __post_init__(self) -> None:
        super().__post_init__()
        dst = self.attrs.get("dst_tensor")
        src = self.attrs.get("src_tensor")
        if not isinstance(dst, SharedRegion):
            raise TypeError("AsyncCopyOp: dst_tensor must be SharedRegion")
        if not isinstance(src, GlobalTensor):
            raise TypeError("AsyncCopyOp: src_tensor must be GlobalTensor")
        if self.results:
            raise ValueError("AsyncCopyOp: must have no results")
        if "count" not in self.attrs:
            raise ValueError("AsyncCopyOp: missing 'count' attr (bytes per copy)")


@dataclass(eq=False)
class AsyncCopyCommitOp(Op):
    """cp.async.commit_group — flushes outstanding async copies into a group."""

    KIND: ClassVar[str] = "async_commit"


@dataclass(eq=False)
class AsyncCopyWaitOp(Op):
    """cp.async.wait_group n — wait until <= n groups remain in flight."""

    KIND: ClassVar[str] = "async_wait"

    def __post_init__(self) -> None:
        super().__post_init__()
        if "n" not in self.attrs:
            raise ValueError("AsyncCopyWaitOp: missing 'n' attr")


_ATOMIC_OPS = frozenset({"add", "min", "max", "and", "or", "xor", "exch"})


@dataclass(eq=False)
class AtomicRmwOp(Op):
    """Atomic read-modify-write. Result is the old value (or new, depending
    on backend; we standardize on 'old value' in IR semantics and let each
    lowerer adjust as needed).
    """

    KIND: ClassVar[str] = "atomic_rmw"

    def __post_init__(self) -> None:
        super().__post_init__()
        t = self.attrs.get("tensor")
        if not isinstance(t, GlobalTensor):
            raise TypeError("AtomicRmwOp: tensor must be GlobalTensor")
        op = self.attrs.get("op")
        if op not in _ATOMIC_OPS:
            raise ValueError(f"AtomicRmwOp: unknown op {op!r}")


# ===========================================================================
# §5.4 Shared memory allocation
# ===========================================================================


@dataclass(eq=False)
class SmemAllocOp(Op):
    """Allocate a block-scoped shared memory region.

    `attrs`: name, dtype, shape (tuple[int,...]), pad (int), align (int)
    `results`: one opaque Value standing in for the backing storage;
    SharedRegions reference it by Value identity.
    """

    KIND: ClassVar[str] = "smem_alloc"

    def __post_init__(self) -> None:
        super().__post_init__()
        for key in ("name", "dtype", "shape"):
            if key not in self.attrs:
                raise ValueError(f"SmemAllocOp: missing '{key}' attr")
        if self.operands:
            raise ValueError("SmemAllocOp: expected no operands")
        if len(self.results) != 1:
            raise ValueError("SmemAllocOp: expected 1 result (the backing)")

    @property
    def dtype(self) -> DType:
        return self.attrs["dtype"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.attrs["shape"])

    @property
    def pad(self) -> int:
        return int(self.attrs.get("pad", 0))

    @property
    def name(self) -> str:
        return self.attrs["name"]


# ===========================================================================
# §5.5 Cross-lane / subgroup
# ===========================================================================


_SHUFFLE_KINDS = frozenset({"bfly", "xor", "up", "down", "idx"})
_SUBGROUP_REDUCE_OPS = frozenset({"sum", "max", "min", "and", "or"})


@dataclass(eq=False)
class ShuffleOp(Op):
    """Warp/simdgroup shuffle. `attrs`: kind, param."""

    KIND: ClassVar[str] = "shuffle"

    def __post_init__(self) -> None:
        super().__post_init__()
        kind = self.attrs.get("kind")
        if kind not in _SHUFFLE_KINDS:
            raise ValueError(f"ShuffleOp: unknown kind {kind!r}")
        if "param" not in self.attrs:
            raise ValueError("ShuffleOp: missing 'param' attr")
        if len(self.operands) != 1:
            raise ValueError("ShuffleOp: expected 1 operand")


@dataclass(eq=False)
class SubgroupReduceOp(Op):
    """Cross-lane reduction over a warp/simdgroup."""

    KIND: ClassVar[str] = "subgroup_reduce"

    def __post_init__(self) -> None:
        super().__post_init__()
        op = self.attrs.get("op")
        if op not in _SUBGROUP_REDUCE_OPS:
            raise ValueError(f"SubgroupReduceOp: unknown op {op!r}")
        if len(self.operands) != 1:
            raise ValueError("SubgroupReduceOp: expected 1 operand")


@dataclass(eq=False)
class SubgroupBroadcastOp(Op):
    """Broadcast a Value from one lane to every lane in the subgroup."""

    KIND: ClassVar[str] = "subgroup_broadcast"

    def __post_init__(self) -> None:
        super().__post_init__()
        if "lane" not in self.attrs:
            raise ValueError("SubgroupBroadcastOp: missing 'lane' attr")
        if len(self.operands) != 1:
            raise ValueError("SubgroupBroadcastOp: expected 1 operand")


# ===========================================================================
# §5.6 Thread identity (SSA constants)
# ===========================================================================


_VALID_DIMS = frozenset({"x", "y", "z"})


@dataclass(eq=False)
class _DimQueryOp(Op):
    """Base for ThreadIdx / BlockIdx / BlockDim / GridDim queries."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attrs.get("dim") not in _VALID_DIMS:
            raise ValueError(f"{type(self).__name__}: missing/invalid 'dim' attr")
        if self.operands:
            raise ValueError(f"{type(self).__name__}: expected no operands")


@dataclass(eq=False)
class ThreadIdxOp(_DimQueryOp):
    KIND: ClassVar[str] = "thread_idx"


@dataclass(eq=False)
class BlockIdxOp(_DimQueryOp):
    KIND: ClassVar[str] = "block_idx"


@dataclass(eq=False)
class BlockDimOp(_DimQueryOp):
    KIND: ClassVar[str] = "block_dim"


@dataclass(eq=False)
class GridDimOp(_DimQueryOp):
    KIND: ClassVar[str] = "grid_dim"


@dataclass(eq=False)
class LaneIdOp(Op):
    KIND: ClassVar[str] = "lane_id"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.operands:
            raise ValueError("LaneIdOp: expected no operands")


@dataclass(eq=False)
class SubgroupIdOp(Op):
    KIND: ClassVar[str] = "subgroup_id"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.operands:
            raise ValueError("SubgroupIdOp: expected no operands")


@dataclass(eq=False)
class GroupIdOp(Op):
    """MMA group id = laneid >> 2.

    The "group" concept comes from the PTX ISA mma fragment formulas
    (§9.7.14.5), which express every A/B/C per-lane (row, col) pair
    in terms of `groupID = %laneid >> 2` and `threadID_in_group =
    %laneid & 3`. Every kernel that loads mma fragments needs this
    quantity once per thread. Materializing it as a first-class op
    keeps callers from having to re-derive it with manual `shr`
    sequences every time.

    Zero operands, one u32 result.
    """

    KIND: ClassVar[str] = "group_id"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.operands:
            raise ValueError("GroupIdOp: expected no operands")
        if len(self.results) != 1:
            raise ValueError("GroupIdOp: expected 1 result")


@dataclass(eq=False)
class ThreadIdInGroupOp(Op):
    """MMA thread-in-group id = laneid & 3.

    Companion to `GroupIdOp`. Every mma fragment formula is written
    against `threadID_in_group`; exposing it as a first-class op
    mirrors `groupID` so kernels don't have to emit a manual
    `and.b32` bit-mask sequence every time they load a tile.

    Zero operands, one u32 result.
    """

    KIND: ClassVar[str] = "thread_id_in_group"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.operands:
            raise ValueError("ThreadIdInGroupOp: expected no operands")
        if len(self.results) != 1:
            raise ValueError("ThreadIdInGroupOp: expected 1 result")


# ===========================================================================
# §5.7 Control flow (structured)
# ===========================================================================


_BARRIER_SCOPES = frozenset({"block", "subgroup", "system"})


@dataclass(eq=False)
class BarrierOp(Op):
    """Synchronization. `attrs['scope']` ∈ {block, subgroup, system}."""

    KIND: ClassVar[str] = "barrier"

    def __post_init__(self) -> None:
        super().__post_init__()
        scope = self.attrs.get("scope", "block")
        if scope not in _BARRIER_SCOPES:
            raise ValueError(f"BarrierOp: unknown scope {scope!r}")
        if self.results or self.operands:
            raise ValueError("BarrierOp: expected no results/operands")


@dataclass(eq=False)
class YieldOp(Op):
    """Terminator for a Region. Produces nothing; operands are the values
    yielded back to the enclosing structured op."""

    KIND: ClassVar[str] = "yield"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.results:
            raise ValueError("YieldOp: must have no results")


@dataclass(eq=False)
class ForLoopOp(Op):
    """Counted for loop with optional loop-carried values.

    operands: (lo, hi, step, *carried_in)
    regions:  (body,)
    results:  one per carried_in, matching yielded values from body
    attrs:    iv_name (str), iv_dtype (DType)

    The induction variable and the per-iteration "body carry" Values
    are Values defined by this op and visible only inside `body`.
    `carried_body_vars[i]` is the Value the body sees as the incoming
    value of the i-th carried slot.
    """

    KIND: ClassVar[str] = "for_loop"

    induction_var: Value | None = None
    carried_body_vars: tuple[Value, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operands) < 3:
            raise ValueError("ForLoopOp: operands must start with (lo, hi, step)")
        if len(self.regions) != 1:
            raise ValueError("ForLoopOp: expected exactly one region (body)")
        if "iv_name" not in self.attrs or "iv_dtype" not in self.attrs:
            raise ValueError("ForLoopOp: missing iv_name/iv_dtype attrs")
        if self.induction_var is None:
            raise ValueError("ForLoopOp: induction_var must be provided at construction")
        lo, hi, step = self.operands[:3]
        # lo/hi/step must share a common integer dtype
        if not (lo.dtype is hi.dtype is step.dtype):
            raise TypeError(
                f"ForLoopOp: lo/hi/step must share a dtype, got {lo.dtype}/{hi.dtype}/{step.dtype}"
            )
        carried_in = self.operands[3:]
        # One result per carried-in; shapes must match.
        if len(self.results) != len(carried_in):
            raise ValueError(
                f"ForLoopOp: {len(carried_in)} carried_in vs {len(self.results)} results"
            )
        for i, (cin, cout) in enumerate(zip(carried_in, self.results, strict=False)):
            if cin.shape != cout.shape:
                raise TypeError(
                    f"ForLoopOp: carried[{i}] shape {cin.shape} != result shape {cout.shape}"
                )

    @property
    def body(self) -> Region:
        return self.regions[0]

    @property
    def lo(self) -> Value:
        return self.operands[0]

    @property
    def hi(self) -> Value:
        return self.operands[1]

    @property
    def step(self) -> Value:
        return self.operands[2]

    @property
    def carried_in(self) -> tuple[Value, ...]:
        return self.operands[3:]


@dataclass(eq=False)
class IfRegionOp(Op):
    """Structured if/then/else.

    operands: (pred, *then_in, *else_in)        # see attrs['n_carried']
    regions:  (then_region, else_region)
    results:  one per yielded value (both regions yield the same shapes)

    `then_body_vars` / `else_body_vars` are the per-arm "carry in" Values
    visible inside the respective arm. They're defined by this op and
    never visible outside it.
    """

    KIND: ClassVar[str] = "if_region"

    then_body_vars: tuple[Value, ...] = ()
    else_body_vars: tuple[Value, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.operands:
            raise ValueError("IfRegionOp: expected at least (pred,)")
        if self.operands[0].dtype is not DType.PRED:
            raise TypeError("IfRegionOp: first operand must be PRED")
        if len(self.regions) != 2:
            raise ValueError("IfRegionOp: expected exactly 2 regions (then, else_)")

    @property
    def pred(self) -> Value:
        return self.operands[0]

    @property
    def then_region(self) -> Region:
        return self.regions[0]

    @property
    def else_region(self) -> Region:
        return self.regions[1]


@dataclass(eq=False)
class WhileLoopOp(Op):
    """Structured while loop with a condition region and a body region.

    operands: (*carried_in,)
    regions:  (cond, body)
    results:  one per carried_in

    The cond region yields a PRED; the body region yields the next
    carried values.
    """

    KIND: ClassVar[str] = "while_loop"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.regions) != 2:
            raise ValueError("WhileLoopOp: expected exactly 2 regions (cond, body)")
        if len(self.results) != len(self.operands):
            raise ValueError("WhileLoopOp: results and carried_in must match")

    @property
    def cond_region(self) -> Region:
        return self.regions[0]

    @property
    def body_region(self) -> Region:
        return self.regions[1]


# ===========================================================================
# §5.8 Matmul and fragment ops
# ===========================================================================


@dataclass(eq=False)
class LoadMatrixOp(Op):
    """Load a matmul fragment from a Global/Shared tensor region.

    operands: (row, col)                     # element-unit tile base
    attrs:    src_tensor, shape_id, which, layout_hint?, reg_offsets?
    result:   one Value representing the fragment (a width-N b32 vec
              where N is the frag's per-thread register count)

    `reg_offsets` is a tuple of `(row_elem_off, col_elem_off)` pairs,
    one per fragment register, in the order the mma instruction expects
    them. The PTX lowerer's default "manual" path emits N scalar
    `ld.shared.b32` instructions at `base + (row+dr)*row_stride +
    (col+dc)*elem_bytes` — exactly the pattern the existing frag
    loaders in mma/frag.py use.

    Callers source the offsets from the authoritative PTX ISA fragment
    formulas (section 9.7.14.5) for their chosen smem layout. For
    preshuffled smem, the offsets just reflect the shuffled positions.
    For the codebase's usual row.col layout (A rowmajor, B^T in smem
    so K is the contiguous dim), see tests/lower/ptx/test_matmul.py
    for worked examples per dtype/shape.

    When `reg_offsets` is omitted, the lowerer must be told to take an
    alternative path via `layout_hint` (e.g. `"ldmatrix"`).
    """

    KIND: ClassVar[str] = "load_matrix"

    def __post_init__(self) -> None:
        super().__post_init__()
        for key in ("src_tensor", "shape_id", "which"):
            if key not in self.attrs:
                raise ValueError(f"LoadMatrixOp: missing '{key}' attr")
        if self.attrs["which"] not in ("a", "b", "c"):
            raise ValueError(f"LoadMatrixOp: which must be a/b/c, got {self.attrs['which']!r}")
        if not isinstance(self.attrs["src_tensor"], (SharedRegion, GlobalTensor)):
            raise TypeError("LoadMatrixOp: src_tensor must be Shared/GlobalTensor")
        if len(self.operands) != 2:
            raise ValueError("LoadMatrixOp: expected 2 operands (row, col)")
        if len(self.results) != 1:
            raise ValueError("LoadMatrixOp: expected 1 result")
        _validate_reg_offsets(self.attrs, self.results[0].width, "LoadMatrixOp")


@dataclass(eq=False)
class StoreMatrixOp(Op):
    """Store a matmul fragment to a Global/Shared tensor region.

    operands: (frag, row, col)
    attrs:    dst_tensor, shape_id, which, layout_hint?, reg_offsets?

    `reg_offsets` semantics mirror LoadMatrixOp — one `(dr, dc)` per
    fragment register. The default PTX lowering emits N scalar
    `st.<space>.b32` at the matching per-register byte offsets.
    """

    KIND: ClassVar[str] = "store_matrix"

    def __post_init__(self) -> None:
        super().__post_init__()
        for key in ("dst_tensor", "shape_id", "which"):
            if key not in self.attrs:
                raise ValueError(f"StoreMatrixOp: missing '{key}' attr")
        if self.attrs["which"] not in ("c", "d"):
            raise ValueError(f"StoreMatrixOp: which must be c/d, got {self.attrs['which']!r}")
        if self.results:
            raise ValueError("StoreMatrixOp: must have no results")
        if len(self.operands) != 3:
            raise ValueError("StoreMatrixOp: expected (frag, row, col)")
        _validate_reg_offsets(self.attrs, self.operands[0].width, "StoreMatrixOp")


def _validate_reg_offsets(attrs: dict, expected_count: int, op_name: str) -> None:
    """Check that `attrs['reg_offsets']` — when present — is a tuple
    of `expected_count` `(int, int)` pairs. The attr is optional."""
    offs = attrs.get("reg_offsets")
    if offs is None:
        return
    if not isinstance(offs, tuple):
        raise TypeError(f"{op_name}: reg_offsets must be a tuple, got {type(offs).__name__}")
    if len(offs) != expected_count:
        raise ValueError(
            f"{op_name}: reg_offsets has {len(offs)} entries, fragment width is {expected_count}"
        )
    for i, entry in enumerate(offs):
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise TypeError(f"{op_name}: reg_offsets[{i}] must be a (row, col) int tuple")
        dr, dc = entry
        if not (isinstance(dr, int) and isinstance(dc, int)):
            raise TypeError(
                f"{op_name}: reg_offsets[{i}] must contain ints, got "
                f"({type(dr).__name__}, {type(dc).__name__})"
            )


@dataclass(eq=False)
class MmaOp(Op):
    """d = a * b + c. Operands are (frag_a, frag_b, frag_c) Values produced
    by LoadMatrixOp (or a previous MmaOp for c)."""

    KIND: ClassVar[str] = "mma"

    def __post_init__(self) -> None:
        super().__post_init__()
        if "shape_id" not in self.attrs:
            raise ValueError("MmaOp: missing 'shape_id' attr")
        if len(self.operands) != 3:
            raise ValueError("MmaOp: expected 3 operands (a, b, c)")
        if len(self.results) != 1:
            raise ValueError("MmaOp: expected 1 result (d)")


@dataclass(eq=False)
class FragApplyOp(Op):
    """Apply a scalar transform to every element of an accumulator fragment.

    The body region holds the per-element subgraph and the lowerer
    inlines it per storage slot. For m16n8
    accumulators that's 4 slots on PTX (one per c_reg) and 2 slots per
    simdgroup tile on MSL (``thread_elements()``). The same body op graph
    applies; only the binding of the input element changes per slot.

    Structure:
      operands: ``(in_frag,)``
      regions:  ``(body,)`` — a Region with exactly one YieldOp terminator;
                the body sees ``body_input_var`` as the per-slot scalar.
      results:  ``(out_frag,)`` — same shape as ``in_frag``.

    Why an IR Region and not a Python callable: the body is a real subgraph
    so the printer / CSE / generic walkers see it. Re-emission per slot is
    handled at lower time by walking the body with ``body_input_var``
    rebound to a per-slot scalar name (see ``_visit_frag_apply`` on each
    backend).

    Attrs:
      * ``shape_id`` — MMA shape name (so the lowerer can resolve the
        MSL ``(mf, nf)`` tile grid or the PTX c_reg count).

    The op ALWAYS returns a fragment of the same dtype/width as the input.
    Layout-changing or dtype-changing transforms belong to ``FragConvertOp``
    (see proposal §1.4), not here.
    """

    KIND: ClassVar[str] = "frag_apply"

    body_input_var: Value | None = None
    body_selector_var: Value | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if "shape_id" not in self.attrs:
            raise ValueError("FragApplyOp: missing 'shape_id' attr")
        if len(self.operands) < 1:
            raise ValueError("FragApplyOp: expected at least 1 operand (in_frag)")
        if len(self.results) != 1:
            raise ValueError("FragApplyOp: expected exactly 1 result (out_frag)")
        if len(self.regions) != 1:
            raise ValueError("FragApplyOp: expected exactly 1 region (body)")
        if self.body_input_var is None:
            raise ValueError("FragApplyOp: body_input_var must be provided at construction")
        in_frag = self.operands[0]
        selectors = self.operands[1:]
        (out_frag,) = self.results
        if in_frag.shape != out_frag.shape:
            raise ValueError(
                f"FragApplyOp: out shape {out_frag.shape} must match in shape {in_frag.shape}"
            )
        if selectors:
            if self.body_selector_var is None:
                raise ValueError(
                    "FragApplyOp: selector operands present but body_selector_var is None"
                )
            if "slot_to_selector_idx" not in self.attrs:
                raise ValueError(
                    "FragApplyOp: selectors present but 'slot_to_selector_idx' attr missing"
                )
            mapping = self.attrs["slot_to_selector_idx"]
            for sel_idx in mapping:
                if not (0 <= sel_idx < len(selectors)):
                    raise ValueError(
                        f"FragApplyOp: slot_to_selector_idx contains {sel_idx} "
                        f"but only {len(selectors)} selectors provided"
                    )
            for s in selectors:
                if s.width != 1:
                    raise ValueError(
                        f"FragApplyOp: selector operands must be scalar (got width={s.width})"
                    )
        elif self.body_selector_var is not None:
            raise ValueError(
                "FragApplyOp: body_selector_var set but no selector operands — "
                "pass selectors along with body_selector_var or leave both None"
            )
        body = self.regions[0]
        term = body.terminator
        if term is None:
            raise ValueError("FragApplyOp: body must end with a YieldOp")
        if len(term.operands) != 1:
            raise ValueError(
                f"FragApplyOp: body yield must produce exactly 1 value (got {len(term.operands)})"
            )
        yielded = term.operands[0]
        if yielded.shape != self.body_input_var.shape:
            raise TypeError(
                f"FragApplyOp: yielded shape {yielded.shape} must match body input shape "
                f"{self.body_input_var.shape} (dtype-changing maps should use FragConvertOp)"
            )

    @property
    def body(self) -> Region:
        return self.regions[0]

    @property
    def in_frag(self) -> Value:
        return self.operands[0]

    @property
    def selectors(self) -> tuple[Value, ...]:
        return self.operands[1:]


@dataclass(eq=False)
class FragForEachOp(Op):
    """Apply a side-effect body to every storage slot of a fragment.

    Produces NO output fragment — the body writes to memory (stores,
    atomics) using the per-slot element AND the slot's tile-local
    ``(row, col)`` position. The position vars are bound per-slot at
    lower time to backend-specific lane-dependent expressions:

      * PTX: ``(row, col) = (groupID + dr, tidIG*2 + dc)`` per cd_offsets.
      * MSL: ``(row, col) = (apple_row + tile_row, apple_col + slot_idx)``
        where apple_row / apple_col are the per-lane 2x2x2 bit-swizzle
        (see frag_tile.apple_lane_to_tile_position) and tile_row is the
        fi*8 offset for the simdgroup_matrix array index.

    This is the "epilogue primitive": the caller computes a gmem address
    from ``(row, col)`` relative to a tile base and emits the store or
    atomic inside the body. Both backends skip the smem round-trip —
    PTX reads c_regs directly, MSL reads thread_elements().

    operands: ``(in_frag,)``
    regions:  ``(body,)`` — body has NO yielded value (no terminator or
              a YieldOp with 0 operands).
    results:  () — no output.
    attrs:
      * ``shape_id``: MMA shape name.
      * ``cd_offsets``: per-c_reg (dr, dc) offsets (tile-local).

    body_input_var: Value — the per-slot scalar element.
    body_row_var:   Value (U32) — the per-slot tile-local row.
    body_col_var:   Value (U32) — the per-slot tile-local col.
    """

    KIND: ClassVar[str] = "frag_for_each"

    body_input_var: Value | None = None
    body_row_var: Value | None = None
    body_col_var: Value | None = None
    body_selector_var: Value | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if "shape_id" not in self.attrs:
            raise ValueError("FragForEachOp: missing 'shape_id' attr")
        if "cd_offsets" not in self.attrs:
            raise ValueError("FragForEachOp: missing 'cd_offsets' attr")
        if len(self.operands) < 1:
            raise ValueError("FragForEachOp: expected ≥1 operands (in_frag [+ selectors])")
        if len(self.results) != 0:
            raise ValueError("FragForEachOp: expected 0 results (side-effect body)")
        if len(self.regions) != 1:
            raise ValueError("FragForEachOp: expected 1 region (body)")
        for field_name in ("body_input_var", "body_row_var", "body_col_var"):
            if getattr(self, field_name) is None:
                raise ValueError(f"FragForEachOp: {field_name} must be provided at construction")
        selectors = self.operands[1:]
        if selectors:
            if self.body_selector_var is None:
                raise ValueError("FragForEachOp: selectors present but body_selector_var is None")
            if "slot_to_selector_idx" not in self.attrs:
                raise ValueError(
                    "FragForEachOp: selectors present but 'slot_to_selector_idx' attr missing"
                )
            for s in selectors:
                if s.width != 1:
                    raise ValueError(
                        f"FragForEachOp: selector operands must be scalar (got width={s.width})"
                    )
        elif self.body_selector_var is not None:
            raise ValueError("FragForEachOp: body_selector_var set but no selector operands")
        # Body must terminate with a void YieldOp (no operands) or nothing.
        body = self.regions[0]
        term = body.terminator
        if term is not None and len(term.operands) != 0:
            raise ValueError("FragForEachOp: body must end with a void YieldOp (no yielded values)")

    @property
    def body(self) -> Region:
        return self.regions[0]

    @property
    def in_frag(self) -> Value:
        return self.operands[0]

    @property
    def selectors(self) -> tuple[Value, ...]:
        return self.operands[1:]


@dataclass(eq=False)
class FragConvertOp(Op):
    """Convert N source fragments to a destination fragment of a
    different (layout, dtype).

    Current supported conversions:
      * ACC f32 → A_FRAG bf16 — the online-softmax P-fragment generator.
        Takes ``kf`` source ACC tiles (each covers 8 K-cols) and produces
        one A-fragment covering ``kf * 8`` K-cols. Optional body applies
        a per-element f32 transform (e.g. exp2((x - m_rc) * log2e))
        before the bf16 cast.

    Later this op will subsume layout/dtype transforms for A ↔ B frags,
    accumulator type promotion, etc. Narrow-scoped for now to keep the
    lowering simple.

    Structure:
      operands: ``(src_frag_0, ..., src_frag_{kf-1}, *selectors)``
      regions: ``(body,)`` — may be empty (no transform, pure cast).
      results: one output Value — width = dst_regs on PTX (a b32 vec);
               on MSL registered in ``frag_values`` as a
               ``simdgroup_matrix<dst_dtype, 8, 8>[mf*kf]`` array.

    Attrs:
      * ``shape_id``: MMA shape name (for mf/kf resolution).
      * ``src_layout``, ``dst_layout``: e.g. "acc", "a_frag".
      * ``src_dtype``, ``dst_dtype``: IR DType.
      * ``num_src_frags``: the kf on the destination side (how many
        source tiles combine into one output).
      * ``cd_offsets``: for row-class selectors on the source.
      * ``slot_to_selector_idx``: optional, if selectors are used.

    body_input_var: the per-slot source element (f32).
    body_selector_var: optional — per-slot selector bound from
                       ``selectors`` via ``slot_to_selector_idx``.
    """

    KIND: ClassVar[str] = "frag_convert"

    body_input_var: Value | None = None
    body_selector_var: Value | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for required in (
            "shape_id",
            "src_layout",
            "dst_layout",
            "src_dtype",
            "dst_dtype",
            "num_src_frags",
        ):
            if required not in self.attrs:
                raise ValueError(f"FragConvertOp: missing {required!r} attr")
        num_src = int(self.attrs["num_src_frags"])
        if num_src < 1:
            raise ValueError(f"FragConvertOp: num_src_frags must be ≥ 1 (got {num_src})")
        if len(self.operands) < num_src:
            raise ValueError(
                f"FragConvertOp: need ≥{num_src} operands (src frags), got {len(self.operands)}"
            )
        if len(self.results) != 1:
            raise ValueError("FragConvertOp: expected exactly 1 result")
        selectors = self.operands[num_src:]
        if selectors:
            if self.body_selector_var is None:
                raise ValueError("FragConvertOp: selectors present but body_selector_var is None")
            if "slot_to_selector_idx" not in self.attrs:
                raise ValueError(
                    "FragConvertOp: selectors present but 'slot_to_selector_idx' attr missing"
                )
            for s in selectors:
                if s.width != 1:
                    raise ValueError(
                        f"FragConvertOp: selector operands must be scalar, got width={s.width}"
                    )
        elif self.body_selector_var is not None:
            raise ValueError("FragConvertOp: body_selector_var set but no selector operands")
        # Body (optional — regions=() means no transform, pure layout/dtype change).
        if self.regions:
            if len(self.regions) != 1:
                raise ValueError("FragConvertOp: expected 0 or 1 region (body)")
            if self.body_input_var is None:
                raise ValueError("FragConvertOp: body region present but body_input_var is None")
            body = self.regions[0]
            term = body.terminator
            if term is None or len(term.operands) != 1:
                raise ValueError(
                    "FragConvertOp: body must yield exactly 1 value (the transformed f32 elem)"
                )

    @property
    def body(self) -> Region | None:
        return self.regions[0] if self.regions else None

    @property
    def src_frags(self) -> tuple[Value, ...]:
        n = int(self.attrs["num_src_frags"])
        return self.operands[:n]

    @property
    def selectors(self) -> tuple[Value, ...]:
        n = int(self.attrs["num_src_frags"])
        return self.operands[n:]


_FRAG_REDUCE_KINDS = frozenset({"max", "min", "add", "mul"})


@dataclass(eq=False)
class FragReduceOp(Op):
    """Reduce a fragment along one axis, producing one scalar per
    perpendicular equivalence class.

    For ``axis="row"`` on an accumulator: reduces across *columns*,
    producing one scalar per row class. For m16n8 bf16 (cd_offsets
    ``((0,0),(0,1),(8,0),(8,1))``), 2 row classes → 2 scalar results.
    Each scalar is broadcast across every lane that shares the class
    (via the per-backend butterfly shuffle pattern) so downstream
    consumers on any lane read the full reduction.

    operands: ``(in_frag,)``
    results:  one width-1 Value per equivalence class (all share
              ``in_frag.dtype``).
    attrs:
      * ``shape_id``: MMA shape name.
      * ``axis``:  ``"row"`` or ``"col"``. ``"row"`` reduces cols, returns
                   one scalar per row class. ``"col"`` reduces rows.
      * ``kind``:  ``"max"``, ``"min"``, ``"add"``, ``"mul"``.
      * ``cd_offsets``: per-c_reg (dr, dc) offsets; the row/col class
                   partition is derived from the sorted distinct dr (for
                   axis=row) or dc (for axis=col) values.
    """

    KIND: ClassVar[str] = "frag_reduce"

    def __post_init__(self) -> None:
        super().__post_init__()
        for required in ("shape_id", "axis", "kind", "cd_offsets"):
            if required not in self.attrs:
                raise ValueError(f"FragReduceOp: missing '{required}' attr")
        axis = self.attrs["axis"]
        if axis not in ("row", "col"):
            raise ValueError(f"FragReduceOp: axis must be 'row'|'col' (got {axis!r})")
        kind = self.attrs["kind"]
        if kind not in _FRAG_REDUCE_KINDS:
            raise ValueError(
                f"FragReduceOp: kind must be one of {sorted(_FRAG_REDUCE_KINDS)} (got {kind!r})"
            )
        if len(self.operands) != 1:
            raise ValueError("FragReduceOp: expected 1 operand (in_frag)")
        (in_frag,) = self.operands
        cd_offsets = self.attrs["cd_offsets"]
        if axis == "row":
            classes = sorted({dr for dr, _ in cd_offsets})
        else:
            classes = sorted({dc for _, dc in cd_offsets})
        if len(self.results) != len(classes):
            raise ValueError(
                f"FragReduceOp: expected {len(classes)} results (one per "
                f"{axis} class from cd_offsets), got {len(self.results)}"
            )
        for r in self.results:
            if r.width != 1:
                raise ValueError(
                    f"FragReduceOp: each result must be scalar (width=1), got width={r.width}"
                )
            if r.dtype is not in_frag.dtype:
                raise TypeError(
                    f"FragReduceOp: result dtype {r.dtype} must match in_frag dtype {in_frag.dtype}"
                )

    @property
    def in_frag(self) -> Value:
        return self.operands[0]
