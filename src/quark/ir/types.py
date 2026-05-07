"""Types for the quark IR.

DType, MemSpace, ValueShape, and the scalar/buffer Type variants are the
building blocks every op, value, and tensor in the IR refers to. They are
deliberately backend-neutral: no PTX register classes, no MSL spellings,
no OpenCL vector aliases. Lowerers map these to target syntax.

See QUARK_IR_PROPOSAL.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DType(str, Enum):  # noqa: UP042
    """Scalar element types expressible in any of our backends.

    Bit-typed variants (B16/B32/B64) exist for raw storage and for matmul
    fragments where the contents aren't a portable float/int.

    Inherits from ``str`` so ``DType.BF16 == "bf16"`` holds and JSON
    round-trips work without a custom encoder — Specs can store
    ``DType`` directly while still accepting string values from
    ``Problem.params``.
    """

    # Float
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    F64 = "f64"
    # Small float (FP8)
    E4M3 = "e4m3"
    E5M2 = "e5m2"
    # Unsigned int
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    # Signed int
    S8 = "s8"
    S16 = "s16"
    S32 = "s32"
    S64 = "s64"
    # Bit-typed
    B16 = "b16"
    B32 = "b32"
    B64 = "b64"
    # Predicate (boolean)
    PRED = "pred"

    @property
    def bytes(self) -> int:
        """Size of one scalar element in bytes. PRED is 1 byte by convention."""
        return _DTYPE_BYTES[self]

    @property
    def bits(self) -> int:
        return self.bytes * 8

    @property
    def elems_per_16B(self) -> int:
        """Number of elements that fit in a 16-byte cp.async line.

        Handy for cp.async line arithmetic (16 // dtype.bytes is a
        ubiquitous pattern across tile loaders and vectorized
        epilogues). Returns 0 for PRED since PRED isn't meaningfully
        used in 16B memory transfers.
        """
        if self.bytes == 0:
            return 0
        return 16 // self.bytes

    @property
    def is_float(self) -> bool:
        return self in _FLOAT_DTYPES

    @property
    def is_int(self) -> bool:
        return self in _INT_DTYPES

    @property
    def is_signed_int(self) -> bool:
        return self in _SIGNED_INT_DTYPES

    @property
    def is_bit(self) -> bool:
        return self in _BIT_DTYPES

    @classmethod
    def from_backend(cls, dt) -> DType:
        """Backend-native dtype → ``DType``.

        Accepts:
          * ``QuarkTensor``-style short strings (``"bf16"``, ``"f32"``, ...).
          * ``numpy.dtype`` (via ``.name`` → ``"float32"`` etc.).
          * Objects with a ``.quark_dtype`` attribute (e.g. legacy
            tagged ndarrays that carry the intended quark dtype).

        On Metal, bf16 is stored as uint16 in numpy. If the object has
        a ``.quark_dtype`` attribute, that takes precedence over the
        numpy dtype name — this lets bf16-as-uint16 carriers resolve
        correctly."""
        # Tagged objects with .quark_dtype: use the tag.
        quark_dt = getattr(dt, "quark_dtype", None)
        if quark_dt is not None:
            return cls(quark_dt)
        # QuarkTensor-style short string ("bf16", "f32", ...).
        if isinstance(dt, str):
            try:
                return cls(dt)
            except ValueError as e:
                raise ValueError(f"DType.from_backend: no mapping for {dt!r}") from e
        # Array-like whose .dtype is a short string (QuarkTensor).
        inner_str = getattr(dt, "dtype", None)
        if isinstance(inner_str, str):
            try:
                return cls(inner_str)
            except ValueError:
                pass  # fall through to numpy name map
        _NAME_MAP = {
            "float32": cls.F32,
            "float16": cls.F16,
            "bfloat16": cls.BF16,
            "int32": cls.S32,
            "int64": cls.S64,
            "uint8": cls.U8,
            "int8": cls.S8,
            "uint16": cls.U16,
            "uint32": cls.U32,
        }
        # Array-like object (numpy ndarray, etc.) — resolve via its
        # .dtype attribute. Check .quark_dtype first (handled above),
        # then fall through to numpy dtype name mapping.
        inner_dtype = getattr(dt, "dtype", dt)
        name = getattr(inner_dtype, "name", str(inner_dtype))
        if name in _NAME_MAP:
            return _NAME_MAP[name]
        raise ValueError(f"DType.from_backend: no mapping for {dt!r}")

    @classmethod
    def coerce(cls, value: DType | str | None) -> DType | None:
        """Narrow a user-supplied ``DType | str | None`` to ``DType | None``.

        Used by ``spec_from_tensors`` and functional wrappers to accept
        either form without forcing callers through Enum construction.
        """
        if value is None or isinstance(value, DType):
            return value
        return cls(value)

    def __repr__(self) -> str:
        return f"DType.{self.name}"


_DTYPE_BYTES: dict[DType, int] = {
    DType.F64: 8,
    DType.F32: 4,
    DType.F16: 2,
    DType.BF16: 2,
    DType.E4M3: 1,
    DType.E5M2: 1,
    DType.U8: 1,
    DType.S8: 1,
    DType.U16: 2,
    DType.S16: 2,
    DType.U32: 4,
    DType.S32: 4,
    DType.U64: 8,
    DType.S64: 8,
    DType.B16: 2,
    DType.B32: 4,
    DType.B64: 8,
    DType.PRED: 1,
}

_FLOAT_DTYPES = frozenset({DType.F64, DType.F32, DType.F16, DType.BF16, DType.E4M3, DType.E5M2})
_INT_DTYPES = frozenset(
    {
        DType.U8,
        DType.S8,
        DType.U16,
        DType.S16,
        DType.U32,
        DType.S32,
        DType.U64,
        DType.S64,
    }
)
_SIGNED_INT_DTYPES = frozenset({DType.S8, DType.S16, DType.S32, DType.S64})
_BIT_DTYPES = frozenset({DType.B16, DType.B32, DType.B64})


class MemSpace(Enum):
    """Where a buffer lives. Maps to gmem/smem/rmem/const on each backend."""

    GLOBAL = "global"
    SHARED = "shared"
    PRIVATE = "private"
    CONSTANT = "constant"
    PARAM = "param"

    def __repr__(self) -> str:
        return f"MemSpace.{self.name}"


# Vector widths we permit for short SIMD-style Values. Anything wider lives
# in a MemRef, not a Value.
#
# Width 32 is the per-lane element count for NAX shapes with M=32 (e.g.
# m32n32k16 — 4 sub-fragments × 8 elements each). Apple's compiler accepts
# vec<T, 32> via cooperative_tensor and the existing IR ops are parametric
# over c_regs, so the only friction is this validity gate.
_VALID_VECTOR_WIDTHS = frozenset({1, 2, 3, 4, 8, 16, 32})


@dataclass(frozen=True)
class ValueShape:
    """Shape of an SSA Value: scalar (width=1) or short vector.

    Larger collections of data are tensor loads/stores, not Values.
    """

    dtype: DType
    width: int = 1

    def __post_init__(self) -> None:
        if self.width not in _VALID_VECTOR_WIDTHS:
            raise ValueError(
                f"ValueShape width must be one of {sorted(_VALID_VECTOR_WIDTHS)}, got {self.width}"
            )

    @property
    def is_scalar(self) -> bool:
        return self.width == 1

    @property
    def is_vector(self) -> bool:
        return self.width > 1

    @property
    def bytes(self) -> int:
        return self.dtype.bytes * self.width

    def __repr__(self) -> str:
        if self.is_scalar:
            return f"{self.dtype.value}"
        return f"{self.dtype.value}x{self.width}"


# ---------------------------------------------------------------------------
# Parameter types (kernel function parameters)
# ---------------------------------------------------------------------------


class Type:
    """Base marker for kernel-parameter types. Not used for SSA Values."""


@dataclass(frozen=True)
class ScalarType(Type):
    """A scalar kernel parameter (e.g. `u32 stride`)."""

    dtype: DType

    def __repr__(self) -> str:
        return f"ScalarType({self.dtype.value})"


@dataclass(frozen=True)
class BufferType(Type):
    """A pointer kernel parameter (e.g. `bf16* X`).

    `space` tells the backend which address space to qualify it with;
    `dtype` is the element type seen at load/store sites.
    """

    dtype: DType
    space: MemSpace = MemSpace.GLOBAL

    def __repr__(self) -> str:
        return f"BufferType({self.dtype.value}, {self.space.name.lower()})"
