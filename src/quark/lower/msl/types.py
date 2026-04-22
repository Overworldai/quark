"""DType -> MSL scalar type names.

Maps quark IR DTypes to the MSL type strings used in variable
declarations, casts, and buffer pointer types. Stays minimal — no
vector type helpers until Phase 2 needs them.
"""

from __future__ import annotations

from quark.ir.types import DType

# MSL scalar type name for each IR DType.
_MSL_TYPE: dict[DType, str] = {
    DType.F64: "double",  # Metal doesn't support double on GPU, but keep for completeness
    DType.F32: "float",
    DType.F16: "half",
    DType.BF16: "bfloat16_t",
    DType.E4M3: "fp8_e4m3",  # MLX fp8 software struct
    DType.E5M2: "fp8_e5m2",  # MLX fp8 software struct
    DType.U8: "uchar",
    DType.S8: "char",
    DType.U16: "ushort",
    DType.S16: "short",
    DType.U32: "uint",
    DType.S32: "int",
    DType.U64: "ulong",
    DType.S64: "long",
    DType.B16: "ushort",
    DType.B32: "uint",
    DType.B64: "ulong",
    DType.PRED: "bool",
}


def msl_type(dtype: DType) -> str:
    """Return the MSL scalar type string for an IR DType."""
    try:
        return _MSL_TYPE[dtype]
    except KeyError as e:
        raise NotImplementedError(f"MSL: no type mapping for {dtype}") from e


def msl_vec_type(dtype: DType, width: int) -> str:
    """Return the MSL vector type string (e.g. float4, half2)."""
    base = msl_type(dtype)
    # MSL native vectors: float2/3/4, half2/3/4, uint2/3/4, etc.
    if width in (2, 3, 4):
        return f"{base}{width}"
    raise NotImplementedError(f"MSL: no native vector type for {dtype} x {width}")
