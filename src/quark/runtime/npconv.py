"""Numpy conversion helpers for quark dtypes.

Single entry point for "give me an f32 numpy array from whatever
tensor-ish I was handed" — used by ``check_correctness`` (the
autotune / fuzz correctness gate) and by ``RefCache`` when it needs
to normalize device outputs against cached numpy references.

Dtypes that numpy doesn't natively carry (bf16, e4m3, e5m2) travel as
raw-byte carrier types (u16 / u8) and widen to f32 through the
helpers below. Narrowing (f32 → bf16 / fp8) is symmetric — bf16 is
straightforward bit truncation, fp8 is LUT-based since the format is
irregular.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------
# bf16 ↔ f32
# ---------------------------------------------------------------


def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    """Widen bf16-in-u16 → f32. bf16 is the top 16 bits of an IEEE
    binary32 word, so the widen is a single left-shift into u32
    followed by a bit-reinterpret as f32."""
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    """Narrow f32 → bf16-in-u16 by truncation. No round-to-nearest —
    matches the ``packed_convert`` / ``cvt.rn.bf16.f32`` PTX path at
    the bit level only when the input bit pattern has the low 16
    bits already zero. Good enough for reference oracles (we accept
    up to ~1e-3 cosine delta on bf16 outputs anyway)."""
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    return (u32 >> np.uint32(16)).astype(np.uint16)


# ---------------------------------------------------------------
# fp8 → f32 (LUT)
# ---------------------------------------------------------------


def _decode_e4m3(byte: int) -> float:
    """Decode one e4m3 byte. Layout: 1 sign, 4 exp (bias 7), 3 mant.
    No infinities — s.1111.111 is the one NaN encoding (per OCP FP8)."""
    sign = (byte >> 7) & 1
    exp = (byte >> 3) & 0xF
    mant = byte & 0x7
    if exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0:
        val = (mant / 8.0) * 2.0 ** (1 - 7)  # subnormal
    else:
        val = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    return -val if sign else val


def _decode_e5m2(byte: int) -> float:
    """Decode one e5m2 byte. Layout: 1 sign, 5 exp (bias 15), 2 mant.
    Has infinities (s.11111.00) and NaNs (s.11111.<nonzero>)."""
    sign = (byte >> 7) & 1
    exp = (byte >> 2) & 0x1F
    mant = byte & 0x3
    if exp == 0x1F:
        if mant == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")
    if exp == 0:
        val = (mant / 4.0) * 2.0 ** (1 - 15)  # subnormal
    else:
        val = (1.0 + mant / 4.0) * 2.0 ** (exp - 15)
    return -val if sign else val


def _build_lut(decode) -> np.ndarray:
    return np.array([decode(b) for b in range(256)], dtype=np.float32)


_E4M3_LUT: np.ndarray | None = None
_E5M2_LUT: np.ndarray | None = None


def e4m3_u8_to_f32(u8: np.ndarray) -> np.ndarray:
    global _E4M3_LUT
    if _E4M3_LUT is None:
        _E4M3_LUT = _build_lut(_decode_e4m3)
    return _E4M3_LUT[u8]


def e5m2_u8_to_f32(u8: np.ndarray) -> np.ndarray:
    global _E5M2_LUT
    if _E5M2_LUT is None:
        _E5M2_LUT = _build_lut(_decode_e5m2)
    return _E5M2_LUT[u8]


# ---------------------------------------------------------------
# f32 → fp8 (LUT-free, per-element encode)
# ---------------------------------------------------------------


def _encode_e4m3_scalar(x: float) -> int:
    import math

    if x != x:  # NaN
        return 0x7F  # canonical NaN encoding
    if x == 0.0:
        # Python scalar ``1.0 / 0.0`` raises ZeroDivisionError regardless
        # of the zero's sign (unlike IEEE 754 / numpy, which yield ±inf).
        # Use ``math.copysign`` to recover the sign bit without dividing.
        return 0x80 if math.copysign(1.0, x) < 0 else 0x00
    sign = 1 if x < 0 else 0
    ax = abs(x)
    # Clamp to max representable e4m3 (448) per OCP FP8.
    if ax >= 448.0:
        return (sign << 7) | 0x7E  # ±448 (not NaN)
    exp = math.floor(math.log2(ax))
    # Unbiased exp range for e4m3: subnormal for exp<-6, normal up to 8.
    if exp < -6:
        # subnormal
        mant_f = ax / 2.0 ** (1 - 7) * 8.0
        mant = round(mant_f) & 0x7
        return (sign << 7) | mant
    mant_f = (ax / 2.0**exp - 1.0) * 8.0
    mant = round(mant_f)
    if mant == 8:
        mant = 0
        exp += 1
    biased = exp + 7
    if biased >= 0xF and mant == 0x7:
        # would encode as NaN; saturate one mantissa below
        mant = 0x6
    return (sign << 7) | ((biased & 0xF) << 3) | (mant & 0x7)


def f32_to_e4m3_u8(arr: np.ndarray) -> np.ndarray:
    """Narrow f32 → e4m3-in-u8. Per-element Python — good enough for
    reference oracles; the kernel path uses hardware ``cvt.satfinite.e4m3x2.f32``."""
    flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    out = np.array([_encode_e4m3_scalar(float(v)) for v in flat], dtype=np.uint8)
    return out.reshape(arr.shape)


# ---------------------------------------------------------------
# Generic "to f32 numpy"
# ---------------------------------------------------------------


def to_f32_numpy(x: Any, *, dtype_hint: str | None = None) -> np.ndarray:
    """Normalize any tensor-ish input to a fresh f32 numpy array.

    Inputs:
      * ``numpy.ndarray`` — returned as f32 (widened via
        ``dtype_hint`` when the numpy dtype alone doesn't disambiguate
        a carrier type, e.g. ``dtype_hint="bf16"`` on a u16 array).
      * ``QuarkTensor`` — copied to host via ``to_bytes()`` +
        reinterpret based on the tensor's dtype string.
      * plain Python scalars / lists — wrapped via ``np.asarray``.

    The output is a fresh array; callers can mutate it without
    aliasing the source.
    """
    # numpy fast-path
    if isinstance(x, np.ndarray):
        if x.dtype == np.float32:
            return x.copy() if not x.flags.owndata else x
        if x.dtype == np.uint16 and dtype_hint == "bf16":
            return bf16_u16_to_f32(x)
        if x.dtype == np.uint8 and dtype_hint == "e4m3":
            return e4m3_u8_to_f32(x)
        if x.dtype == np.uint8 and dtype_hint == "e5m2":
            return e5m2_u8_to_f32(x)
        return x.astype(np.float32, copy=False)

    # QuarkTensor duck-type
    if (
        hasattr(x, "to_bytes")
        and hasattr(x, "dtype")
        and isinstance(getattr(x, "dtype", None), str)
    ):
        shape = tuple(x.shape)
        raw = x.to_bytes()
        dt = x.dtype
        if dt == "f32":
            return np.frombuffer(raw, np.float32).reshape(shape).copy()
        if dt == "f16":
            return np.frombuffer(raw, np.float16).reshape(shape).astype(np.float32)
        if dt == "bf16":
            return bf16_u16_to_f32(np.frombuffer(raw, np.uint16)).reshape(shape)
        if dt == "e4m3":
            return e4m3_u8_to_f32(np.frombuffer(raw, np.uint8)).reshape(shape)
        if dt == "e5m2":
            return e5m2_u8_to_f32(np.frombuffer(raw, np.uint8)).reshape(shape)
        if dt == "s32":
            return np.frombuffer(raw, np.int32).reshape(shape).astype(np.float32)
        if dt == "s64":
            return np.frombuffer(raw, np.int64).reshape(shape).astype(np.float32)
        if dt == "u8":
            return np.frombuffer(raw, np.uint8).reshape(shape).astype(np.float32)
        if dt == "s8":
            return np.frombuffer(raw, np.int8).reshape(shape).astype(np.float32)
        if dt in ("u16", "u32"):
            nt = np.uint16 if dt == "u16" else np.uint32
            return np.frombuffer(raw, nt).reshape(shape).astype(np.float32)
        raise TypeError(f"to_f32_numpy: unsupported QuarkTensor dtype {dt!r}")

    # last resort — list / scalar
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------
# f32 → carrier dtype (for make_tensors_numpy inputs / expected outputs)
# ---------------------------------------------------------------


def astype_numpy(arr_f32: np.ndarray, target) -> np.ndarray:
    """Cast f32 numpy array to the numpy carrier for ``target``.

    ``target`` may be a ``quark.ir.DType`` enum, a short string
    (``"bf16"``), or a numpy dtype. The returned array's numpy
    dtype is:
      * ``f32``  → ``np.float32``
      * ``f16``  → ``np.float16``
      * ``bf16`` → ``np.uint16`` (carries the top-16 bits of the f32)
      * ``e4m3`` / ``e5m2`` → ``np.uint8``
      * ``s32``  → ``np.int32`` (round then cast)
      * ``s64``  → ``np.int64``
      * ``u8`` / ``s8`` → ``np.uint8`` / ``np.int8`` (clamp + cast)

    The carrier convention mirrors ``QuarkTensor.from_numpy`` — pass
    the result straight to ``from_numpy(arr, dtype=<short>)`` and the
    bits go to device unchanged.
    """
    target_str = _target_to_short(target)
    if target_str == "f32":
        return arr_f32.astype(np.float32, copy=False)
    if target_str == "f16":
        return arr_f32.astype(np.float16)
    if target_str == "bf16":
        return f32_to_bf16_u16(arr_f32)
    if target_str == "e4m3":
        return f32_to_e4m3_u8(arr_f32)
    if target_str == "s32":
        return np.rint(arr_f32).astype(np.int32)
    if target_str == "s64":
        return np.rint(arr_f32).astype(np.int64)
    if target_str == "u8":
        return np.clip(np.rint(arr_f32), 0, 255).astype(np.uint8)
    if target_str == "s8":
        return np.clip(np.rint(arr_f32), -128, 127).astype(np.int8)
    if target_str == "u16":
        return np.clip(np.rint(arr_f32), 0, 65535).astype(np.uint16)
    if target_str == "u32":
        return np.clip(np.rint(arr_f32), 0, 2**32 - 1).astype(np.uint32)
    raise TypeError(f"astype_numpy: unsupported target {target!r}")


def _target_to_short(target) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "value") and isinstance(target.value, str):
        return target.value  # DType enum
    raise TypeError(f"_target_to_short: can't resolve {target!r}")


def zeros_for_dtype(shape, target) -> np.ndarray:
    """Allocate a zero-filled numpy array in the carrier for ``target``.
    Equivalent to ``astype_numpy(np.zeros(shape, f32), target)`` but
    skips the cast step for dtypes where the carrier's zero is the
    representation's zero (which is every carrier we support)."""
    target_str = _target_to_short(target)
    carrier_np = {
        "f32": np.float32,
        "f16": np.float16,
        "bf16": np.uint16,
        "e4m3": np.uint8,
        "e5m2": np.uint8,
        "s32": np.int32,
        "s64": np.int64,
        "u8": np.uint8,
        "s8": np.int8,
        "u16": np.uint16,
        "u32": np.uint32,
    }.get(target_str)
    if carrier_np is None:
        raise TypeError(f"zeros_for_dtype: unsupported target {target!r}")
    return np.zeros(shape, dtype=carrier_np)
