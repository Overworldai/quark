"""Pure-Python helpers for tensor data manipulation. No numpy dependency."""

from __future__ import annotations

import math


def flatten_list(data) -> list:
    """Flatten a nested list/tuple of numbers."""
    if isinstance(data, (int, float)):
        return [data]
    result = []
    for item in data:
        result.extend(flatten_list(item))
    return result


def infer_shape(data) -> tuple[int, ...]:
    """Infer shape from a nested list."""
    if isinstance(data, (int, float)):
        return ()
    if not data:
        return (0,)
    inner = infer_shape(data[0])
    return (len(data),) + inner


def reshape_flat_to_nested(flat: list, shape: tuple[int, ...]):
    """Reshape a flat list into nested lists matching shape."""
    if not shape:
        return flat[0] if flat else 0
    if len(shape) == 1:
        return flat[: shape[0]]
    chunk = math.prod(shape[1:])
    return [
        reshape_flat_to_nested(flat[i * chunk : (i + 1) * chunk], shape[1:])
        for i in range(shape[0])
    ]


def f32_bytes_to_bf16_bytes(f32_bytes: bytes) -> bytes:
    """Truncate f32 raw bytes → bf16. Uses ``array.array`` for speed."""
    import array as _array

    u32 = _array.array("I")
    u32.frombytes(f32_bytes)
    u16 = _array.array("H", ((v >> 16) & 0xFFFF for v in u32))
    return u16.tobytes()


def f32_to_bf16_numpy(arr):
    """Truncate f32 → bf16 using numpy. Convenience for from_numpy."""
    import numpy as np

    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16
