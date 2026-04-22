"""Numpy → device tensor boundary helpers.

``make_tensors_numpy`` returns a ``{name: np.ndarray}`` dict in the
TensorDecl carrier dtypes. Tools (fuzz, bench, autotune) need device
tensors to feed the launcher. This module provides the one-call
conversion in both directions.
"""

from __future__ import annotations

import sys
from typing import Any

_IS_METAL = sys.platform == "darwin"


def numpy_to_device_dict(kernel_cls, spec, inputs_np: dict[str, Any]) -> dict[str, Any]:
    """Convert every entry in ``inputs_np`` to a backend-native tensor.

    Dispatch:
      * CUDA (non-Metal) → ``PopcornTensor.from_numpy(arr, dtype=<short>)``.
      * Metal → ``mx.array`` (bf16 carriers widen via u16 view pattern).

    ``kernel_cls.TENSORS`` + ``spec`` tell us the *intended* dtype of each
    buffer so bf16-as-u16 / fp8-as-u8 carriers end up with the right
    on-device dtype regardless of the numpy dtype we were handed.
    """
    # Map TENSORS name → declared popcorn dtype short string.
    name_to_dtype: dict[str, str] = {}
    for decl in kernel_cls.TENSORS:
        dt = decl.dtype(spec, None) if callable(decl.dtype) else decl.dtype
        # DType enum — use its string .value ("bf16" etc.).
        dt_str = dt.value if hasattr(dt, "value") else str(dt)
        name_to_dtype[decl.name] = dt_str

    out: dict[str, Any] = {}
    if _IS_METAL:
        import mlx.core as mx
        import numpy as np

        _PC_TO_MX = {
            "f32": mx.float32,
            "f16": mx.float16,
            "bf16": mx.bfloat16,
            "s32": mx.int32,
            "s64": mx.int64,
            "u8": mx.uint8,
            "s8": mx.int8,
            "u32": mx.uint32,
            "u16": mx.uint16,
        }
        for name, arr in inputs_np.items():
            target = name_to_dtype.get(name)
            if target == "bf16" and arr.dtype == np.uint16:
                # Carrier → bf16 via u16 view.
                out[name] = mx.array(arr).view(mx.bfloat16)
            elif target in _PC_TO_MX:
                out[name] = mx.array(arr).astype(_PC_TO_MX[target])
            else:
                # Dtype not in the lookup (e.g. e4m3) — pass bytes through
                # as u8; MLX has no fp8 today so Metal path can't run
                # fp8-output kernels anyway.
                out[name] = mx.array(arr)
        return out

    from popcorn.runtime.tensor import PopcornTensor

    for name, arr in inputs_np.items():
        target = name_to_dtype.get(name)
        out[name] = PopcornTensor.from_numpy(arr, dtype=target)
    return out
