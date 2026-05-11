"""Numpy → device tensor boundary helpers.

``make_tensors_numpy`` returns a ``{name: np.ndarray}`` dict in the
TensorDecl carrier dtypes. Tools (fuzz, bench, autotune) need device
tensors to feed the launcher. This module provides the one-call
conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Mapping from quark dtype short strings to numpy carrier dtypes.
_PC_TO_NP = {
    "f32": np.float32,
    "f16": np.float16,
    "bf16": np.uint16,  # bf16 stored as uint16
    "s32": np.int32,
    "s64": np.int64,
    "u8": np.uint8,
    "s8": np.int8,
    "u32": np.uint32,
    "u16": np.uint16,
}


def numpy_to_device_dict(kernel_cls, spec, inputs_np: dict[str, Any]) -> dict[str, Any]:
    """Convert every entry in ``inputs_np`` to a ``QuarkTensor``.

    ``kernel_cls.TENSORS`` + ``spec`` tell us the *intended* dtype of each
    buffer so bf16-as-u16 / fp8-as-u8 carriers end up with the right
    on-device dtype regardless of the numpy dtype we were handed.
    """
    from quark.runtime.tensor import QuarkTensor

    name_to_dtype: dict[str, str] = {}
    for decl in kernel_cls.TENSORS:
        dt = decl.dtype(spec, None) if callable(decl.dtype) else decl.dtype
        dt_str = dt.value if hasattr(dt, "value") else str(dt)
        name_to_dtype[decl.name] = dt_str

    out: dict[str, Any] = {}
    for name, arr in inputs_np.items():
        target = name_to_dtype.get(name, "f32")
        if target in _PC_TO_NP and arr.dtype != _PC_TO_NP[target]:
            arr = np.asarray(arr).astype(_PC_TO_NP[target])
        out[name] = QuarkTensor.from_numpy(arr, dtype=target)
    return out
