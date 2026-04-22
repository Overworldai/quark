"""Elementwise numpy reference."""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def elementwise_reference_numpy(spec, *, X, Y=None, Out=None):
    del Out  # reference writes to a fresh buffer
    hint = spec.dtype.value
    x = to_f32_numpy(X, dtype_hint=hint)

    if spec.arity == 2:
        y = to_f32_numpy(Y, dtype_hint=hint)
        if spec.op == "add":
            r = x + y
        elif spec.op == "sub":
            r = x - y
        elif spec.op == "mul":
            r = x * y
        elif spec.op == "div":
            r = x / y
        else:
            raise ValueError(f"unknown binary op: {spec.op}")
    elif spec.op == "neg":
        r = -x
    elif spec.op == "abs":
        r = np.abs(x)
    elif spec.op == "exp":
        r = np.exp(x)
    elif spec.op == "sin":
        r = np.sin(x)
    elif spec.op == "cos":
        r = np.cos(x)
    elif spec.op == "sqrt":
        r = np.sqrt(x)
    elif spec.op == "cast":
        r = x
    else:
        raise ValueError(f"unknown unary op: {spec.op}")

    return astype_numpy(r, spec.effective_out_dtype)
