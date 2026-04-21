"""Backend-agnostic elementwise reference implementations."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def _ref_unary(X, op: str, out_dtype: DType):
    x = PT.astype(X, PT.float32)
    if op == "neg":
        r = -x
    elif op == "abs":
        if PT._is_mx(x):
            import mlx.core as mx

            r = mx.abs(x)
        else:
            import torch

            r = torch.abs(x)
    elif op == "exp":
        r = PT.exp(x)
    elif op == "sin":
        r = PT.sin(x)
    elif op == "cos":
        r = PT.cos(x)
    elif op == "sqrt":
        if PT._is_mx(x):
            import mlx.core as mx

            r = mx.sqrt(x)
        else:
            import torch

            r = torch.sqrt(x)
    elif op == "cast":
        r = x
    else:
        raise ValueError(f"unknown unary op: {op}")
    return PT.astype(r, out_dtype.backend)


def _ref_binary(X, Y, op: str, out_dtype: DType):
    x = PT.astype(X, PT.float32)
    y = PT.astype(Y, PT.float32)
    if op == "add":
        r = x + y
    elif op == "sub":
        r = x - y
    elif op == "mul":
        r = x * y
    elif op == "div":
        r = x / y
    else:
        raise ValueError(f"unknown binary op: {op}")
    return PT.astype(r, out_dtype.backend)


def elementwise_reference_for_spec(kernel, X, Y=None):
    s = kernel.spec
    out_dt = s.effective_out_dtype
    if s.arity == 2:
        return _ref_binary(X, Y, s.op, out_dt)
    return _ref_unary(X, s.op, out_dt)
