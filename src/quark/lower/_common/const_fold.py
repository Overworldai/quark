"""Static evaluation of IR Values to Python ints.

Used by backend lowerers that need to const-fold loop bounds at
lower time — primarily by the ``ForLoopOp(unroll=True)`` paths in
PTX / Metal / NAX, which Python-unroll the body and need the
iteration count as a Python int.

Supported producer ops:

  * ``ConstOp`` — direct literal lookup.
  * ``BlockDimOp`` — reads the lowerer-provided ``local_size``
    tuple (each backend's lowerer passes its own resolved value).
  * ``ArithOp`` — recursively evaluates operands for the simple
    integer ops (add/sub/mul/div/rem). Returns ``None`` if any
    operand fails to fold.

The helper returns ``None`` (rather than raising) when the trace
gives up — callers decide whether that's an error. The unroll
path in PTX raises NotImplementedError with a clear message
pointing at the offending op chain; SPV/OCL never call this
because their unroll is a downstream hint, not a Python-unroll.
"""

from __future__ import annotations

from typing import Any

from quark.ir.op import ArithOp, BlockDimOp, ConstOp


_DIM_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def _arith_eval(kind: str, a: int, b: int) -> int | None:
    if kind == "add":
        return a + b
    if kind == "sub":
        return a - b
    if kind == "mul":
        return a * b
    if kind == "div":
        if b == 0:
            return None
        return a // b
    if kind == "rem":
        if b == 0:
            return None
        return a % b
    return None


def eval_const_int(value: Any, local_size: tuple[int, int, int]) -> int | None:
    """Resolve an IR ``Value`` to a Python int when statically known.

    ``local_size`` is the lowerer's resolved workgroup size — needed
    to fold ``BlockDimOp`` operands (the IR carries them as runtime
    values, but every backend resolves them to a compile-time
    constant from the kernel's local_size attr).

    Returns ``None`` when the trace hits an unsupported producer
    (e.g. ``ThreadIdxOp``, ``LoadOp`` — genuinely runtime values).
    """
    if value is None:
        return None
    prod = getattr(value, "producer", None)
    if prod is None:
        return None
    if isinstance(prod, ConstOp):
        v = prod.attrs.get("value")
        if isinstance(v, (int, bool)):
            return int(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return None
    if isinstance(prod, BlockDimOp):
        dim = prod.attrs.get("dim", "x")
        idx = _DIM_TO_INDEX.get(dim)
        if idx is None:
            return None
        return int(local_size[idx])
    if isinstance(prod, ArithOp):
        kind = prod.attrs.get("kind", "")
        if len(prod.operands) < 2:
            return None
        a = eval_const_int(prod.operands[0], local_size)
        b = eval_const_int(prod.operands[1], local_size)
        if a is None or b is None:
            return None
        return _arith_eval(kind, a, b)
    return None


__all__ = ["eval_const_int"]
