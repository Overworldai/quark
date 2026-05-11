"""TensorDecl, C — declarative tensor and constant specs.

TensorDecl is the single-source-of-truth for a kernel's parameter
tensors; C is the auto-typed constant wrapper. ``_to_value`` coerces
literals / C specs into IR Values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import quark.lang as qk
from quark.ir import Builder, DType, Value

# Role strings for TensorDecl — drive default make_tensors construction
# (randn vs zeros) when base.Kernel.make_tensors walks the manifest.
_ROLES = frozenset({"in", "out", "scratch"})


@dataclass(frozen=True)
class TensorDecl:
    """One entry in a kernel's tensor manifest.

    Replaces the three-way duplication between ``spec`` fields,
    ``make_tensors`` torch calls, and ``emit() → ctx.tensor(...)``
    declarations. Each kernel lists its parameter tensors once as a
    ``TensorDecl``; ``ctx.declare_tensors()`` walks the list at emit
    time, and base ``Kernel.make_tensors`` can walk the same list to
    build test tensors with matching shapes.

    ``dtype`` and ``shape`` each accept either a literal or a
    callable taking ``(spec, config)``. Callables are the common case
    for shapes derived from spec properties (``lambda s, c:
    (s.total_slots, s.H)``) or configs (``lambda s, c: (s.M, c.BN)``).

    ``role`` is one of ``{"in", "out", "scratch"}`` and controls how
    a default ``make_tensors`` builds the torch tensor (randn vs
    zeros). Kernels that need custom construction can still override
    ``make_tensors`` entirely.
    """

    name: str
    dtype: DType | Callable[[Any, Any], DType]
    shape: tuple[int, ...] | Callable[[Any, Any], tuple[int, ...]]
    role: str = "in"

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(
                f"TensorDecl {self.name!r}: role must be one of {_ROLES}, got {self.role!r}"
            )

    def resolve_dtype(self, spec: Any, config: Any) -> DType:
        dt = self.dtype
        if isinstance(dt, DType):
            return dt
        return dt(spec, config)

    def resolve_shape(self, spec: Any, config: Any) -> tuple[int, ...]:
        # ty's narrowing after callable() leaves a residual type it
        # can't unify with tuple[int, ...]; cast explicitly.
        sh = self.shape
        if callable(sh):
            fn = cast(Callable[[Any, Any], tuple[int, ...]], sh)
            return tuple(fn(spec, config))
        return cast(tuple[int, ...], sh)


@dataclass(frozen=True)
class C:
    """Constant spec with automatic type inference.

    Type rules: int → U32, negative int → S32, float → F32.
    Pass dtype= to override.
    """

    value: int | float
    dtype: DType | None = None

    def resolve_dtype(self) -> DType:
        if self.dtype is not None:
            return self.dtype
        if isinstance(self.value, float):
            return DType.F32
        if self.value < 0:
            return DType.S32
        return DType.U32

    def emit(self, bld: Builder) -> Value:
        return qk.const(self.resolve_dtype(), self.value)


def _to_value(bld: Builder, v: Value | C | int | float) -> Value:
    """Coerce a literal or C spec to an IR Value."""
    if isinstance(v, Value):
        return v
    if isinstance(v, C):
        return v.emit(bld)
    if isinstance(v, float):
        return qk.const(DType.F32, v)
    if isinstance(v, int):
        dt = DType.S32 if v < 0 else DType.U32
        return qk.const(dt, v)
    raise TypeError(f"Cannot coerce {type(v).__name__} to Value")
