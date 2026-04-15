"""SSA Values in the popcorn IR.

A Value is the single-assignment result of an Op. Every Op that produces
output produces fresh Values; nothing rebinds them. Ops capture Values
by reference, not by name — textgen walks the IR and assigns
backend-specific names at lowering time.

See POPCORN_IR_PROPOSAL.md §4.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .types import DType, ValueShape

if TYPE_CHECKING:
    from .builder import Builder
    from .op import Op


# Active-Builder ContextVar — set by Builder.begin_function, read by
# Value dunders so `a * b` / `a + b` / etc. can dispatch to the
# in-flight Builder without an explicit reference. Only one Builder
# function can be open at a time (begin_function enforces it), so the
# ContextVar is never ambiguous during kernel construction.
# `default=None` is the runtime behavior; ty infers ContextVar[None]
# from that default, so we don't pass it and rely on begin_function's
# unconditional .set() before any .get(). _active_builder() raises if
# the var is unbound.
_ACTIVE_BUILDER: ContextVar[Builder | None] = ContextVar("popcorn_active_builder")


def _active_builder() -> Builder:
    try:
        bld = _ACTIVE_BUILDER.get()
    except LookupError:
        bld = None
    if bld is None:
        raise RuntimeError(
            "Value operator used outside an active Builder. "
            "Use `builder.mul(a, b)` directly when no function is open."
        )
    return bld


@dataclass(frozen=False, eq=False)
class Value:
    """An SSA value produced either by an Op or by a Function parameter.

    Equality and hashing are by identity — two distinct Values are never
    equal even if they share id/shape, because an `id` is only unique
    within one Function and textgen needs to distinguish the allocated
    objects. The `producer` back-reference is set when the producing Op
    is constructed; for Function params it stays None.

    Lifetime is scoped to the Region in which the producer lives. The
    validator checks dominance; the Builder and printer assume it.
    """

    id: int
    shape: ValueShape
    producer: Optional[Op] = field(default=None, repr=False)
    name: str = ""

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def dtype(self) -> DType:
        return self.shape.dtype

    @property
    def width(self) -> int:
        return self.shape.width

    def __repr__(self) -> str:
        tag = f"%{self.name}" if self.name else f"%{self.id}"
        return f"{tag}:{self.shape!r}"

    # ---- Operator overloading ----------------------------------------
    # Dispatches to the active Builder (ContextVar set by
    # begin_function). Python scalars on the RHS are auto-lifted to
    # constants of *this* Value's dtype — `iv * 32` works without a
    # manual const(). Left-scalar is covered by the __r*__ reflections.
    # Lifted ops participate in the Builder's CSE cache automatically.

    def _as_value(self, other: object) -> Value:
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float, bool)):
            return _active_builder().const(self.dtype, other)
        raise TypeError(f"Value operator: cannot combine Value with {type(other).__name__}")

    def __add__(self, other: object) -> Value:
        return _active_builder().add(self, self._as_value(other))

    def __radd__(self, other: object) -> Value:
        return _active_builder().add(self._as_value(other), self)

    def __sub__(self, other: object) -> Value:
        return _active_builder().sub(self, self._as_value(other))

    def __rsub__(self, other: object) -> Value:
        return _active_builder().sub(self._as_value(other), self)

    def __mul__(self, other: object) -> Value:
        return _active_builder().mul(self, self._as_value(other))

    def __rmul__(self, other: object) -> Value:
        return _active_builder().mul(self._as_value(other), self)

    def __floordiv__(self, other: object) -> Value:
        return _active_builder().div(self, self._as_value(other))

    def __rfloordiv__(self, other: object) -> Value:
        return _active_builder().div(self._as_value(other), self)

    def __mod__(self, other: object) -> Value:
        return _active_builder().rem(self, self._as_value(other))

    def __rmod__(self, other: object) -> Value:
        return _active_builder().rem(self._as_value(other), self)

    def __lshift__(self, other: object) -> Value:
        return _active_builder().shl(self, self._as_value(other))

    def __rshift__(self, other: object) -> Value:
        return _active_builder().shr(self, self._as_value(other))

    def __and__(self, other: object) -> Value:
        return _active_builder().and_(self, self._as_value(other))

    def __rand__(self, other: object) -> Value:
        return _active_builder().and_(self._as_value(other), self)

    def __or__(self, other: object) -> Value:
        return _active_builder().or_(self, self._as_value(other))

    def __ror__(self, other: object) -> Value:
        return _active_builder().or_(self._as_value(other), self)

    def __xor__(self, other: object) -> Value:
        return _active_builder().xor(self, self._as_value(other))

    def __rxor__(self, other: object) -> Value:
        return _active_builder().xor(self._as_value(other), self)

    def __neg__(self) -> Value:
        return _active_builder().neg(self)

    def __abs__(self) -> Value:
        return _active_builder().abs(self)


class ValueAllocator:
    """Monotonic id source for Values within a single Function.

    The Function owns one allocator; Builder methods ask it for a new
    id every time they mint a Value.
    """

    def __init__(self) -> None:
        self._next_id = 0

    def fresh(
        self,
        shape: ValueShape,
        producer: Optional[Op] = None,
        name: str = "",
    ) -> Value:
        v = Value(id=self._next_id, shape=shape, producer=producer, name=name)
        self._next_id += 1
        return v

    def peek(self) -> int:
        return self._next_id
