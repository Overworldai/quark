"""Lowerer protocol + registry.

Every backend lowerer (PTX, MSL, future SPIR-V) registers itself against
a ``DeviceFamily`` so the launcher can look one up without knowing which
backends exist. This is the one-line-per-backend extension point the
portability plan calls for.

Lowerers accept ``DeviceCaps`` in ``__init__`` and expose
``lower_module(module) -> LoweredSomething``. The ``LoweredSomething``
is backend-specific (PTX text vs MSL source); callers dispatch on the
instance type further down the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from quark.device import DeviceFamily


@runtime_checkable
class Lowerer(Protocol):
    """Structural protocol every backend lowerer must satisfy."""

    def lower_module(self, module: Any) -> Any: ...


# family -> factory(caps) -> Lowerer. A factory is either the Lowerer class
# itself (when its __init__ takes only caps) or a small adapter lambda
# that unpacks caps into backend-specific knobs.
LOWERERS: dict[DeviceFamily, Callable[[Any], Lowerer]] = {}


def register_lowerer(family: DeviceFamily) -> Callable[[Callable], Callable]:
    """Register ``factory(caps) -> Lowerer`` for ``family``.

    Usage:

        @register_lowerer(DeviceFamily.CUDA)
        class PtxLowerer:
            def __init__(self, caps): ...

    or for backends whose ctor takes more than caps:

        @register_lowerer(DeviceFamily.METAL)
        def _make(caps):
            return MslLowerer(caps, smem_aliasing=True)
    """

    def _deco(factory):
        if family in LOWERERS:
            existing = LOWERERS[family]
            raise ValueError(
                f"register_lowerer({family}): already registered to "
                f"{existing!r}; refusing to overwrite with {factory!r}"
            )
        LOWERERS[family] = factory
        return factory

    return _deco


def get_lowerer(family: DeviceFamily, caps: Any) -> Lowerer:
    """Construct the lowerer for ``family`` with ``caps``.

    Raises ``KeyError`` with the list of registered families if none
    match — easier to diagnose than a silent fallback.
    """
    try:
        factory = LOWERERS[family]
    except KeyError:
        raise KeyError(
            f"No lowerer registered for {family}. Registered: {sorted(str(f) for f in LOWERERS)}"
        ) from None
    return factory(caps)
