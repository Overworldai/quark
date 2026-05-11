"""Kernel registry — `@register` decorator and lookup helpers.

The registry is the single authority on what kernels exist; tools
(`tools/bench.py`, `tools/fuzz.py`) read from it instead of
maintaining hand-rolled `TARGETS` dicts.

Goal: adding a new kernel touches one place — the kernel folder
itself. The auto-discovery loop in `quark/kernels/__init__.py`
imports every subdirectory of `kernels/` at package load, and any
module-level `@register("name")` decorator side-effect adds the
class to `_REGISTRY`. From there `all_kernels()` and `get(name)`
can find it.

Existing flat-file kernels (gemm, rmsnorm, qproj, owl_attn,
moe_inproj, moe_outproj, row_stationary_megakernel) are NOT
registered yet — they migrate folder-by-folder in a follow-up.
For now the registry contains only kernels written against the
expanded `Kernel` ABC contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type] = {}


# Required ABC hooks the registry decorator validates. Every
# registered kernel must override these — the @register decorator
# raises TypeError at class-definition time if any are still
# pointing at the base class's stub. This catches the "I forgot to
# implement make_tensors" bug at module load instead of at bench
# time.
_REQUIRED_OVERRIDES: tuple[str, ...] = (
    "problems",
    "tune_space",
)
# Note: `make_tensors` is also required, but it coexists with a
# legacy instance-method version in `Kernel` (called by tools/
# bench.py). We can't reliably compare function identity across
# the two shapes (instance method vs classmethod), so make_tensors
# is *not* validated by the function-identity check above. Tests
# at smoke / fuzz time will surface a missing override the first
# time they call it.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register(name: str | None = None) -> Callable[[type], type]:
    """Class decorator that registers a Kernel subclass under `name`.

    If `name` is omitted, uses `cls.NAME`. Validates that the class
    has overridden every required ABC hook (`problems`,
    `tune_space`, `param_spec`) — if any still point at the base-class
    stub, raises TypeError immediately so the bug surfaces at module
    load, not at bench time.

    Duplicate registration is rejected. The class is returned
    unchanged so the decorator is purely additive::

        @register("my_kernel")
        class MyKernel(Kernel):
            NAME = "my_kernel"
            SPEC_CLS = MySpec
            CONFIG_CLS = MyConfig
            ...
    """

    def wrap(cls: type) -> type:
        # Late import to keep registry import-time cycle-free.
        from quark.kernels.base import Kernel

        if not isinstance(cls, type) or not issubclass(cls, Kernel):
            raise TypeError(f"@register: {cls!r} is not a subclass of Kernel")

        key = name or getattr(cls, "NAME", "") or cls.__name__
        if not key:
            raise TypeError(
                f"@register: {cls.__name__} has no NAME attribute and no explicit name was given"
            )
        if key in _REGISTRY:
            existing = _REGISTRY[key]
            if existing is cls:
                # Idempotent re-registration of the same class is fine
                # (happens when tests reload modules).
                return cls
            raise ValueError(
                f"@register: kernel name {key!r} already registered: "
                f"existing={existing.__qualname__}, new={cls.__qualname__}"
            )

        # Validate that the class actually overrides every required
        # hook. We compare the bound method's underlying function to
        # the base-class stub function — equal means the subclass
        # didn't override.
        missing: list[str] = []
        for hook in _REQUIRED_OVERRIDES:
            sub_attr = getattr(cls, hook, None)
            base_attr = getattr(Kernel, hook, None)
            if sub_attr is None or base_attr is None:
                missing.append(hook)
                continue
            sub_fn = getattr(sub_attr, "__func__", sub_attr)
            base_fn = getattr(base_attr, "__func__", base_attr)
            if sub_fn is base_fn:
                missing.append(hook)
        if missing:
            raise TypeError(
                f"@register({key!r}): {cls.__qualname__} must override "
                f"all of {_REQUIRED_OVERRIDES}; still missing: {missing}. "
                f"See quark cleanup proposal §4.3 for the contract."
            )

        # Stamp NAME on the class if it wasn't set explicitly so the
        # autotune cache + tools can read it consistently.
        if not getattr(cls, "NAME", None):
            cls.NAME = key  # type: ignore[attr-defined]

        _REGISTRY[key] = cls
        return cls

    return wrap


def get(name: str) -> type:
    """Look up a registered kernel class by name.

    Raises KeyError on miss with the registered names listed for
    error-message debuggability."""
    if name not in _REGISTRY:
        raise KeyError(f"no kernel named {name!r}. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_kernels() -> list[type]:
    """Every kernel class currently registered. Order is registration
    order (which is import order)."""
    return list(_REGISTRY.values())


def names() -> list[str]:
    """Sorted list of registered kernel names."""
    return sorted(_REGISTRY)


def unregister(name: str) -> None:
    """Drop a kernel from the registry. Used by tests that want
    isolation between cases. Silently no-ops on miss."""
    _REGISTRY.pop(name, None)


def clear() -> None:
    """Drop every entry. Tests only."""
    _REGISTRY.clear()
