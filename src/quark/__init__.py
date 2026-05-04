"""quark — GPU kernel framework with a typed IR and multi-backend lowering."""

from contextlib import contextmanager

from quark.device import Device, DeviceCaps, current_device
from quark.graph import capture_graph
from quark.ir import Builder, DType, Module


def __getattr__(name: str):
    """Lazy top-level attrs.

    ``Engine`` / ``CtrlInput`` pull in torch + the model stack, which
    we don't want to load on every ``import quark`` (the kernel-only
    surface — ``quark.lang``, ``quark.functional`` — has no torch
    dep). Importing them only when the attribute is actually accessed
    keeps the cold-import path lean.
    """
    if name in ("Engine", "CtrlInput"):
        from quark.engine import CtrlInput, Engine

        return {"Engine": Engine, "CtrlInput": CtrlInput}[name]
    raise AttributeError(f"module 'quark' has no attribute {name!r}")


@contextmanager
def max_autotune():
    """Force full genetic search for any autotune cache miss inside this block.

    Equivalent to setting ``QUARK_MAX_AUTOTUNE=1`` but scoped to a
    single ``with`` block.  Thread-safe and async-safe (uses a
    ``contextvars.ContextVar`` internally).

    Usage::

        with quark.max_autotune():
            pcf.gemm(A, B)   # cache miss → full genetic search

    For a per-startup warmup, prefer the explicit ``.autotune()`` API on
    each functional op (e.g. ``pcf.gemm.autotune(A, B)``), which always
    runs a full search regardless of the context depth setting.
    """
    from quark.autotune import _SEARCH_DEPTH

    token = _SEARCH_DEPTH.set("full")
    try:
        yield
    finally:
        _SEARCH_DEPTH.reset(token)


__all__ = [
    "Builder",
    "CtrlInput",
    "DType",
    "Device",
    "DeviceCaps",
    "Engine",
    "Module",
    "capture_graph",
    "current_device",
    "max_autotune",
]
