"""quark — GPU kernel framework with a typed IR and multi-backend lowering."""

from contextlib import contextmanager

from quark.device import Device, DeviceCaps, current_device
from quark.graph import capture_graph
from quark.ir import Builder, DType, Module


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
    "DType",
    "Device",
    "DeviceCaps",
    "Module",
    "capture_graph",
    "current_device",
    "max_autotune",
]
