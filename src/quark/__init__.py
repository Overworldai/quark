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
def lazy(*, sync: bool = True):
    """Defer per-launch GPU sync inside this block; commit + wait at exit.

    By default every kernel launch on Metal commits its command buffer
    and waits for completion before returning, so output arrays are
    immediately readable. That blocks the GPU between kernels and
    erases pipelining benefits across kernel boundaries.

    Inside this block, launches accumulate in a single command buffer
    (auto-committing every 50 dispatches so the GPU starts working
    while Python keeps encoding), and the final sync fires at block
    exit. Output numpy views inside the block reference Metal-pool
    buffers; their *contents* are valid only after the block ends.

    Usage::

        with quark.lazy():
            h = embed(x)
            for layer in layers:
                h = layer(h)
            logits = head(h)
        # logits' data is valid here

    Reading any output array's contents inside the block (e.g.
    ``arr[0]``) returns whatever the GPU has written so far —
    typically uninitialized memory until eval. Call
    :func:`quark.eval` mid-block if you need to materialize early.

    Async-safe via ContextVar; nested ``with quark.lazy():`` is a
    no-op for the inner block.

    Parameters
    ----------
    sync : bool, default True
        When True (default), the block exits with a host-side
        ``waitUntilCompleted`` so any output is immediately host-
        readable.

        When False, the block exits with a *commit-only* — the
        command buffer is sent to the GPU but Python returns before
        the GPU finishes. This is the right choice when the work in
        the block is consumed only by *later* GPU kernels (the next
        frame's denoise reading this frame's K-cache, for example);
        cross-encoder fences carry the dependency. **Do not** read
        the produced data on the host (``ctypes.memmove`` /
        ``data_ptr``) without first calling :func:`quark.eval` —
        the fence machinery only orders GPU→GPU, not GPU→host.

        Nested ``sync=False`` is irrelevant: nesting is a no-op so
        the outer block's ``sync`` is what fires.
    """
    from quark.functional._dispatch import launcher
    from quark.launcher.launcher import _LAZY

    if _LAZY.get():
        # Already inside a lazy block — nesting is a no-op.
        yield
        return
    token = _LAZY.set(True)
    try:
        yield
    finally:
        _LAZY.reset(token)
        # Drain lazy queue first (graph-node ops), then eager ops.
        from quark.drivers import _metal_dispatch as _md

        if _md.has_lazy_pending():
            _md.eval_queue()
        from quark.functional._dispatch import clear_strided_copy_meta

        clear_strided_copy_meta()
        if sync:
            launcher().driver.sync(None)
        else:
            _md.commit_no_wait()


def eval():
    """Force materialization of all pending kernel launches.

    Outside :func:`quark.lazy`, sync already happens per-launch and
    this is a no-op (well, fast-path: nothing pending). Inside lazy,
    use this to commit early when you need to read an intermediate
    result without exiting the block.
    """
    from quark.drivers import _metal_dispatch as _md

    if _md.has_lazy_pending():
        _md.eval_queue()
    from quark.functional._dispatch import clear_strided_copy_meta, launcher

    clear_strided_copy_meta()
    launcher().driver.sync(None)


@contextmanager
def max_autotune():
    """Force full genetic search for any autotune cache miss inside this block.

    Equivalent to setting ``QUARK_MAX_AUTOTUNE=1`` but scoped to a
    single ``with`` block.  Thread-safe and async-safe (uses a
    ``contextvars.ContextVar`` internally).

    Usage::

        with quark.max_autotune():
            qf.gemm(A, B)   # cache miss → full genetic search

    For a per-startup warmup, prefer the explicit ``.autotune()`` API on
    each functional op (e.g. ``qf.gemm.autotune(A, B)``), which always
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
    "eval",
    "lazy",
    "max_autotune",
]
