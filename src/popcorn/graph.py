"""CUDA Graph capture and replay.

    with popcorn.capture_graph() as graph:
        out = model(input_buf, sigma_idx=0, frame_t=0)

    # Replay — zero Python overhead per frame.
    for frame in frames:
        input_buf[:] = frame_data
        graph.replay()
        result = out.to_bytes()

All kernel launches inside the ``capture_graph()`` context manager
are recorded into a CUDA graph instead of being executed. The graph
is instantiated on exit and can be replayed with ``graph.replay()``.

During capture, a thread-local ``_active_capture_stream`` is set so
that ALL CUDA operations (kernel launches, memcpy, memset) use the
capture stream instead of stream 0. This avoids the
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT error.

Constraints:
- All tensor shapes must be fixed during capture.
- Input buffers must not be reallocated between capture and replay.
- No Python control flow that varies between iterations.

On Metal, graph capture is not yet supported — ``capture_graph()``
raises ``NotImplementedError``.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

_IS_METAL = sys.platform == "darwin"

# Thread-local active capture stream. When set (non-zero), all CUDA
# operations should use this stream instead of the default.
_capture_state = threading.local()

# Kernel launch counter during capture — for diagnosing double-launch bugs.
_capture_launch_log: list[str] = []


def active_stream() -> int:
    """Return the active capture stream, or 0 (default stream) if not capturing."""
    return getattr(_capture_state, "stream", 0)


def log_capture_launch(name: str) -> None:
    """Record a kernel launch during graph capture (for diagnostics)."""
    _capture_launch_log.append(name)


# Storages allocated during capture that survive past capture exit.
# After capture, these are transferred to CapturedGraph._held_storages
# and their _alloc_stream is cleared so release() becomes a no-op
# (the graph pool owns this memory).
_capture_storages: list = []


def register_capture_storage(storage) -> None:
    """Track a storage allocated during capture."""
    if active_stream():
        _capture_storages.append(storage)


@dataclass
class CapturedGraph:
    """A recorded sequence of kernel launches, replayable in one shot."""

    _graph: int  # CUgraph handle
    _exec: int  # CUgraphExec handle
    _stream: int
    _held_storages: list = field(default_factory=list)

    def replay(self, sync: bool = False) -> None:
        """Re-execute the captured graph on the original stream.

        ``sync=False`` (default): launch and return immediately. The
        caller is responsible for syncing before reading results.
        ``sync=True``: launch and wait for completion.
        """
        from popcorn.runtime.cuda import CudaRuntime

        rt = CudaRuntime.instance()
        rt.graph_launch(self._exec, self._stream)
        if sync:
            rt.stream_synchronize(self._stream)

    def __del__(self):
        if self._exec:
            try:
                from popcorn.runtime.cuda import CudaRuntime

                rt = CudaRuntime.instance()
                rt.graph_exec_destroy(self._exec)
                rt.graph_destroy(self._graph)
            except Exception:
                pass


@contextmanager
def capture_graph(stream: int = 0):
    """Context manager that captures all kernel launches into a CUDA graph.

    Usage::

        with capture_graph() as graph:
            # All launches here are recorded, not executed.
            out = model(x)

        # Replay
        graph.replay()

    The context manager yields a ``CapturedGraph`` that is populated
    on exit. Calling ``replay()`` before exiting the context raises
    an error.
    """
    if _IS_METAL:
        raise NotImplementedError(
            "capture_graph: Metal graph capture not yet implemented. "
            "See proposals/MLX_FFI_MIGRATION.md Stage 2."
        )

    from popcorn.runtime.cuda import CudaRuntime

    rt = CudaRuntime.instance()

    # Create a dedicated stream for capture.
    capture_stream = stream if stream != 0 else rt.stream_create()

    # Set the thread-local capture stream so all CUDA ops use it.
    # cuMemAllocAsync/cuMemFreeAsync are stream-ordered and work
    # during capture — no alloc cache needed.
    prev_stream = getattr(_capture_state, "stream", 0)
    _capture_state.stream = capture_stream
    _capture_launch_log.clear()
    _capture_storages.clear()

    rt.graph_begin_capture(capture_stream)

    result = CapturedGraph(_graph=0, _exec=0, _stream=capture_stream)
    try:
        yield result
    finally:
        graph_handle = rt.graph_end_capture(capture_stream)
        exec_handle = rt.graph_instantiate(graph_handle)
        result._graph = graph_handle
        result._exec = exec_handle
        result._stream = capture_stream
        # Hold storages that survived capture. Mark them so release()
        # is a no-op — the graph pool owns this memory. Storages that
        # died during capture already got cuMemFreeAsync graph nodes.
        for s in _capture_storages:
            if s.ptr and s._alloc_stream:
                s._alloc_stream = -1  # mark as graph-owned (release = no-op)
                result._held_storages.append(s)
        _capture_storages.clear()
        # Restore previous stream.
        _capture_state.stream = prev_stream
        # Print capture summary.
        n = len(_capture_launch_log)
        # Check for duplicates.
        from collections import Counter

        counts = Counter(_capture_launch_log)
        dupes = {k: v for k, v in counts.items() if v > 1}
        if dupes:
            print(f"[graph capture] WARNING: {n} kernel launches, DUPLICATES:")
            for name, count in sorted(dupes.items()):
                print(f"  {name}: {count}x")
        else:
            print(f"[graph capture] {n} kernel launches recorded")
        # Clear per-capture duplicate-detection flags.
        _clear_capture_flags(capture_stream)


def _clear_capture_flags(capture_stream: int) -> None:
    """Remove ``_graph_captured_<stream>`` attrs from all CompiledKernels."""
    from popcorn.functional._dispatch import launcher

    flag = f"_graph_captured_{capture_stream}"
    for ck in launcher()._kernel_cache.values():
        if hasattr(ck, flag):
            delattr(ck, flag)
