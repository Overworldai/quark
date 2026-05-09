"""QuarkTensor — unified GPU tensor for Metal and CUDA.

EXEMPT FROM 500-LINE RULE: QuarkTensor is the single tensor type for
both backends. Storage, strides, factories, view ops, slicing, and
arithmetic dispatch all belong together — splitting would fragment the
type and force circular imports between the pieces.

A ``QuarkTensor`` wraps a ``cuMemAlloc``'d device pointer (CUDA) or
a Metal-pool-backed buffer handle (Metal) with shape, dtype, strides,
and offset metadata. It supports the operations the launcher needs:

    t.data_ptr()        # → int (device pointer to first element)
    t.shape / t.dtype   # metadata
    t.device            # DeviceSpec(type="cuda", index=0)
    t.is_contiguous()   # True iff strides are row-major with no gaps

And the raw data transfer API (no numpy dependency):

    QuarkTensor.from_bytes(buf, shape, dtype)  # host → device
    t.to_bytes()                                  # device → host

Plus arithmetic, slicing, reshape, permute — all dispatched to
on-device PTX utility kernels (see ``runtime/kernels.py``).

Thread-safety: NOT thread-safe. One CUDA context per process.
"""

from __future__ import annotations

import ctypes
import math
import os as _os
import sys as _sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quark.runtime.utils import (
    f32_bytes_to_bf16_bytes as _f32_bytes_to_bf16_bytes,
)
from quark.runtime.utils import (
    f32_to_bf16_numpy as _f32_to_bf16_numpy,
)
from quark.runtime.utils import (
    flatten_list as _flatten_list,
)
from quark.runtime.utils import (
    infer_shape as _infer_shape,
)
from quark.runtime.utils import (
    reshape_flat_to_nested as _reshape_flat_to_nested,
)

if TYPE_CHECKING:
    pass

# ── Dtype tables ──────────────────────────────────────────────

PC_BYTES: dict[str, int] = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "e4m3": 1,
    "e5m2": 1,
    "s32": 4,
    "s64": 8,
    "u8": 1,
    "s8": 1,
    "u16": 2,
    "u32": 4,
}


@dataclass(frozen=True)
class DeviceSpec:
    type: str = "cuda"
    index: int = 0


def _runtime():
    """Lazy singleton — avoids importing libcuda at module level."""
    from quark.runtime.cuda import CudaRuntime

    return CudaRuntime.instance()


# ── Storage ───────────────────────────────────────────────────


class _CudaStorage:
    """Reference-counted device memory block.

    Multiple QuarkTensors can share the same storage (views from
    slicing, reshape, permute). The storage is freed when the last
    reference is garbage-collected.
    """

    __slots__ = ("_alloc_stream", "_refcount", "nbytes", "ptr")

    def __init__(self, ptr: int, nbytes: int, alloc_stream: int = 0):
        self.ptr = ptr
        self.nbytes = nbytes
        self._refcount = 1
        self._alloc_stream = alloc_stream

    @staticmethod
    def alloc(nbytes: int) -> _CudaStorage:
        if nbytes == 0:
            return _CudaStorage(0, 0)
        from quark.graph import active_stream

        stream = active_stream()
        ptr = _runtime().mem_alloc(nbytes)
        storage = _CudaStorage(ptr, nbytes, alloc_stream=stream)
        if stream:
            from quark.graph import register_capture_storage

            register_capture_storage(storage)
        return storage

    def retain(self) -> _CudaStorage:
        self._refcount += 1
        return self

    def release(self) -> None:
        self._refcount -= 1
        if self._refcount <= 0 and self.ptr and self.nbytes > 0:
            if self._alloc_stream == -1:
                # Held by a CapturedGraph — do not free. The graph's
                # async pool owns this memory.
                self.ptr = 0
                return
            try:  # noqa: SIM105 — contextlib.suppress fails during Python shutdown
                _runtime().mem_free(self.ptr, stream=self._alloc_stream)
            except Exception:
                pass
            self.ptr = 0


class _BorrowedStorage:
    """Non-owning storage that wraps an existing device pointer.

    Holds a reference to the source object (torch.Tensor, np.ndarray)
    to prevent it from being garbage collected. Does NOT free the
    memory on release — the source object owns it.
    """

    __slots__ = ("_owner", "_refcount", "nbytes", "ptr")

    def __init__(self, ptr: int, nbytes: int, owner: object):
        self.ptr = ptr
        self.nbytes = nbytes
        self._owner = owner  # prevent GC of the source tensor
        self._refcount = 1

    def retain(self) -> _BorrowedStorage:
        self._refcount += 1
        return self

    def release(self) -> None:
        self._refcount -= 1
        # Don't free memory — we don't own it.
        if self._refcount <= 0:
            self.ptr = 0
            self._owner = None


# ── Metal storage ────────────────────────────────────────────

_IS_METAL = _sys.platform == "darwin"
# SPV / Intel storage selection. Opt-in for now — the SPV backend lives
# on Linux alongside CUDA, so we can't fall through "Linux → CUDA"
# unconditionally. Set ``QUARK_BACKEND=spv`` (or ``=intel``) to route
# ``QuarkTensor.empty`` / ``zeros`` / ``from_bytes`` through the
# Vulkan storage path; otherwise CUDA is the default on non-Apple.
_IS_SPV = _os.environ.get("QUARK_BACKEND", "").lower() in ("spv", "intel")


class _MetalStorage:
    """Metal-pool-backed buffer (persistent, zero-copy at dispatch time).

    The buffer lives in the C extension's ``g_lazy_buffers`` table,
    indexed by ``handle``. ``ptr`` is the raw ``MTLBuffer->contents()``
    pointer for CPU-side reads (after ``eval_queue``). The buffer is
    never freed — it returns to the pool when the storage is GC'd.

    Created via ``_MetalStorage.alloc`` (empty) or
    ``_MetalStorage.from_host`` (copy from numpy/host memory).
    """

    __slots__ = ("_refcount", "handle", "nbytes", "ptr")

    def __init__(self, handle: int, ptr: int, nbytes: int):
        self.handle = handle
        self.ptr = ptr
        self.nbytes = nbytes
        self._refcount = 1
        # Tell the dispatcher there's a fresh Python wrapper holding
        # this handle. Paired with ``release_handle`` from
        # ``release()`` when the last QuarkTensor referring to this
        # storage goes away.
        try:
            from quark.drivers import _metal_dispatch as _md

            _md.retain_handle(handle)
        except Exception:
            pass

    @staticmethod
    def alloc(nbytes: int) -> _MetalStorage:
        """Allocate an uninitialized Metal pool buffer."""
        from quark.drivers import _metal_dispatch as _md

        # pin_buffer with data_ptr=0 allocates without copying.
        h, p = _md.pin_buffer(0, max(nbytes, 1))
        return _MetalStorage(h, p, max(nbytes, 1))

    @staticmethod
    def from_host(ptr: int, nbytes: int) -> _MetalStorage:
        """Copy host bytes into a new Metal pool buffer."""
        from quark.drivers import _metal_dispatch as _md

        h, p = _md.pin_buffer(ptr, nbytes)
        return _MetalStorage(h, p, nbytes)

    def retain(self) -> _MetalStorage:
        self._refcount += 1
        return self

    def release(self) -> None:
        self._refcount -= 1
        if self._refcount > 0:
            return
        # Last reference dropped — tell the dispatcher this handle is
        # eligible for pool-recycle. The dispatcher waits for the next
        # ``eval()`` (i.e. for any in-flight kernels that bound this
        # buffer to actually complete) before returning the buffer to
        # the pool, so a release racing with a queued kernel is safe.
        try:
            from quark.drivers import _metal_dispatch as _md

            _md.release_handle(self.handle)
        except Exception:
            # Module-shutdown path: _md may already be torn down.
            pass


# ── SPV / Intel Vulkan storage ────────────────────────────────


class _SpvStorage:
    """SPV/Vulkan-backed buffer for ``QuarkTensor`` on Intel.

    The buffer lives in the C extension's internal table, indexed by
    ``handle`` (the same handle ``_sd.allocate_buffer`` returns).
    ``ptr`` is the host-coherent mapped pointer the buffer was created
    with; CPU reads / writes go through it directly (Vulkan's
    ``HOST_VISIBLE | HOST_COHERENT`` memory type makes them visible to
    the GPU on the next dispatch without an explicit flush).

    Lifecycle today: buffers persist until ``teardown_device_state``
    runs at module shutdown. ``_spv_dispatch`` doesn't expose a
    ``free_buffer`` yet — production callers should keep a per-tensor
    pool and reuse handles. Adding free-on-release is a follow-up
    (mirrors the Metal pool-recycle pattern).

    Created via ``_SpvStorage.alloc`` (uninit) or
    ``_SpvStorage.from_host`` (alloc + ctypes.memmove).
    """

    __slots__ = ("_refcount", "handle", "nbytes", "ptr")

    def __init__(self, handle: int, ptr: int, nbytes: int):
        self.handle = handle
        self.ptr = ptr
        self.nbytes = nbytes
        self._refcount = 1

    @staticmethod
    def alloc(nbytes: int) -> _SpvStorage:
        """Allocate an uninitialised Vulkan host-coherent buffer."""
        from quark.drivers import _spv_dispatch as _sd

        h, p = _sd.allocate_buffer(max(nbytes, 1))
        return _SpvStorage(h, p, max(nbytes, 1))

    @staticmethod
    def from_host(ptr: int, nbytes: int) -> _SpvStorage:
        """Copy host bytes into a new Vulkan buffer."""
        import ctypes as _ctypes
        from quark.drivers import _spv_dispatch as _sd

        h, p = _sd.allocate_buffer(max(nbytes, 1))
        if nbytes > 0:
            _ctypes.memmove(p, ptr, nbytes)
        return _SpvStorage(h, p, max(nbytes, 1))

    def retain(self) -> _SpvStorage:
        self._refcount += 1
        return self

    def release(self) -> None:
        self._refcount -= 1
        # No free-buffer API on the dispatch side yet; the buffer
        # persists for the device's lifetime. Track refcount for
        # parity with the other Storage classes.
        if self._refcount <= 0:
            self.ptr = 0


# ── Stride helpers ────────────────────────────────────────────


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major (C-contiguous) strides in elements."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _is_contiguous(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    """Check if shape+strides describe a contiguous row-major layout."""
    expected = _contiguous_strides(shape)
    return strides == expected


# ── QuarkTensor ─────────────────────────────────────────────


class QuarkTensor:
    """Unified GPU tensor for CUDA and Metal.

    Supports views (slicing, reshape, permute) via stride metadata.
    Arithmetic ops dispatch to PTX utility kernels.
    """

    __slots__ = ("_device", "_dtype", "_offset", "_shape", "_storage", "_strides")

    def __init__(
        self,
        storage: _CudaStorage | _BorrowedStorage | _MetalStorage | _SpvStorage,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
        dtype: str,
    ):
        self._storage = storage
        self._shape = shape
        self._strides = strides
        self._offset = offset  # element offset into storage
        self._dtype = dtype
        if isinstance(storage, _MetalStorage):
            dev_family = "metal"
        elif isinstance(storage, _SpvStorage):
            dev_family = "spv"
        else:
            dev_family = "cuda"
        self._device = DeviceSpec(dev_family, 0)

    # ── Metal-specific accessors ──

    @property
    def metal_handle(self) -> int | None:
        """Handle into ``g_lazy_buffers`` for zero-copy ``queue_launch``.
        Returns ``None`` on CUDA."""
        if isinstance(self._storage, _MetalStorage):
            return self._storage.handle
        return None

    @property
    def spv_handle(self) -> int | None:
        """Handle into the Vulkan dispatch's buffer table for
        zero-copy ``compiled.launch``. Returns ``None`` on
        non-SPV storage."""
        if isinstance(self._storage, _SpvStorage):
            return self._storage.handle
        return None

    @property
    def quark_dtype(self) -> str:
        """Short dtype tag (``"bf16"``, ``"f32"``, etc.) for
        ``pcf.*`` dispatch compatibility."""
        return self._dtype

    # ── Factories ──

    @staticmethod
    def empty(*shape: int, dtype: str = "bf16") -> QuarkTensor:
        """Allocate uninitialized device memory.

        On CUDA: ``cuMemAlloc`` (or ``cuMemAllocAsync`` during graph
        capture). On Metal: allocates a persistent Metal pool buffer
        via ``_md.pin_buffer``.
        """
        # Accept numpy dtypes from Metal callers — the on-device layout
        # is byte-identical, but PC_BYTES is keyed by the quark string.
        if not isinstance(dtype, str):
            _NP_TO_PC = {
                "uint16": "bf16",
                "float16": "f16",
                "float32": "f32",
                "int32": "s32",
                "int64": "s64",
                "uint8": "u8",
                "int8": "s8",
            }
            name = getattr(dtype, "name", str(dtype))
            dtype = _NP_TO_PC.get(name, str(dtype))
        numel = math.prod(shape) if shape else 1
        nbytes = numel * PC_BYTES[dtype]
        storage: _CudaStorage | _MetalStorage | _SpvStorage
        if _IS_METAL:
            from quark.drivers import _metal_dispatch as _md

            h, p = _md.pin_buffer(0, max(nbytes, 1))
            storage = _MetalStorage(h, p, nbytes)
        elif _IS_SPV:
            storage = _SpvStorage.alloc(nbytes)
        else:
            storage = _CudaStorage.alloc(nbytes)
        return QuarkTensor(storage, tuple(shape), _contiguous_strides(tuple(shape)), 0, dtype)

    @staticmethod
    def zeros(*shape: int, dtype: str = "bf16") -> QuarkTensor:
        t = QuarkTensor.empty(*shape, dtype=dtype)
        if t._storage.nbytes > 0:
            if _IS_METAL or _IS_SPV:
                # Both Metal pool buffers and Vulkan host-coherent
                # buffers expose the host-mapped pointer at
                # ``storage.ptr`` — memset directly.
                ctypes.memset(t._storage.ptr, 0, t._storage.nbytes)
            else:
                _runtime().memset_d8(t._storage.ptr, 0, t._storage.nbytes)
        return t

    @staticmethod
    def from_bytes(
        buf: bytes | bytearray | memoryview, shape: tuple[int, ...], dtype: str
    ) -> QuarkTensor:
        """Copy raw bytes from host to device.

        Memoryview path is zero-copy: numpy.frombuffer exposes the host
        pointer (e.g. mmap page) straight to cuMemcpyHtoD. Bytes path
        keeps the legacy ctypes-array copy for non-mmap callers.
        """
        numel = math.prod(shape) if shape else 1
        expected = numel * PC_BYTES[dtype]
        if len(buf) != expected:
            raise ValueError(f"from_bytes: want {expected} bytes, got {len(buf)}")
        t = QuarkTensor.empty(*shape, dtype=dtype)
        if expected == 0:
            return t
        if isinstance(buf, memoryview):
            import numpy as _np

            arr = _np.frombuffer(buf, dtype=_np.uint8, count=expected)
            if _IS_METAL or _IS_SPV:
                # Host-coherent (Metal pool buffers + Vulkan
                # ``HOST_VISIBLE | HOST_COHERENT``) memory — the
                # mapped pointer is writable directly. GPU sees it on
                # the next dispatch's implicit cache-flush.
                ctypes.memmove(t._storage.ptr, arr.ctypes.data, expected)
            else:
                _runtime().memcpy_htod(t._storage.ptr, arr.ctypes.data, expected)
        else:
            host_arr = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
            if _IS_METAL or _IS_SPV:
                ctypes.memmove(t._storage.ptr, ctypes.addressof(host_arr), expected)
            else:
                _runtime().memcpy_htod(t._storage.ptr, ctypes.addressof(host_arr), expected)
        return t

    def to_bytes(self) -> bytes:
        """Copy device data to host as raw bytes. If the tensor is
        non-contiguous, a contiguous copy is made first."""
        t = self.contiguous()
        nbytes = t.numel() * PC_BYTES[t._dtype]
        if nbytes == 0:
            return b""
        host_buf = (ctypes.c_ubyte * nbytes)()
        if _IS_METAL:
            # Metal-backed storage already has a host-shared pointer; flush
            # the lazy queue so any pending writes hit memory before we
            # read it, then copy via memmove. CudaRuntime isn't available
            # on macOS — going through ``_runtime()`` here would raise
            # CudaError(LIBRARY_NOT_FOUND).
            from quark.runtime.sync import synchronize

            synchronize()
            ctypes.memmove(ctypes.addressof(host_buf), t.data_ptr(), nbytes)
        elif _IS_SPV:
            # Vulkan host-coherent buffers: drain any in-flight
            # dispatches via ``_sd.sync()`` so the GPU's writes are
            # visible to the host pointer, then memmove from the
            # mapped ``ptr``.
            from quark.drivers import _spv_dispatch as _sd

            _sd.sync()
            ctypes.memmove(ctypes.addressof(host_buf), t.data_ptr(), nbytes)
        else:
            _runtime().memcpy_dtoh(ctypes.addressof(host_buf), t.data_ptr(), nbytes)
        return bytes(host_buf)

    @staticmethod
    def from_numpy(arr, dtype: str | None = None) -> QuarkTensor:
        """Convenience: numpy array → QuarkTensor. Tests/debug only."""
        import numpy as np

        _NP_TO_PC = {
            np.dtype("float32"): "f32",
            np.dtype("float16"): "f16",
            np.dtype("uint16"): "bf16",
            np.dtype("int32"): "s32",
            np.dtype("int64"): "s64",
            np.dtype("uint8"): "u8",
            np.dtype("int8"): "s8",
        }
        if dtype is None:
            dtype = _NP_TO_PC.get(arr.dtype)
            if dtype is None:
                raise ValueError(f"from_numpy: unsupported dtype {arr.dtype}")

        _PC_TO_NP = {
            "f32": np.dtype("float32"),
            "f16": np.dtype("float16"),
            "bf16": np.dtype("uint16"),
            "e4m3": np.dtype("uint8"),
            "e5m2": np.dtype("uint8"),
            "s32": np.dtype("int32"),
            "s64": np.dtype("int64"),
            "u8": np.dtype("uint8"),
            "s8": np.dtype("int8"),
        }
        target_np = _PC_TO_NP[dtype]
        if arr.dtype != target_np:
            if dtype == "bf16" and arr.dtype == np.float32:
                arr = _f32_to_bf16_numpy(arr)
            elif arr.dtype.itemsize == target_np.itemsize:
                arr = arr.view(target_np)
            else:
                arr = arr.astype(target_np)

        arr = np.ascontiguousarray(arr)
        shape = tuple(arr.shape)
        if _IS_METAL:
            storage = _MetalStorage.from_host(arr.ctypes.data, int(arr.nbytes))
            return QuarkTensor(storage, shape, _contiguous_strides(shape), 0, dtype)
        if _IS_SPV:
            storage = _SpvStorage.from_host(arr.ctypes.data, int(arr.nbytes))
            return QuarkTensor(storage, shape, _contiguous_strides(shape), 0, dtype)
        raw = arr.tobytes()
        return QuarkTensor.from_bytes(raw, shape, dtype)

    def to_numpy(self):
        """QuarkTensor → numpy array.

        On Metal: reads directly from the buffer's ``contents()`` pointer
        (triggers ``eval_queue`` if lazy ops are pending). On CUDA:
        ``cuMemcpyDtoH``.
        """
        import numpy as np

        _PC_TO_NP = {
            "f32": np.dtype("float32"),
            "f16": np.dtype("float16"),
            "bf16": np.dtype("uint16"),
            "s32": np.dtype("int32"),
            "s64": np.dtype("int64"),
            "u8": np.dtype("uint8"),
            "s8": np.dtype("int8"),
        }
        np_dt = _PC_TO_NP[self._dtype]
        if isinstance(self._storage, _MetalStorage):
            from quark.drivers import _metal_dispatch as _md

            if _md.has_lazy_pending():
                _md.eval_queue()
                from quark.functional._dispatch import clear_strided_copy_meta

                clear_strided_copy_meta()
            elem_size = np_dt.itemsize
            buf_ptr = self._storage.ptr + self._offset * elem_size
            total = math.prod(self._shape) * elem_size
            # ctypes char arrays implement the buffer protocol; ty's stubs
            # for np.frombuffer don't list it as a buffer-protocol overload.
            return (
                np.frombuffer(  # ty: ignore[no-matching-overload]
                    (ctypes.c_char * total).from_address(buf_ptr), dtype=np_dt
                )
                .reshape(self._shape)
                .copy()
            )
        if isinstance(self._storage, _SpvStorage):
            from quark.drivers import _spv_dispatch as _sd

            _sd.sync()
            elem_size = np_dt.itemsize
            buf_ptr = self._storage.ptr + self._offset * elem_size
            total = math.prod(self._shape) * elem_size
            return (
                np.frombuffer(  # ty: ignore[no-matching-overload]
                    (ctypes.c_char * total).from_address(buf_ptr), dtype=np_dt
                )
                .reshape(self._shape)
                .copy()
            )
        raw = self.to_bytes()
        return np.frombuffer(raw, dtype=np_dt).reshape(self._shape)

    def __array__(self, dtype=None):
        """numpy interop: ``np.asarray(tensor)`` triggers data transfer."""
        import numpy as np

        return np.asarray(self.to_numpy(), dtype=dtype)

    @staticmethod
    def from_list(data, dtype: str = "f32") -> QuarkTensor:
        """Create a tensor from a Python list of numbers. Uses
        ``array.array`` for fast packing of homogeneous data."""
        import array as _array

        flat = _flatten_list(data)
        _TYPECODES = {"f32": "f", "f16": "e", "s32": "i", "s64": "q", "u8": "B", "s8": "b"}
        if dtype == "bf16":
            buf = _array.array("f", flat)
            raw = _f32_bytes_to_bf16_bytes(buf.tobytes())
        elif dtype in _TYPECODES:
            buf = _array.array(_TYPECODES[dtype], flat)
            raw = buf.tobytes()
        else:
            raise ValueError(f"from_list: unsupported dtype {dtype!r}")

        shape = _infer_shape(data)
        return QuarkTensor.from_bytes(raw, shape, dtype)

    # ── Zero-copy wrapping of external tensors ──

    @staticmethod
    def borrow(
        ptr: int,
        nbytes: int,
        shape: tuple[int, ...],
        dtype: str,
        *,
        owner: object = None,
        strides: tuple[int, ...] | None = None,
        offset: int = 0,
    ) -> QuarkTensor:
        """Zero-copy wrap an existing device pointer as a QuarkTensor.

        Non-owning: caller (or ``owner``) must keep the allocation
        alive for the returned tensor's lifetime. ``owner`` anchors GC
        (e.g. the source ``torch.Tensor`` whose ``.data_ptr()`` is
        passed as ``ptr``). Strides default to row-major over ``shape``.
        """
        if strides is None:
            strides = _contiguous_strides(tuple(shape))
        storage = _BorrowedStorage(ptr, nbytes, owner=owner)
        return QuarkTensor(storage, tuple(shape), tuple(strides), offset, dtype)

    # ── Launcher interface ──

    def data_ptr(self) -> int:
        """Pointer to the first element of this view."""
        return self._storage.ptr + self._offset * PC_BYTES[self._dtype]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def device(self) -> DeviceSpec:
        return self._device

    @property
    def strides(self) -> tuple[int, ...]:
        """Strides in elements (not bytes)."""
        return self._strides

    def is_contiguous(self) -> bool:
        return _is_contiguous(self._shape, self._strides)

    def numel(self) -> int:
        return math.prod(self._shape) if self._shape else 1

    @property
    def size(self) -> int:
        return self.numel()

    # ── View ops (no data copy) ──

    def reshape(self, *shape: int) -> QuarkTensor:
        """View-compatible reshape. If the tensor is contiguous, this is
        a zero-copy view. Otherwise, copies to a contiguous buffer first."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        # Resolve -1
        neg_idx = None
        known = 1
        for i, s in enumerate(shape):
            if s == -1:
                if neg_idx is not None:
                    raise ValueError("reshape: only one -1 allowed")
                neg_idx = i
            else:
                known *= s
        old_numel = self.numel()
        if neg_idx is not None:
            if known == 0:
                raise ValueError("reshape: cannot infer with zero dim")
            inferred = old_numel // known
            shape = tuple(inferred if i == neg_idx else s for i, s in enumerate(shape))
            known *= inferred
        if known != old_numel:
            raise ValueError(f"reshape: cannot reshape {self._shape} → {shape}")

        t = self.contiguous() if not self.is_contiguous() else self
        new_strides = _contiguous_strides(shape)
        t._storage.retain()
        return QuarkTensor(t._storage, shape, new_strides, t._offset, t._dtype)

    def permute(self, *dims: int) -> QuarkTensor:
        """Transpose / permute axes. Returns a view (no copy)."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        if sorted(dims) != list(range(self.ndim)):
            raise ValueError(f"permute: invalid dims {dims} for ndim={self.ndim}")
        new_shape = tuple(self._shape[d] for d in dims)
        new_strides = tuple(self._strides[d] for d in dims)
        self._storage.retain()
        return QuarkTensor(self._storage, new_shape, new_strides, self._offset, self._dtype)

    @property
    def T(self) -> QuarkTensor:
        """Transpose last two dims (convenience for matmul)."""
        if self.ndim < 2:
            return self
        dims = list(range(self.ndim))
        dims[-1], dims[-2] = dims[-2], dims[-1]
        return self.permute(*dims)

    def contiguous(self) -> QuarkTensor:
        """Return a contiguous copy, or self if already contiguous."""
        if self.is_contiguous() and self._offset == 0:
            return self
        if _IS_METAL:
            from quark.functional._dispatch import queue_strided_copy

            handle = self.metal_handle
            assert handle is not None, "Metal path requires _MetalStorage"
            t = queue_strided_copy(
                handle,
                self._offset,
                self._strides,
                self._shape,
                PC_BYTES[self._dtype],
            )
            t._dtype = self._dtype
            return t
        # CUDA: copy via _copy_strided utility kernel.
        from quark.runtime.kernels import copy_strided

        return copy_strided(self)

    # ── Slicing ──

    def __getitem__(self, key) -> QuarkTensor:
        """Basic slicing: integer indexing and slice objects.

        Supports: t[i], t[i:j], t[:, i:j], t[i:j, k:l], etc.
        Does NOT support advanced/fancy indexing (boolean masks, index
        tensors). Returns a view when possible.
        """
        if not isinstance(key, tuple):
            key = (key,)

        new_shape = []
        new_strides = []
        offset = self._offset

        dim = 0
        for k in key:
            if dim >= self.ndim:
                raise IndexError(f"too many indices for tensor of dim {self.ndim}")

            if isinstance(k, int):
                # Integer index: collapse this dimension.
                idx = k if k >= 0 else self._shape[dim] + k
                if idx < 0 or idx >= self._shape[dim]:
                    raise IndexError(
                        f"index {k} out of range for dim {dim} with size {self._shape[dim]}"
                    )
                offset += idx * self._strides[dim]
                dim += 1
                # Don't append to new_shape — this dim is collapsed.

            elif isinstance(k, slice):
                start, stop, step = k.indices(self._shape[dim])
                if step != 1:
                    raise NotImplementedError("QuarkTensor: step != 1 slicing not yet supported")
                size = max(0, stop - start)
                offset += start * self._strides[dim]
                new_shape.append(size)
                new_strides.append(self._strides[dim])
                dim += 1

            else:
                raise TypeError(f"QuarkTensor: unsupported index type {type(k).__name__}")

        # Append remaining dimensions.
        for d in range(dim, self.ndim):
            new_shape.append(self._shape[d])
            new_strides.append(self._strides[d])

        self._storage.retain()
        return QuarkTensor(self._storage, tuple(new_shape), tuple(new_strides), offset, self._dtype)

    def __setitem__(self, key, value) -> None:
        """Write into a slice of the tensor. ``value`` must be a
        QuarkTensor or a scalar."""
        dst_view = self[key]
        if isinstance(value, QuarkTensor):
            if value.shape != dst_view.shape:
                raise ValueError(f"__setitem__: shape mismatch {value.shape} vs {dst_view.shape}")
            from quark.runtime.kernels import copy_strided_into

            copy_strided_into(dst_view, value)
        elif isinstance(value, (int, float)):
            from quark.runtime.kernels import fill_scalar

            fill_scalar(dst_view, value)
        else:
            raise TypeError(f"__setitem__: unsupported value type {type(value).__name__}")
        # Release the extra retain from __getitem__.
        dst_view._storage.release()

    # ── Arithmetic (dispatched to utility kernels) ──

    def __add__(self, other) -> QuarkTensor:
        if _IS_METAL:
            return self._metal_binop(other, "add")
        from quark.runtime.kernels import elemwise_binary

        return elemwise_binary("add", self, other)

    def __radd__(self, other) -> QuarkTensor:
        if _IS_METAL:
            return self._metal_binop(other, "add")
        from quark.runtime.kernels import elemwise_binary

        return elemwise_binary("add", self, other)

    def _metal_binop(self, other, op: str) -> QuarkTensor:
        """Element-wise binary op on Metal.

        For bf16 add with both operands having metal_handle, dispatches
        a lazy queue_launch kernel (no pipeline break). Falls back to
        numpy round-trip for other cases.
        """
        # Fast path: lazy bf16 add when both operands have metal_handle.
        if (
            op == "add"
            and self._dtype == "bf16"
            and isinstance(other, QuarkTensor)
            and self.metal_handle is not None
            and other.metal_handle is not None
        ):
            from quark.functional._dispatch import queue_elemwise_add_bf16

            a_numel = self.numel()
            b_numel = other.numel()
            h, ptr, nbytes = queue_elemwise_add_bf16(
                self.metal_handle,
                other.metal_handle,
                a_numel,
                b_numel,
            )
            storage = _MetalStorage(h, ptr, nbytes)
            out_shape = self._shape  # broadcast: result has A's shape
            return QuarkTensor(
                storage,
                out_shape,
                _contiguous_strides(out_shape),
                0,
                self._dtype,
            )

        import numpy as np

        a = self.to_numpy()
        if isinstance(other, QuarkTensor):
            b = other.to_numpy()
        elif hasattr(other, "__array__"):
            b = np.asarray(other)
        else:
            b = other
        if op == "add":
            out = a + b
        elif op == "sub":
            out = a - b
        elif op == "mul":
            out = a * b
        else:
            raise NotImplementedError(f"_metal_binop: {op}")
        return QuarkTensor.from_numpy(out.reshape(self._shape), dtype=self._dtype)

    def __sub__(self, other) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_binary

        return elemwise_binary("sub", self, other)

    def __rsub__(self, other) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_binary_scalar_lhs

        return elemwise_binary_scalar_lhs("sub", other, self)

    def __mul__(self, other) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_binary

        return elemwise_binary("mul", self, other)

    def __rmul__(self, other) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_binary

        return elemwise_binary("mul", self, other)

    def __truediv__(self, other) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_binary

        return elemwise_binary("div", self, other)

    def __rtruediv__(self, other) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_binary_scalar_lhs

        return elemwise_binary_scalar_lhs("div", other, self)

    def __neg__(self) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_unary

        return elemwise_unary("neg", self)

    def __abs__(self) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_unary

        return elemwise_unary("abs", self)

    def __matmul__(self, other) -> QuarkTensor:
        from quark.runtime.kernels import matmul

        return matmul(self, other)

    def exp(self) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_unary

        return elemwise_unary("exp", self)

    def sin(self) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_unary

        return elemwise_unary("sin", self)

    def cos(self) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_unary

        return elemwise_unary("cos", self)

    def sqrt(self) -> QuarkTensor:
        from quark.runtime.kernels import elemwise_unary

        return elemwise_unary("sqrt", self)

    def sum(self, dim: int | None = None) -> QuarkTensor:
        from quark.runtime.kernels import reduce_op

        return reduce_op("sum", self, dim)

    def max(self, dim: int | None = None) -> QuarkTensor:
        from quark.runtime.kernels import reduce_op

        return reduce_op("max", self, dim)

    def astype(self, dtype: str) -> QuarkTensor:
        """Cast to a different dtype. Allocates a new tensor each time."""
        if dtype == self._dtype:
            return self
        from quark.runtime.kernels import cast

        return cast(self, dtype)

    def item(self) -> float:
        """Extract a single-element tensor as a Python float."""
        if self.numel() != 1:
            raise ValueError(f"item(): tensor has {self.numel()} elements, expected 1")
        raw = self.astype("f32").to_bytes()
        import struct as _struct

        return _struct.unpack("<f", raw)[0]

    def clone(self) -> QuarkTensor:
        """Return a contiguous copy with its own storage (async D2D)."""
        dst = QuarkTensor.empty(*self._shape, dtype=self._dtype)
        nbytes = self.numel() * PC_BYTES[self._dtype]
        src = self.contiguous()
        if _IS_METAL:
            ctypes.memmove(dst.data_ptr(), src.data_ptr(), nbytes)
        else:
            from quark.runtime.cuda import CudaRuntime

            CudaRuntime.instance().memcpy_dtod(dst.data_ptr(), src.data_ptr(), nbytes)
        return dst

    def zero_(self) -> QuarkTensor:
        """Zero this tensor in-place (async). Returns self."""
        nbytes = self.numel() * PC_BYTES[self._dtype]
        if _IS_METAL:
            ctypes.memset(self.data_ptr(), 0, nbytes)
        else:
            from quark.runtime.cuda import CudaRuntime

            CudaRuntime.instance().memset_d8(self.data_ptr(), 0, nbytes)
        return self

    def set_value(self, value: int) -> None:
        """Set a scalar s32/u32 tensor to ``value`` (async memset)."""
        if self.numel() != 1 or self._dtype not in ("s32", "u32"):
            raise ValueError("set_value: requires single-element s32/u32")
        _runtime().memset_d32(self.data_ptr(), value & 0xFFFFFFFF, 1)

    def increment(self) -> None:
        """In-place increment a single-element s32 tensor by 1.

        Fully on-device via a tiny PTX kernel — no host sync. The
        kernel is compiled once and cached.
        """
        if self.numel() != 1 or self._dtype not in ("s32", "u32"):
            raise ValueError("increment: requires single-element s32/u32 tensor")
        from quark.runtime.kernels import scalar_increment

        scalar_increment(self)

    def copy_from(self, src: QuarkTensor) -> None:
        """Copy data from ``src`` into this tensor (same shape/dtype)."""
        from quark.runtime.kernels import copy_strided_into

        copy_strided_into(self, src)

    def copy_into_ptr(self, dst_ptr: int, stream: int = 0) -> None:
        """Async D2D copy self → ``dst_ptr``. Hand-off primitive for foreign
        owners (e.g. torch). Non-contiguous tensors are made contiguous first."""
        t = self.contiguous()
        nbytes = t.numel() * PC_BYTES[t._dtype]
        if nbytes == 0:
            return
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memcpy_dtod(dst_ptr, t.data_ptr(), nbytes, stream=stream)

    def copy_from_ptr(self, src_ptr: int, stream: int = 0) -> None:
        """Async D2D copy ``src_ptr`` → self. Counterpart to ``copy_into_ptr``;
        self must be contiguous (common case — borrowed stable buffers)."""
        if not self.is_contiguous():
            raise ValueError("copy_from_ptr: destination tensor must be contiguous")
        nbytes = self.numel() * PC_BYTES[self._dtype]
        if nbytes == 0:
            return
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memcpy_dtod(self.data_ptr(), src_ptr, nbytes, stream=stream)

    @staticmethod
    def cat(tensors: list[QuarkTensor], dim: int = 0) -> QuarkTensor:
        """Concatenate tensors along a dimension."""
        from quark.runtime.kernels import cat

        return cat(tensors, dim)

    @staticmethod
    def randn(*shape: int, dtype: str = "f32") -> QuarkTensor:
        """Random normal tensor (host-side PRNG, then copy to device)."""
        from quark.runtime.kernels import randn

        return randn(shape, dtype)

    # ── Cleanup ──

    def __del__(self):
        try:
            if hasattr(self, "_storage") and self._storage is not None:
                self._storage.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"QuarkTensor(shape={self._shape}, dtype='{self._dtype}', ptr=0x{self.data_ptr():x})"

    def tolist(self) -> list:
        """Convert to nested Python list. Small tensors only."""
        import struct as _struct

        t = self.astype("f32").contiguous()
        raw = t.to_bytes()
        n = t.numel()
        flat = list(_struct.unpack(f"<{n}f", raw))
        return _reshape_flat_to_nested(flat, t._shape)


# ── Backward compat alias ────────────────────────────────────
CudaTensor = QuarkTensor
