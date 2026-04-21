"""PopcornTensor — mutable GPU tensor without torch or numpy.

EXEMPT FROM 500-LINE RULE: PopcornTensor is the single tensor type for
both backends. Storage, strides, factories, view ops, slicing, and
arithmetic dispatch all belong together — splitting would fragment the
type and force circular imports between the pieces.

A ``PopcornTensor`` wraps a ``cuMemAlloc``'d device pointer (CUDA) or
an ``MTLBuffer`` (Metal, Stage 2) with shape, dtype, strides, and
offset metadata. It supports the operations the launcher needs:

    t.data_ptr()        # → int (device pointer to first element)
    t.shape / t.dtype   # metadata
    t.device            # DeviceSpec(type="cuda", index=0)
    t.is_contiguous()   # True iff strides are row-major with no gaps

And the raw data transfer API (no numpy dependency):

    PopcornTensor.from_bytes(buf, shape, dtype)  # host → device
    t.to_bytes()                                  # device → host

Plus arithmetic, slicing, reshape, permute — all dispatched to
on-device PTX utility kernels (see ``runtime/kernels.py``).

Thread-safety: NOT thread-safe. One CUDA context per process.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from popcorn.runtime.utils import (
    f32_bytes_to_bf16_bytes as _f32_bytes_to_bf16_bytes,
)
from popcorn.runtime.utils import (
    f32_to_bf16_numpy as _f32_to_bf16_numpy,
)
from popcorn.runtime.utils import (
    flatten_list as _flatten_list,
)
from popcorn.runtime.utils import (
    infer_shape as _infer_shape,
)
from popcorn.runtime.utils import (
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
    from popcorn.runtime.cuda import CudaRuntime

    return CudaRuntime.instance()


# ── Storage ───────────────────────────────────────────────────


class _CudaStorage:
    """Reference-counted device memory block.

    Multiple PopcornTensors can share the same storage (views from
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
        from popcorn.graph import active_stream

        stream = active_stream()
        ptr = _runtime().mem_alloc(nbytes)
        storage = _CudaStorage(ptr, nbytes, alloc_stream=stream)
        if stream:
            from popcorn.graph import register_capture_storage

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

    Holds a reference to the source object (torch.Tensor, mx.array)
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


# ── PopcornTensor ─────────────────────────────────────────────


class PopcornTensor:
    """Mutable GPU tensor backed by ``cuMemAlloc``.

    Supports views (slicing, reshape, permute) via stride metadata.
    Arithmetic ops dispatch to PTX utility kernels.
    """

    __slots__ = ("_device", "_dtype", "_offset", "_shape", "_storage", "_strides")

    def __init__(
        self,
        storage: _CudaStorage | _BorrowedStorage,
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
        self._device = DeviceSpec("cuda", 0)

    # ── Factories ──

    @staticmethod
    def empty(*shape: int, dtype: str = "bf16") -> PopcornTensor:
        """Allocate uninitialized device memory.

        During graph capture, uses cuMemAllocAsync (stream-ordered
        allocation that works during capture). Otherwise uses cuMemAlloc.
        Tensors allocated during capture are held by the CapturedGraph
        to prevent GC from freeing memory the graph still references.
        """
        numel = math.prod(shape) if shape else 1
        nbytes = numel * PC_BYTES[dtype]
        storage = _CudaStorage.alloc(nbytes)
        return PopcornTensor(storage, tuple(shape), _contiguous_strides(tuple(shape)), 0, dtype)

    @staticmethod
    def zeros(*shape: int, dtype: str = "bf16") -> PopcornTensor:
        t = PopcornTensor.empty(*shape, dtype=dtype)
        if t._storage.nbytes > 0:
            _runtime().memset_d8(t._storage.ptr, 0, t._storage.nbytes)
        return t

    @staticmethod
    def from_bytes(
        buf: bytes | bytearray | memoryview, shape: tuple[int, ...], dtype: str
    ) -> PopcornTensor:
        """Copy raw bytes from host to device.

        Memoryview path is zero-copy: numpy.frombuffer exposes the host
        pointer (e.g. mmap page) straight to cuMemcpyHtoD. Bytes path
        keeps the legacy ctypes-array copy for non-mmap callers.
        """
        numel = math.prod(shape) if shape else 1
        expected = numel * PC_BYTES[dtype]
        if len(buf) != expected:
            raise ValueError(f"from_bytes: want {expected} bytes, got {len(buf)}")
        t = PopcornTensor.empty(*shape, dtype=dtype)
        if expected == 0:
            return t
        if isinstance(buf, memoryview):
            import numpy as _np

            arr = _np.frombuffer(buf, dtype=_np.uint8, count=expected)
            _runtime().memcpy_htod(t._storage.ptr, arr.ctypes.data, expected)
        else:
            host_arr = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
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
        _runtime().memcpy_dtoh(ctypes.addressof(host_buf), t.data_ptr(), nbytes)
        return bytes(host_buf)

    @staticmethod
    def from_numpy(arr, dtype: str | None = None) -> PopcornTensor:
        """Convenience: numpy array → PopcornTensor. Tests/debug only."""
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
        raw = arr.tobytes()
        return PopcornTensor.from_bytes(raw, shape, dtype)

    def to_numpy(self):
        """Convenience: PopcornTensor → numpy array. Tests/debug only."""
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
        raw = self.to_bytes()
        np_dt = _PC_TO_NP[self._dtype]
        return np.frombuffer(raw, dtype=np_dt).reshape(self._shape)

    @staticmethod
    def from_list(data, dtype: str = "f32") -> PopcornTensor:
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
        return PopcornTensor.from_bytes(raw, shape, dtype)

    # ── Zero-copy wrapping of external tensors ──

    @staticmethod
    def from_torch(t) -> PopcornTensor:
        """Zero-copy wrap a ``torch.Tensor`` as a PopcornTensor.

        The torch tensor must be on a CUDA device and contiguous.
        PopcornTensor borrows the data pointer — no memcpy. The
        torch tensor is kept alive via a reference in the storage.
        """
        if not t.is_contiguous():
            raise ValueError("from_torch: tensor must be contiguous")
        if t.device.type != "cuda":
            raise ValueError(f"from_torch: tensor must be on CUDA, got {t.device}")
        import torch

        _TORCH_TO_PC = {
            torch.float32: "f32",
            torch.float16: "f16",
            torch.bfloat16: "bf16",
            torch.int32: "s32",
            torch.int64: "s64",
            torch.int8: "s8",
            torch.uint8: "u8",
        }
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is not None:
            _TORCH_TO_PC[fp8] = "u8"  # store fp8 as u8 raw bytes
        dtype = _TORCH_TO_PC.get(t.dtype)
        if dtype is None:
            raise ValueError(f"from_torch: unsupported dtype {t.dtype}")
        shape = tuple(t.shape)
        nbytes = t.nelement() * t.element_size()
        storage = _BorrowedStorage(t.data_ptr(), nbytes, owner=t)
        return PopcornTensor(storage, shape, _contiguous_strides(shape), 0, dtype)

    @staticmethod
    def from_mlx(t) -> PopcornTensor:
        """Zero-copy wrap an ``mx.array`` as a PopcornTensor.

        Apple silicon uses unified memory, so ``mx.array``'s data
        pointer is valid on both CPU and GPU. PopcornTensor borrows
        the pointer — no memcpy.
        """
        import mlx.core as mx

        _MLX_TO_PC = {
            mx.float32: "f32",
            mx.float16: "f16",
            mx.bfloat16: "bf16",
            mx.int32: "s32",
            mx.int64: "s64",
            mx.int8: "s8",
            mx.uint8: "u8",
            mx.uint16: "u16",
            mx.uint32: "u32",
        }
        dtype = _MLX_TO_PC.get(t.dtype)
        if dtype is None:
            raise ValueError(f"from_mlx: unsupported dtype {t.dtype}")
        shape = tuple(t.shape)
        elem = PC_BYTES.get(dtype, 4)
        nbytes = t.size * elem
        # mx.array on Apple silicon: the data pointer from the array
        # object is in unified memory (CPU + GPU visible).
        import ctypes as _ct

        # Force evaluation so the data is materialized.
        mx.eval(t)
        raw_ptr = _ct.cast(
            _ct.c_void_p(t.__array_interface__["data"][0]),
            _ct.c_void_p,
        ).value
        arr_ptr = raw_ptr if raw_ptr is not None else 0
        storage = _BorrowedStorage(arr_ptr, nbytes, owner=t)
        return PopcornTensor(storage, shape, _contiguous_strides(shape), 0, dtype)

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

    def reshape(self, *shape: int) -> PopcornTensor:
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
        return PopcornTensor(t._storage, shape, new_strides, t._offset, t._dtype)

    def permute(self, *dims: int) -> PopcornTensor:
        """Transpose / permute axes. Returns a view (no copy)."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        if sorted(dims) != list(range(self.ndim)):
            raise ValueError(f"permute: invalid dims {dims} for ndim={self.ndim}")
        new_shape = tuple(self._shape[d] for d in dims)
        new_strides = tuple(self._strides[d] for d in dims)
        self._storage.retain()
        return PopcornTensor(self._storage, new_shape, new_strides, self._offset, self._dtype)

    @property
    def T(self) -> PopcornTensor:
        """Transpose last two dims (convenience for matmul)."""
        if self.ndim < 2:
            return self
        dims = list(range(self.ndim))
        dims[-1], dims[-2] = dims[-2], dims[-1]
        return self.permute(*dims)

    def contiguous(self) -> PopcornTensor:
        """Return a contiguous copy, or self if already contiguous."""
        if self.is_contiguous() and self._offset == 0:
            return self
        # Need to copy via _copy_strided utility kernel.
        from popcorn.runtime.kernels import copy_strided

        return copy_strided(self)

    # ── Slicing ──

    def __getitem__(self, key) -> PopcornTensor:
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
                    raise NotImplementedError("PopcornTensor: step != 1 slicing not yet supported")
                size = max(0, stop - start)
                offset += start * self._strides[dim]
                new_shape.append(size)
                new_strides.append(self._strides[dim])
                dim += 1

            else:
                raise TypeError(f"PopcornTensor: unsupported index type {type(k).__name__}")

        # Append remaining dimensions.
        for d in range(dim, self.ndim):
            new_shape.append(self._shape[d])
            new_strides.append(self._strides[d])

        self._storage.retain()
        return PopcornTensor(
            self._storage, tuple(new_shape), tuple(new_strides), offset, self._dtype
        )

    def __setitem__(self, key, value) -> None:
        """Write into a slice of the tensor. ``value`` must be a
        PopcornTensor or a scalar."""
        dst_view = self[key]
        if isinstance(value, PopcornTensor):
            if value.shape != dst_view.shape:
                raise ValueError(f"__setitem__: shape mismatch {value.shape} vs {dst_view.shape}")
            from popcorn.runtime.kernels import copy_strided_into

            copy_strided_into(dst_view, value)
        elif isinstance(value, (int, float)):
            from popcorn.runtime.kernels import fill_scalar

            fill_scalar(dst_view, value)
        else:
            raise TypeError(f"__setitem__: unsupported value type {type(value).__name__}")
        # Release the extra retain from __getitem__.
        dst_view._storage.release()

    # ── Arithmetic (dispatched to utility kernels) ──

    def __add__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary

        return elemwise_binary("add", self, other)

    def __radd__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary

        return elemwise_binary("add", self, other)

    def __sub__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary

        return elemwise_binary("sub", self, other)

    def __rsub__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary_scalar_lhs

        return elemwise_binary_scalar_lhs("sub", other, self)

    def __mul__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary

        return elemwise_binary("mul", self, other)

    def __rmul__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary

        return elemwise_binary("mul", self, other)

    def __truediv__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary

        return elemwise_binary("div", self, other)

    def __rtruediv__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_binary_scalar_lhs

        return elemwise_binary_scalar_lhs("div", other, self)

    def __neg__(self) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_unary

        return elemwise_unary("neg", self)

    def __abs__(self) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_unary

        return elemwise_unary("abs", self)

    def __matmul__(self, other) -> PopcornTensor:
        from popcorn.runtime.kernels import matmul

        return matmul(self, other)

    def exp(self) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_unary

        return elemwise_unary("exp", self)

    def sin(self) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_unary

        return elemwise_unary("sin", self)

    def cos(self) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_unary

        return elemwise_unary("cos", self)

    def sqrt(self) -> PopcornTensor:
        from popcorn.runtime.kernels import elemwise_unary

        return elemwise_unary("sqrt", self)

    def sum(self, dim: int | None = None) -> PopcornTensor:
        from popcorn.runtime.kernels import reduce_op

        return reduce_op("sum", self, dim)

    def max(self, dim: int | None = None) -> PopcornTensor:
        from popcorn.runtime.kernels import reduce_op

        return reduce_op("max", self, dim)

    def astype(self, dtype: str) -> PopcornTensor:
        """Cast to a different dtype. Allocates a new tensor each time."""
        if dtype == self._dtype:
            return self
        from popcorn.runtime.kernels import cast

        return cast(self, dtype)

    def item(self) -> float:
        """Extract a single-element tensor as a Python float."""
        if self.numel() != 1:
            raise ValueError(f"item(): tensor has {self.numel()} elements, expected 1")
        raw = self.astype("f32").to_bytes()
        import struct as _struct

        return _struct.unpack("<f", raw)[0]

    def clone(self) -> PopcornTensor:
        """Return a contiguous copy with its own storage (async D2D)."""
        dst = PopcornTensor.empty(*self._shape, dtype=self._dtype)
        from popcorn.runtime.cuda import CudaRuntime

        nbytes = self.numel() * PC_BYTES[self._dtype]
        CudaRuntime.instance().memcpy_dtod(dst.data_ptr(), self.contiguous().data_ptr(), nbytes)
        return dst

    def zero_(self) -> PopcornTensor:
        """Zero this tensor in-place (async). Returns self."""
        from popcorn.runtime.cuda import CudaRuntime

        nbytes = self.numel() * PC_BYTES[self._dtype]
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
        from popcorn.runtime.kernels import scalar_increment

        scalar_increment(self)

    def copy_from(self, src: PopcornTensor) -> None:
        """Copy data from ``src`` into this tensor (same shape/dtype)."""
        from popcorn.runtime.kernels import copy_strided_into

        copy_strided_into(self, src)

    @staticmethod
    def cat(tensors: list[PopcornTensor], dim: int = 0) -> PopcornTensor:
        """Concatenate tensors along a dimension."""
        from popcorn.runtime.kernels import cat

        return cat(tensors, dim)

    @staticmethod
    def randn(*shape: int, dtype: str = "f32") -> PopcornTensor:
        """Random normal tensor (host-side PRNG, then copy to device)."""
        from popcorn.runtime.kernels import randn

        return randn(shape, dtype)

    # ── Cleanup ──

    def __del__(self):
        try:
            if hasattr(self, "_storage") and self._storage is not None:
                self._storage.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"PopcornTensor(shape={self._shape}, dtype='{self._dtype}', ptr=0x{self.data_ptr():x})"
        )

    def tolist(self) -> list:
        """Convert to nested Python list. Small tensors only."""
        import struct as _struct

        t = self.astype("f32").contiguous()
        raw = t.to_bytes()
        n = t.numel()
        flat = list(_struct.unpack(f"<{n}f", raw))
        return _reshape_flat_to_nested(flat, t._shape)


# ── Backward compat alias ────────────────────────────────────
CudaTensor = PopcornTensor
