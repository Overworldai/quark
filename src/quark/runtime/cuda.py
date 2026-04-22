"""ctypes binding to libcuda — the CUDA driver API.

EXEMPT FROM 500-LINE RULE

Per quark launcher proposal §4.1. This is a thin wrapper over the
native CUDA driver library, with one Python method per `cu*` symbol
we actually need. **No torch imports** — this layer is the platform
abstraction; torch enters at the Driver / Launcher layer above.

Why the driver API and not the runtime API: the CUDA *runtime* API
(`cudart`) is what `cudaXxx` functions use, and it has its own
context/device management that fights with torch's primary context.
The CUDA *driver* API (`cuXxx`, in `libcuda`) lets us share torch's
primary context cleanly — we call `cuDevicePrimaryCtxRetain` +
`cuCtxSetCurrent` once at startup so every libcuda call we make
operates on the same context torch already uses, and tensor
`data_ptr()` values stay valid across the boundary.

Sizing target per the proposal: ~400 LoC ctypes decls + ~200 LoC
wrapper. We're closer to the lower end here because we ship only
the symbols the launcher actually needs — graph capture, events,
etc. follow the same pattern and can be added incrementally.

Error handling: every `libcuda.cuXxx(...)` call goes through
`_check(res)` which raises `CudaError(name, msg)` on nonzero, with
the message coming from `cuGetErrorString` / `cuGetErrorName`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# CUresult constants we explicitly check for
# ---------------------------------------------------------------------------

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_DEINITIALIZED = 4

# CU_DEVICE_ATTRIBUTE values used by CudaDriver.probe.
# Source: cuda.h (CUdevice_attribute enum). Values are stable ABI.
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82

# CU_FUNC_ATTRIBUTE values.
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

# Stream capture modes.
CU_STREAM_CAPTURE_MODE_GLOBAL = 0
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
CU_STREAM_CAPTURE_MODE_RELAXED = 2


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


class CudaError(RuntimeError):
    """Raised when a CUDA driver API call returns a nonzero CUresult."""

    def __init__(self, code: int, name: str, message: str) -> None:
        super().__init__(f"{name}({code}): {message}")
        self.code = code
        self.name = name
        self.message = message


def _load_libcuda() -> ctypes.CDLL:
    """Load libcuda.so / nvcuda.dll.

    On Linux we try the SONAME (`libcuda.so.1`) first since the bare
    `libcuda.so` is usually a dev-package symlink that may be absent
    on a runtime-only install. Falls back through the Python ctypes
    util resolver. On Windows we look for `nvcuda.dll` directly.
    """
    if sys.platform == "win32":
        for name in ("nvcuda.dll", "nvcuda"):
            try:
                return ctypes.WinDLL(name)  # type: ignore[attr-defined]
            except OSError:
                continue
        raise CudaError(0, "LIBRARY_NOT_FOUND", "nvcuda.dll not found on PATH")

    candidates = ["libcuda.so.1", "libcuda.so"]
    found = ctypes.util.find_library("cuda")
    if found is not None:
        candidates.insert(0, found)
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise CudaError(
        0,
        "LIBRARY_NOT_FOUND",
        "libcuda.so not found. NVIDIA driver may not be installed.",
    )


# ---------------------------------------------------------------------------
# CudaRuntime — the public API surface §4.1 specifies
# ---------------------------------------------------------------------------


class CudaRuntime:
    """ctypes wrapper over libcuda. One process-wide instance shared
    by every CudaDriver — call `CudaRuntime.instance()` to get it."""

    _lock = threading.Lock()
    _instance: Optional[CudaRuntime] = None

    # ---- singleton + lifecycle ----

    def __init__(self) -> None:
        self._lib = _load_libcuda()
        self._setup_signatures()
        self._cu_init(0)
        # Default to sharing the primary context on device 0. Drivers
        # that target a different device call retain_primary_context
        # again — primary contexts are refcounted by libcuda so the
        # double-retain is harmless and ensures any direct
        # CudaRuntime user (without a CudaDriver wrapper) has a
        # current context for module/launch calls.
        if self.device_count() > 0:
            self.retain_primary_context(0)

    @classmethod
    def instance(cls) -> CudaRuntime:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """True if libcuda loads + at least one device is present."""
        try:
            rt = cls.instance()
            return rt.device_count() > 0
        except CudaError:
            return False
        except OSError:
            return False

    # ---- internal: ctypes signature setup ----

    def _setup_signatures(self) -> None:
        """Bind argument and return types for every libcuda symbol we
        wrap. Each wrapper method below assumes these are set."""
        L = self._lib

        L.cuInit.argtypes = [ctypes.c_uint]
        L.cuInit.restype = ctypes.c_int

        L.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        L.cuGetErrorName.restype = ctypes.c_int

        L.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        L.cuGetErrorString.restype = ctypes.c_int

        L.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.cuDeviceGetCount.restype = ctypes.c_int

        L.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.cuDeviceGet.restype = ctypes.c_int

        L.cuDeviceGetName.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.cuDeviceGetName.restype = ctypes.c_int

        L.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,  # CUdevice_attribute
            ctypes.c_int,  # CUdevice
        ]
        L.cuDeviceGetAttribute.restype = ctypes.c_int

        L.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # CUcontext*
            ctypes.c_int,
        ]
        L.cuDevicePrimaryCtxRetain.restype = ctypes.c_int

        L.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        L.cuCtxSetCurrent.restype = ctypes.c_int

        L.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # CUmodule*
            ctypes.c_void_p,  # const void* image
        ]
        L.cuModuleLoadData.restype = ctypes.c_int

        L.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # CUfunction*
            ctypes.c_void_p,  # CUmodule
            ctypes.c_char_p,
        ]
        L.cuModuleGetFunction.restype = ctypes.c_int

        L.cuFuncSetAttribute.argtypes = [
            ctypes.c_void_p,  # CUfunction
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.cuFuncSetAttribute.restype = ctypes.c_int

        L.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,  # CUfunction f
            ctypes.c_uint,  # gridDimX
            ctypes.c_uint,  # gridDimY
            ctypes.c_uint,  # gridDimZ
            ctypes.c_uint,  # blockDimX
            ctypes.c_uint,  # blockDimY
            ctypes.c_uint,  # blockDimZ
            ctypes.c_uint,  # sharedMemBytes
            ctypes.c_void_p,  # CUstream hStream
            ctypes.POINTER(ctypes.c_void_p),  # void** kernelParams
            ctypes.POINTER(ctypes.c_void_p),  # void** extra
        ]
        L.cuLaunchKernel.restype = ctypes.c_int

        L.cuStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # CUstream*
            ctypes.c_uint,  # flags
        ]
        L.cuStreamCreate.restype = ctypes.c_int

        L.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        L.cuStreamSynchronize.restype = ctypes.c_int

        L.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
        L.cuStreamDestroy_v2.restype = ctypes.c_int

        L.cuModuleUnload.argtypes = [ctypes.c_void_p]
        L.cuModuleUnload.restype = ctypes.c_int

        L.cuEventCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        L.cuEventCreate.restype = ctypes.c_int

        L.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.cuEventRecord.restype = ctypes.c_int

        L.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        L.cuEventSynchronize.restype = ctypes.c_int

        L.cuEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        L.cuEventElapsedTime.restype = ctypes.c_int

        L.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
        L.cuEventDestroy_v2.restype = ctypes.c_int

        # Memory management (torch-free tensor path).
        L.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # CUdeviceptr*
            ctypes.c_size_t,
        ]
        L.cuMemAlloc_v2.restype = ctypes.c_int

        L.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        L.cuMemFree_v2.restype = ctypes.c_int

        # Stream-ordered alloc/free (CUDA 11.2+). Works during graph capture.
        L.cuMemAllocAsync.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # CUdeviceptr*
            ctypes.c_size_t,  # bytesize
            ctypes.c_void_p,  # CUstream
        ]
        L.cuMemAllocAsync.restype = ctypes.c_int

        L.cuMemFreeAsync.argtypes = [
            ctypes.c_uint64,  # CUdeviceptr
            ctypes.c_void_p,  # CUstream
        ]
        L.cuMemFreeAsync.restype = ctypes.c_int

        L.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64,  # CUdeviceptr dst
            ctypes.c_void_p,  # const void* src
            ctypes.c_size_t,
        ]
        L.cuMemcpyHtoD_v2.restype = ctypes.c_int

        L.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,  # void* dst
            ctypes.c_uint64,  # CUdeviceptr src
            ctypes.c_size_t,
        ]
        L.cuMemcpyDtoH_v2.restype = ctypes.c_int

        L.cuMemsetD8_v2.argtypes = [
            ctypes.c_uint64,  # CUdeviceptr dst
            ctypes.c_ubyte,  # unsigned char value
            ctypes.c_size_t,
        ]
        L.cuMemsetD8_v2.restype = ctypes.c_int

        # Async memory operations (stream-ordered, required for graph capture).
        L.cuMemcpyDtoDAsync_v2.argtypes = [
            ctypes.c_uint64,  # dst
            ctypes.c_uint64,  # src
            ctypes.c_size_t,  # nbytes
            ctypes.c_void_p,  # CUstream
        ]
        L.cuMemcpyDtoDAsync_v2.restype = ctypes.c_int

        L.cuMemsetD8Async.argtypes = [
            ctypes.c_uint64,
            ctypes.c_ubyte,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        L.cuMemsetD8Async.restype = ctypes.c_int

        L.cuMemsetD16Async.argtypes = [
            ctypes.c_uint64,
            ctypes.c_ushort,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        L.cuMemsetD16Async.restype = ctypes.c_int

        L.cuMemsetD32Async.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        L.cuMemsetD32Async.restype = ctypes.c_int

        # Graph capture API.
        L.cuStreamBeginCapture_v2.argtypes = [
            ctypes.c_void_p,  # CUstream
            ctypes.c_int,  # CUstreamCaptureMode
        ]
        L.cuStreamBeginCapture_v2.restype = ctypes.c_int

        L.cuStreamEndCapture.argtypes = [
            ctypes.c_void_p,  # CUstream
            ctypes.POINTER(ctypes.c_void_p),  # CUgraph*
        ]
        L.cuStreamEndCapture.restype = ctypes.c_int

        L.cuGraphInstantiateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # CUgraphExec*
            ctypes.c_void_p,  # CUgraph
            ctypes.c_uint64,  # flags
        ]
        L.cuGraphInstantiateWithFlags.restype = ctypes.c_int

        L.cuGraphLaunch.argtypes = [
            ctypes.c_void_p,  # CUgraphExec
            ctypes.c_void_p,  # CUstream
        ]
        L.cuGraphLaunch.restype = ctypes.c_int

        L.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
        L.cuGraphExecDestroy.restype = ctypes.c_int

        L.cuGraphDestroy.argtypes = [ctypes.c_void_p]
        L.cuGraphDestroy.restype = ctypes.c_int

        # Typed memset for fill operations.
        L.cuMemsetD16_v2.argtypes = [
            ctypes.c_uint64,  # CUdeviceptr dst
            ctypes.c_ushort,  # unsigned short value
            ctypes.c_size_t,  # count (number of u16 elements)
        ]
        L.cuMemsetD16_v2.restype = ctypes.c_int

        L.cuMemsetD32_v2.argtypes = [
            ctypes.c_uint64,  # CUdeviceptr dst
            ctypes.c_uint,  # unsigned int value
            ctypes.c_size_t,  # count (number of u32 elements)
        ]
        L.cuMemsetD32_v2.restype = ctypes.c_int

    # ---- internal: error checking ----

    def _check(self, res: int) -> None:
        """Raise CudaError on a nonzero CUresult."""
        if res == CUDA_SUCCESS:
            return
        # Resolve name + message via libcuda's error helpers. These
        # themselves return CUresult; if they fail, fall back to a
        # generic message.
        name_p = ctypes.c_char_p()
        msg_p = ctypes.c_char_p()
        try:
            self._lib.cuGetErrorName(res, ctypes.byref(name_p))
            self._lib.cuGetErrorString(res, ctypes.byref(msg_p))
            name = name_p.value.decode() if name_p.value else f"CUresult({res})"
            msg = msg_p.value.decode() if msg_p.value else "<no message>"
        except Exception:
            name = f"CUresult({res})"
            msg = "<error string lookup failed>"
        raise CudaError(res, name, msg)

    # ---- lifecycle ----

    def _cu_init(self, flags: int = 0) -> None:
        """Call cuInit. Idempotent — re-calls return CUDA_SUCCESS."""
        self._check(self._lib.cuInit(flags))

    def device_count(self) -> int:
        n = ctypes.c_int(0)
        self._check(self._lib.cuDeviceGetCount(ctypes.byref(n)))
        return int(n.value)

    def get_device_attribute(self, dev: int, attr: int) -> int:
        cudev = self._cu_device(dev)
        out = ctypes.c_int(0)
        self._check(self._lib.cuDeviceGetAttribute(ctypes.byref(out), attr, cudev))
        return int(out.value)

    def get_device_name(self, dev: int) -> str:
        cudev = self._cu_device(dev)
        buf = ctypes.create_string_buffer(256)
        self._check(self._lib.cuDeviceGetName(buf, 256, cudev))
        return buf.value.decode()

    def _cu_device(self, dev: int) -> int:
        cudev = ctypes.c_int(0)
        self._check(self._lib.cuDeviceGet(ctypes.byref(cudev), dev))
        return cudev.value

    def retain_primary_context(self, dev: int) -> None:
        """Retain + activate the primary context on `dev`. We share
        torch's primary context this way — every libcuda call after
        this point operates on the same context torch is using, so
        tensor `data_ptr()` values are valid pointers in our calls."""
        cudev = self._cu_device(dev)
        ctx = ctypes.c_void_p(0)
        self._check(self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), cudev))
        self._check(self._lib.cuCtxSetCurrent(ctx))

    # ---- module loading (PTX text input) ----

    def module_load_data(self, ptx: bytes) -> int:
        """Pass raw PTX text to the in-driver assembler. Returns the
        CUmodule handle as an int (we treat handles as opaque ints
        across the wrapper boundary)."""
        if not isinstance(ptx, (bytes, bytearray)):
            raise TypeError("module_load_data: ptx must be bytes")
        if not ptx.endswith(b"\x00"):
            ptx = bytes(ptx) + b"\x00"
        mod = ctypes.c_void_p(0)
        self._check(self._lib.cuModuleLoadData(ctypes.byref(mod), ptx))
        return int(mod.value or 0)

    def module_get_function(self, mod: int, name: str) -> int:
        func = ctypes.c_void_p(0)
        self._check(
            self._lib.cuModuleGetFunction(ctypes.byref(func), ctypes.c_void_p(mod), name.encode())
        )
        return int(func.value or 0)

    def func_set_attribute(self, func: int, attr: int, value: int) -> None:
        """Set a function attribute. The most important one for our
        path is MAX_DYNAMIC_SHARED_SIZE_BYTES, which has to be opted
        into for kernels that use >48KB of dynamic shared memory."""
        self._check(self._lib.cuFuncSetAttribute(ctypes.c_void_p(func), attr, value))

    def module_unload(self, mod: int) -> None:
        self._check(self._lib.cuModuleUnload(ctypes.c_void_p(mod)))

    # ---- launch ----

    def launch_kernel(
        self,
        func: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[int],
        smem: int,
        stream: int,
    ) -> None:
        """Launch a kernel. `args` is a list of integer addresses —
        each is the address of a C variable holding one kernel
        parameter (a pointer or a packed scalar). The caller is
        responsible for keeping those backing C variables alive across
        this call (the driver dereferences them but doesn't take
        ownership)."""
        n = len(args)
        ArgArray = ctypes.c_void_p * n if n else ctypes.c_void_p * 1
        kernel_params = ArgArray(*[ctypes.c_void_p(a) for a in args]) if n else None
        gx, gy, gz = grid
        bx, by, bz = block
        self._check(
            self._lib.cuLaunchKernel(
                ctypes.c_void_p(func),
                ctypes.c_uint(gx),
                ctypes.c_uint(gy),
                ctypes.c_uint(gz),
                ctypes.c_uint(bx),
                ctypes.c_uint(by),
                ctypes.c_uint(bz),
                ctypes.c_uint(smem),
                ctypes.c_void_p(stream),
                kernel_params,
                None,
            )
        )

    # ---- streams ----

    def stream_create(self) -> int:
        s = ctypes.c_void_p(0)
        self._check(self._lib.cuStreamCreate(ctypes.byref(s), 0))
        return int(s.value or 0)

    def stream_from_external(self, ptr: int) -> int:
        """No-op: a CUstream IS just an opaque void* in the driver
        API, so a torch-owned stream pointer can be passed straight
        to launch_kernel without wrapping. This method exists for
        symmetry with the OpenCL/Metal runtimes that DO need to
        bridge their native stream type."""
        return int(ptr)

    def stream_synchronize(self, stream: int) -> None:
        self._check(self._lib.cuStreamSynchronize(ctypes.c_void_p(stream)))

    def stream_destroy(self, stream: int) -> None:
        self._check(self._lib.cuStreamDestroy_v2(ctypes.c_void_p(stream)))

    # ---- events (for timing) ----

    def event_create(self, flags: int = 0) -> int:
        ev = ctypes.c_void_p(0)
        self._check(self._lib.cuEventCreate(ctypes.byref(ev), flags))
        return int(ev.value or 0)

    def event_record(self, ev: int, stream: int) -> None:
        self._check(self._lib.cuEventRecord(ctypes.c_void_p(ev), ctypes.c_void_p(stream)))

    def event_synchronize(self, ev: int) -> None:
        self._check(self._lib.cuEventSynchronize(ctypes.c_void_p(ev)))

    def event_elapsed_time(self, start: int, end: int) -> float:
        ms = ctypes.c_float(0.0)
        self._check(
            self._lib.cuEventElapsedTime(
                ctypes.byref(ms), ctypes.c_void_p(start), ctypes.c_void_p(end)
            )
        )
        return float(ms.value)

    def event_destroy(self, ev: int) -> None:
        self._check(self._lib.cuEventDestroy_v2(ctypes.c_void_p(ev)))

    # ---- device memory (torch-free tensor path) ----

    def mem_alloc(self, nbytes: int, stream: int = 0) -> int:
        """Allocate device memory. Stream-aware: uses cuMemAllocAsync
        when a stream is active (required during graph capture)."""
        if stream == 0:
            from quark.graph import active_stream

            stream = active_stream()
        ptr = ctypes.c_uint64(0)
        if stream:
            self._check(
                self._lib.cuMemAllocAsync(
                    ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_void_p(stream)
                )
            )
        else:
            self._check(self._lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return int(ptr.value)

    def mem_free(self, ptr: int, stream: int = 0) -> None:
        """Free device memory. Stream-aware: uses cuMemFreeAsync
        when a stream is active."""
        if stream == 0:
            from quark.graph import active_stream

            stream = active_stream()
        if stream:
            self._check(self._lib.cuMemFreeAsync(ctypes.c_uint64(ptr), ctypes.c_void_p(stream)))
        else:
            self._check(self._lib.cuMemFree_v2(ctypes.c_uint64(ptr)))

    def memcpy_htod(self, dst: int, src: int, nbytes: int) -> None:
        """Host → device. ``src`` is the host pointer (e.g. ``arr.ctypes.data``).

        Runs at full PCIe bandwidth only if ``src`` is page-locked
        (``cuMemHostRegister`` or ``cuMemHostAlloc``). Pageable HtoD
        stages through a small internal pinned buffer at ~1/4 speed
        and serializes per driver call — hence the register dance in
        ``quark.nn.io.load_safetensors``.
        """
        self._check(
            self._lib.cuMemcpyHtoD_v2(
                ctypes.c_uint64(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes)
            )
        )

    # Host-memory flags (CUmemhostregister_flags).
    _MEMHOSTREGISTER_PORTABLE = 0x01
    _MEMHOSTREGISTER_DEVICEMAP = 0x02
    _MEMHOSTREGISTER_READ_ONLY = 0x08

    def memhost_register(
        self,
        ptr: int,
        nbytes: int,
        *,
        read_only: bool = False,
    ) -> None:
        """Page-lock an existing host allocation so HtoD/DtoH use full
        PCIe bandwidth. ``ptr`` must be page-aligned — OS-returned mmaps
        and aligned malloc qualify; arbitrary Python-side buffers do not.

        ``read_only=True`` (CUDA 11.2+) tells the driver it may skip the
        writable-pages path; useful for weight loading since we only
        ever copy *out* of the registered region.
        """
        flags = self._MEMHOSTREGISTER_PORTABLE
        if read_only:
            flags |= self._MEMHOSTREGISTER_READ_ONLY
        self._check(
            self._lib.cuMemHostRegister_v2(
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_uint(flags),
            )
        )

    def memhost_unregister(self, ptr: int) -> None:
        """Undo a prior :meth:`memhost_register`."""
        self._check(self._lib.cuMemHostUnregister(ctypes.c_void_p(ptr)))

    def memcpy_dtoh(self, dst: int, src: int, nbytes: int) -> None:
        """Device → host. ``dst`` is the host pointer."""
        self._check(
            self._lib.cuMemcpyDtoH_v2(
                ctypes.c_void_p(dst), ctypes.c_uint64(src), ctypes.c_size_t(nbytes)
            )
        )

    def memcpy_dtod(self, dst: int, src: int, nbytes: int, stream: int = 0) -> None:
        """Device → device. Always async (cuMemcpyDtoDAsync)."""
        if stream == 0:
            from quark.graph import active_stream

            stream = active_stream()
        self._check(
            self._lib.cuMemcpyDtoDAsync_v2(
                ctypes.c_uint64(dst),
                ctypes.c_uint64(src),
                ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream),
            )
        )

    def memset_d8(self, ptr: int, value: int, nbytes: int, stream: int = 0) -> None:
        """Always async (cuMemsetD8Async)."""
        if stream == 0:
            from quark.graph import active_stream

            stream = active_stream()
        self._check(
            self._lib.cuMemsetD8Async(
                ctypes.c_uint64(ptr),
                ctypes.c_ubyte(value),
                ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream),
            )
        )

    def memset_d16(self, ptr: int, value: int, count: int, stream: int = 0) -> None:
        """Always async (cuMemsetD16Async)."""
        if stream == 0:
            from quark.graph import active_stream

            stream = active_stream()
        self._check(
            self._lib.cuMemsetD16Async(
                ctypes.c_uint64(ptr),
                ctypes.c_ushort(value & 0xFFFF),
                ctypes.c_size_t(count),
                ctypes.c_void_p(stream),
            )
        )

    def memset_d32(self, ptr: int, value: int, count: int, stream: int = 0) -> None:
        """Always async (cuMemsetD32Async)."""
        if stream == 0:
            from quark.graph import active_stream

            stream = active_stream()
        self._check(
            self._lib.cuMemsetD32Async(
                ctypes.c_uint64(ptr),
                ctypes.c_uint(value & 0xFFFFFFFF),
                ctypes.c_size_t(count),
                ctypes.c_void_p(stream),
            )
        )

    # ---- graph capture ----

    def graph_begin_capture(
        self, stream: int, mode: int = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
    ) -> None:
        """Begin capturing all kernel launches on ``stream`` into a graph."""
        self._check(self._lib.cuStreamBeginCapture_v2(ctypes.c_void_p(stream), mode))

    def graph_end_capture(self, stream: int) -> int:
        """End capture and return the CUgraph handle."""
        graph = ctypes.c_void_p(0)
        self._check(self._lib.cuStreamEndCapture(ctypes.c_void_p(stream), ctypes.byref(graph)))
        return int(graph.value or 0)

    def graph_instantiate(self, graph: int) -> int:
        """Instantiate a CUgraph into a CUgraphExec for replay."""
        exec_handle = ctypes.c_void_p(0)
        self._check(
            self._lib.cuGraphInstantiateWithFlags(
                ctypes.byref(exec_handle), ctypes.c_void_p(graph), ctypes.c_uint64(0)
            )
        )
        return int(exec_handle.value or 0)

    def graph_launch(self, exec_handle: int, stream: int) -> None:
        """Replay a captured graph."""
        self._check(self._lib.cuGraphLaunch(ctypes.c_void_p(exec_handle), ctypes.c_void_p(stream)))

    def graph_exec_destroy(self, exec_handle: int) -> None:
        self._check(self._lib.cuGraphExecDestroy(ctypes.c_void_p(exec_handle)))

    def graph_destroy(self, graph: int) -> None:
        self._check(self._lib.cuGraphDestroy(ctypes.c_void_p(graph)))
