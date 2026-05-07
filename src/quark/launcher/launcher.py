"""Launcher and CompiledKernel — top-level entry point for kernel launches.

EXEMPT FROM 500-LINE RULE — Launcher, CompiledKernel, and the dtype
bridge tables form a single cohesive compile/launch entry point.
Splitting would scatter the CUDA/Metal dispatch logic across files
with no readability benefit.

Per quark launcher proposal §6.1 + §6.2.

A `Launcher` owns one device and one driver. It compiles `Kernel`
instances into `CompiledKernel`s on demand and dispatches launches
through the driver. The runtime currency is **torch tensors** —
`CompiledKernel.launch` accepts torch tensors directly, validates
their dtype and contiguity against the `ParamSpec`, extracts
`data_ptr()`, and hands them to the driver.

Bundle 3 ships the CUDA path. Other backends raise NotImplementedError
in `_driver_for(family)`; later bundles wire them up.

The autotune cache (`AutotuneCache`) is referenced from §6.1 but the
proposal defers its implementation to §9 (Bundle 5). For Bundle 3 we
require an explicit `KernelConfig` — `Launcher.compile(spec, config)`
without a config raises a clear error pointing at the future bundle.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as _np

from quark.device import Device, DeviceFamily, current_device

# Quark short-string dtype → numpy carrier dtype. Used in the hot path
# to read tensor metadata without reaching for ``np.asarray`` on a
# QuarkTensor (which copies device→host).
_IR_DT_TO_NP_DT: dict[str, _np.dtype] = {
    "f32": _np.dtype("float32"),
    "f16": _np.dtype("float16"),
    "bf16": _np.dtype("uint16"),
    "s32": _np.dtype("int32"),
    "s64": _np.dtype("int64"),
    "u8": _np.dtype("uint8"),
    "s8": _np.dtype("int8"),
}

# Resolved once at import time to avoid an os.environ.get() on every
# dispatch (~10k/frame). Re-import won't pick up live env changes,
# which matches the rest of the launcher's startup-only knobs. The
# ``import os as _os; ... ; del _os`` shape is intentional — keeps
# ``os`` out of the module's public namespace. The remaining quark
# imports below the env-var read are also intentional (see ruff
# E402 below).
import os as _os  # noqa: E402

_DISABLE_OUTPUT_HANDLES: bool = _os.environ.get("QUARK_DISABLE_OUTPUT_HANDLES") == "1"
del _os
from quark.ir import DType, Module  # noqa: E402
from quark.launcher.param_spec import ParamSpec, ProgramFootprint  # noqa: E402

# Lazy-eval flag for the Metal path. When True, ``CompiledKernel.launch``
# skips the per-call ``driver.sync()`` so dispatches accumulate in a
# single command-buffer and the GPU pipelines across kernel boundaries.
# Flipped by the ``quark.lazy()`` context manager. Async-safe via
# ContextVar (matches ``max_autotune`` / autotune ``_SEARCH_DEPTH``).
_LAZY: ContextVar[bool] = ContextVar("_LAZY", default=False)

if TYPE_CHECKING:
    from quark.autotune import AutotuneCache


def _time_callable(fn, *, warmup_ms: float = 100.0, bench_ms: float = 300.0) -> float:
    """Budget-based timer using CUDA events or Metal sync. Returns μs/call.

    Two-phase design, both phases driven by a wall-clock budget rather
    than a fixed iteration count (which is fragile when per-iter time
    drifts between the probe and the bench — clock ramps, thermal
    throttling, cache warming all bias the probe-then-count approach
    toward the first generation of configs).

    Phase 1 — warmup: launch ``fn()`` eagerly in a tight loop until
    wall-clock elapsed ≥ ``warmup_ms``, periodically draining the
    command queue so the host doesn't run ahead and skew what "warmup"
    means. Sync once at the end so capture starts drained.

    Phase 2 — bench (CUDA): capture ``fn()`` into a CUDA graph once,
    then replay that graph in a wall-clock-bounded loop bracketed by
    CUDA events. Replaying a captured graph eliminates per-call
    Python + driver-API overhead (cublasLt descriptor create/destroy,
    dispatch bookkeeping, stream-ordered alloc setup) that the eager
    loop over-weighted — and matches the regime production runs in
    (QuarkBackend + user code capture the full forward once, then
    replay per frame). This is what makes the PTX vs cuBLAS autotune
    comparison apples-to-apples: both get measured at their true
    steady-state GPU cost, nothing else.

    Falls back to eager timing on capture failure (e.g. when ``fn()``
    internally host-syncs) so autotune still works for any kernel
    that can't be captured.

    Metal path stays eager — graph capture isn't wired up there yet.
    """
    import time as _time

    from quark.device import current_device

    dev = current_device()
    if dev.family is DeviceFamily.METAL:
        # Metal / PyObjC path: dispatch is synchronous (waitUntilCompleted),
        # so no explicit sync needed between calls.
        t0 = _time.perf_counter()
        n_warm = 0
        while (_time.perf_counter() - t0) * 1000 < warmup_ms:
            fn()
            n_warm += 1
        t_bench = _time.perf_counter()
        n_bench = 0
        while (_time.perf_counter() - t_bench) * 1000 < bench_ms:
            fn()
            n_bench += 1
        elapsed_s = _time.perf_counter() - t_bench
        return (elapsed_s * 1e6) / max(n_bench, 1)

    from quark.runtime.cuda import CudaRuntime

    rt = CudaRuntime.instance()
    rt.stream_synchronize(0)

    # Phase 1: warmup. Time-budget loop, periodic drain so the host
    # doesn't race arbitrarily far ahead of the device.
    t0 = _time.perf_counter()
    n_warm = 0
    while (_time.perf_counter() - t0) * 1000 < warmup_ms:
        fn()
        n_warm += 1
        if n_warm % 32 == 0:
            rt.stream_synchronize(0)
    rt.stream_synchronize(0)

    # Phase 2: capture ``fn()`` once, replay the graph in the bench
    # loop. If capture fails (kernel host-syncs internally, uses an
    # unsupported op, etc.) fall back to eager — autotune needs to
    # keep working even for kernels that can't graph.
    captured = None
    try:
        from quark.graph import capture_graph

        with capture_graph(quiet=True) as _g:
            fn()
        captured = _g
    except Exception:
        rt.stream_synchronize(0)
    replay = captured.replay if captured is not None else fn

    start = rt.event_create()
    end = rt.event_create()
    rt.event_record(start, 0)
    t_bench = _time.perf_counter()
    n_bench = 0
    while (_time.perf_counter() - t_bench) * 1000 < bench_ms:
        replay()
        n_bench += 1
    rt.event_record(end, 0)
    rt.event_synchronize(end)
    total_ms = rt.event_elapsed_time(start, end)
    rt.event_destroy(start)
    rt.event_destroy(end)
    # Drain anything lingering so the caller's next timing starts
    # from a known state.
    rt.stream_synchronize(0)
    return total_ms * 1000 / max(n_bench, 1)  # μs per call


# ---------------------------------------------------------------------------
# Backend ↔ IR DType bridge — Metal (numpy) + QuarkTensor (CUDA).
# ---------------------------------------------------------------------------


_IR_TO_NP: dict[DType, _np.dtype] = {
    DType.F32: _np.dtype("float32"),
    DType.F16: _np.dtype("float16"),
    DType.BF16: _np.dtype("uint16"),  # storage type
    DType.S8: _np.dtype("int8"),
    DType.U8: _np.dtype("uint8"),
    DType.S16: _np.dtype("int16"),
    DType.U16: _np.dtype("uint16"),
    DType.S32: _np.dtype("int32"),
    DType.U32: _np.dtype("uint32"),
    DType.S64: _np.dtype("int64"),
}


def _ir_to_np_dtype(dtype: DType) -> _np.dtype:
    """Map an IR DType to a NumPy dtype."""
    result = _IR_TO_NP.get(dtype)
    if result is None:
        raise TypeError(f"_ir_to_np_dtype: no numpy equivalent for {dtype}")
    return result


_CUDA_TENSOR_DTYPE_MAP: dict[str, DType] = {
    "bf16": DType.BF16,
    "f16": DType.F16,
    "f32": DType.F32,
    "s32": DType.S32,
    "s8": DType.S8,
    "u8": DType.U8,
    "e4m3": DType.E4M3,
    "e5m2": DType.E5M2,
    "u16": DType.U16,
    "u32": DType.U32,
    "u64": DType.U64,
    "s16": DType.S16,
    "s64": DType.S64,
}


def _check_dtype_quark(buf, expected: DType) -> None:
    """Validate that a QuarkTensor's dtype matches the kernel expectation."""
    actual = _CUDA_TENSOR_DTYPE_MAP.get(buf.dtype)
    if actual is None:
        raise TypeError(f"_check_dtype_quark: QuarkTensor dtype '{buf.dtype}' has no IR equivalent")
    if actual is not expected:
        raise TypeError(
            f"_check_dtype_quark: buffer dtype mismatch — kernel expects "
            f"{expected!r}, got QuarkTensor of {actual!r} ('{buf.dtype}')"
        )


def _check_contiguous(buf, spec) -> None:
    """Default contiguity check. QuarkTensors are always contiguous."""
    if hasattr(buf, "is_contiguous") and not buf.is_contiguous():
        raise ValueError(f"_check_contiguous: buffer {spec.name!r} must be contiguous")


# ---------------------------------------------------------------------------
# CompiledKernel
# ---------------------------------------------------------------------------


@dataclass
class CompiledKernel:
    """A compiled kernel ready to launch.

    Holds the driver-specific compiled module + the typed parameter
    spec + grid/block functions captured from the source Kernel. The
    `launch()` method validates inputs against the spec and dispatches
    through the driver.

    Bundle 3 supports CUDA only — `buffers` must be on a torch CUDA
    device. The proposal also lists `mps` as valid; that's wired up
    when Bundle 5 (Metal) lands.
    """

    driver: Any  # Driver protocol — typed Any to avoid circular import
    module: Any  # CudaCompiledModule (or other backend equivalent)
    entry: str
    grid_fn: Callable[..., tuple[int, int, int]]
    block_fn: Callable[..., tuple[int, int, int]]
    param_spec: ParamSpec
    footprint: ProgramFootprint
    # The source Kernel instance (same (spec, config) we compiled from).
    # Held so the functional layer can call hooks like
    # ``prepare_launch_tensors`` without reconstructing a fresh Kernel
    # + re-running ``emit()`` on every inference call. Typed Any to
    # avoid a circular import back to quark.kernels.base.
    kernel: Any = None
    # Captured at compile time so launch() can call grid_fn / block_fn
    # without re-passing the spec/config.
    grid_args: tuple = ()
    block_args: tuple = ()
    # Cache of input-role dummy tensors (e.g. has_bias=False Bias, zero
    # Y for unary ops). Populated lazily by the functional dispatch
    # layer on first call.
    _input_dummies: dict = field(default_factory=dict)
    # Lazily-populated dispatch packet for the Metal hot path — see
    # ``_launch_metal``. Holds invariants derived from the spec
    # (output shapes/dtypes, nbytes, default handle lists) so the
    # per-call loop only walks inputs to extract their ``metal_handle``.
    _metal_packet: Any = None

    def launch(
        self,
        *,
        buffers: list,
        scalars: tuple = (),
        stream: int | None = None,
        persistent_outs: set | None = None,
    ) -> list | None:
        """Validate, extract pointers, pack scalars, and dispatch.

        On CUDA: writes into preallocated output buffers, returns None.
        On Metal: returns list of numpy array outputs (driver allocates them).
        """
        if len(buffers) != len(self.param_spec.buffers):
            raise ValueError(
                f"CompiledKernel.launch: expected "
                f"{len(self.param_spec.buffers)} buffers, "
                f"got {len(buffers)}"
            )

        # Detect Metal path via driver type.
        from quark.device import DeviceFamily

        if hasattr(self.driver, "family") and self.driver.family is DeviceFamily.METAL:
            return self._launch_metal(buffers, scalars, persistent_outs=persistent_outs)
        if hasattr(self.driver, "family") and self.driver.family is DeviceFamily.INTEL_GPU:
            return self._launch_spv(buffers, scalars)

        return self._launch_cuda(buffers, scalars, stream)

    def _launch_cuda(self, buffers, scalars, stream) -> None:
        """CUDA launch path — pointer-based, writes in place.

        Every buffer must be a ``QuarkTensor`` (the runtime inference
        path; torch was retired with the numpy-refs migration).
        """
        from quark.runtime.tensor import QuarkTensor

        ptrs: list[int] = []
        for i, (buf, pspec) in enumerate(zip(buffers, self.param_spec.buffers, strict=False)):
            if not isinstance(buf, QuarkTensor):
                raise TypeError(
                    f"CompiledKernel.launch: buffer {i} ({pspec.name!r}) "
                    f"is {type(buf).__name__}, expected QuarkTensor"
                )
            _check_dtype_quark(buf, pspec.dtype)
            _check_contiguous(buf, pspec)
            ptrs.append(buf.data_ptr())

        scalar_bytes = self.param_spec.pack_scalars(tuple(scalars))

        if stream is None:
            from quark.graph import active_stream

            stream_handle = active_stream()  # 0 normally, capture stream during graph capture
        elif isinstance(stream, int):
            stream_handle = stream
        else:
            raise TypeError(
                f"CompiledKernel.launch: stream must be int or None, got {type(stream).__name__}"
            )

        self.driver.launch(
            self.module,
            grid=self.grid_fn(*self.grid_args),
            block=self.block_fn(*self.block_args),
            buffer_ptrs=ptrs,
            scalar_args=scalar_bytes,
            stream=stream_handle,
        )
        return None

    def _launch_metal(self, buffers, scalars, persistent_outs=None) -> list:
        """Metal launch — NumPy arrays / QuarkTensors in, driver
        allocates outputs.

        Inputs that are ``QuarkTensor`` with a Metal pool ``metal_handle``
        go zero-copy via the handle path; everything else goes through
        the input cache (memcpy into a fresh Metal buffer).
        """
        import numpy as np

        # Hot-path packet: precomputed per-CompiledKernel invariants.
        # On first call we walk ``param_spec.buffers``+``buffers`` once
        # to derive output shapes/dtypes/nbytes (fixed by spec) and the
        # ordered lists of readonly-input vs output buffer indices.
        # Subsequent calls reuse these and only re-extract per-buffer
        # ``metal_handle`` for inputs — no shape/dtype branching.
        packet = self._metal_packet
        if packet is None:
            _in_idx: list[int] = []
            _in_names: list[str] = []
            _out_idx: list[int] = []
            _out_names: list[str] = []
            _out_shapes: list[tuple[int, ...]] = []
            _out_dtypes: list = []
            for i, b in enumerate(self.param_spec.buffers):
                if b.readonly:
                    _in_idx.append(i)
                    _in_names.append(b.name)
                else:
                    _out_idx.append(i)
                    _out_names.append(b.name)
                    buf = buffers[i]
                    shape = tuple(int(s) for s in buf.shape)
                    if isinstance(buf.dtype, str):
                        dt = _IR_DT_TO_NP_DT.get(buf.dtype)
                        if dt is None:
                            dt = np.dtype(buf.dtype)
                    else:
                        dt = buf.dtype
                    _out_shapes.append(shape)
                    _out_dtypes.append(dt)
            packet = (
                tuple(_in_idx),
                tuple(_out_idx),
                tuple(_out_names),
                tuple(_out_shapes),
                tuple(_out_dtypes),
            )
            self._metal_packet = packet
        in_idx, out_idx, out_names, output_shapes, output_dtypes = packet
        output_shapes = list(output_shapes)
        output_dtypes = list(output_dtypes)

        _disable_oh = _DISABLE_OUTPUT_HANDLES

        # Inputs: walk readonly indices in a tight loop.
        n_in = len(in_idx)
        input_arrays = [None] * n_in
        input_handles = [-1] * n_in
        for k in range(n_in):
            buf = buffers[in_idx[k]]
            h = getattr(buf, "metal_handle", None)
            if h is not None:
                input_arrays[k] = buf  # placeholder; ndarray view built in driver
                input_handles[k] = int(h)
            else:
                input_arrays[k] = np.asarray(buf)
                # input_handles[k] already -1
        # Outputs: default -1 (pool-alloc), or route to caller-pinned
        # handles. Two trigger paths:
        #   1. ``persistent_outs={...}``: explicit list from the
        #      ``call_with_bindings`` caller, used for state buffers
        #      like KV caches that must survive eval recycling.
        #   2. The caller passed a QuarkTensor with a ``metal_handle``
        #      as the output slot — common for layers that pre-allocate
        #      a cached ``out=`` buffer (``nn.Add``, ``nn.SiLU``,
        #      ``nn.EulerStep``, the ``_maybe_cached_out`` family) and
        #      hand it straight to ``compiled.launch(buffers=[...])``
        #      without going through ``call_with_bindings``. Without
        #      this, the kernel writes into a fresh pool buffer and
        #      the caller's pre-allocated tensor stays at its initial
        #      (typically zero-filled) state — the
        #      ``ctrl_residual``/``mlp_gate`` chain silently produced
        #      zeros for every block that touched a cached-out path.
        n_out = len(out_idx)
        output_handles = [-1] * n_out
        if not _disable_oh:
            for k in range(n_out):
                buf = buffers[out_idx[k]]
                h = getattr(buf, "metal_handle", None)
                if h is None:
                    continue
                if persistent_outs is not None and out_names[k] in persistent_outs:
                    output_handles[k] = int(h)
                else:
                    # Auto-route any caller-provided pinned output buffer
                    # so the kernel writes land where the caller expects.
                    output_handles[k] = int(h)

        scalar_arrays: list = []
        for val, sspec in zip(scalars, self.param_spec.scalars, strict=False):
            scalar_arrays.append(np.array(val, dtype=_ir_to_np_dtype(sspec.dtype)))

        results = self.driver.launch_metal(
            self.module,
            grid=self.grid_fn(*self.grid_args),
            block=self.block_fn(*self.block_args),
            input_arrays=input_arrays,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            scalar_arrays=scalar_arrays,
            input_handles=input_handles,
            output_handles=output_handles,
        )
        if not _LAZY.get():
            self.driver.sync(None)
        return results

    def _launch_spv(self, buffers: list, scalars: tuple) -> None:
        """SPIR-V / Vulkan launch path.

        Per-buffer policy:
          * ``int`` — opaque buffer handle previously returned by
            ``SpvDriver.allocate_buffer``. Passed through as-is.
          * ``(handle, mapped_ptr)`` tuple — same shape
            ``allocate_buffer`` returns; the handle is the first
            element, mapped_ptr is the host-visible mapping the
            caller writes / reads through (already used by the
            test harness; the launcher just consumes the handle).
          * ``QuarkTensor`` — TODO. The pool of Vulkan-backed
            QuarkTensor allocations isn't wired today; once it is,
            zero-copy via ``buffer_handle`` mirrors the Metal
            ``metal_handle`` path. For v1, callers route through
            ``SpvDriver.allocate_buffer`` directly.
          * ``numpy.ndarray`` — copied into a fresh Vulkan buffer
            allocated on the fly and the handle is passed. Cheaper
            for tests; production code should reuse pinned buffers.

        Scalars are packed into push constants via
        ``param_spec.pack_scalars``. The compiled module's
        ``push_size`` validation in ``SpvDriver.launch`` catches
        size mismatches.
        """
        import numpy as np
        from quark.drivers import _spv_dispatch as _sd

        handles: list[int] = []
        for buf, pspec in zip(buffers, self.param_spec.buffers, strict=False):
            del pspec  # only used by the dtype/contiguity checks; SpvDriver
                       # validates via the shader module's binding count.
            if isinstance(buf, int):
                handles.append(buf)
            elif isinstance(buf, tuple) and len(buf) == 2 and isinstance(buf[0], int):
                handles.append(buf[0])
            elif isinstance(buf, np.ndarray):
                arr = np.ascontiguousarray(buf)
                h, ptr = _sd.allocate_buffer(arr.nbytes)
                import ctypes
                ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)
                handles.append(h)
            else:
                raise TypeError(
                    f"_launch_spv: buffer is {type(buf).__name__}; expected "
                    "int handle / (handle, ptr) / numpy.ndarray. QuarkTensor "
                    "support pending pool integration."
                )

        # ``pack_scalars`` returns one bytes object per scalar param.
        # SPIR-V push-constants want a single contiguous blob — flatten.
        scalar_blobs = self.param_spec.pack_scalars(tuple(scalars))
        push_bytes = b"".join(scalar_blobs)

        # The driver's launch signature wants the SpvCompiledModule
        # dataclass (with the pipeline handle + n_buffers + push
        # validation), the workgroup grid, the buffer-handle list,
        # and the push-constant bytes.
        self.driver.launch(
            self.module,
            grid=self.grid_fn(*self.grid_args),
            buffer_handles=handles,
            push_bytes=push_bytes,
        )
        return None

    def launch_metal_fast(
        self,
        input_arrays: list,
        output_shapes: list[tuple[int, ...]],
        output_np_dtypes: list,
    ) -> list:
        """Metal fast path: skip buffer-list splitting and scalar packing.

        Caller provides input arrays and precomputed output shapes/dtypes
        directly. No dict construction, no alloc_from_decl, no
        prepare_launch_tensors. Used by functional wrappers for the
        repeat-call hot path where the launch setup is invariant.
        """
        results = self.driver.launch_metal(
            self.module,
            grid=self.grid_fn(*self.grid_args),
            block=self.block_fn(*self.block_args),
            input_arrays=input_arrays,
            output_shapes=output_shapes,
            output_dtypes=output_np_dtypes,
            scalar_arrays=[],
        )
        if not _LAZY.get():
            self.driver.sync(None)
        return results


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def _driver_for(family: DeviceFamily):
    """Return the Driver class that owns this family. Importing the
    module is lazy so non-CUDA processes never load libcuda."""
    if family is DeviceFamily.CUDA:
        from quark.drivers.cuda import CudaDriver

        return CudaDriver
    if family is DeviceFamily.METAL:
        from quark.drivers.metal import MetalDriver

        return MetalDriver
    if family is DeviceFamily.INTEL_GPU:
        from quark.drivers.spv import SpvDriver

        # SpvDriver doesn't take a ``device=`` kwarg the way Cuda /
        # MetalDriver do — it picks the default Vulkan device via
        # ``pick_default_device`` unless ``device_index`` is passed.
        # Adapt with a small wrapper class that swallows the
        # ``device=`` kwarg the launcher passes through.
        class _SpvDriverAdapter(SpvDriver):
            def __init__(self, *, device=None):
                # ``device`` is the quark Device; we don't need it for
                # SpvDriver (it picks the Vulkan device from the
                # available ICDs). Stash for parity.
                super().__init__()
                self._quark_device = device
                self.family = DeviceFamily.INTEL_GPU

        return _SpvDriverAdapter
    raise NotImplementedError(f"_driver_for: backend {family.value!r} not yet implemented")


class Launcher:
    """Top-level entry point for kernel compilation and launch.

    One Launcher per device per process. Holds the device + driver +
    a kernel cache keyed on (kernel_cls, spec, config). The compile
    pipeline is:

      kernel.emit() → IR Module
                   → PtxLowerer.lower_module(module)
                   → LoweredKernel(ptx, smem_bytes, kernel_name)
                   → driver.compile(ptx, entry, smem_bytes)
                   → CudaCompiledModule
                   → CompiledKernel(driver, module, …)
    """

    def __init__(
        self,
        device: Device | None = None,
        autotune_cache: AutotuneCache | None = None,
    ) -> None:
        if device is None:
            device = current_device()
        self.device = device
        self.driver = _driver_for(device.family)(device=device)
        self._kernel_cache: dict[tuple, CompiledKernel] = {}
        # Lazy-construct the autotune cache so the import doesn't pull
        # in quark.autotune for callers that always pass explicit
        # configs (the cache constructor only does cheap path resolution
        # so this is mostly stylistic).
        if autotune_cache is None:
            from quark.autotune import AutotuneCache

            autotune_cache = AutotuneCache(device=device)
        self._autotune = autotune_cache
        # Inject the compile-and-time hook so the cache's _search can
        # actually call back into us. The hook closes over `self` so
        # the launcher remains the only path that touches the driver.
        self._autotune._compile_and_time = self._compile_and_time_for_autotune  # type: ignore[attr-defined]
        self._autotune._launcher = self  # type: ignore[attr-defined]

    def compile(
        self,
        kernel_cls: type,
        spec,
        config=None,
    ) -> CompiledKernel:
        """Build, lower, and compile a kernel for this device.

        `kernel_cls` is the Kernel subclass; `spec` and `config` are
        its KernelSpec / KernelConfig. The result is cached on
        (kernel_cls, spec, config) so repeat compiles are free.

        When ``config`` is None, the launcher consults its
        ``AutotuneCache``: hot dict → on-disk JSON →
        ``configs/`` bundled defaults → bounded online search. Set
        ``QUARK_DISABLE_AUTOTUNE=1`` to skip the search and fall
        back to the kernel's ``_pick_default_cfg``.
        """
        if config is None:
            config = self._autotune.lookup_or_search(kernel_cls, spec)

        key = (kernel_cls, spec, config)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        kernel = kernel_cls(spec, config)
        if not kernel.is_valid_for(self.device.caps):
            raise ValueError(
                f"{kernel_cls.__name__}: config {config} invalid for "
                f"device {self.device.caps.name} (caps "
                f"max_smem_per_block={self.device.caps.max_smem_per_block})"
            )

        # Lower kernel → IR Module → backend-specific artifact.
        ir_or_program = kernel.emit()
        lowered = self._lower(ir_or_program, kernel)

        # Dispatch compile to the appropriate driver.
        if self.device.family is DeviceFamily.METAL:
            compiled_mod = self.driver.compile(
                lowered=lowered,
                entry_name=lowered.kernel_name or kernel.entry_name(),
                smem_bytes=lowered.smem_bytes,
            )
        elif self.device.family is DeviceFamily.INTEL_GPU:
            # SPIR-V / Vulkan path. The lowerer emits SPIR-V text;
            # ``text_to_binary`` shells out to ``spirv-as`` to get
            # the binary blob the driver consumes. ``n_buffers`` /
            # ``push_constants_size`` come from the lowered kernel
            # metadata so SpvDriver.compile can validate at descriptor-
            # set creation. See PORTABILITY_PLAN §3.1 / §3.2.
            from quark.lower.spv import text_to_binary
            spirv_binary = text_to_binary(lowered.source)
            compiled_mod = self.driver.compile(
                source=spirv_binary,
                entry=lowered.entry_name,
                n_buffers=lowered.n_buffers,
                push_constants_size=lowered.push_constants_size,
                smem_bytes=lowered.smem_bytes,
            )
        else:
            try:
                compiled_mod = self.driver.compile(
                    source=lowered.ptx,
                    entry_name=lowered.kernel_name or kernel.entry_name(),
                    smem_bytes=lowered.smem_bytes,
                )
            except Exception as exc:
                from quark.runtime.cuda import CudaError
                from quark.utils.ptx_dump import classify_compile_error, dump_path_for

                if classify_compile_error(exc) == "ptx":
                    path = dump_path_for(kernel_cls, spec, config)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(lowered.ptx)
                    # ``CudaError(code, name, message)`` doesn't fit the
                    # generic ``type(exc)(msg)`` pattern — the three-arg
                    # ctor raises TypeError when passed a single formatted
                    # string, eating the real driver code (218 INVALID_PTX,
                    # 209 NO_BINARY_FOR_GPU, etc.) so autotune's [cerr]
                    # logs just show the reconstruction TypeError instead.
                    # Preserve the original code/name and fold the PTX
                    # path into the message.
                    if isinstance(exc, CudaError):
                        raise CudaError(
                            exc.code,
                            exc.name,
                            f"{exc.message}\n\nPTX dumped to: {path}",
                        ) from exc
                    raise type(exc)(f"{exc}\n\nPTX dumped to: {path}") from exc
                raise

        ck = CompiledKernel(
            driver=self.driver,
            module=compiled_mod,
            entry=lowered.kernel_name or kernel.entry_name(),
            grid_fn=kernel.grid,
            block_fn=kernel.block,
            param_spec=self._param_spec_for(ir_or_program),
            footprint=ProgramFootprint(smem_bytes=lowered.smem_bytes),
            kernel=kernel,
        )
        self._kernel_cache[key] = ck
        return ck

    def _lower(self, ir_or_program, kernel):
        """Lower the kernel's emit() output to a backend-specific artifact.

        Dispatches through the family-keyed ``LOWERERS`` registry (see
        ``quark.lower.base``). Each backend registers a factory that
        maps ``DeviceCaps`` to a lowerer instance, so adding a new
        backend is a one-line ``@register_lowerer`` call — no edits
        here.

        CUDA: IR Module -> PtxLowerer -> LoweredKernel
        Metal: IR Module -> MslLowerer -> LoweredMslKernel
        """
        if isinstance(ir_or_program, Module):
            from quark.lower import get_lowerer
            from quark.lower.legalize import legalize

            # Legalization: expand ops the backend can't emit natively
            # into equivalent IR it can. No-op when no rewrites are
            # registered (phase-2.1 default). Rewrites land in phase 2.2.
            legalize(ir_or_program, self.device.caps)
            lowerer = get_lowerer(self.device.family, self.device.caps)
            return lowerer.lower_module(ir_or_program)
        raise NotImplementedError(
            "Launcher._lower: legacy Program-based emit() not yet "
            "supported by the driver path. Migrate the kernel to "
            "return an `quark.ir.Module` from emit()."
        )

    def _param_spec_for(self, ir_or_program) -> ParamSpec:
        if isinstance(ir_or_program, Module):
            if not ir_or_program.functions:
                raise ValueError("Launcher._param_spec_for: module has no functions")
            return ParamSpec.from_function(ir_or_program.functions[0])
        raise NotImplementedError("Launcher._param_spec_for: only Module is supported")

    def _compile_and_time_for_autotune(self, kernel_cls, spec, config, *, tensors=None) -> float:
        """Hook the AutotuneCache calls during its bounded search.

        Compiles a single (kernel, spec, config) combo and times one
        launch. When ``tensors`` is provided (the AutotuneCache builds
        a single dict up front and threads it through every candidate),
        we reuse it — matching ``_search_full``'s one-alloc-per-search
        pattern. Per-call ``make_tensors`` was churning the CUDA caching
        allocator across candidates and observed to corrupt subsequent
        launches on sm_120. Falls back to ``make_tensors(problems()[0])``
        only when the cache can't provide tensors (legacy callers).
        Returns the runtime in microseconds.
        """
        kernel = kernel_cls(spec, config)
        if not _kernel_is_valid(kernel, self.device.caps):
            raise ValueError("config invalid for device")

        # Alt-impl configs (e.g. cuBLAS) bypass PTX compile entirely.
        # The kernel class provides a ``time_alt_config`` hook that
        # materializes whatever buffers/runner the alt path needs and
        # returns μs/call under the same timing loop.
        impl = getattr(config, "impl", "ptx")
        if impl != "ptx":
            time_alt = getattr(kernel_cls, "time_alt_config", None)
            if not callable(time_alt):
                raise ValueError(
                    f"config.impl={impl!r} but {kernel_cls.__name__} has no time_alt_config hook"
                )
            return time_alt(self, kernel, tensors=tensors)

        ck = self.compile(kernel_cls, spec, config)

        if tensors is None:
            from quark.refs import ref_cache
            from quark.runtime.device_tensors import numpy_to_device_dict

            problems_fn = getattr(kernel_cls, "problems", None)
            if not callable(problems_fn):
                raise NotImplementedError(
                    "_compile_and_time_for_autotune: kernel needs problems() for bounded search"
                )
            problems = problems_fn()
            if not problems:
                raise ValueError("_compile_and_time_for_autotune: kernel.problems() is empty")
            inputs_np, _outputs_np = ref_cache().get(kernel_cls, problems[0].params)
            tensors = numpy_to_device_dict(kernel_cls, kernel.spec, inputs_np)

        # Route through the kernel's launch-time transform (e.g. GEMM's
        # b_shuffle pre-shuffle) so b_shuffle=True configs receive the
        # layout they expect.
        launch_tensors = kernel.prepare_launch_tensors(tensors)

        # Order buffers by ParamSpec (authoritative IR declaration order).
        buffers = [launch_tensors[b.name] for b in ck.param_spec.buffers]

        for _ in range(self._autotune.warmup):
            ck.launch(buffers=buffers)
        return _time_callable(lambda: ck.launch(buffers=buffers), warmup_ms=10.0, bench_ms=50.0)


def _kernel_is_valid(kernel, caps) -> bool:
    is_valid_for = getattr(kernel, "is_valid_for", None)
    if callable(is_valid_for):
        try:
            return bool(is_valid_for(caps))
        except (TypeError, NotImplementedError):
            pass
    legacy = getattr(kernel, "is_valid", None)
    if callable(legacy):
        try:
            return bool(legacy())
        except (TypeError, NotImplementedError):
            pass
    return True
