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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from quark.device import Device, DeviceFamily, current_device
from quark.ir import DType, Module
from quark.launcher.param_spec import ParamSpec, ProgramFootprint
from quark.lower.ptx import PtxLowerer

if TYPE_CHECKING:
    from quark.autotune import AutotuneCache


def _time_callable(fn, *, warmup_ms: float = 100.0, bench_ms: float = 300.0) -> float:
    """Budget-based timer using CUDA events or MLX sync. Returns μs/call.

    Two-phase design, both phases driven by a wall-clock budget rather
    than a fixed iteration count (which is fragile when per-iter time
    drifts between the probe and the bench — clock ramps, thermal
    throttling, cache warming all bias the probe-then-count approach
    toward the first generation of configs).

    Phase 1 — warmup: launch ``fn()`` in a tight loop until wall-clock
    elapsed ≥ ``warmup_ms``, periodically draining the command queue
    so the host doesn't run ahead and skew what "warmup" means. Sync
    once at the end so the bench starts from a drained pipeline.

    Phase 2 — bench: bracket a wall-clock-bounded loop with CUDA
    events. The events measure actual GPU execution time across every
    launched iter (including queued work that finishes after the host
    exits the loop), so the returned µs/call reflects GPU occupancy,
    not host-side queueing. Sync on the end event before reading the
    elapsed time.

    Sync gates at both ends keep adjacent measurements from bleeding
    into each other — important when the autotuner times dozens of
    configs back-to-back.
    """
    import time as _time

    from quark.device import current_device

    dev = current_device()
    if dev.family is DeviceFamily.METAL:
        # Metal path: drive the timing loop directly over mlx.sync.
        import mlx.core as _mx

        _mx.synchronize()
        t0 = _time.perf_counter()
        n_warm = 0
        while (_time.perf_counter() - t0) * 1000 < warmup_ms:
            fn()
            n_warm += 1
            if n_warm % 32 == 0:
                _mx.synchronize()
        _mx.synchronize()
        t_bench = _time.perf_counter()
        n_bench = 0
        while (_time.perf_counter() - t_bench) * 1000 < bench_ms:
            fn()
            n_bench += 1
        _mx.synchronize()
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

    # Phase 2: bench. CUDA events bracket a wall-clock-bounded loop.
    start = rt.event_create()
    end = rt.event_create()
    rt.event_record(start, 0)
    t_bench = _time.perf_counter()
    n_bench = 0
    while (_time.perf_counter() - t_bench) * 1000 < bench_ms:
        fn()
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
# Backend ↔ IR DType bridge — MLX (Metal) + QuarkTensor (CUDA) only.
# ---------------------------------------------------------------------------


def _mlx_dtype_table() -> dict[Any, DType]:
    """Build the mlx→IR DType map lazily."""
    import mlx.core as mx

    return {
        mx.float32: DType.F32,
        mx.float16: DType.F16,
        mx.bfloat16: DType.BF16,
        mx.int8: DType.S8,
        mx.uint8: DType.U8,
        mx.int32: DType.S32,
        mx.int64: DType.S64,
        mx.uint16: DType.U16,
        mx.uint32: DType.U32,
    }


def _ir_to_mlx_dtype(dtype: DType) -> Any:
    """Map an IR DType to an mlx.core dtype."""
    import mlx.core as mx

    table = {
        DType.F32: mx.float32,
        DType.F16: mx.float16,
        DType.BF16: mx.bfloat16,
        DType.S8: mx.int8,
        DType.U8: mx.uint8,
        DType.S32: mx.int32,
        DType.U32: mx.uint32,
        DType.S64: mx.int64,
    }
    result = table.get(dtype)
    if result is None:
        raise TypeError(f"_ir_to_mlx_dtype: no mlx equivalent for {dtype}")
    return result


def _check_dtype_mlx(buf, expected: DType) -> None:
    """Validate that an mx.array's dtype matches what the kernel expects."""
    table = _mlx_dtype_table()
    actual = table.get(buf.dtype)
    if actual is None:
        raise TypeError(f"_check_dtype_mlx: mlx dtype {buf.dtype} has no quark IR equivalent")
    if actual is not expected:
        raise TypeError(
            f"_check_dtype_mlx: buffer dtype mismatch — kernel expects "
            f"{expected!r}, got array of {actual!r} ({buf.dtype})"
        )


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

    def launch(
        self,
        *,
        buffers: list,
        scalars: tuple = (),
        stream: int | None = None,
    ) -> list | None:
        """Validate, extract pointers, pack scalars, and dispatch.

        On CUDA: writes into preallocated output buffers, returns None.
        On Metal: returns list of mx.array outputs (MLX allocates them).
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
            return self._launch_metal(buffers, scalars)

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

    def _launch_metal(self, buffers, scalars) -> list:
        """Metal launch path — mx.array objects, MLX allocates outputs."""
        import mlx.core as mx

        input_arrays: list = []
        output_shapes: list[tuple[int, ...]] = []
        output_dtypes: list = []

        for buf, pspec in zip(buffers, self.param_spec.buffers, strict=False):
            _check_dtype_mlx(buf, pspec.dtype)
            if pspec.readonly:
                input_arrays.append(buf)
            else:
                # Output buffer — record shape/dtype for MLX to allocate.
                output_shapes.append(tuple(buf.shape))
                output_dtypes.append(buf.dtype)

        # Scalars become 0-d mx.arrays appended to inputs.
        scalar_arrays: list = []
        for val, sspec in zip(scalars, self.param_spec.scalars, strict=False):
            scalar_arrays.append(mx.array(val, dtype=_ir_to_mlx_dtype(sspec.dtype)))

        return self.driver.launch_mlx(
            self.module,
            grid=self.grid_fn(*self.grid_args),
            block=self.block_fn(*self.block_args),
            input_arrays=input_arrays,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            scalar_arrays=scalar_arrays,
        )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def _driver_for(family: DeviceFamily):
    """Return the Driver class that owns this family. Importing the
    module is lazy so non-CUDA processes never load libcuda/MLX."""
    if family is DeviceFamily.CUDA:
        from quark.drivers.cuda import CudaDriver

        return CudaDriver
    if family is DeviceFamily.METAL:
        from quark.drivers.mlx import MlxDriver

        return MlxDriver
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

        CUDA: IR Module -> PtxLowerer -> LoweredKernel
        Metal: IR Module -> MslLowerer -> LoweredMslKernel
        """
        if isinstance(ir_or_program, Module):
            if self.device.family is DeviceFamily.METAL:
                from quark.lower.msl import MslLowerer

                return MslLowerer(self.device.caps).lower_module(ir_or_program)
            cc = self.device.caps.compute_capability
            target_sm = (cc[0] * 10 + cc[1]) if cc is not None else 89
            return PtxLowerer(target_sm=target_sm).lower_module(ir_or_program)
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
