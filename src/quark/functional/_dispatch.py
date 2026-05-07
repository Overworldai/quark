"""Shared plumbing for ``quark.functional.*``: Launcher singleton,
output allocation from ``TensorDecl``, and a thin ``launch_kernel``
helper that hides the CUDA (in-place) vs Metal (returned-list) split.

Per-kernel wrappers (``functional/gemm.py`` etc.) own their positional-
args ↔ TENSORS binding; this module only supplies the utilities they
share.

EXEMPT FROM 500-LINE RULE: this module owns the full call_with_bindings
pipeline (autotune lookup → buffer prep → backend dispatch → output
unwrap) plus the queue_launch_ir fast path and the strided-copy
helper. Splitting them would mean exporting private state (``_LAZY``,
``_QLI_DERIVED_CACHE``, ``_copy_strided_*``) across module boundaries.

WARMUP NOTE: ``call_with_bindings`` issues one throwaway kernel launch
+ synchronize before the real launch on every invocation. This
workarounds an unidentified bug where the first launch of a kernel
in a "fresh" GPU session produces wrong output (NaN / cos≪1) on
CUDA sm_120. Cost: ~2× kernel runtime per call. See
``CompiledKernel.launch`` for the matching per-CompiledKernel warmup
that fixes the same problem on the autotune side.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from quark.device import current_device
from quark.launcher import Launcher

IS_METAL = sys.platform == "darwin"

# Module-level cached Launcher. ``current_device()`` is itself cached
# inside ``quark.device``, so the first call fixes the singleton for
# the process. Re-autotune / reset flows can clear this via
# ``reset_launcher()`` below.
_LAUNCHER: Launcher | None = None
_LAUNCHER_LOCK = threading.Lock()


def launcher() -> Launcher:
    """Return the process-wide Launcher; create on first call.

    Double-checked locking keeps the fast path (``_LAUNCHER is not None``)
    lock-free while preventing two concurrent first-callers from both
    constructing a Launcher.
    """
    global _LAUNCHER
    if _LAUNCHER is not None:
        return _LAUNCHER
    with _LAUNCHER_LOCK:
        if _LAUNCHER is None:
            _LAUNCHER = Launcher(device=current_device())
    return _LAUNCHER


def reset_launcher() -> None:
    """Drop the cached Launcher. Tests / device changes only."""
    global _LAUNCHER
    _LAUNCHER = None


def split_provided_io(
    inputs: dict[str, Any], outputs: dict[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build the ``(provided, auto_alloc)`` pair that ``call_with_bindings``
    expects.

    ``inputs`` are always provided (kernel inputs). ``outputs`` is a
    name→optional-tensor map: entries with a non-None value are pinned
    output buffers (caller wants them written in place); ``None``
    entries are added to ``auto_alloc`` so the dispatch layer creates
    fresh buffers.

    Lets per-kernel functional wrappers stop hand-rolling the same
    "loop over (name, buf), branch on None" boilerplate.
    """
    provided: dict[str, Any] = dict(inputs)
    auto_alloc: tuple[str, ...] = ()
    for name, buf in outputs.items():
        if buf is not None:
            provided[name] = buf
        else:
            auto_alloc = auto_alloc + (name,)
    return provided, auto_alloc


def alloc_from_decl(decl, spec, config, *, like=None):
    """Allocate a fresh tensor sized to ``decl.shape(spec, config)`` in
    ``decl.dtype(spec, config)``.

    Returns a ``QuarkTensor`` on both backends. On Metal, role="out"
    placeholders use numpy zeros (cheap, just for shape/dtype detection
    in ``_launch_metal``); the real output buffer is allocated by
    ``_md.launch`` and surfaced as a ``QuarkTensor``. Input-role tensors
    must be QuarkTensor so they can go zero-copy through the launch.
    """
    shape = decl.shape(spec, config) if callable(decl.shape) else decl.shape
    dtype_val = decl.dtype(spec, config) if callable(decl.dtype) else decl.dtype
    from quark.ir import DType as _DT

    dt_str = dtype_val.value if isinstance(dtype_val, _DT) else str(dtype_val)

    is_output = getattr(decl, "role", "in") == "out"
    if IS_METAL and is_output:
        import numpy as np

        _NP = {
            "bf16": np.uint16,
            "f16": np.float16,
            "f32": np.float32,
            "s32": np.int32,
            "s64": np.int64,
            "u8": np.uint8,
            "s8": np.int8,
            "u16": np.uint16,
            "u32": np.uint32,
            "e4m3": np.uint8,
            "e5m2": np.uint8,
        }
        np_dt = _NP.get(dt_str)
        if np_dt is None:
            raise TypeError(f"alloc_from_decl: dtype {dt_str!r} has no numpy carrier dtype")
        return np.zeros(shape, dtype=np_dt)

    from quark.runtime.tensor import QuarkTensor

    _STORAGE = {
        "b16": "u16",  # bit-typed 16-bit → u16 storage
        "b32": "u32",
    }
    storage = _STORAGE.get(dt_str, dt_str)
    return QuarkTensor.zeros(*shape, dtype=storage)


def launch_kernel(
    kernel_cls,
    spec,
    config,
    *,
    buffers: list,
    scalars: tuple = (),
) -> list | None:
    """Compile + launch; return Metal's output list or None on CUDA.

    The caller is responsible for ordering ``buffers`` to match
    ``kernel.param_spec().buffers``. For kernels that pre-launch-
    transform weights (e.g. GEMM's ``prepare_launch_tensors`` for
    ``b_shuffle``), the caller must call that separately before
    building the buffer list — the functional surface prefers
    explicit, trace-stable inputs over hidden caches.
    """
    compiled = launcher().compile(kernel_cls, spec, config)
    return compiled.launch(buffers=buffers, scalars=scalars)


def call_with_bindings(
    kernel_cls,
    spec,
    config=None,
    *,
    provided: dict,
    auto_alloc: tuple[str, ...] = (),
    like=None,
) -> dict:
    """Assemble the buffers list from a ``{tensor_name: tensor}`` dict
    and launch, returning the full ``{tensor_name: tensor}`` post-launch.

    ``provided`` must contain every kernel buffer *except* those listed
    in ``auto_alloc`` — those are allocated fresh from the kernel's
    ``TENSORS`` manifest.  On Metal the returned dict's auto-allocated
    entries are the fresh ``np.ndarray`` Metal emitted; on CUDA they are
    the pre-allocated ``QuarkTensor`` buffers (populated in place).

    ``like`` is currently unused on CUDA (QuarkTensor picks its own
    device) and ignored on Metal; kept for API stability.

    When ``config`` is ``None`` (the default), the config is resolved
    from the process-wide ``AutotuneCache``: hot dict → on-disk JSON →
    bundled defaults → inline search (fast by default; full when
    ``QUARK_MAX_AUTOTUNE=1`` or inside ``with quark.max_autotune()``).

    Per-invocation warmup: on CUDA, every call issues a throwaway
    kernel launch + sync before the real launch. Without this, the
    first launch in a "fresh" session (i.e. after some other CUDA
    activity / quiescence) produces wrong output on sm_120. Pay 2×
    kernel runtime per invocation. Once the root cause is found this
    can be removed.
    """
    lc = launcher()
    if config is None:
        config = lc._autotune.lookup_or_search(kernel_cls, spec)

    # Alt-impl configs (e.g. cuBLAS) bypass the compile+launch pipeline
    # entirely. The kernel class owns the dispatch via a
    # ``dispatch_alt_config`` hook that fills auto-alloc buffers, runs
    # the alt path, and returns the full {name: tensor} dict the caller
    # expects.
    impl = getattr(config, "impl", "ptx")
    if impl != "ptx":
        dispatch_alt = getattr(kernel_cls, "dispatch_alt_config", None)
        if not callable(dispatch_alt):
            raise ValueError(
                f"config.impl={impl!r} but {kernel_cls.__name__} has no dispatch_alt_config hook"
            )
        return dispatch_alt(
            spec,
            config,
            provided=provided,
            auto_alloc=auto_alloc,
            like=like,
        )

    # Compile is cached on (cls, spec, config); on the hot path this
    # is a dict lookup and returns a ``CompiledKernel`` that already
    # holds the ParamSpec + source Kernel instance. Using those avoids
    # re-running ``kernel.emit()`` (full IR build, hundreds of μs)
    # on every call.
    compiled = lc.compile(kernel_cls, spec, config)
    kernel = compiled.kernel
    pspec = compiled.param_spec

    # Allocate auto-alloc tensors. We split on ``TensorDecl.role``:
    #
    #   * ``role == "out"``: fresh allocation per call. Caching output
    #     buffers silently aliased prior calls when the same spec
    #     repeated (e.g. ada_gate_residual called twice per block, each
    #     consumer expecting its own output). Callers that want a
    #     persistent output pass ``out=`` (Linear does this).
    #
    #   * Input-role dummies (``has_bias=False`` Bias, elementwise Y
    #     for unary ops): cached on the CompiledKernel. The kernel
    #     never writes them — same zeros every call — so reusing the
    #     same storage is safe AND essential: otherwise every small
    #     no-bias GEMM eats a ``cuMemAllocAsync`` + ``cuMemsetD8Async``
    #     per call (~100–500 µs of driver overhead on the default
    #     stream) that autotune's tight-loop timing never sees.
    # ``decls_by_name`` is class-level (TENSORS is a ClassVar), so cache
    # the dict on the kernel class instead of rebuilding the comprehension
    # every call.
    decls_by_name = getattr(kernel_cls, "_decls_by_name_cache", None)
    if decls_by_name is None:
        decls_by_name = {d.name: d for d in kernel_cls.TENSORS}
        kernel_cls._decls_by_name_cache = decls_by_name
    if like is None:
        like = next(iter(provided.values()))
    full: dict = dict(provided)
    input_dummies: dict = compiled._input_dummies
    for name in auto_alloc:
        decl = decls_by_name[name]
        is_output = getattr(decl, "role", "in") == "out"
        if is_output:
            t = alloc_from_decl(decl, spec, config, like=like)
        else:
            t = input_dummies.get(name)
            if t is None:
                t = alloc_from_decl(decl, spec, config, like=like)
                input_dummies[name] = t
        full[name] = t

    full = kernel.prepare_launch_tensors(full)

    from quark.graph import active_stream

    capturing = bool(active_stream())

    if capturing:
        from quark.graph import log_capture_launch

        log_capture_launch(f"{kernel_cls.__name__}({spec})")

    # Pass caller's buffers directly. During graph capture, the
    # caller's tensors (weight Parameters, KV caches, auto-alloc
    # outputs) already have stable addresses — they don't get
    # reallocated between calls. No stable-buffer indirection needed.
    buffers = [full[b.name] for b in pspec.buffers]
    # ``persistent_outs``: names of role="out" tensors that the caller
    # explicitly provided (rather than letting auto-alloc allocate a
    # fresh buffer). On Metal, the launcher routes these straight into
    # the caller's pinned QuarkTensor so writes survive the eval-queue
    # pool recycle — necessary for state buffers like the KV cache.
    #
    # Hot-path optimization: only walk the buffer list when *any*
    # role="out" tensor is in ``provided``. The vast majority of
    # kernels auto-alloc all their outputs (no overlap with provided),
    # so the cheap pre-check avoids the set comprehension entirely.
    persistent_outs = None
    for b in pspec.buffers:
        if (not b.readonly) and (b.name in provided):
            persistent_outs = {
                bb.name for bb in pspec.buffers if not bb.readonly and bb.name in provided
            }
            break
    result = compiled.launch(buffers=buffers, persistent_outs=persistent_outs)

    if IS_METAL:
        # Metal returns one (handle, ptr, nbytes) tuple per non-readonly
        # buffer in pspec order. Wrap each as a QuarkTensor backed by
        # the lazy_alloc_output handle so downstream ops can use it
        # zero-copy through metal_handle.
        from quark.runtime.tensor import QuarkTensor, _contiguous_strides, _MetalStorage

        assert result is not None, "Metal launch must return the output list"
        out_iter = iter(result)
        for b in pspec.buffers:
            if not b.readonly:
                handle, ptr, nbytes = next(out_iter)
                # Output shape comes from the auto-allocated dummy in `full`
                # (alloc_from_decl built it with the right shape/dtype).
                template = full[b.name]
                shape = tuple(int(s) for s in template.shape)
                storage = _MetalStorage(int(handle), int(ptr), int(nbytes))
                full[b.name] = QuarkTensor(
                    storage,
                    shape,
                    _contiguous_strides(shape),
                    0,
                    b.dtype.value,
                )
    # CUDA / QuarkTensor: buffers were written in place; nothing to rebind.
    return full


# queue_launch_ir derived-value cache. Keyed on the same (kernel_cls,
# spec, config) tuple as Launcher._kernel_cache, so each entry mirrors
# one CompiledKernel. The values that don't change across calls — the
# Metal pipeline handle, binding plan, threads_grid, threadgroup size,
# smem bytes — are precomputed once on first call instead of repeated
# attribute access + grid_fn() + block_fn() + tuple math on every
# launch. Saves ~3 µs / call which adds up on the 1000+ launches/frame
# hot path.
_QLI_DERIVED_CACHE: dict = {}

_BYTES_PER_DTYPE = {"bf16": 2, "f16": 2, "f32": 4, "s32": 4}


def queue_launch_ir(
    kernel_cls,
    spec,
    config,
    *,
    inputs: list,
    out_shape: tuple,
    out_dtype: str,
    out_handle: int = -1,
):
    """Compile an IR kernel (cached) and dispatch it via the lazy
    queue_launch path, returning a ``QuarkTensor``.

    Bypasses the heavy ``call_with_bindings`` flow (no per-call output
    alloc, no dict assembly, no autotune lookup at call time): the
    pipeline + binding plan are pulled from the IR-compiled module and
    handed straight to ``_md.queue_launch``.

    ``inputs`` is the ordered list of QuarkTensor / numpy-array inputs
    in the kernel's TENSORS order (skipping role="out" entries). Inputs
    with a ``metal_handle`` are dispatched zero-copy; raw numpy inputs
    are copied into Metal-owned input cache buffers.

    ``out_handle``: when ≥0, the kernel writes into the existing pinned
    Metal buffer at this handle (e.g. a cached ``_out_buf_cache``
    QuarkTensor). When ``-1`` (default), the output is pool-allocated
    and surfaced as a fresh transient ``QuarkTensor``.
    """
    from quark.drivers import _metal_dispatch as _md

    cache_key = (kernel_cls, spec, config)
    derived = _QLI_DERIVED_CACHE.get(cache_key)
    if derived is None:
        compiled = launcher().compile(kernel_cls, spec, config)
        mod = compiled.module
        grid = compiled.grid_fn(*compiled.grid_args)
        block = compiled.block_fn(*compiled.block_args)
        threads_grid = (
            grid[0] * block[0],
            grid[1] * block[1],
            grid[2] * block[2],
        )
        derived = (
            mod.pipeline,
            mod._binding_plan_int,
            threads_grid,
            block,
            mod.smem_bytes,
        )
        _QLI_DERIVED_CACHE[cache_key] = derived
    pipeline, plan, threads_grid, block, smem_bytes = derived

    handles: list[int] = []
    ptrs: list[int] = []
    sizes: list[int] = []
    for arr in inputs:
        # Zero-copy path: QuarkTensor metal_handle.
        metal_h = getattr(arr, "metal_handle", None)
        if metal_h is not None:
            handles.append(metal_h)
            ptrs.append(0)
            sizes.append(0)
        else:
            import numpy as np

            handles.append(-1)
            narr = np.ascontiguousarray(arr).reshape(-1)
            ptrs.append(narr.ctypes.data)
            sizes.append(int(narr.nbytes))

    nbytes = _BYTES_PER_DTYPE.get(out_dtype, 2)
    for d in out_shape:
        nbytes *= int(d)

    out_handle_ret, out_ptr = _md.queue_launch(
        pipeline,
        handles,
        ptrs,
        sizes,
        nbytes,
        plan,
        threads_grid,
        block,
        smem_bytes,
        int(out_handle),
    )
    from quark.runtime.tensor import QuarkTensor, _contiguous_strides, _MetalStorage

    storage = _MetalStorage(out_handle_ret, out_ptr, nbytes)
    return QuarkTensor(
        storage, tuple(out_shape), _contiguous_strides(tuple(out_shape)), 0, out_dtype
    )


def _copy_strided_source(elem_size: int) -> tuple[str, str]:
    """Return (MSL source, kernel name) for a strided copy kernel."""
    type_map = {1: "uchar", 2: "ushort", 4: "uint"}
    t = type_map[elem_size]
    name = f"copy_strided_{elem_size}"
    src = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        f"kernel void {name}(\n"
        f"    device const {t}* src [[buffer(0)]],\n"
        f"    device {t}* dst [[buffer(1)]],\n"
        "    constant int* meta [[buffer(2)]],\n"
        "    uint tid [[thread_position_in_grid]])\n"
        "{\n"
        "    int ndim = meta[0];\n"
        "    int src_offset = meta[1];\n"
        "    int total = meta[2];\n"
        "    if ((int)tid >= total) return;\n"
        "    int remaining = (int)tid;\n"
        "    int src_idx = src_offset;\n"
        "    for (int d = ndim - 1; d >= 0; d--) {\n"
        "        int dim_size = meta[3 + d];\n"
        "        int src_stride = meta[3 + ndim + d];\n"
        "        int dim_idx = remaining % dim_size;\n"
        "        remaining /= dim_size;\n"
        "        src_idx += dim_idx * src_stride;\n"
        "    }\n"
        "    dst[tid] = src[src_idx];\n"
        "}\n"
    )
    return src, name


_copy_strided_pipelines: dict = {}  # elem_size → Pipeline
_copy_strided_meta: list = []  # keep metadata np arrays alive until eval


def clear_strided_copy_meta():
    """Clear pending metadata arrays. Called after eval_queue."""
    _copy_strided_meta.clear()


def queue_strided_copy(
    src_handle: int,
    offset: int,
    strides: tuple[int, ...],
    shape: tuple[int, ...],
    elem_size: int,
):
    """Queue a strided→contiguous copy on Metal via the lazy pipeline.

    Returns a new QuarkTensor backed by a fresh contiguous Metal buffer.
    No GPU sync — the copy is just another queued op.
    """
    import numpy as np

    from quark.drivers import _metal_dispatch as _md
    from quark.runtime.tensor import QuarkTensor, _contiguous_strides, _MetalStorage

    if elem_size not in _copy_strided_pipelines:
        src_code, name = _copy_strided_source(elem_size)
        _copy_strided_pipelines[elem_size] = _md.compile(src_code, name, 196610)

    pipeline = _copy_strided_pipelines[elem_size]
    ndim = len(shape)
    total = 1
    for s in shape:
        total *= int(s)

    # Pack metadata: [ndim, src_offset, total, shape..., strides...]
    meta = np.array(
        [ndim, offset, total] + [int(s) for s in shape] + [int(s) for s in strides],
        dtype=np.int32,
    )
    _copy_strided_meta.append(meta)  # prevent GC until eval

    # Binding plan: buffer(0)=src, buffer(1)=dst(output), buffer(2)=meta
    plan = [(0, 0, 0), (1, 0, 1), (0, 1, 2)]

    out_nbytes = total * elem_size
    tg_size = min(256, total)
    grid = (total, 1, 1)

    out_handle, out_ptr = _md.queue_launch(
        pipeline,
        [src_handle, -1],
        [0, int(meta.ctypes.data)],
        [0, int(meta.nbytes)],
        out_nbytes,
        plan,
        grid,
        (tg_size, 1, 1),
        0,
    )

    storage = _MetalStorage(out_handle, out_ptr, out_nbytes)
    # Caller (``QuarkTensor.contiguous``) overwrites ``_dtype`` with the
    # source's dtype after this returns; "u8" is just a placeholder.
    return QuarkTensor(storage, shape, _contiguous_strides(shape), 0, "u8")


_ADD_BF16_SRC = (
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void add_bf16(\n"
    "    device const bfloat* a [[buffer(0)]],\n"
    "    device const bfloat* b [[buffer(1)]],\n"
    "    device bfloat* out [[buffer(2)]],\n"
    "    constant int* meta [[buffer(3)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    int total = meta[0];\n"
    "    int b_len = meta[1];\n"
    "    if ((int)tid >= total) return;\n"
    "    out[tid] = a[tid] + b[tid % b_len];\n"
    "}\n"
)
_add_bf16_pipeline = None


def queue_elemwise_add_bf16(a_handle, b_handle, a_numel, b_numel):
    """Queue a lazy bf16 element-wise add with broadcast on the smaller operand."""
    import numpy as np

    from quark.drivers import _metal_dispatch as _md

    global _add_bf16_pipeline
    if _add_bf16_pipeline is None:
        _add_bf16_pipeline = _md.compile(_ADD_BF16_SRC, "add_bf16", 196610)

    total = a_numel
    b_len = b_numel
    meta = np.array([total, b_len], dtype=np.int32)
    _copy_strided_meta.append(meta)

    plan = [(0, 0, 0), (0, 1, 1), (1, 0, 2), (0, 2, 3)]
    out_nbytes = total * 2

    out_handle, out_ptr = _md.queue_launch(
        _add_bf16_pipeline,
        [a_handle, b_handle, -1],
        [0, 0, int(meta.ctypes.data)],
        [0, 0, int(meta.nbytes)],
        out_nbytes,
        plan,
        (total, 1, 1),
        (min(256, total), 1, 1),
        0,
    )
    return out_handle, out_ptr, out_nbytes


def _zero_placeholder_s32(n: int, *, like: Any = None) -> Any:
    """Backend-appropriate zeros(n, s32) placeholder — np.ndarray on Metal,
    QuarkTensor on CUDA. ``like`` is accepted for future-proofing
    but currently ignored (QuarkTensor picks its own device)."""
    del like
    if IS_METAL:
        import numpy as np

        return np.zeros((n,), dtype=np.int32)
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.zeros(n, dtype="s32")


def _zero_placeholder_f32(n: int, *, like: Any = None) -> Any:
    del like
    if IS_METAL:
        import numpy as np

        return np.zeros((n,), dtype=np.float32)
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.zeros(n, dtype="f32")


def make_autotune(impl_fn, cls_fn, *, doc: str | None = None):
    """Build a ``pcf.<op>.autotune(...)`` wrapper from the kernel's
    ``_<op>_impl`` and ``_cls`` functions.

    The wrapper:
      1. Resolves the kernel class via ``cls_fn()``.
      2. Calls ``cls.spec_from_tensors(*args, **kwargs)`` — every
         kernel's ``spec_from_tensors`` shares the same signature as
         its impl, so the same args flow straight through.
      3. Runs the full genetic search via ``lookup_or_search`` and
         caches the winner.

    Returns the winning ``KernelConfig``.

    Per-op autotune wrappers used to inline this whole dance and
    drift apart over time. Centralizing it keeps the contract one
    place: ``pcf.<op>.autotune = make_autotune(_<op>_impl, _cls)``.
    """

    def autotune_fn(*args, **kwargs):
        cls = cls_fn()
        spec = cls.spec_from_tensors(*args, **kwargs)
        import quark

        with quark.max_autotune():
            config = launcher()._autotune.lookup_or_search(cls, spec)
        return config

    autotune_fn.__doc__ = doc or (
        "Run full genetic autotune for this kernel's "
        "(shape, dtype, device) triple. Blocks until complete; result "
        "is persisted to disk and to the hot dict so subsequent calls "
        "are immediate cache hits. Returns the winning ``KernelConfig``."
    )
    return autotune_fn
