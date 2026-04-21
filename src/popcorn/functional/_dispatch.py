"""Shared plumbing for ``popcorn.functional.*``: Launcher singleton,
output allocation from ``TensorDecl``, and a thin ``launch_kernel``
helper that hides the CUDA (in-place) vs MLX (returned-list) split.

Per-kernel wrappers (``functional/gemm.py`` etc.) own their positional-
args ↔ TENSORS binding; this module only supplies the utilities they
share.

WARMUP NOTE: ``call_with_bindings`` issues one throwaway kernel launch
+ synchronize before the real launch on every invocation. This
workarounds an unidentified bug where the first launch of a kernel
in a "fresh" GPU session produces wrong output (NaN / cos≪1) on
CUDA sm_120. Cost: ~2× kernel runtime per call. See
``CompiledKernel.launch`` for the matching per-CompiledKernel warmup
that fixes the same problem on the autotune side.
"""

from __future__ import annotations

import threading
from typing import Any

from popcorn.backend import IS_METAL, PT
from popcorn.device import current_device
from popcorn.launcher import Launcher

# Module-level cached Launcher. ``current_device()`` is itself cached
# inside ``popcorn.device``, so the first call fixes the singleton for
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


def alloc_from_decl(decl, spec, config, *, like=None):
    """Allocate a fresh tensor sized to ``decl.shape(spec, config)`` with
    ``decl.dtype(spec, config)`` in the active backend.

    On torch, ``like`` (any reference tensor) provides the device. On
    MLX we'd never call this — MLX allocates outputs itself — but we
    keep the path uniform for the CUDA custom-op wrapper.
    """
    shape = decl.shape(spec, config) if callable(decl.shape) else decl.shape
    dtype_val = decl.dtype(spec, config) if callable(decl.dtype) else decl.dtype
    # decl.dtype may be an IR DType (for static typing) or a callable.
    from popcorn.ir import DType

    if isinstance(dtype_val, DType):
        backend_dt = PT.ir_dtype_to_backend(dtype_val)
    else:
        backend_dt = PT.ir_dtype_to_backend(dtype_val)

    if IS_METAL:
        return PT.zeros(*shape, dtype=backend_dt)

    # CUDA path: use PopcornTensor if the input is a PopcornTensor (no torch),
    # otherwise fall back to torch for torch.Tensor inputs.
    from popcorn.runtime.tensor import PopcornTensor

    if like is not None and isinstance(like, PopcornTensor):
        from popcorn.ir import DType as _DT

        _IR_TO_PC = {
            _DT.BF16: "bf16",
            _DT.F16: "f16",
            _DT.F32: "f32",
            _DT.S32: "s32",
            _DT.U8: "u8",
            _DT.S8: "s8",
            _DT.E4M3: "e4m3",
            _DT.E5M2: "e5m2",
            _DT.B16: "u16",  # bit-typed 16-bit → u16 storage
            _DT.B32: "u32",
            _DT.U16: "u16",
            _DT.U32: "u32",
        }
        return PopcornTensor.zeros(*shape, dtype=_IR_TO_PC[dtype_val])

    import torch

    device = like.device if like is not None else torch.device("cuda")
    return torch.zeros(tuple(shape), dtype=backend_dt, device=device)


def launch_kernel(
    kernel_cls,
    spec,
    config,
    *,
    buffers: list,
    scalars: tuple = (),
) -> list | None:
    """Compile + launch; return MLX's output list or None on CUDA.

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
    ``TENSORS`` manifest.  On MLX the returned dict's auto-allocated
    entries are the fresh ``mx.array`` MLX emitted; on torch they are
    the pre-allocated ``torch.empty`` buffers (now populated in place).

    ``like`` provides the torch device for ``auto_alloc`` entries.
    Defaults to the first value in ``provided``.

    When ``config`` is ``None`` (the default), the config is resolved
    from the process-wide ``AutotuneCache``: hot dict → on-disk JSON →
    bundled defaults → inline search (fast by default; full when
    ``POPCORN_MAX_AUTOTUNE=1`` or inside ``with popcorn.max_autotune()``).

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
    decls_by_name = {d.name: d for d in kernel_cls.TENSORS}
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

    from popcorn.graph import active_stream

    capturing = bool(active_stream())

    if capturing:
        from popcorn.graph import log_capture_launch

        log_capture_launch(f"{kernel_cls.__name__}({spec})")

    # Pass caller's buffers directly. During graph capture, the
    # caller's tensors (weight Parameters, KV caches, auto-alloc
    # outputs) already have stable addresses — they don't get
    # reallocated between calls. No stable-buffer indirection needed.
    buffers = [full[b.name] for b in pspec.buffers]
    result = compiled.launch(buffers=buffers)

    if IS_METAL:
        # MLX returns one array per non-readonly buffer in pspec order.
        # Rebind the dict entries to the fresh arrays.
        assert result is not None, "Metal launch must return the output list"
        out_iter = iter(result)
        for b in pspec.buffers:
            if not b.readonly:
                full[b.name] = next(out_iter)
    # torch: buffers were written in place; nothing to rebind.
    return full


def is_mlx_tensor(x: Any) -> bool:
    """True iff ``x`` is an ``mx.array``. Delegates to PT's private
    helper so we don't repeat the lazy-import dance."""
    return PT._is_mx(x)


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
        import popcorn

        with popcorn.max_autotune():
            config = launcher()._autotune.lookup_or_search(cls, spec)
        return config

    autotune_fn.__doc__ = doc or (
        "Run full genetic autotune for this kernel's "
        "(shape, dtype, device) triple. Blocks until complete; result "
        "is persisted to disk and to the hot dict so subsequent calls "
        "are immediate cache hits. Returns the winning ``KernelConfig``."
    )
    return autotune_fn
