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
    import torch

    device = like.device if like is not None else torch.device("cuda")
    # Zero-init rather than empty: kernels that use atomic scatter-add
    # into the output (moe_outproj) accumulate onto the starting buffer
    # contents. Uninitialized memory leaks garbage into the result —
    # manifests as a data-dependent cosine drop at inference time.
    # The extra memset is a single pass over the output tensor, cheap
    # next to the kernel itself.
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
    if config is None:
        config = launcher()._autotune.lookup_or_search(kernel_cls, spec)

    kernel = kernel_cls(spec, config)
    pspec = kernel.param_spec()

    # Allocate any auto-alloc tensors by looking up their TensorDecl.
    decls_by_name = {d.name: d for d in kernel_cls.TENSORS}
    if like is None:
        like = next(iter(provided.values()))
    full: dict = dict(provided)
    for name in auto_alloc:
        full[name] = alloc_from_decl(decls_by_name[name], spec, config, like=like)

    full = kernel.prepare_launch_tensors(full)
    buffers = [full[b.name] for b in pspec.buffers]

    # NOTE: there used to be an unconditional per-invocation warmup
    # launch here as a workaround for the first-launch-wrong-output
    # bug. That's been moved DOWN into ``CompiledKernel.launch``,
    # where it's keyed on ``(this CompiledKernel, hash(input
    # data_ptrs))``. So the warmup fires the FIRST time a kernel is
    # launched against a particular set of input addresses, then
    # never again for that combination — instead of every call.
    # Inference loops that reuse input tensors pay the warmup once.
    result = launch_kernel(kernel_cls, spec, config, buffers=buffers)

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


def torch_op(name: str, *, mutates_args=()):
    """Decorator that registers a function as a ``torch.library.custom_op``
    on CUDA, no-op on Metal.

    Use to define ``pcf.<op>(...)`` entry points that work on both
    backends with no per-call branching:

        @torch_op("popcorn::gemm", mutates_args=())
        def gemm(A: torch.Tensor, B: torch.Tensor, ...) -> torch.Tensor:
            return _gemm_impl(A, B, ...)

        @gemm.register_fake
        def _(A, B, ...): ...

    Calling ``gemm(...)`` with torch tensors goes through the registered
    custom op (compatible with ``torch.compile`` graph capture). On
    Metal the decorator is a pass-through and the call is a plain
    Python function call. The ``register_fake`` / ``register_autograd``
    attribute hooks are stubbed as no-op decorators on Metal so the
    body of each functional file doesn't need a Metal vs CUDA branch.
    """
    if IS_METAL:

        def _identity(f):
            return f

        def deco(fn):
            fn.register_fake = _identity
            fn.register_autograd = _identity
            return fn

        return deco
    import torch

    def deco(fn):
        # ``torch.library.custom_op`` requires real type objects on the
        # signature (not string annotations). When the caller uses
        # ``from __future__ import annotations`` the annotations come
        # in as strings; resolve them here against the caller's
        # globals so the user can write idiomatic Python.
        import typing

        try:
            globalns = getattr(fn, "__globals__", {})
            hints = typing.get_type_hints(fn, globalns={**globalns, "torch": torch})
            # ``get_type_hints`` resolves ``-> None`` to ``type(None)``,
            # but torch.library.infer_schema only accepts the literal
            # ``None`` for void returns. Normalize it back.
            if hints.get("return") is type(None):
                hints["return"] = None
            fn.__annotations__ = hints
        except Exception:
            pass
        return torch.library.custom_op(name, mutates_args=mutates_args)(fn)

    return deco


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
      4. Issues one ``impl_fn(*args, **kwargs)`` call so
         ``CompiledKernel._warmed_input_keys`` sees the user's input
         data_ptrs. The first inference call with the same tensors
         then skips the warmup launch (see
         ``CompiledKernel.launch`` doc).

    Returns the winning ``KernelConfig``.

    Per-op autotune wrappers used to inline this whole dance and
    drift apart over time. Centralizing it keeps the contract one
    place: ``pcf.<op>.autotune = make_autotune(_<op>_impl, _cls)``.
    """

    def autotune_fn(*args, **kwargs):
        cls = cls_fn()
        spec = cls.spec_from_tensors(*args, **kwargs)
        config = launcher()._autotune.lookup_or_search(cls, spec, depth="full")
        # Pre-warm the per-(CompiledKernel, input-data_ptrs) cache
        # for the user's actual input tensors.
        impl_fn(*args, **kwargs)
        return config

    autotune_fn.__doc__ = doc or (
        "Run full genetic autotune for this kernel's "
        "(shape, dtype, device) triple. Blocks until complete; result "
        "is persisted to disk and to the hot dict so subsequent calls "
        "are immediate cache hits. Also pre-warms the per-input warmup "
        "cache for the tensors passed in here, so the first inference "
        "call that reuses the same input addresses skips the warmup "
        "launch. Returns the winning ``KernelConfig``."
    )
    return autotune_fn
