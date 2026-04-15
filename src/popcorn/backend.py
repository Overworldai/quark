"""Polymorphic tensor operations — one API, two backends.

EXEMPT FROM 500-LINE RULE: consolidating every torch-vs-mlx branch in
one file is the whole point — splitting would fragment the dispatch
table and spread ``if IS_METAL`` over multiple modules.

All ops live on ``PT`` (polymorphic tensor) as static methods, with a
single backend dispatch inside each method. That keeps call sites free
of ``if IS_METAL`` branches and puts the whole torch-vs-mlx divide in
exactly one file.

Platform dispatch: torch on CUDA / CPU (Linux/Windows), mlx on Metal
(darwin). Decided once at import via ``sys.platform``.

Usage:
    from popcorn.backend import PT

    A = PT.randn((M, K), dtype=PT.bfloat16)
    B = PT.zeros((K, N), dtype=PT.bfloat16)
    C = PT.matmul(A, B)

Individual ops are also exported as module-level names (``randn``,
``zeros``, …) for back-compat with existing call sites.
"""

from __future__ import annotations

import sys

IS_METAL = sys.platform == "darwin"


class PT:
    """Polymorphic tensor — static-method dispatch between torch and mlx.

    Every method does the minimum backend branch needed; dtype constants
    live as class attributes that resolve to the active backend's type.
    """

    # ------------------------------------------------------------------
    # Dtype constants
    # ------------------------------------------------------------------
    if IS_METAL:
        import mlx.core as _mx

        float32 = _mx.float32
        float16 = _mx.float16
        bfloat16 = _mx.bfloat16
        int32 = _mx.int32
        int64 = _mx.int64
        uint8 = _mx.uint8
        uint16 = _mx.uint16
        uint32 = _mx.uint32
        int8 = _mx.int8
        int16 = _mx.int16
    else:
        import torch as _torch

        float32 = _torch.float32
        float16 = _torch.float16
        bfloat16 = _torch.bfloat16
        int32 = _torch.int32
        int64 = _torch.int64
        uint8 = _torch.uint8
        uint16 = _torch.int16  # torch has no uint16; alias
        uint32 = _torch.int32  # torch has no uint32; alias
        int8 = _torch.int8
        int16 = _torch.int16

    # ------------------------------------------------------------------
    # Tensor-type handle (for isinstance checks)
    # ------------------------------------------------------------------
    @staticmethod
    def tensor_type() -> type:
        if IS_METAL:
            import mlx.core as mx

            return mx.array
        import torch

        return torch.Tensor

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------
    @staticmethod
    def _shape(shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(shape[0])
        return tuple(shape)

    @staticmethod
    def randn(*shape, dtype=None):
        shape = PT._shape(shape)
        if IS_METAL:
            import mlx.core as mx

            dt = dtype if dtype is not None else mx.float32
            return mx.random.normal(shape=shape).astype(dt)
        import torch

        dt = dtype if dtype is not None else torch.float32
        return torch.randn(*shape, dtype=dt, device="cuda")

    @staticmethod
    def zeros(*shape, dtype=None):
        shape = PT._shape(shape)
        if IS_METAL:
            import mlx.core as mx

            dt = dtype if dtype is not None else mx.float32
            return mx.zeros(shape, dtype=dt)
        import torch

        dt = dtype if dtype is not None else torch.float32
        return torch.zeros(*shape, dtype=dt, device="cuda")

    @staticmethod
    def ones(*shape, dtype=None):
        shape = PT._shape(shape)
        if IS_METAL:
            import mlx.core as mx

            dt = dtype if dtype is not None else mx.float32
            return mx.ones(shape, dtype=dt)
        import torch

        dt = dtype if dtype is not None else torch.float32
        return torch.ones(*shape, dtype=dt, device="cuda")

    @staticmethod
    def arange(*args, dtype=None):
        if IS_METAL:
            import mlx.core as mx

            arr = mx.arange(*args)
            return arr.astype(dtype) if dtype is not None else arr
        import torch

        return torch.arange(*args, dtype=dtype, device="cuda")

    @staticmethod
    def tensor(data, dtype=None):
        if IS_METAL:
            import mlx.core as mx

            arr = mx.array(data)
            return arr.astype(dtype) if dtype is not None else arr
        import torch

        return torch.tensor(data, dtype=dtype, device="cuda")

    @staticmethod
    def linspace(start, stop, n, dtype=None):
        if IS_METAL:
            import mlx.core as mx

            arr = mx.linspace(start, stop, n)
            return arr.astype(dtype) if dtype is not None else arr
        import torch

        return torch.linspace(start, stop, n, dtype=dtype, device="cuda")

    # ------------------------------------------------------------------
    # Shape / combinators — dispatch on the tensor type so a function
    # called with torch tensors on a Metal host (or vice versa) still
    # Does The Right Thing.
    # ------------------------------------------------------------------
    @staticmethod
    def cat(arrays, dim: int = -1):
        arrays = list(arrays)
        if arrays and PT._is_mx(arrays[0]):
            import mlx.core as mx

            return mx.concatenate(arrays, axis=dim)
        import torch

        return torch.cat(arrays, dim=dim)

    @staticmethod
    def broadcast_to(x, shape):
        if PT._is_mx(x):
            import mlx.core as mx

            return mx.broadcast_to(x, tuple(shape))
        return x.expand(*shape)

    @staticmethod
    def repeat_interleave(x, n: int, dim: int = -1):
        if PT._is_mx(x):
            import mlx.core as mx

            return mx.repeat(x, n, axis=dim)
        return x.repeat_interleave(n, dim=dim)

    @staticmethod
    def contiguous(x):
        """Return a contiguous version of ``x``. torch: ``.contiguous()``
        (no-op if already contiguous, else a fresh copy). mlx: identity
        since mlx arrays own their buffer directly — the concept of
        "non-contiguous view" doesn't apply in the same way.

        Use this before handing a tensor to a kernel launch when the
        tensor may have come through ``reshape`` / ``expand`` / slicing
        that could have produced a strided view — the kernel reads raw
        bytes via ``data_ptr()`` and assumes row-major contiguous
        layout.
        """
        if PT._is_mx(x):
            return x
        return x.contiguous()

    @staticmethod
    def transpose(x, dim0: int = -1, dim1: int = -2):
        """Swap two axes. Defaults to the last two (most common case —
        A @ B.T, K_cache ↔ Vt_cache). mlx / torch agree on swapaxes,
        but torch's historical name is transpose; unify here so call
        sites never have to know."""
        if PT._is_mx(x):
            return x.swapaxes(dim0, dim1)
        return x.transpose(dim0, dim1)

    @staticmethod
    def _is_mx(x) -> bool:
        """True iff ``x`` is an mlx.core.array. Type-based dispatch so
        PT methods work even when the caller mixes tensor types (e.g.
        ``reference()`` gets torch CPU tensors on a Metal host)."""
        try:
            import mlx.core as mx

            return isinstance(x, mx.array)
        except ImportError:
            return False

    @staticmethod
    def _as_torch_dtype(dtype):
        """Translate whatever PT.float32 / mx.float32 / torch.float32 is
        into the equivalent torch dtype."""
        import torch

        if isinstance(dtype, torch.dtype):
            return dtype
        # mlx dtype. Use its name.
        name = str(dtype).rsplit(".", 1)[-1]
        return getattr(torch, name, torch.float32)

    @staticmethod
    def _as_mx_dtype(dtype):
        import mlx.core as mx

        if str(type(dtype).__module__).startswith("mlx"):
            return dtype
        name = str(dtype).rsplit(".", 1)[-1]
        return getattr(mx, name, mx.float32)

    @staticmethod
    def astype(x, dtype):
        if PT._is_mx(x):
            return x.astype(PT._as_mx_dtype(dtype))
        return x.to(PT._as_torch_dtype(dtype))

    @staticmethod
    def set_slice(dst, *, axis: int, start: int, stop: int, src):
        """Return ``dst`` with ``dst[..., start:stop, ...]`` along
        ``axis`` replaced by ``src``. Torch mutates in place; MLX
        returns a concat-rebuilt tensor (mlx arrays are immutable)."""
        if PT._is_mx(dst):
            import mlx.core as mx

            pre = (
                mx.take(dst, mx.arange(0, start, dtype=mx.int32), axis=axis) if start > 0 else None
            )
            post = (
                mx.take(dst, mx.arange(stop, dst.shape[axis], dtype=mx.int32), axis=axis)
                if stop < dst.shape[axis]
                else None
            )
            parts = [p for p in (pre, src, post) if p is not None]
            return mx.concatenate(parts, axis=axis) if len(parts) > 1 else parts[0]
        idx = [slice(None)] * dst.ndim
        idx[axis] = slice(start, stop)
        dst[tuple(idx)] = src
        return dst

    # ------------------------------------------------------------------
    # Math — ``a @ b`` / torch.exp work on whatever tensor type the
    # input is, no dispatch needed for most. cos/sin/exp differ across
    # modules so we branch on the input type.
    # ------------------------------------------------------------------
    @staticmethod
    def matmul(a, b):
        return a @ b

    @staticmethod
    def exp(x):
        if PT._is_mx(x):
            import mlx.core as mx

            return mx.exp(x)
        import torch

        return torch.exp(x)

    @staticmethod
    def cos(x):
        if PT._is_mx(x):
            import mlx.core as mx

            return mx.cos(x)
        import torch

        return torch.cos(x)

    @staticmethod
    def sin(x):
        if PT._is_mx(x):
            import mlx.core as mx

            return mx.sin(x)
        import torch

        return torch.sin(x)

    @staticmethod
    def where(cond, a, b):
        if PT._is_mx(cond):
            import mlx.core as mx

            return mx.where(cond, a, b)
        import torch

        return torch.where(cond, a, b)

    @staticmethod
    def index_copy(dst, dim: int, index, src):
        """Out-of-place index_copy — returns a tensor of the same shape
        as ``dst`` where slices along ``dim`` at positions in ``index``
        are replaced by rows from ``src``. On torch this mutates ``dst``
        in place and returns it; on mlx a new tensor is returned.
        Callers should use the return value either way."""
        if PT._is_mx(dst):
            import mlx.core as mx

            # mx.scatter is awkward; use the at[].add semantics via
            # a zero-then-add on the destination axis. For a true
            # replace, reassemble: build a mask and index_put.
            # Simpler: move `dim` to axis 0, update, move back.
            perm = list(range(dst.ndim))
            perm[0], perm[dim] = perm[dim], perm[0]
            d = mx.transpose(dst, perm)
            s = mx.transpose(src, perm)
            # mlx fancy-index assignment via take_along_axis is missing,
            # so emulate with at[idx].set.
            d = d.at[index].add(s - d[index])
            return mx.transpose(d, perm)
        dst.index_copy_(dim, index, src)
        return dst

    @staticmethod
    def zeros_like(x):
        if PT._is_mx(x):
            import mlx.core as mx

            return mx.zeros_like(x)
        import torch

        return torch.zeros_like(x)

    @staticmethod
    def compile(fn=None, **kwargs):
        """torch.compile on CUDA, identity on Metal.

        Usable as either a direct call or a decorator::

            @PT.compile
            def f(x): ...

            @PT.compile(mode="max-autotune")
            def g(x): ...

            g = PT.compile(g, mode="max-autotune-no-cudagraphs")
        """
        if IS_METAL:
            # MLX has no compile; return identity (or decorator-factory if
            # used with kwargs).
            if fn is not None:
                return fn
            return lambda f: f

        import torch

        if fn is None:
            return lambda f: torch.compile(f, **kwargs)
        return torch.compile(fn, **kwargs)

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    @staticmethod
    def synchronize() -> None:
        if IS_METAL:
            import mlx.core as mx

            mx.synchronize()
            return
        import torch

        torch.cuda.synchronize()

    @staticmethod
    def device_name() -> str:
        if IS_METAL:
            import mlx.core as mx

            return str(mx.device_info().get("device_name", "apple-gpu"))
        import torch

        return torch.cuda.get_device_name(0)

    @staticmethod
    def attention(q, k, v, *, mask=None, scale=None):
        """Masked scaled-dot-product attention. Dispatches to each
        backend's fast path so kernels' references never have to
        hand-write a softmax:
          - mlx: ``mx.fast.scaled_dot_product_attention`` (flash-attn
            on Apple silicon).
          - torch: ``F.scaled_dot_product_attention`` (flash / memory-
            efficient attention kernels under the hood on CUDA, plain
            math impl on CPU).

        Shapes: ``q [B, H_q, L_q, D]``, ``k/v [B, H_kv, L_kv, D]``.
        ``mask`` is additive (float, 0 = keep, -inf = drop) with a shape
        that broadcasts to ``[B, H_q, L_q, L_kv]``. GQA (H_q != H_kv)
        is handled by the backend."""
        if PT._is_mx(q):
            import mlx.core as mx

            s = scale if scale is not None else float(q.shape[-1]) ** -0.5
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=s, mask=mask)
        import torch.nn.functional as F

        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)

    @staticmethod
    def has_nan(x) -> bool:
        if PT._is_mx(x):
            import mlx.core as mx

            return bool(mx.any(mx.isnan(x)).item())
        import torch

        return bool(torch.isnan(x).any().item())

    @staticmethod
    def all_finite(x) -> bool:
        if PT._is_mx(x):
            import mlx.core as mx

            return bool((~(mx.isnan(x) | mx.isinf(x))).all().item())
        import torch

        # torch.isfinite errors on fp8 (Float8_e4m3fn / _e5m2) — cast to
        # f32 first. Up-cast is lossless for the finite-vs-not decision.
        if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            x = x.to(torch.float32)
        return bool(torch.isfinite(x).all().item())

    @staticmethod
    def abs_diff_stats(a, b) -> tuple[float, float]:
        """Return ``(mean_abs, max_abs)`` of ``|a - b|`` as Python floats.
        Backend-polymorphic — used by the correctness check for failure
        diagnostics."""
        a32 = PT.astype(a, PT.float32).reshape(-1)
        b32 = PT.astype(b, PT.float32).reshape(-1)
        if PT._is_mx(a32):
            import mlx.core as mx

            d = mx.abs(a32 - b32)
            return float(mx.mean(d).item()), float(mx.max(d).item())
        import torch

        d = torch.abs(a32 - b32)
        return float(d.mean().item()), float(d.max().item())

    @staticmethod
    def first_non_finite_index(x) -> int | None:
        """Flat index of the first non-finite element, or ``None`` if
        all finite."""
        if PT._is_mx(x):
            import mlx.core as mx

            flat = x.reshape(-1)
            bad = mx.isnan(flat) | mx.isinf(flat)
            if not bool(mx.any(bad).item()):
                return None
            # argmax on a boolean — the first True wins.
            return int(mx.argmax(bad.astype(mx.int32)).item())
        import torch

        flat = x.reshape(-1)
        bad = ~torch.isfinite(flat)
        if not bool(bad.any().item()):
            return None
        return int(bad.nonzero()[0].item())

    @staticmethod
    def numel(x) -> int:
        if PT._is_mx(x):
            return int(x.size)
        return int(x.numel())

    @staticmethod
    def cosine_sim(a, b) -> float:
        """Flat cosine similarity between two tensors (any shape, any
        backend). Returns a Python float in ``[-1, 1]`` — computed in
        fp32 to avoid bf16/fp16 drift overwhelming the measurement.
        Adds 1e-30 to the denominator so all-zero inputs return 0
        instead of NaN."""
        a32 = PT.astype(a, PT.float32).reshape(-1)
        b32 = PT.astype(b, PT.float32).reshape(-1)
        num = (a32 * b32).sum()
        # Explicit sqrt(sum(x*x)) — torch and mlx both support it, and
        # unlike .norm() it doesn't differ between the two APIs.
        a_norm = (a32 * a32).sum() ** 0.5
        b_norm = (b32 * b32).sum() ** 0.5
        cs = num / (a_norm * b_norm + 1e-30)
        return float(cs)

    @staticmethod
    def zero_(x):
        """Return an all-zero tensor with the same shape+dtype. MLX
        arrays are immutable so this is not in-place; call sites should
        reassign: ``buf = PT.zero_(buf)``."""
        if PT._is_mx(x):
            import mlx.core as mx

            return mx.zeros(x.shape, dtype=x.dtype)
        return x.zero_()
        return x

    # ------------------------------------------------------------------
    # Cross-backend bridges (for correctness / bench reporting)
    # ------------------------------------------------------------------
    @staticmethod
    def to_cpu_numpy(x):
        import numpy as np

        if IS_METAL:
            return np.array(x)
        return x.detach().cpu().numpy()

    @staticmethod
    def to_torch_cpu(x):
        """Convert either backend's tensor into a CPU torch tensor for
        the correctness checker. Preserves integer dtypes; upcasts
        floats to f32 to sidestep MLX's bf16-numpy limitation."""
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        import mlx.core as mx
        import numpy as np

        if x.dtype in (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.uint16, mx.uint32):
            arr = np.array(x)
        else:
            arr = np.array(x.astype(mx.float32) if x.dtype != mx.float32 else x)
        return torch.from_numpy(arr)

    # ------------------------------------------------------------------
    # IR dtype mapping
    # ------------------------------------------------------------------
    @staticmethod
    def ir_dtype_to_backend(ir_dtype):
        """Map ``popcorn.ir.DType`` → the active backend's dtype."""
        from popcorn.ir import DType

        if IS_METAL:
            import mlx.core as mx

            table = {
                DType.F32: mx.float32,
                DType.F16: mx.float16,
                DType.BF16: mx.bfloat16,
                DType.S32: mx.int32,
                DType.U8: mx.uint8,
                DType.S8: mx.int8,
            }
            return table.get(ir_dtype, mx.float32)
        import torch

        table = {
            DType.F32: torch.float32,
            DType.F16: torch.float16,
            DType.BF16: torch.bfloat16,
            DType.S32: torch.int32,
            DType.U8: torch.uint8,
            DType.S8: torch.int8,
        }
        fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
        if fp8_e4m3 is not None:
            table[DType.E4M3] = fp8_e4m3
        fp8_e5m2 = getattr(torch, "float8_e5m2", None)
        if fp8_e5m2 is not None:
            table[DType.E5M2] = fp8_e5m2
        return table.get(ir_dtype, torch.float32)

    @staticmethod
    def backend_dtype_to_ir(dt):
        """Map the active backend's dtype → ``popcorn.ir.DType``."""
        from popcorn.ir import DType

        if IS_METAL:
            import mlx.core as mx

            table = {
                mx.float32: DType.F32,
                mx.float16: DType.F16,
                mx.bfloat16: DType.BF16,
                mx.int32: DType.S32,
                mx.uint8: DType.U8,
                mx.int8: DType.S8,
            }
            return table.get(dt, DType.F32)
        import torch

        table = {
            torch.float32: DType.F32,
            torch.float16: DType.F16,
            torch.bfloat16: DType.BF16,
            torch.int32: DType.S32,
            torch.uint8: DType.U8,
            torch.int8: DType.S8,
        }
        fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
        if fp8_e4m3 is not None:
            table[fp8_e4m3] = DType.E4M3
        fp8_e5m2 = getattr(torch, "float8_e5m2", None)
        if fp8_e5m2 is not None:
            table[fp8_e5m2] = DType.E5M2
        return table.get(dt, DType.F32)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    @staticmethod
    def time_callable(fn, *, warmup_ms: float = 25.0, bench_ms: float = 100.0) -> float:
        """Budget-based timer. Returns microseconds per call."""
        import time as _time

        if IS_METAL:
            import mlx.core as mx

            fn()
            mx.synchronize()
            # Probe
            elapsed_ms, n = 0.0, 0
            while elapsed_ms < 1.0:
                n = max(n * 2, 1)
                t0 = _time.perf_counter()
                for _ in range(n):
                    fn()
                mx.synchronize()
                elapsed_ms = (_time.perf_counter() - t0) * 1000
            # Warm-up
            for _ in range(max(1, int(warmup_ms / (elapsed_ms / n) if elapsed_ms > 0 else 1))):
                fn()
            mx.synchronize()
            # Bench
            n_bench = max(1, int(bench_ms / (elapsed_ms / n) if elapsed_ms > 0 else 1))
            t0 = _time.perf_counter()
            for _ in range(n_bench):
                fn()
            mx.synchronize()
            return (_time.perf_counter() - t0) / n_bench * 1e6
        import torch

        torch.cuda.synchronize()
        # Warm-up budget
        n_warm = 0
        t0 = _time.perf_counter()
        while (_time.perf_counter() - t0) * 1000 < warmup_ms:
            fn()
            n_warm += 1
        torch.cuda.synchronize()
        # Bench
        n_bench = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = _time.perf_counter()
        while (_time.perf_counter() - t0) * 1000 < bench_ms:
            fn()
            n_bench += 1
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / max(n_bench, 1)
