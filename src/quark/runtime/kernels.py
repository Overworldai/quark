"""Dispatch layer between QuarkTensor ops and quark kernels.

EXEMPT FROM 500-LINE RULE: every QuarkTensor op routes through here.
Splitting by op category (binary, unary, copy, reduce, cat) would
scatter the shared padding/flatten/launch helpers across files with
no readability benefit.

QuarkTensor.__add__ etc. call functions here, which build the
appropriate spec, compile via the Launcher, and launch. All kernels
go through the standard quark pipeline (IR → lowerer → driver).

These functions handle:
- Flattening tensor shapes to 1-D for the elementwise kernel
- Padding N to be divisible by elems_per_block
- Casting to/from f32 when needed
- Building CopyStridedSpec from tensor stride metadata
"""

from __future__ import annotations

import math
import random
import struct as _struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quark.launcher.launcher import CompiledKernel
    from quark.runtime.tensor import QuarkTensor


def _stream() -> int:
    """Return the active CUDA stream (capture stream during graph capture, else 0)."""
    from quark.graph import active_stream

    return active_stream()


def _pad_n(n: int, target: int = 128) -> int:
    """Round up N to the nearest multiple of target."""
    return ((n + target - 1) // target) * target


def _launcher():
    """Get or create the process-wide Launcher."""
    from quark.functional._dispatch import launcher

    return launcher()


def _elemwise_cls():
    from quark.kernels import get

    return get("elementwise")


def _copy_strided_cls():
    from quark.kernels import get

    return get("copy_strided")


# ── Elementwise ops ──────────────────────────────────────────


def _broadcast_rows(t: QuarkTensor, n_rows: int) -> QuarkTensor:
    """Expand [1, D] → [n_rows, D] by repeating the single row."""
    from quark.runtime.tensor import QuarkTensor

    D = t.shape[1]
    parts = [t] * n_rows
    return QuarkTensor.cat(parts, dim=0).reshape(n_rows, D)


def elemwise_binary(op: str, a: QuarkTensor, b, *, out: QuarkTensor | None = None) -> QuarkTensor:
    """Element-wise binary op. ``b`` can be a QuarkTensor or scalar.

    ``out``: optional pre-allocated output buffer (same shape + dtype as
    ``a``). Lets the caller (e.g. ``nn.Add``) reuse a cached buffer and
    skip the per-call ``cuMemAllocAsync`` (~150 µs on Blackwell).
    """
    from quark.runtime.tensor import QuarkTensor

    if isinstance(b, (int, float)):
        return _binary_with_scalar(op, a, float(b))
    if isinstance(b, QuarkTensor):
        if a.shape != b.shape:
            if b.numel() == 1:
                return _binary_with_scalar(op, a, b.item())
            if a.numel() == 1:
                return _binary_with_scalar_lhs(op, a.item(), b)
            # Broadcast: [M, D] op [1, D] → expand b to [M, D].
            if (
                len(a.shape) == 2
                and len(b.shape) == 2
                and b.shape[0] == 1
                and a.shape[1] == b.shape[1]
            ):
                b = _broadcast_rows(b, a.shape[0])
            elif (
                len(a.shape) == 2
                and len(b.shape) == 2
                and a.shape[0] == 1
                and a.shape[1] == b.shape[1]
            ):
                a = _broadcast_rows(a, b.shape[0])
            else:
                raise ValueError(f"elemwise_binary: shape mismatch {a.shape} vs {b.shape}")
        return _binary_tensors(op, a, b, out=out)
    raise TypeError(f"elemwise_binary: unsupported type {type(b).__name__}")


def elemwise_binary_scalar_lhs(op: str, scalar: float, t: QuarkTensor) -> QuarkTensor:
    """scalar <op> tensor."""
    return _binary_with_scalar_lhs(op, float(scalar), t)


def _binary_tensors(
    op: str, a: QuarkTensor, b: QuarkTensor, *, out: QuarkTensor | None = None
) -> QuarkTensor:
    """Binary op on two same-shape tensors."""
    from quark.ir import DType
    from quark.kernels.elementwise.config import ElementwiseConfig
    from quark.kernels.elementwise.spec import ElementwiseSpec
    from quark.runtime.tensor import QuarkTensor

    orig_shape = a.shape
    orig_dtype = a.dtype
    N = a.numel()

    # Flatten to 1-D, pad if needed.
    a_flat = a.reshape(N).contiguous()
    b_flat = b.reshape(N).contiguous()

    padded_N = _pad_n(N)
    if padded_N != N:
        a_flat = _pad_tensor(a_flat, padded_N)
        b_flat = _pad_tensor(b_flat, padded_N)

    spec = ElementwiseSpec(N=padded_N, dtype=DType(orig_dtype), op=op)
    config = ElementwiseConfig.default_for(spec)

    compiled = _launcher().compile(_elemwise_cls(), spec, config)
    # Fast path: caller-provided Out that's already the full unpadded
    # shape. Requires N == padded_N (tile-aligned) so we can launch
    # straight into it.
    if out is not None and padded_N == N:
        Out_flat = out.reshape(N) if out.shape != (N,) else out
        compiled.launch(buffers=[a_flat, b_flat, Out_flat])
        return out.reshape(*orig_shape) if out.shape != orig_shape else out

    # Slow path: allocate a fresh (possibly padded) temp, slice/copy
    # into ``out`` if the caller provided one. ``empty`` instead of
    # ``zeros`` — the kernel writes every element so the memset was
    # pure overhead.
    Out = QuarkTensor.empty(padded_N, dtype=orig_dtype)
    compiled.launch(buffers=[a_flat, b_flat, Out])
    if padded_N != N:
        Out = Out[:N]
    result = Out.reshape(*orig_shape)
    if out is not None:
        from quark.runtime.cuda import CudaRuntime
        from quark.runtime.tensor import PC_BYTES

        CudaRuntime.instance().memcpy_dtod(
            out.data_ptr(), result.contiguous().data_ptr(), N * PC_BYTES[orig_dtype]
        )
        return out
    return result


def _binary_with_scalar(op: str, t: QuarkTensor, scalar: float) -> QuarkTensor:
    """tensor <op> scalar: broadcast scalar to a 1-element tensor."""
    from quark.runtime.tensor import QuarkTensor

    # Create a scalar tensor, broadcast via the kernel.
    # The elementwise kernel doesn't support broadcasting natively,
    # so we fill a full-size tensor with the scalar value.
    orig_shape = t.shape
    orig_dtype = t.dtype
    N = t.numel()

    t_flat = t.reshape(N).contiguous()
    padded_N = _pad_n(N)
    if padded_N != N:
        t_flat = _pad_tensor(t_flat, padded_N)

    # Create scalar-filled tensor.
    scalar_t = _fill_tensor(padded_N, scalar, orig_dtype)

    from quark.ir import DType
    from quark.kernels.elementwise.config import ElementwiseConfig
    from quark.kernels.elementwise.spec import ElementwiseSpec

    spec = ElementwiseSpec(N=padded_N, dtype=DType(orig_dtype), op=op)
    config = ElementwiseConfig.default_for(spec)

    compiled = _launcher().compile(_elemwise_cls(), spec, config)
    Out = QuarkTensor.zeros(padded_N, dtype=orig_dtype)
    buffers = [t_flat, scalar_t, Out]
    compiled.launch(buffers=buffers)

    if padded_N != N:
        Out = Out[:N]
    return Out.reshape(*orig_shape)


def _binary_with_scalar_lhs(op: str, scalar: float, t: QuarkTensor) -> QuarkTensor:
    """scalar <op> tensor. For sub/div, order matters."""
    from quark.runtime.tensor import QuarkTensor

    orig_shape = t.shape
    orig_dtype = t.dtype
    N = t.numel()

    t_flat = t.reshape(N).contiguous()
    padded_N = _pad_n(N)
    if padded_N != N:
        t_flat = _pad_tensor(t_flat, padded_N)

    scalar_t = _fill_tensor(padded_N, scalar, orig_dtype)

    from quark.ir import DType
    from quark.kernels.elementwise.config import ElementwiseConfig
    from quark.kernels.elementwise.spec import ElementwiseSpec

    spec = ElementwiseSpec(N=padded_N, dtype=DType(orig_dtype), op=op)
    config = ElementwiseConfig.default_for(spec)

    compiled = _launcher().compile(_elemwise_cls(), spec, config)
    Out = QuarkTensor.zeros(padded_N, dtype=orig_dtype)
    # scalar is LHS: X=scalar_t, Y=t_flat
    buffers = [scalar_t, t_flat, Out]
    compiled.launch(buffers=buffers)

    if padded_N != N:
        Out = Out[:N]
    return Out.reshape(*orig_shape)


def elemwise_unary(op: str, t: QuarkTensor) -> QuarkTensor:
    """Element-wise unary op."""
    from quark.ir import DType
    from quark.kernels.elementwise.config import ElementwiseConfig
    from quark.kernels.elementwise.spec import ElementwiseSpec
    from quark.runtime.tensor import QuarkTensor

    orig_shape = t.shape
    orig_dtype = t.dtype
    N = t.numel()

    t_flat = t.reshape(N).contiguous()
    padded_N = _pad_n(N)
    if padded_N != N:
        t_flat = _pad_tensor(t_flat, padded_N)

    spec = ElementwiseSpec(N=padded_N, dtype=DType(orig_dtype), op=op)
    config = ElementwiseConfig.default_for(spec)

    compiled = _launcher().compile(_elemwise_cls(), spec, config)
    Out = QuarkTensor.zeros(padded_N, dtype=orig_dtype)
    # Dummy Y for unary ops.
    dummy_Y = QuarkTensor.zeros(1, dtype=orig_dtype)
    buffers = [t_flat, dummy_Y, Out]
    compiled.launch(buffers=buffers)

    if padded_N != N:
        Out = Out[:N]
    return Out.reshape(*orig_shape)


def cast(
    t: QuarkTensor,
    target_dtype: str,
    *,
    out: QuarkTensor | None = None,
    dummy_y: QuarkTensor | None = None,
) -> QuarkTensor:
    """Cast between dtypes via the elementwise kernel.

    ``out`` (optional): caller-provided pre-allocated output buffer
    with matching shape and ``target_dtype``. When supplied, bypasses
    the per-call ``QuarkTensor.empty`` — used by ``nn.Cast`` to
    reuse a shape-keyed output cache and eliminate the per-cast
    ``cuMemAllocAsync``.

    ``dummy_y`` (optional): caller-provided 1-element tensor of
    ``t.dtype`` that stands in for the elementwise kernel's unused
    ``Y`` input. Lets ``nn.Cast`` keep a stable pointer across calls
    so the kernel's ``input_key`` doesn't drift and the first-launch
    warmup never refires. When None, a fresh dummy is allocated.

    Fast path: when ``t`` is contiguous and its element count is a
    multiple of ``epb`` (the kernel's tile), the kernel runs straight
    on ``t``'s storage into ``out`` (or a fresh output). No reshape,
    no contiguous() copy, no pad, no slice.

    Slow path only when ``N`` isn't tile-aligned — round ``N`` up to
    the next tile boundary, zero-pad in a temporary, cast, then slice.
    Autotune's default config is picked so the fast path covers every
    spec we actually hit in the hot loop.
    """
    from quark.ir import DType
    from quark.kernels.elementwise.config import ElementwiseConfig
    from quark.kernels.elementwise.spec import ElementwiseSpec
    from quark.runtime.tensor import QuarkTensor

    if t.dtype == target_dtype:
        return t

    orig_shape = t.shape
    N = t.numel()

    # We need a flat, contiguous view of the source. Avoid the
    # ``reshape(N).contiguous()`` sequence — the reshape is a no-op
    # for contiguous input anyway, and ``contiguous()`` still forces
    # a D→D copy when the tensor is already contiguous in its current
    # shape. Since the elementwise kernel indexes into a flat address
    # space, we only need to know it's contiguous in storage.
    t_flat = t if t.is_contiguous() else t.contiguous()

    # Probe the default config — its ``elems_per_block`` tells us
    # whether N is tile-aligned for the fast path. The spec we
    # actually compile is sized to match the chosen path so
    # ``is_valid`` always accepts it.
    probe_spec = ElementwiseSpec(
        N=N,
        dtype=DType(t.dtype),
        op="cast",
        out_dtype=DType(target_dtype),
    )
    config = ElementwiseConfig.default_for(probe_spec)
    epb = config.elems_per_block

    src_dt = DType(t.dtype)
    dst_dt = DType(target_dtype)
    y = dummy_y if dummy_y is not None else QuarkTensor.zeros(1, dtype=t.dtype)

    if N % epb == 0:
        # Fast path: launch into ``out`` (caller-provided) or a fresh
        # output tensor of the true shape.
        spec = probe_spec
        compiled = _launcher().compile(_elemwise_cls(), spec, config)
        Out = out if out is not None else QuarkTensor.empty(*orig_shape, dtype=target_dtype)
        compiled.launch(buffers=[t_flat, y, Out])
        return Out

    # Slow path: tile-align via a padded temp. Rebuild the spec at
    # ``padded_N`` so the compiled kernel's ``is_valid`` passes. The
    # ``out=`` fast-path output-cache doesn't help here because the
    # kernel writes into a padded temporary that we then slice; the
    # slow path is for tile-misaligned Ns we don't hit in hot loops.
    padded_N = _pad_n(N, target=epb)
    spec = ElementwiseSpec(N=padded_N, dtype=src_dt, op="cast", out_dtype=dst_dt)
    compiled = _launcher().compile(_elemwise_cls(), spec, config)
    t_padded = _pad_tensor(t_flat, padded_N)
    Out_padded = QuarkTensor.empty(padded_N, dtype=target_dtype)
    compiled.launch(buffers=[t_padded, y, Out_padded])
    result = Out_padded[:N].reshape(*orig_shape)
    if out is not None:
        # Honor the caller's output buffer contract by copying into it —
        # this path is cold so one D→D memcpy is acceptable.
        from quark.runtime.cuda import CudaRuntime
        from quark.runtime.tensor import PC_BYTES

        CudaRuntime.instance().memcpy_dtod(
            out.data_ptr(), result.contiguous().data_ptr(), N * PC_BYTES[target_dtype]
        )
        return out
    return result


# ── Scalar increment ─────────────────────────────────────────


# Cached compile of the increment kernel (one per dtype).
_incr_compiled: dict[str, CompiledKernel] = {}


def scalar_increment(t: QuarkTensor) -> None:
    """In-place ``t[0] += 1`` on a 1-element s32/u32 tensor.

    Launches the dedicated 1-block, 1-thread ``increment`` kernel so
    the op stays on-device and graph-capturable.
    """
    from quark.ir import DType
    from quark.kernels import get
    from quark.kernels.increment.kernel import IncrementConfig, IncrementSpec

    if t.numel() != 1 or t._dtype not in ("s32", "u32"):
        raise ValueError("scalar_increment: requires single-element s32/u32 tensor")

    dt = t._dtype
    compiled = _incr_compiled.get(dt)
    if compiled is None:
        dtype_ir = DType.S32 if dt == "s32" else DType.U32
        spec = IncrementSpec(dtype=dtype_ir)
        config = IncrementConfig.default_for(spec)
        compiled = _launcher().compile(get("increment"), spec, config)
        _incr_compiled[dt] = compiled
    compiled.launch(buffers=[t])


# ── Copy strided ─────────────────────────────────────────────


def copy_strided(t: QuarkTensor) -> QuarkTensor:
    """Copy a non-contiguous tensor into a fresh contiguous buffer."""
    from quark.runtime.tensor import PC_BYTES, QuarkTensor, _contiguous_strides

    N = t.numel()
    if N == 0:
        return QuarkTensor.empty(*t.shape, dtype=t.dtype)

    elem = PC_BYTES[t.dtype]

    # Fast path: contiguous data at an offset → simple D2D memcpy.
    from quark.runtime.tensor import _is_contiguous

    if _is_contiguous(t.shape, t.strides) and t._offset > 0:
        out = QuarkTensor.empty(*t.shape, dtype=t.dtype)
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memcpy_dtod(out.data_ptr(), t.data_ptr(), N * elem)
        return out

    # General strided copy via the copy_strided kernel.
    # Pad shape/strides to 4D.
    ndim = t.ndim
    shape_4d = list(t.shape) + [1] * (4 - ndim)
    strides_4d = list(t.strides) + [1] * (4 - ndim)

    padded_N = _pad_n(N)

    from quark.ir import DType
    from quark.kernels.copy_strided.config import CopyStridedConfig
    from quark.kernels.copy_strided.spec import CopyStridedSpec

    spec = CopyStridedSpec(
        N=padded_N,
        dtype=DType(t.dtype),
        ndim=min(ndim, 4),
        shape0=shape_4d[0],
        shape1=shape_4d[1],
        shape2=shape_4d[2],
        shape3=shape_4d[3],
        stride0=strides_4d[0],
        stride1=strides_4d[1],
        stride2=strides_4d[2],
        stride3=strides_4d[3],
        offset=t._offset,
    )
    config = CopyStridedConfig.default_for(spec)

    compiled = _launcher().compile(_copy_strided_cls(), spec, config)

    # The Src buffer must be the FULL storage (not the view), because
    # the kernel indexes into it with computed offsets.
    # We create a "fake" QuarkTensor that points to the full storage.

    src_full = QuarkTensor(
        t._storage.retain() and t._storage,
        (padded_N,),
        _contiguous_strides((padded_N,)),
        0,  # offset handled in the kernel via spec.offset
        t.dtype,
    )
    # Make sure storage covers padded_N elements.
    # If not, we'll read garbage for the padding — that's fine since
    # we trim the output below.

    out = QuarkTensor.zeros(padded_N, dtype=t.dtype)
    buffers = [src_full, out]
    compiled.launch(buffers=buffers)

    if padded_N != N:
        out = out[:N]
    return out.reshape(*t.shape) if out.shape != t.shape else out


def copy_strided_into(dst: QuarkTensor, src: QuarkTensor) -> None:
    """Copy src into dst (which may be a strided view)."""
    from quark.runtime.tensor import PC_BYTES

    if dst.is_contiguous() and src.is_contiguous():
        n = dst.numel()
        elem = PC_BYTES[dst.dtype]
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memcpy_dtod(dst.data_ptr(), src.data_ptr(), n * elem)
        return

    if dst.is_contiguous():
        src_c = src.contiguous()
        n = dst.numel()
        elem = PC_BYTES[dst.dtype]
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memcpy_dtod(dst.data_ptr(), src_c.data_ptr(), n * elem)
        return

    raise NotImplementedError("copy_strided_into: strided dst not yet supported")


def fill_scalar(t: QuarkTensor, value: float) -> None:
    """Fill a tensor view with a scalar value."""
    from quark.runtime.tensor import PC_BYTES

    if value == 0.0 and t.is_contiguous():
        n = t.numel()
        elem = PC_BYTES[t.dtype]
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memset_d8(t.data_ptr(), 0, n * elem)
        return

    # General fill: create a scalar-filled tensor, copy into the view.
    filled = _fill_tensor(t.numel(), value, t.dtype)
    copy_strided_into(t, filled.reshape(*t.shape) if filled.shape != t.shape else filled)


# ── Reduction (host-side for now) ─────────────────────────────


def reduce_op(op: str, t: QuarkTensor, dim: int | None) -> QuarkTensor:
    """Sum or max reduction. Host-side via array.array for speed."""
    import array as _array

    from quark.runtime.tensor import QuarkTensor

    t_f32 = cast(t, "f32") if t.dtype != "f32" else t
    t_c = t_f32.contiguous()
    raw = t_c.to_bytes()

    vals = _array.array("f")
    vals.frombytes(raw)

    if dim is None:
        result = math.fsum(vals) if op == "sum" else max(vals)
        out_buf = _array.array("f", [result])
        out = QuarkTensor.from_bytes(out_buf.tobytes(), (1,), "f32")
        return cast(out, t.dtype) if t.dtype != "f32" else out

    shape = list(t.shape)
    d = dim if dim >= 0 else len(shape) + dim
    outer = math.prod(shape[:d])
    inner = shape[d]
    trailing = math.prod(shape[d + 1 :])

    out_list = _array.array("f")
    for o in range(outer):
        for tr in range(trailing):
            if op == "sum":
                s = math.fsum(vals[o * inner * trailing + i * trailing + tr] for i in range(inner))
            else:
                s = max(vals[o * inner * trailing + i * trailing + tr] for i in range(inner))
            out_list.append(s)

    out_shape = tuple(shape[:d] + shape[d + 1 :]) or (1,)
    out = QuarkTensor.from_bytes(out_list.tobytes(), out_shape, "f32")
    return cast(out, t.dtype) if t.dtype != "f32" else out


# ── Matmul (host-side for small tensors) ──────────────────────


def matmul(a: QuarkTensor, b: QuarkTensor) -> QuarkTensor:
    """Matmul via pcf.gemm — the real tiled GEMM kernel, not Python loops."""
    if a.ndim != 2 or b.ndim != 2:
        raise NotImplementedError("matmul: only 2D x 2D supported")

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"matmul: inner dim mismatch {K} vs {K2}")

    import quark.functional as pcf

    return pcf.gemm(a, b)


# ── Cat ───────────────────────────────────────────────────────


def cat(tensors: list[QuarkTensor], dim: int = 0) -> QuarkTensor:
    """Concatenate tensors along a dimension."""
    from quark.runtime.tensor import PC_BYTES, QuarkTensor

    if not tensors:
        raise ValueError("cat: empty tensor list")
    if len(tensors) == 1:
        return tensors[0].contiguous()

    dtype = tensors[0].dtype
    ndim = tensors[0].ndim
    d = dim if dim >= 0 else ndim + dim

    for i, t in enumerate(tensors[1:], 1):
        if t.dtype != dtype:
            raise ValueError(f"cat: dtype mismatch at index {i}")
        if t.ndim != ndim:
            raise ValueError(f"cat: ndim mismatch at index {i}")
        for j in range(ndim):
            if j != d and t.shape[j] != tensors[0].shape[j]:
                raise ValueError(f"cat: shape mismatch at dim {j}")

    out_shape = list(tensors[0].shape)
    out_shape[d] = sum(t.shape[d] for t in tensors)
    out = QuarkTensor.empty(*out_shape, dtype=dtype)

    elem = PC_BYTES[dtype]
    if d == 0:
        # Simple case: concat along dim 0 = sequential memcpy.
        offset = 0
        for t in tensors:
            tc = t.contiguous()
            nbytes = tc.numel() * elem
            if nbytes > 0:
                from quark.runtime.cuda import CudaRuntime

                CudaRuntime.instance().memcpy_dtod(
                    out._storage.ptr + offset * elem,
                    tc.data_ptr(),
                    nbytes,
                )
                offset += tc.numel()
    else:
        # General case: slice assignment.
        offset = 0
        for t in tensors:
            start = offset
            end = offset + t.shape[d]
            idx = [slice(None)] * ndim
            idx[d] = slice(start, end)
            out[tuple(idx)] = t.contiguous()
            offset = end

    return out


# ── Randn ─────────────────────────────────────────────────────


def randn(shape: tuple[int, ...], dtype: str) -> QuarkTensor:
    """Random normal tensor. Uses ``random.gauss`` (C-implemented) with
    ``array.array`` for fast bulk packing."""
    import array as _array

    from quark.runtime.tensor import QuarkTensor

    n = math.prod(shape) if shape else 1
    # random.gauss is implemented in C — much faster than manual Box-Muller.
    buf = _array.array("f", (random.gauss(0.0, 1.0) for _ in range(n)))

    t = QuarkTensor.from_bytes(buf.tobytes(), shape, "f32")
    if dtype != "f32":
        t = cast(t, dtype)
    return t


# ── Helpers ───────────────────────────────────────────────────


def _pad_tensor(t: QuarkTensor, padded_N: int) -> QuarkTensor:
    """Pad a 1-D contiguous tensor to padded_N with zeros."""
    from quark.runtime.tensor import PC_BYTES, QuarkTensor

    if t.numel() == padded_N:
        return t
    out = QuarkTensor.zeros(padded_N, dtype=t.dtype)
    elem = PC_BYTES[t.dtype]
    from quark.runtime.cuda import CudaRuntime

    CudaRuntime.instance().memcpy_dtod(out.data_ptr(), t.data_ptr(), t.numel() * elem)
    return out


def _fill_tensor(n: int, value: float, dtype: str) -> QuarkTensor:
    """Create an n-element tensor filled with ``value``.

    Uses cuMemsetD16/D32 for a single driver call — no Python-side
    byte packing of millions of elements.
    """
    from quark.runtime.tensor import QuarkTensor

    if value == 0.0:
        return QuarkTensor.zeros(n, dtype=dtype)

    t = QuarkTensor.empty(n, dtype=dtype)
    from quark.runtime.cuda import CudaRuntime

    rt = CudaRuntime.instance()

    if dtype == "f32":
        # Reinterpret f32 as u32 for memset.
        bits = _struct.unpack("<I", _struct.pack("<f", value))[0]
        rt.memset_d32(t.data_ptr(), bits, n)
    elif dtype == "bf16":
        # Truncate f32 → bf16: top 16 bits of f32.
        bits = _struct.unpack("<I", _struct.pack("<f", value))[0]
        bf16_bits = (bits >> 16) & 0xFFFF
        rt.memset_d16(t.data_ptr(), bf16_bits, n)
    elif dtype == "f16":
        # Pack one f16 value and memset.
        raw = _struct.pack("<e", value)
        bits = _struct.unpack("<H", raw)[0]
        rt.memset_d16(t.data_ptr(), bits, n)
    elif dtype in ("s32", "u32"):
        rt.memset_d32(t.data_ptr(), int(value) & 0xFFFFFFFF, n)
    elif dtype in ("s8", "u8"):
        rt.memset_d8(t.data_ptr(), int(value) & 0xFF, n)
    else:
        # Fallback: pack one element, memset with appropriate width.
        rt.memset_d8(t.data_ptr(), int(value) & 0xFF, n)

    return t
