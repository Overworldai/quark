"""``quark.functional.gemm`` — torch/MLX-native C = A @ B^T.

Signature:

    C = pcf.gemm(A, B, *, out_dtype=None, compute_dtype=None, b_shuffled=False)

A: [M, K], B: [N, K] (weight, stored transposed so K is fast axis).
Returns C: [M, N] in ``out_dtype`` (defaults to A's dtype).
"""

from __future__ import annotations

import os
import sys

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_GemmCls = None
_IS_METAL = sys.platform == "darwin"

# Dtype combos where ``cublasLtMatmul`` gives us scale-free row-major
# A × B^T directly. fp8 A+B requires both operands fp8 (cublasLt's fp8
# matmul doesn't do mixed half/fp8), so Linear pre-casts x to e4m3 when
# the weight is fp8 to land here. Mixed input combos + pre-shuffled B +
# fused activations stay on the custom kernel.
_CUBLAS_COMBOS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("bf16", "bf16", "bf16"),
        ("bf16", "bf16", "f32"),
        ("f16", "f16", "f16"),
        ("f16", "f16", "f32"),
        ("e4m3", "e4m3", "bf16"),
        ("e4m3", "e4m3", "f16"),
        ("e4m3", "e4m3", "e4m3"),
        ("e4m3", "e4m3", "f32"),
    }
)


def _cls():
    """Lazy import so module import order doesn't matter."""
    global _GemmCls
    if _GemmCls is None:
        _GemmCls = get("gemm")
    return _GemmCls


def _dtype_str(t) -> str:
    dt = t.dtype
    return dt if isinstance(dt, str) else str(dt)


def _try_cublas(A, B, *, out_dtype, compute_dtype, b_shuffled, activation, bias, out):
    """Short-circuit to ``cublasLtMatmul`` when the combo supports it.

    Returns the output tensor on a hit, or ``None`` to tell the caller
    to fall through to the custom-kernel path. Kept deliberately
    conservative: anything the gate can't prove cuBLAS handles in one
    call (scales, fused epilogues, weight pre-shuffle, non-standard
    dtypes) falls back immediately.
    """
    if _IS_METAL:
        return None
    if os.environ.get("QUARK_DISABLE_CUBLAS") == "1":
        return None
    if b_shuffled or activation is not None:
        return None

    from quark.runtime.tensor import QuarkTensor

    if not isinstance(A, QuarkTensor) or not isinstance(B, QuarkTensor):
        return None
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        return None

    # cublasLt's BIAS epilogue requires a 1D [N] vector.
    if bias is not None:
        if not isinstance(bias, QuarkTensor):
            return None
        if bias.ndim != 1 or int(bias.shape[0]) != int(B.shape[0]):
            return None

    a_dt = _dtype_str(A)
    b_dt = _dtype_str(B)

    # compute_dtype: custom kernel uses it to down-cast A during the
    # smem load (bf16 → e4m3 for mixed-dtype fp8 MMA). cuBLAS has no
    # equivalent zero-cost path without scales, so any explicit
    # compute_dtype that differs from a_dt forces fallback.
    if compute_dtype is not None and str(compute_dtype) != a_dt:
        return None

    if out is not None:
        c_dt = _dtype_str(out)
    else:
        c_dt = str(out_dtype) if out_dtype is not None else a_dt

    if (a_dt, b_dt, c_dt) not in _CUBLAS_COMBOS:
        return None

    from quark.runtime.cublas import CublasRuntime

    if not CublasRuntime.is_available():
        return None

    M = int(A.shape[0])
    K = int(A.shape[1])
    N = int(B.shape[0])

    if out is None:
        out = QuarkTensor.empty(M, N, dtype=c_dt)
    elif tuple(out.shape) != (M, N):
        return None

    from quark.graph import active_stream

    stream = active_stream() or 0
    bias_ptr = bias.data_ptr() if bias is not None else 0
    bias_dt = _dtype_str(bias) if bias is not None else None
    if os.environ.get("QUARK_CUBLAS_VERBOSE") == "1":
        tag = f" bias={bias_dt}" if bias is not None else ""
        print(
            f"[cublas] matmul M={M} N={N} K={K} a={a_dt} b={b_dt} c={c_dt}{tag} stream={stream}",
            flush=True,
        )
    CublasRuntime.instance().matmul(
        a_ptr=A.data_ptr(),
        b_ptr=B.data_ptr(),
        c_ptr=out.data_ptr(),
        M=M,
        N=N,
        K=K,
        a_dtype=a_dt,
        b_dtype=b_dt,
        c_dtype=c_dt,
        bias_ptr=bias_ptr,
        bias_dtype=bias_dt,
        stream=stream,
    )
    return out


def _gemm_impl(
    A,
    B,
    *,
    out_dtype=None,
    compute_dtype=None,
    b_shuffled=False,
    activation=None,
    bias=None,
    out=None,
):
    cublas_out = _try_cublas(
        A,
        B,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        bias=bias,
        out=out,
    )
    if cublas_out is not None:
        return cublas_out

    cls = _cls()
    has_bias = bias is not None
    spec = cls.spec_from_tensors(
        A,
        B,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        has_bias=has_bias,
    )
    provided = {"A": A, "B": B}
    if has_bias:
        provided["Bias"] = bias
    if out is not None:
        provided["Out"] = out
    auto_alloc_names: tuple[str, ...] = ()
    if not has_bias:
        auto_alloc_names = auto_alloc_names + ("Bias",)
    if out is None:
        auto_alloc_names = auto_alloc_names + ("Out",)
    result = call_with_bindings(
        cls,
        spec,
        provided=provided,
        auto_alloc=auto_alloc_names,
        like=A,
    )
    return result["Out"]


def gemm(
    A,
    B,
    out_dtype=None,
    compute_dtype=None,
    b_shuffled=False,
    activation=None,
    bias=None,
    out=None,
):
    """Compute ``C = A @ B.T``.

    ``out``: optional pre-allocated output buffer to write into. When
    provided, skips the auto_alloc path — lets callers (e.g. Linear
    layers) reuse a cached buffer keyed by M and avoid per-call allocs.
    """
    return _gemm_impl(
        A,
        B,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        bias=bias,
        out=out,
    )


gemm.autotune = make_autotune(_gemm_impl, _cls)  # ty: ignore[unresolved-attribute]
