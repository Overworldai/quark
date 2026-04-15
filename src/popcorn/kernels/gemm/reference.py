"""Backend-agnostic reference for the universal GEMM.

``C[M, N] = A[M, K] @ B^T[N, K]^T``

Accumulates in f32, casts to the requested output dtype at the end.
Uses ``popcorn.backend.PT`` for every tensor op so one reference runs
on both torch (CUDA/CPU) and mlx (Metal).
"""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def gemm_reference(
    A, B, *, out_dtype: DType | str = DType.BF16, compute_dtype: DType | str | None = None
):
    """``C = A @ B^T``, accumulated in f32, cast to ``out_dtype``.

    A: ``[M, K]`` (or ``[*, K]`` — flattened to 2D first).
    B: ``[N, K]`` (the weight matrix).
    Returns ``[M, N]`` in the requested output dtype.

    When ``compute_dtype`` is set and differs from the source A/B
    dtype, the tensor is roundtripped through that dtype first — the
    same on-load cast the kernel applies, so the cos-sim gate sees
    the same precision loss on both sides.
    """
    if A.ndim > 2:
        A = A.reshape(-1, A.shape[-1])
    K = A.shape[-1]
    assert B.shape[1] == K, f"K mismatch: A has K={K}, B has K={B.shape[1]}"

    out_dt = DType(out_dtype) if isinstance(out_dtype, str) else out_dtype

    # Simulate the compute-dtype cast.
    if compute_dtype is not None:
        compute_dt = DType(compute_dtype) if isinstance(compute_dtype, str) else compute_dtype
        ct = compute_dt.backend
        if A.dtype != ct:
            A = PT.astype(A, ct)
        if B.dtype != ct:
            B = PT.astype(B, ct)

    A_f32 = PT.astype(A, PT.float32)
    B_f32 = PT.astype(B, PT.float32)
    C_f32 = PT.matmul(A_f32, PT.transpose(B_f32))

    return PT.astype(C_f32, out_dt.backend)


def gemm_reference_for_spec(kernel, A, B):
    s = kernel.spec
    return gemm_reference(A, B, out_dtype=s.out_dtype, compute_dtype=s.compute_dtype_resolved)
