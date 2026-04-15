"""Backend-agnostic reference for the MoE in-projection.

Uses ``popcorn.backend.PT`` for every op so the reference runs on both
torch (CUDA/CPU) and mlx (Metal) tensors without per-caller branching.
"""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def moe_inproj_reference(
    X,
    W_in,
    token_ids,
    work_list,
    *,
    BM: int,
    H: int,
    total_slots: int,
    out_dtype: DType | str = DType.BF16,
    compute_dtype: DType | str | None = None,
):
    """``h = SiLU(X[token_ids] @ W_in[expert].T)``, accumulated in f32.

    Shapes:
      * ``X``: ``[M, D]``
      * ``W_in``: ``[n_experts * H, D]`` (or ``[n_experts, H, D]``)
      * ``token_ids``: ``[total_slots]`` int — per-slot row index in X
      * ``work_list``: flat ``[2 * n_work_items]`` int, ``(grp_start, expert)`` pairs
    Returns ``[total_slots, H]`` in ``out_dtype``.

    When ``compute_dtype`` is set and differs from X/W dtype the
    corresponding tensor is roundtripped through that dtype first so
    the cos-sim gate sees the same precision loss the kernel's on-load
    cast applies.
    """
    out_dt = DType(out_dtype) if isinstance(out_dtype, str) else out_dtype

    # Simulate the kernel's compute-dtype cast.
    if compute_dtype is not None:
        compute_dt = DType(compute_dtype) if isinstance(compute_dtype, str) else compute_dtype
        ct = compute_dt.backend
        if X.dtype != ct:
            X = PT.astype(X, ct)
        if W_in.dtype != ct:
            W_in = PT.astype(W_in, ct)

    D = X.shape[-1]
    n_experts = W_in.shape[0] if W_in.ndim == 3 else W_in.shape[0] // H

    W_3d = PT.astype(W_in, PT.float32).reshape(n_experts, H, D)
    X_f32 = PT.astype(X, PT.float32)

    # work_list and token_ids to Python for the per-group dispatch —
    # tiny lists, cheap to materialise once per reference call.
    wl = PT.to_cpu_numpy(work_list).reshape(-1, 2).tolist()
    tok = PT.to_cpu_numpy(token_ids).tolist()

    h = PT.zeros(total_slots, H, dtype=PT.float32)
    for grp_start, expert in wl:
        gs = int(grp_start)
        e = int(expert)
        tids = [int(tok[gs + i]) for i in range(BM)]
        # Fancy-index rows of X via a list — works on both backends.
        x_g = X_f32[PT.tensor(tids, dtype=PT.int32)]
        block = PT.matmul(x_g, PT.transpose(W_3d[e]))
        h = PT.set_slice(h, axis=0, start=gs, stop=gs + BM, src=block)

    # SiLU(x) = x · sigmoid(x) = x / (1 + exp(-x))
    h = h / (1.0 + PT.exp(-h))

    return PT.astype(h, out_dt.backend)


def moe_inproj_reference_for_spec(kernel, X, W_in, token_ids, work_list):
    s = kernel.spec
    return moe_inproj_reference(
        X,
        W_in,
        token_ids,
        work_list,
        BM=kernel.config.BM,
        H=s.H,
        total_slots=s.total_slots,
        out_dtype=s.out_dtype,
        compute_dtype=s.compute_dtype_resolved,
    )
