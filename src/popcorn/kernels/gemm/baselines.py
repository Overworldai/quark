"""Platform-dispatched baselines for the GEMM kernel.

Split out from ``kernel.py`` so the kernel file stays focused on
IR emission; backend-specific baseline construction lives alongside
its kernel but in its own module."""

from __future__ import annotations

import torch

from popcorn.ir import DType
from popcorn.kernels.base import Baseline


def gemm_baselines(tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    A, B = tensors["A"], tensors["B"]
    if IS_METAL:
        import mlx.core as mx

        B_t = mx.transpose(B)

        def _mlx_matmul(a=A, b=B_t):
            mx.eval(a @ b)

        return [Baseline("mx.matmul[bf16]", _mlx_matmul)]

    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    A_bf = A if A.dtype == torch.bfloat16 else A.to(torch.bfloat16)
    B_bf = B if B.dtype == torch.bfloat16 else B.to(torch.bfloat16)
    out = [Baseline("torch.matmul[bf16]", lambda a=A_bf, b=B_bf: torch.matmul(a, b.T))]
    if A.dtype in fp8 and A.dtype == B.dtype and hasattr(torch, "_scaled_mm"):
        sa = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        sb = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        out.append(
            Baseline(
                f"torch._scaled_mm[{DType.from_backend(A.dtype)}]",
                lambda a=A, b=B, sa=sa, sb=sb: torch._scaled_mm(a, b.T, sa, sb),
            )
        )
    return out
