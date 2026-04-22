"""Universal GEMM kernel — C[M, N] = A[M, K] @ B^T[N, K]^T.

Parameterized over (a_dtype, b_dtype, acc_dtype, out_dtype). Uses
the new IR Builder + Launcher stack.
"""

from quark.kernels.gemm.config import GemmConfig
from quark.kernels.gemm.kernel import GemmKernel
from quark.kernels.gemm.spec import GemmSpec

__all__ = ["GemmConfig", "GemmKernel", "GemmSpec"]
