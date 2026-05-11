"""HeadRMSNormSpec — per-head RMSNorm on column ranges of a packed QKV tensor.

Operates on a ``[M, D_full]`` buffer (the packed QKV output of a
single GEMM). Normalizes ``n_heads`` independent ``Dh``-wide chunks
starting at column ``col_offset``. Columns outside the normalized
range are copied unchanged to the output.

Grid: one block per ``(token, head_in_qkv)`` where head_in_qkv
spans all columns (Q heads normalized, K heads normalized, V heads
copied). Each block touches exactly ``Dh`` elements.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class HeadRMSNormSpec(KernelSpec):
    M: int  # tokens
    D_full: int  # total columns (qkv_dim, e.g. 4096)
    n_q_heads: int  # number of Q heads to normalize (first n_q*Dh cols)
    n_kv_heads: int  # number of K heads to normalize (next n_kv*Dh cols)
    Dh: int  # head dim (e.g. 64)
    dtype: DType = DType.BF16
    eps: float = 1.1920929e-07  # torch.finfo(torch.float32).eps

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        expected = (self.n_q_heads + 2 * self.n_kv_heads) * self.Dh
        if expected != self.D_full:
            raise ValueError(
                f"HeadRMSNormSpec: D_full={self.D_full} != "
                f"(n_q_heads + 2*n_kv_heads)*Dh = {expected}"
            )

    @property
    def total_heads(self) -> int:
        return self.n_q_heads + 2 * self.n_kv_heads

    @property
    def q_end(self) -> int:
        return self.n_q_heads * self.Dh

    @property
    def k_end(self) -> int:
        return (self.n_q_heads + self.n_kv_heads) * self.Dh
