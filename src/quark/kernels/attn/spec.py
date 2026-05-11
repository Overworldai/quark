"""AttnSpec — problem definition for flash attention with GQA."""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class AttnSpec(KernelSpec):
    B: int  # batch size
    n_kv_heads: int  # number of KV heads
    gqa_ratio: int  # Q heads per KV head (n_q_heads = n_kv_heads * gqa_ratio)
    seq_len: int  # query sequence length (T)
    kv_len: int  # key/value sequence length (capacity)
    Dh: int = 64  # head dimension
    a_dtype: DType = DType.BF16
    b_dtype: DType = DType.BF16

    def __post_init__(self):
        for field in ("a_dtype", "b_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))

    @property
    def n_q_heads(self) -> int:
        return self.n_kv_heads * self.gqa_ratio

    @property
    def total_q(self) -> int:
        return self.B * self.n_q_heads * self.seq_len
