"""OwlAttnIntSpec — int8 attention spec.

Same shape descriptors as ``OwlAttnSpec``. Compute dtype is fixed
to s8 (a/b operands) with s32 accumulators; output dtype is the
configurable ``out_dtype`` (bf16 by default).
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class OwlAttnIntSpec(KernelSpec):
    B: int
    n_kv_heads: int
    gqa_ratio: int
    H_spatial: int
    W_spatial: int
    num_buckets: int
    pinned_dilation: int
    Dh: int = 64
    a_dtype: DType = DType.BF16  # Q gmem dtype (Q is bf16, quantized on-load)
    out_dtype: DType = DType.BF16
    scale_dtype: DType = DType.F32
    max_segments: int = 3
    packed_qkv: bool = False
    quilt_factor: int = 1
    quilt_offset: int = 0

    def __post_init__(self):
        for field in ("a_dtype", "out_dtype", "scale_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))

    @property
    def n_q_heads(self) -> int:
        return self.n_kv_heads * self.gqa_ratio

    @property
    def tpf(self) -> int:
        return self.H_spatial * self.W_spatial

    @property
    def tpf_cached(self) -> int:
        return self.tpf // self.quilt_factor

    @property
    def seq_len(self) -> int:
        return self.tpf

    @property
    def L(self) -> int:
        return self.num_buckets * self.tpf_cached

    @property
    def capacity(self) -> int:
        return self.L + self.tpf_cached

    @property
    def total_q(self) -> int:
        return self.B * self.n_q_heads * self.seq_len

    @property
    def qkv_dim(self) -> int:
        return (self.n_q_heads + 2 * self.n_kv_heads) * self.Dh

    @property
    def q_rows(self) -> int:
        return (self.B * self.tpf) if self.packed_qkv else self.total_q

    @property
    def q_cols(self) -> int:
        return self.qkv_dim if self.packed_qkv else self.Dh

    @property
    def out_rows(self) -> int:
        return (self.B * self.tpf) if self.packed_qkv else self.total_q

    @property
    def out_cols(self) -> int:
        return (self.n_q_heads * self.Dh) if self.packed_qkv else self.Dh
