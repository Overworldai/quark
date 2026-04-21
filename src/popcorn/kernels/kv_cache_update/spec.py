"""KVCacheUpdateSpec — problem definition for ortho-RoPE + ring KV cache write.

Models one transformer-layer's per-step KV cache write. Each call:
  - takes one frame's K/V (shape [B, n_kv_heads, tpf, Dh])
  - applies ortho-RoPE to K (in fp32; Q is roped separately by the consumer)
  - writes K, V_t into the layer's ring cache + a tail "current frame" slot

Spec is per-layer because ``pinned_dilation`` (1=dense, 8=dilated) is baked
into the cache geometry. tpf == H_spatial * W_spatial == one frame.

Capacity layout (compact, no wasted slots):
    K_cache[B, n_kv_heads, capacity, Dh]
    V_cache_t[B, n_kv_heads, Dh, capacity]
    capacity = num_buckets * tpf + tpf       (ring + tail current)
"""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec


@dataclass(frozen=True)
class KVCacheUpdateSpec(KernelSpec):
    B: int
    n_kv_heads: int
    Dh: int
    H_spatial: int
    W_spatial: int
    num_buckets: int
    pinned_dilation: int
    in_dtype: DType = DType.BF16
    kv_dtype: DType = DType.BF16
    max_segments: int = 3
    # When True, K and V are read from a packed QKV tensor [B*tpf, qkv_dim]
    # using column offsets (k_col_offset, v_col_offset) instead of from
    # separate [B*Hk*tpf, Dh] tensors. The packed tensor replaces K; V
    # is unused.
    packed_qkv: bool = False
    n_q_heads: int = 0  # only used when packed_qkv=True (for column offsets)
    # Number of frames in the cos/sin RoPE table. 1 = pre-sliced (legacy),
    # >1 = full table (the kernel indexes via frame_t * tpf internally).
    rope_n_frames: int = 1

    def __post_init__(self):
        for field in ("in_dtype", "kv_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))

    @property
    def tpf(self) -> int:
        return self.H_spatial * self.W_spatial

    @property
    def L(self) -> int:
        return self.num_buckets * self.tpf

    @property
    def capacity(self) -> int:
        return self.L + self.tpf

    @property
    def qkv_dim(self) -> int:
        return (self.n_q_heads + 2 * self.n_kv_heads) * self.Dh

    @property
    def k_col_offset(self) -> int:
        return self.n_q_heads * self.Dh

    @property
    def v_col_offset(self) -> int:
        return (self.n_q_heads + self.n_kv_heads) * self.Dh

    @property
    def k_input_rows(self) -> int:
        """Row count of the K input tensor (or QKV when packed)."""
        return (self.B * self.tpf) if self.packed_qkv else (self.B * self.n_kv_heads * self.tpf)

    @property
    def k_input_cols(self) -> int:
        return self.qkv_dim if self.packed_qkv else self.Dh
