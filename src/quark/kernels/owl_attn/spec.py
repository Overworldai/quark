"""OwlAttnSpec — flash-attention with ortho-RoPE on Q (in registers) and
sparse segment-based KV iteration (matches kv_cache_update output).

Differences vs ``AttnSpec``:
  - ``capacity`` replaces ``kv_len`` (= num_buckets*tpf + tpf — ring + tail
    layout produced by the kv_cache_update kernel).
  - ``H_spatial`` / ``W_spatial``: spatial grid; tpf = H * W = seq_len.
  - ``num_buckets`` / ``pinned_dilation``: per-layer, must match the
    associated kv_cache_update spec.
  - ``max_segments``: max ring+tail segment count from the cache update.
  - ``kv_dtype``: K_cache / Vt_cache element dtype (bf16 or e4m3).
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class OwlAttnSpec(KernelSpec):
    B: int
    n_kv_heads: int
    gqa_ratio: int
    H_spatial: int
    W_spatial: int
    num_buckets: int
    pinned_dilation: int
    Dh: int = 64
    a_dtype: DType = DType.BF16  # Q gmem dtype
    kv_dtype: DType = DType.BF16  # K_cache / Vt_cache gmem dtype
    out_dtype: DType = DType.BF16  # output gmem dtype
    # MMA / smem compute dtype. Defaults to a_dtype (no cast). Set to
    # E4M3 to drive both GEMMs in fp8 even when Q comes in as bf16 — the
    # cast happens during Q-RoPE (packed_convert) and during K/V load
    # (TileLoad with cast=). Mirrors the GEMM kernel's compute_dtype axis.
    compute_dtype: DType | None = None
    max_segments: int = 3
    # When True, Q input is packed QKV [B*tpf, qkv_dim] and output is
    # [B*tpf, n_q_heads*Dh]. The kernel indexes into the packed Q
    # columns using per-head column offsets rather than head-strided rows.
    packed_qkv: bool = False
    # Quilt attention — see KVCacheUpdateSpec for the full description.
    # owl_attn doesn't read these directly, but they shrink ``L`` /
    # ``capacity`` so the cache shape, segment lengths, and the iter
    # loop count all line up with the matching kv_cache_update spec.
    quilt_factor: int = 1
    quilt_offset: int = 0
    # RoPE cos/sin are computed inline from (h, w, frame_t, Dh).
    # No precomputed tables — frame_t is a runtime device buffer.

    def __post_init__(self):
        for field in ("a_dtype", "kv_dtype", "out_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))
        if (
            self.compute_dtype is not None
            and isinstance(self.compute_dtype, str)
            and not isinstance(self.compute_dtype, DType)
        ):
            object.__setattr__(self, "compute_dtype", DType(self.compute_dtype))

    @property
    def compute_dtype_resolved(self) -> DType:
        return self.compute_dtype if self.compute_dtype is not None else self.a_dtype

    @property
    def n_q_heads(self) -> int:
        return self.n_kv_heads * self.gqa_ratio

    @property
    def tpf(self) -> int:
        return self.H_spatial * self.W_spatial

    @property
    def tpf_cached(self) -> int:
        """Per-frame token count stored in K/V cache (after quilt)."""
        return self.tpf // self.quilt_factor

    @property
    def seq_len(self) -> int:
        return self.tpf  # one frame per call

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
        """Row count of the Q/QKV input tensor."""
        return (self.B * self.tpf) if self.packed_qkv else self.total_q

    @property
    def q_cols(self) -> int:
        """Col count of the Q/QKV input tensor."""
        return self.qkv_dim if self.packed_qkv else self.Dh

    @property
    def out_rows(self) -> int:
        return (self.B * self.tpf) if self.packed_qkv else self.total_q

    @property
    def out_cols(self) -> int:
        return (self.n_q_heads * self.Dh) if self.packed_qkv else self.Dh
