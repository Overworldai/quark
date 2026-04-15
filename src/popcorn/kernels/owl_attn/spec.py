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

from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec


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
    def seq_len(self) -> int:
        return self.tpf  # one frame per call

    @property
    def L(self) -> int:
        return self.num_buckets * self.tpf

    @property
    def capacity(self) -> int:
        return self.L + self.tpf

    @property
    def total_q(self) -> int:
        return self.B * self.n_q_heads * self.seq_len
