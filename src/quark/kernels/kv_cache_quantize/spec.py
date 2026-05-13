"""KVQuantizeSpec — per-token symmetric int8 quantization of a
row-major [num_tokens, Dh] bf16 tensor (e.g. K_cache).

Phase 2.2 will add a transposed (Vt) path with column-wise reduce."""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class KVQuantizeSpec(KernelSpec):
    """Per-token quant of a bf16 tensor.

    Two layouts via ``transposed``:
      * ``False`` (default, K-cache style):
        in:  [num_tokens, Dh] bf16
        out: [num_tokens, Dh] s8 + [num_tokens] f32 scales
        Each token = one ROW; reduce is row-wise.
      * ``True`` (Vt-cache style):
        in:  [n_heads, Dh, num_tokens] bf16 (flattened first dim
             = n_groups × Dh = ``Dh × n_kv_heads × B``)
        out: same shape s8 + [n_heads × num_tokens] f32 scales
        Each token = one COL within a (head, batch) Dh-row block;
        reduce is column-wise across Dh rows.

    For the transposed layout, the spec carries the flattened
    ``num_heads`` dim (= ``n_kv_heads × B``) explicitly so the kernel
    can index the right Dh-rows for each (head, batch, token) WG.
    """

    num_tokens: int
    Dh: int = 64
    in_dtype: DType = DType.BF16
    scale_dtype: DType = DType.F32
    transposed: bool = False
    # When ``transposed=True``, the input first dim is
    # ``n_heads * Dh`` and the output scales have shape
    # ``(n_heads * num_tokens,)``.
    n_heads: int = 1

    def __post_init__(self):
        if self.num_tokens <= 0:
            raise ValueError(f"KVQuantizeSpec: num_tokens must be positive (got {self.num_tokens})")
        if self.Dh <= 0:
            raise ValueError(f"KVQuantizeSpec: Dh must be positive (got {self.Dh})")
        if self.in_dtype not in (DType.BF16, DType.F16):
            raise ValueError(f"KVQuantizeSpec: in_dtype {self.in_dtype!r} must be BF16 or F16")
        if self.transposed and self.n_heads <= 0:
            raise ValueError(
                f"KVQuantizeSpec(transposed=True): n_heads must be positive (got {self.n_heads})"
            )
