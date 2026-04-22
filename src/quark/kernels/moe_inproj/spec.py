"""MoeInprojSpec — problem definition for the MoE in-projection.

h[total_slots, H] = SiLU( X[token_ids, :] @ W_in[expert, :, :].T )

The kernel processes a work_list of (grp_start, expert) pairs,
each covering BM consecutive slots. Each slot's source row in X
is looked up via token_ids[slot_index].
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_AB = frozenset({DType.BF16, DType.F16, DType.E4M3, DType.E5M2})
_VALID_OUT = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class MoeInprojSpec(KernelSpec):
    M: int  # number of input tokens (rows of X)
    D: int  # input feature dim (K dimension of GEMM)
    H: int  # hidden feature dim (N dimension, expert output)
    n_experts: int
    top_k: int = 2  # slots per token
    a_dtype: DType = DType.BF16
    b_dtype: DType = DType.BF16
    out_dtype: DType = DType.BF16
    # Optional compute dtype — separate from a/b_dtype, like GemmSpec.
    # When set and differs from a source dtype, the tile loader casts
    # the element during the gmem→smem load so the smem and mma
    # fragments run in compute_dtype. Defaults to ``a_dtype`` (no cast).
    compute_dtype: DType | None = None

    def __post_init__(self):
        for field in ("a_dtype", "b_dtype", "out_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))
        if (
            self.compute_dtype is not None
            and isinstance(self.compute_dtype, str)
            and not isinstance(self.compute_dtype, DType)
        ):
            object.__setattr__(self, "compute_dtype", DType(self.compute_dtype))
        if self.a_dtype not in _VALID_AB:
            raise ValueError(f"MoeInprojSpec: a_dtype {self.a_dtype!r} not in {_VALID_AB}")
        if self.b_dtype not in _VALID_AB:
            raise ValueError(f"MoeInprojSpec: b_dtype {self.b_dtype!r} not in {_VALID_AB}")
        if self.out_dtype not in _VALID_OUT:
            raise ValueError(f"MoeInprojSpec: out_dtype {self.out_dtype!r} not in {_VALID_OUT}")
        if self.compute_dtype is not None and self.compute_dtype not in _VALID_AB:
            raise ValueError(
                f"MoeInprojSpec: compute_dtype {self.compute_dtype!r} not in {_VALID_AB}"
            )

    @property
    def total_slots(self) -> int:
        return self.M * self.top_k

    @property
    def compute_dtype_resolved(self) -> DType:
        """Effective compute dtype — source a_dtype when not set."""
        return self.compute_dtype if self.compute_dtype is not None else self.a_dtype
