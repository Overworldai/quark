"""MoeReduceSpec — gather-and-sum the per-slot bf16 partials produced
by ``moe_outproj`` into a per-token output, weighted by the router's
``slot_weights``.

Layout:
  * ``partials[total_slots, D]``    bf16, output of moe_outproj
  * ``slot_weights[total_slots]``   f32, output of moe_router_correct
  * ``token_slot_table[M, top_k]``  s32, output of moe_router_correct
                                    — token-major inverse of the slot
                                    permutation
  * ``out[M, D]``                   bf16, residual-stream dtype

f32 accumulation in registers; cast to ``out_dtype`` on store.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_PARTIAL_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32})
_VALID_OUT_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32})


@dataclass(frozen=True)
class MoeReduceSpec(KernelSpec):
    M: int  # output tokens
    D: int  # output feature dim
    n_experts: int  # E
    capacity: int  # per-expert slot count; total_slots = E * capacity
    top_k: int
    partials_dtype: DType = DType.BF16
    out_dtype: DType = DType.BF16

    def __post_init__(self):
        for field in ("partials_dtype", "out_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))
        if self.partials_dtype not in _VALID_PARTIAL_DTYPES:
            raise ValueError(
                f"MoeReduceSpec: partials_dtype {self.partials_dtype!r} "
                f"not in {_VALID_PARTIAL_DTYPES}"
            )
        if self.out_dtype not in _VALID_OUT_DTYPES:
            raise ValueError(
                f"MoeReduceSpec: out_dtype {self.out_dtype!r} not in {_VALID_OUT_DTYPES}"
            )
        for name in ("M", "D", "n_experts", "capacity", "top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"MoeReduceSpec: {name} must be positive")

    @property
    def total_slots(self) -> int:
        return self.n_experts * self.capacity
