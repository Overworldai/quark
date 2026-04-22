"""AdaGateResidualSpec — fused ``out = x + gate_bmcast * y``.

Matches world_engine's ``x + ada_gate(y, gate)`` (see
``world_engine/src/model/nn.py::ada_gate``): the gate is applied
directly, no sigmoid. The sigmoid on the gate slice — if any —
happens upstream in the CondHead projection pipeline, not here.

Shapes:
    x, y:  [G*M, D]   (residual stream and attn/MLP output)
    gate:  [G, D]     (per-group gate from the CondHead projection;
                        broadcast over the M rows of its group)
    out:   [G*M, D]

Per-element:
    out[b, d] = x[b, d] + gate[b // M, d] * y[b, d]

Fires twice per transformer block (post-attn residual, post-MLP
residual) in waypoint-1.5.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class AdaGateResidualSpec(KernelSpec):
    G: int
    M: int
    D: int
    dtype: DType = DType.BF16

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"AdaGateResidualSpec: dtype {self.dtype!r} not in {_VALID_DTYPES}")
        for name in ("G", "M", "D"):
            if getattr(self, name) <= 0:
                raise ValueError(f"AdaGateResidualSpec: {name} must be positive")

    @property
    def B(self) -> int:
        return self.G * self.M
