"""RMSNormSpec — plain (gainless) RMSNorm over the last dim.

``y[b, d] = x[b, d] * rsqrt(mean(x[b, :]²) + eps)``

The waypoint-1.5 model uses ``F.rms_norm(x)`` with no learnable gain
(Q/K pre-RoPE, cross-attn inputs, and the base of every AdaLN).
This kernel implements exactly that form. AdaLN's ``(1 + a) * y + b``
per-row modulation is a sibling kernel (``ada_rmsnorm_scale_bias``)
that builds on the same reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class RMSNormSpec(KernelSpec):
    """One row per block; D features reduced per row.

    ``B`` is the flattened product of all leading dims (``[*, D] →
    [B, D]``). ``dtype`` is the same for in/out; the reduction is
    always done in f32 regardless.
    """

    B: int
    D: int
    dtype: DType = DType.BF16
    eps: float = 1.1920929e-07  # torch.finfo(torch.float32).eps

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"RMSNormSpec: dtype {self.dtype!r} not in {_VALID_DTYPES}")
        if self.D <= 0 or self.B <= 0:
            raise ValueError(f"RMSNormSpec: B, D must be positive; got B={self.B}, D={self.D}")
        if self.eps <= 0:
            raise ValueError(f"RMSNormSpec: eps must be positive; got {self.eps}")
