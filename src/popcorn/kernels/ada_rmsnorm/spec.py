"""AdaRMSNormSpec — RMSNorm + ``(1 + scale) * y + bias`` epilogue.

World-engine's AdaLN composes as::

    y = rmsnorm(x) * (1 + scale) + bias

where ``x`` is a token stream ``[G, M, D]`` (G groups of M consecutive
tokens sharing one (scale, bias)) and ``scale, bias`` are ``[G, D]``.
The kernel flattens X/Out to ``[G*M, D]`` and broadcasts scale/bias
over M.

For waypoint-1.5 this fires twice per transformer block (pre-attn,
pre-MLP), plus once in the final out-norm, with M=tokens_per_frame=512
and G=1 per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec

_VALID_DTYPES = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class AdaRMSNormSpec(KernelSpec):
    G: int  # number of scale/bias groups
    M: int  # rows per group (broadcast factor)
    D: int  # feature dim
    dtype: DType = DType.BF16
    eps: float = 1.1920929e-07  # torch.finfo(torch.float32).eps
    activation: str | None = None

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"AdaRMSNormSpec: dtype {self.dtype!r} not in {_VALID_DTYPES}")
        for name in ("G", "M", "D"):
            if getattr(self, name) <= 0:
                raise ValueError(f"AdaRMSNormSpec: {name} must be positive")
        if self.eps <= 0:
            raise ValueError(f"AdaRMSNormSpec: eps must be positive; got {self.eps}")

    @property
    def B(self) -> int:
        """Total rows in X / Out."""
        return self.G * self.M
