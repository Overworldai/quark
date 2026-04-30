"""MoE purely-correct router kernel.

Each token routes to its actual top-K experts (no capacity bound, no
substitution). Buffer is sized for the worst-case BM-padded layout
(``M*K + E*(BM-1)``); chunks past the actual active region get
``expert=-1`` so the inproj/outproj kernels can sentinel-skip them.
"""

from quark.kernels.moe_router_correct.config import MoeRouterCorrectConfig
from quark.kernels.moe_router_correct.kernel import MoeRouterCorrectKernel
from quark.kernels.moe_router_correct.spec import MoeRouterCorrectSpec

__all__ = ["MoeRouterCorrectConfig", "MoeRouterCorrectKernel", "MoeRouterCorrectSpec"]
