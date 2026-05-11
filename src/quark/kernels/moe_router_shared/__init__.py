"""MoE shared-experts router kernel.

Picks K experts globally by cumulative softmax preference across all
tokens, then routes every token to those same K experts. Emits the
same buffer layout as ``moe_router`` so ``moe_inproj`` / ``moe_outproj``
can be reused unchanged. See ``spec.py``.
"""

from quark.kernels.moe_router_shared.config import MoeRouterSharedConfig
from quark.kernels.moe_router_shared.kernel import MoeRouterSharedKernel
from quark.kernels.moe_router_shared.spec import MoeRouterSharedSpec

__all__ = ["MoeRouterSharedConfig", "MoeRouterSharedKernel", "MoeRouterSharedSpec"]
