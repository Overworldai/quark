"""MoE router kernel — capacity-bounded top-K dispatch.

Consumes router logits ``[M, E]`` and emits expert-major-sorted
``token_ids`` / ``slot_weights`` directly consumable by
``moe_inproj`` and ``moe_outproj``. See ``spec.py`` for the routing
contract.
"""

from quark.kernels.moe_router.config import MoeRouterConfig
from quark.kernels.moe_router.kernel import MoeRouterKernel
from quark.kernels.moe_router.spec import MoeRouterSpec

__all__ = ["MoeRouterConfig", "MoeRouterKernel", "MoeRouterSpec"]
