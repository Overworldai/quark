"""MoE reduce kernel — gather-and-sum over per-token expert partials.

``out[m, d] = sum over k of slot_weights[idx] * partials[idx, d]``
where ``idx = token_slot_table[m, k]``.

Pairs with the modified ``moe_outproj`` (writes per-slot bf16 partials)
and ``moe_router_correct`` (emits ``token_slot_table``). Replaces the
non-deterministic atomic scatter-add the previous outproj epilogue
used.
"""

from quark.kernels.moe_reduce.config import MoeReduceConfig
from quark.kernels.moe_reduce.kernel import MoeReduceKernel
from quark.kernels.moe_reduce.spec import MoeReduceSpec

__all__ = ["MoeReduceConfig", "MoeReduceKernel", "MoeReduceSpec"]
