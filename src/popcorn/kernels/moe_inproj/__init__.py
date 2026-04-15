"""MoE in-projection kernel — h = SiLU(X[tok_ids] @ W_in[expert].T).

Migrated to the new IR Builder + L2 blocks from the legacy
`popcorn.kernels.moe.inproj` module.
"""

from popcorn.kernels.moe_inproj.config import MoeInprojConfig
from popcorn.kernels.moe_inproj.kernel import MoeInprojKernel
from popcorn.kernels.moe_inproj.spec import MoeInprojSpec

__all__ = ["MoeInprojConfig", "MoeInprojKernel", "MoeInprojSpec"]
