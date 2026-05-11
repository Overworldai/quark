"""MoE in-projection kernel — h = SiLU(X[tok_ids] @ W_in[expert].T).

Migrated to the new IR Builder + L2 blocks from the legacy
`quark.kernels.moe.inproj` module.
"""

from quark.kernels.moe_inproj.config import MoeInprojConfig
from quark.kernels.moe_inproj.kernel import MoeInprojKernel
from quark.kernels.moe_inproj.spec import MoeInprojSpec

__all__ = ["MoeInprojConfig", "MoeInprojKernel", "MoeInprojSpec"]
