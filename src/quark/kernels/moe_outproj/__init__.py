"""MoE out-projection kernel — output = Σ_e weight_e * (h_e @ W_out_e.T).

Migrated to the new IR Builder + L2 blocks.
"""

from quark.kernels.moe_outproj.config import MoeOutprojConfig
from quark.kernels.moe_outproj.kernel import MoeOutprojKernel
from quark.kernels.moe_outproj.spec import MoeOutprojSpec

__all__ = ["MoeOutprojConfig", "MoeOutprojKernel", "MoeOutprojSpec"]
