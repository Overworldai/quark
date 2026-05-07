"""owl_attn — segment-sparse flash attention with ortho-RoPE on Q.

Consumer of `kv_cache_update`'s K_cache + Vt_cache + segments. Per-call
inputs include the cos/sin freq slice for this frame so Q can be
rotated in registers before GEMM1.

The package contains two implementations:
  * ``kernel.py`` — generic CUDA / Metal path via the high-level DSL
    (cp.async + simdgroup_matrix MMAs), used by the kernel framework.
  * ``nax.py`` — IR-emitted M5+ Metal NAX path (Apple matmul2d), used
    by ``functional/owl_attn.py``'s fast-path on supported devices.
"""

import sys

if sys.platform == "darwin":
    from quark.kernels.owl_attn import nax  # noqa: F401
from quark.kernels.owl_attn.kernel import OwlAttnKernel  # noqa: F401
