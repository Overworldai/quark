"""owl_attn — segment-sparse flash attention with ortho-RoPE on Q.

Consumer of `kv_cache_update`'s K_cache + Vt_cache + segments. Per-call
inputs include the cos/sin freq slice for this frame so Q can be
rotated in registers before GEMM1.
"""

from popcorn.kernels.owl_attn.kernel import OwlAttnKernel  # noqa: F401
