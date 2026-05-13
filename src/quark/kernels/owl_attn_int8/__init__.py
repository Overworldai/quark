"""OwlAttn int8 — SageAttention-style int8 attention on Intel Xe2.

Mirrors the OwlAttn kernel API but uses the m8n16k32 s8/s8/s32
cooperative_matrix path for both QK and AV. Consumes the s8 KV
cache + per-token scale tensors produced by
:class:`quark.kernels.kv_cache_quantize.KVQuantizeKernel`.

Validated quant algorithm at GLSL level
(``/tmp/run_int8_gemm_scales.py`` for the GEMM half,
``/tmp/run_kv_quantize_probe.py`` for the KV-quant half).
"""

from quark.kernels.owl_attn_int8.config import OwlAttnIntConfig
from quark.kernels.owl_attn_int8.kernel import OwlAttnIntKernel
from quark.kernels.owl_attn_int8.spec import OwlAttnIntSpec

__all__ = ["OwlAttnIntConfig", "OwlAttnIntKernel", "OwlAttnIntSpec"]
