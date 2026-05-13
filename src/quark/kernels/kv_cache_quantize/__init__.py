"""Per-token symmetric int8 quantization of bf16 KV caches.

Inputs: bf16 K_cache (row-major, [num_tokens, Dh]) and Vt_cache
(transposed, [Dh*n_heads*B, capacity]). Outputs the s8 quantized
versions plus per-token f32 scales.

Algorithm: per-token absmax → scale = absmax/127 → round(x/scale).
Validated end-to-end against numpy reference in
`/tmp/run_kv_quantize_probe.py` (cos_dequant > 0.9999).

First cut handles K-quant only; Vt-quant (transposed reduce) lands
in Phase 2.2.
"""

from quark.kernels.kv_cache_quantize.config import KVQuantizeConfig
from quark.kernels.kv_cache_quantize.kernel import KVQuantizeKernel
from quark.kernels.kv_cache_quantize.spec import KVQuantizeSpec

__all__ = ["KVQuantizeConfig", "KVQuantizeKernel", "KVQuantizeSpec"]
