"""``popcorn.functional`` — torch/MLX-native callables over popcorn kernels.

    import popcorn.functional as pcf

    C = pcf.gemm(A, B)
    y = pcf.attention(Q, K, V_t, B=..., n_kv_heads=..., ...)

Each entry accepts framework-native tensors (``torch.Tensor`` or
``mx.array``) and returns the same type. On torch, every op is also
registered via ``torch.library.custom_op`` so ``torch.compile`` can
trace through it — use the ``torch.ops.popcorn.<name>`` form inside
compiled regions or rely on the public ``pcf.<name>`` wrapper which
dispatches on input type.

"""

from __future__ import annotations

# Per-kernel modules register their torch custom ops at import time.
from popcorn.functional.attention import attention
from popcorn.functional.gemm import gemm
from popcorn.functional.kv_cache_update import kv_cache_update
from popcorn.functional.moe_inproj import moe_inproj
from popcorn.functional.moe_outproj import moe_outproj
from popcorn.functional.owl_attn import owl_attn
from popcorn.functional.shuffle import (
    shuffle_b_for_gemm,
    shuffle_b_for_moe_inproj,
    shuffle_b_for_moe_outproj,
)

__all__ = [
    "attention",
    "gemm",
    "kv_cache_update",
    "moe_inproj",
    "moe_outproj",
    "owl_attn",
    "shuffle_b_for_gemm",
    "shuffle_b_for_moe_inproj",
    "shuffle_b_for_moe_outproj",
]
