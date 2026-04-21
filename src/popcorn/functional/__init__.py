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
from popcorn.functional.ada_gate_residual import ada_gate_residual
from popcorn.functional.ada_rmsnorm import ada_rmsnorm
from popcorn.functional.attention import attention
from popcorn.functional.euler_step import euler_step
from popcorn.functional.gemm import gemm
from popcorn.functional.head_rmsnorm import head_rmsnorm
from popcorn.functional.kv_cache_update import kv_cache_update
from popcorn.functional.moe_inproj import moe_inproj
from popcorn.functional.moe_outproj import moe_outproj
from popcorn.functional.noise_cond import precompute_noise_lut
from popcorn.functional.owl_attn import owl_attn
from popcorn.functional.patchify import patchify, patchify_2x2
from popcorn.functional.quantize_e4m3 import quantize_e4m3
from popcorn.functional.randn import randn
from popcorn.functional.rmsnorm import rmsnorm
from popcorn.functional.shuffle import (
    shuffle_b_for_gemm,
    shuffle_b_for_moe_inproj,
    shuffle_b_for_moe_outproj,
)
from popcorn.functional.shuffle_weight import shuffle_weight
from popcorn.functional.silu import silu
from popcorn.functional.unpatchify import unpatchify
from popcorn.functional.value_residual import value_residual
from popcorn.functional.value_residual_packed import value_residual_packed

__all__ = [
    "ada_gate_residual",
    "ada_rmsnorm",
    "attention",
    "euler_step",
    "gemm",
    "head_rmsnorm",
    "kv_cache_update",
    "moe_inproj",
    "moe_outproj",
    "owl_attn",
    "patchify",
    "patchify_2x2",
    "precompute_noise_lut",
    "quantize_e4m3",
    "randn",
    "rmsnorm",
    "shuffle_b_for_gemm",
    "shuffle_b_for_moe_inproj",
    "shuffle_b_for_moe_outproj",
    "shuffle_weight",
    "silu",
    "unpatchify",
    "value_residual",
    "value_residual_packed",
]
