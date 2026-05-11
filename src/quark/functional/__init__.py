"""``quark.functional`` — torch/Metal-native callables over quark kernels.

    import quark.functional as qf

    C = qf.gemm(A, B)
    y = qf.attention(Q, K, V_t, B=..., n_kv_heads=..., ...)

Each entry accepts ``QuarkTensor`` (or ``torch.Tensor`` on CUDA) and
returns the same type.
On torch, every op is also registered via ``torch.library.custom_op``
so ``torch.compile`` can trace through it — use the
``torch.ops.quark.<name>`` form inside compiled regions or rely on the
public ``qf.<name>`` wrapper which dispatches on input type.

"""

from __future__ import annotations

# Per-kernel modules register their torch custom ops at import time.
from quark.functional.ada_gate_residual import ada_gate_residual
from quark.functional.ada_rmsnorm import ada_rmsnorm
from quark.functional.attention import attention
from quark.functional.euler_step import euler_step
from quark.functional.gemm import gemm
from quark.functional.head_rmsnorm import head_rmsnorm
from quark.functional.kv_cache_update import kv_cache_update
from quark.functional.moe_inproj import moe_inproj
from quark.functional.moe_outproj import moe_outproj
from quark.functional.moe_reduce import moe_reduce
from quark.functional.moe_router import moe_router
from quark.functional.moe_router_correct import moe_router_correct
from quark.functional.moe_router_shared import moe_router_shared
from quark.functional.noise_cond import precompute_noise_lut
from quark.functional.owl_attn import owl_attn
from quark.functional.patchify import patchify, patchify_2x2
from quark.functional.quantize_e4m3 import quantize_e4m3
from quark.functional.randn import randn
from quark.functional.rmsnorm import rmsnorm
from quark.functional.shuffle import (
    shuffle_b_for_gemm,
    shuffle_b_for_moe_inproj,
    shuffle_b_for_moe_outproj,
)
from quark.functional.shuffle_weight import shuffle_weight
from quark.functional.silu import silu
from quark.functional.unpatchify import unpatchify
from quark.functional.value_residual import value_residual
from quark.functional.value_residual_packed import value_residual_packed

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
    "moe_reduce",
    "moe_router",
    "moe_router_correct",
    "moe_router_shared",
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
