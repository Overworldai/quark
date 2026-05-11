"""``quark.functional.shuffle_weight`` — GPU-side weight pre-shuffle.

    W_shuffled = qf.shuffle_weight(W, K_CHUNK=64, mma_k=16, bpad=0)

Permutes the B-matrix so each MMA lane's fragment data is contiguous
in shared memory. The shuffle is the same permutation as
``weight_shuffle.cached_shuffle_b`` but runs on-GPU via the
``shuffle_weight`` kernel.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("shuffle_weight")
    return _Cls


def shuffle_weight(W, *, K_CHUNK: int = 64, mma_k: int = 16, bpad: int = 0):
    """Pre-shuffle a weight tensor for vectorized frag loads.

    ``W``: ``[N, K]`` weight tensor (bf16/f16/e4m3).
    Returns a same-shape (or wider if bpad > 0) shuffled tensor.

    Default ``K_CHUNK`` / ``mma_k`` work for the common bf16 GEMM
    configs. The autotune-resolved ``Linear.prepare()`` path
    determines the right values automatically.
    """
    from quark.ir import DType
    from quark.kernels.shuffle_weight.config import ShuffleWeightConfig
    from quark.kernels.shuffle_weight.spec import ShuffleWeightSpec

    N = int(W.shape[0])
    K = int(W.shape[1])
    dtype_str = W.dtype if isinstance(W.dtype, str) else str(W.dtype)

    spec = ShuffleWeightSpec(
        N=N,
        K=K,
        dtype=DType(dtype_str),
        K_CHUNK=K_CHUNK,
        mma_k=mma_k,
        bpad=bpad,
    )
    config = ShuffleWeightConfig.default_for(spec)

    result = call_with_bindings(
        _cls(),
        spec,
        config=config,
        provided={"Src": W},
        auto_alloc=("Dst",),
        like=W,
    )
    return result["Dst"]
