"""``popcorn.functional.quantize_e4m3`` — bf16/f16 → e4m3 quantization.

    W_fp8 = pcf.quantize_e4m3(W)

Quantizes a tensor to e4m3 format using the GPU via packed_convert
(PTX: ``cvt.rn.satfinite.e4m3x2.f32``). Returns a tensor with
dtype="e4m3" (1 byte per element).
"""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("quantize_e4m3")
    return _Cls


def quantize_e4m3(X):
    """Quantize a bf16/f16/f32 tensor to e4m3.

    Input: any shape, bf16/f16/f32 dtype.
    Output: same shape, e4m3 dtype (1 byte per element).
    """
    from popcorn.ir import DType
    from popcorn.kernels.quantize_e4m3.config import QuantizeE4M3Config
    from popcorn.kernels.quantize_e4m3.spec import QuantizeE4M3Spec

    orig_shape = tuple(X.shape)
    N = 1
    for d in orig_shape:
        N *= int(d)

    # Pad to even if needed.
    padded = N if N % 2 == 0 else N + 1

    X_flat = X.reshape(N)
    if padded != N:
        # Pad with zero.
        from popcorn.runtime.tensor import PopcornTensor

        pad = PopcornTensor.zeros(1, dtype=X_flat.dtype)
        X_flat = PopcornTensor.cat([X_flat.contiguous(), pad], dim=0)

    src_dt = DType.from_backend(X.dtype) if not isinstance(X.dtype, str) else DType(X.dtype)
    spec = QuantizeE4M3Spec(N=padded, src_dtype=src_dt)
    config = QuantizeE4M3Config.default_for(spec)

    result = call_with_bindings(
        _cls(),
        spec,
        config=config,
        provided={"X": X_flat},
        auto_alloc=("Out",),
        like=X_flat,
    )

    out = result["Out"]  # [N] e4m3

    if padded != N:
        out = out[:N]

    return out.reshape(*orig_shape)
