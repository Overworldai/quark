"""``quark.functional.quantize_e4m3`` — bf16/f16 → e4m3 quantization.

    W_fp8 = qf.quantize_e4m3(W)

Quantizes a tensor to e4m3 format using the GPU via packed_convert
(PTX: ``cvt.rn.satfinite.e4m3x2.f32``). Returns a tensor with
dtype="e4m3" (1 byte per element).
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings
from quark.kernels import get

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
    from quark.ir import DType
    from quark.kernels.quantize_e4m3.config import QuantizeE4M3Config
    from quark.kernels.quantize_e4m3.spec import QuantizeE4M3Spec

    orig_shape = tuple(X.shape)
    N = 1
    for d in orig_shape:
        N *= int(d)

    # Pad to even if needed.
    padded = N if N % 2 == 0 else N + 1

    X_flat = X.reshape(N)
    if padded != N:
        # Pad with zero.
        from quark.runtime.tensor import QuarkTensor

        pad = QuarkTensor.zeros(1, dtype=X_flat.dtype)
        X_flat = QuarkTensor.cat([X_flat.contiguous(), pad], dim=0)

    src_dt = DType.from_backend(X)
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
