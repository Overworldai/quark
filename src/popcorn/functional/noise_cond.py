"""``popcorn.functional.noise_cond`` — cached noise-conditioner LUT.

The sigma schedule is known ahead of time (5 fixed values for
waypoint-1.5). Fourier features + the 2-layer MLP are pre-computed
once at setup and stored as a ``[n_sigmas, d_model]`` device tensor.
At inference time the forward path is a single row-select + zero
popcorn kernel calls (the MLP result is already baked in).

Setup (one-time, at model init):

    lut = precompute_noise_lut(sigmas, freqs, W1, W2)

Inference (hot path, zero non-popcorn ops):

    emb = lut[sigma_idx]   # pure tensor slice — no compute
"""

from __future__ import annotations

import sys

from popcorn.functional.gemm import gemm as _pcf_gemm
from popcorn.ir import DType

_IS_METAL = sys.platform == "darwin"
_SQRT2 = 1.4142135623730951


def _dtype_str(backend_dtype) -> str:
    return DType.from_backend(backend_dtype).value


def precompute_noise_lut(sigmas, freqs, W1, W2):
    """Pre-compute the full noise-conditioner embedding for every sigma
    in the schedule.

    ``sigmas``: list/tuple of float sigma values (e.g. [1.0, 0.9, 0.75, 0.3, 0.0]).
    ``freqs``, ``W1``, ``W2``: device tensors (the NoiseConditioner's weights).

    Returns a device tensor ``[n_sigmas, d_model]`` — the complete LUT.
    At inference time, index by sigma position: ``emb = lut[i]``.

    The Fourier features (sin/cos) are computed on the CPU via
    pure Python (no numpy). The MLP (2x pcf.gemm) runs on the GPU.
    """
    n = len(sigmas)
    freqs_list = _to_list(freqs)
    fourier_dim = len(freqs_list) * 2

    if _IS_METAL:
        return _precompute_metal(sigmas, freqs_list, fourier_dim, n, W1, W2)
    return _precompute_cuda(sigmas, freqs_list, fourier_dim, n, W1, W2)


def _precompute_cuda(sigmas, freqs_list, fourier_dim, n, W1, W2):
    """CUDA path: numpy f32 MLP (no tensor core MMA for f32×f32)."""
    import numpy as np

    from popcorn.runtime.tensor import PopcornTensor

    # Build Fourier features in f32.
    fourier = np.zeros((n, fourier_dim), dtype=np.float32)
    for i, s in enumerate(sigmas):
        phase = np.array([float(s) * 1000.0 * f for f in freqs_list], dtype=np.float32)
        fourier[i, : len(freqs_list)] = _SQRT2 * np.sin(phase)
        fourier[i, len(freqs_list) :] = _SQRT2 * np.cos(phase)

    # Get weights as f32 numpy.
    def _to_f32_np(t):
        if hasattr(t, "astype"):
            return (
                t.astype("f32").to_numpy() if hasattr(t, "to_numpy") else np.array(t.float().cpu())
            )
        return np.array(t)

    w1 = _to_f32_np(W1)  # [d_mid, fourier_dim]
    w2 = _to_f32_np(W2)  # [d_model, d_mid]

    # MLP: h = silu(fourier @ W1.T), emb = h @ W2.T
    h = fourier @ w1.T  # [n, d_mid]
    h = h * (1.0 / (1.0 + np.exp(-h)))  # silu = x * sigmoid(x)
    emb = h @ w2.T  # [n, d_model]

    # Transfer to device as bf16 (or f16 depending on weight dtype).
    out_dt = str(W2.dtype)
    if out_dt in ("f16", "bf16"):
        return PopcornTensor.from_numpy(emb, dtype=out_dt)
    return PopcornTensor.from_numpy(emb, dtype="bf16")


def _precompute_metal(sigmas, freqs_list, fourier_dim, n, W1, W2):
    """Metal path — mlx directly, no PT."""
    import mlx.core as mx
    import numpy as np

    BM_MIN = 16
    freqs_np = np.array(freqs_list, dtype=np.float32)

    rows_np = []
    for s in sigmas:
        phase = float(s) * 1000.0 * freqs_np
        row = _SQRT2 * np.concatenate([np.sin(phase), np.cos(phase)])
        rows_np.append(row)
    fourier_np = np.stack(rows_np, axis=0)

    if n < BM_MIN:
        pad_np = np.zeros((BM_MIN - n, fourier_np.shape[1]), dtype=np.float32)
        fourier_np = np.concatenate([fourier_np, pad_np], axis=0)

    fourier_u16 = (fourier_np.view(np.uint32) >> 16).astype(np.uint16)
    fourier_dev = mx.array(fourier_u16).view(mx.bfloat16)
    out_dt = _dtype_str(W2.dtype)
    h = _pcf_gemm(fourier_dev, W1, activation="silu", out_dtype=out_dt)
    emb = _pcf_gemm(h, W2, out_dtype=out_dt)
    mx.synchronize()

    return emb[:n]


def _to_list(tensor) -> list[float]:
    """Extract a 1-D tensor to a Python list of floats, backend-agnostic."""
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    # PopcornTensor path.
    if hasattr(tensor, "to_bytes"):
        import struct as _struct

        t = tensor.astype("f32") if tensor.dtype != "f32" else tensor
        raw = t.contiguous().to_bytes()
        n = t.numel()
        return list(_struct.unpack(f"<{n}f", raw))
    return list(tensor)
