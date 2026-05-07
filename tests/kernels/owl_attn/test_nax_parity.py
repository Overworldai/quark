"""Numerical parity: IR-emitted NAX attn vs hand-written nax_attn.py.

OBSOLETE — both sides of the comparison no longer exist on this branch:
``drivers/nax_attn.py`` was retired in commit ``32771c0`` once the IR
version (``kernels/owl_attn/nax.py``) reached parity. This test stays
as a marker so future readers understand the parity bench's outcome,
but it always skips. End-to-end verification of the surviving IR path
runs in ``scripts/verify_against_mlx.py`` and the per-frame bench at
``scripts/bench_quark_world_engine.py``.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from quark.device import DeviceFamily, current_device


def _skip_if_no_nax():
    pytest.skip(
        "drivers/nax_attn.py retired in 32771c0 — parity vs the "
        "hand-written kernel is no longer applicable. Run "
        "scripts/verify_against_mlx.py for end-to-end correctness."
    )
    # Unreachable, kept so the helper's signature stays intact for
    # any future re-use (the function shape predated the retirement).
    dev = current_device()
    if dev.family is not DeviceFamily.METAL:
        pytest.skip("not a Metal device")
    if not dev.caps.supports_nax:
        pytest.skip("Metal device without NAX support (M5+ required)")


def _make_inputs(
    *, n_kv_heads=16, gqa_ratio=2, tpf=128, num_buckets=16, Dh=64, max_segments=3, seed=0xA17E
):
    """Build a small but realistic owl_attn problem.

    Layout matches drivers/nax_attn.py's expectations:
      Q: [n_q_heads * tpf, Dh] bf16
      K_cache: [n_kv_heads * capacity, Dh] bf16
      Vt_cache: [n_kv_heads * Dh, capacity] bf16
      segments: [max_segments * 2] s32 — (start, length) pairs
      n_segments: [1] s32 — number of valid segments
      frame_t: [1] s32 — unused by the kernels but in the binding
    """
    n_q_heads = n_kv_heads * gqa_ratio
    capacity = num_buckets * tpf + tpf
    rng = np.random.default_rng(seed)

    # bf16 round-trip via float32 → uint16 ushort to keep bit patterns.
    def _bf16(x):
        # Truncate float32 to bf16 by zeroing the low 16 bits.
        x = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((x + 0x8000) >> 16).astype(np.uint16)

    Q = _bf16(rng.standard_normal((n_q_heads * tpf, Dh)) * 0.3)
    K_cache = _bf16(rng.standard_normal((n_kv_heads * capacity, Dh)) * 0.3)
    Vt_cache = _bf16(rng.standard_normal((n_kv_heads * Dh, capacity)) * 0.3)

    # Two valid segments covering [0, num_buckets*tpf) and [num_buckets*tpf, capacity).
    L = num_buckets * tpf
    seg_pairs = [0, L, L, tpf] + [0, 0] * (max_segments - 2)
    segments = np.array(seg_pairs, dtype=np.int32)
    n_segments = np.array([2], dtype=np.int32)
    frame_t = np.array([num_buckets * 1], dtype=np.int32)

    return {
        "Q": Q,
        "K_cache": K_cache,
        "Vt_cache": Vt_cache,
        "segments": segments,
        "n_segments": n_segments,
        "frame_t": frame_t,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "gqa_ratio": gqa_ratio,
        "tpf": tpf,
        "Dh": Dh,
        "capacity": capacity,
        "max_segments": max_segments,
    }


def _bf16_to_f32(arr_u16):
    """Convert a uint16-viewed bf16 array back to float32."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def _run_handwritten(inputs):
    """Dispatch the existing drivers/nax_attn.py kernel."""
    from quark.drivers.nax_attn import compile_nax_attn

    from quark.drivers import _metal_dispatch as _md

    BK = 32  # baseline; matches NaxAttnSpec(BK=32)
    pipeline = compile_nax_attn(BK)

    n_q_heads = inputs["n_q_heads"]
    tpf = inputs["tpf"]
    Dh = inputs["Dh"]
    capacity = inputs["capacity"]
    max_segments = inputs["max_segments"]
    scale = 1.0 / math.sqrt(Dh)
    q_row_stride = Dh
    q_head_stride = tpf * Dh

    params = struct.pack(
        "IIIIIIIfII",
        n_q_heads,
        inputs["n_kv_heads"],
        inputs["gqa_ratio"],
        tpf,
        capacity,
        Dh,
        max_segments,
        scale,
        q_row_stride,
        q_head_stride,
    )
    params_np = np.frombuffer(params, dtype=np.uint8).copy()

    arrs = (
        inputs["Q"],
        inputs["K_cache"],
        inputs["Vt_cache"],
        inputs["segments"],
        inputs["n_segments"],
        inputs["frame_t"],
    )
    handles, ptrs, nbytes = [], [], []
    for arr in arrs:
        narr = np.ascontiguousarray(arr)
        handles.append(-1)
        ptrs.append(narr.ctypes.data)
        nbytes.append(narr.nbytes)
    handles.append(-1)
    ptrs.append(params_np.ctypes.data)
    nbytes.append(params_np.nbytes)

    out_rows = n_q_heads * tpf
    out_nbytes = out_rows * Dh * 2
    plan = [(0, i, i) for i in range(6)] + [(1, 0, 6), (0, 6, 7)]
    n_q_tiles = tpf // 16
    grid = (n_q_tiles * 32, n_q_heads, 1)
    tg = (32, 1, 1)

    out_handle, out_ptr = _md.queue_launch(
        pipeline,
        handles,
        ptrs,
        nbytes,
        out_nbytes,
        plan,
        grid,
        tg,
        0,
    )
    # Wrap in QuarkTensor → np.asarray triggers the metal-cpp sync path.
    from quark.runtime.tensor import QuarkTensor, _contiguous_strides, _MetalStorage

    storage = _MetalStorage(out_handle, out_ptr, out_nbytes)
    qt = QuarkTensor(
        storage,
        (out_rows, Dh),
        _contiguous_strides((out_rows, Dh)),
        0,
        "bf16",
    )
    return np.asarray(qt).view(np.uint16).reshape(out_rows, Dh)


def _run_ir(inputs):
    """Dispatch the IR-emitted equivalent (now living in
    ``kernels/owl_attn/nax.py``)."""
    from quark.kernels.owl_attn.nax import NaxAttnSpec, dispatch_nax_attn

    spec = NaxAttnSpec(
        BK=32,
        n_q_heads=inputs["n_q_heads"],
        n_kv_heads=inputs["n_kv_heads"],
        gqa_ratio=inputs["gqa_ratio"],
        tpf=inputs["tpf"],
        capacity=inputs["capacity"],
        Dh=inputs["Dh"],
        max_segments=inputs["max_segments"],
    )
    out_qt = dispatch_nax_attn(
        spec=spec,
        Q=inputs["Q"],
        K_cache=inputs["K_cache"],
        Vt_cache=inputs["Vt_cache"],
        segments=inputs["segments"],
        n_segments=inputs["n_segments"],
        frame_t=inputs["frame_t"],
    )
    # QuarkTensor → numpy uint16 view of bf16.
    return (
        np.asarray(out_qt)
        .view(np.uint16)
        .reshape(
            inputs["n_q_heads"] * inputs["tpf"],
            inputs["Dh"],
        )
    )


def test_ir_vs_handwritten_cosine():
    """Outputs of the IR-emitted and hand-written kernels should be
    near-identical: same kernel, same inputs, just different MSL
    generators."""
    _skip_if_no_nax()
    inputs = _make_inputs()
    out_hw_u16 = _run_handwritten(inputs)
    out_ir_u16 = _run_ir(inputs)

    out_hw = _bf16_to_f32(out_hw_u16).reshape(-1)
    out_ir = _bf16_to_f32(out_ir_u16).reshape(-1)

    # Cosine similarity — robust to scale differences.
    cos = float(np.dot(out_hw, out_ir) / (np.linalg.norm(out_hw) * np.linalg.norm(out_ir) + 1e-30))
    print(f"cosine={cos:.6f}, max|diff|={np.max(np.abs(out_hw - out_ir)):.4f}")
    assert cos >= 0.999, f"IR vs hand-written cosine={cos:.6f} below 0.999"
