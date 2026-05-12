"""Pin model parameters + persistent state to the active device pool.

Shared between every backend that uses ``QuarkTensor`` pool buffers
(today: Metal, OCL/Intel). The Metal description below applies to
every host-coherent pool backend — OCL USM-shared has the same
recycle-on-lazy-eval shape.
"""

from __future__ import annotations


def _pin_params_to_device(model) -> None:
    """Move every numpy-carrier ``Parameter`` + persistent state buffer
    onto the active device pool as a ``QuarkTensor``.

    ``quark.nn.Module`` parameters arrive as tagged numpy arrays (host
    memory) by default. The dispatcher's input cache wraps them into
    pool buffers on first launch but *recycles* the wrapped buffers
    at every ``quark.lazy()`` eval boundary — meaning a per-frame
    ``gen_frame()`` would re-memcpy every weight tensor (~3.7 GB at
    360p bf16) on each iteration.

    Pinning the parameters as ``QuarkTensor`` (which holds a refcount
    on its pool slot) makes the dispatcher skip the input-cache write
    entirely and read straight from the pinned buffer. KV-cache state
    buffers (``K_cache`` / ``Vt_cache`` / ``segments`` / ``n_segments``
    / ``frame_t`` / ``frozen``) must also be pinned, otherwise the
    pool recycle silently throws away every cache update between
    frames.
    """
    import numpy as _np

    from quark.nn.module import Module as _NN
    from quark.nn.module import Parameter as _NN_Param
    from quark.runtime.tensor import QuarkTensor

    def _is_numpy_carrier(t) -> bool:
        return isinstance(t, _np.ndarray)

    def _to_qt(arr):
        qd = getattr(arr, "quark_dtype", None) or {
            "uint16": "bf16",
            "float16": "f16",
            "float32": "f32",
            "int32": "s32",
            "uint8": "u8",
            "int8": "s8",
        }.get(arr.dtype.name, "bf16")
        return QuarkTensor.from_numpy(arr, dtype=qd)

    # Persistent state attrs on KVCacheUpdate that need pinning so the
    # ring buffer survives eval recycle. Anything else is a Parameter
    # (handled in the walk below) or a transient.
    _STATE_BUF_ATTRS = ("K_cache", "Vt_cache", "segments", "n_segments", "frame_t", "frozen")

    def _walk(mod):
        for _, val in mod._own_members():
            if isinstance(val, _NN_Param):
                arr = val.data
                if _is_numpy_carrier(arr):
                    val.data = _to_qt(_np.ascontiguousarray(arr))
            elif isinstance(val, _NN):
                _walk(val)
        for attr in _STATE_BUF_ATTRS:
            v = getattr(mod, attr, None)
            if _is_numpy_carrier(v):
                setattr(mod, attr, _to_qt(_np.ascontiguousarray(v)))

    _walk(model)

    # Per-block ``_cond_lut`` tuples (populated by ``Waypoint15.prepare()``)
    # and the model-level ``_out_norm_luts`` are read on every forward;
    # pin them too or they re-upload per eval.
    for block in getattr(model, "blocks", []):
        lut = getattr(block, "_cond_lut", None)
        if lut is None:
            continue
        block._cond_lut = [
            tuple(_to_qt(_np.ascontiguousarray(t)) if _is_numpy_carrier(t) else t for t in tup)
            for tup in lut
        ]

    on_luts = getattr(model, "_out_norm_luts", None)
    if on_luts is not None:
        model._out_norm_luts = [
            (
                _to_qt(_np.ascontiguousarray(s_on)) if _is_numpy_carrier(s_on) else s_on,
                _to_qt(_np.ascontiguousarray(b_on)) if _is_numpy_carrier(b_on) else b_on,
            )
            for s_on, b_on in on_luts
        ]
