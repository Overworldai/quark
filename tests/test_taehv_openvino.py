"""OpenVINO TAEHV smoke test.

Skipped automatically when ``openvino`` / ``taehv`` / ``torch`` aren't
importable, or when ``QUARK_TAEHV_OPENVINO_DIR`` doesn't point at a
pre-exported IR directory. Run the exporter once before invoking::

    python -m quark.taehv.openvino.export \\
        --ae-repo Overworld-Models/taehv1_5 \\
        --latent-height 16 --latent-width 32 \\
        --cache-dir ./openvino_cache
    QUARK_TAEHV_OPENVINO_DIR=./openvino_cache pytest tests/test_taehv_openvino.py
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

LAT_H, LAT_W = 16, 32  # 360p
PIXEL_H, PIXEL_W = 360, 640

_skip = pytest.mark.skipif(
    True, reason="set at import time below"
)


def _runtime_ready() -> tuple[bool, str]:
    """Check whether the test environment can run the smoke."""
    try:
        import openvino  # noqa: F401
    except Exception as e:
        return False, f"openvino not importable: {e}"
    ov_dir = os.environ.get("QUARK_TAEHV_OPENVINO_DIR")
    if not ov_dir:
        return False, "QUARK_TAEHV_OPENVINO_DIR not set (run `python -m quark.taehv.openvino.export` first)"
    path = pathlib.Path(ov_dir) / f"{LAT_H}x{LAT_W}" / "decoder.xml"
    if not path.exists():
        return False, f"missing decoder IR at {path}"
    return True, ""


_ready, _skip_reason = _runtime_ready()


@pytest.mark.skipif(not _ready, reason=_skip_reason)
def test_decode_shape_and_dtype() -> None:
    """``ae.decode(latent)`` returns ``(4, H_pix, W_pix, 3) uint8``."""
    from quark.taehv.openvino import load as load_ov

    ae = load_ov(
        os.environ["QUARK_TAEHV_OPENVINO_DIR"],
        latent_height=LAT_H, latent_width=LAT_W,
        device=os.environ.get("QUARK_TAEHV_OPENVINO_DEVICE", "CPU"),
    )

    latent = np.zeros((1, 32, LAT_H, LAT_W), dtype=np.float32)
    frames = ae.decode(latent)
    assert frames.shape == (4, PIXEL_H, PIXEL_W, 3), frames.shape
    assert frames.dtype == np.uint8


@pytest.mark.skipif(not _ready, reason=_skip_reason)
def test_encode_decode_roundtrip() -> None:
    """``encode(decode(rand_latent))`` cos_sim is in the TAEHV lossy range."""
    from quark.taehv.openvino import load as load_ov

    ae = load_ov(
        os.environ["QUARK_TAEHV_OPENVINO_DIR"],
        latent_height=LAT_H, latent_width=LAT_W,
        device=os.environ.get("QUARK_TAEHV_OPENVINO_DEVICE", "CPU"),
    )

    rng = np.random.default_rng(0)
    rand_latent = rng.standard_normal((1, 32, LAT_H, LAT_W)).astype(np.float32)
    decoded = ae.decode(rand_latent)
    re_encoded = ae.encode(decoded)

    a = rand_latent.astype(np.float32).ravel()
    b = re_encoded.astype(np.float32).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    # TAEHV is lossy by design; cos in [0.4, 0.7] is the typical envelope
    # on a random latent. Anything outside means the model is mis-wired.
    assert 0.3 < cos < 0.9, f"round-trip cos out of expected range: {cos:.4f}"


@pytest.mark.skipif(not _ready, reason=_skip_reason)
def test_reset_clears_state() -> None:
    """``ae.reset()`` re-zeroes the decoder MemBlock state."""
    from quark.taehv.openvino import load as load_ov

    ae = load_ov(
        os.environ["QUARK_TAEHV_OPENVINO_DIR"],
        latent_height=LAT_H, latent_width=LAT_W,
        device=os.environ.get("QUARK_TAEHV_OPENVINO_DEVICE", "CPU"),
    )
    # Decode once to populate state
    latent = np.ones((1, 32, LAT_H, LAT_W), dtype=np.float32) * 0.5
    _ = ae.decode(latent)
    assert ae._state_lo.any() or ae._state_mid.any() or ae._state_hi.any()
    ae.reset()
    assert not ae._state_lo.any()
    assert not ae._state_mid.any()
    assert not ae._state_hi.any()
    assert not ae._primed


@pytest.mark.skipif(not _ready, reason=_skip_reason)
def test_device_fallback_to_cpu() -> None:
    """Requesting an unavailable device (GPU/NPU on a CPU-only host)
    falls back to CPU with a warning rather than crashing."""
    import warnings

    from quark.taehv.openvino import load as load_ov
    import openvino as ov

    avail = set(ov.Core().available_devices)
    if "GPU" in avail:
        pytest.skip("host has GPU; fallback path not exercised here")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ae = load_ov(
            os.environ["QUARK_TAEHV_OPENVINO_DIR"],
            latent_height=LAT_H, latent_width=LAT_W,
            device="GPU",
        )
    assert ae._device == "CPU", f"expected CPU fallback, got {ae._device}"
    assert any("falling back to cpu" in str(_w.message).lower() for _w in w), \
        f"expected fallback warning, got: {[str(_.message) for _ in w]}"


def _pytorch_ref_ready() -> tuple[bool, str]:
    if not _ready:
        return False, _skip_reason
    try:
        import torch  # noqa: F401
        from taehv import TAEHV  # noqa: F401
    except Exception as e:
        return False, f"torch / upstream taehv not importable: {e}"
    return True, ""


_pt_ready, _pt_skip_reason = _pytorch_ref_ready()


@pytest.mark.skipif(not _pt_ready, reason=_pt_skip_reason)
def test_pytorch_reference_parity() -> None:
    """OpenVINO IR matches the upstream PyTorch TAEHV at fp32 reference:
    encoder cos=1.0 exact, decoder cos=1.0 with max|diff| in the
    fp16-storage rounding envelope (~0.015).

    This is the strongest correctness signal — bypasses any
    intermediate-format ambiguity (CoreML round-trip etc.) and
    compares the IR's compute path directly against the source.
    """
    import torch
    import openvino as ov

    from quark.taehv._pt_traces import (
        DecoderExplicitState,
        EncoderStatic,
        load_taehv_pytorch,
    )
    from quark.taehv.openvino import load as load_ov

    taehv = load_taehv_pytorch("Overworld-Models/taehv1_5")
    enc_pt = EncoderStatic(taehv, h=LAT_H * 8, w=LAT_W * 8).eval()
    dec_pt = DecoderExplicitState(taehv, lat_h=LAT_H, lat_w=LAT_W).eval()

    ae = load_ov(
        os.environ["QUARK_TAEHV_OPENVINO_DIR"],
        latent_height=LAT_H, latent_width=LAT_W,
        device=os.environ.get("QUARK_TAEHV_OPENVINO_DEVICE", "CPU"),
    )

    rng = np.random.default_rng(7)

    def _cos(a, b):
        a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Encoder
    enc_input = rng.standard_normal((4, 12, LAT_H * 8, LAT_W * 8)).astype(np.float32)
    with torch.no_grad():
        pt_out = enc_pt(torch.from_numpy(enc_input)).numpy()
    enc_req = ae._encoder.create_infer_request()
    enc_req.set_input_tensor(0, ov.Tensor(enc_input))
    enc_req.infer()
    ov_out = np.array(enc_req.get_output_tensor(0).data, copy=True)

    # CPU device computes in fp32, GPU device computes in fp16 by
    # default — pick the right tolerance band for each.
    enc_max_diff_tol = 5e-2 if ae._device != "CPU" else 1e-3
    assert _cos(pt_out, ov_out) > 0.999999, f"encoder cos drift: {_cos(pt_out, ov_out)}"
    assert np.abs(pt_out - ov_out).max() < enc_max_diff_tol, \
        f"encoder max|diff|: {np.abs(pt_out - ov_out).max()} > {enc_max_diff_tol}"

    # Decoder
    lat = rng.standard_normal((1, 32, LAT_H, LAT_W)).astype(np.float32)
    state_lo = rng.standard_normal((3, 256, LAT_H, LAT_W)).astype(np.float32)
    state_mid = rng.standard_normal((3, 128, LAT_H * 2, LAT_W * 2)).astype(np.float32)
    state_hi = rng.standard_normal((3, 64, LAT_H * 4, LAT_W * 4)).astype(np.float32)

    with torch.no_grad():
        pt_frames, pt_lo, pt_mid, pt_hi = dec_pt(
            torch.from_numpy(lat),
            torch.from_numpy(state_lo),
            torch.from_numpy(state_mid),
            torch.from_numpy(state_hi),
        )

    ae.reset()
    dec_req = ae._dec_req
    dec_req.set_input_tensor(0, ov.Tensor(lat))
    dec_req.set_input_tensor(1, ov.Tensor(state_lo))
    dec_req.set_input_tensor(2, ov.Tensor(state_mid))
    dec_req.set_input_tensor(3, ov.Tensor(state_hi))
    dec_req.infer()
    ov_frames = np.array(dec_req.get_output_tensor(ae._dec_frames_idx).data, copy=True)
    ov_lo = np.array(dec_req.get_output_tensor(ae._dec_lo_idx).data, copy=True)
    ov_mid = np.array(dec_req.get_output_tensor(ae._dec_mid_idx).data, copy=True)
    ov_hi = np.array(dec_req.get_output_tensor(ae._dec_hi_idx).data, copy=True)

    # GPU device's fp16 compute path has a larger error envelope than
    # CPU's fp32. Scale the per-output tolerances by ~10× on GPU.
    _dev_scale = 10.0 if ae._device != "CPU" else 1.0
    for label, pt, ov_t, tol in (
        ("frames", pt_frames.numpy(), ov_frames, 5e-3 * _dev_scale),
        ("state_lo", pt_lo.numpy(), ov_lo, 5e-3 * _dev_scale),
        ("state_mid", pt_mid.numpy(), ov_mid, 2e-2 * _dev_scale),
        ("state_hi", pt_hi.numpy(), ov_hi, 1e-2 * _dev_scale),
    ):
        c = _cos(pt, ov_t)
        d = float(np.abs(pt - ov_t).max())
        assert c > 0.99999, f"{label} cos drift: {c}"
        assert d < tol, f"{label} max|diff| {d} exceeds tol {tol}"


@pytest.mark.skipif(not _ready, reason=_skip_reason)
def test_dispatcher_routes_to_openvino_on_linux() -> None:
    """``load_taehv(uri)`` on Linux without ``-coreml`` suffix routes to OpenVINO."""
    import sys
    from quark.taehv import load_taehv

    if sys.platform == "darwin":
        pytest.skip("default dispatch is CoreML on darwin")

    ae = load_taehv(
        os.environ["QUARK_TAEHV_OPENVINO_DIR"],
        latent_height=LAT_H, latent_width=LAT_W,
        compute_units=os.environ.get("QUARK_TAEHV_OPENVINO_DEVICE", "CPU"),
    )
    # Backend object should be the OpenVINO runtime, identifiable by
    # the ``_device`` attribute (CoreML uses ``compute_units``).
    assert hasattr(ae, "_device"), f"expected OpenVINOTAEHV, got {type(ae).__name__}"
