"""Phase 3.3 — EngineIntel.gen_frame end-to-end with real OpenVINO TAEHV.

Constructs the engine with the production Waypoint-1.5-1B-360P config
(height=8, width=16, patch=[2,2] → latent grid 16×32, matching the
pre-built OpenVINO IR at ``/tmp/taehv_openvino/16x32/``), skips weight
load (DiT runs random), and exercises ``gen_frame`` which routes:

    encode_ctrl → DiT denoise (OCL) → DiT commit (OCL) →
        PipelinedDecoder.submit(latent) → OpenVINO decode → pixels

Gate: pixels return as the documented ``[temporal_compression, H_pix,
W_pix, 3]`` torch.uint8 tensor and contain no NaN / Inf. Visual
correctness is not asserted (random DiT weights produce noise) — this
is a wiring check, not a correctness check. With real weights the
same flow drives the actual demo.

The Phase 3.1 ``_compute_latent`` smoke covers the DiT half in
isolation (``QUARK_SKIP_VAE=1``). This adds the VAE half — exercising
the OpenVINO IR + ``PipelinedDecoder`` worker thread + host snapshot
on the same dispatch path the production engine uses.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest


def _ocl_available() -> bool:
    try:
        from quark.runtime.sync import _IS_OCL
        return bool(_IS_OCL)
    except Exception:
        return False


def _openvino_ir_local() -> str | None:
    """Return the local TAEHV-OpenVINO export dir if it exists.

    The ``quark.taehv.openvino.fetch._mirror_dir`` shortcut treats a
    local directory containing ``<latH>x<latW>/{encoder,decoder}.xml``
    as a drop-in for the HF repo. ``/tmp/taehv_openvino`` is where
    ``python -m quark.taehv.openvino.export`` writes by default.
    """
    cand = pathlib.Path("/tmp/taehv_openvino")
    if (cand / "16x32" / "encoder.xml").is_file():
        return str(cand)
    return None


pytestmark = [
    pytest.mark.skipif(
        not _ocl_available(),
        reason="OCL backend not available — Phase 3.3 is DEVKIT-only",
    ),
    pytest.mark.skipif(
        _openvino_ir_local() is None,
        reason=(
            "Local TAEHV OpenVINO IR not found at /tmp/taehv_openvino/16x32 — "
            "run ``python -m quark.taehv.openvino.export "
            "--ae-repo Overworld-Models/taehv1_5 "
            "--latent-height 16 --latent-width 32 "
            "--cache-dir /tmp/taehv_openvino`` on the devkit first."
        ),
    ),
]


@pytest.fixture
def gen_frame_env(monkeypatch):
    """Point the engine at the real Waypoint 360P model URI + local IR.

    The HF cache already holds ``Overworld/Waypoint-1.5-1B-360P``
    (config + weights) and ``Overworld-Models/taehv1_5`` (PyTorch
    reference; not used at runtime — kept for the export script). The
    OpenVINO IR is at ``/tmp/taehv_openvino`` after the export step.
    """
    monkeypatch.setenv("QUARK_FORCE_ENGINE", "intel")
    # Override the OpenVINO TAEHV repo URI with the local-dir path —
    # ``fetch_openvino_artifacts`` honours a local dir before falling
    # back to ``huggingface_hub.snapshot_download``.
    monkeypatch.setenv("QUARK_TAEHV_OPENVINO_URI", "/tmp/taehv_openvino")
    yield


def test_gen_frame_one_frame(gen_frame_env):
    """One call to ``engine.gen_frame()`` returns a pixel tensor.

    Random DiT weights → meaningless pixel content; we only assert
    shape + dtype + finiteness. Real-weights visual inspection is a
    separate test (load_weights=True, save the rendered frame).
    """
    from quark.engine import Engine, EngineIntel

    # ``Overworld/Waypoint-1.5-1B-360P`` is already cached locally
    # from earlier devkit work; load_weights=False keeps the DiT in
    # random-init state (fast construction, no .safetensors deser).
    with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
        engine = Engine("Overworld/Waypoint-1.5-1B-360P", load_weights=False)
    assert isinstance(engine, EngineIntel)
    assert engine._taehv is not None, "VAE handle not constructed"
    assert engine._pipe is not None, "PipelinedDecoder not constructed"

    pixels = engine.gen_frame(ctrl=None)
    assert pixels is not None, "gen_frame returned None"
    # Documented contract: [temporal_compression=4, H_pix, W_pix, 3]
    # uint8. ``temporal_compression`` lives on the raw config dict
    # rather than the Waypoint15Config dataclass; TAEHV is fixed at 4.
    assert pixels.ndim == 4, f"pixels shape {tuple(pixels.shape)} ndim != 4"
    assert pixels.shape[0] == 4, f"expected 4 subframes, got {pixels.shape[0]}"
    assert pixels.shape[-1] == 3
    assert pixels.dtype == np.uint8 or str(pixels.dtype) == "torch.uint8"

    # uint8 can't be NaN/Inf — sanity range gate.
    pix_np = pixels.numpy() if hasattr(pixels, "numpy") else np.asarray(pixels)
    assert pix_np.min() >= 0 and pix_np.max() <= 255


def test_gen_frame_stability_short_loop(gen_frame_env):
    """Phase 3.4 — short stability loop (8 frames).

    Verifies the gen_frame path is reusable across frames — KV-cache
    commit + rollover, ``frame_counter`` advance, OpenVINO encoder
    state reset semantics, and the PipelinedDecoder's depth-1
    submit/drain. RSS isn't asserted (would need an external probe),
    but a hard crash / hang / NaN propagation across iters is caught.

    8 frames is a fast-running proxy for the full 100-frame stability
    target in ``OCL_E2E_PLAN.md`` Phase 3.4 — keeps CI under a minute
    while still exercising commit-write rollover at every frame.
    """
    from quark.engine import Engine

    with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
        engine = Engine("Overworld/Waypoint-1.5-1B-360P", load_weights=False)

    for i in range(8):
        pixels = engine.gen_frame(ctrl=None)
        assert pixels is not None, f"gen_frame returned None at frame {i}"
        assert pixels.shape[0] == 4 and pixels.shape[-1] == 3
        pix_np = pixels.numpy() if hasattr(pixels, "numpy") else np.asarray(pixels)
        # Catch silent NaN propagation through the DiT → TAEHV chain:
        # an inf/NaN latent decodes to all-zero (or all-255) saturated
        # pixels rather than the gradient noise random weights produce.
        # Random DiT → noise pixels covering most of [0, 255]; flag if
        # the entire frame collapses to a single value.
        assert pix_np.std() > 0, f"frame {i} collapsed to constant pixel value"
