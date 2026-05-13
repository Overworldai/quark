"""Phase 4.4 (preliminary) — steady-state lfps on Battlemage.

Warmup + measure pattern. The first ``gen_frame`` pays for kernel
autotune + IGC compile + OpenVINO IR load; we drop that frame and
measure the next N frames' wall time. Reports both the per-frame
latency (s/frame) and lfps.

Not a strict gate; emits a print line so the run shows up in CI
logs. The Vulkan SPV baseline (per ``project_spv_dispatch_floor.md``)
is **3.76 lfps**; OCL should approach or beat that after Phase 4.1
(batched submit via ``quark.lazy()``).
"""

from __future__ import annotations

import os
import pathlib
import time

import pytest


def _ocl_available() -> bool:
    try:
        from quark.runtime.sync import _IS_OCL
        return bool(_IS_OCL)
    except Exception:
        return False


def _openvino_ir_local() -> str | None:
    cand = pathlib.Path("/tmp/taehv_openvino")
    if (cand / "16x32" / "encoder.xml").is_file():
        return str(cand)
    return None


pytestmark = [
    pytest.mark.skipif(
        not _ocl_available(),
        reason="OCL backend not available — Phase 4 is DEVKIT-only",
    ),
    pytest.mark.skipif(
        _openvino_ir_local() is None,
        reason="Local TAEHV OpenVINO IR not found at /tmp/taehv_openvino/16x32",
    ),
]


@pytest.fixture
def gen_frame_env(monkeypatch):
    monkeypatch.setenv("QUARK_FORCE_ENGINE", "intel")
    monkeypatch.setenv("QUARK_TAEHV_OPENVINO_URI", "/tmp/taehv_openvino")
    yield


def test_steady_state_lfps(gen_frame_env):
    """N=16 measured frames after a 1-frame warmup. Reports lfps.

    The first frame includes autotune + IGC compile + OV IR load —
    not comparable to steady-state. Subsequent frames hit cached
    kernels and the per-call ``clFinish`` (or, post-Phase-4.1, the
    batched lazy-mode submit).
    """
    from quark.engine import Engine

    with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
        engine = Engine("Overworld/Waypoint-1.5-1B-360P", load_weights=False)

    # Warmup
    _ = engine.gen_frame(ctrl=None)

    n_measured = 16
    t0 = time.perf_counter()
    for _ in range(n_measured):
        engine.gen_frame(ctrl=None)
    elapsed = time.perf_counter() - t0

    s_per_frame = elapsed / n_measured
    lfps = n_measured / elapsed
    vae_units = engine._vae_compute_units  # "GPU" or "CPU"
    print(
        f"\n  Battlemage steady-state: "
        f"{lfps:.2f} lfps ({s_per_frame*1000:.1f} ms/frame) over "
        f"{n_measured} frames; VAE compute_units={vae_units}"
    )
    # Sanity: any non-trivial measurement should clear 0.1 lfps. The
    # Vulkan SPV baseline is 3.76 lfps. We assert only the trivial
    # lower bound here; the real Phase 4 work is to close the gap.
    assert lfps > 0.1, f"lfps={lfps:.4f} — something's catastrophically slow"
