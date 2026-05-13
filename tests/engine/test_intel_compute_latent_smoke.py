"""Phase 3.1 — EngineIntel._compute_latent stub-VAE smoke.

Builds an EngineIntel with a tiny synthetic config, no weight load,
``QUARK_SKIP_VAE=1`` so the OpenVINO decode path is bypassed, and
runs one denoise+commit through the full DiT on Battlemage via OCL.

Gate: the returned latent has the expected ``_flat_shape``, contains
no NaN / Inf, and per-element absolute range stays under 1e3 (the
post-Euler latent is roughly standard-normal so 1e3 is a generous
sanity ceiling; any blown-up kernel-internal output trips this).

This is the lightest possible "does the full DiT graph make it
through the OCL backend without crashing or producing garbage" check
and is the entry point for Phase 3.1 in ``docs/OCL_E2E_PLAN.md``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Mirror the tiny config used in test_intel_smoke.py.
_SMOKE_CFG = {
    "model_type": "test",
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "fourier_dim": 32,
    "channels": 32,
    "patch": [2, 2],
    "height": 4,
    "width": 4,
    "local_window": 4,
    "global_window": 8,
    "global_attn_period": 2,
    "n_buttons": 16,
    "ctrl_conditioning": False,
    "scheduler_sigmas": [1.0, 0.9, 0.5, 0.0],
    "ae_uri": "dummy/dummy",
    "taehv_ae": True,
    "inference_fps": 8,
    "base_fps": 16,
    "temporal_compression": 4,
    "prompt_conditioning": None,
}


def _ocl_available() -> bool:
    try:
        from quark.runtime.sync import _IS_OCL
        return bool(_IS_OCL)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ocl_available(),
    reason="OCL backend not available — skipping (Phase 3 is DEVKIT-only)",
)


@pytest.fixture
def smoke_env(monkeypatch):
    monkeypatch.setenv("QUARK_FORCE_ENGINE", "intel")
    monkeypatch.setenv("QUARK_SKIP_VAE", "1")
    monkeypatch.setattr("quark.engine.intel._resolve_path", lambda uri, **kw: uri)
    monkeypatch.setattr(
        "quark.engine.intel.load_yaml_config", lambda path: dict(_SMOKE_CFG),
    )
    yield


def test_compute_latent_one_frame(smoke_env):
    from quark.engine import Engine, EngineIntel

    with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
        engine = Engine("dummy", load_weights=False)
    assert isinstance(engine, EngineIntel)

    latent_qt = engine._compute_latent(ctrl=None)
    # Drain the lazy queue so to_numpy() returns the final tensor.
    from quark.runtime.sync import synchronize as _sync
    _sync()
    latent_np = latent_qt.to_numpy()

    # Shape sanity: EngineIntel._flat_shape is set at __init__ from
    # channels * height * width / (patch[0] * patch[1]) packed flat.
    assert latent_np.shape == tuple(engine._flat_shape), (
        f"latent shape {latent_np.shape} != expected {tuple(engine._flat_shape)}"
    )

    # bf16 carrier — decode to f32 for nan/inf/range checks.
    f32 = (latent_np.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(f32).all(), (
        f"non-finite values in latent: nans={int(np.isnan(f32).sum())} "
        f"infs={int(np.isinf(f32).sum())}"
    )
    max_abs = float(np.abs(f32).max())
    assert max_abs < 1e3, f"latent max|x|={max_abs:.4e} exceeds 1e3 sanity ceiling"


def test_compute_latent_two_frames_no_crash(smoke_env):
    """Run two consecutive frames — exercises the KV-cache commit path
    (frame_counter advance + frozen=False second call) and the
    pipelined-decode-disabled main thread loop.

    Gate is just "doesn't crash"; numerics already covered by frame 1.
    """
    from quark.engine import Engine

    with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
        engine = Engine("dummy", load_weights=False)
    for _ in range(2):
        latent_qt = engine._compute_latent(ctrl=None)
    from quark.runtime.sync import synchronize as _sync
    _sync()
    latent_np = latent_qt.to_numpy()
    f32 = (latent_np.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(f32).all()
