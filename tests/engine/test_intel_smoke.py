"""EngineIntel construction smoke — Mac-compatible.

Exercises the factory dispatch + EngineIntel.__init__ wiring (model
build, quant resolution, weight pin to QuarkTensor, scratch buffer
alloc) without needing an Intel GPU or OpenVINO. Uses a tiny
synthetic config, skips weight load (``load_weights=False``), and
sets ``QUARK_SKIP_VAE=1`` so the VAE block early-returns.

On Mac, the underlying ``QuarkTensor`` storage routes to the Metal
pool (the auto-detect picks ``_MetalStorage`` whenever ``_IS_METAL``
is true regardless of ``QUARK_FORCE_ENGINE``). That's fine — this
test checks engine-level wiring, not the OCL storage path. The OCL
storage path is exercised by ``tests/lower/ocl`` and the [DEVKIT]
smokes in the OCL-E2E plan.
"""

from __future__ import annotations

import pytest

from quark.engine import Engine, EngineIntel
from quark.models.waypoint_15 import QuantConfig
from quark.runtime.tensor import QuarkTensor

# Tiny config — every dim shrunk to keep model construction fast.
# Honors Waypoint15Config defaults for everything not overridden.
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


@pytest.fixture
def smoke_env(monkeypatch):
    """Stub config load + path resolve; force the intel engine;
    skip VAE; suppress the no-fp8 RuntimeWarning we expect."""
    monkeypatch.setenv("QUARK_FORCE_ENGINE", "intel")
    monkeypatch.setenv("QUARK_SKIP_VAE", "1")
    # intel.py imports these at module level, so monkeypatch the
    # bindings inside intel.py's namespace (not the source module).
    monkeypatch.setattr("quark.engine.intel._resolve_path", lambda uri, **kw: uri)
    monkeypatch.setattr(
        "quark.engine.intel.load_yaml_config", lambda path: dict(_SMOKE_CFG),
    )
    yield


def _is_quark_tensor(t) -> bool:
    return isinstance(t, QuarkTensor)


class TestEngineIntelSmoke:
    def test_constructs_via_factory(self, smoke_env):
        """``Engine(...)`` dispatches to EngineIntel and runs through
        __init__ without raising."""
        with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
            engine = Engine("dummy", load_weights=False)
        assert isinstance(engine, EngineIntel)

    def test_model_built_and_quant_forced(self, smoke_env):
        with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
            engine = Engine("dummy", load_weights=False)
        assert engine.model is not None
        assert engine.cfg.quant == QuantConfig.all_bf16()

    def test_vae_skipped(self, smoke_env):
        """``QUARK_SKIP_VAE=1`` leaves the VAE handles None — the
        engine is unusable for gen_frame in this mode, which is the
        point. The smoke just validates the dispatch wiring."""
        with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
            engine = Engine("dummy", load_weights=False)
        assert engine._taehv is None
        assert engine._pipe is None
        assert engine._vae_compute_units is None

    def test_params_pinned_to_quark_tensor(self, smoke_env):
        """After construction, every nn.Parameter on the model is a
        QuarkTensor (no numpy carriers left). The state buffers on
        ``KVCacheUpdate`` (``K_cache`` / ``Vt_cache`` / ``segments``
        / ``n_segments`` / ``frame_t`` / ``frozen``) get the same
        treatment."""
        with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
            engine = Engine("dummy", load_weights=False)
        from quark.nn.module import Module as _NN
        from quark.nn.module import Parameter as _NN_Param

        leaks = []

        def walk(mod, prefix=""):
            for name, val in mod._own_members():
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(val, _NN_Param):
                    if not _is_quark_tensor(val.data):
                        leaks.append((path, type(val.data).__name__))
                elif isinstance(val, _NN):
                    walk(val, path)

        walk(engine.model)
        assert not leaks, f"Parameters not pinned to QuarkTensor: {leaks[:5]}"

    def test_scratch_buffers_are_quark_tensor(self, smoke_env):
        with pytest.warns(RuntimeWarning, match="forcing QuantConfig"):
            engine = Engine("dummy", load_weights=False)
        assert _is_quark_tensor(engine._noise_qt)
        assert _is_quark_tensor(engine._frame_t_qt)
