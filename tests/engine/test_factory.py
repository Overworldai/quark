"""Engine factory dispatch — ``Engine.__new__`` picks the right
concrete subclass for the host platform + ``QUARK_FORCE_ENGINE``
env override.

These tests exercise the dispatch path WITHOUT constructing an
``Engine`` (the subclass ``__init__`` would need a real model URI
+ network access). ``__new__`` returns the appropriate subclass
instance with default-init state; we just check the type.
"""

from __future__ import annotations

import sys

import pytest

from quark.engine import Engine, EngineCUDA, EngineIntel, EngineMetal
from quark.engine.base import _detect_engine_family


class TestDetectEngineFamily:
    def test_darwin_default_is_metal(self, monkeypatch):
        monkeypatch.delenv("QUARK_FORCE_ENGINE", raising=False)
        monkeypatch.delenv("QUARK_PROBE_INTEL_FIRST", raising=False)
        monkeypatch.setattr("quark.engine.base._IS_METAL", True)
        monkeypatch.setattr("quark.engine.base._IS_LINUX", False)
        assert _detect_engine_family() == "metal"

    def test_linux_default_is_cuda(self, monkeypatch):
        monkeypatch.delenv("QUARK_FORCE_ENGINE", raising=False)
        monkeypatch.delenv("QUARK_PROBE_INTEL_FIRST", raising=False)
        monkeypatch.setattr("quark.engine.base._IS_METAL", False)
        monkeypatch.setattr("quark.engine.base._IS_LINUX", True)
        assert _detect_engine_family() == "cuda"

    def test_force_env_overrides_platform_detection(self, monkeypatch):
        monkeypatch.setattr("quark.engine.base._IS_METAL", True)
        for forced in ("cuda", "metal", "intel"):
            monkeypatch.setenv("QUARK_FORCE_ENGINE", forced)
            assert _detect_engine_family() == forced

    def test_force_env_uppercase_normalised(self, monkeypatch):
        monkeypatch.setattr("quark.engine.base._IS_METAL", False)
        monkeypatch.setenv("QUARK_FORCE_ENGINE", "INTEL")
        assert _detect_engine_family() == "intel"

    def test_force_env_unknown_value_falls_through(self, monkeypatch):
        # Garbage values fall back to platform detection rather than
        # erroring — preserves the "set it and forget it" property of
        # downstream test harnesses that pin QUARK_FORCE_ENGINE.
        monkeypatch.setattr("quark.engine.base._IS_METAL", True)
        monkeypatch.setenv("QUARK_FORCE_ENGINE", "rocm")
        assert _detect_engine_family() == "metal"


class TestEngineNew:
    def test_construct_metal_subclass_via_force(self, monkeypatch):
        """``Engine.__new__`` returns an ``EngineMetal`` instance when
        the family resolves to ``metal``. ``__init__`` is *not* called
        when ``__new__`` is invoked directly — the test only checks the
        instance class, not constructed state."""
        monkeypatch.setenv("QUARK_FORCE_ENGINE", "metal")
        instance = Engine.__new__(Engine)
        assert isinstance(instance, EngineMetal)

    def test_construct_cuda_subclass_via_force(self, monkeypatch):
        monkeypatch.setenv("QUARK_FORCE_ENGINE", "cuda")
        instance = Engine.__new__(Engine)
        assert isinstance(instance, EngineCUDA)

    def test_construct_intel_subclass_via_force(self, monkeypatch):
        monkeypatch.setenv("QUARK_FORCE_ENGINE", "intel")
        instance = Engine.__new__(Engine)
        assert isinstance(instance, EngineIntel)

    def test_subclass_construction_skips_factory(self):
        """Constructing the subclass directly bypasses ``__new__``'s
        family dispatch — it just allocates the requested class."""
        # ``EngineIntel.__new__`` falls through to ``object.__new__``
        # because ``cls is Engine`` is False. ``__init__`` is skipped
        # since we call ``__new__`` directly.
        instance = EngineIntel.__new__(EngineIntel)
        assert type(instance) is EngineIntel
        assert isinstance(instance, Engine)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="EngineIntel construction needs Vulkan, Linux-only",
)
class TestEngineIntelConstruction:
    """End-to-end EngineIntel construction — runs ONLY when Vulkan is
    actually available. Skipped on Mac CI; on Linux without an ICD,
    construction itself raises a clear RuntimeError that the rest of
    the suite asserts."""

    def test_raises_when_vulkan_unavailable(self, monkeypatch):
        # Stub `is_available()` False to simulate a Vulkan-less Linux
        # host (CI runner with no ICD installed).
        from quark.drivers import spv
        monkeypatch.setattr(spv, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="Vulkan is not available"):
            EngineIntel(model_uri="dummy", load_weights=False)

    def test_caps_after_construct(self, monkeypatch):
        from quark.drivers import spv
        if not spv.is_available():
            pytest.skip("no Vulkan device on this host")
        # Stub the YAML loader so we don't need a real model URI on disk.
        monkeypatch.setattr(
            "quark.models.config._resolve_path",
            lambda uri: uri,
        )
        monkeypatch.setattr(
            "quark.models.config.load_yaml_config",
            lambda path: {
                "model_type": "test",
                "channels": 32,
                "patch": [2, 2],
                "height": 8,
                "width": 16,
                "d_model": 2048,
                "n_layers": 24,
                "n_heads": 32,
                "n_kv_heads": 16,
                "fourier_dim": 512,
                "ae_uri": "dummy/dummy",
                "scheduler_sigmas": [1.0, 0.9, 0.75, 0.3, 0.0],
                "ctrl_conditioning": False,
            },
        )
        engine = EngineIntel(model_uri="dummy", load_weights=False)
        # Caps should reflect the actual Battlemage probe.
        from quark.device import DeviceFamily
        assert engine.caps.family is DeviceFamily.INTEL_GPU
        # And the inference methods should still raise — stub mode.
        with pytest.raises(NotImplementedError, match="PORTABILITY_PLAN"):
            engine.gen_frame()
