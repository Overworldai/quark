"""Engine factory dispatch — ``Engine.__new__`` picks the right
concrete subclass for the host platform + ``QUARK_FORCE_ENGINE``
env override.

These tests exercise the dispatch path WITHOUT constructing an
``Engine`` (the subclass ``__init__`` would need a real model URI
+ network access). ``__new__`` returns the appropriate subclass
instance with default-init state; we just check the type.
"""

from __future__ import annotations

from quark.engine import Engine, EngineCUDA, EngineIntel, EngineMetal
from quark.engine.base import _detect_engine_family


class TestDetectEngineFamily:
    def test_darwin_default_is_metal(self, monkeypatch):
        monkeypatch.delenv("QUARK_FORCE_ENGINE", raising=False)
        monkeypatch.setattr("quark.engine.base._IS_METAL", True)
        monkeypatch.setattr("quark.engine.base._IS_LINUX", False)
        assert _detect_engine_family() == "metal"

    def test_linux_default_is_cuda_when_no_ocl(self, monkeypatch):
        monkeypatch.delenv("QUARK_FORCE_ENGINE", raising=False)
        monkeypatch.setattr("quark.engine.base._IS_METAL", False)
        monkeypatch.setattr("quark.engine.base._IS_LINUX", True)
        monkeypatch.setattr("quark.runtime.sync._IS_OCL", False)
        assert _detect_engine_family() == "cuda"

    def test_linux_default_is_intel_when_ocl_available(self, monkeypatch):
        """``_IS_OCL=True`` on Linux flips the default from cuda to
        intel — the OCL backend probed and found a device."""
        monkeypatch.delenv("QUARK_FORCE_ENGINE", raising=False)
        monkeypatch.setattr("quark.engine.base._IS_METAL", False)
        monkeypatch.setattr("quark.engine.base._IS_LINUX", True)
        monkeypatch.setattr("quark.runtime.sync._IS_OCL", True)
        assert _detect_engine_family() == "intel"

    def test_force_env_overrides_platform_detection(self, monkeypatch):
        monkeypatch.setattr("quark.engine.base._IS_METAL", True)
        for forced in ("cuda", "metal", "intel"):
            monkeypatch.setenv("QUARK_FORCE_ENGINE", forced)
            assert _detect_engine_family() == forced

    def test_force_env_uppercase_normalised(self, monkeypatch):
        monkeypatch.setattr("quark.engine.base._IS_METAL", False)
        monkeypatch.setattr("quark.engine.base._IS_LINUX", False)
        monkeypatch.setenv("QUARK_FORCE_ENGINE", "CUDA")
        assert _detect_engine_family() == "cuda"

    def test_force_env_unknown_value_falls_through(self, monkeypatch):
        # Garbage values fall back to platform detection rather than
        # erroring — preserves the "set it and forget it" property of
        # downstream test harnesses that pin QUARK_FORCE_ENGINE.
        monkeypatch.setattr("quark.engine.base._IS_METAL", True)
        monkeypatch.setattr("quark.engine.base._IS_LINUX", False)
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
