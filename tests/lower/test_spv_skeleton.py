"""Tests pinning the SPIR-V skeleton: registry slot + stub surface.

The SPIR-V lowerer itself is ~3 weeks of engineering behind Intel
Arc / Xe hardware (see docs/PORTABILITY_PLAN.md §3). This file
pins the *scaffolding* the refactor left behind: the registry
slot is reserved, the stub surface raises a clear error, and the
family enum has the expected value. Real SPIR-V output tests
will replace these once the lowerer is implemented.
"""

from __future__ import annotations

import pytest

from quark.device import DeviceFamily
from quark.lower import LOWERERS, get_lowerer
from quark.lower.spv import LoweredSpirVKernel, SpirVLowerer


def test_intel_gpu_family_exists():
    """DeviceFamily has the INTEL_GPU slot (refactors elsewhere rely
    on this enum value)."""
    assert DeviceFamily.INTEL_GPU.value == "intel_gpu"


def test_spv_lowerer_registered():
    """The SPIR-V factory is in the lowerer registry under
    DeviceFamily.INTEL_GPU."""
    assert DeviceFamily.INTEL_GPU in LOWERERS


def test_get_lowerer_returns_spv_lowerer():
    """``get_lowerer`` constructs a ``SpirVLowerer`` for INTEL_GPU.
    Caps can be ``None`` until a real Vulkan / Level Zero probe lands."""
    lowerer = get_lowerer(DeviceFamily.INTEL_GPU, caps=None)
    assert isinstance(lowerer, SpirVLowerer)


def test_lower_module_raises_with_pointed_error():
    """The stub surface raises NotImplementedError pointing at the
    plan — not a silent fallback or a KeyError."""
    lowerer = SpirVLowerer(caps=None)
    with pytest.raises(NotImplementedError, match="PORTABILITY_PLAN.md"):
        lowerer.lower_module(None)


def test_lowered_kernel_constructor_raises():
    """``LoweredSpirVKernel`` is also a placeholder; constructing one
    directly raises the same pointed error."""
    with pytest.raises(NotImplementedError, match="PORTABILITY_PLAN.md"):
        LoweredSpirVKernel()
