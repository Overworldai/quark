"""Tests pinning the SPIR-V backend scaffolding.

Originally pinned the §3.0 "stub" surface (NotImplementedError); the
§3.2 first-cut lowerer replaced the stubs with a working
``vec_add``-class lowerer (see ``tests/lower/spv/test_lower.py`` for
the visitor-level + end-to-end coverage). What this file now pins is
the *registration* contract — the family enum + registry slot + the
adapter shape the launcher consumes — independent of which visitors
exist.
"""

from __future__ import annotations

from quark.device import DeviceFamily
from quark.lower import LOWERERS, get_lowerer
from quark.lower.spv import LoweredSpirVKernel, SpirVLowerer


def test_intel_gpu_family_exists():
    """``DeviceFamily`` has the ``INTEL_GPU`` slot (refactors elsewhere
    rely on this enum value)."""
    assert DeviceFamily.INTEL_GPU.value == "intel_gpu"


def test_spv_lowerer_registered():
    """The SPIR-V factory is in the lowerer registry under
    ``DeviceFamily.INTEL_GPU``."""
    assert DeviceFamily.INTEL_GPU in LOWERERS


def test_get_lowerer_returns_spv_lowerer():
    """``get_lowerer`` constructs a ``SpirVLowerer`` for ``INTEL_GPU``.
    Caps can be ``None`` until a real probe is wired."""
    lowerer = get_lowerer(DeviceFamily.INTEL_GPU, caps=None)
    assert isinstance(lowerer, SpirVLowerer)


def test_lowered_kernel_dataclass_shape():
    """``LoweredSpirVKernel`` carries SPIR-V text + launch metadata.

    The launcher consumes ``source`` / ``entry_name`` / ``n_buffers``
    / ``push_constants_size`` / ``smem_bytes`` / ``local_size`` to
    populate ``SpvDriver.compile`` + ``.launch``. Any rename of those
    fields needs a coordinated launcher edit; the test catches drift
    early.
    """
    kernel = LoweredSpirVKernel(source="; placeholder")
    assert kernel.source == "; placeholder"
    assert kernel.entry_name == "main"
    assert kernel.n_buffers == 0
    assert kernel.push_constants_size == 0
    assert kernel.smem_bytes == 0
    assert kernel.local_size == (1, 1, 1)
