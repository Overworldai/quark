"""SPIR-V lowerer package (Intel Arc / Xe iGPU) — skeleton.

Imports ``SpirVLowerer`` and registers it against
``DeviceFamily.INTEL_GPU``. The lowerer itself is a stub that
raises on ``lower_module`` until the phase-3 implementation lands;
the registration keeps the rest of the pipeline consistent
(``Launcher._lower`` uses ``get_lowerer(family, caps)`` blindly,
so having the slot filled even with a stub lets the launcher
surface a meaningful error instead of a KeyError from
``LOWERERS``).
"""

from __future__ import annotations

from quark.device import DeviceFamily
from quark.lower.base import register_lowerer

from .lower import LoweredSpirVKernel, SpirVLowerer


@register_lowerer(DeviceFamily.INTEL_GPU)
def _make_spirv_lowerer(caps) -> SpirVLowerer:
    """Adapter: DeviceCaps → SpirVLowerer. Caps come from a Vulkan or
    Level Zero probe (also TBD)."""
    return SpirVLowerer(caps)


__all__ = [
    "LoweredSpirVKernel",
    "SpirVLowerer",
]
