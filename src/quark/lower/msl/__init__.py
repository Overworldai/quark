"""MSL lowerer (Apple Metal).

Walks a quark IR `Module` and emits an MSL kernel body for the
metal-cpp + nanobind driver (`drivers/metal.py`). See `lower.py` for
the visitor implementation.
"""

from quark.device import DeviceFamily
from quark.lower.base import register_lowerer

from .lower import LoweredMslKernel, MslLowerer
from .names import NameAlloc
from .types import msl_type


@register_lowerer(DeviceFamily.METAL)
def _make_msl_lowerer(caps) -> MslLowerer:
    """Adapter: DeviceCaps → MslLowerer."""
    return MslLowerer(caps)


__all__ = [
    "LoweredMslKernel",
    "MslLowerer",
    "NameAlloc",
    "msl_type",
]
