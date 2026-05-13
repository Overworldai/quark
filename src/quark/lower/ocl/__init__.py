"""OpenCL/IGC backend lowerer.

Targets Intel's IGC compiler via OpenCL-flavor SPIR-V (consumed by
``clCreateProgramWithIL`` on Intel NEO / opencl-icd). The dialect:

  ``OpCapability Kernel``
  ``OpMemoryModel Physical64 OpenCL``
  ``OpEntryPoint Kernel``
  ``OpenCL.std`` extended instruction set
  ``OpFunctionParameter`` (CrossWorkgroup pointer) for buffer args
  ``SPV_INTEL_subgroup_matrix_multiply_accumulate`` for the MMA path
"""

from __future__ import annotations

from quark.device import DeviceFamily
from quark.lower.base import register_lowerer

from .lower import LoweredOclSpirVKernel, OpenClSpirVLowerer


@register_lowerer(DeviceFamily.INTEL_GPU)
def _make_ocl_lowerer(caps) -> OpenClSpirVLowerer:
    """Adapter: ``DeviceCaps`` → ``OpenClSpirVLowerer``. Caps come from
    an OCL probe (``quark.drivers.ocl.probe`` →
    ``caps_from_probe``)."""
    return OpenClSpirVLowerer(caps)


__all__ = [
    "LoweredOclSpirVKernel",
    "OpenClSpirVLowerer",
]
