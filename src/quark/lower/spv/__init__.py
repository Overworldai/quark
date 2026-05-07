"""SPIR-V lowerer package (Intel Arc / Xe iGPU).

PORTABILITY_PLAN §3.2 first cut — lowers a quark IR ``Module`` to
SPIR-V text + assembles via ``spirv-as``.

Public surface:

  * :class:`SpirVLowerer` — the lowerer; ``lower_module(module) ->
    LoweredSpirVKernel``.
  * :class:`LoweredSpirVKernel` — the artifact: SPIR-V text +
    launch metadata.
  * :func:`text_to_binary` — assemble the text via ``spirv-as``
    (external CLI; raise ``SpirvAsNotFound`` if missing).

Visitor coverage is intentionally narrow in this first cut (a
``vec_add``-class kernel — scalar arith, scalar load/store, single-
workgroup dispatch via ``LocalInvocationId``). Extending to the
full ~45-op surface is incremental — each additional visitor adds
one entry to ``_DISPATCH`` in ``lower.py``. See ``PORTABILITY_PLAN``
§3.2 for the full table.
"""

from __future__ import annotations

from quark.device import DeviceFamily
from quark.lower.base import register_lowerer

from .assemble import SpirvAsNotFound, text_to_binary
from .lower import LoweredSpirVKernel, SpirVLowerer


@register_lowerer(DeviceFamily.INTEL_GPU)
def _make_spirv_lowerer(caps) -> SpirVLowerer:
    """Adapter: ``DeviceCaps`` → ``SpirVLowerer``. Caps come from a
    Vulkan probe (``drivers.spv.probe``)."""
    return SpirVLowerer(caps)


__all__ = [
    "LoweredSpirVKernel",
    "SpirVLowerer",
    "SpirvAsNotFound",
    "text_to_binary",
]
