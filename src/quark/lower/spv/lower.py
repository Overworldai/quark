"""SPIR-V compute-shader lowerer (Intel Arc / Xe iGPU).

**Status: skeleton.** This module exists so the registry has a
slot for ``DeviceFamily.INTEL_GPU`` — the legalization pass, the
per-backend mnemonic tables, and the caps flag split all already
assume this slot will exist, and adding an enum value without a
lowerer would leave a dangling reference. The real implementation
is ~3 weeks of work behind Intel hardware and a chosen runtime
(Vulkan vs Level Zero vs pyopencl).

Calling :meth:`SpirVLowerer.lower_module` raises
``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any


class LoweredSpirVKernel:
    """Placeholder for the SPIR-V-side ``LoweredKernel`` equivalent.

    Will carry the SPIR-V binary blob, entry point name, and launch
    metadata once the lowerer is implemented. Shape mirrors
    ``LoweredKernel`` (PTX) and ``LoweredMslKernel`` (MSL).
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "SpirVLowerer: LoweredSpirVKernel is a skeleton — "
            "the SPIR-V backend has not been implemented yet."
        )


class SpirVLowerer:
    """SPIR-V compute-shader lowerer — skeleton.

    The eventual visitor set mirrors ``PtxLowerer`` / ``MslLowerer``
    with three additions that have no PTX analogue:

      * ``OpCooperativeMatrixLoadKHR`` / ``StoreKHR`` / ``MulAddKHR``
        for the ``LoadMatrixOp`` / ``StoreMatrixOp`` / ``MmaOp`` path.
      * Driver-pinned ``reqd_sub_group_size(32)`` so the v1 plan
        doesn't need per-kernel variants.
      * ``FragApplyOp`` lowering gated on body shape
        (coordinate-free element body only).

    The constructor accepts ``DeviceCaps`` so the factory in
    ``lower/spv/__init__.py`` can pass the queried Vulkan
    ``VkPhysicalDeviceSubgroupProperties`` / cooperative-matrix
    properties through. For now it stashes caps and rejects
    ``lower_module`` calls.
    """

    def __init__(self, caps: Any = None) -> None:
        self.caps = caps

    def lower_module(self, module: Any) -> LoweredSpirVKernel:
        raise NotImplementedError(
            "SpirVLowerer.lower_module: SPIR-V backend is scaffolded but "
            "not yet implemented. The registry slot, legalization pass, "
            "and caps split that make this backend a one-package "
            "addition are in place; the lowerer body itself is the next "
            "engineering bundle. This error fires only on devices "
            "probed as DeviceFamily.INTEL_GPU — no currently-wired "
            "hardware trips it."
        )
