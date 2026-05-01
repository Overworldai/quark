"""SPIR-V compute-shader lowerer (Intel Arc / Xe iGPU).

**Status: skeleton.** PORTABILITY_PLAN.md §3 describes the full
scope; the real implementation is ~3 weeks of work behind Intel
hardware and a chosen runtime (Vulkan vs Level Zero vs pyopencl).
This module exists so the registry has a slot for
``DeviceFamily.INTEL_GPU`` — every phase-1 and phase-2 refactor
already assumes this slot will exist (legalization pass, capture
of per-backend mnemonics, caps flag split), and adding an enum
value without a lowerer would leave a dangling reference.

Calling :meth:`SpirVLowerer.lower_module` raises
``NotImplementedError`` with a message pointing the maintainer at
the plan. The first cohort of kernels (per §3.5 v1) will drop in
here when the prototype converges.
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
            "SpirVLowerer: LoweredSpirVKernel is a skeleton. See "
            "docs/PORTABILITY_PLAN.md §3.2 for the implementation plan."
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
        (coordinate-free element body only — §3.3 in the plan).

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
            "not yet implemented. The phase-1/2 refactors landed the "
            "registry slot, legalization pass, and caps split that make "
            "this backend a one-package addition; the lowerer itself is "
            "the next engineering bundle. See docs/PORTABILITY_PLAN.md "
            "§3 for the full plan (driver choice, cooperative-matrix "
            "shape probing, FragApplyOp strategy). This error will fire "
            "only on devices probed as DeviceFamily.INTEL_GPU — no "
            "currently-wired hardware trips it."
        )
