"""PTX lowerer (NVIDIA).

Walks a quark IR `Module` and emits a PTX text artifact wrapped in a
`.visible .entry` kernel. See `lower.py` for the visitor implementation.
"""

from quark.device import DeviceFamily
from quark.lower.base import register_lowerer

from .lower import LoweredKernel, PtxLowerer
from .regs import RegAllocator, arith_suffix, reg_class


@register_lowerer(DeviceFamily.CUDA)
def _make_ptx_lowerer(caps) -> PtxLowerer:
    """Adapter: DeviceCaps → PtxLowerer with target_sm derived from CC."""
    cc = caps.compute_capability
    target_sm = (cc[0] * 10 + cc[1]) if cc is not None else 89
    return PtxLowerer(target_sm=target_sm, caps=caps)


__all__ = [
    "LoweredKernel",
    "PtxLowerer",
    "RegAllocator",
    "arith_suffix",
    "reg_class",
]
