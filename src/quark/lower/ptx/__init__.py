"""PTX lowerer (NVIDIA).

Walks a quark IR `Module` and emits a PTX text artifact wrapped in a
`.visible .entry` kernel. See `lower.py` for the visitor implementation.
"""

from .lower import LoweredKernel, PtxLowerer
from .regs import RegAllocator, arith_suffix, reg_class

__all__ = [
    "LoweredKernel",
    "PtxLowerer",
    "RegAllocator",
    "arith_suffix",
    "reg_class",
]
