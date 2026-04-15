"""MSL lowerer (Apple Metal).

Walks a popcorn IR `Module` and emits an MSL kernel body suitable for
`mx.fast.metal_kernel`. See `lower.py` for the visitor implementation.
"""

from .lower import LoweredMslKernel, MslLowerer
from .names import NameAlloc
from .types import msl_type

__all__ = [
    "LoweredMslKernel",
    "MslLowerer",
    "NameAlloc",
    "msl_type",
]
