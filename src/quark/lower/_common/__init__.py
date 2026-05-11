"""Cross-backend helpers shared between PTX / MSL / (future SPIR-V).

These modules carry no backend-specific knowledge — anything here has
to work for every lowerer. Backend-specific bits (PTX register
classes, MSL dtype prefixes, SPIR-V storage classes) live inside the
respective backend package and compose with the shared helpers.
"""

from quark.lower._common.name_alloc import NameAllocator

__all__ = ["NameAllocator"]
