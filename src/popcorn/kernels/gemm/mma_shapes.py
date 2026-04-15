"""MMA shape registry — back-compat shim.

The real registry moved to :mod:`popcorn.ir.mma_registry` in M1 of the
MMA_SHAPES proposal. Kernels still import ``MmaConfig`` and
``lookup_mma`` from here during the migration; M4 will remove this
module entirely after every kernel switches to ``mma_dtype_spec`` +
chip-driven shape autotune.

Do not add new descriptors here — edit ``popcorn.ir.mma_registry``.
"""

from __future__ import annotations

from popcorn.ir.mma_registry import (
    _BF16_K16,
    _E4M3_K16,
    _E4M3_K32,
    _E5M2_K16,
    _E5M2_K32,
    _F16_K16,
    MmaConfig,
    _BF16xE4M3_K16,
    lookup_mma,
)

__all__ = (  # noqa: RUF022 — intentionally orders public before private _-prefixed
    "MmaConfig",
    "lookup_mma",
    # Re-exported private descriptors — tests consume them by name to
    # verify offset tables against the PTX ISA. Removed in M4 once the
    # test suite migrates to ``popcorn.ir.mma_registry.ALL_SHAPES``.
    "_BF16_K16",
    "_F16_K16",
    "_E4M3_K16",
    "_E5M2_K16",
    "_E4M3_K32",
    "_E5M2_K32",
    "_BF16xE4M3_K16",
)
