"""Variable-name allocator for MSL codegen.

MSL uses C-style local variables — no register classes, no ``.reg``
declaration blocks. The shared :class:`NameAllocator` provides the
counter-per-prefix mechanics; this module specializes the prefix
selection (dtype-keyed instead of register-class-keyed) and the
vector allocation policy (one base name plus ``_i`` suffixes).
"""

from __future__ import annotations

from quark.ir import DType, Value
from quark.lower._common import NameAllocator


class NameAlloc(NameAllocator):
    """Counter-based local-variable name allocator for MSL.

    Each IR ``Value.id`` maps to one or more C-style local names.
    Scalars get a single name; width-N vectors get a base name plus
    ``_i`` suffixes (``v0_0``, ``v0_1``, …) for cases where the
    lowerer decomposes vectors into scalars.
    """

    def name_for(self, value: Value) -> str:
        """Return the canonical MSL name for ``value``.

        For width-1 Values, returns the single name. For width-N
        returns the *first* component — most MSL visitors iterate
        components explicitly rather than consuming a braced form
        the way PTX does. Use ``components()`` for all N.
        """
        return self.components(value)[0]

    @property
    def _names(self) -> dict[int, tuple[str, ...]]:
        """Back-compat alias for the shared base's ``_components`` dict.

        MSL mma visitors (``quark.lower.msl.mma``) reach into this
        private attribute to re-bind loop-body locals across walks of
        the ``FragApplyOp`` body. Keep the old name available so those
        call sites continue to work without touching the MMA emitter.
        """
        return self._components

    # -- backend-specific bind: supports ``force=`` override -----------

    # MSL keeps the ``force=`` escape hatch inherited from the shared
    # base — no override needed. PTX's subclass explicitly rejects
    # ``force=``; MSL leaves it available for the loop-carry re-
    # declaration path.

    # -- allocation ----------------------------------------------------

    def _allocate_components(self, value: Value) -> tuple[str, ...]:
        prefix = _PREFIX.get(value.dtype, "v")
        if value.width == 1:
            return (self.fresh(prefix),)
        # Vector: one base name + component suffixes. Avoids burning
        # N separate counters and keeps related components clustered
        # in emitted MSL.
        base = self.fresh(prefix)
        return tuple(f"{base}_{i}" for i in range(value.width))


# Prefix per dtype. Uses `_pc_` (quark) namespace to guarantee no
# collisions with MSL keywords, type names (uchar, ushort, half, float,
# bool), stdlib identifiers, or Metal builtin names.
_PREFIX: dict[DType, str] = {
    DType.F64: "_pc_f64_",
    DType.F32: "_pc_f32_",
    DType.F16: "_pc_f16_",
    DType.BF16: "_pc_bf_",
    DType.E4M3: "_pc_e4_",
    DType.E5M2: "_pc_e5_",
    DType.U8: "_pc_u8_",
    DType.S8: "_pc_s8_",
    DType.U16: "_pc_u16_",
    DType.S16: "_pc_s16_",
    DType.U32: "_pc_u32_",
    DType.S32: "_pc_s32_",
    DType.U64: "_pc_u64_",
    DType.S64: "_pc_s64_",
    DType.B16: "_pc_b16_",
    DType.B32: "_pc_b32_",
    DType.B64: "_pc_b64_",
    DType.PRED: "_pc_p_",
}
