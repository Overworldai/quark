"""Variable-name allocator for MSL codegen.

Replaces the PTX RegAllocator. MSL uses C-style local variables —
no register classes, no `.reg` declaration blocks. Names are allocated
per-prefix with a simple counter.
"""

from __future__ import annotations

from quark.ir import DType, Value


class NameAlloc:
    """Counter-based local-variable name allocator for MSL.

    Each IR Value.id maps to a single C-style local name. Width-N
    vector Values get N component names (v0_0, v0_1, …) for cases
    where the lowerer decomposes vectors into scalars.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._names: dict[int, tuple[str, ...]] = {}

    def fresh(self, prefix: str) -> str:
        """Allocate a fresh name with the given prefix."""
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        return f"{prefix}{n}"

    def name_for(self, value: Value) -> str:
        """Return the canonical MSL name for a scalar Value.

        First access allocates; subsequent accesses return the same name.
        For width-1 Values, returns a single name. For width-N, returns
        the first component (use `components()` for all of them).
        """
        comps = self._names.get(value.id)
        if comps is None:
            comps = self._allocate(value)
        return comps[0]

    def components(self, value: Value) -> tuple[str, ...]:
        """Return all component names for a Value."""
        comps = self._names.get(value.id)
        if comps is None:
            comps = self._allocate(value)
        return comps

    def bind(self, value: Value, names: tuple[str, ...], *, force: bool = False) -> None:
        """Bind a Value to pre-existing names (for aliasing / carry vars).

        Set ``force=True`` to override an existing binding (used when
        a for-loop carry variable is re-declared from scalar to fragment).
        """
        existing = self._names.get(value.id)
        if existing is not None and existing != names and not force:
            raise RuntimeError(
                f"NameAlloc.bind: Value {value.id} already bound to "
                f"{existing}, cannot rebind to {names}"
            )
        self._names[value.id] = names

    def alias(self, target: Value, source: Value) -> tuple[str, ...]:
        """Bind `target` to `source`'s names."""
        src_names = self.components(source)
        self.bind(target, src_names)
        return src_names

    def alias_component(self, target: Value, source: Value, index: int) -> str:
        """Bind scalar `target` to one component of `source`."""
        src = self.components(source)
        name = src[index]
        self.bind(target, (name,))
        return name

    def has(self, value: Value) -> bool:
        return value.id in self._names

    def _allocate(self, value: Value) -> tuple[str, ...]:
        """Allocate names for a Value based on its dtype."""
        prefix = _PREFIX.get(value.dtype, "v")
        if value.width == 1:
            name = self.fresh(prefix)
            self._names[value.id] = (name,)
            return (name,)
        # Vector: allocate N component names
        base = self.fresh(prefix)
        comps = tuple(f"{base}_{i}" for i in range(value.width))
        self._names[value.id] = comps
        return comps


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
