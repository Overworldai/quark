"""PTX register allocator for the IR lowerer.

Every IR `Value` gets mapped to one or more PTX register names. A
scalar Value (width=1) has a single name; a vector Value (width=N)
has N component names that can be emitted either as a braced form
for vector-typed PTX instructions (`ld.global.v4.b32 {r0,r1,r2,r3}`)
or individually for scalar ops that consume one component.

The allocator groups names by register class so they can be emitted
in one `.reg .f32 %f0, %f1, ...;` block at the top of the kernel body.

Aliasing is supported: a Value can be bound to registers that were
already allocated, so that the backedge of a loop, the yield of an
if/else, a `VecBuildOp` (which just groups four scalar Values into
one vec Value), or a `VecExtractOp` (which exposes one component of
a vec as a scalar) can share storage without emitting spurious movs.
"""

from __future__ import annotations

from popcorn.ir import DType, Value

# ---------------------------------------------------------------------------
# DType → PTX register class + operation suffix
# ---------------------------------------------------------------------------

# The register class is what appears in a `.reg .f32 %f0;` declaration.
# Most arithmetic ops use the same suffix; a few (e.g. bitwise) prefer a
# bit-typed suffix. The lowerer calls `op_suffix(dtype, kind)` to pick
# the right one.

_REG_CLASS: dict[DType, str] = {
    DType.F64: "f64",
    DType.F32: "f32",
    DType.F16: "b16",  # fp16 values live in b16 regs in PTX
    DType.BF16: "b16",  # same for bf16
    DType.E4M3: "b8",
    DType.E5M2: "b8",
    DType.U64: "u64",
    DType.S64: "s64",
    DType.U32: "u32",
    DType.S32: "s32",
    DType.U16: "u16",
    DType.S16: "s16",
    DType.U8: "u8",
    DType.S8: "s8",
    DType.B16: "b16",
    DType.B32: "b32",
    DType.B64: "b64",
    DType.PRED: "pred",
}


def reg_class(dtype: DType) -> str:
    """Return the PTX register class string for a DType (no leading dot)."""
    try:
        return _REG_CLASS[dtype]
    except KeyError as e:
        raise NotImplementedError(f"PTX: no register class for {dtype}") from e


def arith_suffix(dtype: DType) -> str:
    """PTX arith-instruction suffix for `dtype`.

    Mostly mirrors `reg_class`, but F16/BF16 use their own native arith
    suffix even though their register class is `b16`. Most arith on
    subword FP goes through f32 anyway (via ConvertOp), so we leave the
    fp16/bf16 arith path for a later pass.
    """
    if dtype in (DType.F16, DType.BF16):
        return dtype.value  # "f16" or "bf16"
    return reg_class(dtype)


# Prefix used for each reg class when allocating names. Keeps emitted
# PTX readable (e.g. %f12 vs %r7 vs %p3) and avoids name clashes
# between classes.
_PREFIX: dict[str, str] = {
    "f64": "fd",
    "f32": "f",
    "u64": "rd",
    "s64": "rs",
    "u32": "r",
    "s32": "rs",
    "u16": "rh",
    "s16": "rhs",
    "u8": "rb",
    "s8": "rsb",
    "b16": "h",
    "b32": "b",
    "b64": "bd",
    "b8": "rb",
    "pred": "p",
}


def _prefix_for(cls: str) -> str:
    try:
        return _PREFIX[cls]
    except KeyError as e:
        # Strict: unknown classes here used to silently fall through to
        # the `"r"` prefix, which then produced `.reg .r %r0, ...;` —
        # invalid PTX that only ptxas would catch. Reject loudly at
        # emit time so typo'd callers surface immediately.
        raise ValueError(
            f"RegAllocator: unknown register class {cls!r}; valid classes: {sorted(_PREFIX)}"
        ) from e


# ---------------------------------------------------------------------------
# RegAllocator
# ---------------------------------------------------------------------------


class RegAllocator:
    """Per-Function PTX register allocator.

    Terminology:
      - "components" — the tuple of reg names behind one Value. Scalars
        have len=1, vectors have len=width.
      - "name" — the canonical textual form for a Value at a use site:
        a single `%f0` for a scalar or `{%f0, %f1, %f2, %f3}` for a vec
        (the braced form accepted by `ld.global.v4.*`).

    Public API:
      - `name_for(value)` → str (canonical form)
      - `components(value)` → tuple[str, ...]
      - `bind(value, names)` — register `value` with pre-allocated names
      - `alias(target, source)` — target shares source's components
      - `declare(cls)` — allocate an anonymous scratch reg
      - `declarations()` → list of `.reg` decl lines
    """

    def __init__(self) -> None:
        # Per-value component tuple.
        self._components: dict[int, tuple[str, ...]] = {}
        # Per-class list of declared names for the .reg block.
        self._by_class: dict[str, list[str]] = {}
        self._counter: dict[str, int] = {}

    # -- allocation ----------------------------------------------------

    def declare(self, cls: str) -> str:
        """Allocate one fresh anonymous register in `cls`."""
        n = self._counter.get(cls, 0)
        self._counter[cls] = n + 1
        name = f"%{_prefix_for(cls)}{n}"
        self._by_class.setdefault(cls, []).append(name)
        return name

    def name_for(self, value: Value) -> str:
        """Return the canonical PTX form for a Value.

        Scalars → `%f0`. Vectors → `{%f0, %f1, %f2, %f3}`. First access
        lazily allocates component registers; subsequent accesses are
        stable."""
        comps = self._components.get(value.id)
        if comps is None:
            comps = self._allocate_components(value)
            self._components[value.id] = comps
        if len(comps) == 1:
            return comps[0]
        return "{" + ", ".join(comps) + "}"

    def components(self, value: Value) -> tuple[str, ...]:
        """Return the tuple of component reg names for a Value,
        allocating on first access."""
        comps = self._components.get(value.id)
        if comps is None:
            comps = self._allocate_components(value)
            self._components[value.id] = comps
        return comps

    def _allocate_components(self, value: Value) -> tuple[str, ...]:
        cls = reg_class(value.dtype)
        return tuple(self.declare(cls) for _ in range(value.width))

    # -- binding / aliasing -------------------------------------------

    def bind(self, value: Value, names: tuple[str, ...]) -> None:
        """Register `value`'s components as `names`. No new regs are
        allocated. Fails if `value` already has a binding."""
        existing = self._components.get(value.id)
        if existing is not None and existing != names:
            raise RuntimeError(
                f"RegAllocator.bind: Value {value.id} already bound to "
                f"{existing}, cannot rebind to {names}"
            )
        if len(names) != value.width:
            raise ValueError(f"RegAllocator.bind: Value width {value.width} vs {len(names)} names")
        self._components[value.id] = names

    def alias(self, target: Value, source: Value) -> tuple[str, ...]:
        """Bind `target` to `source`'s component tuple."""
        src_components = self.components(source)
        if target.width != source.width:
            # Allow a scalar target aliased onto one component of a vec
            # only via `alias_component` — not this method.
            raise ValueError(
                f"RegAllocator.alias: width mismatch — source width "
                f"{source.width}, target width {target.width}"
            )
        self.bind(target, src_components)
        return src_components

    def alias_component(self, target: Value, source: Value, component_index: int) -> str:
        """Bind scalar `target` to one component of vec `source`."""
        if target.width != 1:
            raise ValueError("RegAllocator.alias_component: target must be a scalar Value")
        src_components = self.components(source)
        if not (0 <= component_index < len(src_components)):
            raise IndexError(
                f"RegAllocator.alias_component: index {component_index} "
                f"out of range for width {len(src_components)}"
            )
        name = src_components[component_index]
        self.bind(target, (name,))
        return name

    # -- introspection -------------------------------------------------

    def has(self, value: Value) -> bool:
        return value.id in self._components

    def declarations(self) -> list[str]:
        """Return sorted `.reg` declaration lines, one per class."""
        lines: list[str] = []
        for cls in sorted(self._by_class):
            names = self._by_class[cls]
            if not names:
                continue
            lines.append(f"    .reg .{cls} {', '.join(names)};")
        return lines
