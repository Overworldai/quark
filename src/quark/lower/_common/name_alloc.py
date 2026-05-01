"""Shared counter-per-prefix name allocator.

Every backend lowerer needs the same core machinery:

  * unique-name generation keyed by some backend-specific "class"
    string (PTX register class, MSL dtype prefix, SPIR-V storage
    class / type),
  * an IR ``Value.id`` → ``tuple[str, ...]`` component map so the
    second visit to a Value returns the same name,
  * ``bind`` / ``alias`` / ``alias_component`` / ``has`` helpers
    for sub-register carveouts, loop carries, and vec/scalar
    plumbing.

``NameAllocator`` carves out that shared surface. Backend allocators
(``quark.lower.ptx.regs.RegAllocator``, ``quark.lower.msl.names.NameAlloc``)
subclass it and only implement the backend-specific bits:

  * ``_class_for(value)`` — which prefix/class bucket a Value's
    components land in,
  * ``_allocate_components(value)`` — how many physical names back
    a vec Value (PTX: one per component; MSL: one base name plus
    ``_i`` suffixes).
  * (PTX only) ``name_for`` override for the ``{a, b, c, d}``
    braced vector form that PTX vec ops require.

Zero behavior change from the previous stand-alone implementations —
the shared base is factored-out mechanics, not a semantic rewrite.
"""

from __future__ import annotations

from quark.ir import Value


class NameAllocator:
    """Counter-keyed name allocator with per-Value component tracking.

    Subclasses:

      * override ``_class_for(value)`` to pick the class key used when
        allocating components,
      * override ``_allocate_components(value)`` to decide how many
        names a vec Value needs and how they're spelled,
      * optionally override ``name_for`` when the canonical form at
        a use site is something other than the first component.

    Everything else — ``fresh``, ``components``, ``bind``, ``alias``,
    ``alias_component``, ``has`` — lives here.
    """

    def __init__(self) -> None:
        # prefix → next index. Each prefix (or class) has its own
        # counter so the emitted names stay readable (``%f0 %f1 …``
        # rather than a single monotonic counter that interleaves).
        self._counters: dict[str, int] = {}
        # Value.id → tuple of physical names. Scalars have len 1; vecs
        # have len equal to the physical register / variable count
        # (which may differ from ``value.width`` when packed storage
        # is used — e.g. bf16x2 packed into one b32).
        self._components: dict[int, tuple[str, ...]] = {}

    # -- allocation ----------------------------------------------------

    def fresh(self, prefix: str) -> str:
        """Allocate one unique name in ``prefix``'s bucket."""
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        return self._format_name(prefix, n)

    def _format_name(self, prefix: str, n: int) -> str:
        """Default: ``{prefix}{n}``. Override for backends that decorate
        names (PTX's leading ``%`` lives in its subclass)."""
        return f"{prefix}{n}"

    # -- per-Value lookup ---------------------------------------------

    def name_for(self, value: Value) -> str:
        """Canonical name for ``value`` at a use site.

        Default: the first component (matches MSL / any C-style backend).
        PTX overrides to emit the braced vec form when len > 1.
        """
        return self.components(value)[0]

    def components(self, value: Value) -> tuple[str, ...]:
        """Tuple of component names for ``value``, allocating on first
        access."""
        comps = self._components.get(value.id)
        if comps is None:
            comps = self._allocate_components(value)
            self._components[value.id] = comps
        return comps

    def has(self, value: Value) -> bool:
        return value.id in self._components

    # -- subclass hooks ------------------------------------------------

    def _allocate_components(self, value: Value) -> tuple[str, ...]:
        """Backend-specific: return the physical names backing ``value``.

        Subclasses must implement. PTX allocates one register per
        logical component; MSL allocates one base name with ``_i``
        suffixes for width > 1.
        """
        raise NotImplementedError("NameAllocator._allocate_components")

    # -- binding / aliasing -------------------------------------------

    def bind(self, value: Value, names: tuple[str, ...], *, force: bool = False) -> None:
        """Register ``value``'s components as the given names.

        By default refuses to rebind if the Value is already bound to
        a different tuple (catches accidental double-binding in the
        visitor path). ``force=True`` overrides — used when a loop
        carry variable is re-declared from scalar to vector form."""
        existing = self._components.get(value.id)
        if existing is not None and existing != names and not force:
            raise RuntimeError(
                f"{type(self).__name__}.bind: Value {value.id} already bound to "
                f"{existing}, cannot rebind to {names}"
            )
        self._components[value.id] = names

    def alias(self, target: Value, source: Value) -> tuple[str, ...]:
        """Bind ``target`` to ``source``'s components. Widths must match
        (use ``alias_component`` for scalar-of-vec)."""
        src_components = self.components(source)
        if target.width != source.width:
            raise ValueError(
                f"{type(self).__name__}.alias: width mismatch — source width "
                f"{source.width}, target width {target.width}"
            )
        self.bind(target, src_components)
        return src_components

    def alias_component(self, target: Value, source: Value, index: int) -> str:
        """Bind scalar ``target`` to one component of vec ``source``."""
        if target.width != 1:
            raise ValueError(f"{type(self).__name__}.alias_component: target must be scalar")
        src_components = self.components(source)
        if not (0 <= index < len(src_components)):
            raise IndexError(
                f"{type(self).__name__}.alias_component: index {index} "
                f"out of range for width {len(src_components)}"
            )
        name = src_components[index]
        self.bind(target, (name,))
        return name
