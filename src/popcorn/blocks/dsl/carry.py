"""Stage + Carry — typed per-iteration stage and named loop carry.

Both are attribute-addressable containers:

* ``Stage(k=..., vt=...)`` wraps the per-iteration smem tiles so
  ``ictx.stage.k`` / ``ictx.stage.vt`` typechecks.
* ``Carry(o=..., m=..., l=...)`` wraps the loop-carried state; slots
  expose attribute access (``carry.o``) and flatten / rebind under the
  hood so ``pop.for_range``'s ``carried=`` API still sees a flat tuple.
"""

from __future__ import annotations

from typing import Any

from popcorn.blocks.dsl.accumulators import Accumulators
from popcorn.ir import Value


class Stage:
    """Attribute-addressable stage container for pipeline iterations.

    Kernels bundle per-iteration smem tiles into a Stage so consume
    callbacks can read them by name::

        stages = [Stage(k=..., vt=...) for _ in range(n_stages)]

        def consume(ictx):
            ictx.stage.k.load_from(...)
            ictx.stage.vt.load_from(...)

    Subscript access (``stage["k"]``) is also supported for back-compat
    with the dict-shaped stages that pre-date this class.
    """

    def __init__(self, **tiles: Any) -> None:
        object.__setattr__(self, "_tiles", dict(tiles))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        tiles = object.__getattribute__(self, "_tiles")
        if name not in tiles:
            raise AttributeError(f"Stage has no tile {name!r} — declared tiles: {list(tiles)}")
        return tiles[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._tiles[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._tiles[key]

    def __contains__(self, key: str) -> bool:
        return key in self._tiles

    def __iter__(self) -> Any:
        return iter(self._tiles)

    @classmethod
    def staged(cls, n: int, **tile_specs: Any) -> list[Stage]:
        """Build ``n`` Stages from shared tile specs — the pipeline
        rotation factory. Replaces the per-kernel stage-loop boilerplate::

            stages = Stage.staged(
                c.n_stages,
                k=SmemTile.spec(s.b_dtype, (KvTile, Dh), pad=c.KvPad, lane_col_step=lcs),
                vt=SmemTile.spec(s.b_dtype, (Dh, KvTile), pad=c.KvPad, lane_col_step=lcs),
            )

        Each spec is materialized into a uniquely-named :class:`SmemTile`
        per stage (``K_s0``, ``K_s1``, ``Vt_s0``, ``Vt_s1``, …). The
        name is built from the tile key (capitalised, underscore-preserved)
        plus a ``_s{i}`` suffix matching the legacy convention so smem
        layout / dump output stays unchanged.

        Values that aren't :class:`SmemTileSpec` are passed through
        verbatim to every stage (shared across all stages) — useful for
        per-stage scratch that happens to be identical across stages.
        """
        from popcorn.blocks.dsl.smem_tile import SmemTile, SmemTileSpec

        stages: list[Stage] = []
        for i in range(n):
            tiles: dict[str, Any] = {}
            for key, spec in tile_specs.items():
                if isinstance(spec, SmemTileSpec):
                    tile_name = f"{_stage_name(key)}_s{i}"
                    tiles[key] = SmemTile(
                        name=tile_name,
                        dtype=spec.dtype,
                        shape=spec.shape,
                        pad=spec.pad,
                        lane_col_step=spec.lane_col_step,
                    )
                else:
                    tiles[key] = spec
            stages.append(cls(**tiles))
        return stages


def _stage_name(key: str) -> str:
    """``k`` → ``K``, ``vt`` → ``Vt``, ``a_tile`` → ``A_tile`` — mirror
    the legacy ``K_s{i}`` / ``Vt_s{i}`` naming."""
    if not key:
        return key
    return key[0].upper() + key[1:]


class Carry:
    """Named multi-subset loop carry with attribute access.

    The simple GEMM-shape uses ``carry=acc`` directly — one
    Accumulators in, final Values on ``acc.results``. Attention-style
    kernels carry multiple disjoint groups (``O`` accumulator grid +
    per-row-class ``m`` / ``l`` scalar vectors). ``Carry`` gives each
    group a name you can read and write directly::

        carry = Carry(
            o=o_acc,                          # Accumulators
            m=(n_ml, -1e30, DType.F32),       # (count, init_value, dtype)
            l=(n_ml,  0.0,  DType.F32),
        )

        def consume(ictx):
            carry = ictx.carry                # a Carry instance
            # compute with carry.o, carry.m, carry.l (each is list[Value])
            carry.o = new_o
            carry.m = new_m
            carry.l = new_l
            return carry                      # consume returns the Carry

        final = run_pipeline(body=PipelineBody(..., carry=carry), ...)
        # `final` is the same Carry, rebound to the loop's final values;
        # epilogue reads `final.o`, `final.l` directly.

    Supported slot spec forms (any mix in one Carry):

    * :class:`Accumulators` — emits ``MT * NT`` loop-init vec Values.
    * ``(count, init_value, dtype)`` — scalar array; ``count`` copies
      of ``pop.const(dtype, init_value)`` at init time.
    * ``list[Value]`` / tuple of Values — pre-built, no init callback.
    * Single ``Value`` — a one-slot scalar.

    The carry flattens to a tuple under the hood (``for_range``'s
    ``carried`` API still wants a flat tuple of Values); ``Carry`` hides
    that by flattening on yield and re-binding on loop-body entry.
    """

    def __init__(self, **slots: Any) -> None:
        # Parse slot specs up-front; cache per-slot ``count`` and an
        # init callable so ``.init()`` can emit the flat init tuple
        # under an active Builder. Accepts four shapes (see class doc).
        specs: dict[str, tuple[int, Any]] = {}
        values: dict[str, list[Value]] = {}
        raw: dict[str, Any] = {}
        for name, spec in slots.items():
            count, init_fn, current = self._parse_spec(spec)
            specs[name] = (count, init_fn)
            raw[name] = spec
            if current is not None:
                values[name] = current
        # Bypass our __setattr__ — these are internal state.
        object.__setattr__(self, "_specs", specs)
        object.__setattr__(self, "_values", values)
        # Keeps the original spec objects so ``run_pipeline`` can find
        # Accumulators slots post-loop and stash their ``.results``.
        object.__setattr__(self, "_raw_specs", raw)

    @staticmethod
    def _parse_spec(spec: Any) -> tuple[int, Any, list[Value] | None]:
        """Return (count, init_fn(bld) -> list[Value], current-values-or-None)."""
        if isinstance(spec, Accumulators):
            acc = spec

            def _init_fn(bld: Any, _acc: Accumulators = acc) -> list[Value]:
                return list(_acc.init())

            return acc.count(), _init_fn, None
        if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], int):
            count, init_val, dtype = spec

            def _init_fn(bld: Any, _c=count, _v=init_val, _d=dtype) -> list[Value]:
                from popcorn.lang import const as _const

                return [_const(_d, _v)] * int(_c)

            return int(count), _init_fn, None
        if isinstance(spec, (list, tuple)):
            current = list(spec)

            def _init_fn(bld: Any, _c: list[Value] = current) -> list[Value]:
                return list(_c)

            return len(current), _init_fn, current
        if isinstance(spec, Value):

            def _init_fn(bld: Any, _v: Value = spec) -> list[Value]:
                return [_v]

            return 1, _init_fn, [spec]
        raise TypeError(
            f"Carry slot spec {spec!r} not recognized — expected Accumulators, "
            "(count, init_value, dtype), list[Value], or a single Value."
        )

    # ── Attribute-style read/write on slots ──────────────────────────
    #
    # Reads return the current list[Value] for that slot. Writes
    # normalize list/tuple/Value into list[Value] and validate the
    # count matches the slot's declared size.

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        values = object.__getattribute__(self, "_values")
        if name not in values:
            specs = object.__getattribute__(self, "_specs")
            if name in specs:
                raise AttributeError(
                    f"Carry slot {name!r} not bound — call .init() or pass the "
                    "Carry as PipelineBody.carry so run_pipeline binds it."
                )
            raise AttributeError(f"Carry has no slot {name!r}")
        return values[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        specs = object.__getattribute__(self, "_specs")
        if name not in specs:
            raise AttributeError(
                f"Carry: slot {name!r} was not declared at construction; "
                f"declared slots: {list(specs)}"
            )
        count, _ = specs[name]
        vs = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(vs) != count:
            raise ValueError(f"Carry.{name}: expected {count} Values, got {len(vs)}")
        self._values[name] = vs

    # ── Flatten / rebind — used by run_pipeline internally ───────────

    def init(self) -> tuple[Value, ...]:
        """Emit initial values for every slot and flatten to a tuple.

        If every slot is already bound (e.g. the Carry was produced via
        :meth:`bound_to`) the flat tuple is returned as-is — the caller
        already handed us concrete Values. Otherwise the per-slot init
        callable runs: Accumulators slots emit ``.init()``, scalar-array
        slots emit ``count`` copies of ``pop.const(dtype, init_value)``,
        and pre-built-list slots pass through. Must run under an active
        Builder (same convention as other DSL free-functions).
        """
        if all(name in self._values for name in self._specs):
            return self.flatten()
        from popcorn.ir.value import _ACTIVE_BUILDER

        bld = _ACTIVE_BUILDER.get(None)
        if bld is None:
            raise RuntimeError(
                "Carry.init(): no active Builder. Call from inside a kernel "
                "emit body or wrap in `pop.kernel_scope(bld):`."
            )
        out: list[Value] = []
        for _name, (_, init_fn) in self._specs.items():
            vs = init_fn(bld)
            self._values[_name] = list(vs)
            out.extend(vs)
        return tuple(out)

    def bound_to(self, flat: Any) -> Carry:
        """Return a fresh Carry sharing this Carry's specs, with slot
        values bound from ``flat``.

        Used when an outer loop holds the current carry as a flat tuple
        (``pop.for_range``'s ``carried``) and an inner pipeline needs
        to start from those values rather than re-emitting init. The
        returned Carry's ``.init()`` returns ``flat`` as-is (all slots
        pre-bound), so passing it as ``PipelineBody.carry`` on the
        inner pipeline makes the inner loop start where the outer left
        off.
        """
        new = Carry.__new__(Carry)
        object.__setattr__(new, "_specs", dict(self._specs))
        object.__setattr__(new, "_raw_specs", dict(self._raw_specs))
        object.__setattr__(new, "_values", {})
        new.rebind(flat)
        return new

    def flatten(self) -> tuple[Value, ...]:
        """Concat current slot values in declaration order."""
        out: list[Value] = []
        for name in self._specs:
            if name not in self._values:
                raise RuntimeError(
                    f"Carry.flatten: slot {name!r} is unbound; did you call .init()?"
                )
            out.extend(self._values[name])
        return tuple(out)

    def rebind(self, flat: Any) -> Carry:
        """Slice ``flat`` into per-slot lists (using declared counts),
        store in-place, return ``self``. ``run_pipeline`` calls this on
        every loop iteration to bind the fresh carry values before
        handing the Carry to ``consume``.
        """
        flat_t = tuple(flat)
        i = 0
        for name, (count, _) in self._specs.items():
            self._values[name] = list(flat_t[i : i + count])
            i += count
        if i != len(flat_t):
            raise ValueError(
                f"Carry.rebind: flat tuple has {len(flat_t)} Values but slots expect {i} total."
            )
        return self

    @property
    def slots(self) -> dict[str, Any]:
        """Read-only view of slot names → specs — retained for the
        ``_stash_results_on_carry`` hook in run_pipeline that walks
        Accumulators slots to populate ``.results``.
        """
        return dict(self._specs)
