"""Lifetime annotations for shared-memory regions.

A ``Lifetime`` declares when a ``SharedRegion`` is live. The smem
layout pass (in ``popcorn.lower.smem_layout``) consumes lifetimes to:

  1. Compute the true peak smem usage by intersecting region intervals
     instead of summing all declared sizes.
  2. Auto-alias regions whose lifetimes don't overlap — e.g., owl_attn's
     Q region and the post-MMA O staging region never co-exist, so they
     share storage automatically.
  3. Catch read-after-write hazards across aliased boundaries when the
     code path between writer and aliased reader doesn't include a
     ``threadgroup_barrier``.

Lifetimes that don't reference an explicit Region default to "auto":
the layout pass infers ``[earliest_use, latest_use]`` from the IR
structure (op-index walk). Explicit ``in_region(R)`` extends or pins
the lifetime past the apparent use range — useful when a region is
defined outside a loop but only used inside it, and you want the
lowerer to free it before the loop's epilogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .region import Region


class LifetimeKind(Enum):
    #: Live for the entire kernel — never aliasable. Default for regions
    #: without an explicit lifetime annotation when their use spans more
    #: than one structured-region boundary (the layout pass falls back to
    #: kernel-wide if it can't prove tighter).
    KERNEL = "kernel"
    #: Live only while ``region`` is on the IR control-flow stack. Common
    #: case: scratch declared at function scope but used only inside a
    #: ``for_loop`` body — declare with ``lifetime=in_region(loop.body)``.
    IN_REGION = "in_region"
    #: Live up to (but NOT including) the entry into ``region``. Lets the
    #: layout pass alias the storage with anything declared
    #: ``after_region(region)``.
    BEFORE_REGION = "before_region"
    #: Live from the exit of ``region`` to the kernel end. Symmetric
    #: counterpart to ``BEFORE_REGION``.
    AFTER_REGION = "after_region"
    #: Lifetime is inferred from dataflow — the layout pass computes
    #: ``[earliest_use, latest_use]`` over the IR's op-index walk. This is
    #: the default when no lifetime is supplied.
    AUTO = "auto"


@dataclass(frozen=True)
class Lifetime:
    """When a ``SharedRegion`` is live.

    Use the classmethod constructors (``Lifetime.kernel()``,
    ``Lifetime.in_region(R)``, etc.) — direct construction is supported
    but the helpers spell out the intent.
    """

    kind: LifetimeKind = LifetimeKind.AUTO
    #: For ``IN_REGION`` / ``BEFORE_REGION`` / ``AFTER_REGION`` only —
    #: the IR Region this lifetime is anchored to. Region uses object
    #: identity for hashing (``eq=False`` on the dataclass), so the
    #: layout pass can map back to the structural-control-flow op that
    #: owns it.
    region: Region | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        needs_region = {
            LifetimeKind.IN_REGION,
            LifetimeKind.BEFORE_REGION,
            LifetimeKind.AFTER_REGION,
        }
        if self.kind in needs_region and self.region is None:
            raise ValueError(f"Lifetime kind {self.kind.name} requires a region argument")
        if self.kind in (LifetimeKind.KERNEL, LifetimeKind.AUTO) and self.region is not None:
            raise ValueError(f"Lifetime kind {self.kind.name} must not carry a region argument")

    @classmethod
    def kernel(cls) -> Lifetime:
        """Live the whole kernel — never aliased away."""
        return cls(kind=LifetimeKind.KERNEL)

    @classmethod
    def auto(cls) -> Lifetime:
        """Inferred from earliest/latest use (default)."""
        return cls(kind=LifetimeKind.AUTO)

    @classmethod
    def in_region(cls, region: Region) -> Lifetime:
        """Live only while ``region`` is on the control-flow stack."""
        return cls(kind=LifetimeKind.IN_REGION, region=region)

    @classmethod
    def before_region(cls, region: Region) -> Lifetime:
        """Live up to (but not entering) ``region``. Aliasable with
        ``after_region(region)``-lifetime regions."""
        return cls(kind=LifetimeKind.BEFORE_REGION, region=region)

    @classmethod
    def after_region(cls, region: Region) -> Lifetime:
        """Live from the exit of ``region`` onward."""
        return cls(kind=LifetimeKind.AFTER_REGION, region=region)


__all__ = ("Lifetime", "LifetimeKind")
