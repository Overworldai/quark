"""Region: an ordered sequence of Ops.

A Region is the body of a Function or the body/then/else/cond of a
structured control-flow op. Regions own their ops; ops reference Values
across regions only when SSA dominance allows.

See QUARK_IR_PROPOSAL.md §2, §5.7.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .op import Op


@dataclass(eq=False)
class Region:
    """An ordered list of Ops. The terminator (a YieldOp or nothing for
    the function body) is the last element when present.

    Regions are not first-class Values. They're a structural container.
    """

    ops: list[Op] = field(default_factory=list)
    parent_op: Op | None = field(default=None, repr=False)

    def append(self, op: Op) -> None:
        self.ops.append(op)

    def __iter__(self) -> Iterator[Op]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __bool__(self) -> bool:  # non-empty when it has any ops
        return bool(self.ops)

    @property
    def terminator(self) -> Op | None:
        """Return the last op if it's a region terminator (YieldOp), else None."""
        from .op import YieldOp

        if not self.ops:
            return None
        last = self.ops[-1]
        return last if isinstance(last, YieldOp) else None
