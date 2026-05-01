"""Backend lowerers for the quark IR.

Each backend is a visitor package under `quark.lower.<target>` that
walks a `Module` and emits target source text. V1 ships PTX only.

Importing this package eagerly imports all backends so their
``@register_lowerer`` decorators fire into ``quark.lower.base.LOWERERS``.
Callers then use ``get_lowerer(family, caps)`` to construct the
appropriate lowerer without knowing which backends exist.

See QUARK_IR_PROPOSAL.md §9–§10.
"""

# Trigger registration side-effects. MSL is pure Python so the import
# is safe on any platform (it only needs `mlx` at actual emit time).
# Populate the legalization registry with every keep-or-expand pattern
# for ops whose lowering is backend-conditional. Importing
# ``quark.lower.legalize`` alone gets the empty-registry no-op path;
# importing this package picks up the concrete rewrites too.
from quark.lower import legalizations as _legalizations  # noqa: F401
from quark.lower import msl as _msl  # noqa: F401 — registration side-effect
from quark.lower import ptx as _ptx  # noqa: F401 — registration side-effect
from quark.lower import spv as _spv  # noqa: F401 — SPIR-V stub registration
from quark.lower.base import LOWERERS, Lowerer, get_lowerer, register_lowerer

__all__ = [
    "LOWERERS",
    "Lowerer",
    "get_lowerer",
    "register_lowerer",
]
