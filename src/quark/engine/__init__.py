"""``quark.Engine`` — public entry point.

The engine is a package: this ``__init__.py`` is just the public-
import shim. ``Engine(model_uri, ...)`` constructs a platform-
appropriate concrete subclass via the ``__new__`` factory in
:mod:`quark.engine.base` — :class:`EngineCUDA` on Linux / Windows /
CUDA Macs, :class:`EngineMetal` on Apple Silicon.

Backend modules:

  * :mod:`quark.engine.base` — base class + factory + shared
    helpers (``_qt_from_torch``, ``_resolve_quant``, ``_encode_ctrl``).
  * :mod:`quark.engine.cuda` — :class:`EngineCUDA`. Torch tensors at
    the latent boundary (zero-copy ``QuarkTensor.borrow``), CUDA graph
    capture via ``GenerateFrame``, torch-side
    ``ChunkedStreamingTAEHV`` VAE.
  * :mod:`quark.engine.metal` — :class:`EngineMetal`. DiT runs
    eagerly under ``quark.lazy()`` (Metal has no graph capture), VAE
    runs on the Apple Neural Engine via ``quark.taehv``, latent
    boundary is plain numpy.

Adding a new backend is one new ``EngineX(Engine)`` subclass plus a
branch in :meth:`Engine.__new__`; nothing in :class:`Engine` itself
or in callers needs to change.
"""

from __future__ import annotations

from quark.engine.base import Engine
from quark.engine.cuda import EngineCUDA
from quark.engine.metal import EngineMetal
from quark.models.waypoint_15 import CtrlInput

__all__ = ["CtrlInput", "Engine", "EngineCUDA", "EngineMetal"]
