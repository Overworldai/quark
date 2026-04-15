"""@kernel decorator — autogenerates emit() from a build() body.

Replaces the ritual opening of every kernel (construct KernelContext,
walk the TENSORS manifest, expose the tensor namespace, finalize the
module) with a single decorator that runs around a user-supplied
``build(self)`` method.

Usage:

    @kernel("moe_outproj", spec=MoeOutprojSpec, config=MoeOutprojConfig)
    class MoeOutprojKernel(Kernel):
        TENSORS = [TensorDecl("h_in", ...), ...]

        def build(self):
            s, c = self.spec, self.config
            g = self.g          # .h_in, .W_out, ...
            bctx = self.make_bctx(mma_cfg)
            ...                 # kernel logic

The decorator:
  - sets NAME / SPEC_CLS / CONFIG_CLS / OUTPUT_IDX class attributes,
  - wires @register(NAME) so the kernel lands in the registry,
  - generates an ``emit(self) -> Module`` that creates the
    KernelContext, walks ``cls.TENSORS``, publishes the tensor
    dict on ``self.g`` (``SimpleNamespace``), runs ``self.build()``,
    and returns ``ctx.finalize()``.

The kernel's build() is responsible for:
  - resolving the MmaConfig (the lookup is kernel-specific: some
    kernels don't use mma.sync at all);
  - calling ``self.make_bctx(mma_cfg)`` to materialize the
    BlockContext. Auto-publishes into ``_ACTIVE_BCTX`` so the
    free-function helpers (``block_base``, ``barrier``, ``c``, …)
    resolve.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from popcorn.ir import Module
from popcorn.kernels.registry import register as _register


def kernel(
    name: str,
    *,
    spec: type,
    config: type,
    output_idx: int = -1,
    problems: Callable | list | None = None,
    baselines: Callable | None = None,
    reference: Callable | None = None,
) -> Callable[[type], type]:
    """Class decorator declaring a popcorn kernel.

    Optional hooks:
      * ``problems``  — either a ``list[Problem]`` or a zero-arg callable
        returning one. Installed as the ``problems()`` classmethod.
      * ``baselines`` — ``fn(kernel, tensors) -> list[Baseline]``.
        Installed as the instance ``baselines(self, tensors)`` method;
        the function gets the full kernel so it can reach ``kernel.spec``
        / ``kernel.config`` as needed.
      * ``reference`` — ``fn(kernel, *tensors) -> out``. Installed as
        the instance ``reference(self, *tensors)`` method.

    Each hook lets the kernel keep its bookkeeping (problems table,
    baseline construction, torch reference) in sibling modules so
    ``kernel.py`` stays focused on IR emission.
    """

    def decorate(cls: type) -> type:
        cls.NAME = name  # type: ignore
        cls.SPEC_CLS = spec  # type: ignore
        cls.CONFIG_CLS = config  # type: ignore
        cls.OUTPUT_IDX = output_idx  # type: ignore

        if problems is not None:
            # Accept either a callable returning the problem list or
            # the list itself; normalize to a zero-arg callable.
            _fn: Callable[[], list] = (  # type: ignore
                problems if callable(problems) else (lambda _p=problems: list(_p))
            )
            cls.problems = classmethod(lambda _cls: _fn())  # type: ignore

        if baselines is not None:

            def _baselines(self, tensors, _fn=baselines):
                return _fn(self, tensors)

            cls.baselines = _baselines  # type: ignore

        if reference is not None:

            def _reference(self, *tensors, _fn=reference):
                return _fn(self, *tensors)

            cls.reference = _reference  # type: ignore
            cls.__abstractmethods__ = frozenset(  # type: ignore
                m for m in getattr(cls, "__abstractmethods__", ()) if m != "reference"
            )

        # Generated emit(): opens the ctx, walks the manifest, runs
        # build(), finalizes. The kernel only writes build().
        if "build" not in cls.__dict__:
            raise TypeError(
                f"@kernel({name!r}): {cls.__name__} must define a "
                f"`build(self)` method (replaces the old emit())."
            )
        if "emit" in cls.__dict__:
            raise TypeError(
                f"@kernel({name!r}): {cls.__name__} must not define "
                f"`emit()` — the decorator generates it from build()."
            )

        def emit(self: Any) -> Module:
            # Lazy import — popcorn.blocks pulls in mma_body which
            # (transitively) imports popcorn.kernels.gemm, so we can't
            # import at module level without a circular import.
            from popcorn.blocks import KernelContext, block_base

            # n_warps: most configs carry it directly; owl_attn computes
            # gqa_ratio*NCW and exposes _n_warps() as an override hook.
            if hasattr(self, "_n_warps"):
                n_warps = self._n_warps()
            else:
                n_warps = getattr(self.config, "n_warps", 4)
            ctx = KernelContext(self.NAME, n_warps=n_warps)
            self.ctx = ctx
            self.bld = ctx.bld
            # Walk the manifest once. Every tensor is available as
            # self.g.NAME from inside build().
            g_dict = ctx.declare_tensors(self.TENSORS, self.spec, self.config)
            self.g = SimpleNamespace(**g_dict)

            # Auto-publish bctx so build() can drop the opening ritual.
            # Resolve mma_cfg via _mma_cfg() if the kernel declares any
            # MMA sites; otherwise pass None.
            import contextlib

            mma_cfg = None
            if type(self).mma_sites(self.spec):
                # Invalid main_shape → leave mma_cfg=None; build() decides.
                with contextlib.suppress(KeyError):
                    mma_cfg = self._mma_cfg()
            self.make_bctx(mma_cfg)  # publishes self.bctx

            # Block-base convention: self.m_base = block_base("y", BM),
            # self.n_base = block_base("x", BN). Set automatically when
            # the config carries those fields. Kernels whose grid
            # doesn't follow the (N/BN, M/BM, 1) convention (attn /
            # owl_attn / kv_cache_update) don't declare BM/BN and thus
            # don't get these bases — they compute their own.
            if hasattr(self.config, "BM"):
                self.m_base = block_base("y", self.config.BM)
            if hasattr(self.config, "BN"):
                self.n_base = block_base("x", self.config.BN)

            # Run kernel-supplied build body.
            self.build()
            return ctx.finalize()

        cls.emit = emit  # type: ignore
        # Kernel base declares `emit` abstract via ABC; patching the
        # method post-class-creation doesn't clear __abstractmethods__
        # automatically. Drop it explicitly so the class instantiates.
        cls.__abstractmethods__ = frozenset(  # type: ignore
            m for m in getattr(cls, "__abstractmethods__", ()) if m != "emit"
        )
        return _register(name)(cls)

    return decorate
