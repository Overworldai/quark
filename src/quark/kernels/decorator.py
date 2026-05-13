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

from quark.ir import Module
from quark.kernels.registry import register as _register


def _resolve_caps(self: Any):
    """Return DeviceCaps for this kernel, or None if unavailable.

    Priority: explicitly bound ``self._caps`` (set by tests / autotune
    via ``Kernel.bind_caps``), then ``current_device().caps`` cached
    probe. ``None`` means "no caps known" — emit() falls back to the
    default ``build()``.
    """
    caps = getattr(self, "_caps", None)
    if caps is not None:
        return caps
    try:
        from quark.device import current_device

        return current_device().caps
    except Exception:
        return None


def _pick_build_method(cls: type, caps) -> str:
    """Return the most-specific build method name available on ``cls``.

    Lookup order: ``build_<family>`` (e.g. ``build_metal``,
    ``build_cuda``), then plain ``build``. The family name is
    ``caps.family.value`` (already lowercase: 'metal' / 'cuda' / ...).
    Picking the method by name lets a kernel diverge per-backend
    without duplicating its TENSORS / spec / config / tune_space.
    """
    if caps is not None:
        family = getattr(caps.family, "value", None)
        if family:
            specific = f"build_{family}"
            if specific in cls.__dict__ or any(specific in base.__dict__ for base in cls.__mro__):
                return specific
    return "build"


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
    """Class decorator declaring a quark kernel.

    Optional hooks:
      * ``problems``  — either a ``list[Problem]`` or a zero-arg callable
        returning one. Installed as the ``problems()`` classmethod.
      * ``baselines`` — ``fn(kernel, tensors) -> list[Baseline]``.
        Installed as the instance ``baselines(self, tensors)`` method;
        the function gets the full kernel so it can reach ``kernel.spec``
        / ``kernel.config`` as needed.
      * ``reference`` — ``fn(spec, **inputs_np) -> dict[str, ndarray] | ndarray``.
        Installed as the ``reference_numpy(cls, spec, **inputs)`` classmethod.
        Single-output kernels may return a bare ``np.ndarray``; multi-output
        kernels return a ``{name: ndarray}`` dict keyed on the role="out"
        TensorDecl names.

    Each hook lets the kernel keep its bookkeeping (problems table,
    baseline construction, numpy reference) in sibling modules so
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
            # ``reference`` is the numpy oracle. Store it as a
            # classmethod so ``RefCache.get()`` can invoke it as
            # ``kernel_cls.reference_numpy(spec, **inputs)``.
            _ref_fn = reference

            def _reference_numpy(_cls, spec, _fn=_ref_fn, **inputs):
                return _fn(spec, **inputs)

            cls.reference_numpy = classmethod(_reference_numpy)  # type: ignore
            cls.__abstractmethods__ = frozenset(  # type: ignore
                m for m in getattr(cls, "__abstractmethods__", ()) if m != "reference"
            )

        # Generated emit(): opens the ctx, walks the manifest, runs
        # build(), finalizes. The kernel only writes build().
        if "build" not in cls.__dict__ and not any(
            k.startswith("build_") and callable(v) for k, v in cls.__dict__.items()
        ):
            raise TypeError(
                f"@kernel({name!r}): {cls.__name__} must define a "
                f"`build(self)` method (or a backend-specific "
                f"`build_<family>` such as ``build_metal``)."
            )
        if "emit" in cls.__dict__:
            raise TypeError(
                f"@kernel({name!r}): {cls.__name__} must not define "
                f"`emit()` — the decorator generates it from build()."
            )

        def emit(self: Any) -> Module:
            # Lazy import — quark.blocks pulls in mma_body which
            # (transitively) imports quark.kernels.gemm, so we can't
            # import at module level without a circular import.
            from quark.blocks import KernelContext, block_base

            # n_warps: most configs carry it directly; owl_attn computes
            # gqa_ratio*NCW and exposes _n_warps() as an override hook.
            if hasattr(self, "_n_warps"):
                n_warps = self._n_warps()
            else:
                n_warps = getattr(self.config, "n_warps", 4)

            # Resolve mma_cfg before KernelContext so the per-kernel SG
            # width (derived from MMA shape; see resolve_subgroup_size)
            # is known. Intel MMA shapes pin SG=16 — autotuners do NOT
            # set ``config.subgroup_size``; the resolver picks it from
            # the shape, and the OCL lowerer's OpExecutionMode pins
            # the matching width at compile time.
            import contextlib

            mma_cfg = None
            if type(self).mma_sites(self.spec):
                # Invalid main_shape → leave mma_cfg=None; build() decides.
                with contextlib.suppress(KeyError):
                    mma_cfg = self._mma_cfg()

            sgs = self.resolve_subgroup_size()
            ctx = KernelContext(self.NAME, n_warps=n_warps, subgroup_size=sgs)
            self.ctx = ctx
            self.bld = ctx.bld
            # Walk the manifest once. Every tensor is available as
            # self.g.NAME from inside build().
            g_dict = ctx.declare_tensors(self.TENSORS, self.spec, self.config)
            self.g = SimpleNamespace(**g_dict)
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

            # Run kernel-supplied build body. Backend-specific
            # ``build_<family>`` overrides take precedence over the
            # default ``build`` when caps are known.
            self.caps = _resolve_caps(self)
            method_name = _pick_build_method(type(self), self.caps)
            getattr(self, method_name)()
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
