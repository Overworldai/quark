"""KernelContext — top-level kernel builder driven by ``@kernel``.

Manages function setup, tensor declarations, and block emission; the
``@kernel`` decorator instantiates it and publishes it into the active
ContextVar so every free-function helper finds it implicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from popcorn.blocks.dsl.block_context import BlockContext
from popcorn.blocks.dsl.context import _ACTIVE_BCTX, _ACTIVE_KCTX
from popcorn.blocks.dsl.tensors import TensorDecl
from popcorn.ir import BufferType, Builder, DType, GlobalTensor, Module, Value

if TYPE_CHECKING:
    pass


class KernelContext:
    """Top-level kernel builder.

    Manages function setup, tensor declarations, and block emission.
    Owned and driven by the ``@kernel`` decorator — kernel authors
    write a ``build(self)`` body and never instantiate this class
    directly.
    """

    def __init__(self, name: str, *, mma_shapes: list | None = None, n_warps: int = 4):
        self._name = name
        self._mma_shapes = mma_shapes or []
        self._n_warps = n_warps
        self._bld = Builder(f"{name}_module")
        for shape in self._mma_shapes:
            self._bld.register_shape(shape)
        self._fn = self._bld.begin_function(name)
        self._tensors: list[GlobalTensor] = []
        self._param_idx = 0
        self._block_indices: dict[str, Value] = {}
        # Publish this context into the ContextVar so free-function
        # helpers (block_base, barrier, c, etc.) and auto-emitting
        # SetupBlocks can find it without threading `ctx` everywhere.
        # Reset in `finalize()`.
        self._kctx_token = _ACTIVE_KCTX.set(self)
        self._bctx_token = None  # set by make_ctx

    @property
    def bld(self) -> Builder:
        return self._bld

    def tensor(
        self,
        name: str,
        dtype: DType,
        shape: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
        readonly: bool = False,
    ) -> GlobalTensor:
        """Declare a kernel parameter + GlobalTensor in one call.

        Infers row-major stride from shape if stride is not given.
        Set ``readonly=True`` for input-only buffers (matters for Metal
        where MLX separates inputs from outputs).
        """
        from popcorn.ir.module import ParamAttrs

        self._bld.param(name, BufferType(dtype), attrs=ParamAttrs(readonly=readonly))
        if stride is None:
            # Row-major: stride[i] = product of shape[i+1:]
            stride = tuple(_product(shape[i + 1 :]) for i in range(len(shape)))
        gt = GlobalTensor(
            dtype=dtype,
            shape=shape,
            stride=stride,
            name=name,
            param=self._fn.params[self._param_idx],
        )
        self._param_idx += 1
        self._tensors.append(gt)
        return gt

    def declare_tensors(
        self,
        decls: list[TensorDecl],
        spec: Any,
        config: Any = None,
    ) -> dict[str, GlobalTensor]:
        """Declare every tensor in `decls` against this kernel's spec+config.

        Replaces the ``g_x = ctx.tensor(...); g_w = ctx.tensor(...)`` ritual
        with a single manifest walk. Tensors are declared in list order so
        the parameter order in the emitted function signature is stable.
        Returns a name→GlobalTensor dict; callers typically destructure by
        name or access by key.

        Each TensorDecl resolves its dtype and shape from (spec, config) —
        both can be literals or callables, so the same manifest covers
        shapes that depend on derived spec properties (``s.total_slots``)
        and config knobs (``c.BM``, ``c.BN``) without branching.
        """
        out: dict[str, GlobalTensor] = {}
        for decl in decls:
            out[decl.name] = self.tensor(
                decl.name,
                dtype=decl.resolve_dtype(spec, config),
                shape=decl.resolve_shape(spec, config),
                readonly=decl.role == "in",
            )
        return out

    def block_idx(self, axis: str) -> Value:
        """Cached block index."""
        if axis not in self._block_indices:
            self._block_indices[axis] = self._bld.block_idx(axis)
        return self._block_indices[axis]

    def block_base(self, axis: str, size: int) -> Value:
        """Shorthand for ``block_idx(axis) * size``.

        Every kernel computes at least one per-block base offset like
        this (``m_base = block_x * BM``, ``n_base = block_y * BN``).
        Auto-CSE in the Builder already dedupes the multiplication, but
        the one-liner is cleaner at call sites than the open-coded mul.
        """
        return self._bld.mul(self.block_idx(axis), self._bld.const(DType.U32, size))

    @property
    def n_threads(self) -> int:
        """Total threads per block. Exposed for kernels that need the
        value before they call :meth:`make_ctx` (e.g. for sizing
        per-thread work quanta in smem allocation math).
        """
        return self._n_warps * 32

    def make_ctx(self, mma_cfg: Any = None) -> BlockContext:
        """Create a BlockContext and publish it as the active ctx.

        The BlockContext is published in ``_ACTIVE_BCTX`` so that
        free-function helpers (``block_base``, ``barrier``, ``c``, …)
        and auto-emitting SetupBlocks find it implicitly. Callers can
        still capture the returned bctx for explicit access.
        """
        bctx = BlockContext(
            self._bld,
            n_threads=self._n_warps * 32,
            mma_cfg=mma_cfg,
        )
        self._bctx_token = _ACTIVE_BCTX.set(bctx)
        return bctx

    def finalize(self) -> Module:
        """Close the function, reset the active ContextVars, return
        the module. Called by the @kernel decorator (or manually by
        kernels not yet using the decorator).
        """
        self._bld.end_function()
        if self._bctx_token is not None:
            _ACTIVE_BCTX.reset(self._bctx_token)
            self._bctx_token = None
        _ACTIVE_KCTX.reset(self._kctx_token)
        return self._bld.module


def _product(seq: tuple[int, ...]) -> int:
    result = 1
    for x in seq:
        result *= x
    return result
