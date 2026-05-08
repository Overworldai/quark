"""Builder — the L1/L2-facing IR construction API.

The Builder owns:
 - the current Module and Function under construction
 - a Region stack for structured control flow
 - a Value allocator (delegated to the Function)

Nothing in this file emits text. Every method records an IR node.
Control flow is exposed through Python context managers so nested
regions read like ordinary `for`/`if` blocks.

See QUARK_IR_PROPOSAL.md §6.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import ClassVar

from . import op as _op
from .lifetime import Lifetime
from .module import (
    Function,
    FunctionAttrs,
    MmaShape,
    Module,
    ParamAttrs,
    Type,
)
from .region import Region
from .tensor import GlobalTensor, SharedRegion, Tensor
from .types import DType, ValueShape
from .value import Value


class Builder:
    """Construction API for IR modules.

    Typical usage:
        b = Builder("my_module")
        fn = b.begin_function("gemm")
        x_ptr = b.param("X", BufferType(DType.BF16))
        ...
        b.end_function()
        module = b.module
    """

    def __init__(self, module_name: str = "") -> None:
        self.module = Module(name=module_name)
        self._fn: Function | None = None
        self._region_stack: list[Region] = []
        # CSE stack — parallel to _region_stack. Each entry is a dict
        # {(kind, operand_tuple, shape_key): Value} of pure-op results
        # emitted in that region. Lookup walks the stack top→bottom so
        # ancestor-region results stay visible while descendant regions
        # don't leak outward. Set QUARK_DISABLE_CSE=1 to disable.
        import os as _os

        self._cse_enabled = _os.environ.get("QUARK_DISABLE_CSE") not in ("1", "true", "yes")
        self._cse_stack: list[dict] = []
        # Incremented on every ``async_copy`` emission. ``run_pipeline``
        # snapshots the counter around ``produce`` to decide whether the
        # iteration needs ``async_commit`` / ``async_wait``. Replaces the
        # caller-maintained ``has_async`` flag on PipelineBody.
        self.async_emissions: int = 0

    # ---------------------------------------------------------------
    # Function / parameter management
    # ---------------------------------------------------------------

    def begin_function(self, name: str, attrs: FunctionAttrs | None = None) -> Function:
        if self._fn is not None:
            raise RuntimeError("Builder.begin_function: a function is already open")
        fn = Function(name=name, attrs=attrs or FunctionAttrs())
        self.module.add_function(fn)
        self._fn = fn
        self._region_stack = [fn.body]
        self._cse_stack = [{}]
        # Publish as the active Builder so Value operator overloads
        # (`a * b`, etc.) can dispatch here.
        from .value import _ACTIVE_BUILDER

        self._active_token = _ACTIVE_BUILDER.set(self)
        return fn

    def end_function(self) -> Function:
        if self._fn is None:
            raise RuntimeError("Builder.end_function: no open function")
        if len(self._region_stack) != 1:
            raise RuntimeError(
                "Builder.end_function: unclosed regions on the stack — "
                "did you forget to exit a for_loop/if_region?"
            )
        fn = self._fn
        self._fn = None
        self._region_stack = []
        self._cse_stack = []
        from .value import _ACTIVE_BUILDER

        _ACTIVE_BUILDER.reset(self._active_token)
        return fn

    @property
    def function(self) -> Function:
        if self._fn is None:
            raise RuntimeError("Builder: no active function")
        return self._fn

    @property
    def current_region(self) -> Region:
        if not self._region_stack:
            raise RuntimeError("Builder: no active region")
        return self._region_stack[-1]

    def param(self, name: str, type: Type, attrs: ParamAttrs | None = None) -> Value:
        return self.function.add_param(name, type, attrs)

    def register_shape(self, shape: MmaShape) -> None:
        self.module.register_shape(shape)

    # ---------------------------------------------------------------
    # Helpers: minting results + emitting ops
    # ---------------------------------------------------------------

    def _fresh(self, shape: ValueShape, name: str = "") -> Value:
        return self.function.fresh_value(shape, producer=None, name=name)

    def _emit(self, op: _op.Op) -> _op.Op:
        self.current_region.append(op)
        return op

    # ---------------------------------------------------------------
    # CSE: pure-op result memoization, scoped by region for dominance.
    # ---------------------------------------------------------------
    # Commutative arith kinds whose operand order doesn't matter for CSE
    # — a+b == b+a, so canonicalize operand ids before caching.
    _COMMUTATIVE_ARITH: ClassVar[frozenset] = frozenset(
        {"add", "mul", "mul_hi", "min", "max", "and", "or", "xor"}
    )

    def _cse_key_for_arith(
        self, kind: str, operands: tuple[Value, ...], result_shape: ValueShape
    ) -> tuple:
        ids = tuple(v.id for v in operands)
        if kind in self._COMMUTATIVE_ARITH and len(ids) == 2:
            ids = tuple(sorted(ids))
        return (f"arith:{kind}", ids, (result_shape.dtype, result_shape.width))

    def _cse_lookup(self, key: tuple) -> Value | None:
        if not self._cse_enabled or not self._cse_stack:
            return None
        for scope in reversed(self._cse_stack):
            if key in scope:
                return scope[key]
        return None

    def _cse_store(self, key: tuple, value: Value) -> None:
        if not self._cse_enabled or not self._cse_stack:
            return
        self._cse_stack[-1][key] = value

    # ---------------------------------------------------------------
    # §5.1 Arithmetic and math
    # ---------------------------------------------------------------

    def const(self, dtype: DType, value: int | float | bool, name: str = "") -> Value:
        # Region-scoped CSE: reusing a const Value from an outer region
        # inside an inner region is legal (SSA dominance holds), but the
        # reverse is not. ``_cse_lookup`` walks the _cse_stack from the
        # current frame outward, so it only returns values that dominate
        # the current insertion point. Matches how arith ops are CSE-ed.
        key = ("const", dtype, value)
        hit = self._cse_lookup(key)
        if hit is not None:
            return hit
        out = self._fresh(ValueShape(dtype), name)
        self._emit(
            _op.ConstOp(
                results=(out,),
                attrs={"dtype": dtype, "value": value},
            )
        )
        self._cse_store(key, out)
        return out

    def _binary_arith(
        self,
        kind: str,
        a: Value,
        b: Value,
        result_shape: ValueShape | None = None,
        name: str = "",
    ) -> Value:
        if result_shape is None:
            if a.shape != b.shape:
                raise TypeError(
                    f"ArithOp({kind}): operands must share shape, got {a.shape}/{b.shape}"
                )
            result_shape = a.shape
        # CSE: pure arith — same (kind, operands, shape) in a
        # dominating region returns the prior result. Kills the need
        # for manual bctx.hoist("name", lambda: ...) around repeated
        # index math.
        key = self._cse_key_for_arith(kind, (a, b), result_shape)
        cached = self._cse_lookup(key)
        if cached is not None:
            return cached
        out = self._fresh(result_shape, name)
        self._emit(
            _op.ArithOp(
                results=(out,),
                operands=(a, b),
                attrs={"kind": kind},
            )
        )
        self._cse_store(key, out)
        return out

    def add(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("add", a, b, name=name)

    def sub(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("sub", a, b, name=name)

    def mul(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("mul", a, b, name=name)

    def mul_hi(self, a: Value, b: Value, name: str = "") -> Value:
        """High 32 bits of a 32×32→64 unsigned multiply.

        Lowers to PTX ``mul.hi.u32``. Used by Philox-family counter-based
        RNGs (``randn``) — every round needs both halves of the product.
        """
        return self._binary_arith("mul_hi", a, b, name=name)

    def min(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("min", a, b, name=name)

    def max(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("max", a, b, name=name)

    def div(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("div", a, b, name=name)

    def rem(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("rem", a, b, name=name)

    def shl(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("shl", a, b, name=name)

    def shr(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("shr", a, b, name=name)

    def and_(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("and", a, b, name=name)

    def or_(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("or", a, b, name=name)

    def xor(self, a: Value, b: Value, name: str = "") -> Value:
        return self._binary_arith("xor", a, b, name=name)

    def neg(self, a: Value, name: str = "") -> Value:
        out = self._fresh(a.shape, name)
        self._emit(_op.ArithOp(results=(out,), operands=(a,), attrs={"kind": "neg"}))
        return out

    def abs(self, a: Value, name: str = "") -> Value:
        out = self._fresh(a.shape, name)
        self._emit(_op.ArithOp(results=(out,), operands=(a,), attrs={"kind": "abs"}))
        return out

    def fma(self, a: Value, b: Value, c: Value, name: str = "") -> Value:
        if not (a.shape == b.shape == c.shape):
            raise TypeError(f"fma: a/b/c must share shape, got {a.shape}/{b.shape}/{c.shape}")
        out = self._fresh(a.shape, name)
        self._emit(_op.ArithOp(results=(out,), operands=(a, b, c), attrs={"kind": "fma"}))
        return out

    def fma_bf16x2(self, a: Value, b: Value, c: Value, name: str = "") -> Value:
        """Packed BF16×2 fused multiply-add: d = a*b + c.

        All operands and result are B32, each holding two packed BF16 values.
        Lowers to `fma.rn.bf16x2` on sm_80+."""
        if not (a.dtype == b.dtype == c.dtype == DType.B32):
            raise TypeError(
                f"fma_bf16x2: all operands must be B32, got {a.dtype}/{b.dtype}/{c.dtype}"
            )
        out = self._fresh(ValueShape(DType.B32), name)
        self._emit(_op.ArithOp(results=(out,), operands=(a, b, c), attrs={"kind": "fma_bf16x2"}))
        return out

    def cvt_rn_bf16x2_f32(self, a: Value, b: Value, name: str = "") -> Value:
        """Pack two F32 values into a B32 holding two packed BF16 values.

        Lowers to `cvt.rn.bf16x2.f32 rD, rA, rB;` on sm_80+. Equivalent to
        two `cvt.rn.bf16.f32` + `mov.b32 {lo,hi}` but in a single instruction.
        Used by the bf16x2 atomic-add scatter path."""
        if not (a.dtype == DType.F32 and b.dtype == DType.F32):
            raise TypeError(
                f"cvt_rn_bf16x2_f32: both operands must be F32, got {a.dtype}/{b.dtype}"
            )
        out = self._fresh(ValueShape(DType.B32), name)
        self._emit(
            _op.ArithOp(results=(out,), operands=(a, b), attrs={"kind": "cvt_rn_bf16x2_f32"})
        )
        return out

    def cmp(self, kind: str, a: Value, b: Value, name: str = "") -> Value:
        # CSE: comparisons are pure. Not commutative, so preserve
        # operand order in the cache key.
        key = (f"cmp:{kind}", (a.id, b.id), (DType.PRED, 1))
        cached = self._cse_lookup(key)
        if cached is not None:
            return cached
        out = self._fresh(ValueShape(DType.PRED), name)
        self._emit(_op.CmpOp(results=(out,), operands=(a, b), attrs={"kind": kind}))
        self._cse_store(key, out)
        return out

    def select(self, pred: Value, t: Value, f: Value, name: str = "") -> Value:
        out = self._fresh(t.shape, name)
        self._emit(_op.SelectOp(results=(out,), operands=(pred, t, f)))
        return out

    def convert(self, v: Value, dst: DType, rounding: str = "rn", name: str = "") -> Value:
        out = self._fresh(ValueShape(dst, width=v.width), name)
        self._emit(
            _op.ConvertOp(
                results=(out,),
                operands=(v,),
                attrs={"src_dtype": v.dtype, "dst_dtype": dst, "rounding": rounding},
            )
        )
        return out

    def packed_convert(
        self,
        lo: Value,
        hi: Value,
        dst: DType,
        rounding: str = "rn",
        name: str = "",
    ) -> Value:
        """Packed two-into-one cvt — PTX's only cvt form for fp8 dst.

        Consumes two scalar Values of the same dtype and returns a
        width-1 B16 Value whose two bytes hold the packed ``<fp8>x2``
        result. Callers write it to contiguous fp8 storage via a single
        b16 store (``vec_store(..., width=1, dtype=B16)`` or similar).

        Valid destination dtypes are E4M3 and E5M2; valid source dtypes
        are F32, F16, BF16 (those are the src operand types PTX's
        ``cvt.rn.satfinite.<fp8>x2.*`` instructions accept).
        """
        if dst not in (DType.E4M3, DType.E5M2):
            raise TypeError(
                f"packed_convert: dst must be an fp8 dtype, got {dst}. "
                f"For other dsts use `convert()`."
            )
        if lo.dtype is not hi.dtype:
            raise TypeError(
                f"packed_convert: operand dtypes must match, got {lo.dtype} vs {hi.dtype}"
            )
        # Output is one 16-bit value carrying the two packed fp8 bytes.
        out = self._fresh(ValueShape(DType.B16, width=1), name or "fp8x2")
        self._emit(
            _op.PackedConvertOp(
                results=(out,),
                operands=(lo, hi),
                attrs={"src_dtype": lo.dtype, "dst_dtype": dst, "rounding": rounding},
            )
        )
        return out

    def unpacked_convert(
        self,
        packed: Value,
        src_dtype: DType,
        dst_dtype: DType,
        rounding: str = "rn",
        name: str = "",
    ) -> Value:
        """Inverse of `packed_convert` — split a packed fp8x2 (carried as
        a B16 value) into a width-2 vector of `dst_dtype` (typically BF16
        or F16).

        Required because PTX has no scalar `cvt.<wider>.e4m3`. The only
        cvt form for fp8 sources is `cvt.rn{.satfinite}{.relu}.<dst>x2.<src>x2`,
        which takes a packed b16 and produces a packed b32 of two wider
        floats. This op exposes that as a typed width-2 vector — callers
        consume the components via `vec_extract` (or `vec_store` if the
        destination layout is contiguous).

        Valid source dtypes: E4M3, E5M2.
        Valid destination dtypes: BF16, F16, F32 (lowered through f16x2 or
        bf16x2 then split).
        """
        if src_dtype not in (DType.E4M3, DType.E5M2):
            raise TypeError(
                f"unpacked_convert: src must be an fp8 dtype, got {src_dtype}. "
                f"For other source types use `convert()`."
            )
        if packed.dtype is not DType.B16:
            raise TypeError(
                f"unpacked_convert: input must be a B16 (carrying packed fp8x2), got {packed.dtype}"
            )
        out = self._fresh(ValueShape(dst_dtype, width=2), name or "unpack_fp8")
        self._emit(
            _op.UnpackedConvertOp(
                results=(out,),
                operands=(packed,),
                attrs={"src_dtype": src_dtype, "dst_dtype": dst_dtype, "rounding": rounding},
            )
        )
        return out

    def bitcast(self, v: Value, dst: DType, name: str = "") -> Value:
        out = self._fresh(ValueShape(dst, width=v.width), name)
        self._emit(_op.BitcastOp(results=(out,), operands=(v,), attrs={"dst_dtype": dst}))
        return out

    # Math
    def _math(self, kind: str, v: Value, name: str = "") -> Value:
        out = self._fresh(v.shape, name)
        self._emit(_op.MathOp(results=(out,), operands=(v,), attrs={"kind": kind}))
        return out

    def rcp_approx(self, v: Value, name: str = "") -> Value:
        return self._math("rcp_approx", v, name)

    def rsqrt_approx(self, v: Value, name: str = "") -> Value:
        return self._math("rsqrt_approx", v, name)

    def ex2_approx(self, v: Value, name: str = "") -> Value:
        return self._math("ex2_approx", v, name)

    def sqrt(self, v: Value, name: str = "") -> Value:
        return self._math("sqrt", v, name)

    def exp(self, v: Value, name: str = "") -> Value:
        return self._math("exp", v, name)

    def exp_approx(self, v: Value, name: str = "") -> Value:
        """Fast exp (``metal::fast::exp`` on MSL, ``ex2.approx`` + mul on PTX)."""
        return self._math("exp_approx", v, name)

    def exp2(self, v: Value, name: str = "") -> Value:
        return self._math("exp2", v, name)

    def log2(self, v: Value, name: str = "") -> Value:
        return self._math("log2", v, name)

    def log2_approx(self, v: Value, name: str = "") -> Value:
        """Fast SFU-pipe log2 (PTX ``lg2.approx.f32``). ~4-cycle
        throughput vs ~16 for the precise ``lg2.f32``. Used in the
        Philox + Box-Muller randn path where ULP-level accuracy is
        meaningless (the output is a random draw, not a bit-exact
        math function).
        """
        return self._math("log2_approx", v, name)

    def sqrt_approx(self, v: Value, name: str = "") -> Value:
        """Fast SFU-pipe sqrt (PTX ``sqrt.approx.f32``). Same rationale
        as :meth:`log2_approx` — used in Box-Muller for the
        ``r = sqrt(-2 * ln(u))`` magnitude.
        """
        return self._math("sqrt_approx", v, name)

    def sin(self, v: Value, name: str = "") -> Value:
        return self._math("sin", v, name)

    def cos(self, v: Value, name: str = "") -> Value:
        return self._math("cos", v, name)

    def tanh(self, v: Value, name: str = "") -> Value:
        return self._math("tanh", v, name)

    # ---------------------------------------------------------------
    # §5.2 Bit / vector manipulation
    # ---------------------------------------------------------------

    def vec_build(self, elems: Sequence[Value], name: str = "") -> Value:
        if not elems:
            raise ValueError("vec_build: need at least one element")
        dtype = elems[0].dtype
        for e in elems:
            if e.dtype is not dtype or e.width != 1:
                raise TypeError("vec_build: elements must be scalars of the same dtype")
        out = self._fresh(ValueShape(dtype, width=len(elems)), name)
        self._emit(_op.VecBuildOp(results=(out,), operands=tuple(elems)))
        return out

    def vec_extract(self, v: Value, index: int, name: str = "") -> Value:
        out = self._fresh(ValueShape(v.dtype, width=1), name)
        self._emit(_op.VecExtractOp(results=(out,), operands=(v,), attrs={"index": index}))
        return out

    def packed_extract_b32(self, vec: Value, pair_idx: int, name: str = "") -> Value:
        """Extract physical b32 register for BF16/F16 pair at pair_idx.

        Returns the B32 register holding elements [2*pair_idx, 2*pair_idx+1]
        of a packed sub-register vector.  Zero cost — no PTX emitted."""
        if vec.dtype not in (DType.BF16, DType.F16):
            raise TypeError(f"packed_extract_b32: expected BF16/F16 vec, got {vec.dtype}")
        out = self._fresh(ValueShape(DType.B32), name)
        self._emit(
            _op.VecExtractOp(
                results=(out,), operands=(vec,), attrs={"index": pair_idx, "packed_b32": True}
            )
        )
        return out

    def vec_build_packed_b32(
        self, b32_scalars: list[Value], *, elem_dtype, width: int, name: str = ""
    ) -> Value:
        """Build a BF16/F16 vector from pre-packed B32 values.

        b32_scalars: B32 values, each holding 2 elements of elem_dtype.
        width = 2 * len(b32_scalars). Zero PTX emitted — just rebinds registers."""
        if not all(v.dtype == DType.B32 for v in b32_scalars):
            raise TypeError("vec_build_packed_b32: all inputs must be B32")
        if width != 2 * len(b32_scalars):
            raise ValueError(f"vec_build_packed_b32: width={width} != 2 * {len(b32_scalars)}")
        out = self._fresh(ValueShape(elem_dtype, width), name)
        self._emit(
            _op.VecBuildOp(results=(out,), operands=tuple(b32_scalars), attrs={"packed_b32": True})
        )
        return out

    def split_b32(self, v: Value) -> tuple[Value, Value]:
        lo = self._fresh(ValueShape(DType.B16))
        hi = self._fresh(ValueShape(DType.B16))
        self._emit(_op.SplitB32Op(results=(lo, hi), operands=(v,)))
        return lo, hi

    def merge_b32(self, lo: Value, hi: Value, name: str = "") -> Value:
        out = self._fresh(ValueShape(DType.B32), name)
        self._emit(_op.MergeB32Op(results=(out,), operands=(lo, hi)))
        return out

    # ---------------------------------------------------------------
    # §5.3 Memory
    # ---------------------------------------------------------------

    def load(
        self,
        tensor: Tensor,
        *indices: Value,
        pred: Value | None = None,
        dtype: DType | None = None,
        name: str = "",
    ) -> Value:
        """Scalar load.

        `dtype` overrides the result type — useful for "bytes are bytes"
        re-interprets, e.g. loading 2 contiguous fp8 bytes from an e4m3
        tensor as a single B16 to feed `unpacked_convert`. Total bytes are
        the responsibility of the caller; the lowerer just emits
        `ld.<space>.<cls(dtype)>` at the address derived from `tensor +
        indices` (which uses tensor.dtype for stride bytes).
        """
        result_dtype = dtype if dtype is not None else tensor.dtype
        out = self._fresh(ValueShape(result_dtype), name)
        ops: tuple[Value, ...] = tuple(indices)
        if pred is not None:
            ops = ops + (pred,)
        self._emit(
            _op.LoadOp(
                results=(out,),
                operands=ops,
                attrs={"tensor": tensor, "pred": pred},
            )
        )
        return out

    def store(
        self,
        tensor: Tensor,
        value: Value,
        *indices: Value,
        pred: Value | None = None,
    ) -> None:
        ops: tuple[Value, ...] = (value,) + tuple(indices)
        if pred is not None:
            ops = ops + (pred,)
        self._emit(
            _op.StoreOp(
                operands=ops,
                attrs={"tensor": tensor, "pred": pred},
            )
        )

    def vec_load(
        self,
        tensor: Tensor,
        *indices: Value,
        width: int,
        dtype: DType | None = None,
        pred: Value | None = None,
        name: str = "",
    ) -> Value:
        dt = dtype if dtype is not None else tensor.dtype
        out = self._fresh(ValueShape(dt, width=width), name)
        ops: tuple[Value, ...] = tuple(indices)
        if pred is not None:
            ops = ops + (pred,)
        self._emit(
            _op.VecLoadOp(
                results=(out,),
                operands=ops,
                attrs={"tensor": tensor, "width": width, "pred": pred},
            )
        )
        return out

    def vec_store(
        self,
        tensor: Tensor,
        vec: Value,
        *indices: Value,
        pred: Value | None = None,
    ) -> None:
        ops: tuple[Value, ...] = (vec,) + tuple(indices)
        if pred is not None:
            ops = ops + (pred,)
        self._emit(
            _op.VecStoreOp(
                operands=ops,
                attrs={"tensor": tensor, "pred": pred},
            )
        )

    def async_copy(
        self,
        dst: SharedRegion,
        src: GlobalTensor,
        *,
        dst_idx: Sequence[Value],
        src_idx: Sequence[Value],
        count: int,
        pred: Value | None = None,
    ) -> None:
        ops = tuple(dst_idx) + tuple(src_idx)
        if pred is not None:
            ops = ops + (pred,)
        self._emit(
            _op.AsyncCopyOp(
                operands=ops,
                attrs={
                    "dst_tensor": dst,
                    "src_tensor": src,
                    "count": count,
                    "pred": pred,
                    "n_dst_idx": len(dst_idx),
                    "n_src_idx": len(src_idx),
                },
            )
        )
        self.async_emissions += 1

    def async_commit(self) -> None:
        self._emit(_op.AsyncCopyCommitOp())

    def async_wait(self, n: int = 0) -> None:
        self._emit(_op.AsyncCopyWaitOp(attrs={"n": n}))

    def atomic_rmw(
        self,
        tensor: GlobalTensor,
        op: str,
        value: Value,
        *indices: Value,
        atomic_type: str | None = None,
        name: str = "",
    ) -> Value:
        out = self._fresh(ValueShape(value.dtype, width=value.width), name)
        attrs: dict = {"tensor": tensor, "op": op}
        if atomic_type:
            attrs["atomic_type"] = atomic_type
        self._emit(
            _op.AtomicRmwOp(
                results=(out,),
                operands=(value,) + tuple(indices),
                attrs=attrs,
            )
        )
        return out

    # ---------------------------------------------------------------
    # §5.4 Shared memory allocation
    # ---------------------------------------------------------------

    def smem_alloc(
        self,
        name: str,
        dtype: DType,
        shape: tuple[int, ...],
        pad: int = 0,
        align: int = 0,
        *,
        align_bytes: int = 16,
        lifetime: Lifetime | None = None,
        readonly_after_init: bool = False,
    ) -> SharedRegion:
        """Allocate a shared-memory region and return a SharedRegion view of it.

        The underlying SmemAllocOp is stored on the Function so backends
        can hoist it to kernel entry. The returned SharedRegion covers
        the full allocation; take `.view()`s from it for sub-regions
        (e.g. pipeline stages).

        Layout-aware fields:
          * ``align_bytes``: minimum alignment guarantee. The
            ``smem_layout`` pass picks per-region offsets that respect
            this. Defaults to 16 (matches the natural alignment for
            v4.b32 / v8.b16 vec ops).
          * ``lifetime``: when this region is live. Defaults to
            ``Lifetime.auto()`` which infers from earliest/latest use.
            Pass ``Lifetime.in_region(R)`` to pin lifetime explicitly so
            the layout pass can alias storage with disjoint regions.
          * ``readonly_after_init``: marks regions written once at entry
            and only read afterwards — lets the layout pass elide some
            barriers around aliased reads.
        """
        from .lifetime import Lifetime as _Lifetime

        if lifetime is None:
            lifetime = _Lifetime.auto()
        backing = self._fresh(ValueShape(DType.U64), name=f"{name}_alloc")
        smem_op = _op.SmemAllocOp(
            results=(backing,),
            attrs={
                "name": name,
                "dtype": dtype,
                "shape": tuple(shape),
                "pad": pad,
                "align": align,
                "align_bytes": align_bytes,
                "lifetime": lifetime,
                "readonly_after_init": readonly_after_init,
            },
        )
        self._emit(smem_op)
        self.function.smem_allocs.append(smem_op)

        # Compute element strides from shape (row-major, optional pad on
        # the innermost stride for bank-conflict avoidance).
        stride = _rowmajor_stride(shape, pad=pad)
        return SharedRegion(
            dtype=dtype,
            shape=tuple(shape),
            stride=stride,
            name=name,
            alloc=backing,
            pad=pad,
            align_bytes=align_bytes,
            lifetime=lifetime,
            readonly_after_init=readonly_after_init,
        )

    # ---------------------------------------------------------------
    # §5.5 Cross-lane / subgroup
    # ---------------------------------------------------------------

    def shuffle(self, kind: str, v: Value, param: int, name: str = "") -> Value:
        out = self._fresh(v.shape, name)
        self._emit(
            _op.ShuffleOp(
                results=(out,),
                operands=(v,),
                attrs={"kind": kind, "param": param},
            )
        )
        return out

    def subgroup_reduce(self, op: str, v: Value, name: str = "") -> Value:
        out = self._fresh(v.shape, name)
        self._emit(
            _op.SubgroupReduceOp(
                results=(out,),
                operands=(v,),
                attrs={"op": op},
            )
        )
        return out

    def subgroup_broadcast(self, v: Value, lane: int, name: str = "") -> Value:
        out = self._fresh(v.shape, name)
        self._emit(
            _op.SubgroupBroadcastOp(
                results=(out,),
                operands=(v,),
                attrs={"lane": lane},
            )
        )
        return out

    # ---------------------------------------------------------------
    # §5.6 Thread identity
    # ---------------------------------------------------------------

    def thread_idx(self, dim: str = "x", name: str = "") -> Value:
        out = self._fresh(ValueShape(DType.U32), name or f"tid_{dim}")
        self._emit(_op.ThreadIdxOp(results=(out,), attrs={"dim": dim}))
        return out

    def block_idx(self, dim: str = "x", name: str = "") -> Value:
        out = self._fresh(ValueShape(DType.U32), name or f"bid_{dim}")
        self._emit(_op.BlockIdxOp(results=(out,), attrs={"dim": dim}))
        return out

    def block_dim(self, dim: str = "x", name: str = "") -> Value:
        out = self._fresh(ValueShape(DType.U32), name or f"ntid_{dim}")
        self._emit(_op.BlockDimOp(results=(out,), attrs={"dim": dim}))
        return out

    def grid_dim(self, dim: str = "x", name: str = "") -> Value:
        out = self._fresh(ValueShape(DType.U32), name or f"ngrid_{dim}")
        self._emit(_op.GridDimOp(results=(out,), attrs={"dim": dim}))
        return out

    def lane_id(self, name: str = "laneid") -> Value:
        out = self._fresh(ValueShape(DType.U32), name)
        self._emit(_op.LaneIdOp(results=(out,)))
        return out

    def subgroup_id(self, name: str = "warpid") -> Value:
        out = self._fresh(ValueShape(DType.U32), name)
        self._emit(_op.SubgroupIdOp(results=(out,)))
        return out

    def group_id(self, name: str = "groupID") -> Value:
        """MMA group id = `%laneid >> 2`.

        First-class op for the per-lane coordinate every mma fragment
        formula is written against. Use alongside `thread_id_in_group()`
        when computing per-thread smem base offsets for A/B/C loads.
        """
        out = self._fresh(ValueShape(DType.U32), name)
        self._emit(_op.GroupIdOp(results=(out,)))
        return out

    def thread_id_in_group(self, name: str = "tidIG") -> Value:
        """MMA thread-in-group id = `%laneid & 3`.

        Companion to `group_id()`. Every per-lane fragment address
        computation uses `tidIG * elems_per_lane` as the inner
        coordinate.
        """
        out = self._fresh(ValueShape(DType.U32), name)
        self._emit(_op.ThreadIdInGroupOp(results=(out,)))
        return out

    # ---------------------------------------------------------------
    # §5.7 Control flow
    # ---------------------------------------------------------------

    def barrier(self, scope: str = "block") -> None:
        self._emit(_op.BarrierOp(attrs={"scope": scope}))

    @contextmanager
    def for_loop(
        self,
        lo: Value,
        hi: Value,
        step: Value,
        *,
        iv_name: str = "i",
        carried: Sequence[Value] = (),
    ) -> Iterator[tuple[Value, tuple[Value, ...]]]:
        """Build a ForLoopOp with an explicit induction variable and
        loop-carried values. The context manager yields `(iv, (carried_in, ...))`
        for use inside the body.

        On exit, the loop is finalized:
         - the body Region is required to end with a YieldOp producing
           one Value per carried-in, with matching shapes
         - the ForLoopOp's results are exposed on the context manager
           via `for_loop.results` (use `builder.for_loop_results` too)
        """
        if lo.dtype is not hi.dtype or lo.dtype is not step.dtype:
            raise TypeError(
                f"for_loop: lo/hi/step must share a dtype, got {lo.dtype}/{hi.dtype}/{step.dtype}"
            )
        iv = self._fresh(ValueShape(lo.dtype), name=iv_name)
        carried = tuple(carried)
        # Fresh in-body values for each carried arg — these are what the
        # body sees as the loop-carried "in" values. Yielded values from
        # the body update them.
        body_carried: tuple[Value, ...] = tuple(
            self._fresh(c.shape, name=f"{iv_name}_carry{i}") for i, c in enumerate(carried)
        )
        # Results live at the for-loop level and represent the final
        # yielded values after the last iteration.
        results: tuple[Value, ...] = tuple(
            self._fresh(c.shape, name=f"{iv_name}_out{i}") for i, c in enumerate(carried)
        )

        body_region = Region()
        for_op = _op.ForLoopOp(
            results=results,
            operands=(lo, hi, step) + carried,
            attrs={"iv_name": iv_name, "iv_dtype": lo.dtype},
            regions=(body_region,),
            induction_var=iv,
            carried_body_vars=body_carried,
        )
        # Rebind iv producer to the ForLoopOp (they're both owned by it).
        iv.producer = for_op
        for c in body_carried:
            c.producer = for_op
        # The for-op itself lives in the outer region.
        self.current_region.append(for_op)
        # Enter the body region.
        self._region_stack.append(body_region)
        self._cse_stack.append({})
        try:
            yield iv, body_carried
            # Auto-close: if the body didn't yield, require the op to
            # have no carried values (so the empty yield is OK).
            if body_region.terminator is None:
                if carried:
                    raise RuntimeError(
                        "for_loop: body must end with builder.yield_(...) "
                        "when the loop has carried values"
                    )
                self._emit(_op.YieldOp())
            # Validate the terminator signature matches carried shapes.
            term = body_region.terminator
            assert term is not None
            if len(term.operands) != len(carried):
                raise RuntimeError(
                    f"for_loop: yield produced {len(term.operands)} values, "
                    f"loop carries {len(carried)}"
                )
            for i, (yielded, expected) in enumerate(zip(term.operands, carried, strict=False)):
                if yielded.shape != expected.shape:
                    raise TypeError(
                        f"for_loop: yield[{i}] shape {yielded.shape} != "
                        f"carried[{i}] shape {expected.shape}"
                    )
        finally:
            popped = self._region_stack.pop()
            assert popped is body_region
            self._cse_stack.pop()

        # Stash results on the for_op for consumers via builder.last_results
        self._last_for_results = results

    @property
    def last_results(self) -> tuple[Value, ...]:
        """Results of the most recently closed for_loop / if_region."""
        return getattr(self, "_last_for_results", ())

    @contextmanager
    def _region_scope(self, region: Region) -> Iterator[None]:
        self._region_stack.append(region)
        self._cse_stack.append({})
        try:
            yield
        finally:
            popped = self._region_stack.pop()
            assert popped is region
            self._cse_stack.pop()

    @contextmanager
    def if_(
        self,
        pred: Value,
        *,
        carried: Sequence[Value] = (),
    ) -> Iterator[tuple[tuple[Value, ...], tuple[Value, ...], _IfArms]]:
        """Ergonomic structured-if builder.

        Usage:
            with builder.if_(pred, carried=[v0]) as (then_in, else_in, arms):
                with arms.then_():
                    ... use then_in[0] ...
                    builder.yield_(new_v0_then)
                with arms.else_():
                    ... use else_in[0] ...
                    builder.yield_(new_v0_else)
            result = builder.last_results[0]
        """
        if pred.dtype is not DType.PRED:
            raise TypeError("if_: pred must be a PRED Value")

        carried = tuple(carried)
        then_in = tuple(self._fresh(c.shape, name=f"then_in{i}") for i, c in enumerate(carried))
        else_in = tuple(self._fresh(c.shape, name=f"else_in{i}") for i, c in enumerate(carried))
        results = tuple(self._fresh(c.shape, name=f"if_out{i}") for i, c in enumerate(carried))

        then_region = Region()
        else_region = Region()
        if_op = _op.IfRegionOp(
            results=results,
            operands=(pred,) + carried + carried,
            attrs={"n_carried": len(carried)},
            regions=(then_region, else_region),
            then_body_vars=then_in,
            else_body_vars=else_in,
        )
        for v in then_in:
            v.producer = if_op
        for v in else_in:
            v.producer = if_op
        self.current_region.append(if_op)

        arms = _IfArms(
            builder=self,
            then_region=then_region,
            else_region=else_region,
            n_carried=len(carried),
            if_op=if_op,
        )
        try:
            yield then_in, else_in, arms
        finally:
            arms._finalize()
            self._last_for_results = results

    def yield_(self, *values: Value) -> None:
        """Terminator for the currently-open Region."""
        if not self._region_stack:
            raise RuntimeError("yield_: no active region")
        self._emit(_op.YieldOp(operands=tuple(values)))

    # ---------------------------------------------------------------
    # §5.8 Matmul
    # ---------------------------------------------------------------

    def load_matrix(
        self,
        src: Tensor,
        shape_id: str,
        which: str,
        row: Value | int = 0,
        col: Value | int = 0,
        layout_hint: str = "manual",
        reg_offsets: tuple[tuple[int, int], ...] | None = None,
        name: str = "",
    ) -> Value:
        """Emit a fragment load.

        The fragment width (number of per-thread registers) comes from
        `MmaShape.{a,b,c}_regs`. The PTX lowerer's default path uses
        `reg_offsets` — one `(row_elem, col_elem)` pair per register —
        to emit N scalar `ld.shared.b32` instructions at the correct
        per-register positions. Callers source these offsets from the
        PTX ISA fragment formulas (§9.7.14.5) for their chosen smem
        layout; for row.col (A rowmajor, B^T in smem so K is the
        contiguous dim), see tests/lower/ptx/test_matmul.py for the
        authoritative tables.

        Passing `layout_hint="ldmatrix"` opts into the `ldmatrix.sync
        .aligned.x<N>.m8n8.shared.b16` fast path — in that case
        `reg_offsets` is not needed (and is ignored by the lowerer).
        """
        shape = self.module.kernel_shapes.get(shape_id)
        if shape is None:
            raise KeyError(f"load_matrix: shape_id {shape_id!r} not registered in module")
        # Coerce int offsets to ConstOps of U32 for the IR.
        row_v = row if not isinstance(row, int) else self.const(DType.U32, row)
        col_v = col if not isinstance(col, int) else self.const(DType.U32, col)
        # Fragment Values are width-N vecs where N is the per-thread
        # register count declared by the MmaShape. The carrier dtype
        # depends on the role:
        #   - a / b (always packed subword elements): b32
        #   - c / d (accumulator): matches MmaShape.acc_dtype so the
        #     PTX mma.sync.aligned.*.f32 C slot gets f32-typed regs,
        #     .s32 slot gets s32-typed regs, etc. Fallback to b32 if
        #     the shape has no reg counts (legacy smoke tests).
        reg_count = {"a": shape.a_regs, "b": shape.b_regs, "c": shape.c_regs}[which]
        frag_width = reg_count if reg_count > 0 else 1
        carrier_dtype = _frag_carrier_dtype(shape, which)
        out = self._fresh(ValueShape(carrier_dtype, width=frag_width), name or f"frag_{which}")
        attrs: dict = {
            "src_tensor": src,
            "shape_id": shape_id,
            "which": which,
            "layout_hint": layout_hint,
        }
        if reg_offsets is not None:
            attrs["reg_offsets"] = tuple((int(dr), int(dc)) for (dr, dc) in reg_offsets)
        self._emit(
            _op.LoadMatrixOp(
                results=(out,),
                operands=(row_v, col_v),
                attrs=attrs,
            )
        )
        return out

    def store_matrix(
        self,
        dst: Tensor,
        frag: Value,
        shape_id: str,
        which: str,
        row: Value | int = 0,
        col: Value | int = 0,
        layout_hint: str = "manual",
        reg_offsets: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Emit a fragment store.

        Default lowering emits N scalar `st.<space>.b32` instructions at
        per-register offsets derived from `reg_offsets`. Callers supply
        the offsets from the PTX ISA fragment formulas for the chosen
        smem layout.
        """
        if shape_id not in self.module.kernel_shapes:
            raise KeyError(f"store_matrix: shape_id {shape_id!r} not registered")
        row_v = row if not isinstance(row, int) else self.const(DType.U32, row)
        col_v = col if not isinstance(col, int) else self.const(DType.U32, col)
        attrs: dict = {
            "dst_tensor": dst,
            "shape_id": shape_id,
            "which": which,
            "layout_hint": layout_hint,
        }
        if reg_offsets is not None:
            attrs["reg_offsets"] = tuple((int(dr), int(dc)) for (dr, dc) in reg_offsets)
        self._emit(
            _op.StoreMatrixOp(
                operands=(frag, row_v, col_v),
                attrs=attrs,
            )
        )

    def store_matrix_gate_residual(
        self,
        dst: Tensor,
        residual: Tensor,
        gate: Tensor,
        frag: Value,
        shape_id: str,
        row: Value | int = 0,
        col: Value | int = 0,
        m_per_group: int = 1,
    ) -> None:
        """Emit a fragment store fused with the AdaGate-residual epilogue.

        Writes ``dst[r, c] = residual[r, c] + gate[r // m_per_group, c]
        * frag[r, c]``. NAX-only (Metal M5+); the frag must be the
        destination of a NAX MmaOp (width-16 F32). See
        ``StoreMatrixGateResidualOp`` for the full contract.

        ``m_per_group`` is ``M / G`` from the GemmSpec — the row-stride
        between gate rows. ``m_per_group >= M`` (the spec ``G == 1``
        case) gives the constant-row-0 broadcast.
        """
        if shape_id not in self.module.kernel_shapes:
            raise KeyError(f"store_matrix_gate_residual: shape_id {shape_id!r} not registered")
        row_v = row if not isinstance(row, int) else self.const(DType.U32, row)
        col_v = col if not isinstance(col, int) else self.const(DType.U32, col)
        attrs: dict = {
            "dst_tensor": dst,
            "residual_tensor": residual,
            "gate_tensor": gate,
            "shape_id": shape_id,
            "which": "d",
            "m_per_group": int(m_per_group),
        }
        self._emit(
            _op.StoreMatrixGateResidualOp(
                operands=(frag, row_v, col_v),
                attrs=attrs,
            )
        )

    def mma(
        self,
        shape_id: str,
        a: Value,
        b: Value,
        c: Value,
        *,
        transpose_b: bool = True,
        accumulate: bool = True,
        name: str = "",
    ) -> Value:
        """Emit an MMA: ``d = a * (b^T if transpose_b else b) [+ c]``.

        ``transpose_b`` is only meaningful for backends whose MMA
        primitive exposes the choice — NAX ``matmul2d_descriptor`` does;
        PTX ``mma.sync`` doesn't (the layout is fixed by the shape's
        mnemonic). Default ``True`` matches every existing call site
        and the registered NAX shape's current ``transpose_b=true``
        emission. PTX visitor asserts this is ``True``; NAX visitor
        reads it. Used by attention's GEMM2 (``S @ V`` with V loaded
        K-contiguous → no transpose).

        ``accumulate=False`` selects matmul2d's ``multiply`` mode (D =
        A*B; C is ignored). The first MMA in a K-loop chain can use
        this to skip both reading C and the upstream zero-init that
        would have produced it. NAX-only; PTX asserts True.
        """
        shape = self.module.kernel_shapes.get(shape_id)
        if shape is None:
            raise KeyError(f"mma: shape_id {shape_id!r} not registered")
        # d fragment shares c's layout: same reg count, same carrier dtype
        # (f32 / s32 / b32 depending on acc_dtype — see
        # `_frag_carrier_dtype` for the mapping).
        reg_count = shape.c_regs if shape.c_regs > 0 else 1
        carrier_dtype = _frag_carrier_dtype(shape, "d")
        out = self._fresh(ValueShape(carrier_dtype, width=reg_count), name or "mma_d")
        attrs: dict = {"shape_id": shape_id}
        if not transpose_b:
            # Only set when non-default to keep IR text concise; visitors
            # treat the absence of the attr as ``True``.
            attrs["transpose_b"] = False
        if not accumulate:
            attrs["accumulate"] = False
        self._emit(
            _op.MmaOp(
                results=(out,),
                operands=(a, b, c),
                attrs=attrs,
            )
        )
        return out

    def frag_apply(
        self,
        shape_id: str,
        frag: Value,
        fn,
        *,
        selectors: Sequence[Value] = (),
        slot_to_selector_idx: tuple[int, ...] | None = None,
        name: str = "",
    ) -> Value:
        """Apply a scalar transform ``fn`` to every element of an accumulator
        fragment. Returns a new fragment of the same shape and dtype.

        Ergonomically this is ``tile.map(fn)`` in the RegisterTile fluent API;
        at the IR level it's a ``FragApplyOp`` with a body region built from
        ``fn``. ``fn`` takes one scalar Value (the element) and returns one
        scalar Value (the transformed element). The body is emitted ONCE at
        IR build time and replicated per storage slot at lower time.

        On PTX this compiles to per-reg scalar ops. On MSL it compiles to
        ``thread_elements()`` reads and writes — no threadgroup round-trip.

        Per-slot selectors (optional). When ``selectors=`` is given, ``fn``
        receives ``(elem, selector)`` and the lowerer binds ``selector``
        per slot to ``selectors[slot_to_selector_idx[slot]]``. This is how
        ``.map_per_row_class(scales, fn)`` builds the O-rescale pattern
        (2 scales, 4 slots mapped ``(0, 0, 1, 1)`` per cd_offsets).
        """
        if frag.dtype is not DType.F32:
            raise TypeError(f"frag_apply: frag must be f32 accumulator (got {frag.dtype})")
        shape = self.module.kernel_shapes.get(shape_id)
        if shape is None:
            raise KeyError(f"frag_apply: shape {shape_id!r} not registered")
        selectors = tuple(selectors)
        if selectors:
            # ``slot_to_selector_idx=None`` opts into dynamic-row-dispatch
            # mode: the lowerer computes the row from the smem slot index
            # at runtime and OpSelect-chains the right selector. Used by
            # the SPV/Intel coopmat path where the lane↔(r,c) mapping is
            # implementation-private — no compile-time per-c_reg
            # selector index is recoverable. In dynamic mode, selectors
            # must be ``rows``-many (one per matrix row).
            if slot_to_selector_idx is not None:
                n_slots = shape.c_regs
                if len(slot_to_selector_idx) != n_slots:
                    raise ValueError(
                        f"frag_apply: slot_to_selector_idx length {len(slot_to_selector_idx)} "
                        f"≠ shape.c_regs {n_slots}"
                    )

        # The body sees the per-slot scalar element as a fresh Value.
        # Lower time binds this Value to a different per-slot scalar name
        # on each re-walk.
        elem = self._fresh(ValueShape(frag.dtype), "apply_elem")
        sel_var: Value | None = None
        if selectors:
            sel_dtype = selectors[0].dtype
            sel_var = self._fresh(ValueShape(sel_dtype), "apply_sel")

        body_region = Region()
        self._region_stack.append(body_region)
        self._cse_stack.append({})
        try:
            if sel_var is not None:
                out_elem = fn(elem, sel_var)
            else:
                out_elem = fn(elem)
            if not isinstance(out_elem, Value):
                raise TypeError(
                    f"frag_apply: fn must return a Value, got {type(out_elem).__name__}"
                )
            if out_elem.shape != elem.shape:
                raise TypeError(
                    f"frag_apply: fn produced shape {out_elem.shape}, expected {elem.shape}"
                )
            self._emit(_op.YieldOp(operands=(out_elem,)))
        finally:
            self._region_stack.pop()
            self._cse_stack.pop()

        out = self._fresh(frag.shape, name or "frag_apply")
        attrs: dict = {"shape_id": shape_id}
        if selectors and slot_to_selector_idx is not None:
            attrs["slot_to_selector_idx"] = tuple(slot_to_selector_idx)
        # else: dynamic-row-dispatch mode — slot_to_selector_idx absent.
        op = _op.FragApplyOp(
            results=(out,),
            operands=(frag, *selectors),
            attrs=attrs,
            regions=(body_region,),
            body_input_var=elem,
            body_selector_var=sel_var,
        )
        elem.producer = op
        if sel_var is not None:
            sel_var.producer = op
        self._emit(op)
        return out

    def frag_for_each(
        self,
        shape_id: str,
        frag: Value,
        fn,
        cd_offsets: tuple[tuple[int, int], ...],
        *,
        selectors: Sequence[Value] = (),
        slot_to_selector_idx: tuple[int, ...] | None = None,
    ) -> None:
        """Apply ``fn(elem, row, col [, selector])`` to every storage
        slot of a fragment, with side effects only — no output fragment.

        This is the "epilogue primitive". ``fn`` takes:
          * ``elem``: scalar IR Value of the slot's element
          * ``row``: U32 IR Value, tile-local row (lane-dependent — the
            lowerer binds it to ``groupID + dr`` on PTX, the Apple row
            formula + tile offset on MSL)
          * ``col``: U32 IR Value, tile-local col
          * ``selector`` (iff ``selectors=`` given): the per-slot value
            from ``selectors[slot_to_selector_idx[slot]]`` (e.g., per
            row-class ``l_rcp`` scalar for normalize-and-store).

        Body emits store/atomic ops using these Values; it returns None.
        """
        if frag.dtype is not DType.F32:
            raise TypeError(f"frag_for_each: frag must be f32 accumulator (got {frag.dtype})")
        shape = self.module.kernel_shapes.get(shape_id)
        if shape is None:
            raise KeyError(f"frag_for_each: shape {shape_id!r} not registered")
        selectors = tuple(selectors)
        if selectors:
            if slot_to_selector_idx is None:
                raise ValueError("frag_for_each: selectors= given without slot_to_selector_idx")
            n_slots = shape.c_regs
            if len(slot_to_selector_idx) != n_slots:
                raise ValueError(
                    f"frag_for_each: slot_to_selector_idx length "
                    f"{len(slot_to_selector_idx)} ≠ shape.c_regs {n_slots}"
                )

        elem = self._fresh(ValueShape(frag.dtype), "foreach_elem")
        row = self._fresh(ValueShape(DType.U32), "foreach_row")
        col = self._fresh(ValueShape(DType.U32), "foreach_col")
        sel_var: Value | None = None
        if selectors:
            sel_var = self._fresh(ValueShape(selectors[0].dtype), "foreach_sel")

        body_region = Region()
        self._region_stack.append(body_region)
        self._cse_stack.append({})
        try:
            if sel_var is not None:
                ret = fn(elem, row, col, sel_var)
            else:
                ret = fn(elem, row, col)
            if ret is not None:
                raise TypeError(
                    "frag_for_each: fn must return None (side-effect body), "
                    f"got {type(ret).__name__}"
                )
            self._emit(_op.YieldOp())
        finally:
            self._region_stack.pop()
            self._cse_stack.pop()

        attrs: dict = {"shape_id": shape_id, "cd_offsets": cd_offsets}
        if selectors:
            assert slot_to_selector_idx is not None
            attrs["slot_to_selector_idx"] = tuple(slot_to_selector_idx)
        op = _op.FragForEachOp(
            results=(),
            operands=(frag, *selectors),
            attrs=attrs,
            regions=(body_region,),
            body_input_var=elem,
            body_row_var=row,
            body_col_var=col,
            body_selector_var=sel_var,
        )
        elem.producer = op
        row.producer = op
        col.producer = op
        if sel_var is not None:
            sel_var.producer = op
        self._emit(op)

    def frag_convert(
        self,
        shape_id: str,
        src_frags: Sequence[Value],
        *,
        src_layout: str,
        dst_layout: str,
        src_dtype: DType,
        dst_dtype: DType,
        cd_offsets: tuple[tuple[int, int], ...],
        fn=None,
        selectors: Sequence[Value] = (),
        slot_to_selector_idx: tuple[int, ...] | None = None,
        name: str = "",
    ) -> Value:
        """Convert ``kf`` source fragments of ``(src_layout, src_dtype)`` to
        one output fragment of ``(dst_layout, dst_dtype)``.

        Currently supports only ``src_layout="acc", dst_layout="a_frag"``
        with ``src_dtype=F32, dst_dtype=BF16`` — the P-fragment path in
        online_softmax. The output is a width-``shape.a_regs`` b32 Value
        suitable to feed directly into ``b.mma``; on MSL the lowerer
        produces a ``simdgroup_matrix<bf16, 8, 8>[mf*kf]`` array registered
        in ``frag_values`` so MMA consumes it without a b32→simdgroup
        unpack.

        ``fn(elem)`` (optional) applies a per-element f32 transform before
        the bf16 cast. With ``selectors=`` given, ``fn(elem, sel)`` receives
        the per-slot selector (e.g., per-row-class ``m_new``).
        """
        if (src_layout, dst_layout) != ("acc", "a_frag"):
            raise NotImplementedError(
                f"frag_convert: only acc→a_frag supported today (got {src_layout}→{dst_layout})"
            )
        if (src_dtype, dst_dtype) != (DType.F32, DType.BF16):
            raise NotImplementedError(
                f"frag_convert: only f32→bf16 supported for acc→a_frag "
                f"(got {src_dtype}→{dst_dtype})"
            )
        shape = self.module.kernel_shapes.get(shape_id)
        if shape is None:
            raise KeyError(f"frag_convert: shape {shape_id!r} not registered")
        src_frags = tuple(src_frags)
        kf = len(src_frags)
        if kf < 1:
            raise ValueError("frag_convert: need ≥1 source fragment")
        for sf in src_frags:
            if sf.dtype is not src_dtype:
                raise TypeError(f"frag_convert: src frag dtype {sf.dtype} ≠ src_dtype {src_dtype}")
        selectors = tuple(selectors)
        if selectors and slot_to_selector_idx is None:
            raise ValueError("frag_convert: selectors= given without slot_to_selector_idx")

        # Body vars + optional body region.
        elem = self._fresh(ValueShape(src_dtype), "conv_elem")
        sel_var: Value | None = None
        if selectors:
            sel_var = self._fresh(ValueShape(selectors[0].dtype), "conv_sel")

        body_region = Region()
        regions: tuple[Region, ...] = ()
        if fn is not None:
            self._region_stack.append(body_region)
            self._cse_stack.append({})
            try:
                if sel_var is not None:
                    yielded = fn(elem, sel_var)
                else:
                    yielded = fn(elem)
                if not isinstance(yielded, Value):
                    raise TypeError("frag_convert: fn must return a Value")
                if yielded.shape != elem.shape:
                    raise TypeError(
                        f"frag_convert: fn yielded shape {yielded.shape}, expected {elem.shape}"
                    )
                self._emit(_op.YieldOp(operands=(yielded,)))
            finally:
                self._region_stack.pop()
                self._cse_stack.pop()
            regions = (body_region,)

        # Output: width=a_regs b32 — the A-fragment carrier.
        out_width = shape.a_regs
        out = self._fresh(
            ValueShape(DType.B32, width=out_width),
            name or "frag_conv",
        )
        attrs: dict = {
            "shape_id": shape_id,
            "src_layout": src_layout,
            "dst_layout": dst_layout,
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "num_src_frags": kf,
            "cd_offsets": cd_offsets,
        }
        if selectors:
            assert slot_to_selector_idx is not None
            attrs["slot_to_selector_idx"] = tuple(slot_to_selector_idx)
        op = _op.FragConvertOp(
            results=(out,),
            operands=(*src_frags, *selectors),
            attrs=attrs,
            regions=regions,
            body_input_var=elem if fn is not None else None,
            body_selector_var=sel_var if fn is not None else None,
        )
        if fn is not None:
            elem.producer = op
            if sel_var is not None:
                sel_var.producer = op
        self._emit(op)
        return out

    def frag_slice(
        self,
        frag: Value,
        *,
        start: int,
        length: int,
        name: str = "",
    ) -> Value:
        """Take a contiguous component slice of a fragment Value.

        Used by NAX attention to split a width-16 GEMM1 accumulator
        (two 16×16 N-tiles per lane) into two width-8 A operands for
        GEMM2. Pure component-rebinding at lower time — no MSL emitted,
        the slice's storage class is inherited from the source.
        """
        if start < 0 or length < 1:
            raise ValueError(f"frag_slice: start={start}, length={length} must be ≥0/≥1")
        if start + length > frag.width:
            raise ValueError(
                f"frag_slice: slice [{start}:{start + length}] exceeds source width {frag.width}"
            )
        out = self._fresh(ValueShape(frag.dtype, width=length), name or "frag_slice")
        self._emit(
            _op.FragSliceOp(
                results=(out,),
                operands=(frag,),
                attrs={"start": int(start), "length": int(length)},
            )
        )
        return out

    def frag_reduce(
        self,
        shape_id: str,
        frag: Value,
        *,
        kind: str,
        axis: str,
        cd_offsets: tuple[tuple[int, int], ...],
        n_classes_override: int | None = None,
        name: str = "",
    ) -> tuple[Value, ...]:
        """Reduce an accumulator fragment along one axis.

        For ``axis="row"``: reduce across columns, produce one scalar
        per row class. For ``axis="col"``: reduce across rows, produce
        one per col class.

        Row/col classes are derived from ``cd_offsets``: sorted distinct
        ``dr`` values (axis=row) or ``dc`` values (axis=col). The lowerer
        emits both the local per-c_reg/per-slot reduction AND the
        cross-lane butterfly shuffle so every lane in the class ends up
        holding the reduced value.

        Returns a tuple of scalar Values, one per class, in the sorted
        order of the class indices (so for cd_offsets ``((0,0),(0,1),
        (8,0),(8,1))`` with axis=row, ``result[0]`` is the row class
        dr=0 reduction and ``result[1]`` is dr=8).

        ``n_classes_override`` lets a backend with an implementation-
        private lane mapping (Intel KHR coopmat) ask for ``rows`` (or
        ``cols``) results without fitting through the per-c_reg
        ``cd_offsets`` convention. Set this to ``shape.m`` (axis=row)
        on the SPV/Intel path; the lowerer derives row classes from
        the smem layout instead of cd_offsets.
        """
        if frag.dtype is not DType.F32:
            raise TypeError(f"frag_reduce: frag must be f32 accumulator (got {frag.dtype})")
        shape = self.module.kernel_shapes.get(shape_id)
        if shape is None:
            raise KeyError(f"frag_reduce: shape {shape_id!r} not registered")
        if axis == "row":
            classes = sorted({dr for dr, _ in cd_offsets})
        elif axis == "col":
            classes = sorted({dc for _, dc in cd_offsets})
        else:
            raise ValueError(f"frag_reduce: axis must be 'row'|'col' (got {axis!r})")

        n_results = (
            int(n_classes_override) if n_classes_override is not None else len(classes)
        )
        results = tuple(
            self._fresh(ValueShape(frag.dtype), name or f"frag_red{i}") for i in range(n_results)
        )
        attrs: dict = {
            "shape_id": shape_id,
            "axis": axis,
            "kind": kind,
            "cd_offsets": cd_offsets,
        }
        if n_classes_override is not None:
            attrs["n_classes_override"] = int(n_classes_override)
        self._emit(
            _op.FragReduceOp(
                results=results,
                operands=(frag,),
                attrs=attrs,
            )
        )
        return results


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _frag_carrier_dtype(shape: MmaShape, which: str) -> DType:
    """Return the per-thread carrier DType for a fragment role.

    A / B multiplicand fragments are always packed into b32 registers
    (two bf16, two f16, four e4m3, four s8, etc. per reg). C / D
    accumulator fragments take their carrier from the shape's
    `acc_dtype` so the PTX mma.sync's typed accumulator slot gets the
    right register class:

      - F32 acc → f32 regs (one f32 per reg)
      - S32 acc → s32 regs (one s32 per reg)
      - F16 acc → b32 regs (packed f16x2)
      - anything else → b32 regs (safe fallback)
    """
    if which in ("a", "b"):
        return DType.B32
    acc = shape.acc_dtype
    if acc is DType.F32:
        return DType.F32
    if acc is DType.S32:
        return DType.S32
    # F16 / BF16 / etc. accumulators ride in b32-packed regs.
    return DType.B32


def _rowmajor_stride(shape: tuple[int, ...], pad: int = 0) -> tuple[int, ...]:
    """Row-major element stride with optional padding on the row stride.

    For a 2D shape (R, C) with pad P, the stride is (C + P, 1). Higher
    ranks ignore `pad` (we only pad the row of 2D tiles today).
    """
    if len(shape) == 0:
        return ()
    if len(shape) == 1:
        return (1,)
    if len(shape) == 2:
        return (shape[1] + pad, 1)
    # General row-major for rank >= 3
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class _IfArms:
    """Handle returned by `builder.if_()` to enter the then/else arms
    in order. Each arm is a context manager that enters the corresponding
    Region and validates the yield on exit."""

    def __init__(
        self,
        builder: Builder,
        then_region: Region,
        else_region: Region,
        n_carried: int,
        if_op: _op.IfRegionOp,
    ) -> None:
        self._builder = builder
        self._then_region = then_region
        self._else_region = else_region
        self._n_carried = n_carried
        self._if_op = if_op
        self._then_done = False
        self._else_done = False

    @contextmanager
    def then_(self) -> Iterator[None]:
        if self._then_done:
            raise RuntimeError("if_.then_(): already entered once")
        self._builder._region_stack.append(self._then_region)
        self._builder._cse_stack.append({})
        try:
            yield
        finally:
            popped = self._builder._region_stack.pop()
            assert popped is self._then_region
            self._builder._cse_stack.pop()
            self._then_done = True

    @contextmanager
    def else_(self) -> Iterator[None]:
        if self._else_done:
            raise RuntimeError("if_.else_(): already entered once")
        self._builder._region_stack.append(self._else_region)
        self._builder._cse_stack.append({})
        try:
            yield
        finally:
            popped = self._builder._region_stack.pop()
            assert popped is self._else_region
            self._builder._cse_stack.pop()
            self._else_done = True

    def _finalize(self) -> None:
        # Ensure both arms ran at least once; require a matching yield
        # signature from each arm.
        if not self._then_done:
            raise RuntimeError("if_: then_() arm was never entered")
        if not self._else_done:
            raise RuntimeError("if_: else_() arm was never entered")
        self._require_yield(self._then_region, "then")
        self._require_yield(self._else_region, "else")
        then_yield = self._then_region.terminator
        else_yield = self._else_region.terminator
        assert then_yield is not None and else_yield is not None
        if len(then_yield.operands) != self._n_carried:
            raise RuntimeError(
                f"if_: then yield has {len(then_yield.operands)} values, expected {self._n_carried}"
            )
        if len(else_yield.operands) != self._n_carried:
            raise RuntimeError(
                f"if_: else yield has {len(else_yield.operands)} values, expected {self._n_carried}"
            )
        for i, (t, e) in enumerate(zip(then_yield.operands, else_yield.operands, strict=False)):
            if t.shape != e.shape:
                raise TypeError(f"if_: yield[{i}] shape mismatch then={t.shape} vs else={e.shape}")

    def _require_yield(self, region: Region, which: str) -> None:
        if region.terminator is None:
            if self._n_carried == 0:
                # OK to omit for zero-carried: add an empty yield.
                region.ops.append(_op.YieldOp())
                return
            raise RuntimeError(
                f"if_: {which} arm must end with builder.yield_(...) when the if has carried values"
            )
