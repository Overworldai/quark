"""Tests for the concrete legalization rewrites.

EXEMPT FROM 500-LINE RULE: pairs 1:1 with ``quark.lower.legalizations``
(5 rewrites + driver properties + idempotency + nested-region walks).
Splitting forces test-class scatter or cross-file fixture sharing;
the single file reads top-to-bottom as one class per rewrite plus
a cross-cutting properties block. Hard cap 800 still applies.

Pairs with :mod:`quark.lower.legalizations`. Each rewrite gets a test
for its keep-path (backend has the primitive) and its expansion
path (backend lacks it — asserts the produced IR chain). Plus two
property sections: idempotency (``legalize`` twice == once) and
nested-region walks (rewrites fire inside ForLoopOp bodies).
"""

from __future__ import annotations

from quark.ir import DType
from quark.ir.builder import Builder
from quark.ir.op import ArithOp, AtomicRmwOp, SubgroupReduceOp

# Ensure concrete legalizations are registered.
from quark.lower import legalizations as _legalizations  # noqa: F401
from quark.lower.legalize import legalize

# -----------------------------------------------------------------------------
# Caps fakes
# -----------------------------------------------------------------------------


class _Caps:
    """Minimal caps stand-in with only the flags the rewrites consult."""

    def __init__(
        self,
        *,
        has_fma_bf16x2: bool = True,
        atomic_add_vector: frozenset = frozenset(),
        supports_async_copy: bool = True,
        has_native_subgroup_reduce: bool = False,
    ):
        self.has_fma_bf16x2 = has_fma_bf16x2
        self.atomic_add_vector = atomic_add_vector
        self.supports_async_copy = supports_async_copy
        self.has_native_subgroup_reduce = has_native_subgroup_reduce


_CUDA_SM90 = _Caps(
    has_fma_bf16x2=True,
    atomic_add_vector=frozenset({(DType.BF16, 2), (DType.F16, 2)}),
    supports_async_copy=True,
)
_NO_BF16X2 = _Caps(has_fma_bf16x2=False)
_NO_ATOMIC_BF16X2 = _Caps(atomic_add_vector=frozenset())
_NO_ASYNC = _Caps(supports_async_copy=False)


# -----------------------------------------------------------------------------
# ArithOp(kind="fma_bf16x2") / ArithOp(kind="cvt_rn_bf16x2_f32")
# -----------------------------------------------------------------------------


class TestBf16x2Arith:
    def _build_fma_bf16x2(self):
        b = Builder("m")
        b.begin_function("fn")
        z = b.const(DType.B32, 0)
        b.fma_bf16x2(z, z, z)
        b.end_function()
        return b.module

    def _build_cvt_rn_bf16x2_f32(self):
        b = Builder("m")
        b.begin_function("fn")
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        b.cvt_rn_bf16x2_f32(a, c)
        b.end_function()
        return b.module

    def test_keeps_fma_bf16x2_when_cap_true(self):
        m = self._build_fma_bf16x2()
        legalize(m, _CUDA_SM90)
        # FMA op remains as-is.
        fmas = [o for o in m.functions[0].body.ops if isinstance(o, ArithOp)]
        assert any(o.attrs.get("kind") == "fma_bf16x2" for o in fmas)

    def test_keeps_cvt_rn_bf16x2_f32_when_cap_true(self):
        m = self._build_cvt_rn_bf16x2_f32()
        legalize(m, _CUDA_SM90)
        cvts = [o for o in m.functions[0].body.ops if isinstance(o, ArithOp)]
        assert any(o.attrs.get("kind") == "cvt_rn_bf16x2_f32" for o in cvts)

    def test_expands_fma_bf16x2_without_cap(self):
        """Real expansion: ``fma_bf16x2`` → 3 ``SplitB32Op`` + 8
        ``BitcastOp`` (6 B16→BF16 on inputs, 2 BF16→B16 on outputs) +
        8 ``ConvertOp`` (6 BF16→F32, 2 F32→BF16) + 2 scalar F32
        ``ArithOp(kind="fma")`` + 1 ``MergeB32Op``. Final merge reuses
        the original op's result Value."""
        from quark.ir import DType
        from quark.ir.op import ArithOp, BitcastOp, ConvertOp, MergeB32Op, SplitB32Op

        m = self._build_fma_bf16x2()
        orig_ops = list(m.functions[0].body.ops)
        orig_fma = next(o for o in orig_ops if o.attrs.get("kind") == "fma_bf16x2")
        orig_result = orig_fma.results[0]
        assert orig_result.dtype is DType.B32

        legalize(m, _NO_BF16X2)
        ops = m.functions[0].body.ops

        splits = [o for o in ops if isinstance(o, SplitB32Op)]
        bitcasts = [o for o in ops if isinstance(o, BitcastOp)]
        converts = [o for o in ops if isinstance(o, ConvertOp)]
        fmas = [o for o in ops if isinstance(o, ArithOp) and o.attrs.get("kind") == "fma"]
        merges = [o for o in ops if isinstance(o, MergeB32Op)]

        assert len(splits) == 3, f"expected 3 SplitB32Op, got {len(splits)}"
        assert len(bitcasts) == 8, f"expected 8 BitcastOp, got {len(bitcasts)}"
        # 6 BF16-bound bitcasts (inputs) + 2 B16-bound bitcasts (outputs).
        bf16_casts = [b for b in bitcasts if b.attrs["dst_dtype"] is DType.BF16]
        b16_casts = [b for b in bitcasts if b.attrs["dst_dtype"] is DType.B16]
        assert len(bf16_casts) == 6
        assert len(b16_casts) == 2
        assert len(converts) == 8, f"expected 8 ConvertOp, got {len(converts)}"
        # 6 widening converts (BF16→F32), 2 narrowing (F32→BF16).
        widen = [c for c in converts if c.attrs["dst_dtype"] is DType.F32]
        narrow = [c for c in converts if c.attrs["dst_dtype"] is DType.BF16]
        assert len(widen) == 6
        assert len(narrow) == 2
        assert len(fmas) == 2, f"expected 2 scalar f32 FMAs, got {len(fmas)}"
        assert len(merges) == 1
        # Identity preservation: the final merge's result is the
        # original fma_bf16x2's result Value.
        assert merges[0].results[0] is orig_result

    def test_expands_cvt_rn_bf16x2_f32_without_cap(self):
        """Real expansion: ``cvt_rn_bf16x2_f32`` → two ``ConvertOp``
        (F32→BF16), two ``BitcastOp`` (BF16→B16), one ``MergeB32Op``.
        The final MergeB32Op's result reuses the original op's result
        Value so downstream operand tuples stay valid without a
        separate use-replacement pass."""
        from quark.ir import DType
        from quark.ir.op import BitcastOp, ConvertOp, MergeB32Op

        m = self._build_cvt_rn_bf16x2_f32()
        # Capture the pre-rewrite result Value so we can assert the
        # post-rewrite chain preserves its identity.
        orig_ops = list(m.functions[0].body.ops)
        orig_cvt = next(o for o in orig_ops if o.attrs.get("kind") == "cvt_rn_bf16x2_f32")
        orig_result = orig_cvt.results[0]
        assert orig_result.dtype is DType.B32

        legalize(m, _NO_BF16X2)
        ops = m.functions[0].body.ops

        # Chain contents: 2 consts (unchanged), 2 converts, 2 bitcasts, 1 merge.
        converts = [o for o in ops if isinstance(o, ConvertOp)]
        bitcasts = [o for o in ops if isinstance(o, BitcastOp)]
        merges = [o for o in ops if isinstance(o, MergeB32Op)]
        assert len(converts) == 2
        assert all(c.attrs["dst_dtype"] is DType.BF16 for c in converts)
        assert len(bitcasts) == 2
        assert all(b.attrs["dst_dtype"] is DType.B16 for b in bitcasts)
        assert len(merges) == 1
        # Identity preservation: the merge's result Value is the
        # original cvt_rn_bf16x2_f32's result Value.
        assert merges[0].results[0] is orig_result

    def test_other_arith_kinds_passthrough(self):
        """Regular ``ArithOp(kind="fma")`` is not claimed by this
        rewrite — the driver leaves it alone even when
        ``has_fma_bf16x2`` is False."""
        b = Builder("m")
        b.begin_function("fn")
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        b.fma(a, c, a)  # scalar f32 FMA — not bf16x2
        b.end_function()
        legalize(b.module, _NO_BF16X2)
        # No NotImplementedError raised: scalar fma stays.
        ops = b.module.functions[0].body.ops
        assert any(isinstance(o, ArithOp) and o.attrs.get("kind") == "fma" for o in ops)


# -----------------------------------------------------------------------------
# AtomicRmwOp vector-atomic paths
# -----------------------------------------------------------------------------


class TestVectorAtomicRmw:
    def _build_atomic_bf16x2(self):
        from quark.ir import BufferType, GlobalTensor

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.BF16))
        g = GlobalTensor(
            dtype=DType.BF16,
            shape=(128,),
            stride=(1,),
            name="X",
            param=b.function.params[-1],
        )
        idx = b.const(DType.U32, 0)
        packed = b.const(DType.B32, 0)
        b.atomic_rmw(g, "add", packed, idx, atomic_type="bf16x2")
        b.end_function()
        return b.module

    def test_keeps_bf16x2_atomic_when_hw_supports(self):
        m = self._build_atomic_bf16x2()
        legalize(m, _CUDA_SM90)
        atomics = [o for o in m.functions[0].body.ops if isinstance(o, AtomicRmwOp)]
        assert atomics and atomics[0].attrs.get("atomic_type") == "bf16x2"

    def test_expands_bf16x2_atomic_without_caps(self):
        """Real expansion: ``atomic_rmw(..., atomic_type="bf16x2")``
        → ``split_b32`` + 2 ``bitcast`` (B16→BF16) + ``const(1)`` +
        ``arith(add, col, one)`` + 2 scalar ``atomic_rmw`` (no
        ``atomic_type``) on adjacent columns. The result Value of the
        original vector atomic was unused (red.add has no writeback),
        so this expansion doesn't preserve it — orphan Value GC'd."""
        from quark.ir import DType
        from quark.ir.op import (
            ArithOp,
            AtomicRmwOp,
            BitcastOp,
            ConstOp,
            SplitB32Op,
        )

        m = self._build_atomic_bf16x2()
        legalize(m, _NO_ATOMIC_BF16X2)
        ops = m.functions[0].body.ops

        splits = [o for o in ops if isinstance(o, SplitB32Op)]
        bitcasts = [
            o for o in ops if isinstance(o, BitcastOp) and o.attrs["dst_dtype"] is DType.BF16
        ]
        consts = [o for o in ops if isinstance(o, ConstOp) and o.attrs.get("value") == 1]
        adds = [o for o in ops if isinstance(o, ArithOp) and o.attrs.get("kind") == "add"]
        scalars = [o for o in ops if isinstance(o, AtomicRmwOp)]
        # No vector atomic should remain — the rewrite claims any op
        # with ``atomic_type in {"bf16x2","f16x2"}``, so the 2 scalar
        # atomics below are the only AtomicRmwOps left.
        vector_atomics = [o for o in scalars if o.attrs.get("atomic_type") in ("bf16x2", "f16x2")]
        assert vector_atomics == []
        assert len(splits) == 1
        assert len(bitcasts) == 2
        assert len(consts) == 1
        assert len(adds) == 1
        assert len(scalars) == 2

    def test_expands_f16x2_atomic_without_caps(self):
        """The rewrite claims both ``bf16x2`` and ``f16x2`` — this
        pins the f16x2 path since bf16x2 already has full golden
        coverage above. F16 bitcasts should replace BF16 bitcasts in
        the expansion; otherwise the chain is identical."""
        from quark.ir import BufferType, DType, GlobalTensor
        from quark.ir.op import AtomicRmwOp, BitcastOp

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.F16))
        g = GlobalTensor(
            dtype=DType.F16,
            shape=(128,),
            stride=(1,),
            name="X",
            param=b.function.params[-1],
        )
        idx = b.const(DType.U32, 0)
        packed = b.const(DType.B32, 0)
        b.atomic_rmw(g, "add", packed, idx, atomic_type="f16x2")
        b.end_function()

        # Caps lacking F16 vector atomic — triggers the expansion.
        no_f16x2 = _Caps(
            has_fma_bf16x2=True,
            atomic_add_vector=frozenset(),  # no (F16, 2)
            supports_async_copy=True,
        )
        legalize(b.module, no_f16x2)
        ops = b.module.functions[0].body.ops

        bitcasts = [o for o in ops if isinstance(o, BitcastOp)]
        # Two bitcasts, both targeting F16 (not BF16).
        assert len(bitcasts) == 2
        assert all(bc.attrs["dst_dtype"] is DType.F16 for bc in bitcasts)
        scalars = [o for o in ops if isinstance(o, AtomicRmwOp)]
        # No vector-typed atomics survive.
        vector_atomics = [o for o in scalars if o.attrs.get("atomic_type") in ("bf16x2", "f16x2")]
        assert vector_atomics == []
        # Two scalar atomics on adjacent columns.
        assert len(scalars) == 2

    def test_expands_bf16x2_atomic_with_multi_dim_indices(self):
        """The expansion preserves leading tensor indices and
        increments only the last one (the column axis). 2D tensors
        — the typical shape for GEMM output scatter — must end up
        with ``(row, col)`` and ``(row, col+1)`` respectively."""
        from quark.ir import BufferType, GlobalTensor
        from quark.ir.op import AtomicRmwOp

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.BF16))
        # 2D tensor: shape (M, N) with row-major stride.
        g = GlobalTensor(
            dtype=DType.BF16,
            shape=(64, 128),
            stride=(128, 1),
            name="X",
            param=b.function.params[-1],
        )
        row = b.const(DType.U32, 5)
        col = b.const(DType.U32, 10)
        packed = b.const(DType.B32, 0)
        b.atomic_rmw(g, "add", packed, row, col, atomic_type="bf16x2")
        b.end_function()

        legalize(b.module, _NO_ATOMIC_BF16X2)
        ops = b.module.functions[0].body.ops
        scalars = [o for o in ops if isinstance(o, AtomicRmwOp)]
        assert len(scalars) == 2
        # Each scalar atomic has (value, *indices) as operands.
        # First atomic: (lo_bf, row, col).
        # Second atomic: (hi_bf, row, col_plus_1).
        # Row is the shared value; col differs.
        first_row = scalars[0].operands[1]
        second_row = scalars[1].operands[1]
        first_col = scalars[0].operands[2]
        second_col = scalars[1].operands[2]
        assert first_row is row, (
            f"first atomic's row should be the original row Value, got {first_row}"
        )
        assert second_row is row, (
            f"second atomic's row should be the same row Value, got {second_row}"
        )
        assert first_col is col, (
            f"first atomic's col should be the original col Value, got {first_col}"
        )
        assert second_col is not col, (
            "second atomic's col must be a fresh Value (col+1), not the original col"
        )

    def test_scalar_atomic_unaffected(self):
        """Scalar AtomicRmwOp (no ``atomic_type``) passes through
        regardless of the vector-atomic caps."""
        from quark.ir import BufferType, GlobalTensor

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.F32))
        g = GlobalTensor(
            dtype=DType.F32,
            shape=(128,),
            stride=(1,),
            name="X",
            param=b.function.params[-1],
        )
        idx = b.const(DType.U32, 0)
        v = b.const(DType.F32, 1.0)
        b.atomic_rmw(g, "add", v, idx)  # no atomic_type kwarg
        b.end_function()
        # Empty atomic_add_vector caps — would trip the vec rewrite
        # if this op were claimed. Scalar atomics aren't.
        legalize(b.module, _NO_ATOMIC_BF16X2)
        atomics = [o for o in b.module.functions[0].body.ops if isinstance(o, AtomicRmwOp)]
        assert atomics and atomics[0].attrs.get("atomic_type") is None


# -----------------------------------------------------------------------------
# AsyncCopyOp + Commit / Wait
# -----------------------------------------------------------------------------


class TestAsyncCopy:
    def _build_async_copy(self):
        from quark.ir import BufferType, GlobalTensor

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.F32))
        g = GlobalTensor(
            dtype=DType.F32,
            shape=(128,),
            stride=(1,),
            name="X",
            param=b.function.params[-1],
        )
        smem = b.smem_alloc("S", DType.F32, (128,), pad=0)
        idx = b.const(DType.U32, 0)
        b.async_copy(dst=smem, src=g, dst_idx=(idx,), src_idx=(idx,), count=16)
        b.async_commit()
        b.async_wait(0)
        b.end_function()
        return b.module

    def test_keeps_async_copy_when_supported(self):
        m = self._build_async_copy()
        legalize(m, _CUDA_SM90)
        kinds = {type(o).__name__ for o in m.functions[0].body.ops}
        assert "AsyncCopyOp" in kinds

    def test_expansion_replaces_async_copy_with_vec_load_store(self):
        """Real expansion: ``async_copy`` → ``vec_load`` + ``vec_store``;
        ``async_commit`` / ``async_wait`` strip to empty."""
        from quark.ir.op import (
            AsyncCopyCommitOp,
            AsyncCopyOp,
            AsyncCopyWaitOp,
            VecLoadOp,
            VecStoreOp,
        )

        m = self._build_async_copy()
        legalize(m, _NO_ASYNC)
        ops = m.functions[0].body.ops

        assert [o for o in ops if isinstance(o, AsyncCopyOp)] == []
        assert [o for o in ops if isinstance(o, AsyncCopyCommitOp)] == []
        assert [o for o in ops if isinstance(o, AsyncCopyWaitOp)] == []
        vec_loads = [o for o in ops if isinstance(o, VecLoadOp)]
        vec_stores = [o for o in ops if isinstance(o, VecStoreOp)]
        assert len(vec_loads) == 1
        assert len(vec_stores) == 1
        # width = count (16 bytes) / src.dtype.bytes (4 for F32) = 4.
        assert vec_loads[0].attrs["width"] == 4

    def test_expansion_forwards_pred_to_both_vec_load_and_vec_store(self):
        """Predicated ``async_copy(..., pred=P)`` must wire the
        predicate to both the VecLoad and the VecStore — otherwise
        one side executes unconditionally and the other doesn't,
        producing partial stores on lanes where the predicate is
        false. Covers the ``pred`` branch of ``_expand_async_copy``."""
        from quark.ir import BufferType, GlobalTensor
        from quark.ir.op import VecLoadOp, VecStoreOp

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.F32))
        g = GlobalTensor(
            dtype=DType.F32,
            shape=(128,),
            stride=(1,),
            name="X",
            param=b.function.params[-1],
        )
        smem = b.smem_alloc("S", DType.F32, (128,), pad=0)
        idx = b.const(DType.U32, 0)
        # Build a PRED-typed Value to pass as pred.
        zero_f = b.const(DType.F32, 0.0)
        one_f = b.const(DType.F32, 1.0)
        p = b.cmp("lt", zero_f, one_f)  # always-true PRED
        b.async_copy(dst=smem, src=g, dst_idx=(idx,), src_idx=(idx,), count=16, pred=p)
        b.async_commit()
        b.async_wait(0)
        b.end_function()

        legalize(b.module, _NO_ASYNC)
        ops = b.module.functions[0].body.ops
        vec_loads = [o for o in ops if isinstance(o, VecLoadOp)]
        vec_stores = [o for o in ops if isinstance(o, VecStoreOp)]
        assert len(vec_loads) == 1
        assert len(vec_stores) == 1
        # Both ops carry the same pred Value on their attrs AND as
        # the last operand (see VecLoadOp/VecStoreOp validators —
        # they check ``attrs['pred']`` is the last operand).
        assert vec_loads[0].attrs["pred"] is p
        assert vec_stores[0].attrs["pred"] is p
        assert vec_loads[0].operands[-1] is p
        assert vec_stores[0].operands[-1] is p

    def test_expansion_rejects_sub_vector_count(self):
        """``_expand_async_copy`` raises NotImplementedError when
        ``count`` produces a ``width < 2`` — the ``VecLoadOp`` /
        ``VecStoreOp`` validators require width >= 2, so a genuinely
        scalar async_copy (``count == elem_bytes``) has no vec-op
        fallback wired up. Pins the error message so a future
        refactor can't silently fall back to an uninitialized path."""
        import pytest

        from quark.ir import BufferType, GlobalTensor

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.F32))
        g = GlobalTensor(
            dtype=DType.F32,
            shape=(128,),
            stride=(1,),
            name="X",
            param=b.function.params[-1],
        )
        smem = b.smem_alloc("S", DType.F32, (128,), pad=0)
        idx = b.const(DType.U32, 0)
        # count = 4 bytes = 1 F32 element → width=1 → triggers the guard.
        b.async_copy(dst=smem, src=g, dst_idx=(idx,), src_idx=(idx,), count=4)
        b.end_function()

        with pytest.raises(NotImplementedError, match="count=4"):
            legalize(b.module, _NO_ASYNC)


# -----------------------------------------------------------------------------
# SubgroupReduceOp — always kept in the current wiring.
# -----------------------------------------------------------------------------


class TestSubgroupReduce:
    def _build_subgroup_reduce_module(self):
        b = Builder("m")
        b.begin_function("fn")
        v = b.const(DType.F32, 1.0)
        b.subgroup_reduce("sum", v)
        b.end_function()
        return b.module

    def test_keeps_subgroup_reduce_on_native_reduce_backend(self):
        """Backends with a native single-op reduce (Metal simd_sum,
        SPIR-V OpGroupNonUniformAdd) keep the op and let the lowerer
        emit the intrinsic directly."""
        m = self._build_subgroup_reduce_module()
        legalize(m, _Caps(has_native_subgroup_reduce=True))
        assert any(isinstance(o, SubgroupReduceOp) for o in m.functions[0].body.ops)

    def test_expands_subgroup_reduce_on_butterfly_backend(self):
        """Backends without a native reduce (CUDA — only has
        ``shfl.sync.bfly``) now get the butterfly expansion from the
        legalize pass: ``log2(W)`` ShuffleOp(bfly) + matching
        ArithOp(combine) pairs. PTX still carries an inline butterfly
        as a safety net for code paths that bypass legalize (direct-
        to-lowerer tests), but the preferred path is through this
        rewrite."""
        from quark.ir.op import ArithOp, ShuffleOp

        m = self._build_subgroup_reduce_module()
        # Subgroup width 32 (default) → 5 iterations (16, 8, 4, 2, 1).
        legalize(m, _Caps(has_native_subgroup_reduce=False))
        ops = m.functions[0].body.ops

        assert [o for o in ops if isinstance(o, SubgroupReduceOp)] == []
        shuffles = [o for o in ops if isinstance(o, ShuffleOp) and o.attrs["kind"] == "bfly"]
        combines = [o for o in ops if isinstance(o, ArithOp) and o.attrs["kind"] == "add"]
        assert len(shuffles) == 5
        assert len(combines) == 5
        assert [s.attrs["param"] for s in shuffles] == [16, 8, 4, 2, 1]

    def test_subgroup_reduce_butterfly_respects_subgroup_width(self):
        """Butterfly depth scales with ``caps.subgroup_width``: W=8 →
        3 iterations (4, 2, 1)."""
        from quark.ir.op import ArithOp, ShuffleOp

        m = self._build_subgroup_reduce_module()

        class _NarrowCaps:
            has_native_subgroup_reduce = False
            subgroup_width = 8

        legalize(m, _NarrowCaps())
        ops = m.functions[0].body.ops
        shuffles = [o for o in ops if isinstance(o, ShuffleOp)]
        combines = [o for o in ops if isinstance(o, ArithOp) and o.attrs["kind"] == "add"]
        assert len(shuffles) == 3
        assert len(combines) == 3
        assert [s.attrs["param"] for s in shuffles] == [4, 2, 1]

    def test_subgroup_reduce_butterfly_rejects_non_power_of_two_width(self):
        """Butterfly expansion requires ``W`` to be a power of two
        (``W & (W-1) == 0``). Non-power-of-2 values don't produce a
        well-defined butterfly pattern and would silently emit
        wrong offsets. Pins the ValueError so a future accidental
        ``subgroup_width=24`` (or similar) is caught at legalize
        time rather than producing garbage at runtime."""
        import pytest

        m = self._build_subgroup_reduce_module()

        class _BadCaps:
            has_native_subgroup_reduce = False
            subgroup_width = 24  # not a power of two

        with pytest.raises(ValueError, match="power of two"):
            legalize(m, _BadCaps())


# -----------------------------------------------------------------------------
# Idempotency — the driver docstring claims running ``legalize`` twice with
# the same caps produces the same module as running it once. A rewrite that
# returned its own output (a loop bug the driver protects against via
# position-advance) or a pattern-match that survived its own expansion
# would violate this; the test pins the property for every currently-wired
# rewrite + caps combination.
# -----------------------------------------------------------------------------


class TestIdempotency:
    """Running ``legalize`` twice with the same caps must produce the
    same module (op count + op kinds at each position) as running it
    once. Without this property a subtle rewrite bug could silently
    churn IR on every compile."""

    def _ops_signature(self, m):
        """Stable structural summary of the module's top-level ops.
        Uses (type name, attrs-kind-if-any) per op — enough to catch
        any rewrite that accidentally re-matches its own output."""
        return [
            (
                type(o).__name__,
                o.attrs.get("kind") or o.attrs.get("op") or o.attrs.get("atomic_type"),
            )
            for o in m.functions[0].body.ops
        ]

    def test_idempotent_on_bf16x2_expansions(self):
        from quark.ir.builder import Builder

        b = Builder("m")
        b.begin_function("fn")
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        packed = b.cvt_rn_bf16x2_f32(a, c)
        b.fma_bf16x2(packed, packed, packed)
        b.end_function()

        # First pass expands both bf16x2 ops.
        legalize(b.module, _NO_BF16X2)
        sig_once = self._ops_signature(b.module)

        # Second pass should be a no-op — nothing matches anymore.
        legalize(b.module, _NO_BF16X2)
        sig_twice = self._ops_signature(b.module)

        assert sig_once == sig_twice

    def test_idempotent_on_subgroup_reduce_butterfly(self):
        from quark.ir.builder import Builder

        b = Builder("m")
        b.begin_function("fn")
        v = b.const(DType.F32, 1.0)
        b.subgroup_reduce("sum", v)
        b.end_function()

        legalize(b.module, _Caps(has_native_subgroup_reduce=False))
        sig_once = self._ops_signature(b.module)
        legalize(b.module, _Caps(has_native_subgroup_reduce=False))
        sig_twice = self._ops_signature(b.module)

        assert sig_once == sig_twice

    def test_nested_region_rewrites_fire_two_levels_deep(self):
        """Rewrites fire at arbitrary nesting depth — regression
        against a half-recursion bug where the walk might descend
        one level but stop there. Builds ``subgroup_reduce`` inside
        a ``ForLoopOp`` inside another ``ForLoopOp`` (realistic
        shape for a 2D attention or conv K-loop inside a tile-loop)
        and asserts the butterfly lands at the innermost level."""
        from quark.ir.op import ArithOp, ShuffleOp, SubgroupReduceOp

        b = Builder("m")
        b.begin_function("fn")
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step) as (_, _):
            with b.for_loop(lo, hi, step) as (_, _):
                v = b.const(DType.F32, 1.0)
                b.subgroup_reduce("sum", v)
        b.end_function()

        legalize(b.module, _Caps(has_native_subgroup_reduce=False))

        # Walk two levels down and confirm the butterfly landed.
        top_for = next(o for o in b.module.functions[0].body.ops if o.regions)
        inner_for = next(o for o in top_for.regions[0].ops if o.regions)
        innermost = inner_for.regions[0].ops
        assert [o for o in innermost if isinstance(o, SubgroupReduceOp)] == []
        shuffles = [o for o in innermost if isinstance(o, ShuffleOp)]
        combines = [o for o in innermost if isinstance(o, ArithOp) and o.attrs["kind"] == "add"]
        assert len(shuffles) == 5  # W=32 default → 5 butterfly iterations
        assert len(combines) == 5

    def test_nested_region_rewrites_fire(self):
        """A rewrite registered for a particular op type fires inside
        nested ``ForLoopOp`` bodies too — not just top-level function
        ops. Without this the SPIR-V backend (no PTX inline-butterfly
        safety net) would silently miscompile kernels that emit a
        ``SubgroupReduceOp`` inside an online-softmax K-loop."""
        from quark.ir.op import ArithOp, ShuffleOp, SubgroupReduceOp

        b = Builder("m")
        b.begin_function("fn")
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step) as (iv, _):
            # Emit a subgroup_reduce inside the loop body.
            v = b.const(DType.F32, 1.0)
            b.subgroup_reduce("sum", v)
        b.end_function()

        # Before: SubgroupReduceOp lives inside the ForLoopOp's body.
        top_ops = b.module.functions[0].body.ops
        for_loops = [o for o in top_ops if o.regions]
        assert len(for_loops) == 1
        inner = for_loops[0].regions[0].ops
        assert any(isinstance(o, SubgroupReduceOp) for o in inner)

        legalize(b.module, _Caps(has_native_subgroup_reduce=False))

        # After: the inner SubgroupReduceOp is gone, replaced by
        # ShuffleOp/ArithOp pairs *inside the loop body*.
        inner_after = for_loops[0].regions[0].ops
        assert [o for o in inner_after if isinstance(o, SubgroupReduceOp)] == []
        shuffles = [o for o in inner_after if isinstance(o, ShuffleOp)]
        combines = [o for o in inner_after if isinstance(o, ArithOp) and o.attrs["kind"] == "add"]
        assert len(shuffles) == 5  # W=32 default → 5 butterfly iterations
        assert len(combines) == 5

    def test_idempotent_with_nested_region_rewrites(self):
        """Idempotency holds even when rewrites fire inside a
        ForLoopOp body. Regression for the interaction between
        nested-region walks (8197cff) and the per-function shared
        ID counter (0578d32): the second pass must allocate fresh
        IDs starting past the first pass's nested-region allocations,
        not collide with them."""
        b = Builder("m")
        b.begin_function("fn")
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step) as (_, _):
            v = b.const(DType.F32, 1.0)
            b.subgroup_reduce("sum", v)
        b.end_function()

        legalize(b.module, _Caps(has_native_subgroup_reduce=False))
        first_ids = {
            v.id
            for region in (op.regions for op in b.module.functions[0].body.ops)
            for r in region
            for op in r.ops
            for v in op.results
        }

        legalize(b.module, _Caps(has_native_subgroup_reduce=False))
        second_ids = {
            v.id
            for region in (op.regions for op in b.module.functions[0].body.ops)
            for r in region
            for op in r.ops
            for v in op.results
        }

        # Second pass must not have added any new Values inside the
        # loop body (no new rewrites claim the expanded ops), and no
        # existing Value.id was overwritten.
        assert first_ids == second_ids

    def test_idempotent_on_keep_paths(self):
        """Keep-path rewrites (native backend) run twice without
        producing extra ops."""
        from quark.ir.builder import Builder

        b = Builder("m")
        b.begin_function("fn")
        v = b.const(DType.F32, 1.0)
        b.subgroup_reduce("sum", v)
        b.end_function()

        legalize(b.module, _CUDA_SM90)
        sig_once = self._ops_signature(b.module)
        legalize(b.module, _CUDA_SM90)
        sig_twice = self._ops_signature(b.module)

        # Metal-style caps preserve SubgroupReduceOp; two legalize
        # passes leave the module unchanged.
        metal_caps = _Caps(has_native_subgroup_reduce=True)
        b2 = Builder("m2")
        b2.begin_function("fn")
        v2 = b2.const(DType.F32, 1.0)
        b2.subgroup_reduce("sum", v2)
        b2.end_function()
        legalize(b2.module, metal_caps)
        metal_sig_once = self._ops_signature(b2.module)
        legalize(b2.module, metal_caps)
        metal_sig_twice = self._ops_signature(b2.module)

        assert sig_once == sig_twice
        assert metal_sig_once == metal_sig_twice
