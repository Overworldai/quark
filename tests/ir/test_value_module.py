"""Tests for Value, ValueAllocator, Function, Module, and parameters."""

import pytest

from popcorn.ir import (
    BufferType,
    DType,
    Function,
    Module,
    Param,
    ScalarType,
    Value,
    ValueAllocator,
    ValueShape,
)


class TestValueAllocator:
    def test_monotonic_ids(self):
        alloc = ValueAllocator()
        v0 = alloc.fresh(ValueShape(DType.F32))
        v1 = alloc.fresh(ValueShape(DType.U32))
        v2 = alloc.fresh(ValueShape(DType.BF16))
        assert (v0.id, v1.id, v2.id) == (0, 1, 2)
        assert alloc.peek() == 3

    def test_value_has_shape_and_dtype(self):
        alloc = ValueAllocator()
        v = alloc.fresh(ValueShape(DType.F32, 4))
        assert v.dtype is DType.F32
        assert v.width == 4

    def test_identity_hash_not_by_fields(self):
        # Two Values with the same id/shape must still hash distinctly —
        # hashing is by identity because `id` is only unique within one
        # Function, and textgen may allocate temporary Value wrappers.
        alloc = ValueAllocator()
        v0 = alloc.fresh(ValueShape(DType.F32))
        v0_copy = Value(id=v0.id, shape=v0.shape)
        assert v0 is not v0_copy
        assert v0 != v0_copy
        assert hash(v0) != hash(v0_copy)


class TestFunctionAndParams:
    def test_add_scalar_param(self):
        fn = Function(name="f")
        v = fn.add_param("n", ScalarType(DType.U32))
        assert v.dtype is DType.U32
        assert len(fn.params) == 1
        assert fn.params[0].name == "n"

    def test_add_buffer_param_has_u64_value(self):
        fn = Function(name="f")
        v = fn.add_param("X", BufferType(DType.BF16))
        assert v.dtype is DType.U64, "buffer params expose a u64 base pointer"
        assert fn.params[0].type == BufferType(DType.BF16)

    def test_param_ids_are_fresh(self):
        fn = Function(name="f")
        a = fn.add_param("a", ScalarType(DType.U32))
        b = fn.add_param("b", ScalarType(DType.U32))
        assert a.id != b.id

    def test_add_invalid_param_type_rejected(self):
        with pytest.raises(TypeError):
            Param(name="bad", type="not-a-type")  # type: ignore[arg-type]


class TestModule:
    def test_add_function(self):
        m = Module(name="m")
        fn = Function(name="f")
        m.add_function(fn)
        assert m.get_function("f") is fn
        assert m.get_function("missing") is None

    def test_shape_registry_dedup(self):
        from popcorn.ir import MmaShape

        m = Module()
        s = MmaShape(
            name="s",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
        )
        m.register_shape(s)
        with pytest.raises(ValueError, match="already registered"):
            m.register_shape(s)
