"""Tests for popcorn.launcher.param_spec — pure-python, no CUDA needed."""

import struct

import pytest

from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    ParamAttrs,
    ScalarType,
)
from popcorn.launcher import ParamSpec, ProgramFootprint, ScalarSpec


def _function_with_params(specs):
    """Build a small Function with the requested params and return it."""
    b = Builder("m")
    fn = b.begin_function("k")
    for name, type_, attrs in specs:
        b.param(name, type_, attrs)
    b.end_function()
    return fn


class TestParamSpecFromFunction:
    def test_buffers_and_scalars_split_by_type(self):
        fn = _function_with_params(
            [
                ("X", BufferType(DType.F32), None),
                ("N", ScalarType(DType.U32), None),
                ("Y", BufferType(DType.BF16), None),
                ("scale", ScalarType(DType.F32), None),
            ]
        )
        spec = ParamSpec.from_function(fn)
        assert [b.name for b in spec.buffers] == ["X", "Y"]
        assert [s.name for s in spec.scalars] == ["N", "scale"]
        assert spec.buffers[0].dtype is DType.F32
        assert spec.buffers[1].dtype is DType.BF16
        assert spec.scalars[0].dtype is DType.U32
        assert spec.scalars[1].dtype is DType.F32

    def test_buffer_attrs_propagate(self):
        fn = _function_with_params(
            [
                ("X", BufferType(DType.F32), ParamAttrs(readonly=True, align=16)),
            ]
        )
        spec = ParamSpec.from_function(fn)
        assert spec.buffers[0].readonly is True
        assert spec.buffers[0].align == 16

    def test_empty_function(self):
        fn = _function_with_params([])
        spec = ParamSpec.from_function(fn)
        assert spec.n_buffers() == 0
        assert spec.n_scalars() == 0


class TestPackScalars:
    def test_single_u32(self):
        spec = ParamSpec(scalars=(ScalarSpec("N", DType.U32),))
        blobs = spec.pack_scalars((42,))
        assert len(blobs) == 1
        assert blobs[0] == struct.pack("=I", 42)

    def test_multiple_returns_one_blob_per_scalar(self):
        spec = ParamSpec(
            scalars=(
                ScalarSpec("a", DType.U32),
                ScalarSpec("b", DType.S64),
                ScalarSpec("c", DType.F32),
            )
        )
        blobs = spec.pack_scalars((10, -5, 3.14))
        assert len(blobs) == 3
        assert blobs[0] == struct.pack("=I", 10)
        assert blobs[1] == struct.pack("=q", -5)
        assert blobs[2] == struct.pack("=f", 3.14)

    def test_empty(self):
        spec = ParamSpec()
        assert spec.pack_scalars(()) == []

    def test_value_count_mismatch_raises(self):
        spec = ParamSpec(
            scalars=(
                ScalarSpec("a", DType.U32),
                ScalarSpec("b", DType.U32),
            )
        )
        with pytest.raises(ValueError, match="expected 2 values"):
            spec.pack_scalars((10,))

    def test_unsupported_dtype_raises(self):
        spec = ParamSpec(scalars=(ScalarSpec("p", DType.PRED),))
        with pytest.raises(TypeError, match="unsupported dtype"):
            spec.pack_scalars((True,))


class TestProgramFootprint:
    def test_default_fields(self):
        fp = ProgramFootprint(smem_bytes=1024)
        assert fp.smem_bytes == 1024
        assert fp.instruction_count == 0
        assert fp.reg_total == 0
        assert fp.reg_counts == {}
