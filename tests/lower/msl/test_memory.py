"""Tests for memory (load/store/smem) MSL lowering."""

from quark.ir import BufferType, DType, GlobalTensor, ParamAttrs
from tests.lower.msl.conftest import lower, lower_full


class TestScalarLoad:
    def _make_g(self, b, shape: tuple[int, ...] = (16,)):
        b.param("X", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
        stride = (1,) if len(shape) == 1 else (shape[1], 1)
        return GlobalTensor(
            dtype=DType.F32,
            shape=shape,
            stride=stride,
            name="X",
            param=b.function.params[-1],
        )

    def test_global_load(self, fresh_builder):
        b = fresh_builder
        g = self._make_g(b)
        idx = b.const(DType.U32, 0)
        b.load(g, idx)
        out = lower(b)
        assert "X[" in out
        assert "float _pc_f32_0 =" in out

    def test_global_store(self, fresh_builder):
        b = fresh_builder
        b.param("Y", BufferType(DType.F32))
        g = GlobalTensor(
            dtype=DType.F32,
            shape=(16,),
            stride=(1,),
            name="Y",
            param=b.function.params[-1],
        )
        idx = b.const(DType.U32, 0)
        val = b.const(DType.F32, 1.0)
        b.store(g, val, idx)
        out = lower(b)
        assert "Y[" in out


class TestSmemAlloc:
    def test_smem_declared(self, fresh_builder):
        b = fresh_builder
        b.smem_alloc("A", DType.F32, (64, 64))
        out = lower(b)
        assert "threadgroup float" in out
        assert "[" in out

    def test_smem_bytes_reported(self, fresh_builder):
        b = fresh_builder
        b.smem_alloc("A", DType.F32, (64, 64))
        result = lower_full(b)
        assert result.smem_bytes == 64 * 64 * 4

    def test_smem_load_store(self, fresh_builder):
        b = fresh_builder
        A = b.smem_alloc("A", DType.F32, (8, 16))
        row = b.const(DType.U32, 2)
        col = b.const(DType.U32, 3)
        v = b.load(A, row, col)
        b.store(A, v, row, col)
        out = lower(b)
        # Should reference the smem variable name.
        assert "smem_" in out


class TestParamClassification:
    def test_readonly_is_input(self, fresh_builder):
        b = fresh_builder
        b.param("X", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
        b.param("Y", BufferType(DType.F32))
        result = lower_full(b)
        assert "X" in result.input_names
        assert "Y" in result.output_names
