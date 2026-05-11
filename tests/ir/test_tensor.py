"""Tests for GlobalTensor/SharedRegion views, FragTensor invariants,
and stride-from-shape helpers."""

import pytest

from quark.ir import (
    BufferType,
    Builder,
    DType,
    FragTensor,
    GlobalTensor,
    SharedRegion,
    Tensor,
)
from quark.ir.builder import _rowmajor_stride


class TestStrideHelper:
    def test_2d_no_pad(self):
        assert _rowmajor_stride((8, 16)) == (16, 1)

    def test_2d_with_pad(self):
        assert _rowmajor_stride((8, 16), pad=4) == (20, 1)

    def test_rank_3(self):
        # shape (A, B, C) row-major: strides (B*C, C, 1)
        assert _rowmajor_stride((4, 8, 16)) == (8 * 16, 16, 1)

    def test_rank_1_and_0(self):
        assert _rowmajor_stride((5,)) == (1,)
        assert _rowmajor_stride(()) == ()


class TestSharedRegionFromBuilder:
    def test_smem_alloc_returns_tensor_with_correct_stride(self):
        b = Builder()
        b.begin_function("f")
        t = b.smem_alloc("A", DType.BF16, (8, 16), pad=4)
        b.end_function()
        assert isinstance(t, SharedRegion)
        assert t.dtype is DType.BF16
        assert t.shape == (8, 16)
        assert t.stride == (20, 1)
        assert t.pad == 4
        assert t.name == "A"

    def test_smem_view_accumulates_static_offset(self):
        b = Builder()
        b.begin_function("f")
        t = b.smem_alloc("A", DType.BF16, (32, 16))
        b.end_function()
        v1 = t.view(static_offset_add=100)
        v2 = v1.view(static_offset_add=50)
        assert v1.static_offset == 100
        assert v2.static_offset == 150
        assert v2.alloc is t.alloc  # same backing

    def test_smem_dyn_offset_is_not_replaceable_without_folding(self):
        b = Builder()
        b.begin_function("f")
        t = b.smem_alloc("A", DType.BF16, (32, 16))
        k = b.const(DType.U32, 4)
        j = b.const(DType.U32, 7)
        v1 = t.view(dyn_offset=k)
        # Passing a second dyn_offset without folding must error.
        with pytest.raises(ValueError, match="already has a dyn_offset"):
            v1.view(dyn_offset=j)
        b.end_function()


class TestGlobalTensorView:
    def _make(self) -> tuple[Builder, GlobalTensor]:
        b = Builder()
        fn = b.begin_function("f")
        b.param("X", BufferType(DType.BF16))

        param = fn.params[-1]
        g = GlobalTensor(
            dtype=DType.BF16,
            shape=(64, 128),
            stride=(128, 1),
            name="X",
            param=param,
        )
        return b, g

    def test_static_row_offset_accumulates(self):
        b, g = self._make()
        v1 = g.view(row=8)
        v2 = v1.view(row=16)
        assert v1.static_row_offset == 8
        assert v2.static_row_offset == 24
        b.end_function()

    def test_dyn_row_and_col_once(self):
        b, g = self._make()
        r = b.const(DType.U32, 3)
        c = b.const(DType.U32, 4)
        v = g.view(row=r, col=c)
        assert v.dyn_row_offset is r
        assert v.dyn_col_offset is c
        b.end_function()

    def test_dyn_row_applied_twice_errors(self):
        b, g = self._make()
        r = b.const(DType.U32, 3)
        r2 = b.const(DType.U32, 5)
        v = g.view(row=r)
        with pytest.raises(ValueError, match="already has a dyn_row_offset"):
            v.view(row=r2)
        b.end_function()

    def test_bad_row_type(self):
        b, g = self._make()
        with pytest.raises(TypeError):
            g.view(row="bad")  # type: ignore[arg-type]
        b.end_function()


class TestFragTensor:
    def test_which_must_be_a_b_c_d(self):
        with pytest.raises(ValueError):
            FragTensor(
                dtype=DType.BF16,
                shape=(16, 16),
                stride=(16, 1),
                shape_id="m16n8k16_bf16",
                which="e",
            )

    def test_happy_path(self):
        f = FragTensor(
            dtype=DType.BF16,
            shape=(16, 16),
            stride=(16, 1),
            shape_id="m16n8k16_bf16",
            which="a",
        )
        assert f.shape_id == "m16n8k16_bf16"
        assert f.which == "a"
        assert f.rank == 2


class TestBaseTensor:
    def test_rank_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same rank"):
            Tensor(dtype=DType.F32, shape=(4, 8), stride=(8,))  # type: ignore[abstract]
