"""Compile-time OOB validation tests.

Covers the constant-index OOB check in ``validate_module``:

  * SharedRegion accesses (Load / Store / VecLoad / VecStore) with
    out-of-range constant indices → ValidationError.
  * GlobalTensor accesses — same.
  * Dynamic indices → skipped (can't prove).
  * Legal in-bounds accesses → no error.
"""

from __future__ import annotations

import pytest

from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    GlobalTensor,
    validate_module,
)
from popcorn.ir.validator import ValidationError


def _builder() -> Builder:
    b = Builder("t")
    b.begin_function("f")
    return b


def _g(b, name, dtype, shape):
    b.param(name, BufferType(dtype))
    param = b.function.params[-1]
    return GlobalTensor(
        dtype=dtype,
        shape=shape,
        stride=(shape[1], 1),
        name=name,
        param=param,
    )


# ---------------------------------------------------------------------------
# SharedRegion
# ---------------------------------------------------------------------------


def test_smem_oob_row_raises():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (4, 8))
    A[4, 0] = b.const(DType.F32, 1.0)
    b.end_function()
    with pytest.raises(ValidationError, match="axis 0 index=4 out of bounds"):
        validate_module(b.module)


def test_smem_oob_col_raises():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (4, 8))
    A[0, 8] = b.const(DType.F32, 1.0)
    b.end_function()
    with pytest.raises(ValidationError, match="axis 1 index=8 out of bounds"):
        validate_module(b.module)


def test_smem_in_bounds_ok():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (4, 8))
    A[3, 7] = b.const(DType.F32, 1.0)
    _ = A[0, 0]
    b.end_function()
    validate_module(b.module)  # no raise


def test_smem_dynamic_index_skips_check():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (4, 8))
    # Index comes from an ArithOp, not a ConstOp — validator skips.
    row = b.add(b.const(DType.U32, 100), b.const(DType.U32, 200))  # way OOB but dynamic
    A[row, b.const(DType.U32, 0)] = b.const(DType.F32, 1.0)
    b.end_function()
    # Validator can't statically prove OOB here → no raise. The user
    # gets to shoot themselves in the foot, but at least it's a
    # runtime UB rather than a silent validator miss.
    validate_module(b.module)


def test_smem_vec_store_oob_raises():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (4, 8))
    val = b.const(DType.F32, 1.0)
    vec = b.vec_build([val, val, val, val])
    # row=5 is OOB.
    b.vec_store(A, vec, b.const(DType.U32, 5), b.const(DType.U32, 0))
    b.end_function()
    with pytest.raises(ValidationError, match="axis 0 index=5 out of bounds"):
        validate_module(b.module)


# ---------------------------------------------------------------------------
# GlobalTensor
# ---------------------------------------------------------------------------


def test_gmem_oob_row_raises():
    b = _builder()
    g = _g(b, "X", DType.F32, (16, 16))
    _ = g[16, 0]
    b.end_function()
    with pytest.raises(ValidationError, match="row=16 out of bounds"):
        validate_module(b.module)


def test_gmem_oob_col_raises():
    b = _builder()
    g = _g(b, "X", DType.F32, (16, 16))
    _ = g[0, 16]
    b.end_function()
    with pytest.raises(ValidationError, match="col=16 out of bounds"):
        validate_module(b.module)


def test_gmem_in_bounds_ok():
    b = _builder()
    g = _g(b, "X", DType.F32, (16, 16))
    _ = g[15, 15]
    _ = g[0, 0]
    b.end_function()
    validate_module(b.module)


def test_gmem_dynamic_offset_skips_check():
    b = _builder()
    g = _g(b, "X", DType.F32, (16, 16))
    # View with a dyn_row_offset — can't statically resolve, so OOB
    # check skips. Even with a huge static index.
    dyn = b.const(DType.U32, 0)
    gview = g.view(row=dyn)
    _ = gview[1000, 0]  # huge, but dyn_row_offset poisoned the analysis
    b.end_function()
    validate_module(b.module)  # no raise


def test_gmem_view_static_offset_accumulates():
    b = _builder()
    g = _g(b, "X", DType.F32, (16, 16))
    # View starts at row 10; accessing g_view[6, 0] hits row 16 which
    # IS in bounds of the parent shape; g_view[7, 0] hits row 17 → OOB.
    g_view = g.view(row=10)
    _ = g_view[7, 0]  # effective row = 17
    b.end_function()
    with pytest.raises(ValidationError, match="row=17 out of bounds"):
        validate_module(b.module)
