"""Pythonic SharedRegion / GlobalTensor API tests.

Covers the new sugar layered on top of ``bld.load`` / ``bld.store`` /
``bld.vec_load`` / ``bld.vec_store``:

  * ``region[r, c]`` → load
  * ``region[r, c] = v`` → store
  * ``region[r, c:c+W]`` → vec_load width=W
  * ``region[r, c:c+W] = v`` → vec_store
  * ``region.stage(i)`` → 3D → 2D view at stage i
  * ``region.warp_view(rows, warp_id=)`` → per-warp slice with
    ``warp_dyn_offset`` set
  * ``g_tensor.tile(row=, col=, shape=)`` → typed gmem tile

These are pure IR-level tests — no lowering needed beyond what each
underlying op already supports.
"""

from __future__ import annotations

import pytest

from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    GlobalTensor,
    Lifetime,
    LoadOp,
    StoreOp,
    VecLoadOp,
    VecStoreOp,
)


def _builder() -> Builder:
    b = Builder("t")
    b.begin_function("f")
    return b


def _gtensor(b: Builder, name: str, dtype: DType, shape: tuple[int, int]) -> GlobalTensor:
    """Tests need a GlobalTensor — declare a Param, then wrap in a
    GlobalTensor with the requested shape (row-major stride)."""
    param = b.param(name, BufferType(dtype))
    return GlobalTensor(
        dtype=dtype,
        shape=shape,
        stride=(shape[1], 1),
        name=name,
        param=param,
    )


def test_subscript_scalar_load_emits_load_op():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    v = A[3, 5]
    assert v.dtype is DType.F32
    ops = [op for op in b.current_region.ops if isinstance(op, LoadOp)]
    assert len(ops) == 1


def test_subscript_scalar_store_emits_store_op():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    val = b.const(DType.F32, 1.5)
    A[3, 5] = val
    ops = [op for op in b.current_region.ops if isinstance(op, StoreOp)]
    assert len(ops) == 1


def test_subscript_vec_load_emits_vec_load_op():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    v = A[3, 0:4]
    assert v.dtype is DType.F32
    assert v.width == 4
    ops = [op for op in b.current_region.ops if isinstance(op, VecLoadOp)]
    assert len(ops) == 1


def test_subscript_vec_store_emits_vec_store_op():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    v = A[0, 0:4]
    A[3, 0:4] = v
    ops = [op for op in b.current_region.ops if isinstance(op, VecStoreOp)]
    assert len(ops) == 1


def test_subscript_rejects_slice_step():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    with pytest.raises(IndexError, match="step must be 1"):
        _ = A[0, 0:4:2]


def test_subscript_rejects_slice_on_non_last_axis():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    with pytest.raises(IndexError, match="last axis"):
        _ = A[0:4, 0]


def test_subscript_rejects_dynamic_slice_bounds():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    n = b.const(DType.U32, 4)
    with pytest.raises(IndexError, match="statically known"):
        _ = A[0, 0:n]


def test_subscript_rejects_wrong_arity():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (16, 16))
    with pytest.raises(IndexError, match="expected 2 indices"):
        _ = A[3]


def test_warp_view_sets_dyn_offset():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (32, 16))
    warp_id = b.const(DType.U32, 1)
    warp_a = A.warp_view(rows=16, warp_id=warp_id)
    assert warp_a.shape == (16, 16)
    assert warp_a.dyn_offset is not None


def test_warp_view_no_warp_id_just_reshapes():
    b = _builder()
    A = b.smem_alloc("A", DType.F32, (32, 16))
    sub = A.warp_view(rows=16)
    assert sub.shape == (16, 16)
    assert sub.dyn_offset is None


def test_warp_view_rejects_3d_region():
    b = _builder()
    A3 = b.smem_alloc("A3", DType.F32, (2, 16, 16))
    warp_id = b.const(DType.U32, 0)
    with pytest.raises(ValueError, match="2D"):
        A3.warp_view(rows=8, warp_id=warp_id)


def test_stage_const_idx_static_offset():
    b = _builder()
    A3 = b.smem_alloc("A3", DType.F32, (3, 16, 16), pad=4)
    s0 = A3.stage(0)
    s1 = A3.stage(1)
    s2 = A3.stage(2)
    # 16 rows × (16 + 4) cols = 320 elements per stage.
    assert s0.shape == (16, 16)
    assert s1.shape == (16, 16)
    assert s2.shape == (16, 16)
    assert s0.static_offset == 0
    assert s1.static_offset == 320
    assert s2.static_offset == 640
    # Stride excludes the leading stage dim — drops to (cols+pad, 1).
    assert s0.stride == (20, 1)


def test_stage_dyn_idx_emits_dyn_offset():
    b = _builder()
    A3 = b.smem_alloc("A3", DType.F32, (3, 16, 16))
    iv = b.const(DType.U32, 0)
    sd = A3.stage(iv)
    assert sd.shape == (16, 16)
    assert sd.dyn_offset is not None


def test_stage_rejects_non_3d():
    b = _builder()
    A2 = b.smem_alloc("A2", DType.F32, (16, 16))
    with pytest.raises(ValueError, match="3D regions"):
        A2.stage(0)


def test_stage_lifetime_inheritance():
    """A staged view inherits the parent's Lifetime so the layout pass
    sees them as the same region."""
    b = _builder()
    A3 = b.smem_alloc("A3", DType.F32, (2, 8, 8), lifetime=Lifetime.kernel())
    s0 = A3.stage(0)
    assert s0.lifetime == Lifetime.kernel()


def test_global_tensor_tile_returns_view():
    b = _builder()
    g_x = _gtensor(b, "X", DType.BF16, (64, 64))
    tile = g_x.tile(row=0, col=0, shape=(16, 16))
    assert tile.shape == (16, 16)
    assert tile.static_row_offset == 0
    assert tile.static_col_offset == 0


def test_global_tensor_tile_with_offsets():
    b = _builder()
    g_x = _gtensor(b, "X", DType.BF16, (64, 64))
    tile = g_x.tile(row=16, col=32, shape=(16, 16))
    assert tile.static_row_offset == 16
    assert tile.static_col_offset == 32


def test_global_tensor_subscript_load():
    b = _builder()
    g_x = _gtensor(b, "X", DType.F32, (16, 16))
    v = g_x[3, 5]
    assert v.dtype is DType.F32


def test_copy_from_emits_loads_and_stores():
    """Unified tile-load wraps ``emit_tile_load`` — emits per-thread
    scalar loads + stores for the scalar path."""
    from popcorn.ir.op import LoadOp as _Load
    from popcorn.ir.op import StoreOp as _Store

    b = _builder()
    g_x = _gtensor(b, "X", DType.F32, (64, 64))
    A = b.smem_alloc("A", DType.F32, (16, 16))
    g_tile = g_x.tile(row=0, col=0, shape=(16, 16))
    tid = b.const(DType.U32, 0)
    A.copy_from(g_tile, tid=tid, n_threads=32, async_load=False)
    # Scalar path emits 256/32 = 8 loads and 8 stores per thread.
    loads = [op for op in b.current_region.ops if isinstance(op, _Load)]
    stores = [op for op in b.current_region.ops if isinstance(op, _Store)]
    assert len(loads) == 8
    assert len(stores) == 8


def test_copy_from_rejects_3d_region():
    b = _builder()
    g_x = _gtensor(b, "X", DType.F32, (16, 16))
    A3 = b.smem_alloc("A3", DType.F32, (2, 16, 16))
    g_tile = g_x.tile(row=0, col=0, shape=(16, 16))
    tid = b.const(DType.U32, 0)
    with pytest.raises(ValueError, match="2D"):
        A3.copy_from(g_tile, tid=tid, n_threads=32)


def test_copy_from_respects_stage_view():
    """``region.stage(i).copy_from(gmem_tile)`` is the pipeline-stage
    tile-load pattern."""
    from popcorn.ir.op import StoreOp as _Store

    b = _builder()
    g_x = _gtensor(b, "X", DType.F32, (64, 64))
    A3 = b.smem_alloc("A3", DType.F32, (2, 16, 16))
    g_tile = g_x.tile(row=0, col=0, shape=(16, 16))
    tid = b.const(DType.U32, 0)
    A3.stage(0).copy_from(g_tile, tid=tid, n_threads=32, async_load=False)
    stores = [op for op in b.current_region.ops if isinstance(op, _Store)]
    assert len(stores) == 8
