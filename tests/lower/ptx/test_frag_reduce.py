"""PTX lowering tests for FragReduceOp.

FragReduceOp produces one scalar per row (or col) equivalence class,
with each scalar broadcast across every lane that shares the class
via a butterfly shuffle. On PTX m16n8 the butterfly is [1, 2] over
tig (the 2 low bits of lane within a group of 4).
"""

from __future__ import annotations

import re

from popcorn.ir import (
    Builder,
    DType,
    MmaShape,
    validate_module,
)
from popcorn.lower.ptx import PtxLowerer

_M16N8K16_BF16 = MmaShape(
    name="m16n8k16_bf16",
    m=16,
    n=8,
    k=16,
    a_dtype=DType.BF16,
    b_dtype=DType.BF16,
    acc_dtype=DType.F32,
    a_regs=4,
    b_regs=2,
    c_regs=4,
    ptx="m16n8k16.row.col.f32.bf16.bf16.f32",
)

_A_OFFS = ((0, 0), (8, 0), (0, 8), (8, 8))
_B_OFFS = ((0, 0), (0, 8))
_CD_OFFS = ((0, 0), (0, 1), (8, 0), (8, 1))


def _live_acc(b: Builder):
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    return b.mma("m16n8k16_bf16", a, bf, cf)


def test_frag_reduce_returns_n_row_classes_scalars():
    """m16n8 with cd_offsets giving 2 row classes → 2 scalar results."""
    b = Builder("test")
    d = _live_acc(b)
    results = b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    assert len(results) == 2
    for r in results:
        assert r.width == 1
        assert r.dtype is DType.F32


def test_frag_reduce_emits_local_fold_plus_butterfly():
    """Lowering should emit the per-class local fold of c_regs followed
    by shfl.sync.bfly.b32 at distances 1 and 2."""
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # 2 shuffle distances × 2 row classes = 4 shuffle instructions.
    shuffles = re.findall(r"shfl\.sync\.bfly\.b32 %\w+, %\w+, (\d+),", ptx)
    dists = [int(d) for d in shuffles]
    assert dists.count(1) >= 2 and dists.count(2) >= 2, (
        f"expected XOR distances [1, 2] × 2 row classes, got {dists}\n{ptx}"
    )


def test_frag_reduce_max_kind_uses_max_instr():
    """max reduction emits max.f32, add uses add.f32."""
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # Each row class: 1 local fold + 2 butterfly folds = 3 max.f32 ops.
    # 2 classes → 6 total max.f32.
    assert ptx.count("max.f32") >= 6, f"expected ≥6 max.f32, got\n{ptx}"


def test_frag_reduce_add_kind_uses_add_instr():
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="add", axis="row", cd_offsets=_CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # The only add.f32 ops in this module come from the reduction — the
    # load/mma path doesn't emit any. 2 classes × 3 folds = 6.
    assert ptx.count("add.f32") == 6, f"expected 6 add.f32, got\n{ptx}"


def test_frag_reduce_validates():
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    b.end_function()
    validate_module(b.module)


def test_frag_reduce_rejects_unknown_kind():
    import pytest

    b = Builder("test")
    d = _live_acc(b)
    with pytest.raises(ValueError, match="kind"):
        b.frag_reduce("m16n8k16_bf16", d, kind="avg", axis="row", cd_offsets=_CD_OFFS)
