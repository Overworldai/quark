"""MSL lowering tests for FragReduceOp on accumulator fragments.

The MSL lowering for axis='row' uses Apple's 8×8 simdgroup_matrix
layout: thread_elements()[0] and [1] are in the same row of a tile,
and the 4 lanes sharing a row differ in bits 0 and 3 of the lane
index. So the lowering emits:

  * Local fold of the 2 thread_elements slots (max/min/+/*)
  * simd_shuffle_xor at distance 1 (bit 0) + fold
  * simd_shuffle_xor at distance 8 (bit 3) + fold

After both shuffles, every lane in the row holds the full reduction.
"""

from __future__ import annotations

from quark.ir import Builder, DType, MmaShape, validate_module
from quark.lower.msl import MslLowerer
from tests.lower.msl.conftest import METAL_CAPS_FAKE

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
    msl="bfloat16_t:2:1:2",
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


def _lower(b: Builder) -> str:
    b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module).source


def test_msl_frag_reduce_emits_thread_elements_and_shuffles():
    """Row reduction emits thread_elements() reads, local fold, plus
    simd_shuffle_xor at distances 1 and 8 per row class."""
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    msl = _lower(b)
    assert msl.count("thread_elements()") >= 2
    assert msl.count("simd_shuffle_xor") >= 4, (
        f"expected ≥4 shuffles (2 classes × 2 dists), got {msl.count('simd_shuffle_xor')}"
    )
    assert "1u" in msl  # xor distance 1
    assert "8u" in msl  # xor distance 8


def test_msl_frag_reduce_no_smem_roundtrip():
    """The whole reduction stays in register-space — no simdgroup_store
    or threadgroup_barrier should appear in the emitted MSL from the
    reduce itself."""
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    msl = _lower(b)
    assert "simdgroup_store" not in msl, f"FragReduceOp should not emit simdgroup_store\n{msl}"


def test_msl_frag_reduce_max_uses_metal_max():
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    msl = _lower(b)
    assert "metal::max" in msl


def test_msl_frag_reduce_add_uses_plus():
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="add", axis="row", cd_offsets=_CD_OFFS)
    msl = _lower(b)
    # Local fold of thread_elements[0] + [1] and per-shuffle folds.
    # Count standalone " + " appearing inside the reduction body.
    assert msl.count(" + ") >= 6  # 2 classes × 3 folds (local + 2 shfl)


def test_msl_frag_reduce_validates():
    b = Builder("test")
    d = _live_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)
    b.end_function()
    validate_module(b.module)
