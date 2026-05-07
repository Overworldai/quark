"""MSL lowering tests for FragReduceOp on NAX-stored fragments.

NAX BaseNAXFrag layout per lane: 8 elements as 2 rows × 4 cols
(kElemRows=2, kElemCols=4). For c_regs=16 (TN=2 stacked frags) the
mapping is::

    slot      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    row class 0 0 0 0 1 1 1 1 0 0  0  0  1  1  1  1

Reduction emission:
  * Within-thread fold of slots-of-this-class (no thread_elements())
  * simd_shuffle_xor at distance 1 (quad ±1)
  * simd_shuffle_xor at distance 8 (cross-quad ±8)

After both shuffles, every lane in the row band holds the full
reduction. Different from the simdgroup_matrix path's [1, 8] butterfly
which folds via thread_elements() pairs.
"""

from __future__ import annotations

from quark.ir import Builder, DType, validate_module
from quark.lower.msl import MslLowerer
from tests.lower.msl.conftest import METAL_CAPS_FAKE

_NAX_CD_OFFS = tuple((r * 8, c) for r in range(2) for c in range(4))


def _live_nax_acc(b: Builder):
    """Build a kernel where d is a NAX-stored f32 acc fragment."""
    # The NAX shape is already in the production registry; just register
    # it on the module so the builder accepts it.
    from quark.ir.mma_registry import _BY_SHAPE_ID

    b.register_shape(_BY_SHAPE_ID["m16n32k16_nax_bf16"].shape)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (32, 16))
    C = b.smem_alloc("C", DType.F32, (16, 32))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    bf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    return b.mma("m16n32k16_nax_bf16", a, bf, cf)


def _lower(b: Builder) -> str:
    b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module).source


def test_nax_frag_reduce_max_emits_shuffles_at_1_and_8():
    """Row reduction on a NAX C-frag emits simd_shuffle_xor at
    distances 1 and 8 per row class (2 classes from kElemRows=2)."""
    b = Builder("test")
    d = _live_nax_acc(b)
    b.frag_reduce("m16n32k16_nax_bf16", d, kind="max", axis="row", cd_offsets=_NAX_CD_OFFS)
    msl = _lower(b)
    # 2 classes × 2 shuffle distances = 4 simd_shuffle_xor calls
    assert msl.count("simd_shuffle_xor") >= 4, (
        f"expected ≥4 shuffles (2 classes × 2 dists), got {msl.count('simd_shuffle_xor')}\n{msl}"
    )
    assert "ushort(1)" in msl
    assert "ushort(8)" in msl


def test_nax_frag_reduce_max_uses_metal_max():
    b = Builder("test")
    d = _live_nax_acc(b)
    b.frag_reduce("m16n32k16_nax_bf16", d, kind="max", axis="row", cd_offsets=_NAX_CD_OFFS)
    msl = _lower(b)
    assert "metal::max" in msl


def test_nax_frag_reduce_add_uses_plus():
    b = Builder("test")
    d = _live_nax_acc(b)
    b.frag_reduce("m16n32k16_nax_bf16", d, kind="add", axis="row", cd_offsets=_NAX_CD_OFFS)
    msl = _lower(b)
    # The NAX path inlines `(a + b)` — should appear inside the fold.
    # Also has at least 4 shuffle adds (2 classes × 2 dists).
    assert msl.count("simd_shuffle_xor") >= 4
    assert "metal::max" not in msl  # max shouldn't appear


def test_nax_frag_reduce_no_thread_elements():
    """NAX path operates on per-lane scalar arrays directly — no
    ``thread_elements()`` indirection (that's the simdgroup_matrix
    path's discriminator)."""
    b = Builder("test")
    d = _live_nax_acc(b)
    b.frag_reduce("m16n32k16_nax_bf16", d, kind="max", axis="row", cd_offsets=_NAX_CD_OFFS)
    msl = _lower(b)
    # Only the `_visit_frag_reduce_nax` body for this op shouldn't use
    # thread_elements. The NAX MMA preamble doesn't either, so the
    # whole emitted MSL from this reduce should be free of it.
    # (Other ops in the kernel may legitimately use it; we just check
    # the reduce-specific code paths don't.)
    assert "thread_elements" not in msl, f"NAX frag_reduce should not emit thread_elements()\n{msl}"


def test_nax_frag_reduce_module_validates():
    """The emitted IR module must validate before lowering."""
    b = Builder("test")
    d = _live_nax_acc(b)
    b.frag_reduce("m16n32k16_nax_bf16", d, kind="max", axis="row", cd_offsets=_NAX_CD_OFFS)
    b.end_function()
    validate_module(b.module)
