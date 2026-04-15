"""Tests for the structural validator.

We drive the validator with hand-built IR that violates specific
invariants, to exercise each error path. The happy path is covered
implicitly by the builder tests (which call validate_module).
"""

import pytest

from popcorn.ir import (
    ArithOp,
    Builder,
    DType,
    MmaOp,
    SharedRegion,
    ValidationError,
    ValueShape,
    validate_module,
)


def test_builder_happy_path_passes():
    b = Builder("m")
    b.begin_function("f")
    x = b.const(DType.F32, 1.0)
    y = b.const(DType.F32, 2.0)
    b.add(x, y)
    b.end_function()
    validate_module(b.module)  # must not raise


def test_ssa_out_of_scope_detected():
    """Create a ForLoopOp body that uses a Value from a sibling's
    for-loop body — the validator must catch the leak."""
    b = Builder("m")
    b.begin_function("f")
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 4)
    step = b.const(DType.U32, 1)
    leaked = {}
    with b.for_loop(lo, hi, step, iv_name="k") as (k, _):
        # Inside the body, create a Value that should only be visible
        # inside this region.
        inner = b.const(DType.F32, 7.0)
        leaked["v"] = inner
        b.yield_()
    # Now after the loop, try to use `leaked['v']` as an operand to
    # an op — the validator should reject this because `inner` was
    # defined in a region that's already exited.
    other = b.const(DType.F32, 1.0)
    # Hand-construct an ArithOp that references the leaked Value. We
    # bypass builder.add so we don't get stopped at op construction
    # time (the Builder doesn't check dominance, only shapes).
    from popcorn.ir.value import ValueShape as _Shape

    out = b.function.fresh_value(_Shape(DType.F32))
    bad_op = ArithOp(
        results=(out,),
        operands=(other, leaked["v"]),
        attrs={"kind": "add"},
    )
    b.current_region.append(bad_op)
    b.end_function()
    with pytest.raises(ValidationError, match="not in scope"):
        validate_module(b.module)


def test_unregistered_mma_shape_detected():
    """Hand-build an MmaOp with a shape_id the module doesn't know."""
    b = Builder("m")
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    b.register_shape(
        __import__("popcorn.ir", fromlist=["MmaShape"]).MmaShape(
            name="known",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
        )
    )
    fa = b.load_matrix(A, "known", which="a", row=0, col=0)
    fb = b.load_matrix(A, "known", which="b", row=0, col=0)
    fc = b.load_matrix(A, "known", which="c", row=0, col=0)
    # Forge an MmaOp referencing an unregistered shape.
    out = b.function.fresh_value(ValueShape(DType.B32))
    bad = MmaOp(
        results=(out,),
        operands=(fa, fb, fc),
        attrs={"shape_id": "unregistered"},
    )
    b.current_region.append(bad)
    b.end_function()
    with pytest.raises(ValidationError, match="shape_id 'unregistered'"):
        validate_module(b.module)


def test_shared_tensor_backing_not_in_function_rejected():
    """A SharedRegion that references an alloc Value that isn't a real
    SmemAllocOp of this function must be rejected. The validator can
    catch this either as an SSA-dominance failure (the Value was never
    produced) or as a smem_backings failure — both indicate the same
    bug, so we accept either error message."""
    b = Builder("m")
    b.begin_function("f")
    # Produce *some* real U64 Value (a ConstOp) that is in SSA scope,
    # but is NOT an SmemAllocOp — so dominance passes and the
    # smem_backings check is what fires.
    not_a_real_alloc = b.const(DType.U64, 0, name="not_alloc")
    rogue = SharedRegion(
        dtype=DType.F32,
        shape=(8, 8),
        stride=(8, 1),
        name="rogue",
        alloc=not_a_real_alloc,
    )
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.load(rogue, row, col)
    b.end_function()
    with pytest.raises(ValidationError, match="SmemAllocOp"):
        validate_module(b.module)


def test_yield_must_be_last_op_in_region():
    """If someone directly appends a YieldOp not at the end of a region,
    validation flags it."""
    from popcorn.ir.op import YieldOp

    b = Builder("m")
    b.begin_function("f")
    # Stuff a YieldOp followed by a const inside the function body
    # by hand.
    term = YieldOp()
    extra = b.function.fresh_value(ValueShape(DType.F32))
    from popcorn.ir.op import ConstOp

    const_op = ConstOp(results=(extra,), attrs={"dtype": DType.F32, "value": 0.0})
    b.current_region.append(term)
    b.current_region.append(const_op)
    b.end_function()
    with pytest.raises(ValidationError, match="YieldOp must be the terminator"):
        validate_module(b.module)
