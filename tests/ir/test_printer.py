"""Tests for the human-readable IR printer."""

from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    MmaShape,
    print_function,
    print_module,
)


def test_empty_module_prints():
    b = Builder("empty")
    out = print_module(b.module)
    assert "module @empty" in out
    assert out.endswith("}\n")


def test_function_param_and_const():
    b = Builder("m")
    b.begin_function("f")
    b.param("X", BufferType(DType.BF16))
    b.const(DType.F32, 1.5)
    b.end_function()

    out = print_module(b.module)
    assert "func @f" in out
    assert "X: BufferType(bf16, global)" in out
    assert "const" in out
    assert "dtype=f32" in out
    assert "value=1.5" in out


def test_smem_alloc_prints_readable_tensor():
    b = Builder()
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (8, 16), pad=4)
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.load(A, row, col)
    b.end_function()
    out = print_module(b.module)
    assert "smem_alloc" in out
    assert "'A'" in out
    assert "smem<A:bf16[8, 16]>" in out


def test_mma_shape_header_printed():
    b = Builder("m")
    b.register_shape(
        MmaShape(
            name="m16n8k16_bf16",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
        )
    )
    b.begin_function("f")
    b.end_function()
    out = print_module(b.module)
    assert "mma_shape @m16n8k16_bf16" in out
    assert "m=16" in out and "k=16" in out


def test_print_function_alone():
    b = Builder()
    fn = b.begin_function("solo")
    b.const(DType.F32, 42.0)
    b.end_function()
    out = print_function(fn)
    assert "func @solo" in out
    assert "value=42.0" in out
