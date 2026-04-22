"""Tests for the PTX RegAllocator and DType→class mappings."""

import pytest

from quark.ir import DType, Value, ValueShape
from quark.lower.ptx.regs import RegAllocator, arith_suffix, reg_class


def _val(dtype: DType, vid: int = 0) -> Value:
    return Value(id=vid, shape=ValueShape(dtype))


class TestDTypeMapping:
    @pytest.mark.parametrize(
        "dtype,cls",
        [
            (DType.F32, "f32"),
            (DType.F64, "f64"),
            (DType.F16, "b16"),
            (DType.BF16, "b16"),
            (DType.U32, "u32"),
            (DType.S32, "s32"),
            (DType.U64, "u64"),
            (DType.B32, "b32"),
            (DType.PRED, "pred"),
        ],
    )
    def test_reg_class(self, dtype: DType, cls: str):
        assert reg_class(dtype) == cls

    def test_arith_suffix_f16(self):
        # F16/BF16 arith suffix is their native name (not b16).
        assert arith_suffix(DType.F16) == "f16"
        assert arith_suffix(DType.BF16) == "bf16"
        # But the register class for them is b16.
        assert reg_class(DType.F16) == "b16"


class TestRegAllocator:
    def test_allocate_distinct_regs_per_value(self):
        a = RegAllocator()
        v0 = _val(DType.F32, 0)
        v1 = _val(DType.F32, 1)
        n0 = a.name_for(v0)
        n1 = a.name_for(v1)
        assert n0 != n1
        assert n0.startswith("%f") and n1.startswith("%f")

    def test_name_for_is_stable(self):
        a = RegAllocator()
        v = _val(DType.U32)
        assert a.name_for(v) == a.name_for(v)

    def test_prefixes_differ_per_class(self):
        a = RegAllocator()
        fv = _val(DType.F32, 10)
        iv = _val(DType.U32, 11)
        pv = _val(DType.PRED, 12)
        assert a.name_for(fv).startswith("%f")
        assert a.name_for(iv).startswith("%r")
        assert a.name_for(pv).startswith("%p")

    def test_declarations_grouped_and_sorted(self):
        a = RegAllocator()
        _ = a.name_for(_val(DType.F32, 0))
        _ = a.name_for(_val(DType.U32, 1))
        _ = a.name_for(_val(DType.F32, 2))
        decls = a.declarations()
        # Expect one decl line per class, classes in sorted order.
        classes = [line.split(" .")[1].split(" ")[0] for line in decls]
        assert classes == sorted(classes)
        # F32 decl should list both f32 registers.
        f32_line = next(line for line in decls if ".f32 " in line)
        assert f32_line.count("%f") == 2

    def test_alias_reuses_source_register(self):
        a = RegAllocator()
        src = _val(DType.F32, 5)
        tgt = _val(DType.F32, 6)
        src_name = a.name_for(src)
        a.alias(tgt, src)
        assert a.name_for(tgt) == src_name

    def test_alias_conflict_raises(self):
        a = RegAllocator()
        src = _val(DType.F32, 5)
        tgt = _val(DType.F32, 6)
        a.name_for(src)
        a.name_for(tgt)  # pre-allocated with its own reg
        # Aliasing a Value that already has a different binding is rejected.
        with pytest.raises(RuntimeError, match="already bound"):
            a.alias(tgt, src)

    def test_declare_anonymous(self):
        a = RegAllocator()
        n = a.declare("u32")
        assert n.startswith("%r")
        # It must also show up in the declarations list.
        decls = a.declarations()
        assert any(n in line for line in decls)

    def test_declare_rejects_unknown_class(self):
        """A typo like ``declare("r")`` (prefix, not class) used to fall
        through to the default ``"r"`` prefix and emit
        ``.reg .r %r0, ...;`` — invalid PTX that only ptxas caught.
        The allocator now raises at emit time."""
        a = RegAllocator()
        with pytest.raises(ValueError, match="unknown register class"):
            a.declare("r")
        with pytest.raises(ValueError, match="unknown register class"):
            a.declare("f")  # prefix, not class — should be "f32"
