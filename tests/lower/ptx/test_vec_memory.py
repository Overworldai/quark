"""Tests for vec load/store lowering (v2 / v4)."""

import re

import pytest

from quark.ir import BufferType, DType, GlobalTensor
from tests.lower.ptx.conftest import lower


def _gmem(b, dtype=DType.F32, shape=(16, 32)):
    b.param("X", BufferType(dtype))
    return GlobalTensor(
        dtype=dtype,
        shape=shape,
        stride=(shape[1], 1),
        name="X",
        param=b.function.params[-1],
    )


class TestVecLoad:
    def test_v4_gmem_load(self, fresh_builder):
        g = _gmem(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=4)
        out = lower(fresh_builder)
        assert re.search(
            r"ld\.global\.v4\.f32 \{%f\d+, %f\d+, %f\d+, %f\d+\}, \[%rd0\];",
            out,
        )

    def test_v2_gmem_load(self, fresh_builder):
        g = _gmem(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=2)
        out = lower(fresh_builder)
        assert re.search(r"ld\.global\.v2\.f32 \{%f\d+, %f\d+\}, ", out)

    def test_smem_v4_load(self, fresh_builder):
        A = fresh_builder.smem_alloc("A", DType.F32, (16, 32))
        row = fresh_builder.const(DType.U32, 1)
        col = fresh_builder.const(DType.U32, 4)
        fresh_builder.vec_load(A, row, col, width=4)
        out = lower(fresh_builder)
        # Offset = 1*32*4 + 4*4 = 144
        assert re.search(
            r"ld\.shared\.v4\.f32 \{%f\d+, %f\d+, %f\d+, %f\d+\}, \[%r\d+ \+ 144\];",
            out,
        )

    def test_v3_rejected(self, fresh_builder):
        g = _gmem(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        # IR allows width=3; PTX doesn't.
        fresh_builder.vec_load(g, row, col, width=3)
        with pytest.raises(NotImplementedError, match="width"):
            lower(fresh_builder)

    def test_v8_b16_lowers_to_single_v4_b32(self, fresh_builder):
        """A 16-byte transfer requested as v8.b16 is a single 16B ld —
        the lowerer canonicalizes the reg class to b32 and emits ONE
        `ld.global.v4.b32` (not two ld.v4.b16). Bytes are bytes; this
        keeps the common 16B path on one instruction.

        Regression from kv_cache_update asking for v8.b16: previously
        the lowerer either rejected it outright or split into 2 ops.
        """
        g = _gmem(fresh_builder, dtype=DType.BF16, shape=(8, 16))
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=8, dtype=DType.B16)
        out = lower(fresh_builder)
        assert out.count("ld.global.v4.b32") == 1
        # The b16 form must NOT be emitted — the rewrite picks b32.
        assert "ld.global.v4.b16" not in out
        assert "ld.global.v4.b16" not in out

    def test_v8_b16_store_lowers_to_single_v4_b32(self, fresh_builder):
        """vec_store of a value loaded as v8.b16 uses the same b32
        physical regs and emits one st.global.v4.b32."""
        g = _gmem(fresh_builder, dtype=DType.BF16, shape=(8, 16))
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        v = fresh_builder.vec_load(g, row, col, width=8, dtype=DType.B16)
        fresh_builder.vec_store(g, v, row, col)
        out = lower(fresh_builder)
        assert out.count("st.global.v4.b32") == 1
        assert "st.global.v4.b16" not in out

    def test_v16_e4m3_lowers_to_single_v4_b32(self, fresh_builder):
        """16-byte transfer via 16 fp8 elements lowers to one v4.b32."""
        g = _gmem(fresh_builder, dtype=DType.E4M3, shape=(8, 16))
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=16, dtype=DType.E4M3)
        out = lower(fresh_builder)
        assert out.count("ld.global.v4.b32") == 1

    def test_v4_b16_kept_as_v4_b16(self, fresh_builder):
        """A v4.b16 (8B) request stays as v4.b16 — width is already a
        legal PTX vec width, no canonicalization needed."""
        g = _gmem(fresh_builder, dtype=DType.BF16, shape=(8, 16))
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=4, dtype=DType.B16)
        out = lower(fresh_builder)
        assert "ld.global.v4.b16" in out

    def test_v2_b16_supported(self, fresh_builder):
        """4-byte transfer via v2.b16 stays as v2.b16."""
        g = _gmem(fresh_builder, dtype=DType.BF16, shape=(8, 16))
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=2, dtype=DType.B16)
        out = lower(fresh_builder)
        assert out.count("ld.global.v2.b16") == 1

    def test_e4m3_v2_load_canonicalizes_to_b16(self, fresh_builder):
        """fp8 dtypes have no PTX reg class — `ld.global.v2.e4m3` isn't a
        thing. The lowerer must canonicalize a 2-byte fp8 load to a
        single-instruction byte-equivalent form (b16 = 2 bytes).
        Regression: tile_loader's pack2_src path needs this to work.
        """
        g = _gmem(fresh_builder, dtype=DType.E4M3, shape=(8, 16))
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=2, dtype=DType.E4M3)
        out = lower(fresh_builder)
        # Single bit-typed load — NOT v2.e4m3 (illegal) and NOT split.
        assert "ld.global.b16" in out
        assert "e4m3" not in out  # no leaked dtype suffix

    def test_e4m3_v4_load_canonicalizes_to_b32(self, fresh_builder):
        """4 fp8 bytes → one ld.global.b32 (bytes are bytes)."""
        g = _gmem(fresh_builder, dtype=DType.E4M3, shape=(8, 16))
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_load(g, row, col, width=4, dtype=DType.E4M3)
        out = lower(fresh_builder)
        assert "ld.global.b32" in out
        assert "e4m3" not in out


class TestVecStore:
    def test_v4_gmem_store(self, fresh_builder):
        g = _gmem(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        v = fresh_builder.vec_load(g, row, col, width=4)
        fresh_builder.vec_store(g, v, row, col)
        out = lower(fresh_builder)
        assert re.search(
            r"st\.global\.v4\.f32 \[%rd0\], \{%f\d+, %f\d+, %f\d+, %f\d+\};",
            out,
        )

    def test_build_then_vec_store(self, fresh_builder):
        """vec_build + vec_store should emit only one store (the build is
        pure aliasing)."""
        A = fresh_builder.smem_alloc("A", DType.F32, (4, 4))
        a0 = fresh_builder.const(DType.F32, 1.0)
        a1 = fresh_builder.const(DType.F32, 2.0)
        a2 = fresh_builder.const(DType.F32, 3.0)
        a3 = fresh_builder.const(DType.F32, 4.0)
        v = fresh_builder.vec_build([a0, a1, a2, a3])
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.vec_store(A, v, row, col)
        out = lower(fresh_builder)
        assert out.count("st.shared.v4.f32") == 1
