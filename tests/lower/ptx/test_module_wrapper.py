"""Tests for the PTX module/kernel wrapper: headers, params, reg decls."""

import re

import pytest

from popcorn.ir import BufferType, Builder, DType, ScalarType
from popcorn.lower.ptx import LoweredKernel, PtxLowerer


def _lower(module):
    return PtxLowerer().lower_module(module)


def test_module_header_has_version_target_address_size():
    b = Builder("m")
    b.begin_function("f")
    b.end_function()
    out = _lower(b.module).ptx
    assert ".version 8.4" in out
    assert ".target sm_89" in out
    assert ".address_size 64" in out


def test_version_and_target_are_configurable():
    b = Builder("m")
    b.begin_function("f")
    b.end_function()
    out = PtxLowerer(ptx_version="8.7", target_sm=90).lower_module(b.module).ptx
    assert ".version 8.7" in out
    assert ".target sm_90" in out


def test_visible_entry_with_kernel_name():
    b = Builder("m")
    b.begin_function("my_kernel")
    b.end_function()
    lowered = _lower(b.module)
    assert ".visible .entry my_kernel(" in lowered.ptx
    assert lowered.kernel_name == "my_kernel"


def test_buffer_params_use_u64():
    b = Builder("m")
    b.begin_function("f")
    b.param("X", BufferType(DType.BF16))
    b.param("Y", BufferType(DType.F32))
    b.end_function()
    out = _lower(b.module).ptx
    assert ".param .u64 X" in out
    assert ".param .u64 Y" in out


def test_scalar_param_uses_its_dtype():
    b = Builder("m")
    b.begin_function("f")
    b.param("stride", ScalarType(DType.U32))
    b.end_function()
    out = _lower(b.module).ptx
    assert ".param .u32 stride" in out


def test_reg_decls_appear_before_instructions():
    b = Builder("m")
    b.begin_function("f")
    b.param("X", BufferType(DType.F32))
    b.const(DType.F32, 1.0)
    b.end_function()
    out = _lower(b.module).ptx
    reg_pos = out.index(".reg")
    first_instr = out.index("ld.param")
    assert reg_pos < first_instr


def test_body_ends_with_ret():
    b = Builder("m")
    b.begin_function("f")
    b.end_function()
    out = _lower(b.module).ptx
    assert re.search(r"ret;\n\}", out)


def test_multi_function_module_rejected():
    b = Builder("m")
    b.begin_function("f1")
    b.end_function()
    b.begin_function("f2")
    b.end_function()
    with pytest.raises(NotImplementedError):
        _lower(b.module)


# ---- LoweredKernel wrapper itself ----


def test_lowered_kernel_exposes_smem_bytes_zero_when_no_allocs():
    b = Builder("m")
    b.begin_function("f")
    b.end_function()
    lowered = _lower(b.module)
    assert isinstance(lowered, LoweredKernel)
    assert lowered.smem_bytes == 0
    # With zero smem, the declaration must not appear.
    assert ".shared" not in lowered.ptx


def test_lowered_kernel_reports_smem_bytes_and_declares_extern_shared():
    b = Builder("m")
    b.begin_function("f")
    # Two allocs: 8*16*4 = 512 bytes bf16 + 4*4*4 = 64 bytes f32
    # padded to 16 → 512 + 64 = 576.
    b.smem_alloc("A", DType.BF16, (8, 16))  # 256 bytes
    b.smem_alloc("B", DType.F32, (4, 4))  # 64 bytes, aligned start
    b.end_function()
    lowered = _lower(b.module)
    # A = 256 bytes → next alloc starts at offset 256 (16-aligned).
    # B = 64 bytes → total = 320.
    assert lowered.smem_bytes == 320
    # Dynamic smem pool declaration uses `.extern .shared` with no size.
    assert ".extern .shared .align 16 .b8 _smem_pool[];" in lowered.ptx
    # The static bracketed size form must NOT appear.
    assert ".shared .align 16 .b8 _smem_pool[320]" not in lowered.ptx


def test_lowered_kernel_str_returns_ptx():
    b = Builder("m")
    b.begin_function("kern")
    b.end_function()
    lowered = _lower(b.module)
    assert str(lowered) == lowered.ptx
