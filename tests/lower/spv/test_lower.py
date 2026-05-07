"""Tests for the SPIR-V lowerer's first cut.

Two tiers:

1. **Text-emit goldens** (cross-platform). Build a small IR module,
   run it through ``SpirVLowerer.lower_module``, assert the text
   output mentions the expected SPIR-V opcodes / decorations.
   Doesn't need ``spirv-as`` or Vulkan; runs everywhere.

2. **End-to-end smoke** (Linux + Vulkan + ``spirv-as``). Lower →
   assemble → dispatch via ``SpvDriver`` → read back → numerics
   match. Skipped on Mac and on hosts without ``spirv-as``.

Tier 1 is the regression-prevention net for visitor-by-visitor
work; tier 2 proves the whole pipeline (lowerer + assembler +
driver + dispatch) doesn't drift.
"""

from __future__ import annotations

import ctypes
import shutil
import sys

import numpy as np
import pytest

from quark.ir import Builder, DType
from quark.ir.tensor import GlobalTensor
from quark.ir.types import BufferType
from quark.lower.spv import SpirVLowerer


def _build_vec_add_ir(n: int = 64):
    """Construct a synthetic ``Z[i] = X[i] + Y[i]`` IR module.
    Mirrors ``tests/launcher/test_cuda_e2e.py``'s shape, ported to
    work without the launcher's compile machinery."""
    b = Builder("vec_add_module")
    fn = b.begin_function("vec_add")
    b.param("X", BufferType(DType.F32))
    b.param("Y", BufferType(DType.F32))
    b.param("Z", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_y = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Y", param=fn.params[1])
    g_z = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[2])
    tid = b.thread_idx("x")
    b.store(g_z, b.add(b.load(g_x, tid), b.load(g_y, tid)), tid)
    b.end_function()
    return b.module


# ─────────────────────────────────────────────────────────────────
# Tier 1 — text-emit goldens.
# ─────────────────────────────────────────────────────────────────


class TestTextEmit:
    """Cross-platform: just check the text output's shape."""

    def test_vec_add_emits_expected_skeleton(self):
        result = SpirVLowerer(local_size=(64, 1, 1)).lower_module(_build_vec_add_ir())
        src = result.source

        # Capabilities + memory model.
        assert "OpCapability Shader" in src
        assert "OpMemoryModel Logical GLSL450" in src

        # Entry point + execution mode.
        assert 'OpEntryPoint GLCompute' in src
        assert '"main"' in src
        assert "OpExecutionMode" in src and "LocalSize 64 1 1" in src

        # Exactly one LocalInvocationId builtin.
        assert src.count("BuiltIn LocalInvocationId") == 1

        # Three storage-buffer bindings, descriptor set 0, sequential.
        assert src.count("DescriptorSet 0") == 3
        assert "Binding 0" in src
        assert "Binding 1" in src
        assert "Binding 2" in src

        # Block decoration appears exactly once on the struct (de-dupe
        # works — the lowerer otherwise emits it per-buffer).
        block_lines = [
            line for line in src.splitlines()
            if line.startswith("OpDecorate") and " Block" in line and "MemberDecorate" not in line
        ]
        assert len(block_lines) == 1, (
            f"expected single 'Block' decoration; got {block_lines}"
        )

        # Body ops we expect.
        assert "OpFAdd" in src
        assert "OpAccessChain" in src
        assert "OpLoad" in src
        assert "OpStore" in src
        assert "OpReturn" in src
        assert "OpFunctionEnd" in src

    def test_metadata_matches_kernel_shape(self):
        result = SpirVLowerer(local_size=(32, 1, 1)).lower_module(_build_vec_add_ir())
        assert result.entry_name == "main"
        assert result.n_buffers == 3
        assert result.local_size == (32, 1, 1)
        assert result.push_constants_size == 0


# ─────────────────────────────────────────────────────────────────
# Tier 2 — end-to-end through spirv-as + SpvDriver.
# ─────────────────────────────────────────────────────────────────


pytestmark_e2e = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("spirv-as") is None,
    reason="needs Linux + Vulkan + spirv-as for end-to-end dispatch",
)


@pytest.fixture(scope="module")
def driver():
    """SpvDriver bound to the default device. Skipped on hosts where
    Vulkan isn't reachable."""
    if sys.platform != "linux":
        pytest.skip("Linux-only")
    from quark.drivers import spv
    if not spv.is_available():
        pytest.skip("no Vulkan device")
    return spv.SpvDriver()


def _build_vec_add_with_bounds_ir(n: int):
    """Bounds-check pattern: dispatch a multiple-of-local-size grid,
    use ``if (i < n)`` to skip out-of-range elements. Mirrors the
    GLSL fixture from §3.1's compile/launch test, but built from
    quark IR so the ``CmpOp`` + ``IfRegionOp`` + ``BlockIdxOp`` +
    ``BlockDimOp`` visitors get exercised.

    Global thread id is the standard ``block_idx*block_dim +
    thread_idx`` composition — the same shape every multi-workgroup
    kernel emits.
    """
    import quark.lang as qk

    b = Builder("vec_add_bounds_module")
    fn = b.begin_function("vec_add_bounds")
    b.param("X", BufferType(DType.F32))
    b.param("Y", BufferType(DType.F32))
    b.param("Z", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_y = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Y", param=fn.params[1])
    g_z = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[2])

    gid = b.add(b.mul(b.block_idx("x"), b.block_dim("x")), b.thread_idx("x"))
    n_const = b.const(DType.U32, n)
    pred = b.cmp("lt", gid, n_const)
    with qk.if_(pred) as (then_in, else_in, arms):
        with arms.then_():
            b.store(g_z, b.add(b.load(g_x, gid), b.load(g_y, gid)), gid)
            qk.yield_()
        with arms.else_():
            qk.yield_()
    b.end_function()
    return b.module


class TestBoundsCheckedKernel:
    """Tier 1 goldens for the bounds-check / multi-workgroup path —
    the visitor set the §3.2 second-cut adds."""

    def test_emits_workgroup_id_for_block_idx(self):
        result = SpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_with_bounds_ir(100)
        )
        # Both LocalInvocationId AND WorkgroupId in interface, since
        # the kernel composes a global id from both.
        assert "BuiltIn LocalInvocationId" in result.source
        assert "BuiltIn WorkgroupId" in result.source

    def test_emits_constant_for_block_dim(self):
        result = SpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_with_bounds_ir(100)
        )
        # block_dim("x") is a compile-time constant (LocalSize.x).
        assert "OpConstant " in result.source
        assert "64" in result.source

    def test_emits_structured_if(self):
        result = SpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_with_bounds_ir(100)
        )
        src = result.source
        assert "OpSelectionMerge" in src
        assert "OpBranchConditional" in src
        # Two OpBranch (one per arm), one OpReturn at function exit.
        assert src.count("OpBranch ") >= 2
        # The bounds check uses ULessThan on u32.
        assert "OpULessThan" in src


@pytestmark_e2e
def test_vec_add_end_to_end_through_lowerer(driver):
    """The full pipeline: build IR → lower → assemble → compile →
    dispatch → read back → numerics. Sister test to
    ``test_spv_compile_launch.test_vector_add_end_to_end``, but
    going through the framework lowerer instead of the hand-written
    GLSL fixture. If both pass, the lowerer's text emit is good."""
    from quark.lower.spv import text_to_binary

    n = 64
    rng = np.random.default_rng(0xBEEFCAFE)
    X = rng.standard_normal(n).astype(np.float32)
    Y = rng.standard_normal(n).astype(np.float32)
    expected = X + Y

    # Lower + assemble.
    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(_build_vec_add_ir(n))
    binary = text_to_binary(result.source)

    # Compile + dispatch via the public driver.
    nbytes = n * 4
    a_h, a_map = driver.allocate_buffer(nbytes)
    b_h, b_map = driver.allocate_buffer(nbytes)
    c_h, c_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(a_map, X.ctypes.data, X.nbytes)
    ctypes.memmove(b_map, Y.ctypes.data, Y.nbytes)

    compiled = driver.compile(
        source=binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    # Single-workgroup dispatch: grid is (1, 1, 1), workgroup size
    # is local_size = (n, 1, 1), so n threads cover the n elements.
    driver.launch(compiled, (1, 1, 1), [a_h, b_h, c_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, c_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def _build_math_kernel_ir(n: int):
    """Per-thread kernel: ``Z[i] = sin(X[i] * 2.0) + sqrt(Y[i] + 1.0)``.

    Exercises ``MathOp`` (sin / sqrt) + ``ConvertOp`` (constant
    promotion) + ``ArithOp`` (mul / add) in one shape. Ground truth
    is computed in numpy at the test site; SPIR-V output compared
    with a small fp tolerance to absorb the GLSL.std.450 ULP slack
    PORTABILITY_PLAN §3.7 v1 flagged.
    """
    import quark.lang as qk

    b = Builder("math_kernel")
    fn = b.begin_function("math_kernel")
    b.param("X", BufferType(DType.F32))
    b.param("Y", BufferType(DType.F32))
    b.param("Z", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_y = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Y", param=fn.params[1])
    g_z = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[2])

    tid = b.thread_idx("x")
    x = b.load(g_x, tid)
    y = b.load(g_y, tid)
    two = b.const(DType.F32, 2.0)
    one = b.const(DType.F32, 1.0)
    sin_term = qk.sin(b.mul(x, two))
    sqrt_term = qk.sqrt(b.add(y, one))
    out = b.add(sin_term, sqrt_term)
    b.store(g_z, out, tid)
    b.end_function()
    return b.module


class TestMathConvertSelect:
    """Tier 1 goldens for the new visitors."""

    def test_math_emits_glsl_extinst(self):
        result = SpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_math_kernel_ir(64)
        )
        src = result.source
        assert 'OpExtInstImport "GLSL.std.450"' in src
        # spirv-as wants symbolic names, not numeric instruction codes.
        assert "OpExtInst" in src
        assert " Sin " in src
        assert " Sqrt " in src

    def test_convert_short_circuits_same_dtype(self):
        """``convert(x, dst=x.dtype)`` is a no-op — no OpFConvert /
        OpUConvert emitted."""
        b = Builder("noop_convert")
        b.begin_function("f")
        x = b.const(DType.F32, 1.0)
        b.convert(x, dst=DType.F32)
        b.end_function()
        src = SpirVLowerer().lower_module(b.module).source
        assert "OpFConvert" not in src
        assert "OpUConvert" not in src

    def test_convert_f32_f16_emits_fconvert_and_caps(self):
        b = Builder("fcvt")
        b.begin_function("f")
        x = b.const(DType.F32, 1.5)
        b.convert(x, dst=DType.F16)
        b.end_function()
        src = SpirVLowerer().lower_module(b.module).source
        assert "OpCapability Float16" in src
        assert "OpFConvert" in src
        assert "OpTypeFloat 16" in src

    def test_select_emits_opselect(self):
        b = Builder("sel")
        b.begin_function("f")
        x = b.const(DType.F32, 1.0)
        y = b.const(DType.F32, 2.0)
        p = b.cmp("lt", x, y)
        b.select(p, x, y)
        b.end_function()
        src = SpirVLowerer().lower_module(b.module).source
        assert "OpSelect" in src


@pytestmark_e2e
def test_math_kernel_end_to_end(driver):
    """End-to-end: lower a kernel using ``MathOp`` (sin / sqrt) +
    ``ArithOp`` (mul / add), dispatch on Battlemage, compare to a
    numpy reference. Tolerance accounts for GLSL.std.450 ULP slack —
    Vulkan's transcendentals carry up to 4 ULP of error vs PTX's
    ``ex2.approx.f32``."""
    from quark.lower.spv import text_to_binary

    n = 64
    rng = np.random.default_rng(0xFEED)
    X = rng.standard_normal(n).astype(np.float32)
    Y = (rng.standard_normal(n).astype(np.float32) ** 2)  # ensure y+1 > 0
    expected = np.sin(X * 2.0) + np.sqrt(Y + 1.0)

    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_math_kernel_ir(n)
    )
    binary = text_to_binary(result.source)

    nbytes = n * 4
    a_h, a_map = driver.allocate_buffer(nbytes)
    b_h, b_map = driver.allocate_buffer(nbytes)
    c_h, c_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(a_map, X.ctypes.data, X.nbytes)
    ctypes.memmove(b_map, Y.ctypes.data, Y.nbytes)

    compiled = driver.compile(
        source=binary, entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (1, 1, 1), [a_h, b_h, c_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, c_map, got.nbytes)
    # 1e-5 absolute tolerance covers Vulkan GLSL.std.450's ULP slack
    # (per spec, sin/sqrt carry up to 4 ULP of error). Tighter than
    # 1e-4; loose enough to absorb driver variance.
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytestmark_e2e
def test_vec_add_with_bounds_end_to_end(driver):
    """Multi-workgroup dispatch + bounds check. The kernel computes
    a global thread id from ``block_idx*block_dim + thread_idx``,
    early-exits when ``gid >= n``. Dispatch grid is
    ``ceil(n / local_size)`` workgroups; the trailing partial
    workgroup hits the bounds check and skips work without writing
    out of range.
    """
    from quark.lower.spv import text_to_binary

    n = 100  # not a multiple of 64 — partial trailing workgroup
    local_size_x = 64
    n_workgroups = (n + local_size_x - 1) // local_size_x
    rng = np.random.default_rng(0xABCDEF)
    X = rng.standard_normal(n).astype(np.float32)
    Y = rng.standard_normal(n).astype(np.float32)
    expected = X + Y

    result = SpirVLowerer(local_size=(local_size_x, 1, 1)).lower_module(
        _build_vec_add_with_bounds_ir(n)
    )
    binary = text_to_binary(result.source)

    nbytes = n * 4
    a_h, a_map = driver.allocate_buffer(nbytes)
    b_h, b_map = driver.allocate_buffer(nbytes)
    c_h, c_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(a_map, X.ctypes.data, X.nbytes)
    ctypes.memmove(b_map, Y.ctypes.data, Y.nbytes)

    compiled = driver.compile(
        source=binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (n_workgroups, 1, 1), [a_h, b_h, c_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, c_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
