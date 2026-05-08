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


def _build_subgroup_reduce_ir(n: int):
    """Per-thread loads X[i], subgroup-reduce-sum across the
    workgroup, lane 0 writes the total to Y[0].

    Exercises ``SubgroupReduceOp`` (sum, f32) + ``ThreadIdxOp`` +
    ``IfRegionOp`` (lane-0 gate). N must be ≤ subgroup_size for
    a single-subgroup workgroup; on Battlemage that's 32.
    """
    import quark.lang as qk

    b = Builder("subgroup_reduce")
    fn = b.begin_function("subgroup_reduce")
    b.param("X", BufferType(DType.F32))
    b.param("Y", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_y = GlobalTensor(dtype=DType.F32, shape=(1,), stride=(1,),
                       name="Y", param=fn.params[1])

    tid = b.thread_idx("x")
    x_i = b.load(g_x, tid)
    sum_v = b.subgroup_reduce("sum", x_i)  # cross-lane sum
    zero = b.const(DType.U32, 0)
    is_lane_zero = b.cmp("eq", tid, zero)
    with qk.if_(is_lane_zero) as (then_in, else_in, arms):
        with arms.then_():
            b.store(g_y, sum_v, zero)
            qk.yield_()
        with arms.else_():
            qk.yield_()
    b.end_function()
    return b.module


class TestSmemBarrierSubgroup:
    """Tier 1 goldens for §3.2 v4 visitors: smem + barrier +
    subgroup reductions."""

    def test_subgroup_reduce_emits_op_group_nonuniform(self):
        result = SpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_subgroup_reduce_ir(32)
        )
        src = result.source
        assert "OpCapability GroupNonUniformArithmetic" in src
        assert "OpGroupNonUniformFAdd" in src
        # Subgroup scope (3) used for the reduction execution scope.
        assert " Reduce " in src

    def test_smem_alloc_emits_workgroup_variable(self):
        b = Builder("smem")
        b.begin_function("f")
        smem = b.smem_alloc("scratch", DType.F32, (32,))
        # Write index 0 = a constant, then read it back. Just exercises
        # the smem allocation + load/store paths.
        v = b.const(DType.F32, 1.5)
        zero = b.const(DType.U32, 0)
        b.store(smem, v, zero)
        b.load(smem, zero)
        b.end_function()
        src = SpirVLowerer().lower_module(b.module).source
        assert "OpVariable" in src and "Workgroup" in src
        # The flat array sized to 32.
        assert "OpTypeArray" in src

    def test_barrier_emits_op_control_barrier(self):
        b = Builder("barrier")
        b.begin_function("f")
        import quark.lang as qk
        qk.barrier()
        b.end_function()
        src = SpirVLowerer().lower_module(b.module).source
        assert "OpControlBarrier" in src


@pytestmark_e2e
def test_subgroup_reduce_end_to_end(driver):
    """Full subgroup-reduce kernel runs on Battlemage. N=32 matches
    the device's subgroup width (per the §3.1 probe), so a single
    subgroup covers all the work."""
    from quark.lower.spv import text_to_binary

    n = 32
    rng = np.random.default_rng(0xDEADBEEF)
    X = rng.standard_normal(n).astype(np.float32)
    expected = X.sum()

    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_subgroup_reduce_ir(n)
    )
    binary = text_to_binary(result.source)

    a_h, a_map = driver.allocate_buffer(n * 4)
    y_h, y_map = driver.allocate_buffer(4)
    ctypes.memmove(a_map, X.ctypes.data, X.nbytes)

    compiled = driver.compile(
        source=binary, entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (1, 1, 1), [a_h, y_h], push_bytes=b"")

    got = np.empty(1, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, y_map, got.nbytes)
    # Subgroup reduction order is implementation-defined; allow some
    # ULP slack on the float-add tree.
    np.testing.assert_allclose(got[0], expected, rtol=1e-5, atol=1e-5)


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


def _build_for_loop_accum_ir(n: int, n_iters: int):
    """Per-thread accumulator: ``Out[t] = sum_{k=0}^{n_iters-1} X[t]``.

    Exercises ``ForLoopOp`` + carries: the loop carries an F32
    accumulator, the body adds ``X[t]`` to it once per iteration. After
    ``n_iters`` iterations the carry equals ``X[t] * n_iters``, which we
    write to ``Out[t]``. Catches OpPhi at the loop header, OpCopyObject
    for the yielded carry, and the merge-block exit value.
    """
    import quark.lang as qk

    b = Builder("for_loop_module")
    fn = b.begin_function("for_loop_accum")
    b.param("X", BufferType(DType.F32))
    b.param("Out", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_o = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Out", param=fn.params[1])

    tid = b.thread_idx("x")
    x_t = b.load(g_x, tid)
    zero_f = b.const(DType.F32, 0.0)
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, n_iters)
    step = b.const(DType.U32, 1)
    with b.for_loop(lo, hi, step, iv_name="k", carried=(zero_f,)) as (
        _k, (acc_in,),
    ):
        new_acc = b.add(acc_in, x_t)
        b.yield_(new_acc)
    final_acc = b.last_results[0]
    b.store(g_o, final_acc, tid)
    b.end_function()
    return b.module


class TestForLoopTextEmit:
    """Tier 1 — for_loop visitor must emit the structured-loop skeleton
    (preheader + header with phi + cond + loop merge + body + continue
    with iv increment + merge), independent of any Vulkan device."""

    def test_emits_loop_skeleton(self):
        result = SpirVLowerer(local_size=(8, 1, 1)).lower_module(
            _build_for_loop_accum_ir(8, n_iters=4)
        )
        src = result.source
        # Structured loop primitives.
        assert "OpLoopMerge" in src
        assert "OpPhi" in src
        # iv increment via OpIAdd in the continue block.
        assert "OpIAdd" in src
        # Carry materialisation at YieldOp.
        assert "OpCopyObject" in src
        # Loop predicate uses ULessThan on u32 induction.
        assert "OpULessThan" in src


@pytestmark_e2e
def test_for_loop_accum_end_to_end(driver):
    """End-to-end: build a for_loop kernel with an F32 carry, lower it,
    dispatch it on the Vulkan device, and verify
    ``Out[t] == X[t] * n_iters``. This is the first kernel through the
    SPIR-V backend that uses ``ForLoopOp`` + carries."""
    from quark.lower.spv import text_to_binary

    n = 64
    n_iters = 5
    rng = np.random.default_rng(0xF00DCAFE)
    X = rng.standard_normal(n).astype(np.float32)
    expected = X * n_iters

    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_for_loop_accum_ir(n, n_iters=n_iters)
    )
    binary = text_to_binary(result.source)

    nbytes = n * 4
    x_h, x_map = driver.allocate_buffer(nbytes)
    o_h, o_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

    compiled = driver.compile(
        source=binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (1, 1, 1), [x_h, o_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def _build_for_loop_window_sum_ir(n: int, win: int):
    """Per-thread sliding-window sum: ``Out[t] = sum_{k=0}^{win-1} X[t+k]``.

    Loop body uses the induction variable to compute a varying load
    address, so this kernel exercises the iv path through OpPhi (not
    just the carry path of ``test_for_loop_accum_end_to_end``). X is
    sized ``n + win`` so the trailing window is in-bounds for every
    thread ``t in [0, n)``.
    """
    import quark.lang as qk

    b = Builder("for_loop_window")
    fn = b.begin_function("for_loop_window")
    b.param("X", BufferType(DType.F32))
    b.param("Out", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n + win,), stride=(1,),
                       name="X", param=fn.params[0])
    g_o = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Out", param=fn.params[1])

    tid = b.thread_idx("x")
    zero_f = b.const(DType.F32, 0.0)
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, win)
    step = b.const(DType.U32, 1)
    with b.for_loop(lo, hi, step, iv_name="k", carried=(zero_f,)) as (
        k, (acc_in,),
    ):
        addr = b.add(tid, k)
        x_v = b.load(g_x, addr)
        new_acc = b.add(acc_in, x_v)
        b.yield_(new_acc)
    final_acc = b.last_results[0]
    b.store(g_o, final_acc, tid)
    b.end_function()
    return b.module


def _build_cmp_select_ir(n: int):
    """Per-thread bounds-gated copy:

      Out[t] = X[t]              if t < n/2 else
               X[t] + 100.0      otherwise   (the "lerp" branch — easy
                                              to detect when wrong)

    Lowers cmp + select on F32 directly in the body. Used as a
    cross-platform sanity check that ``OpSelect`` is emitted with
    the (cond, t-arm, f-arm) operand order spirv-as expects.
    """
    import quark.lang as qk

    b = Builder("cmp_sel_module")
    fn = b.begin_function("cmp_sel")
    b.param("X", BufferType(DType.F32))
    b.param("Out", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_o = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Out", param=fn.params[1])

    tid = b.thread_idx("x")
    half = b.const(DType.U32, n // 2)
    is_lower = b.cmp("lt", tid, half)
    x_t = b.load(g_x, tid)
    bonus = b.const(DType.F32, 100.0)
    upper_val = b.add(x_t, bonus)
    out = b.select(is_lower, x_t, upper_val)
    b.store(g_o, out, tid)
    b.end_function()
    return b.module


def _build_cmp_and_select_ir(n: int):
    """Per-thread bounds-gated kernel using ``and(cmp, cmp)``:

      in_range = (t >= lo) && (t < hi)
      Out[t]   = X[t] if in_range else X[t] + 100.0

    With lo=n/4 and hi=3n/4, threads ``[n/4, 3n/4)`` get the X[t]
    arm; everyone else gets X[t]+100. Mirrors the
    ``value_residual_packed`` v-col bounds-check pattern that broke
    in the wider kernel.
    """
    import quark.lang as qk

    b = Builder("cmp_and_sel_module")
    fn = b.begin_function("cmp_and_sel")
    b.param("X", BufferType(DType.F32))
    b.param("Out", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_o = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Out", param=fn.params[1])

    tid = b.thread_idx("x")
    lo_b = b.const(DType.U32, n // 4)
    hi_b = b.const(DType.U32, 3 * n // 4)
    cmp_lo = b.cmp("ge", tid, lo_b)
    cmp_hi = b.cmp("lt", tid, hi_b)
    in_range = b.and_(cmp_lo, cmp_hi)
    x_t = b.load(g_x, tid)
    bonus = b.const(DType.F32, 100.0)
    upper_val = b.add(x_t, bonus)
    out = b.select(in_range, x_t, upper_val)
    b.store(g_o, out, tid)
    b.end_function()
    return b.module


@pytestmark_e2e
def test_cmp_and_select_runs_end_to_end(driver):
    """Per-thread cmp + and + OpSelect on a F32 payload — verifies
    the boolean ``OpLogicalAnd`` of two ``OpUGreaterThanEqual`` /
    ``OpULessThan`` results feeds OpSelect correctly. Same pattern
    that ``value_residual_packed`` uses for its V-col bounds gate."""
    from quark.lower.spv import text_to_binary

    n = 64
    lo, hi = n // 4, 3 * n // 4  # in-range = [16, 48)
    rng = np.random.default_rng(0xFADE_FACE & 0xFFFFFFFF)
    X = rng.standard_normal(n).astype(np.float32)
    in_range_mask = (np.arange(n) >= lo) & (np.arange(n) < hi)
    expected = np.where(in_range_mask, X, X + 100.0).astype(np.float32)

    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_cmp_and_select_ir(n)
    )
    binary = text_to_binary(result.source)

    nbytes = n * 4
    x_h, x_map = driver.allocate_buffer(nbytes)
    o_h, o_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

    compiled = driver.compile(
        source=binary, entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (1, 1, 1), [x_h, o_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


@pytestmark_e2e
def test_cmp_select_runs_end_to_end(driver):
    """Per-thread cmp + OpSelect — verifies the SPV backend's
    select operand ordering is correct (cond, t-arm, f-arm), which
    is the lurking suspect when complex kernels see wrong-arm
    output. Lower-half writes ``X[t]``; upper-half writes
    ``X[t] + 100`` — diff is huge enough to catch a swapped arm
    even with rounding."""
    from quark.lower.spv import text_to_binary

    n = 64
    rng = np.random.default_rng(0xCAFE_FEED)
    X = rng.standard_normal(n).astype(np.float32)
    expected = X.copy()
    expected[n // 2 :] = X[n // 2 :] + 100.0

    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_cmp_select_ir(n)
    )
    binary = text_to_binary(result.source)

    nbytes = n * 4
    x_h, x_map = driver.allocate_buffer(nbytes)
    o_h, o_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

    compiled = driver.compile(
        source=binary, entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (1, 1, 1), [x_h, o_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def _build_coopmat_mma_ir():
    """Synthetic m8n16k16 bf16→f32 MMA: ``D = A @ B + C`` on three
    ``GlobalTensor`` buffers in row-major.

    A is ``[8, 16] bf16``, B is ``[16, 16] bf16``, C/D is ``[8, 16] f32``.
    Single-workgroup, single-MMA shape — proves the
    ``LoadMatrixOp`` / ``MmaOp`` / ``StoreMatrixOp`` visitor chain
    emits valid SPIR-V (i.e. spirv-as accepts the output)."""
    from quark.ir.module import MmaShape

    SHAPE_NAME = "m8n16k16_intel_bf16_f32"
    b = Builder("coopmat_mma")
    fn = b.begin_function("coopmat_mma")
    b.param("A", BufferType(DType.BF16))
    b.param("B", BufferType(DType.BF16))
    b.param("C", BufferType(DType.F32))
    b.param("D", BufferType(DType.F32))

    g_a = GlobalTensor(dtype=DType.BF16, shape=(8, 16), stride=(16, 1),
                       name="A", param=fn.params[0])
    # B follows the framework's gemm convention: stored (N, K)
    # row-major so the K axis is contiguous. The SPV ``b`` load
    # treats this as a column-major K×N tile, which the MMA reads
    # as the standard ``B`` matrix (K×N math view).
    g_b = GlobalTensor(dtype=DType.BF16, shape=(16, 16), stride=(16, 1),
                       name="B", param=fn.params[1])
    g_c = GlobalTensor(dtype=DType.F32, shape=(8, 16), stride=(16, 1),
                       name="C", param=fn.params[2])
    g_d = GlobalTensor(dtype=DType.F32, shape=(8, 16), stride=(16, 1),
                       name="D", param=fn.params[3])

    # Register the Intel m8n16k16 shape for this kernel module so
    # ``load_matrix`` finds it.
    b.register_shape(
        MmaShape(
            name=SHAPE_NAME,
            m=8, n=16, k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
            a_regs=4, b_regs=8, c_regs=4,
        ),
    )

    zero = b.const(DType.U32, 0)
    a_frag = b.load_matrix(g_a, SHAPE_NAME, which="a", row=zero, col=zero)
    b_frag = b.load_matrix(g_b, SHAPE_NAME, which="b", row=zero, col=zero)
    c_frag = b.load_matrix(g_c, SHAPE_NAME, which="c", row=zero, col=zero)
    d_frag = b.mma(SHAPE_NAME, a_frag, b_frag, c_frag)
    b.store_matrix(g_d, d_frag, SHAPE_NAME, which="d", row=zero, col=zero)
    b.end_function()
    return b.module


def _build_frag_convert_multi_src_ir():
    """Two MMAs producing two ACC fragments → one ``frag_convert``
    merging both into a single A-fragment (kf=2). Just lowers and
    drops the result on the floor — the test only asserts that the
    multi-source dispatch path emits the OpSelect chain.

    Single workgroup, single warp, no dispatch grid math; exists
    purely to exercise the lowerer's ``num_src=2`` branch."""
    from quark.ir.module import MmaShape

    SHAPE_NAME = "m8n16k16_intel_bf16_f32"
    b = Builder("frag_cvt_multi")
    fn = b.begin_function("frag_cvt_multi")
    b.param("A0", BufferType(DType.BF16))
    b.param("B0", BufferType(DType.BF16))
    b.param("A1", BufferType(DType.BF16))
    b.param("B1", BufferType(DType.BF16))
    b.param("Out", BufferType(DType.U32))

    g_a0 = GlobalTensor(dtype=DType.BF16, shape=(8, 16), stride=(16, 1),
                        name="A0", param=fn.params[0])
    g_b0 = GlobalTensor(dtype=DType.BF16, shape=(16, 16), stride=(16, 1),
                        name="B0", param=fn.params[1])
    g_a1 = GlobalTensor(dtype=DType.BF16, shape=(8, 16), stride=(16, 1),
                        name="A1", param=fn.params[2])
    g_b1 = GlobalTensor(dtype=DType.BF16, shape=(16, 16), stride=(16, 1),
                        name="B1", param=fn.params[3])
    g_out = GlobalTensor(dtype=DType.U32, shape=(8, 16), stride=(16, 1),
                         name="Out", param=fn.params[4])

    b.register_shape(
        MmaShape(
            name=SHAPE_NAME,
            m=8, n=16, k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
            a_regs=4, b_regs=8, c_regs=4,
        ),
    )

    zero = b.const(DType.U32, 0)
    zero_f = b.const(DType.F32, 0.0)
    c_init = b.vec_build([zero_f] * 4)

    a0 = b.load_matrix(g_a0, SHAPE_NAME, which="a", row=zero, col=zero)
    b0 = b.load_matrix(g_b0, SHAPE_NAME, which="b", row=zero, col=zero)
    src0 = b.mma(SHAPE_NAME, a0, b0, c_init)

    a1 = b.load_matrix(g_a1, SHAPE_NAME, which="a", row=zero, col=zero)
    b1 = b.load_matrix(g_b1, SHAPE_NAME, which="b", row=zero, col=zero)
    src1 = b.mma(SHAPE_NAME, a1, b1, c_init)

    # Merge two ACC frags into one A-fragment of bf16. cd_offsets is
    # the Intel placeholder; the lowerer's slot iteration is driven
    # by tile dimensions, not cd_offsets.
    b.frag_convert(
        SHAPE_NAME,
        [src0, src1],
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=((0, 0),) * 4,
    )
    # Sink something to keep the function non-empty after frag_convert
    # (the result is unused; we don't store it anywhere — but we still
    # need the entry-point interface to advertise Out).
    b.store(g_out, b.const(DType.U32, 0), zero, zero)
    b.end_function()
    return b.module


class TestCoopMatTextEmit:
    """Tier 1 — the cooperative-matrix visitor surface emits the
    expected SPIR-V opcodes regardless of Vulkan availability."""

    def test_emits_coopmat_skeleton(self):
        result = SpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_coopmat_mma_ir()
        )
        src = result.source
        assert "OpCapability CooperativeMatrixKHR" in src
        assert "OpExtension \"SPV_KHR_cooperative_matrix\"" in src
        assert "OpTypeCooperativeMatrixKHR" in src
        assert "OpCooperativeMatrixLoadKHR" in src
        assert "OpCooperativeMatrixStoreKHR" in src
        assert "OpCooperativeMatrixMulAddKHR" in src

    def test_frag_convert_multi_src_emits_opselect_chain(self):
        """num_src=2 FragConvert lowers to an OpSelect chain — one
        OpIEqual + OpSelect per source-1 — instead of a single direct
        load. Locks in the multi-src dispatch path that the kf>1 GEMM2
        P-fragment build needs (e.g. PTX m16n8k16 attention)."""
        result = SpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_frag_convert_multi_src_ir()
        )
        src = result.source
        # Per-slot OpSelect chain: with num_src=2 there's one OpIEqual
        # + one OpSelect per slot. With dst_n_elems=8*32=256 elements
        # over 32 lanes → 8 slots → exactly 8 OpSelects (one per slot,
        # selecting between src0 and src1 loads on src_idx==0).
        assert src.count("OpIEqual") == 8
        assert src.count("OpSelect ") == 8
        # Three Workgroup smem regions: src0 + src1 scratch (one each)
        # plus the dst scratch.
        assert src.count("OpVariable") >= 3


@pytestmark_e2e
def test_coopmat_mma_assembles(driver):
    """End-to-end through ``spirv-as``: build the synthetic m8n16k16
    MMA IR, lower, assemble. Doesn't dispatch (the kernel uses
    coordinates wired for a single workgroup at row=col=0; running
    it requires the host-side dispatch grid + descriptor wiring the
    smoke tests already cover).

    Acceptance gate: ``spirv-as --target-env vulkan1.4`` accepts the
    text output. Catches type-id mismatches and the
    ``OpCooperativeMatrix*KHR`` operand orderings without needing a
    GEMM kernel that lowers all the way through the launcher."""
    from quark.lower.spv import text_to_binary

    result = SpirVLowerer(local_size=(32, 1, 1)).lower_module(
        _build_coopmat_mma_ir()
    )
    binary = text_to_binary(result.source)
    assert len(binary) > 0


@pytestmark_e2e
def test_coopmat_mma_runs_end_to_end(driver):
    """Dispatch the synthetic m8n16k16 MMA on Battlemage and verify
    ``D == A @ B + C`` against the numpy reference.

    The first SPV kernel that actually exercises ``OpCooperativeMatrix
    LoadKHR`` / ``MulAddKHR`` / ``StoreKHR`` end-to-end on hardware.
    Single-workgroup, single-MMA tile — no FragForEach needed (the
    epilogue path that gemm uses); the lowerer emits the coopmat
    type once, runs one load per operand, one MulAdd, one store."""
    from quark.lower.spv import text_to_binary

    rng = np.random.default_rng(0xC00FCAFE & 0xFFFFFFFF)
    M, N, K = 8, 16, 16

    def to_bf16_bits(arr_f32):
        # Round-to-nearest-even: add 0x7FFF + (LSB of high half) before
        # the truncating shift. The numpy reference does the same.
        u32 = arr_f32.astype("<f4").view("<u4")
        return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype("<u2")

    def bf16_to_f32(u16):
        return (u16.astype("<u4") << 16).view("<f4")

    A_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    # B in storage: (N, K) row-major (the framework's gemm
    # convention). The math view is K×N, accessed as B^T-of-storage.
    Bt_f32 = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    C_f32 = rng.standard_normal((M, N)).astype(np.float32)

    A_bf16 = to_bf16_bits(A_f32)
    Bt_bf16 = to_bf16_bits(Bt_f32)
    # Round through bf16 so the GPU sees the same input precision.
    A_round = bf16_to_f32(A_bf16)
    Bt_round = bf16_to_f32(Bt_bf16)
    expected = (A_round.astype(np.float32) @ Bt_round.astype(np.float32).T
                + C_f32).astype(np.float32)

    result = SpirVLowerer(local_size=(32, 1, 1)).lower_module(
        _build_coopmat_mma_ir()
    )
    binary = text_to_binary(result.source)

    a_h, a_p = driver.allocate_buffer(A_bf16.nbytes)
    b_h, b_p = driver.allocate_buffer(Bt_bf16.nbytes)
    c_h, c_p = driver.allocate_buffer(C_f32.nbytes)
    d_h, d_p = driver.allocate_buffer(M * N * 4)
    ctypes.memmove(a_p, A_bf16.ctypes.data, A_bf16.nbytes)
    ctypes.memmove(b_p, Bt_bf16.ctypes.data, Bt_bf16.nbytes)
    ctypes.memmove(c_p, C_f32.ctypes.data, C_f32.nbytes)

    compiled = driver.compile(
        source=binary, entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (1, 1, 1), [a_h, b_h, c_h, d_h], push_bytes=b"")

    got = np.empty((M, N), dtype=np.float32)
    ctypes.memmove(got.ctypes.data, d_p, got.nbytes)
    # F32 acc with bf16 inputs: bench at moderate atol — the GPU's
    # mma path matches ``A @ B + C`` to ~bf16-input × f32-acc precision,
    # which is dominated by the bf16-rounding of the inputs.
    np.testing.assert_allclose(got, expected, rtol=1e-2, atol=1e-2)


@pytestmark_e2e
def test_for_loop_window_sum_end_to_end(driver):
    """Sliding-window sum exercising both iv-derived loads AND a carry.
    Catches the case where the OpPhi-emitted iv is mis-routed (e.g.
    confused with the next-iteration value) in the body — a windowed
    load makes that bug visible whereas an iv-free body doesn't."""
    from quark.lower.spv import text_to_binary

    n = 64
    win = 4
    rng = np.random.default_rng(0xC0FFEE_5A5A & 0xFFFFFFFF)
    X = rng.standard_normal(n + win).astype(np.float32)
    expected = np.array([X[t : t + win].sum() for t in range(n)], dtype=np.float32)

    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_for_loop_window_sum_ir(n, win=win)
    )
    binary = text_to_binary(result.source)

    x_h, x_map = driver.allocate_buffer((n + win) * 4)
    o_h, o_map = driver.allocate_buffer(n * 4)
    ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

    compiled = driver.compile(
        source=binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    driver.launch(compiled, (1, 1, 1), [x_h, o_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────
# Tier 1 — registry-survey (lower-only, no driver / no spirv-as).
# ─────────────────────────────────────────────────────────────────


def _intel_caps():
    """Synthetic Battlemage-like ``DeviceCaps`` for lower-only tests.

    Pinned to ``INTEL_GPU`` family with Xe2 chip generation so MMA
    shapes filter to the ``m8n16k16_intel_*`` cohort. Smem cap mirrors
    Battlemage's 64 KB; full byte-accurate caps don't matter here
    because the lowerer doesn't consume them — it's the kernel's
    ``is_valid_for`` that uses them, and we want the kernels to *pass*
    that check on this fake caps object."""
    from quark.device import DeviceCaps, DeviceFamily

    return DeviceCaps(
        family=DeviceFamily.INTEL_GPU,
        name="battlemage_synth",
        compute_unit_count=20,
        subgroup_width=32,
        max_threads_per_block=1024,
        max_smem_per_block=64 * 1024,
        max_regs_per_thread=0,
        max_regs_per_block=0,
        arch_tag="xe2",
        compute_capability=None,
        supports_async_copy=False,
        supports_graph_capture=False,
        supports_fp8_e4m3=False,
        supports_bf16_mma=True,
        matmul_shapes=frozenset(["m8n16k16_intel_bf16_f32",
                                 "m8n16k16_intel_f16_f32"]),
    )


# Kernels known to lower cleanly through the SPV backend on Battlemage-
# like caps using their FIRST registered problem + default config.
# This is a lock-in list — if a kernel here starts failing to lower on
# the SPV path, the test below catches the regression. Kernels NOT on
# this list either lack default-config compatibility with the Intel MMA
# shapes (need an explicit ``main_shape="m8n16k16_intel_*"``) or use
# dtypes / ops the SPV lowerer doesn't yet support; tracking them here
# would be churn.
_KNOWN_LOWERS_ON_SPV = (
    "ada_gate_residual",
    "ada_rmsnorm",
    "elementwise",
    "head_rmsnorm",
    "kv_cache_update",
    "moe_reduce",
    "moe_router",
    "moe_router_correct",
    "moe_router_shared",
    "rmsnorm",
    "silu",
    "value_residual",
    "value_residual_packed",
)


@pytest.mark.parametrize("kernel_name", _KNOWN_LOWERS_ON_SPV)
def test_known_kernel_lowers_on_spv(kernel_name):
    """Tier-1 regression net: each kernel in ``_KNOWN_LOWERS_ON_SPV``
    produces parseable SPIR-V text from its first ``problems()`` entry
    + default config. No driver, no ``spirv-as`` — pure IR-build →
    ``SpirVLowerer.lower_module`` round-trip on synthetic
    Battlemage-like caps. Locks in the platform-agnostic claim that
    the framework's ``Kernel.emit() → IR Module → SpirVLowerer`` chain
    works on these kernels with zero kernel-side edits."""
    from quark.kernels import all_kernels
    from quark.lower.legalize import legalize
    from quark.ir.module import Module

    kc = next((k for k in all_kernels()
               if getattr(k, "NAME", k.__name__) == kernel_name), None)
    if kc is None:
        pytest.skip(f"{kernel_name}: not in registry")

    problems = kc.problems()
    if not problems:
        pytest.skip(f"{kernel_name}: empty problems")
    p = problems[0]

    caps = _intel_caps()
    k = kc.from_problem(p.params)
    assert k.is_valid_for(caps), (
        f"{kernel_name}: first problem {p.name} no longer valid on Intel "
        f"caps — config-selection drift?"
    )
    ir = k.emit()
    if isinstance(ir, Module):
        legalize(ir, caps)
    lowerer = SpirVLowerer(local_size=k.block())
    result = lowerer.lower_module(ir)
    # Sanity checks on the lowered output. ``n_buffers > 0`` covers the
    # entry-point interface; the source must mention ``OpEntryPoint``
    # (every kernel emits one); ``LocalSize`` baked from ``k.block()``
    # so the kernel runs at the geometry it was authored against.
    assert result.n_buffers > 0
    assert "OpEntryPoint" in result.source
    assert "LocalSize" in result.source


# ── Additional MMA-using kernels that lower with an explicit Intel
# shape (their default ``main_shape=""`` resolves to a PTX-only shape
# via ``lookup_mma`` and fails ``is_valid_for`` on Intel caps). These
# need a small kernel-side shim — device-aware default shape — but in
# the meantime the lock-in test threads the Intel shape in via an
# explicit config.
_INTEL_SHAPE = "m8n16k16_intel_bf16_f32"


def _build_kernel_with_intel_shape(kernel_name):
    """Construct ``(kernel, kernel_cls)`` for a kernel that needs an
    explicit Intel ``main_shape`` to validate. Returns the
    instantiated kernel + class; raises ``KeyError`` for an unknown
    name. Kept as a small dispatch table here rather than per-kernel
    helpers so the test stays grouped."""
    if kernel_name == "gemm":
        from quark.kernels.gemm.kernel import GemmKernel
        from quark.kernels.gemm.config import GemmConfig
        from quark.kernels.gemm.spec import GemmSpec
        # gemm's default problems use e4m3 (fp8) which Intel doesn't
        # support; build a bf16/bf16 spec instead.
        spec = GemmSpec(
            M=128, N=128, K=128,
            a_dtype=DType.BF16, b_dtype=DType.BF16,
            acc_dtype=DType.F32, out_dtype=DType.BF16,
            compute_dtype=DType.BF16,
        )
        cfg = GemmConfig(
            BM=32, BN=32, BK=16, n_warps=1, n_stages=1,
            main_shape=_INTEL_SHAPE, impl="ws",
        )
        return GemmKernel(spec, cfg), GemmKernel
    if kernel_name == "attn":
        from quark.kernels.attn.kernel import AttnKernel
        from quark.kernels.attn.config import AttnConfig
        p = AttnKernel.problems()[0]
        k0 = AttnKernel.from_problem(p.params)
        cfg = AttnConfig(
            KvTile=32, MTiles=1, NCW=1, KvPad=8, n_stages=1,
            main_shape=_INTEL_SHAPE,
        )
        return AttnKernel(k0.spec, cfg), AttnKernel
    if kernel_name == "moe_inproj":
        from quark.kernels.moe_inproj.kernel import MoeInprojKernel
        from quark.kernels.moe_inproj.config import MoeInprojConfig
        p = MoeInprojKernel.problems()[0]
        k0 = MoeInprojKernel.from_problem(p.params)
        cfg = MoeInprojConfig(
            BM=32, BN=64, BK=16, n_warps=1, n_stages=1,
            main_shape=_INTEL_SHAPE,
        )
        return MoeInprojKernel(k0.spec, cfg), MoeInprojKernel
    if kernel_name == "moe_outproj":
        from quark.kernels.moe_outproj.kernel import MoeOutprojKernel
        from quark.kernels.moe_outproj.config import MoeOutprojConfig
        p = MoeOutprojKernel.problems()[0]
        k0 = MoeOutprojKernel.from_problem(p.params)
        cfg = MoeOutprojConfig(
            BM=32, BN=64, BK=16, n_warps=1, n_stages=1,
            main_shape=_INTEL_SHAPE,
        )
        return MoeOutprojKernel(k0.spec, cfg), MoeOutprojKernel
    if kernel_name == "patchify":
        from quark.kernels.patchify.kernel import PatchifyKernel
        from quark.kernels.patchify.config import PatchifyConfig
        p = PatchifyKernel.problems()[0]
        k0 = PatchifyKernel.from_problem(p.params)
        cfg = PatchifyConfig(
            BM=8, BN=16, BK=16, n_warps=1, n_stages=1,
            main_shape=_INTEL_SHAPE,
        )
        return PatchifyKernel(k0.spec, cfg), PatchifyKernel
    if kernel_name == "unpatchify":
        from quark.kernels.unpatchify.kernel import UnpatchifyKernel
        from quark.kernels.unpatchify.config import UnpatchifyConfig
        p = UnpatchifyKernel.problems()[0]
        k0 = UnpatchifyKernel.from_problem(p.params)
        cfg = UnpatchifyConfig(
            BM=8, BN=16, BK=16, n_warps=1, n_stages=1,
            main_shape=_INTEL_SHAPE,
        )
        return UnpatchifyKernel(k0.spec, cfg), UnpatchifyKernel
    raise KeyError(f"unknown kernel: {kernel_name}")


_MMA_KERNELS_WITH_FORCED_INTEL_SHAPE = (
    "gemm", "attn", "moe_inproj", "moe_outproj", "patchify", "unpatchify",
)


@pytest.mark.parametrize("kernel_name", _MMA_KERNELS_WITH_FORCED_INTEL_SHAPE)
def test_mma_kernel_lowers_on_spv_with_intel_shape(kernel_name):
    """Cooperative-matrix kernels that compile on Intel SPV when the
    Intel ``main_shape`` is threaded in via config. The reason they're
    not in ``_KNOWN_LOWERS_ON_SPV`` is that their default
    ``main_shape=""`` resolves to a PTX-only shape (``lookup_mma`` is
    not yet device-aware), so ``is_valid_for(intel_caps)`` rejects
    them. This test pins forward progress on the visitor surface
    (``LoadMatrixKHR``, ``MmaOp``, ``StoreMatrixKHR``, ``FragForEach``,
    ``FragApply``, ``FragReduce``, ``FragConvert``) — every visitor
    that the cooperative-matrix path needs."""
    from quark.lower.legalize import legalize
    from quark.ir.module import Module

    caps = _intel_caps()
    k, _kc = _build_kernel_with_intel_shape(kernel_name)
    assert k.is_valid_for(caps), (
        f"{kernel_name}: not valid for Intel caps even with explicit "
        f"main_shape={_INTEL_SHAPE}"
    )
    ir = k.emit()
    if isinstance(ir, Module):
        legalize(ir, caps)
    lowerer = SpirVLowerer(local_size=k.block())
    result = lowerer.lower_module(ir)
    assert result.n_buffers > 0
    assert "OpEntryPoint" in result.source
    # Every MMA kernel must declare ``CooperativeMatrixKHR`` and emit
    # the load/mma/store triple.
    assert "OpCapability CooperativeMatrixKHR" in result.source
    assert "OpCooperativeMatrixLoadKHR" in result.source
    assert "OpCooperativeMatrixMulAddKHR" in result.source
    assert "OpCooperativeMatrixStoreKHR" in result.source


def test_multi_warp_frag_visitors_partition_smem_per_warp():
    """Multi-warp kernels (n_warps > 1) need per-warp partitioned smem
    scratch in the FragForEach / FragApply / FragReduce / FragConvert
    visitors. Without that, every warp's
    ``OpCooperativeMatrixStoreKHR`` writes the SAME ``scratch[0..N]``
    range and the per-lane reads after the subgroup-scope barrier pull
    a mix of warps' data — silently corrupting any kernel that uses
    the smem-roundtrip pattern (e.g. multi-warp GEMM's epilogue).

    Lower a multi-warp GEMM and assert the partition is emitted:
    SubgroupId BuiltIn declared, ``warp_base`` SSA ids in the function
    body, and one OpIMul per Frag* op (``subgroup_id * n_per_warp``)."""
    from quark.kernels.gemm.kernel import GemmKernel
    from quark.kernels.gemm.config import GemmConfig
    from quark.kernels.gemm.spec import GemmSpec
    from quark.lower.legalize import legalize
    from quark.ir.module import Module

    caps = _intel_caps()
    spec = GemmSpec(
        M=128, N=128, K=128,
        a_dtype=DType.BF16, b_dtype=DType.BF16,
        acc_dtype=DType.F32, out_dtype=DType.BF16,
        compute_dtype=DType.BF16,
    )

    # Single-warp baseline: no warp partition needed.
    cfg1 = GemmConfig(
        BM=32, BN=32, BK=16, n_warps=1, n_stages=1,
        main_shape=_INTEL_SHAPE, impl="ws",
    )
    k1 = GemmKernel(spec, cfg1)
    ir1 = k1.emit()
    if isinstance(ir1, Module):
        legalize(ir1, caps)
    src1 = SpirVLowerer(local_size=k1.block()).lower_module(ir1).source
    # Single-warp shouldn't emit any warp_base offset.
    assert "warp_base" not in src1
    assert "smem_base" not in src1

    # Multi-warp (n_warps=4) should emit per-warp offsets in every
    # FragForEach in the epilogue.
    cfg4 = GemmConfig(
        BM=64, BN=64, BK=16, n_warps=4, n_stages=1,
        main_shape=_INTEL_SHAPE, impl="ws",
    )
    k4 = GemmKernel(spec, cfg4)
    ir4 = k4.emit()
    if isinstance(ir4, Module):
        legalize(ir4, caps)
    src4 = SpirVLowerer(local_size=k4.block()).lower_module(ir4).source
    # Per-FragForEach: ``warp_base`` (= ``subgroup_id * n_per_warp``,
    # the warp's offset into the scratch array) + ``smem_base``
    # (= ``warp_base + lane_id``, the per-lane base inside the slice).
    # Both ssa names come from the partition helper.
    assert "warp_base" in src4
    assert "smem_base" in src4
    # SubgroupId BuiltIn must be declared (used for warp partition).
    assert "BuiltIn SubgroupId" in src4


@pytest.mark.parametrize(
    "shape,a_dt,b_dt,out_dt",
    [
        ("m8n16k16_intel_bf16_f32", "BF16", "BF16", "BF16"),
        ("m8n16k16_intel_f16_f32", "F16", "F16", "F16"),
    ],
)
def test_gemm_lowers_on_intel_shapes(shape, a_dt, b_dt, out_dt):
    """GEMM lowers on every Intel cooperative-matrix shape that the
    framework supports today (bf16 and f16 inputs, both with f32
    accumulator since GemmSpec requires F32 acc). Locks in dtype
    coverage for the SPV cooperative-matrix path so a regression in
    the f16 type emit (e.g. missing capability, wrong OpTypeFloat
    width) doesn't slip past the bf16-only suite."""
    from quark.kernels.gemm.kernel import GemmKernel
    from quark.kernels.gemm.config import GemmConfig
    from quark.kernels.gemm.spec import GemmSpec
    from quark.lower.legalize import legalize
    from quark.ir.module import Module

    a = getattr(DType, a_dt)
    b = getattr(DType, b_dt)
    o = getattr(DType, out_dt)
    spec = GemmSpec(
        M=128, N=128, K=128,
        a_dtype=a, b_dtype=b, acc_dtype=DType.F32, out_dtype=o,
        compute_dtype=a,
    )
    cfg = GemmConfig(
        BM=32, BN=32, BK=16, n_warps=1, n_stages=1,
        main_shape=shape, impl="ws",
    )
    k = GemmKernel(spec, cfg)
    # Synthesize Intel caps that include both shapes so either
    # parametrisation validates.
    from quark.device import DeviceCaps, DeviceFamily

    caps = DeviceCaps(
        family=DeviceFamily.INTEL_GPU, name="battlemage_synth",
        compute_unit_count=20, subgroup_width=32,
        max_threads_per_block=1024, max_smem_per_block=64 * 1024,
        max_regs_per_thread=0, max_regs_per_block=0,
        arch_tag="xe2", compute_capability=None,
        supports_async_copy=False, supports_graph_capture=False,
        supports_fp8_e4m3=False, supports_bf16_mma=True,
        matmul_shapes=frozenset(["m8n16k16_intel_bf16_f32",
                                 "m8n16k16_intel_f16_f32"]),
    )
    assert k.is_valid_for(caps), (
        f"GEMM bf16/f16 ({shape}): not valid on Intel caps"
    )
    ir = k.emit()
    if isinstance(ir, Module):
        legalize(ir, caps)
    src = SpirVLowerer(local_size=k.block()).lower_module(ir).source
    assert "OpCapability CooperativeMatrixKHR" in src
    assert "OpCooperativeMatrixMulAddKHR" in src
    if a is DType.F16:
        # f16 input requires the Float16 capability.
        assert "OpCapability Float16" in src
    elif a is DType.BF16:
        assert "OpCapability BFloat16TypeKHR" in src
        assert "OpCapability BFloat16CooperativeMatrixKHR" in src


def test_attn_lowers_on_spv_exercises_full_visitor_surface():
    """Attention is the headline coopmat kernel — it exercises every
    visitor on the SPV path: cooperative-matrix Load/Store/MulAdd,
    FragApply (per-row scale + exp), FragReduce (online-softmax max +
    sum), FragConvert (f32 ACC → bf16 A-frag for GEMM2), control flow
    (KV-tile for-loop with carries), GLSL.std.450 ext-inst (Exp2 from
    ``ex2_approx``), and BFloat16 capability + memory model.

    This test pins the visitor-emission count for the smallest attn
    problem so regressions in any one visitor (e.g. silent skip,
    accidental short-circuit, missing capability) surface here. The
    counts come from a known-good lowering and are tied to the
    config: ``B=1, n_kv_heads=2, gqa_ratio=2, seq_len=64, kv_len=64,
    Dh=64, KvTile=32, MTiles=1, NCW=1, n_stages=1`` with the Intel
    m8n16k16 bf16/f32 shape. Update on intentional config change."""
    from quark.lower.legalize import legalize
    from quark.ir.module import Module

    caps = _intel_caps()
    k, _kc = _build_kernel_with_intel_shape("attn")
    ir = k.emit()
    if isinstance(ir, Module):
        legalize(ir, caps)
    src = SpirVLowerer(local_size=k.block()).lower_module(ir).source

    # Vulkan + KHR coopmat capabilities + memory model.
    assert "OpCapability CooperativeMatrixKHR" in src
    assert "OpCapability VulkanMemoryModel" in src
    assert "OpCapability BFloat16TypeKHR" in src
    assert "OpCapability BFloat16CooperativeMatrixKHR" in src
    assert "OpCapability GroupNonUniformArithmetic" in src
    assert "OpExtension \"SPV_KHR_cooperative_matrix\"" in src
    assert "OpExtension \"SPV_KHR_bfloat16\"" in src
    assert "OpMemoryModel Logical Vulkan" in src

    # MMA chain: GEMM1 (Q@K^T) + GEMM2 (P@V) + Q + K + V + output
    # loads, all over MTiles × n_kv_heads × KV-iters. Lower bounds
    # rather than equalities — autotune-driven config drift would
    # change exact counts but the structural floor stays.
    assert src.count("OpCooperativeMatrixLoadKHR") >= 8
    assert src.count("OpCooperativeMatrixMulAddKHR") >= 8
    assert src.count("OpCooperativeMatrixStoreKHR") >= 4

    # FragReduce: per online-softmax step we do max + sum reductions.
    # ``OpGroupNonUniformF{Max,Add}`` covers both. Floor at 2 to ensure
    # at least one max + one sum landed.
    assert src.count("OpGroupNonUniformF") >= 2

    # FragConvert: f32 ACC → bf16 A-frag for GEMM2's P-fragment build,
    # one OpFConvert per slot per acc tile. Lower bound: at least one.
    assert src.count("OpFConvert") >= 1

    # GLSL.std.450 ext-inst for ``ex2_approx`` (online softmax exp2)
    # and rcp/rsqrt approx ops. Imported once.
    assert src.count("GLSL.std.450") >= 1
    assert src.count("Exp2") >= 1

    # KV-tile for-loop: one OpLoopMerge + OpPhi for each carry across
    # KV iterations. Lower bound at one of each — there are several
    # carries (output acc, max, sum) so we expect more in practice.
    assert src.count("OpLoopMerge") >= 1
    assert src.count("OpPhi") >= 1

    # Workgroup-mem barriers: smem roundtrip pattern in FragApply /
    # FragReduce / FragConvert + cp.async loads. At least 1 must
    # appear; the actual count is much higher (~32 in practice).
    assert src.count("OpControlBarrier") >= 4

    # Multi-warp partition: attn's default config has n_warps =
    # gqa_ratio * NCW = 2 * 1 = 2, so every Frag* visitor must emit
    # the per-warp scratch partition (subgroup_id * n_per_warp). The
    # 36ef967 fix ensures FragApply / FragReduce / FragConvert all
    # honour the partition — attention's n_warps>1 IS the
    # exercise-it-end-to-end test for that fix.
    assert "BuiltIn SubgroupId" in src
    assert src.count("warp_base") >= 1, (
        "attn n_warps=2 must emit per-warp smem partition"
    )
    assert src.count("smem_base") >= 1
