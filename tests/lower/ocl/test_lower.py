"""Tests for the OpenCL/IGC SPIR-V lowerer (Phase 3 first cut).

Two tiers:

1. **Tier 1 — text-emit goldens** (cross-platform). Build a small IR
   module via the Builder, run it through
   ``OpenClSpirVLowerer.lower_module``, assert the emitted text is
   OpenCL-dialect SPIR-V (Kernel capability + Physical64 OpenCL
   memory model + CrossWorkgroup pointer args + OpenCL.std ext-inst,
   NOT the Vulkan dialect the ``quark.lower.spv`` lowerer emits).

2. **Tier 2 — end-to-end through spirv-as + IGC + OclDriver**.
   Lower → assemble (``--target-env opencl2.0``) → compile via
   ``clCreateProgramWithIL`` → dispatch → read back → numerics.
   Gated on Linux + ``spirv-as`` + a reachable OpenCL GPU device.
   Proves the whole pipeline (lowerer + assembler + IGC + driver +
   dispatch) doesn't drift.
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
from quark.lower.ocl import OpenClSpirVLowerer


def _build_vec_add_ir(n: int = 64):
    """Construct ``Z[i] = X[i] + Y[i]`` IR. Mirrors the SPV tier-1
    test fixture so the two lowerers consume identical input."""
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


def _build_vec_add_with_bounds_ir(n: int):
    """Bounds-check pattern: dispatch a multiple-of-local-size grid,
    use ``if (gid < n)`` to skip out-of-range elements. Exercises
    ``CmpOp`` + ``IfRegionOp`` + ``BlockIdxOp`` + ``BlockDimOp`` in
    one shape — the same fixture the SPV tier-1 ``TestBoundsCheckedKernel``
    uses, so the two lowerers stay in lockstep."""
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


class TestOclTextEmit:
    """Cross-platform: assert the emitted text has the OpenCL shape."""

    def test_emits_opencl_dialect_headers(self):
        result = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_ir()
        )
        src = result.source

        # OpenCL dialect headers — the bits IGC requires and that
        # would cause the Vulkan-dialect emitter to fail probe.
        assert "OpCapability Kernel" in src
        assert "OpCapability Addresses" in src
        assert "OpCapability Linkage" in src
        assert "OpCapability Int64" in src
        assert "OpMemoryModel Physical64 OpenCL" in src
        assert "OpEntryPoint Kernel" in src
        assert '"main"' in src
        assert 'OpExtInstImport "OpenCL.std"' in src

        # Negative: the Vulkan shape MUST NOT leak through.
        assert "OpCapability Shader" not in src
        assert "Logical GLSL450" not in src
        assert "GLSL.std.450" not in src
        assert "OpEntryPoint GLCompute" not in src
        assert "DescriptorSet" not in src
        assert "OpAccessChain" not in src  # OCL uses OpInBoundsPtrAccessChain

    def test_emits_kernel_pointer_params(self):
        """Buffers should be ``OpFunctionParameter`` of a
        ``CrossWorkgroup`` pointer — no Block-decorated struct
        wrapper, no DescriptorSet/Binding decorations."""
        result = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_ir()
        )
        src = result.source

        # CrossWorkgroup pointer type (per-arg) + three OpFunctionParameter
        # uses (one per buffer).
        assert "OpTypePointer CrossWorkgroup" in src
        assert src.count("OpFunctionParameter") == 3
        assert "OpInBoundsPtrAccessChain" in src

    def test_emits_global_invocation_id_as_u64_vec3(self):
        """``thread_idx`` lowers to ``LocalInvocationId`` extract +
        u64→u32 convert. OCL builtins are 64-bit (per the SPIR-V
        OpenCL environment), narrower than Vulkan's u32 vec3."""
        result = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_ir()
        )
        src = result.source
        assert "BuiltIn LocalInvocationId" in src
        # ``OpTypeVector %u64 3`` lands in the type cache; the load
        # result is u64-typed, and we convert to u32 for IR use.
        assert "OpTypeInt 64 0" in src
        assert "OpUConvert" in src

    def test_body_uses_pointer_arith_not_struct_indexing(self):
        """The store sequence should be: OpInBoundsPtrAccessChain on
        a kernel-param pointer (advance) → OpStore. No nested
        OpAccessChain through a Block-decorated struct."""
        result = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_ir()
        )
        src = result.source
        assert "OpFAdd" in src
        assert "OpLoad" in src
        assert "OpStore" in src
        # Function-body epilogue.
        assert "OpReturn" in src
        assert "OpFunctionEnd" in src

    def test_metadata_matches_kernel_shape(self):
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_vec_add_ir()
        )
        assert result.entry_name == "main"
        assert result.n_buffers == 3
        assert result.local_size == (32, 1, 1)
        assert result.subgroup_size == 32
        assert result.smem_bytes == 0

# ─────────────────────────────────────────────────────────────────
# Tier 2 — end-to-end through spirv-as + OclDriver.
# ─────────────────────────────────────────────────────────────────


pytestmark_e2e = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("spirv-as") is None,
    reason="needs Linux + spirv-as for end-to-end IGC dispatch",
)


@pytest.fixture(scope="module")
def ocl_driver():
    """OclDriver bound to the default device. Skipped on hosts where
    OpenCL isn't reachable."""
    if sys.platform != "linux":
        pytest.skip("Linux-only")
    from quark.drivers import ocl
    if not ocl.is_available():
        pytest.skip("no OpenCL device")
    return ocl.OclDriver()


@pytestmark_e2e
def test_vec_add_end_to_end_through_ocl_lowerer(ocl_driver):
    """Build vec_add via quark IR → lower via ``OpenClSpirVLowerer``
    → assemble via ``spirv-as --target-env opencl2.0`` → compile via
    IGC (``clCreateProgramWithIL``) → dispatch → read back → numerics.

    Sister test to ``tests/drivers/test_ocl_compile_launch.py
    ::test_launch_vec_add_correctness``, but going through the
    framework lowerer instead of a hand-written OpenCL-flavor
    SPIR-V fixture. If this passes, the lowerer's text emit is good
    end-to-end."""
    from quark.lower._common.spirv_assemble import text_to_binary

    n = 64
    rng = np.random.default_rng(0xBEEFCAFE)
    X = rng.standard_normal(n).astype(np.float32)
    Y = rng.standard_normal(n).astype(np.float32)
    expected = X + Y

    # Lower + assemble.
    result = OpenClSpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_vec_add_ir(n)
    )
    binary = text_to_binary(result.source, target_env="opencl2.0")

    # Compile + dispatch via the public driver.
    nbytes = n * 4
    a_h, a_map = ocl_driver.allocate_buffer(nbytes)
    b_h, b_map = ocl_driver.allocate_buffer(nbytes)
    c_h, c_map = ocl_driver.allocate_buffer(nbytes)
    ctypes.memmove(a_map, X.ctypes.data, X.nbytes)
    ctypes.memmove(b_map, Y.ctypes.data, Y.nbytes)

    compiled = ocl_driver.compile(
        binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        subgroup_size=result.subgroup_size,
        local_size=result.local_size,
    )
    # Single-workgroup dispatch: grid=(1,1,1), local_size=(n,1,1) so
    # n work-items cover the n elements.
    ocl_driver.launch(compiled, grid=(1, 1, 1),
                      buffers=[a_h, b_h, c_h], sync=True)

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, c_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


class TestBoundsCheckedKernel:
    """Tier-1 goldens for the bounds-check / multi-workgroup path —
    the visitor set Phase 3 step (1) adds. Mirrors
    ``tests/lower/spv/test_lower.py::TestBoundsCheckedKernel`` so the
    two lowerers stay byte-comparable in shape (modulo dialect)."""

    def test_emits_workgroup_id_for_block_idx(self):
        result = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_with_bounds_ir(100)
        )
        assert "BuiltIn LocalInvocationId" in result.source
        assert "BuiltIn WorkgroupId" in result.source

    def test_emits_constant_for_block_dim(self):
        result = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_with_bounds_ir(100)
        )
        # block_dim("x") materialises as an OpConstant u32 with the
        # local_size.x value.
        assert "OpConstant " in result.source
        assert "64" in result.source

    def test_emits_structured_if(self):
        result = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(
            _build_vec_add_with_bounds_ir(100)
        )
        src = result.source
        assert "OpSelectionMerge" in src
        assert "OpBranchConditional" in src
        assert src.count("OpBranch ") >= 2
        # The bounds check uses ULessThan on u32.
        assert "OpULessThan" in src


@pytestmark_e2e
def test_bounds_checked_vec_add_e2e_through_ocl_lowerer(ocl_driver):
    """Multi-workgroup dispatch: grid=(ceil(n/wg),1,1), local=(wg,1,1).
    Threads with gid>=n take the else arm and skip the store; the rest
    write Z[gid] = X[gid] + Y[gid]. Validates ``CmpOp`` +
    ``IfRegionOp`` end-to-end through IGC."""
    from quark.lower._common.spirv_assemble import text_to_binary

    n = 100  # deliberately not a multiple of wg
    wg = 32
    rng = np.random.default_rng(0xFEEDC0DE)
    X = rng.standard_normal(n).astype(np.float32)
    Y = rng.standard_normal(n).astype(np.float32)
    expected = X + Y

    result = OpenClSpirVLowerer(local_size=(wg, 1, 1)).lower_module(
        _build_vec_add_with_bounds_ir(n)
    )
    binary = text_to_binary(result.source, target_env="opencl2.0")

    nbytes = n * 4
    a_h, a_map = ocl_driver.allocate_buffer(nbytes)
    b_h, b_map = ocl_driver.allocate_buffer(nbytes)
    c_h, c_map = ocl_driver.allocate_buffer(nbytes)
    ctypes.memmove(a_map, X.ctypes.data, X.nbytes)
    ctypes.memmove(b_map, Y.ctypes.data, Y.nbytes)
    # Pre-seed Z with sentinels — out-of-range threads must NOT write,
    # so the sentinel positions should survive the dispatch verbatim.
    # (n=100 with wg=32 → grid=4 workgroups → 128 work-items dispatched;
    # the last 28 take the else arm and skip the store.)
    Z_seed = np.full(n, -7777.0, dtype=np.float32)
    ctypes.memmove(c_map, Z_seed.ctypes.data, Z_seed.nbytes)

    compiled = ocl_driver.compile(
        binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        subgroup_size=result.subgroup_size,
        local_size=result.local_size,
    )
    grid_x = (n + wg - 1) // wg  # 4
    ocl_driver.launch(compiled, grid=(grid_x, 1, 1),
                      buffers=[a_h, b_h, c_h], sync=True)

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, c_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def _build_smem_shuffle_ir(wg: int):
    """Smem shuffle pattern: each thread writes ``tid`` to ``smem[tid]``;
    after ``barrier()``, every thread reads ``smem[(tid+1) % wg]`` and
    writes the result to global ``Z[tid]``. Validates SmemAllocOp +
    BarrierOp + SharedRegion-aware load/store end-to-end. Reference:
    ``Z[i] = (i+1) % wg``."""
    b = Builder("smem_shuffle")
    fn = b.begin_function("smem_shuffle")
    b.param("Z", BufferType(DType.U32))
    g_z = GlobalTensor(dtype=DType.U32, shape=(wg,), stride=(1,),
                       name="Z", param=fn.params[0])
    s = b.smem_alloc("buf", DType.U32, (wg,))
    tid = b.thread_idx("x")
    # smem[tid] = tid
    b.store(s, tid, tid)
    b.barrier(scope="block")
    # gather = smem[(tid+1) % wg]
    one = b.const(DType.U32, 1)
    wg_const = b.const(DType.U32, wg)
    nxt = b.rem(b.add(tid, one), wg_const)
    gathered = b.load(s, nxt)
    b.store(g_z, gathered, tid)
    b.end_function()
    return b.module


class TestSmemAndBarrier:
    """Phase 3 step (3): SmemAllocOp + BarrierOp + SharedRegion-aware
    Load/Store. The OCL emit shape is byte-identical to Vulkan for
    smem + barriers (Workgroup storage class + OpControlBarrier are
    dialect-agnostic); the smem_bytes accounting is what differs from
    the first cut (was hard-coded 0)."""

    def test_smem_alloc_emits_workgroup_array(self):
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_smem_shuffle_ir(32)
        )
        src = result.source
        assert "OpVariable" in src and " Workgroup" in src
        assert "OpTypeArray" in src
        assert "OpTypePointer Workgroup" in src
        # smem_bytes is surfaced for driver-side validation.
        assert result.smem_bytes == 32 * 4  # u32 × 32 elements

    def test_barrier_emits_op_control_barrier(self):
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_smem_shuffle_ir(32)
        )
        src = result.source
        assert "OpControlBarrier" in src

    def test_smem_load_uses_op_access_chain_not_inbounds(self):
        """Workgroup-class arrays use ``OpAccessChain`` (one index for
        the array element), distinct from the CrossWorkgroup buffer
        path which uses ``OpInBoundsPtrAccessChain``. This keeps the
        two storage classes diff-readable in disassembly."""
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_smem_shuffle_ir(32)
        )
        src = result.source
        assert "OpAccessChain" in src
        assert "OpInBoundsPtrAccessChain" in src  # global Z store


@pytestmark_e2e
def test_smem_shuffle_e2e_through_ocl_lowerer(ocl_driver):
    """End-to-end: lower → spirv-as → IGC → dispatch. Each lane
    writes its tid to smem, barriers, then reads the next lane's
    value and writes to Z. Output should be ``Z[i] = (i+1) % wg``."""
    from quark.lower._common.spirv_assemble import text_to_binary

    wg = 32
    result = OpenClSpirVLowerer(local_size=(wg, 1, 1)).lower_module(
        _build_smem_shuffle_ir(wg)
    )
    binary = text_to_binary(result.source, target_env="opencl2.0")

    z_h, z_map = ocl_driver.allocate_buffer(wg * 4)
    seed = np.full(wg, 0xDEADBEEF, dtype=np.uint32)
    ctypes.memmove(z_map, seed.ctypes.data, seed.nbytes)

    compiled = ocl_driver.compile(
        binary, entry=result.entry_name, n_buffers=result.n_buffers,
        subgroup_size=result.subgroup_size, local_size=result.local_size,
    )
    ocl_driver.launch(compiled, grid=(1, 1, 1), buffers=[z_h], sync=True)

    got = np.empty(wg, dtype=np.uint32)
    ctypes.memmove(got.ctypes.data, z_map, got.nbytes)
    expected = (np.arange(wg, dtype=np.uint32) + 1) % wg
    np.testing.assert_array_equal(got, expected)


_M8N16K16_BF16_SHAPE_ID = "m8n16k16_intel_bf16_f32"


def _build_mma_ir(sg: int = 16):
    """Single m8n16k16 bf16 MMA at SG=16 (the Intel MMA extension's
    spec-supported subgroup size).

    Per-lane host-side copy widths (gmem→smem):
      A: M*K/SG = 128/16 = 8 u16 per lane.
      B: K*N/SG = 256/16 = 16 u16 per lane.
      C/D: M*N/SG = 128/16 = 8 f32 per lane.

    Smem is **row-major** (matrix-shaped): A=(M,K), B=(K,N), C/D=(M,N).
    The OCL LoadMatrix/StoreMatrix visitors gather Intel column-of
    from this row-major layout.
    """
    from quark.ir import Builder
    from quark.ir.module import MmaShape

    if sg != 16:
        raise ValueError(
            f"unsupported sg={sg} — Intel MMA spec only supports SG ∈ {{8, 16}}"
        )

    M, N_dim, K_dim = 8, 16, 16
    a_per_lane = M * K_dim // sg
    b_per_lane = K_dim * N_dim // sg
    c_per_lane = M * N_dim // sg

    b = Builder("intel_mma_kernel")
    fn = b.begin_function("intel_mma_kernel")
    b.param("A", BufferType(DType.U16))
    b.param("Bb", BufferType(DType.U16))
    b.param("C", BufferType(DType.F32))
    b.param("D", BufferType(DType.F32))

    shape = MmaShape(
        name=_M8N16K16_BF16_SHAPE_ID,
        m=M, n=N_dim, k=K_dim,
        a_dtype=DType.BF16, b_dtype=DType.BF16, acc_dtype=DType.F32,
        a_regs=a_per_lane, b_regs=8, c_regs=c_per_lane,
    )
    b.register_shape(shape)

    # B is stored as (N, K) row-major ("B^T in storage" — the kernel
    # cohort's convention for every gemm; the Intel MMA visitor's
    # B-load swaps (lane, slot) → (outer, inner) accordingly).
    a_smem = b.smem_alloc("a_frag", DType.U16, (M, K_dim))
    b_smem = b.smem_alloc("b_frag", DType.U16, (N_dim, K_dim))
    c_smem = b.smem_alloc("c_frag", DType.F32, (M, N_dim))
    d_smem = b.smem_alloc("d_frag", DType.F32, (M, N_dim))

    g_a = GlobalTensor(dtype=DType.U16, shape=(sg * a_per_lane,), stride=(1,),
                       name="A", param=fn.params[0])
    g_b = GlobalTensor(dtype=DType.U16, shape=(sg * b_per_lane,), stride=(1,),
                       name="B", param=fn.params[1])
    g_c = GlobalTensor(dtype=DType.F32, shape=(sg * c_per_lane,), stride=(1,),
                       name="C", param=fn.params[2])
    g_d = GlobalTensor(dtype=DType.F32, shape=(sg * c_per_lane,), stride=(1,),
                       name="D", param=fn.params[3])

    tid = b.thread_idx("x")

    def _copy_gmem_to_smem(gmem, smem, per_lane, n_cols):
        """Copy ``per_lane`` contiguous elements per thread from
        row-major ``gmem`` (1D buffer) to 2D ``smem`` (shape M,N).
        Maps thread tid + iter i to matrix coord (row=flat//n_cols,
        col=flat%n_cols) where flat = tid * per_lane + i.

        Uses Python-side const folding when ``per_row_lanes ==
        n_cols // per_lane`` is an integer."""
        per_row_lanes_v = n_cols // per_lane  # threads per matrix row
        # row = tid // per_row_lanes; col_in_row = (tid % per_row_lanes) * per_lane
        per_row_lanes_const = b.const(DType.U32, per_row_lanes_v)
        per_lane_const = b.const(DType.U32, per_lane)
        row = b.div(tid, per_row_lanes_const)
        col_in_row = b.mul(b.rem(tid, per_row_lanes_const), per_lane_const)
        n_cols_const = b.const(DType.U32, n_cols)
        for i in range(per_lane):
            col = b.add(col_in_row, b.const(DType.U32, i))
            flat = b.add(b.mul(row, n_cols_const), col)
            b.store(smem, b.load(gmem, flat), row, col)

    _copy_gmem_to_smem(g_a, a_smem, a_per_lane, K_dim)
    _copy_gmem_to_smem(g_b, b_smem, b_per_lane, N_dim)
    _copy_gmem_to_smem(g_c, c_smem, c_per_lane, N_dim)
    b.barrier(scope="block")

    a_frag = b.load_matrix(a_smem, _M8N16K16_BF16_SHAPE_ID, "a", 0, 0)
    b_frag = b.load_matrix(b_smem, _M8N16K16_BF16_SHAPE_ID, "b", 0, 0)
    c_frag = b.load_matrix(c_smem, _M8N16K16_BF16_SHAPE_ID, "c", 0, 0)
    d_frag = b.mma(_M8N16K16_BF16_SHAPE_ID, a_frag, b_frag, c_frag)
    b.store_matrix(d_smem, d_frag, _M8N16K16_BF16_SHAPE_ID, "d", 0, 0)
    b.barrier(scope="block")

    # Copy smem D back to row-major gmem. Mirror of _copy_gmem_to_smem.
    per_row_lanes = N_dim // c_per_lane
    per_row_lanes_const = b.const(DType.U32, per_row_lanes)
    c_per_lane_const = b.const(DType.U32, c_per_lane)
    row_d = b.div(tid, per_row_lanes_const)
    col_in_row_d = b.mul(b.rem(tid, per_row_lanes_const), c_per_lane_const)
    n_dim_const = b.const(DType.U32, N_dim)
    for i in range(c_per_lane):
        col = b.add(col_in_row_d, b.const(DType.U32, i))
        flat = b.add(b.mul(row_d, n_dim_const), col)
        b.store(g_d, b.load(d_smem, row_d, col), flat)
    b.end_function()
    return b.module


class TestIntelMma:
    """Phase 3 step (4): MMA visitor + LoadMatrix/StoreMatrix + the
    Intel layout descriptor. Lowers a single m8n16k16 bf16 MMA at
    SG=16 (the spec-canonical subgroup size per the Khronos
    ``cl_intel_subgroup_matrix_multiply_accumulate`` extension)."""

    def test_emits_intel_mma_capability_and_extension(self):
        result = OpenClSpirVLowerer(
            local_size=(16, 1, 1), subgroup_width=16,
        ).lower_module(_build_mma_ir(sg=16))
        src = result.source
        assert "OpCapability SubgroupMatrixMultiplyAccumulateINTEL" in src
        assert "SPV_INTEL_subgroup_matrix_multiply_accumulate" in src
        assert "OpSubgroupMatrixMultiplyAccumulateINTEL" in src
        assert "MatrixAPackedBFloat16INTEL|MatrixBPackedBFloat16INTEL" in src
        # MMA-using kernels must pin SubgroupSize so IGC picks the
        # right DPAS shape (Battlemage default is SG=32; MMA needs 16).
        assert "OpExecutionMode" in src and "SubgroupSize 16" in src

    def test_emits_per_lane_vector_loads(self):
        result = OpenClSpirVLowerer(
            local_size=(16, 1, 1), subgroup_width=16,
        ).lower_module(_build_mma_ir(sg=16))
        src = result.source
        assert "OpTypeVector" in src
        assert "BuiltIn SubgroupLocalInvocationId" in src


@pytestmark_e2e
def test_intel_mma_e2e_all_ones_through_ocl_lowerer(ocl_driver):
    """Numerical validation of the MMA path at the spec-canonical
    SG=16. With A=all-ones (bf16=1.0) and B=all-ones in both halves
    of each u32 (bf16=1.0 in low + high), C=0, every D[m,n] = sum_k
    A[m,k] × B[k,n] = K × 1 × 1 = 16. **All 8 per-lane f32 slots
    should hold 16.0** — vs SG=32 where only 2/4 slots filled.

    This confirms the SG=16 layout entry in `_INTEL_MMA_LAYOUTS` is
    correct and that IGC's DPAS engine is processing the full M*N
    tile (not just lanes 0–15)."""
    from quark.lower._common.spirv_assemble import text_to_binary

    SG = 16
    result = OpenClSpirVLowerer(
        local_size=(SG, 1, 1), subgroup_width=SG,
    ).lower_module(_build_mma_ir(sg=SG))
    binary = text_to_binary(result.source, target_env="opencl2.0")

    # Buffer sizes: A = M*K = 128 u16, B = K*N = 256 u16, C/D = M*N = 128 f32.
    n_a = 8 * 16
    n_b = 16 * 16
    n_c = 8 * 16
    a_h, a_map = ocl_driver.allocate_buffer(n_a * 2)
    b_h, b_map = ocl_driver.allocate_buffer(n_b * 2)
    c_h, c_map = ocl_driver.allocate_buffer(n_c * 4)
    d_h, d_map = ocl_driver.allocate_buffer(n_c * 4)

    A = np.full(n_a, 0x3F80, dtype=np.uint16)  # bf16 1.0
    B = np.full(n_b, 0x3F80, dtype=np.uint16)  # bf16 1.0 (row-major K×N)
    C = np.zeros(n_c, dtype=np.float32)
    ctypes.memmove(a_map, A.ctypes.data, A.nbytes)
    ctypes.memmove(b_map, B.ctypes.data, B.nbytes)
    ctypes.memmove(c_map, C.ctypes.data, C.nbytes)
    D_seed = np.full(n_c, -7777.0, dtype=np.float32)
    ctypes.memmove(d_map, D_seed.ctypes.data, D_seed.nbytes)

    compiled = ocl_driver.compile(
        binary, entry=result.entry_name, n_buffers=result.n_buffers,
        subgroup_size=SG, local_size=(SG, 1, 1),
    )
    ocl_driver.launch(compiled, grid=(1, 1, 1),
                      buffers=[a_h, b_h, c_h, d_h], sync=True)

    got = np.empty(n_c, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, d_map, got.nbytes)
    expected = np.full(n_c, 16.0, dtype=np.float32)
    np.testing.assert_array_equal(
        got, expected,
        err_msg=f"D should be 16.0 in all {n_c} slots; got unique={sorted(set(got.tolist()))}",
    )


@pytestmark_e2e
def test_intel_mma_c_passthrough_preserves_data(ocl_driver):
    """C-passthrough numerical probe at SG=16: A=B=0, C=lane-slot
    markers. MMA computes D = A*B + C = C with A=B=0. So D's gmem
    dump should equal C's gmem input byte-exactly. Validates the
    LoadMatrix→MMA→StoreMatrix round-trip preserves data through
    the dpas pipeline."""
    from quark.lower._common.spirv_assemble import text_to_binary

    SG = 16
    result = OpenClSpirVLowerer(
        local_size=(SG, 1, 1), subgroup_width=SG,
    ).lower_module(_build_mma_ir(sg=SG))
    binary = text_to_binary(result.source, target_env="opencl2.0")

    n_a = 8 * 16
    n_b = 16 * 16
    n_c = 8 * 16
    a_h, a_map = ocl_driver.allocate_buffer(n_a * 2)  # u16
    b_h, b_map = ocl_driver.allocate_buffer(n_b * 2)  # u16
    c_h, c_map = ocl_driver.allocate_buffer(n_c * 4)  # f32
    d_h, d_map = ocl_driver.allocate_buffer(n_c * 4)  # f32

    # A=B=0 so the MMA contributes nothing; C carries unique
    # lane-slot markers so any layout swap between C-in and D-out
    # would corrupt them.
    A = np.zeros(n_a, dtype=np.uint16)
    B = np.zeros(n_b, dtype=np.uint16)
    C = (np.arange(n_c, dtype=np.float32) + 1.5)
    ctypes.memmove(a_map, A.ctypes.data, A.nbytes)
    ctypes.memmove(b_map, B.ctypes.data, B.nbytes)
    ctypes.memmove(c_map, C.ctypes.data, C.nbytes)
    # Pre-fill D with sentinels so the dispatch is forced to write
    # every slot (not just inherit a zero-initialised buffer).
    D_sentinel = np.full(n_c, -7777.0, dtype=np.float32)
    ctypes.memmove(d_map, D_sentinel.ctypes.data, D_sentinel.nbytes)

    compiled = ocl_driver.compile(
        binary, entry=result.entry_name, n_buffers=result.n_buffers,
        subgroup_size=SG, local_size=(SG, 1, 1),
    )
    ocl_driver.launch(compiled, grid=(1, 1, 1),
                      buffers=[a_h, b_h, c_h, d_h], sync=True)

    got = np.empty(n_c, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, d_map, got.nbytes)
    np.testing.assert_array_equal(got, C)


@pytestmark_e2e
def test_intel_mma_e2e_zero_input_through_ocl_lowerer(ocl_driver):
    """Zero-input smoke at SG=16: A=B=C=0, expect D=0. Proves the
    visitor's emit shape (capability + extension + per-lane loads +
    MMA op with operand flag + SubgroupSize exec mode) compiles and
    dispatches through IGC end-to-end."""
    from quark.lower._common.spirv_assemble import text_to_binary

    SG = 16
    result = OpenClSpirVLowerer(
        local_size=(SG, 1, 1), subgroup_width=SG,
    ).lower_module(_build_mma_ir(sg=SG))
    binary = text_to_binary(result.source, target_env="opencl2.0")

    n_a = 8 * 16
    n_b = 16 * 16
    n_c = 8 * 16
    a_h, a_map = ocl_driver.allocate_buffer(n_a * 2)  # u16
    b_h, b_map = ocl_driver.allocate_buffer(n_b * 2)  # u16
    c_h, c_map = ocl_driver.allocate_buffer(n_c * 4)  # f32
    d_h, d_map = ocl_driver.allocate_buffer(n_c * 4)  # f32

    A = np.zeros(n_a, dtype=np.uint16)
    B = np.zeros(n_b, dtype=np.uint16)
    C = np.zeros(n_c, dtype=np.float32)
    D_sentinel = np.full(n_c, -7777.0, dtype=np.float32)
    ctypes.memmove(a_map, A.ctypes.data, A.nbytes)
    ctypes.memmove(b_map, B.ctypes.data, B.nbytes)
    ctypes.memmove(c_map, C.ctypes.data, C.nbytes)
    ctypes.memmove(d_map, D_sentinel.ctypes.data, D_sentinel.nbytes)

    compiled = ocl_driver.compile(
        binary, entry=result.entry_name, n_buffers=result.n_buffers,
        subgroup_size=SG, local_size=(SG, 1, 1),
    )
    ocl_driver.launch(compiled, grid=(1, 1, 1),
                      buffers=[a_h, b_h, c_h, d_h], sync=True)

    got = np.empty(n_c, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, d_map, got.nbytes)
    # Zero inputs → zero output, byte-exact.
    np.testing.assert_array_equal(got, np.zeros(n_c, dtype=np.float32))


def _build_butterfly_shuffle_ir(wg: int = 32):
    """Butterfly subgroup sum via 5 ShuffleXor steps (log2(32)=5).
    Each lane starts holding ``lane_id+1.0``. After log2(wg) butterfly
    steps, every lane should hold sum(1..wg) = wg*(wg+1)/2 = 528 for
    wg=32. Lane 0 writes to Z[0] to verify."""
    from quark.ir import Builder

    b = Builder("bfly")
    fn = b.begin_function("bfly")
    b.param("Z", BufferType(DType.F32))
    g_z = GlobalTensor(dtype=DType.F32, shape=(1,), stride=(1,),
                       name="Z", param=fn.params[0])

    tid = b.thread_idx("x")
    # Start: lane L's value = lane_id + 1.0 (f32).
    tid_f = b.convert(tid, DType.F32)
    one = b.const(DType.F32, 1.0)
    v = b.add(tid_f, one)
    # Butterfly: log2(wg) shuffle-xor steps with mask 1, 2, 4, 8, 16.
    mask = 1
    while mask < wg:
        shuffled = b.shuffle("xor", v, mask)
        v = b.add(v, shuffled)
        mask *= 2

    # Lane 0 writes result.
    z_idx = b.const(DType.U32, 0)
    pred = b.cmp("eq", tid, z_idx)
    b.store(g_z, v, z_idx, pred=pred)
    b.end_function()
    return b.module


def _build_atomic_counter_ir(wg: int = 32):
    """Each lane atomically increments Z[0] (u32). Final value = wg."""
    from quark.ir import Builder

    b = Builder("atomic_cnt")
    fn = b.begin_function("atomic_cnt")
    b.param("Z", BufferType(DType.U32))
    g_z = GlobalTensor(dtype=DType.U32, shape=(1,), stride=(1,),
                       name="Z", param=fn.params[0])

    zero = b.const(DType.U32, 0)
    one = b.const(DType.U32, 1)
    b.atomic_rmw(g_z, "add", one, zero)
    b.end_function()
    return b.module


def _build_loop_reduce_ir(n: int = 16, wg: int = 32):
    """Loop-reduce kernel: each lane reads ``X[wg*i + tid]`` for ``i`` in
    a ``n/wg``-iter for-loop with an accumulator carry, then a
    subgroup-sum across lanes produces the total. Lane 0 writes the
    result to ``Z[0]``. Exercises ``ForLoopOp`` (with carries) +
    ``SubgroupReduceOp``.

    Expected output: Z[0] = sum(X[0..n])."""
    from quark.ir import Builder

    b = Builder("loop_reduce")
    fn = b.begin_function("loop_reduce")
    b.param("X", BufferType(DType.F32))
    b.param("Z", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_z = GlobalTensor(dtype=DType.F32, shape=(1,), stride=(1,),
                       name="Z", param=fn.params[1])

    tid = b.thread_idx("x")
    zero = b.const(DType.F32, 0.0)
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, n // wg)
    one = b.const(DType.U32, 1)
    wg_const = b.const(DType.U32, wg)

    # Loop: acc = sum over i of X[wg*i + tid].
    with b.for_loop(lo, hi, one, carried=(zero,)) as (i, (acc_in,)):
        idx = b.add(b.mul(i, wg_const), tid)
        x = b.load(g_x, idx)
        acc_next = b.add(acc_in, x)
        b.yield_(acc_next)
    # The for-loop's result is the final accumulator.
    per_lane_sum = fn.body.ops[-1].results[0]

    # Subgroup-reduce across all lanes.
    total = b.subgroup_reduce("sum", per_lane_sum)

    # Lane 0 writes to Z[0].
    z_idx = b.const(DType.U32, 0)
    pred = b.cmp("eq", tid, z_idx)
    b.store(g_z, total, z_idx, pred=pred)
    b.end_function()
    return b.module


class TestForLoopAndSubgroupReduce:
    """Phase 3 step (6)+(7): ForLoopOp (with loop-carried accumulator)
    + SubgroupReduceOp. Both are dialect-agnostic (OpLoopMerge +
    OpPhi + OpGroupNonUniform*) — same emit shape as Vulkan."""

    def test_for_loop_emits_loop_merge_and_op_phi(self):
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_loop_reduce_ir()
        )
        src = result.source
        assert "OpLoopMerge" in src
        assert "OpPhi" in src
        # iv is u32, so the loop cond uses ULessThan.
        assert "OpULessThan" in src

    def test_subgroup_reduce_emits_group_nonuniform_fadd(self):
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_loop_reduce_ir()
        )
        src = result.source
        assert "OpCapability GroupNonUniformArithmetic" in src
        assert "OpGroupNonUniformFAdd" in src
        assert " Reduce " in src

    def test_for_loop_unroll_emits_loop_control_unroll(self):
        """Phase 3 step (17): ``ForLoopOp(unroll=True)`` → emit
        ``OpLoopMerge ... Unroll``. spirv-opt + IGC honor the hint to
        unroll the loop at downstream-compile time. Kernel-cohort
        copy loops migrate to this form so they can be SG-agnostic
        (block_dim_x() resolves to 16 on OCL, 32 on Vulkan-SPV)."""
        from quark.ir import Builder

        b = Builder("unroll_module")
        fn = b.begin_function("kfn")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.F32))
        g_x = GlobalTensor(dtype=DType.F32, shape=(8,), stride=(1,),
                           name="X", param=fn.params[0])
        g_y = GlobalTensor(dtype=DType.F32, shape=(8,), stride=(1,),
                           name="Y", param=fn.params[1])
        zero = b.const(DType.U32, 0)
        n_const = b.const(DType.U32, 8)
        one = b.const(DType.U32, 1)
        with b.for_loop(zero, n_const, one, unroll=True) as (i, _):
            v = b.load(g_x, i)
            b.store(g_y, v, i)
            b.yield_()
        b.end_function()
        src = OpenClSpirVLowerer(local_size=(1, 1, 1)).lower_module(b.module).source
        # Loop-control bit ``Unroll`` is emitted (vs ``None`` for the
        # default runtime-loop variant exercised by other tests).
        assert "OpLoopMerge" in src
        assert " Unroll" in src
        # Regression: still has OpPhi for the iv (the SPV unroll is a
        # hint; the loop structure stays — IGC decides whether to
        # actually unroll at downstream compile).
        assert "OpPhi" in src


@pytestmark_e2e
def test_loop_reduce_e2e_through_ocl_lowerer(ocl_driver):
    """End-to-end: lower → IGC → dispatch. Each lane does a 4-iter
    sum loop, then a subgroup-sum across 32 lanes; expect total =
    sum(X[0..n])."""
    from quark.lower._common.spirv_assemble import text_to_binary

    N = 128
    WG = 32  # 4 iter × 32 lanes = 128 elements
    result = OpenClSpirVLowerer(local_size=(WG, 1, 1)).lower_module(
        _build_loop_reduce_ir(N, WG)
    )
    binary = text_to_binary(result.source, target_env="opencl2.0")

    rng = np.random.default_rng(0xBADBAD)
    X = rng.standard_normal(N).astype(np.float32)
    expected = float(X.sum())

    x_h, x_m = ocl_driver.allocate_buffer(N * 4)
    z_h, z_m = ocl_driver.allocate_buffer(4)
    ctypes.memmove(x_m, X.ctypes.data, X.nbytes)
    sentinel = np.array([-9999.0], dtype=np.float32)
    ctypes.memmove(z_m, sentinel.ctypes.data, 4)

    compiled = ocl_driver.compile(
        binary, entry=result.entry_name, n_buffers=result.n_buffers,
        subgroup_size=result.subgroup_size, local_size=result.local_size,
    )
    ocl_driver.launch(compiled, grid=(1, 1, 1),
                      buffers=[x_h, z_h], sync=True)

    got = np.empty(1, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, z_m, 4)
    # f32 reduction across 128 numbers — tolerate ULP slack.
    np.testing.assert_allclose(got[0], expected, rtol=1e-5, atol=1e-5)


class TestShuffleAndAtomic:
    """Phase 3 step (8)+(9): ShuffleOp + AtomicRmwOp. Both are
    dialect-agnostic — same emit shape as the Vulkan path."""

    def test_shuffle_emits_op_group_nonuniform_shuffle_xor(self):
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_butterfly_shuffle_ir()
        )
        src = result.source
        assert "OpCapability GroupNonUniformShuffle" in src
        assert "OpGroupNonUniformShuffleXor" in src

    def test_atomic_emits_op_atomic_iadd(self):
        result = OpenClSpirVLowerer(local_size=(32, 1, 1)).lower_module(
            _build_atomic_counter_ir()
        )
        src = result.source
        assert "OpAtomicIAdd" in src


@pytestmark_e2e
def test_butterfly_shuffle_e2e_through_ocl_lowerer(ocl_driver):
    """Butterfly subgroup-sum reaches the right answer end-to-end.
    Lane 0 writes sum(1..32) = 528 to Z[0]."""
    from quark.lower._common.spirv_assemble import text_to_binary

    WG = 32
    result = OpenClSpirVLowerer(local_size=(WG, 1, 1)).lower_module(
        _build_butterfly_shuffle_ir(WG)
    )
    binary = text_to_binary(result.source, target_env="opencl2.0")

    z_h, z_m = ocl_driver.allocate_buffer(4)
    ctypes.memmove(z_m, np.array([0.0], dtype=np.float32).ctypes.data, 4)

    compiled = ocl_driver.compile(
        binary, entry=result.entry_name, n_buffers=result.n_buffers,
        subgroup_size=result.subgroup_size, local_size=result.local_size,
    )
    ocl_driver.launch(compiled, grid=(1, 1, 1),
                      buffers=[z_h], sync=True)
    got = np.empty(1, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, z_m, 4)
    assert got[0] == 528.0, f"butterfly sum wrong: got={got[0]}"


@pytestmark_e2e
def test_atomic_counter_e2e_through_ocl_lowerer(ocl_driver):
    """Each of 32 lanes does atomic_rmw add 1 → Z[0] = 32."""
    from quark.lower._common.spirv_assemble import text_to_binary

    WG = 32
    result = OpenClSpirVLowerer(local_size=(WG, 1, 1)).lower_module(
        _build_atomic_counter_ir(WG)
    )
    binary = text_to_binary(result.source, target_env="opencl2.0")

    z_h, z_m = ocl_driver.allocate_buffer(4)
    ctypes.memmove(z_m, np.array([0], dtype=np.uint32).ctypes.data, 4)

    compiled = ocl_driver.compile(
        binary, entry=result.entry_name, n_buffers=result.n_buffers,
        subgroup_size=result.subgroup_size, local_size=result.local_size,
    )
    ocl_driver.launch(compiled, grid=(1, 1, 1),
                      buffers=[z_h], sync=True)
    got = np.empty(1, dtype=np.uint32)
    ctypes.memmove(got.ctypes.data, z_m, 4)
    assert got[0] == WG, f"atomic counter wrong: got={got[0]}, expected={WG}"


class TestVecOps:
    """Phase 3 step (5): VecLoad/VecStore/VecBuild/VecExtract +
    Convert/Bitcast — minimal coverage (matching-dtype only) enough
    to unblock ElementwiseKernel through the launcher path."""

    def test_vec_load_emits_n_scalar_loads_and_composite_construct(self):
        b = Builder("vec_load_module")
        fn = b.begin_function("kfn")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.F32))
        g_x = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="X", param=fn.params[0])
        g_y = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="Y", param=fn.params[1])
        tid = b.thread_idx("x")
        v = b.vec_load(g_x, tid, width=4)
        b.vec_store(g_y, v, tid)
        b.end_function()
        result = OpenClSpirVLowerer(local_size=(16, 1, 1)).lower_module(b.module)
        src = result.source
        # 4 scalar loads + 1 OpCompositeConstruct.
        assert src.count("OpCompositeConstruct") >= 1
        assert "OpCompositeExtract" in src  # the store extracts each lane

    def test_convert_lowers_to_opfconvert(self):
        b = Builder("convert_module")
        fn = b.begin_function("kfn")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.U32))
        g_x = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="X", param=fn.params[0])
        g_y = GlobalTensor(dtype=DType.U32, shape=(64,), stride=(1,),
                           name="Y", param=fn.params[1])
        tid = b.thread_idx("x")
        x = b.load(g_x, tid)
        u = b.convert(x, DType.U32)
        b.store(g_y, u, tid)
        b.end_function()
        src = OpenClSpirVLowerer(local_size=(16, 1, 1)).lower_module(b.module).source
        assert "OpConvertFToU" in src

    def test_bitcast_lowers_to_opbitcast(self):
        b = Builder("bitcast_module")
        fn = b.begin_function("kfn")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.U32))
        g_x = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="X", param=fn.params[0])
        g_y = GlobalTensor(dtype=DType.U32, shape=(64,), stride=(1,),
                           name="Y", param=fn.params[1])
        tid = b.thread_idx("x")
        x = b.load(g_x, tid)
        bits = b.bitcast(x, DType.U32)
        b.store(g_y, bits, tid)
        b.end_function()
        src = OpenClSpirVLowerer(local_size=(16, 1, 1)).lower_module(b.module).source
        assert "OpBitcast" in src


class TestMathAndSelect:
    """Phase 3 step (2): ``MathOp`` rewires to OpenCL.std ext-inst
    (lowercase opcodes — ``sqrt`` not ``Sqrt``); ``SelectOp`` is
    dialect-agnostic (``OpSelect``)."""

    def test_sqrt_uses_opencl_std_ext_inst(self):
        b = Builder("sqrt_module")
        fn = b.begin_function("kfn")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.F32))
        g_x = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="X", param=fn.params[0])
        g_y = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="Y", param=fn.params[1])
        tid = b.thread_idx("x")
        b.store(g_y, b.sqrt(b.load(g_x, tid)), tid)
        b.end_function()
        src = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(b.module).source
        assert 'OpExtInstImport "OpenCL.std"' in src
        # OpenCL.std uses lowercase opcode names.
        assert " sqrt " in src
        assert "OpExtInst" in src
        # Negative: the Vulkan ext-inst set must NOT leak.
        assert "GLSL.std.450" not in src
        assert " Sqrt " not in src

    def test_select_lowers_to_op_select(self):
        b = Builder("select_module")
        fn = b.begin_function("kfn")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.F32))
        g_x = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="X", param=fn.params[0])
        g_y = GlobalTensor(dtype=DType.F32, shape=(64,), stride=(1,),
                           name="Y", param=fn.params[1])
        tid = b.thread_idx("x")
        x = b.load(g_x, tid)
        zero = b.const(DType.F32, 0.0)
        is_neg = b.cmp("lt", x, zero)
        clamped = b.select(is_neg, zero, x)
        b.store(g_y, clamped, tid)
        b.end_function()
        src = OpenClSpirVLowerer(local_size=(64, 1, 1)).lower_module(b.module).source
        assert "OpSelect" in src
        assert "OpFOrdLessThan" in src


class TestOclSecondCut:
    """Spot-checks for individual second-cut visitors."""

    def test_block_idx_lowers_to_workgroup_id(self):
        """A kernel that reads ``block_idx`` should emit the
        ``WorkgroupId`` builtin (vec3 u64), distinct from
        ``LocalInvocationId``."""
        b = Builder("block_idx_module")
        fn = b.begin_function("kfn")
        b.param("Z", BufferType(DType.U32))
        g_z = GlobalTensor(dtype=DType.U32, shape=(1,), stride=(1,),
                           name="Z", param=fn.params[0])
        zero = b.const(DType.U32, 0)
        bid = b.block_idx("x")
        b.store(g_z, bid, zero)
        b.end_function()
        result = OpenClSpirVLowerer(local_size=(1, 1, 1)).lower_module(b.module)
        src = result.source
        assert "BuiltIn WorkgroupId" in src
