"""Tests for the OCL driver's compile + launch path.

Exercises a hand-written OpenCL-flavor SPIR-V kernel through the raw
OclDriver API. The launcher's compile path is NOT exercised yet —
that requires Phase 3's Intel-flavor SPIR-V emitter (the Vulkan emitter
in ``quark.lower.spv`` produces output IGC rejects, as confirmed by
the 2026-05-12 probe).

These tests validate the C-extension's compile/launch surface in
isolation, so Phase 3's lowerer work has a stable target to emit into.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="_ocl_dispatch is Linux-only (OpenCL + IGC + spirv-as)",
)


@pytest.fixture(scope="module")
def spirv_as_available():
    return shutil.which("spirv-as") is not None


@pytest.fixture(scope="module")
def ocl_available():
    from quark.drivers import ocl
    return ocl.is_available()


# OpenCL-flavor SPIR-V for a vec_add kernel:
#   kernel void main(global float *a, global float *b, global float *c) {
#       size_t i = get_global_id(0);
#       c[i] = a[i] + b[i];
#   }
# Hand-written to be IGC-compatible (Physical64 OpenCL memory model,
# Kernel capability, GlobalInvocationId builtin, etc.). Phase 3 will
# generate this kind of SPV from quark's IR.
_VEC_ADD_SPVASM = textwrap.dedent("""\
               OpCapability Addresses
               OpCapability Linkage
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int8
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "main" %a %b %c
               OpSource OpenCL_C 200000
               OpName %main "main"
               OpName %a "a"
               OpName %b "b"
               OpName %c "c"
               OpDecorate %a FuncParamAttr NoCapture
               OpDecorate %b FuncParamAttr NoCapture
               OpDecorate %c FuncParamAttr NoCapture
               OpDecorate %_gid BuiltIn GlobalInvocationId
       %void = OpTypeVoid
        %u32 = OpTypeInt 32 0
        %u64 = OpTypeInt 64 0
        %f32 = OpTypeFloat 32
       %v3u64 = OpTypeVector %u64 3
       %ptr_v3u64_in = OpTypePointer Input %v3u64
       %ptr_f32 = OpTypePointer CrossWorkgroup %f32
       %fn_type = OpTypeFunction %void %ptr_f32 %ptr_f32 %ptr_f32
       %_gid = OpVariable %ptr_v3u64_in Input
       %main = OpFunction %void None %fn_type
          %a = OpFunctionParameter %ptr_f32
          %b = OpFunctionParameter %ptr_f32
          %c = OpFunctionParameter %ptr_f32
      %entry = OpLabel
       %gidv = OpLoad %v3u64 %_gid
       %gid0 = OpCompositeExtract %u64 %gidv 0
       %a_p = OpInBoundsPtrAccessChain %ptr_f32 %a %gid0
       %b_p = OpInBoundsPtrAccessChain %ptr_f32 %b %gid0
       %c_p = OpInBoundsPtrAccessChain %ptr_f32 %c %gid0
       %a_v = OpLoad %f32 %a_p
       %b_v = OpLoad %f32 %b_p
       %sum = OpFAdd %f32 %a_v %b_v
               OpStore %c_p %sum
               OpReturn
               OpFunctionEnd
""")


_INTEL_MMA_PROBE_SPVASM = textwrap.dedent("""\
               OpCapability Addresses
               OpCapability Linkage
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int16
               OpCapability SubgroupMatrixMultiplyAccumulateINTEL
               OpExtension "SPV_INTEL_subgroup_matrix_multiply_accumulate"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "main" %a %b %c %d
               OpName %main "main"
        %void = OpTypeVoid
         %u32 = OpTypeInt 32 0
         %i16 = OpTypeInt 16 0
         %f32 = OpTypeFloat 32
        %v4i16 = OpTypeVector %i16 4
        %v8u32 = OpTypeVector %u32 8
        %v4f32 = OpTypeVector %f32 4
         %u_16 = OpConstant %u32 16
       %ptr_v4i16 = OpTypePointer CrossWorkgroup %v4i16
       %ptr_v8u32 = OpTypePointer CrossWorkgroup %v8u32
       %ptr_v4f32 = OpTypePointer CrossWorkgroup %v4f32
       %fn_type = OpTypeFunction %void %ptr_v4i16 %ptr_v8u32 %ptr_v4f32 %ptr_v4f32
        %main = OpFunction %void None %fn_type
          %a = OpFunctionParameter %ptr_v4i16
          %b = OpFunctionParameter %ptr_v8u32
          %c = OpFunctionParameter %ptr_v4f32
          %d = OpFunctionParameter %ptr_v4f32
       %entry = OpLabel
         %av = OpLoad %v4i16 %a
         %bv = OpLoad %v8u32 %b
         %cv = OpLoad %v4f32 %c
         %dv = OpSubgroupMatrixMultiplyAccumulateINTEL %v4f32 %u_16 %av %bv %cv MatrixAPackedBFloat16INTEL|MatrixBPackedBFloat16INTEL
               OpStore %d %dv
               OpReturn
               OpFunctionEnd
""")


def _assemble(spvasm: str) -> bytes:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".spvasm", delete=False) as f:
        f.write(spvasm)
        asm_path = f.name
    spv_path = asm_path.replace(".spvasm", ".spv")
    result = subprocess.run(
        ["spirv-as", "--target-env", "opencl2.0", asm_path, "-o", spv_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"spirv-as failed: {result.stderr}")
    with open(spv_path, "rb") as f:
        return f.read()


def _assemble_vec_add() -> bytes:
    """Assemble the hand-written kernel via spirv-as."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".spvasm", delete=False) as f:
        f.write(_VEC_ADD_SPVASM)
        asm_path = f.name
    spv_path = asm_path.replace(".spvasm", ".spv")
    result = subprocess.run(
        ["spirv-as", "--target-env", "opencl2.0", asm_path, "-o", spv_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"spirv-as failed: {result.stderr}")
    with open(spv_path, "rb") as f:
        return f.read()


def test_compile_vec_add_through_ocl(spirv_as_available, ocl_available):
    if not spirv_as_available:
        pytest.skip("spirv-as not on PATH")
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")

    from quark.drivers.ocl import OclDriver

    spv = _assemble_vec_add()
    assert len(spv) > 0

    drv = OclDriver()
    ck = drv.compile(
        spv, entry="main", n_buffers=3,
        subgroup_size=32, local_size=(32, 1, 1),
    )
    assert ck.handle > 0
    assert ck.n_buffers == 3


def test_launch_vec_add_correctness(spirv_as_available, ocl_available):
    """End-to-end vec_add through the OCL driver — validates compile,
    USM allocation, kernel arg binding via clSetKernelArgMemPointerINTEL,
    enqueue, and host-side readback are all correct."""
    if not spirv_as_available:
        pytest.skip("spirv-as not on PATH")
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")

    from quark.drivers.ocl import OclDriver

    drv = OclDriver()
    ck = drv.compile(
        _assemble_vec_add(), entry="main", n_buffers=3,
        subgroup_size=32, local_size=(32, 1, 1),
    )

    N = 1024
    a_h, a_ptr = drv.allocate_buffer(N * 4)
    b_h, b_ptr = drv.allocate_buffer(N * 4)
    c_h, c_ptr = drv.allocate_buffer(N * 4)

    a_np = (np.arange(N, dtype=np.float32) + 1.0).copy()
    b_np = (np.arange(N, dtype=np.float32) * 0.5).copy()
    ctypes.memmove(a_ptr, a_np.ctypes.data, N * 4)
    ctypes.memmove(b_ptr, b_np.ctypes.data, N * 4)

    drv.launch(ck, grid=(N // 32, 1, 1),
               buffers=[a_h, b_h, c_h], sync=True)

    c_np = np.empty(N, dtype=np.float32)
    ctypes.memmove(c_np.ctypes.data, c_ptr, N * 4)
    expected = a_np + b_np
    np.testing.assert_allclose(c_np, expected, rtol=0, atol=0)


def test_intel_mma_spv_compiles_through_igc(spirv_as_available, ocl_available):
    """Compile-gate probe for ``SPV_INTEL_subgroup_matrix_multiply_
    accumulate``: assembles the hand-written m8n16k16 bf16 MMA kernel
    and confirms IGC accepts it via ``clCreateProgramWithIL``. Locks
    in the per-lane fragment shapes for Battlemage subgroup_size=32
    that the Phase 3 step (4) ``_visit_mma`` emitter must produce:

        A = v4i16   (4 packed bf16, 4 elems × 32 lanes = M*K = 128)
        B = v8u32   (8 i32-class regs holding bf16 pairs)
        C = v4f32   (4 f32 acc, 4 × 32 = M*N = 128)
        K-Dim = 16
        Operands = MatrixAPackedBFloat16INTEL|MatrixBPackedBFloat16INTEL

    The shape was discovered by probe (IGC rejects v4u32 for B with
    "Matrix B argument must have 8 components for targeted HW. Actual:
    4"; v8u32 lands cleanly). This test guards against IGC syntax
    drift in advance of the visitor work."""
    if not spirv_as_available:
        pytest.skip("spirv-as not on PATH")
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")

    from quark.drivers.ocl import OclDriver

    spv = _assemble(_INTEL_MMA_PROBE_SPVASM)
    assert len(spv) > 0

    drv = OclDriver()
    ck = drv.compile(
        spv, entry="main", n_buffers=4,
        subgroup_size=32, local_size=(32, 1, 1),
    )
    assert ck.handle > 0
    assert ck.n_buffers == 4


def test_launcher_dispatch_path_recognizes_ocl_family(ocl_available):
    """Confirm the launcher's family dispatch picks up INTEL_GPU
    when given an OclDriver device, and that the registered lowerer
    resolves to the real Phase-3 emitter (``OpenClSpirVLowerer``)."""
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")

    from quark.device import DeviceFamily
    from quark.drivers.ocl import OclDriver
    from quark.launcher import Launcher
    from quark.lower import get_lowerer
    from quark.lower.ocl import OpenClSpirVLowerer

    drv = OclDriver()
    launcher = Launcher(device=drv.device)
    assert launcher.device.family is DeviceFamily.INTEL_GPU

    lowerer = get_lowerer(DeviceFamily.INTEL_GPU, drv.caps)
    assert isinstance(lowerer, OpenClSpirVLowerer)
    assert hasattr(lowerer, "lower_module")
