"""End-to-end SPIR-V tests for the Launcher integration.

Mirror of ``test_cuda_e2e.py`` — builds a small ``_VecAddKernel`` via
the IR Builder, lowers it through ``SpirVLowerer``, compiles via
``SpvDriver``, and verifies the round-trip succeeds at the launcher
level.

PORTABILITY_PLAN §3.2 → §3.7 v1: this is the "framework drives the
SPIR-V backend like any other family" smoke. If this passes, the
existing kernel-framework machinery (Kernel / Spec / Config / IR
emission / Launcher.compile) reaches the SPIR-V backend without
edits at the kernel side — the platform-agnostic IR contract works.

Skipped on non-Linux + on hosts without a Vulkan ICD or
``spirv-as``.
"""

from __future__ import annotations

import ctypes
import shutil
import sys
from dataclasses import dataclass

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("spirv-as") is None,
    reason="needs Linux + Vulkan + spirv-as",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vulkan_available():
    if sys.platform != "linux":
        return False
    from quark.drivers import spv
    return spv.is_available()


@pytest.fixture(scope="module")
def spv_device(vulkan_available):
    """Build a quark ``Device`` for the bound Vulkan physical device.

    Reuses ``SpvDriver.device`` which probes caps from the Vulkan
    runtime; the ``Launcher`` consumes ``Device`` directly without
    an extra cap-translation step.
    """
    if not vulkan_available:
        pytest.skip("no Vulkan device")
    from quark.drivers import spv
    drv = spv.SpvDriver()
    return drv.device


# ---------------------------------------------------------------------------
# Tiny vec_add kernel — mirrors the shape used in test_cuda_e2e.py.
# ---------------------------------------------------------------------------


from quark.ir import Builder, DType
from quark.ir.module import Module
from quark.ir.tensor import GlobalTensor
from quark.ir.types import BufferType
from quark.kernels.base import Kernel, KernelConfig, KernelSpec


@dataclass(frozen=True)
class _VecAddSpec(KernelSpec):
    n: int = 64


@dataclass(frozen=True)
class _VecAddConfig(KernelConfig):
    block: int = 64


class _VecAddKernel(Kernel):
    """Tiny IR-built vec_add kernel for the SPIR-V launcher e2e test.

    Same shape as ``test_cuda_e2e._VecAddKernel`` — three F32 buffer
    params, ``thread_idx`` as the index, scalar load/add/store. The
    ``block`` dimension is chosen so a single-workgroup dispatch
    covers the whole buffer (``n == block``); the kernel itself
    doesn't care about workgroup geometry.
    """

    def __init__(self, spec, config):
        self.spec = spec
        self.config = config
        self._compiled = None

    def is_valid(self) -> bool:
        return self.spec.n == self.config.block

    def smem_estimate(self) -> int:
        return 0

    def emit(self) -> Module:
        b = Builder("spv_vec_add_module")
        fn = b.begin_function("vec_add")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.F32))
        b.param("Z", BufferType(DType.F32))
        g_x = GlobalTensor(
            dtype=DType.F32, shape=(self.spec.n,), stride=(1,),
            name="X", param=fn.params[0],
        )
        g_y = GlobalTensor(
            dtype=DType.F32, shape=(self.spec.n,), stride=(1,),
            name="Y", param=fn.params[1],
        )
        g_z = GlobalTensor(
            dtype=DType.F32, shape=(self.spec.n,), stride=(1,),
            name="Z", param=fn.params[2],
        )
        tid = b.thread_idx("x")
        x_v = b.load(g_x, tid)
        y_v = b.load(g_y, tid)
        z_v = b.add(x_v, y_v)
        b.store(g_z, z_v, tid)
        b.end_function()
        return b.module

    def global_tensors(self):
        return []

    def grid(self):
        return (1, 1, 1)

    def block(self):
        return (self.config.block, 1, 1)

    def reference(self, *tensors):
        x, y, _z = tensors
        return x + y

    def flops(self):
        return self.spec.n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpirVLauncherCompile:
    def test_launcher_can_be_constructed_for_intel_gpu(self, spv_device):
        """``Launcher(device=intel_device)`` constructs without
        explosion — i.e. the ``_driver_for(INTEL_GPU)`` path resolves
        to a working ``SpvDriver`` adapter."""
        from quark.launcher import Launcher
        launcher = Launcher(device=spv_device)
        assert launcher.driver is not None
        assert launcher.device.family.value == "intel_gpu"

    def test_launcher_compile_produces_compiled_kernel(self, spv_device):
        """``Launcher.compile(_VecAddKernel, spec, config)`` runs the
        full lowering + assembly + Vulkan compile path and returns
        a ``CompiledKernel`` with a non-zero pipeline handle."""
        from quark.launcher import Launcher
        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            _VecAddKernel,
            _VecAddSpec(n=64),
            _VecAddConfig(block=64),
        )
        assert ck is not None
        # The compiled module should be the SpvCompiledModule the
        # SPIR-V backend produces (handle, n_buffers, push_size, ...).
        from quark.drivers.spv import SpvCompiledModule
        assert isinstance(ck.module, SpvCompiledModule)
        assert ck.module.handle != 0
        assert ck.module.n_buffers == 3

    def test_launcher_compile_lowers_through_framework(self, spv_device):
        """Smoke that the framework's ``Kernel.emit() → IR Module
        → SpirVLowerer → spirv-as → SpvDriver.compile`` chain works
        end-to-end on the existing in-tree machinery (no Kernel-side
        edits). This is the central platform-agnostic IR claim."""
        from quark.launcher import Launcher
        launcher = Launcher(device=spv_device)
        # Validity check: spec/config invariant.
        kernel = _VecAddKernel(_VecAddSpec(n=64), _VecAddConfig(block=64))
        assert kernel.is_valid()
        # Compile through launcher — exercises every layer of the
        # SPIR-V backend stack at once.
        ck = launcher.compile(
            _VecAddKernel,
            _VecAddSpec(n=64),
            _VecAddConfig(block=64),
        )
        # Param spec inferred from the IR function — three F32
        # buffers, no scalars.
        assert len(ck.param_spec.buffers) == 3
        for buf_spec in ck.param_spec.buffers:
            assert buf_spec.dtype == "f32"

    def test_compile_caches_result(self, spv_device):
        """Re-compiling the same (kernel_cls, spec, config) returns
        the cached ``CompiledKernel`` — same shape as CUDA / Metal."""
        from quark.launcher import Launcher
        launcher = Launcher(device=spv_device)
        ck1 = launcher.compile(
            _VecAddKernel,
            _VecAddSpec(n=64),
            _VecAddConfig(block=64),
        )
        ck2 = launcher.compile(
            _VecAddKernel,
            _VecAddSpec(n=64),
            _VecAddConfig(block=64),
        )
        assert ck1 is ck2


class TestExistingKernelsCompile:
    """Surveys: which in-tree kernel suites lower through SPIR-V?

    These tests pin "existing kernels work without edits" — the
    central platform-agnostic claim. New visitor coverage in the
    SPIR-V lowerer will cause more existing kernels to start
    passing here over time.
    """

    def test_increment_kernel_compiles(self, spv_device):
        """``IncrementKernel`` (in-tree, used elsewhere by the
        framework's auto-incrementing frame counter / loop-step
        machinery) lowers cleanly through the SPIR-V launcher with
        zero kernel-side edits. Smoke that the framework's
        load/store/arith path on an existing kernel works on Intel.
        """
        from quark.kernels.increment.kernel import (
            IncrementConfig,
            IncrementKernel,
            IncrementSpec,
        )
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            IncrementKernel,
            IncrementSpec(dtype=DType.S32),
            IncrementConfig(),
        )
        assert ck.module.handle != 0
        assert ck.module.n_buffers == 1  # single ``T`` buffer
