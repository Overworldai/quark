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

    def test_euler_step_runs_end_to_end_against_numpy_reference(self, spv_device):
        """``EulerStepKernel`` runs end-to-end on Battlemage —
        ``Out = X + Dsig[0] * V`` per element. Multi-buffer (4 storage
        buffers) + multi-axis dispatch (the kernel uses ``block_idx
        ('y')``) + scalar f32 dsig parameter.
        """
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.euler_step.kernel import (
            EulerStepConfig,
            EulerStepKernel,
            EulerStepSpec,
            _reference,
        )
        from quark.launcher import Launcher

        N = 128
        rng = np.random.default_rng(0xFEEDBEEF)
        X = rng.standard_normal(N).astype(np.float32)
        V = rng.standard_normal(N).astype(np.float32)
        Dsig = np.array([0.1], dtype=np.float32)
        spec = EulerStepSpec(N=N, dtype=DType.F32)
        expected = _reference(spec, X=X, V=V, Dsig=Dsig)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            EulerStepKernel, spec, EulerStepConfig(n_warps=2, elems_per_block=128),
        )

        nbytes = N * 4
        x_h, x_map = _sd.allocate_buffer(nbytes)
        v_h, v_map = _sd.allocate_buffer(nbytes)
        d_h, d_map = _sd.allocate_buffer(4)
        o_h, o_map = _sd.allocate_buffer(nbytes)
        ctypes.memmove(x_map, X.ctypes.data, X.nbytes)
        ctypes.memmove(v_map, V.ctypes.data, V.nbytes)
        ctypes.memmove(d_map, Dsig.ctypes.data, Dsig.nbytes)

        ck.launch(buffers=[x_h, v_h, d_h, o_h])

        got = np.empty(N, dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)

    def test_rmsnorm_runs_end_to_end_against_numpy_reference(self, spv_device):
        """``RMSNormKernel`` runs end-to-end on Battlemage —
        ``y = x * rsqrt(mean(x², axis=-1) + eps)``. Exercises the
        full reduction surface: smem + barrier + cross-lane shuffle
        ``OpGroupNonUniformShuffleXor`` for the partial-sum tree."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.rmsnorm.kernel import RMSNormKernel
        from quark.kernels.rmsnorm.spec import RMSNormSpec
        from quark.kernels.rmsnorm.config import RMSNormConfig
        from quark.kernels.rmsnorm.reference import rmsnorm_reference_numpy
        from quark.launcher import Launcher

        B, D = 4, 64
        rng = np.random.default_rng(0xCAFEBABE)
        X = rng.standard_normal((B, D)).astype(np.float32)
        spec = RMSNormSpec(B=B, D=D, dtype=DType.F32)
        expected = rmsnorm_reference_numpy(spec, X=X)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(RMSNormKernel, spec, RMSNormConfig(n_warps=1))

        nbytes = B * D * 4
        x_h, x_map = _sd.allocate_buffer(nbytes)
        o_h, o_map = _sd.allocate_buffer(nbytes)
        ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

        ck.launch(buffers=[x_h, o_h])

        got = np.empty((B, D), dtype=np.float32).reshape(-1)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        got = got.reshape(B, D)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4)

    def test_silu_runs_end_to_end_against_numpy_reference(self, spv_device):
        """``SiLUKernel`` runs end-to-end on Battlemage and produces
        output matching the kernel's own ``silu_reference_numpy``.

        This is the platform-agnostic IR claim verified at the
        numerics level — the same kernel source produces the same
        output across CUDA / Metal / Intel SPIR-V. Compare against
        the kernel's reference function (used by the autotune
        cache's correctness gate on every backend) at fp32 ULP
        tolerance — the math goes through ``GLSL.std.450 Exp2`` on
        Vulkan, which is spec-allowed up to 4 ULP of slack.
        """
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.silu.kernel import (
            SiLUConfig,
            SiLUKernel,
            SiLUSpec,
        )
        from quark.kernels.silu.reference import silu_reference_numpy
        from quark.launcher import Launcher

        N = 128
        rng = np.random.default_rng(0xC0FFEE)
        X = rng.standard_normal(N).astype(np.float32) * 2.0  # spread for sigmoid
        expected = silu_reference_numpy(SiLUSpec(N=N, dtype=DType.F32), X=X)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            SiLUKernel,
            SiLUSpec(N=N, dtype=DType.F32),
            SiLUConfig(n_warps=2, elems_per_block=N),
        )

        # Two buffers: X (in), Out (out).
        nbytes = N * 4
        x_h, x_map = _sd.allocate_buffer(nbytes)
        o_h, o_map = _sd.allocate_buffer(nbytes)
        ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

        ck.launch(buffers=[x_h, o_h])

        got = np.empty(N, dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)

        # 1e-4 absolute matches what the framework's autotune-cache
        # correctness gate asks of f32 elementwise kernels — covers
        # GLSL.std.450 Exp2's 4-ULP slack vs PTX exp2.approx.
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4)

    def test_increment_kernel_runs_end_to_end(self, spv_device):
        """``IncrementKernel`` not just compiles — actually executes
        on Battlemage and produces the correct output. ``T[0] += 1``
        per the kernel; we initialise to 41, expect 42 back.

        Uses ``SpvDriver.allocate_buffer`` for the storage backing,
        passes the handle through ``CompiledKernel.launch``, reads
        back via the host-visible mapping. This is the closing
        framework-to-hardware loop: a kernel written for the CUDA +
        Metal backends now runs unchanged on Intel, dispatched
        through the same Launcher API.
        """
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
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

        # 1-element s32 scratch buffer, init to 41.
        handle, mapped = _sd.allocate_buffer(4)
        init = np.array([41], dtype=np.int32)
        ctypes.memmove(mapped, init.ctypes.data, init.nbytes)

        ck.launch(buffers=[handle])

        out = np.empty(1, dtype=np.int32)
        ctypes.memmove(out.ctypes.data, mapped, out.nbytes)
        assert int(out[0]) == 42

    def test_euler_step_kernel_compiles(self, spv_device):
        """``EulerStepKernel`` (in-tree, the diffusion-scheduler euler
        step that runs every denoise iter) lowers cleanly. Exercises
        the multi-buffer + multi-axis (block_idx 'y' / 'x') visitor
        surface."""
        from quark.kernels.euler_step.kernel import (
            EulerStepConfig,
            EulerStepKernel,
            EulerStepSpec,
        )
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            EulerStepKernel,
            EulerStepSpec(N=128, dtype=DType.F32),
            EulerStepConfig(n_warps=2, elems_per_block=128),
        )
        assert ck.module.handle != 0

    def test_copy_strided_kernel_compiles(self, spv_device):
        """``CopyStridedKernel`` (in-tree) lowers cleanly with a
        small enough config to fit Battlemage's 1024-thread
        workgroup ceiling. ``CopyStridedConfig`` defaults exceed
        the ceiling — the test pins a fitted config."""
        from quark.kernels.copy_strided.kernel import (
            CopyStridedConfig,
            CopyStridedKernel,
            CopyStridedSpec,
        )
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            CopyStridedKernel,
            CopyStridedSpec(N=64, dtype=DType.F32),
            CopyStridedConfig(n_warps=1, elems_per_block=32),
        )
        assert ck.module.handle != 0

    def test_rmsnorm_kernel_compiles(self, spv_device):
        """``RMSNormKernel`` (in-tree, the activations rmsnorm in
        the transformer block) lowers cleanly. Exercises the
        smem + barrier + cross-lane shuffle reduction path that
        the §3.2 v4 / v7 visitor bundles unlocked, plus the
        Vulkan-1.4 "all globals in entry-point interface" rule
        for Workgroup-class smem variables.
        """
        from quark.kernels.rmsnorm.kernel import RMSNormKernel
        from quark.kernels.rmsnorm.spec import RMSNormSpec
        from quark.kernels.rmsnorm.config import RMSNormConfig
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            RMSNormKernel,
            RMSNormSpec(B=4, D=64, dtype=DType.F32),
            RMSNormConfig(n_warps=1),
        )
        assert ck.module.handle != 0

    def test_head_rmsnorm_kernel_compiles(self, spv_device):
        """``HeadRMSNormKernel`` (per-head RMSNorm in attention) lowers
        cleanly — multi-head fused norm over the QKV-packed buffer."""
        from quark.kernels.head_rmsnorm.kernel import HeadRMSNormKernel
        from quark.kernels.head_rmsnorm.spec import HeadRMSNormSpec
        from quark.kernels.head_rmsnorm.config import HeadRMSNormConfig
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        # D_full = (n_q + 2*n_kv) * Dh = (4 + 4) * 64 = 512
        ck = launcher.compile(
            HeadRMSNormKernel,
            HeadRMSNormSpec(M=4, D_full=512, n_q_heads=4, n_kv_heads=2,
                            Dh=64, dtype=DType.F32),
            HeadRMSNormConfig(n_warps=1),
        )
        assert ck.module.handle != 0

    def test_value_residual_kernel_compiles(self, spv_device):
        """``ValueResidualKernel`` (the residual-stream value path in
        attention) lowers cleanly. 4 storage buffers, exercises the
        per-frame state-update load/store pattern."""
        from quark.kernels.value_residual.kernel import ValueResidualKernel
        from quark.kernels.value_residual.spec import ValueResidualSpec
        from quark.kernels.value_residual.config import ValueResidualConfig
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            ValueResidualKernel,
            ValueResidualSpec(N=128, dtype=DType.F32),
            ValueResidualConfig(n_warps=1, elems_per_block=128),
        )
        assert ck.module.handle != 0

    def test_silu_kernel_compiles(self, spv_device):
        """``SiLUKernel`` (in-tree, the elementwise activation that
        sits in the MLP fc1 epilogue) lowers cleanly through the
        SPIR-V launcher with zero kernel-side edits. Exercises the
        vec ops + math (``rcp_approx`` / ``ex2_approx``) +
        ``ConvertOp`` + ``ArithOp.neg`` visitor surface — about 60%
        of the elementwise kernel cohort's coverage.
        """
        from quark.kernels.silu.kernel import (
            SiLUConfig,
            SiLUKernel,
            SiLUSpec,
        )
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            SiLUKernel,
            SiLUSpec(N=128, dtype=DType.F32),
            SiLUConfig(n_warps=2, elems_per_block=128),
        )
        assert ck.module.handle != 0
        assert ck.module.n_buffers == 2  # X (in) + Out
