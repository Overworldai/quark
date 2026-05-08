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

    def test_ada_rmsnorm_kernel_compiles(self, spv_device):
        """``AdaRMSNormKernel`` — adaptive RMSNorm with per-group affine.
        Uses ForLoopOp over chunks, smem reduction, broadcast scale —
        the first kernel through the SPV backend that needs the for-loop
        + carries visitor wiring."""
        from quark.kernels.ada_rmsnorm.kernel import AdaRMSNormKernel
        from quark.kernels.ada_rmsnorm.spec import AdaRMSNormSpec
        from quark.kernels.ada_rmsnorm.config import AdaRMSNormConfig
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            AdaRMSNormKernel,
            AdaRMSNormSpec(G=1, M=64, D=128, dtype=DType.F32),
            AdaRMSNormConfig(n_warps=1, chunk_D=128),
        )
        assert ck.module.handle != 0

    def test_ada_gate_residual_kernel_compiles(self, spv_device):
        """``AdaGateResidualKernel`` — chunk-loop gated residual stream.
        Exercises both ``direct=True`` and ``direct=False`` config paths
        (the `False` path adds an extra indirection lookup)."""
        from quark.kernels.ada_gate_residual.kernel import AdaGateResidualKernel
        from quark.kernels.ada_gate_residual.spec import AdaGateResidualSpec
        from quark.kernels.ada_gate_residual.config import AdaGateResidualConfig
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        for direct in (True, False):
            ck = launcher.compile(
                AdaGateResidualKernel,
                AdaGateResidualSpec(G=1, M=64, D=128, dtype=DType.F32),
                AdaGateResidualConfig(n_warps=1, chunk_D=128, direct=direct),
            )
            assert ck.module.handle != 0

    def test_ada_gate_residual_bf16_runs_end_to_end_against_numpy_reference(
        self, spv_device,
    ):
        """``AdaGateResidualKernel`` runs E2E natively at bf16 — exercises
        the ``packed_b32`` path (``vec_extract`` + ``vec_build`` with the
        ``packed_b32=True`` attr) for ``fma_bf16x2``-style packed compute.

        Output is bf16-bit-identical to the numpy reference (no rounding
        slack) — the kernel emits the same fused-multiply-add chain,
        the numpy reference upconverts to f32 for the math then rounds
        back to bf16. The bit-identity confirms the packed b32 visitor
        chain (split → bitcast → merge) and the wide-vec
        ``OpTypeArray`` fallback both round-trip correctly."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.ada_gate_residual.kernel import (
            AdaGateResidualKernel,
        )
        from quark.kernels.ada_gate_residual.spec import AdaGateResidualSpec
        from quark.kernels.ada_gate_residual.config import (
            AdaGateResidualConfig,
        )
        from quark.kernels.ada_gate_residual.reference import (
            ada_gate_residual_reference_numpy,
        )
        from quark.launcher import Launcher

        spec = AdaGateResidualSpec(G=1, M=64, D=256, dtype=DType.BF16)
        cfg = AdaGateResidualConfig(n_warps=1, chunk_D=256, direct=True)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(AdaGateResidualKernel, spec, cfg)

        tensors = AdaGateResidualKernel.make_tensors_numpy(
            {"G": 1, "M": 64, "D": 256, "dtype": DType.BF16}
        )
        ref = ada_gate_residual_reference_numpy(
            spec, X=tensors["X"], Y=tensors["Y"], gate=tensors["gate"]
        )

        handles: list[int] = []
        out_ptr = 0
        for buf in ck.param_spec.buffers:
            arr = tensors[buf.name]
            h, ptr = _sd.allocate_buffer(arr.nbytes)
            if buf.name == "Out":
                out_ptr = ptr
            else:
                ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)
            handles.append(h)

        ck.launch(buffers=handles)

        got = np.empty(ref.shape, dtype=np.uint16)
        ctypes.memmove(got.ctypes.data, out_ptr, got.nbytes)
        # Bit-identical comparison — bf16 storage is uint16, the
        # math chain is the same on both sides.
        np.testing.assert_array_equal(got, ref.view(np.uint16))

    def test_head_rmsnorm_runs_end_to_end_against_numpy_reference(
        self, spv_device,
    ):
        """``HeadRMSNormKernel`` runs E2E and matches its reference.
        Per-head RMS over Dh-wide column ranges of a packed QKV tensor;
        the K/V heads are normalized after the Q heads, V columns are
        copied through. Multiple per-block reductions exercise smem +
        subgroup_reduce."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.head_rmsnorm.kernel import HeadRMSNormKernel
        from quark.kernels.head_rmsnorm.spec import HeadRMSNormSpec
        from quark.kernels.head_rmsnorm.config import HeadRMSNormConfig
        from quark.kernels.head_rmsnorm.reference import (
            head_rmsnorm_reference_numpy,
        )
        from quark.launcher import Launcher

        n_q, n_kv, Dh = 4, 2, 64
        M = 4
        D_full = (n_q + 2 * n_kv) * Dh  # 8 * 64 = 512
        spec = HeadRMSNormSpec(
            M=M, D_full=D_full, n_q_heads=n_q, n_kv_heads=n_kv, Dh=Dh,
            dtype=DType.F32,
        )

        rng = np.random.default_rng(0xC0FFEE_5EED & 0xFFFFFFFF)
        X = rng.standard_normal((M, D_full)).astype(np.float32)
        expected = head_rmsnorm_reference_numpy(spec, X=X)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(HeadRMSNormKernel, spec, HeadRMSNormConfig(n_warps=1))

        x_h, x_map = _sd.allocate_buffer(X.nbytes)
        o_h, o_map = _sd.allocate_buffer(M * D_full * 4)
        ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

        ck.launch(buffers=[x_h, o_h])

        got = np.empty((M, D_full), dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4)

    def test_value_residual_runs_end_to_end_against_numpy_reference(
        self, spv_device,
    ):
        """``ValueResidualKernel`` runs E2E and matches its reference.
        ``out = v + lamb * (v1 - v)`` — flat 1D residual lerp; 4 buffers."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.value_residual.kernel import ValueResidualKernel
        from quark.kernels.value_residual.spec import ValueResidualSpec
        from quark.kernels.value_residual.config import ValueResidualConfig
        from quark.kernels.value_residual.reference import (
            value_residual_reference_numpy,
        )
        from quark.launcher import Launcher

        N = 256
        spec = ValueResidualSpec(N=N, dtype=DType.F32)

        rng = np.random.default_rng(0xBADC0FFEE & 0xFFFFFFFF)
        V = rng.standard_normal(N).astype(np.float32)
        V1 = rng.standard_normal(N).astype(np.float32)
        lamb = np.array([0.31], dtype=np.float32)
        expected = value_residual_reference_numpy(spec, V=V, V1=V1, lamb=lamb)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            ValueResidualKernel, spec,
            ValueResidualConfig(n_warps=1, elems_per_block=N),
        )

        v_h, v_map = _sd.allocate_buffer(V.nbytes)
        v1_h, v1_map = _sd.allocate_buffer(V1.nbytes)
        l_h, l_map = _sd.allocate_buffer(lamb.nbytes)
        o_h, o_map = _sd.allocate_buffer(N * 4)
        ctypes.memmove(v_map, V.ctypes.data, V.nbytes)
        ctypes.memmove(v1_map, V1.ctypes.data, V1.nbytes)
        ctypes.memmove(l_map, lamb.ctypes.data, lamb.nbytes)

        ck.launch(buffers=[v_h, v1_h, l_h, o_h])

        got = np.empty(N, dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)

    def test_ada_gate_residual_runs_end_to_end_against_numpy_reference(
        self, spv_device,
    ):
        """``AdaGateResidualKernel`` runs E2E and matches its reference.
        ``out = x + gate * y`` with gate broadcast from [G, D] to [B, D].
        4 buffers; chunk-D loop over the feature dim (uses ForLoopOp)."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.ada_gate_residual.kernel import (
            AdaGateResidualKernel,
        )
        from quark.kernels.ada_gate_residual.spec import AdaGateResidualSpec
        from quark.kernels.ada_gate_residual.config import (
            AdaGateResidualConfig,
        )
        from quark.kernels.ada_gate_residual.reference import (
            ada_gate_residual_reference_numpy,
        )
        from quark.launcher import Launcher

        G, M, D = 1, 32, 128
        B = G * M
        spec = AdaGateResidualSpec(G=G, M=M, D=D, dtype=DType.F32)

        rng = np.random.default_rng(0xA9A_DEC0DE & 0xFFFFFFFF)
        X = rng.standard_normal((B, D)).astype(np.float32)
        Y = rng.standard_normal((B, D)).astype(np.float32)
        gate = rng.standard_normal((G, D)).astype(np.float32) * 0.1
        expected = ada_gate_residual_reference_numpy(spec, X=X, Y=Y, gate=gate)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            AdaGateResidualKernel, spec,
            AdaGateResidualConfig(n_warps=1, chunk_D=128, direct=True),
        )

        x_h, x_map = _sd.allocate_buffer(X.nbytes)
        y_h, y_map = _sd.allocate_buffer(Y.nbytes)
        g_h, g_map = _sd.allocate_buffer(gate.nbytes)
        o_h, o_map = _sd.allocate_buffer(B * D * 4)
        ctypes.memmove(x_map, X.ctypes.data, X.nbytes)
        ctypes.memmove(y_map, Y.ctypes.data, Y.nbytes)
        ctypes.memmove(g_map, gate.ctypes.data, gate.nbytes)

        ck.launch(buffers=[x_h, y_h, g_h, o_h])

        got = np.empty((B, D), dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)

    def test_elementwise_kernel_compiles(self, spv_device):
        """``ElementwiseKernel`` — vec_load + per-elem op (add/mul/etc.) +
        vec_store. Cohort backbone for the attention block's scalar
        epilogues."""
        from quark.kernels.elementwise.kernel import ElementwiseKernel
        from quark.kernels.elementwise.spec import ElementwiseSpec
        from quark.kernels.elementwise.config import ElementwiseConfig
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            ElementwiseKernel,
            ElementwiseSpec(N=256, dtype=DType.F32, op="add"),
            ElementwiseConfig(n_warps=1, elems_per_block=256),
        )
        assert ck.module.handle != 0

    def test_elementwise_runs_end_to_end_against_numpy_reference(self, spv_device):
        """Run ``ElementwiseKernel`` (op="add") on Battlemage and verify
        it matches ``elementwise_reference_numpy``. Cohort backbone for
        attention's scalar epilogues — proves the SPV launcher hands the
        binary-op + vec_load/store path through correctly."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.elementwise.kernel import ElementwiseKernel
        from quark.kernels.elementwise.spec import ElementwiseSpec
        from quark.kernels.elementwise.config import ElementwiseConfig
        from quark.kernels.elementwise.reference import (
            elementwise_reference_numpy,
        )
        from quark.launcher import Launcher

        N = 256
        rng = np.random.default_rng(0xCAFE_FACE & 0xFFFFFFFF)
        X = rng.standard_normal(N).astype(np.float32)
        Y = rng.standard_normal(N).astype(np.float32)
        spec = ElementwiseSpec(N=N, dtype=DType.F32, op="add")
        expected = elementwise_reference_numpy(spec, X=X, Y=Y)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            ElementwiseKernel,
            spec,
            ElementwiseConfig(n_warps=1, elems_per_block=N),
        )

        nbytes = N * 4
        x_h, x_map = _sd.allocate_buffer(nbytes)
        y_h, y_map = _sd.allocate_buffer(nbytes)
        o_h, o_map = _sd.allocate_buffer(nbytes)
        ctypes.memmove(x_map, X.ctypes.data, X.nbytes)
        ctypes.memmove(y_map, Y.ctypes.data, Y.nbytes)

        ck.launch(buffers=[x_h, y_h, o_h])

        got = np.empty(N, dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    def test_ada_rmsnorm_runs_end_to_end_against_numpy_reference(self, spv_device):
        """``AdaRMSNormKernel`` runs E2E on Battlemage and matches its
        own ``ada_rmsnorm_reference_numpy``. First kernel through the
        SPV backend that uses ``ForLoopOp`` + carries (chunk-D
        accumulation), proving the loop visitor is correct on a real
        kernel — not just the synthetic ``test_for_loop_*`` cases."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.ada_rmsnorm.kernel import AdaRMSNormKernel
        from quark.kernels.ada_rmsnorm.spec import AdaRMSNormSpec
        from quark.kernels.ada_rmsnorm.config import AdaRMSNormConfig
        from quark.kernels.ada_rmsnorm.reference import (
            ada_rmsnorm_reference_numpy,
        )
        from quark.launcher import Launcher

        G, M, D = 1, 32, 128
        B = G * M
        spec = AdaRMSNormSpec(G=G, M=M, D=D, dtype=DType.F32)

        rng = np.random.default_rng(0xADA_DEC0DE & 0xFFFFFFFF)
        X = rng.standard_normal((B, D)).astype(np.float32)
        scale = rng.standard_normal((G, D)).astype(np.float32) * 0.1
        bias = rng.standard_normal((G, D)).astype(np.float32) * 0.1
        expected = ada_rmsnorm_reference_numpy(spec, X=X, scale=scale, bias=bias)

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            AdaRMSNormKernel, spec, AdaRMSNormConfig(n_warps=1, chunk_D=128),
        )

        x_h, x_map = _sd.allocate_buffer(X.nbytes)
        s_h, s_map = _sd.allocate_buffer(scale.nbytes)
        b_h, b_map = _sd.allocate_buffer(bias.nbytes)
        o_h, o_map = _sd.allocate_buffer(B * D * 4)
        ctypes.memmove(x_map, X.ctypes.data, X.nbytes)
        ctypes.memmove(s_map, scale.ctypes.data, scale.nbytes)
        ctypes.memmove(b_map, bias.ctypes.data, bias.nbytes)

        ck.launch(buffers=[x_h, s_h, b_h, o_h])

        got = np.empty((B, D), dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4)

    def test_value_residual_packed_runs_end_to_end_against_numpy_reference(
        self, spv_device,
    ):
        """``ValueResidualPackedKernel`` runs E2E and matches its
        reference. First kernel through SPV that uses
        - PRED-typed values + ``OpLogicalAnd`` (the v-col bounds gate),
        - 4 chunks (D_full // chunk_D = 4), so ``run_pipeline`` engages
          the double-buffer path with two consume calls in the loop
          body — both the iter_a (stage 0) and iter_b (stage 1) writes
          must hit Out, which depends on the kernel's expected
          workgroup size matching ``LocalSize`` (was wrong before — see
          launcher's INTEL_GPU branch in ``_lower``)."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.value_residual_packed.kernel import (
            ValueResidualPackedKernel,
        )
        from quark.kernels.value_residual_packed.spec import (
            ValueResidualPackedSpec,
        )
        from quark.kernels.value_residual_packed.config import (
            ValueResidualPackedConfig,
        )
        from quark.kernels.value_residual_packed.reference import (
            value_residual_packed_reference_numpy,
        )
        from quark.launcher import Launcher

        M, D_full = 4, 512
        v_col_offset, v_width = 256, 256
        spec = ValueResidualPackedSpec(
            M=M, D_full=D_full, v_col_offset=v_col_offset, v_width=v_width,
            dtype=DType.F32,
        )

        rng = np.random.default_rng(0xC0DE_FEED)
        QKV_curr = rng.standard_normal((M, D_full)).astype(np.float32)
        QKV_first = rng.standard_normal((M, D_full)).astype(np.float32)
        lamb = np.array([0.42], dtype=np.float32)
        expected = value_residual_packed_reference_numpy(
            spec, QKV_curr=QKV_curr, QKV_first=QKV_first, lamb=lamb,
        )

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            ValueResidualPackedKernel, spec,
            ValueResidualPackedConfig(n_warps=1, chunk_D=128),
        )

        c_h, c_map = _sd.allocate_buffer(QKV_curr.nbytes)
        f_h, f_map = _sd.allocate_buffer(QKV_first.nbytes)
        l_h, l_map = _sd.allocate_buffer(lamb.nbytes)
        o_h, o_map = _sd.allocate_buffer(M * D_full * 4)
        ctypes.memmove(c_map, QKV_curr.ctypes.data, QKV_curr.nbytes)
        ctypes.memmove(f_map, QKV_first.ctypes.data, QKV_first.nbytes)
        ctypes.memmove(l_map, lamb.ctypes.data, lamb.nbytes)

        ck.launch(buffers=[c_h, f_h, l_h, o_h])

        got = np.empty((M, D_full), dtype=np.float32)
        ctypes.memmove(got.ctypes.data, o_map, got.nbytes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)

    def test_gemm_single_k_iter_runs_end_to_end(self, spv_device):
        """``GemmKernel`` runs E2E at the smallest config (M=8, N=16,
        K=16, BM/BN/BK=8/16/16, n_warps=1, n_stages=1) and matches
        the numpy reference bit-identically.

        Single K-iteration so the for-loop accumulator carry is set
        up but the multi-iter tile-offset path doesn't kick in
        (tile offsets in the gmem→smem produce don't yet thread
        through the SPV vec_load chain — multi-iter K is the next
        gap). K=16 single-iter exercises:
          - cooperative-matrix LoadKHR for A (RowMajor) and B
            (ColumnMajor — the framework's gemm convention with B
            stored as N×K row-major).
          - MulAddKHR with the bf16/f32 ``m8n16k16`` shape.
          - StoreKHR via the FragForEach smem-roundtrip path.
          - For-loop carry phi typed as a coop-matrix accumulator
            (the ``coopmat-typed accumulator carry`` commit).
        """
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.gemm.kernel import GemmKernel
        from quark.kernels.gemm.spec import GemmSpec
        from quark.kernels.gemm.config import GemmConfig
        from quark.kernels.gemm.reference import gemm_reference_numpy
        from quark.launcher import Launcher

        M, N, K = 8, 16, 16
        spec = GemmSpec(
            M=M, N=N, K=K,
            a_dtype=DType.BF16, b_dtype=DType.BF16,
            acc_dtype=DType.F32, out_dtype=DType.BF16,
        )
        cfg = GemmConfig(
            BM=8, BN=16, BK=16,
            n_warps=1, n_stages=1,
            main_shape="m8n16k16_intel_bf16_f32",
        )

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(GemmKernel, spec, cfg)

        tensors = GemmKernel.make_tensors_numpy({
            "M": M, "N": N, "K": K,
            "a_dtype": DType.BF16, "b_dtype": DType.BF16,
            "out_dtype": DType.BF16,
        })
        ref = gemm_reference_numpy(spec, A=tensors["A"], B=tensors["B"])

        handles: list[int] = []
        out_ptr = 0
        for buf in ck.param_spec.buffers:
            arr = tensors[buf.name]
            h, ptr = _sd.allocate_buffer(arr.nbytes)
            if buf.name == "Out":
                out_ptr = ptr
            else:
                ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)
            handles.append(h)

        ck.launch(buffers=handles)

        got = np.empty((M, N), dtype=np.uint16)
        ctypes.memmove(got.ctypes.data, out_ptr, got.nbytes)
        # bf16 bit-identity vs the numpy reference — the SPV
        # cooperative-matrix MulAdd path matches exactly.
        np.testing.assert_array_equal(got, ref.view(np.uint16))

    def test_gemm_multi_k_iter_runs_end_to_end(self, spv_device):
        """``GemmKernel`` runs E2E with K > BK so the for-loop runs
        multiple iterations of the produce/consume tile pipeline.

        The bug this regression-tests against: ``_flatten_index`` used
        ``tensor.shape[-1]`` for the stride and ignored
        ``GlobalTensor.view`` static_/dyn_ offsets — every K iteration
        wrote the same gmem slice into smem, so the second iter's
        accumulator added a duplicate of the first iter's contribution
        instead of the next K-block. The offset-aware
        ``_flatten_global_index`` threads the tile's
        ``(static_row_offset, static_col_offset, dyn_row_offset,
        dyn_col_offset)`` and the parent's per-axis ``stride`` through
        the address arithmetic, fixing every gmem→smem produce inside
        a multi-iter K loop."""
        import ctypes
        import numpy as np

        from quark.drivers import _spv_dispatch as _sd
        from quark.kernels.gemm.kernel import GemmKernel
        from quark.kernels.gemm.spec import GemmSpec
        from quark.kernels.gemm.config import GemmConfig
        from quark.kernels.gemm.reference import gemm_reference_numpy
        from quark.launcher import Launcher

        M, N, K = 8, 16, 32  # K > BK → 2 iters
        spec = GemmSpec(
            M=M, N=N, K=K,
            a_dtype=DType.BF16, b_dtype=DType.BF16,
            acc_dtype=DType.F32, out_dtype=DType.BF16,
        )
        cfg = GemmConfig(
            BM=8, BN=16, BK=16,
            n_warps=1, n_stages=2,
            main_shape="m8n16k16_intel_bf16_f32",
        )

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(GemmKernel, spec, cfg)

        tensors = GemmKernel.make_tensors_numpy({
            "M": M, "N": N, "K": K,
            "a_dtype": DType.BF16, "b_dtype": DType.BF16,
            "out_dtype": DType.BF16,
        })
        ref = gemm_reference_numpy(spec, A=tensors["A"], B=tensors["B"])

        handles: list[int] = []
        out_ptr = 0
        for buf in ck.param_spec.buffers:
            arr = tensors[buf.name]
            h, ptr = _sd.allocate_buffer(arr.nbytes)
            if buf.name == "Out":
                out_ptr = ptr
            else:
                ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)
            handles.append(h)

        ck.launch(buffers=handles)

        got = np.empty((M, N), dtype=np.uint16)
        ctypes.memmove(got.ctypes.data, out_ptr, got.nbytes)
        np.testing.assert_array_equal(got, ref.view(np.uint16))

    def test_value_residual_packed_kernel_compiles(self, spv_device):
        """``ValueResidualPackedKernel`` — packed-QKV variant. First
        kernel through SPV that needs PRED-typed values (boolean SSA
        from cmp + and-fold) and ``OpLogicalAnd``."""
        from quark.kernels.value_residual_packed.kernel import (
            ValueResidualPackedKernel,
        )
        from quark.kernels.value_residual_packed.spec import (
            ValueResidualPackedSpec,
        )
        from quark.kernels.value_residual_packed.config import (
            ValueResidualPackedConfig,
        )
        from quark.launcher import Launcher

        launcher = Launcher(device=spv_device)
        ck = launcher.compile(
            ValueResidualPackedKernel,
            ValueResidualPackedSpec(M=4, D_full=512, v_col_offset=256,
                                     v_width=256, dtype=DType.F32),
            ValueResidualPackedConfig(n_warps=1, chunk_D=128),
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
