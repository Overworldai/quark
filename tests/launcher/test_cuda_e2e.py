"""End-to-end CUDA tests for the Bundle 3 stack.

Builds a small kernel via the IR Builder, lowers it through
PtxLowerer, compiles via the CUDA driver, launches with torch
tensors, and validates the result. Skipped automatically if libcuda
isn't available.
"""
# ruff: noqa: E402  -- imports below pytest.skip(allow_module_level=True)
# are intentional; they would fail at module load on a CUDA-less system,
# so they sit behind the runtime gate above.

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from popcorn.drivers.cuda import CudaCompiledModule, CudaDriver
from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    GlobalTensor,
    Module,
)
from popcorn.kernels.base import Kernel, KernelConfig, KernelSpec
from popcorn.launcher import Launcher
from popcorn.runtime.cuda import CudaRuntime

# ---------------------------------------------------------------------------
# CudaRuntime — direct API
# ---------------------------------------------------------------------------


class TestCudaRuntime:
    def test_is_available(self):
        assert CudaRuntime.is_available()

    def test_device_count_at_least_one(self):
        rt = CudaRuntime.instance()
        assert rt.device_count() >= 1

    def test_device_name_returns_string(self):
        rt = CudaRuntime.instance()
        name = rt.get_device_name(0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_module_load_data_accepts_minimal_ptx(self):
        rt = CudaRuntime.instance()
        ptx = b""".version 8.7
.target sm_120
.address_size 64

.visible .entry noop()
{
    ret;
}
"""
        mod = rt.module_load_data(ptx)
        assert mod != 0
        rt.module_unload(mod)


# ---------------------------------------------------------------------------
# CudaDriver — probe + compile + launch
# ---------------------------------------------------------------------------


class TestCudaDriverProbe:
    def test_probe_populates_realistic_caps(self):
        driver = CudaDriver()
        caps = driver.device.caps
        assert caps.compute_unit_count > 0
        assert caps.warp_size == 32
        assert caps.max_threads_per_block >= 256
        assert caps.max_smem_per_block >= 48 * 1024
        assert caps.compute_capability is not None
        assert caps.arch_tag.startswith("sm_")
        assert caps.supports_bf16_mma is True or caps.compute_capability < (8, 0)


# ---------------------------------------------------------------------------
# Full vec_add kernel via Launcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VecAddSpec(KernelSpec):
    n: int = 128


@dataclass(frozen=True)
class _VecAddConfig(KernelConfig):
    block: int = 128


class _VecAddKernel(Kernel):
    """Tiny IR-built vec_add kernel for the e2e test."""

    def __init__(self, spec, config):
        self.spec = spec
        self.config = config
        self._compiled = None

    def is_valid(self) -> bool:
        return self.spec.n == self.config.block

    def smem_estimate(self) -> int:
        return 0

    def emit(self) -> Module:
        b = Builder("vec_add_module")
        fn = b.begin_function("vec_add")
        b.param("X", BufferType(DType.F32))
        b.param("Y", BufferType(DType.F32))
        b.param("Z", BufferType(DType.F32))
        g_x = GlobalTensor(
            dtype=DType.F32,
            shape=(self.spec.n,),
            stride=(1,),
            name="X",
            param=fn.params[0],
        )
        g_y = GlobalTensor(
            dtype=DType.F32,
            shape=(self.spec.n,),
            stride=(1,),
            name="Y",
            param=fn.params[1],
        )
        g_z = GlobalTensor(
            dtype=DType.F32,
            shape=(self.spec.n,),
            stride=(1,),
            name="Z",
            param=fn.params[2],
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


class TestLauncherEndToEnd:
    def test_vec_add_compiles_launches_and_matches_reference(self):
        launcher = Launcher()
        ck = launcher.compile(
            _VecAddKernel,
            _VecAddSpec(n=128),
            _VecAddConfig(block=128),
        )
        x = torch.arange(128, dtype=torch.float32, device="cuda")
        y = torch.arange(128, dtype=torch.float32, device="cuda") * 10
        z = torch.zeros_like(x)
        ck.launch(buffers=[x, y, z])
        torch.cuda.synchronize()
        assert torch.equal(z, x + y)

    def test_compile_cache_returns_same_object(self):
        launcher = Launcher()
        spec = _VecAddSpec(n=128)
        config = _VecAddConfig(block=128)
        ck1 = launcher.compile(_VecAddKernel, spec, config)
        ck2 = launcher.compile(_VecAddKernel, spec, config)
        assert ck1 is ck2

    def test_invalid_config_rejected_via_is_valid_for(self):
        launcher = Launcher()
        spec = _VecAddSpec(n=128)
        bad = _VecAddConfig(block=64)  # mismatched
        with pytest.raises(ValueError, match="invalid for device"):
            launcher.compile(_VecAddKernel, spec, bad)

    def test_no_config_consults_autotune_cache(self):
        """Bundle 5 wired AutotuneCache into Launcher.compile. When
        the caller passes config=None, the launcher consults the
        cache (hot → disk → bundled → search).

        _VecAddKernel doesn't expose tune_space / _pick_default_cfg /
        a default-constructible CONFIG_CLS, so the cache's fallback
        path runs out of options and raises a clear RuntimeError.
        That's the expected behavior for kernels that haven't
        provided any autotune hooks — it tells the author to either
        pass an explicit config or implement one of the hooks."""
        launcher = Launcher()
        with pytest.raises(RuntimeError, match="no fallback config available"):
            launcher.compile(_VecAddKernel, _VecAddSpec(n=128))


# ---------------------------------------------------------------------------
# Direct CudaCompiledModule round-trip via the driver
# ---------------------------------------------------------------------------


class TestCompileAndLaunchDirect:
    def test_minimal_ptx_compiles(self):
        driver = CudaDriver()
        ptx = """.version 8.7
.target sm_120
.address_size 64

.visible .entry noop()
{
    ret;
}
"""
        mod = driver.compile(ptx, "noop", smem_bytes=0)
        assert isinstance(mod, CudaCompiledModule)
        assert mod.smem_bytes == 0
        mod.close()
