"""End-to-end tests for the PyObjC Metal driver.

Runs on Apple Silicon only — skips everywhere else.
"""

from __future__ import annotations

import platform

import numpy as np
import pytest

_SKIP = platform.system() != "Darwin"
_SKIP_REASON = "Requires Metal-capable Apple Silicon Mac"

# Guard the imports so the module is importable on Linux for collection.
if not _SKIP:
    try:
        from quark.drivers.metal import MetalDriver, is_available

        _SKIP = not is_available()
    except ImportError:
        _SKIP = True


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestProbe:
    def test_probe_returns_valid_caps(self):
        driver = MetalDriver()
        caps = driver.device.caps
        assert caps.name  # non-empty
        assert caps.subgroup_width == 32
        assert caps.max_threads_per_block >= 512
        assert caps.max_smem_per_block >= 16384
        assert caps.chip_gen.is_metal

    def test_metal4_detection(self):
        driver = MetalDriver()
        caps = driver.device.caps
        # On M4+ this should be True; on older chips False.
        # Just verify the field exists and is a bool.
        assert isinstance(caps.supports_metal4, bool)
        assert isinstance(caps.supports_nax, bool)
        if caps.supports_nax:
            assert caps.supports_metal4  # NAX implies Metal 4


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestCompileAndLaunch:
    """Compile and dispatch a vec_add kernel through the full driver stack."""

    def _make_lowered(self):
        """Build a minimal LoweredMslKernel for vec_add."""
        from quark.lower.msl.lower import LoweredMslKernel

        return LoweredMslKernel(
            source=(
                "uint i = thread_position_in_grid.x;\n"
                "if (i < A_shape[0]) {\n"
                "    C[i] = A[i] + B[i];\n"
                "}\n"
            ),
            header="",
            kernel_name="quark_vec_add",
            smem_bytes=0,
            input_names=["A", "B"],
            output_names=["C"],
            scalar_names=[],
            atomic_outputs=False,
            template_args=[],
            input_dtypes=["float", "float"],
            output_dtypes=["float"],
            scalar_dtypes=[],
        )

    def test_compile(self):
        driver = MetalDriver()
        lowered = self._make_lowered()
        mod = driver.compile(lowered, lowered.kernel_name, 0)
        assert mod.pipeline is not None
        assert "A_shape" in mod.source
        assert "const device float* A" in mod.source

    def test_launch_matches_numpy(self):
        driver = MetalDriver()
        lowered = self._make_lowered()
        mod = driver.compile(lowered, lowered.kernel_name, 0)

        N = 1024
        rng = np.random.default_rng(42)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)

        tg = 256
        grid_tg = ((N + tg - 1) // tg, 1, 1)

        outputs = driver.launch_mlx(
            mod,
            grid=grid_tg,
            block=(tg, 1, 1),
            input_arrays=[A, B],
            output_shapes=[(N,)],
            output_dtypes=[np.float32],
        )
        driver.sync(None)
        import ctypes

        _h, ptr, nbytes = outputs[0]
        C = np.frombuffer((ctypes.c_char * nbytes).from_address(ptr), dtype=np.float32).reshape(N)
        np.testing.assert_array_equal(C, A + B)

    def test_pipeline_cache(self):
        """Second compile of identical source should return cached pipeline."""
        driver = MetalDriver()
        lowered = self._make_lowered()
        mod1 = driver.compile(lowered, lowered.kernel_name, 0)
        mod2 = driver.compile(lowered, lowered.kernel_name, 0)
        assert mod1.pipeline is mod2.pipeline


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestHalfPrecision:
    def test_half_vec_add(self):
        from quark.lower.msl.lower import LoweredMslKernel

        lowered = LoweredMslKernel(
            source=("uint i = thread_position_in_grid.x;\nC[i] = A[i] + B[i];\n"),
            header="",
            kernel_name="quark_vec_add_f16",
            smem_bytes=0,
            input_names=["A", "B"],
            output_names=["C"],
            scalar_names=[],
            atomic_outputs=False,
            template_args=[],
            input_dtypes=["half", "half"],
            output_dtypes=["half"],
            scalar_dtypes=[],
        )
        driver = MetalDriver()
        mod = driver.compile(lowered, lowered.kernel_name, 0)

        N = 512
        A = np.ones(N, dtype=np.float16) * 1.5
        B = np.ones(N, dtype=np.float16) * 2.5

        outputs = driver.launch_mlx(
            mod,
            grid=((N + 255) // 256, 1, 1),
            block=(256, 1, 1),
            input_arrays=[A, B],
            output_shapes=[(N,)],
            output_dtypes=[np.float16],
        )
        driver.sync(None)
        import ctypes

        _h, ptr, nbytes = outputs[0]
        out = np.frombuffer((ctypes.c_char * nbytes).from_address(ptr), dtype=np.float16).reshape(N)
        np.testing.assert_array_equal(out, np.float16(4.0))
