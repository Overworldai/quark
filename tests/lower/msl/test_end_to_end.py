"""End-to-end Metal launch test: IR -> MSL -> compile -> launch -> verify.

Requires a Metal device. Skips on CUDA-only or non-Darwin.
Works with both MLX and PyObjC drivers (QUARK_METAL_DRIVER env var).
"""

import numpy as np
import pytest

try:
    from quark.drivers.metal import is_available

    HAS_METAL = is_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="no Metal device")


def test_vecadd_end_to_end():
    """Build a 256-element vec-add kernel via the IR, lower to MSL,
    compile via the active Metal driver, launch, and verify correctness."""
    from quark.device import DeviceFamily, current_device
    from quark.ir import BufferType, Builder, DType, GlobalTensor, ParamAttrs

    device = current_device()
    if device.family is not DeviceFamily.METAL:
        pytest.skip("not a Metal device")

    N = 256
    b = Builder("vecadd")
    b.begin_function("vecadd_kernel")
    b.param("A", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("B", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("C", BufferType(DType.F32))

    g_a = GlobalTensor(
        dtype=DType.F32, shape=(N,), stride=(1,), name="A", param=b.function.params[0]
    )
    g_b = GlobalTensor(
        dtype=DType.F32, shape=(N,), stride=(1,), name="B", param=b.function.params[1]
    )
    g_c = GlobalTensor(
        dtype=DType.F32, shape=(N,), stride=(1,), name="C", param=b.function.params[2]
    )

    tid = b.thread_idx("x")
    a_val = b.load(g_a, tid)
    b_val = b.load(g_b, tid)
    c_val = b.add(a_val, b_val)
    b.store(g_c, c_val, tid)
    b.end_function()

    from quark.lower.msl import MslLowerer

    lowered = MslLowerer(device.caps).lower_module(b.module)
    assert "thread_position_in_threadgroup" in lowered.source
    assert lowered.input_names == ["A", "B"]
    assert lowered.output_names == ["C"]

    # Compile via the active driver (respects QUARK_METAL_DRIVER).
    from quark.launcher.launcher import Launcher

    driver = Launcher(device=device).driver
    compiled = driver.compile(lowered, entry_name="vecadd_kernel", smem_bytes=0)

    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)

    outputs = driver.launch_metal(
        compiled,
        grid=(N, 1, 1),
        block=(N, 1, 1),
        input_arrays=[A, B],
        output_shapes=[(N,)],
        output_dtypes=[np.float32],
    )
    driver.sync(None)
    # launch_metal returns list of (handle, ptr, nbytes) — wrap as numpy.
    import ctypes

    _h, ptr, nbytes = outputs[0]
    C = np.frombuffer((ctypes.c_char * nbytes).from_address(ptr), dtype=np.float32).reshape(N)
    expected = A + B
    np.testing.assert_allclose(C, expected, atol=1e-5)


def test_vecadd_via_launcher():
    """Same vec-add but through the full Launcher -> CompiledKernel path."""
    from quark.device import DeviceFamily, current_device
    from quark.ir import BufferType, Builder, DType, GlobalTensor, ParamAttrs
    from quark.launcher.launcher import CompiledKernel, Launcher
    from quark.launcher.param_spec import ParamSpec, ProgramFootprint

    device = current_device()
    if device.family is not DeviceFamily.METAL:
        pytest.skip("not a Metal device")

    N = 128
    b = Builder("vecadd")
    b.begin_function("vecadd_kernel")
    b.param("A", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("B", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("C", BufferType(DType.F32))

    g_a = GlobalTensor(
        dtype=DType.F32, shape=(N,), stride=(1,), name="A", param=b.function.params[0]
    )
    g_b = GlobalTensor(
        dtype=DType.F32, shape=(N,), stride=(1,), name="B", param=b.function.params[1]
    )
    g_c = GlobalTensor(
        dtype=DType.F32, shape=(N,), stride=(1,), name="C", param=b.function.params[2]
    )
    tid = b.thread_idx("x")
    a_val = b.load(g_a, tid)
    b_val = b.load(g_b, tid)
    c_val = b.add(a_val, b_val)
    b.store(g_c, c_val, tid)
    b.end_function()

    from quark.lower.msl import MslLowerer

    lowered = MslLowerer(device.caps).lower_module(b.module)
    launcher = Launcher(device=device)
    driver = launcher.driver
    compiled_mod = driver.compile(lowered, entry_name="vecadd_kernel", smem_bytes=0)

    param_spec = ParamSpec.from_function(b.module.functions[0])
    ck = CompiledKernel(
        driver=driver,
        module=compiled_mod,
        entry="vecadd_kernel",
        grid_fn=lambda: (1, 1, 1),
        block_fn=lambda: (N, 1, 1),
        param_spec=param_spec,
        footprint=ProgramFootprint(smem_bytes=0),
    )

    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)

    result = ck.launch(buffers=[A, B, C])
    assert result is not None
    import ctypes

    _h, ptr, nbytes = result[0]
    out = np.frombuffer((ctypes.c_char * nbytes).from_address(ptr), dtype=np.float32).reshape(N)
    expected = A + B
    np.testing.assert_allclose(out, expected, atol=1e-5)
