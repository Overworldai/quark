"""End-to-end Metal launch test: IR → MSL → MLX compile → launch → verify.

Requires a Metal device (mx.metal.is_available()). Skips on CUDA-only.
"""

import pytest

try:
    import mlx.core as mx

    HAS_METAL = mx.metal.is_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="no Metal device")


def test_vecadd_end_to_end():
    """Build a 256-element vec-add kernel via the IR, lower to MSL,
    compile via MLX, launch, and verify correctness."""
    from popcorn.device import DeviceFamily, current_device
    from popcorn.ir import BufferType, Builder, DType, ParamAttrs

    device = current_device()
    if device.family is not DeviceFamily.METAL:
        pytest.skip("not a Metal device")

    # Build IR for: C[tid] = A[tid] + B[tid]
    N = 256
    b = Builder("vecadd")
    b.begin_function("vecadd_kernel")
    b.param("A", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("B", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("C", BufferType(DType.F32))

    from popcorn.ir import GlobalTensor

    g_a = GlobalTensor(
        dtype=DType.F32,
        shape=(N,),
        stride=(1,),
        name="A",
        param=b.function.params[0],
    )
    g_b = GlobalTensor(
        dtype=DType.F32,
        shape=(N,),
        stride=(1,),
        name="B",
        param=b.function.params[1],
    )
    g_c = GlobalTensor(
        dtype=DType.F32,
        shape=(N,),
        stride=(1,),
        name="C",
        param=b.function.params[2],
    )

    tid = b.thread_idx("x")
    a_val = b.load(g_a, tid)
    b_val = b.load(g_b, tid)
    c_val = b.add(a_val, b_val)
    b.store(g_c, c_val, tid)
    b.end_function()

    # Lower to MSL
    from popcorn.lower.msl import MslLowerer

    lowered = MslLowerer(device.caps).lower_module(b.module)
    assert "thread_position_in_threadgroup" in lowered.source
    assert lowered.input_names == ["A", "B"]
    assert lowered.output_names == ["C"]

    # Compile via MLX driver
    from popcorn.drivers.mlx import MlxDriver

    driver = MlxDriver(device=device)
    compiled = driver.compile(lowered, entry_name="vecadd_kernel", smem_bytes=0)

    # Launch
    A = mx.random.normal(shape=(N,)).astype(mx.float32)
    B = mx.random.normal(shape=(N,)).astype(mx.float32)
    mx.eval(A, B)

    outputs = driver.launch_mlx(
        compiled,
        grid=(N, 1, 1),
        block=(N, 1, 1),
        input_arrays=[A, B],
        output_shapes=[(N,)],
        output_dtypes=[mx.float32],
    )
    C = outputs[0]

    # Verify
    expected = A + B
    mx.eval(expected)
    diff = mx.abs(C - expected)
    mx.eval(diff)
    max_err = float(mx.max(diff))
    assert max_err < 1e-5, f"max error {max_err} too large"


def test_vecadd_via_launcher():
    """Same vec-add but through the full Launcher → CompiledKernel path."""
    from popcorn.device import DeviceFamily, current_device
    from popcorn.ir import BufferType, Builder, DType, ParamAttrs
    from popcorn.launcher.launcher import CompiledKernel, Launcher
    from popcorn.launcher.param_spec import ParamSpec, ProgramFootprint

    device = current_device()
    if device.family is not DeviceFamily.METAL:
        pytest.skip("not a Metal device")

    N = 128

    # Build the IR module
    b = Builder("vecadd")
    b.begin_function("vecadd_kernel")
    b.param("A", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("B", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
    b.param("C", BufferType(DType.F32))

    from popcorn.ir import GlobalTensor

    g_a = GlobalTensor(
        dtype=DType.F32,
        shape=(N,),
        stride=(1,),
        name="A",
        param=b.function.params[0],
    )
    g_b = GlobalTensor(
        dtype=DType.F32,
        shape=(N,),
        stride=(1,),
        name="B",
        param=b.function.params[1],
    )
    g_c = GlobalTensor(
        dtype=DType.F32,
        shape=(N,),
        stride=(1,),
        name="C",
        param=b.function.params[2],
    )
    tid = b.thread_idx("x")
    a_val = b.load(g_a, tid)
    b_val = b.load(g_b, tid)
    c_val = b.add(a_val, b_val)
    b.store(g_c, c_val, tid)
    b.end_function()

    # Lower + compile through the launcher's internal path
    from popcorn.lower.msl import MslLowerer

    lowered = MslLowerer(device.caps).lower_module(b.module)
    driver = Launcher(device=device).driver
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

    # Launch via CompiledKernel.launch()
    A = mx.random.normal(shape=(N,)).astype(mx.float32)
    B = mx.random.normal(shape=(N,)).astype(mx.float32)
    C = mx.zeros((N,), dtype=mx.float32)
    mx.eval(A, B, C)

    result = ck.launch(buffers=[A, B, C])
    assert result is not None, "Metal launch should return output arrays"
    out = result[0]

    expected = A + B
    mx.eval(expected)
    diff = mx.abs(out - expected)
    mx.eval(diff)
    max_err = float(mx.max(diff))
    assert max_err < 1e-5, f"max error {max_err} too large"
