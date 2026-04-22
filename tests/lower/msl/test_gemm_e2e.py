"""End-to-end GEMM kernel test on Metal.

Compiles the real GemmKernel for bf16 through the full pipeline:
IR emit → MslLowerer → MlxDriver compile → launch → correctness.

Requires a Metal device. Skips on CUDA-only.
"""

import pytest

pytest.importorskip(
    "torch",
    reason="torch removed from runtime; numpy-refs migration — test kept for dev-only cross-check when torch is installed",
)

import pytest

try:
    import mlx.core as mx

    HAS_METAL = mx.metal.is_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="no Metal device")


def test_gemm_bf16_compile_and_lower():
    """Compile the GEMM kernel for a small bf16 problem and verify
    the MSL source contains expected patterns."""
    from quark.device import DeviceFamily, current_device
    from quark.kernels.gemm import GemmConfig, GemmKernel, GemmSpec
    from quark.lower.msl import MslLowerer

    device = current_device()
    if device.family is not DeviceFamily.METAL:
        pytest.skip("not a Metal device")

    spec = GemmSpec(M=64, N=64, K=64, a_dtype="bf16", b_dtype="bf16", out_dtype="bf16")
    # Metal's bf16 shape set is now m8n8k8_bf16 only (m16n8 dropped
    # after autotune showed m8n8k8 dominated every M3 problem).
    config = GemmConfig(BM=64, BN=64, BK=16, n_warps=4, n_stages=1, main_shape="m8n8k8_bf16")
    kernel = GemmKernel(spec, config)

    assert kernel.is_valid_for(device.caps), "config should be valid on Metal"

    # Emit IR and lower to MSL.
    module = kernel.emit()
    lowered = MslLowerer(device.caps).lower_module(module)
    src = lowered.source

    # Verify key MSL patterns.
    assert "thread_position_in_threadgroup" in src
    assert "threadgroup_position_in_grid" in src
    assert "threadgroup" in src  # smem declarations
    assert "simdgroup_multiply_accumulate" in src  # MMA
    assert "simdgroup_load" in src


def test_gemm_bf16_launch():
    """Full end-to-end: compile + launch + correctness check."""

    from quark.device import DeviceFamily, current_device
    from quark.kernels.gemm import GemmConfig, GemmKernel, GemmSpec
    from quark.launcher.launcher import Launcher

    device = current_device()
    if device.family is not DeviceFamily.METAL:
        pytest.skip("not a Metal device")

    M, N, K = 64, 64, 64
    spec = GemmSpec(M=M, N=N, K=K, a_dtype="bf16", b_dtype="bf16", out_dtype="bf16")
    # Metal's bf16 shape set is now m8n8k8_bf16 only (m16n8 dropped
    # after autotune showed m8n8k8 dominated every M3 problem).
    config = GemmConfig(BM=64, BN=64, BK=16, n_warps=4, n_stages=1, main_shape="m8n8k8_bf16")

    launcher = Launcher(device=device)
    compiled = launcher.compile(GemmKernel, spec, config)

    # Create tensors via the kernel's make_tensors.
    tensors = GemmKernel.make_tensors(
        {"M": M, "N": N, "K": K, "a_dtype": "bf16", "b_dtype": "bf16", "out_dtype": "bf16"}
    )
    A, B, Bias, Out = tensors["A"], tensors["B"], tensors["Bias"], tensors["Out"]

    # Launch.
    result = compiled.launch(buffers=[A, B, Bias, Out])
    assert result is not None, "Metal launch should return output arrays"
    C_metal = result[0]
    mx.eval(C_metal)

    # Cosine similarity check.
    import numpy as np
    import torch

    A_np = np.array(A.astype(mx.float32))
    B_np = np.array(B.astype(mx.float32))
    C_ref = torch.from_numpy(A_np) @ torch.from_numpy(B_np).T
    C_np = np.array(C_metal.astype(mx.float32))
    cos = float(
        torch.nn.functional.cosine_similarity(
            torch.from_numpy(C_np).flatten().float().unsqueeze(0),
            C_ref.flatten().unsqueeze(0),
        )
    )
    assert cos > 0.999, f"cosine similarity {cos:.6f} too low"
