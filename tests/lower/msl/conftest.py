"""Helpers for MSL-lowerer tests."""

import pytest

from quark.device import DeviceCaps, DeviceFamily
from quark.ir import Builder
from quark.lower.msl import LoweredMslKernel, MslLowerer

# Fake Metal caps for tests that don't need a real device.
METAL_CAPS_FAKE = DeviceCaps(
    family=DeviceFamily.METAL,
    name="test-metal",
    compute_unit_count=32,
    warp_size=32,
    max_threads_per_block=1024,
    max_smem_per_block=32 * 1024,
    max_regs_per_thread=None,
    max_regs_per_block=None,
    arch_tag="metal3",
    compute_capability=None,
    supports_async_copy=False,
    supports_graph_capture=False,
    supports_fp8_e4m3=False,
    supports_bf16_mma=True,
    matmul_shapes=frozenset({"m16n8k16_bf16", "m16n8k16_f16"}),
    supported_dtypes=frozenset({"f32", "f16", "bf16", "u32", "s32", "u8", "s8"}),
    cpu_features=frozenset(),
)


@pytest.fixture
def fresh_builder() -> Builder:
    b = Builder("t")
    b.begin_function("f")
    return b


def lower(b: Builder) -> str:
    """Close the function (if needed) and return the MSL source string."""
    if b._fn is not None:  # type: ignore[attr-defined]
        b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module).source


def lower_full(b: Builder) -> LoweredMslKernel:
    """Close the function and return the full `LoweredMslKernel`."""
    if b._fn is not None:  # type: ignore[attr-defined]
        b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module)
