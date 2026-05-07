"""Tests for the SPIR-V lowerer's first cut.

Two tiers:

1. **Text-emit goldens** (cross-platform). Build a small IR module,
   run it through ``SpirVLowerer.lower_module``, assert the text
   output mentions the expected SPIR-V opcodes / decorations.
   Doesn't need ``spirv-as`` or Vulkan; runs everywhere.

2. **End-to-end smoke** (Linux + Vulkan + ``spirv-as``). Lower →
   assemble → dispatch via ``SpvDriver`` → read back → numerics
   match. Skipped on Mac and on hosts without ``spirv-as``.

Tier 1 is the regression-prevention net for visitor-by-visitor
work; tier 2 proves the whole pipeline (lowerer + assembler +
driver + dispatch) doesn't drift.
"""

from __future__ import annotations

import ctypes
import shutil
import sys

import numpy as np
import pytest

from quark.ir import Builder, DType
from quark.ir.tensor import GlobalTensor
from quark.ir.types import BufferType
from quark.lower.spv import SpirVLowerer


def _build_vec_add_ir(n: int = 64):
    """Construct a synthetic ``Z[i] = X[i] + Y[i]`` IR module.
    Mirrors ``tests/launcher/test_cuda_e2e.py``'s shape, ported to
    work without the launcher's compile machinery."""
    b = Builder("vec_add_module")
    fn = b.begin_function("vec_add")
    b.param("X", BufferType(DType.F32))
    b.param("Y", BufferType(DType.F32))
    b.param("Z", BufferType(DType.F32))
    g_x = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_y = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Y", param=fn.params[1])
    g_z = GlobalTensor(dtype=DType.F32, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[2])
    tid = b.thread_idx("x")
    b.store(g_z, b.add(b.load(g_x, tid), b.load(g_y, tid)), tid)
    b.end_function()
    return b.module


# ─────────────────────────────────────────────────────────────────
# Tier 1 — text-emit goldens.
# ─────────────────────────────────────────────────────────────────


class TestTextEmit:
    """Cross-platform: just check the text output's shape."""

    def test_vec_add_emits_expected_skeleton(self):
        result = SpirVLowerer(local_size=(64, 1, 1)).lower_module(_build_vec_add_ir())
        src = result.source

        # Capabilities + memory model.
        assert "OpCapability Shader" in src
        assert "OpMemoryModel Logical GLSL450" in src

        # Entry point + execution mode.
        assert 'OpEntryPoint GLCompute' in src
        assert '"main"' in src
        assert "OpExecutionMode" in src and "LocalSize 64 1 1" in src

        # Exactly one LocalInvocationId builtin.
        assert src.count("BuiltIn LocalInvocationId") == 1

        # Three storage-buffer bindings, descriptor set 0, sequential.
        assert src.count("DescriptorSet 0") == 3
        assert "Binding 0" in src
        assert "Binding 1" in src
        assert "Binding 2" in src

        # Block decoration appears exactly once on the struct (de-dupe
        # works — the lowerer otherwise emits it per-buffer).
        block_lines = [
            line for line in src.splitlines()
            if line.startswith("OpDecorate") and " Block" in line and "MemberDecorate" not in line
        ]
        assert len(block_lines) == 1, (
            f"expected single 'Block' decoration; got {block_lines}"
        )

        # Body ops we expect.
        assert "OpFAdd" in src
        assert "OpAccessChain" in src
        assert "OpLoad" in src
        assert "OpStore" in src
        assert "OpReturn" in src
        assert "OpFunctionEnd" in src

    def test_metadata_matches_kernel_shape(self):
        result = SpirVLowerer(local_size=(32, 1, 1)).lower_module(_build_vec_add_ir())
        assert result.entry_name == "main"
        assert result.n_buffers == 3
        assert result.local_size == (32, 1, 1)
        assert result.push_constants_size == 0


# ─────────────────────────────────────────────────────────────────
# Tier 2 — end-to-end through spirv-as + SpvDriver.
# ─────────────────────────────────────────────────────────────────


pytestmark_e2e = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("spirv-as") is None,
    reason="needs Linux + Vulkan + spirv-as for end-to-end dispatch",
)


@pytest.fixture(scope="module")
def driver():
    """SpvDriver bound to the default device. Skipped on hosts where
    Vulkan isn't reachable."""
    if sys.platform != "linux":
        pytest.skip("Linux-only")
    from quark.drivers import spv
    if not spv.is_available():
        pytest.skip("no Vulkan device")
    return spv.SpvDriver()


@pytestmark_e2e
def test_vec_add_end_to_end_through_lowerer(driver):
    """The full pipeline: build IR → lower → assemble → compile →
    dispatch → read back → numerics. Sister test to
    ``test_spv_compile_launch.test_vector_add_end_to_end``, but
    going through the framework lowerer instead of the hand-written
    GLSL fixture. If both pass, the lowerer's text emit is good."""
    from quark.lower.spv import text_to_binary

    n = 64
    rng = np.random.default_rng(0xBEEFCAFE)
    X = rng.standard_normal(n).astype(np.float32)
    Y = rng.standard_normal(n).astype(np.float32)
    expected = X + Y

    # Lower + assemble.
    result = SpirVLowerer(local_size=(n, 1, 1)).lower_module(_build_vec_add_ir(n))
    binary = text_to_binary(result.source)

    # Compile + dispatch via the public driver.
    nbytes = n * 4
    a_h, a_map = driver.allocate_buffer(nbytes)
    b_h, b_map = driver.allocate_buffer(nbytes)
    c_h, c_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(a_map, X.ctypes.data, X.nbytes)
    ctypes.memmove(b_map, Y.ctypes.data, Y.nbytes)

    compiled = driver.compile(
        source=binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        push_constants_size=result.push_constants_size,
    )
    # Single-workgroup dispatch: grid is (1, 1, 1), workgroup size
    # is local_size = (n, 1, 1), so n threads cover the n elements.
    driver.launch(compiled, (1, 1, 1), [a_h, b_h, c_h], push_bytes=b"")

    got = np.empty(n, dtype=np.float32)
    ctypes.memmove(got.ctypes.data, c_map, got.nbytes)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
