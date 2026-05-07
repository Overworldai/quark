"""End-to-end smoke for the _spv_dispatch compile/launch path.

Compiles a hand-written ``vector_add`` GLSL shader (checked in as
both ``.comp`` source and a pre-built ``.spv`` blob) through
``compile()`` + ``launch()``, verifies ``C = A + B`` against numpy.

PORTABILITY_PLAN §3.1's compile/launch deliverable. Mirrors the
existing ``test_spv_probe.py`` skip pattern — Linux + Vulkan-ICD
required.

Once the §3.2 ``SpirVLowerer`` lands, this test stays as a smoke
that the C-side dispatch path doesn't regress; the lowerer adds
its own visitor goldens that exercise the same compile/launch
pipeline against framework-emitted SPIR-V.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="_spv_dispatch is Linux-only (Vulkan + nanobind ext)",
)


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def spv_module():
    from quark.drivers import spv
    return spv


@pytest.fixture(scope="module")
def vulkan_available(spv_module):
    return spv_module.is_available()


@pytest.fixture(scope="module")
def dispatch(spv_module, vulkan_available):
    """The ``_spv_dispatch`` C extension itself, with a device bound.

    Tests below operate at the C-ext level (``compile`` / ``launch``)
    rather than going through ``SpvDriver`` because the latter
    raises NotImplementedError on those methods today — that error
    is itself tested in ``test_spv_probe.py``. When the Python
    facade gets wired in a follow-up, the tests below will move
    to using ``SpvDriver`` directly.
    """
    if not vulkan_available:
        pytest.skip("no Vulkan device available on this host")
    from quark.drivers import _spv_dispatch
    idx = spv_module.pick_default_device()
    if idx is None:
        pytest.skip("no non-CPU Vulkan device — vector_add smoke needs a GPU")
    _spv_dispatch.bind_device(idx)
    return _spv_dispatch


@pytest.fixture(scope="module")
def vector_add_spirv() -> bytes:
    """Pre-built SPIR-V binary for the vector_add smoke kernel.

    Re-build from ``vector_add.comp`` via:
        glslangValidator -V vector_add.comp -o vector_add.spv
    """
    blob_path = _FIXTURE_DIR / "vector_add.spv"
    if not blob_path.exists():
        pytest.skip(
            f"vector_add.spv not built — re-run "
            f"``glslangValidator -V {_FIXTURE_DIR / 'vector_add.comp'} "
            f"-o {blob_path}``"
        )
    return blob_path.read_bytes()


def _upload_f32(dispatch, mapped_ptr: int, data: np.ndarray) -> None:
    """Copy a numpy f32 array into the host-visible mapping at
    ``mapped_ptr``. Equivalent of ``buf[:] = data`` against the
    Vulkan storage buffer."""
    arr = np.ascontiguousarray(data, dtype=np.float32)
    ctypes.memmove(mapped_ptr, arr.ctypes.data, arr.nbytes)


def _download_f32(dispatch, mapped_ptr: int, n: int) -> np.ndarray:
    """Read a f32 vector of ``n`` elements out of the host-visible
    mapping at ``mapped_ptr``."""
    out = np.empty(n, dtype=np.float32)
    ctypes.memmove(out.ctypes.data, mapped_ptr, out.nbytes)
    return out


def test_bind_device_idempotent(dispatch, spv_module):
    """Re-binding to the same device is a no-op (no resource churn,
    no leaked descriptor sets / command buffers)."""
    idx = spv_module.pick_default_device()
    dispatch.bind_device(idx)
    dispatch.bind_device(idx)  # would crash on double-create otherwise


def test_allocate_buffer_returns_writable_mapping(dispatch):
    """Allocated buffers are zero-initialised and writable from
    Python via ``ctypes.memmove`` against the returned mapped_ptr."""
    handle, mapped = dispatch.allocate_buffer(64)
    assert handle != 0
    assert mapped != 0
    # Read it — should be all zeros.
    out = (ctypes.c_float * 16).from_address(mapped)
    for v in out:
        assert v == 0.0


def test_compile_runs_to_pipeline(dispatch, vector_add_spirv):
    """SPIR-V binary compiles cleanly to a Vulkan compute pipeline.
    Returns a non-zero handle that ``launch`` can resolve."""
    handle = dispatch.compile(
        spirv=vector_add_spirv,
        entry="main",
        n_buffers=3,
        push_constants_size=4,  # uint n
    )
    assert handle != 0


def test_vector_add_end_to_end(dispatch, vector_add_spirv):
    """The full §3.1 smoke: compile + dispatch + readback for
    ``C = A + B``. If this runs to completion and the numerics
    match, the SPIR-V backend's compile/launch path is correct."""
    n = 1024
    rng = np.random.default_rng(0xC0FFEE)
    A = rng.standard_normal(n).astype(np.float32)
    B = rng.standard_normal(n).astype(np.float32)
    expected = A + B

    nbytes = n * 4
    a_h, a_map = dispatch.allocate_buffer(nbytes)
    b_h, b_map = dispatch.allocate_buffer(nbytes)
    c_h, c_map = dispatch.allocate_buffer(nbytes)
    _upload_f32(dispatch, a_map, A)
    _upload_f32(dispatch, b_map, B)

    pipeline = dispatch.compile(
        spirv=vector_add_spirv,
        entry="main",
        n_buffers=3,
        push_constants_size=4,
    )

    # local_size_x = 64; grid covers ceil(n / 64) workgroups.
    n_groups = (n + 63) // 64

    # Push constant: one uint32 carrying the array length, used by
    # the shader's "if (i >= pc.n) return;" tail-trim.
    push = np.array([n], dtype=np.uint32).tobytes()

    dispatch.launch(
        pipeline,
        (n_groups, 1, 1),
        [a_h, b_h, c_h],
        push_bytes=push,
    )

    got = _download_f32(dispatch, c_map, n)

    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_launch_buffer_count_mismatch_raises(dispatch, vector_add_spirv):
    """The launch path validates buffer count against compile-time
    ``n_buffers``. A mismatch raises a clear RuntimeError."""
    pipeline = dispatch.compile(
        spirv=vector_add_spirv,
        entry="main",
        n_buffers=3,
        push_constants_size=4,
    )
    a_h, _ = dispatch.allocate_buffer(64)
    b_h, _ = dispatch.allocate_buffer(64)
    push = np.array([16], dtype=np.uint32).tobytes()
    with pytest.raises(RuntimeError, match="n_buffers"):
        dispatch.launch(pipeline, (1, 1, 1), [a_h, b_h], push_bytes=push)


def test_launch_push_size_mismatch_raises(dispatch, vector_add_spirv):
    """Push-constant size mismatch is also caught at launch."""
    pipeline = dispatch.compile(
        spirv=vector_add_spirv,
        entry="main",
        n_buffers=3,
        push_constants_size=4,
    )
    a_h, _ = dispatch.allocate_buffer(64)
    b_h, _ = dispatch.allocate_buffer(64)
    c_h, _ = dispatch.allocate_buffer(64)
    bad_push = b"\x00" * 8  # wrong size
    with pytest.raises(RuntimeError, match="push_bytes"):
        dispatch.launch(
            pipeline, (1, 1, 1), [a_h, b_h, c_h], push_bytes=bad_push,
        )
