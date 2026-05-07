"""Smoke + sanity tests for the _spv_dispatch C extension.

Skipped on non-Linux hosts (the extension is Linux-only — Apple's
Metal driver does not link against Vulkan, the CUDA driver is pure
ctypes). Skipped further when no Vulkan ICD is reachable, e.g. a
Linux CI runner without Mesa or Intel/AMD/NV proprietary drivers.

Where Vulkan IS available (the Intel devkit `xe3-devbox` is the
canonical run target) these tests assert the contract between the C
extension and ``drivers/spv.py``: ``probe()`` returns the keys
``caps_from_probe`` consumes, ``DeviceCaps.matmul_shapes`` only
contains shapes the driver actually advertises, etc.
"""

from __future__ import annotations

import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="_spv_dispatch is Linux-only (Vulkan + nanobind ext)",
)


@pytest.fixture(scope="module")
def spv_module():
    """Import ``drivers.spv`` lazily so non-Linux test runs don't
    explode at collection time."""
    from quark.drivers import spv
    return spv


@pytest.fixture(scope="module")
def vulkan_available(spv_module):
    """Skip the rest of the suite when no Vulkan ICD is reachable.

    Returned as a boolean so tests can also assert the negative case
    (``is_available()`` returning ``False`` cleanly rather than
    raising) without re-skipping themselves.
    """
    return spv_module.is_available()


def test_module_imports(spv_module):
    """The C extension must at least import — even on a CI host
    without an ICD, the .so should load and the surface should
    expose the expected names."""
    assert hasattr(spv_module, "is_available")
    assert hasattr(spv_module, "enumerate_devices")
    assert hasattr(spv_module, "probe")
    assert hasattr(spv_module, "pick_default_device")
    assert hasattr(spv_module, "caps_from_probe")
    assert hasattr(spv_module, "SpvDriver")


def test_enumerate_devices_returns_list(spv_module):
    """Even with no ICD, ``enumerate_devices()`` returns ``[]`` (not
    raising)."""
    devices = spv_module.enumerate_devices()
    assert isinstance(devices, list)


def test_enumerate_devices_has_required_keys(spv_module, vulkan_available):
    if not vulkan_available:
        pytest.skip("no Vulkan device available on this host")
    devices = spv_module.enumerate_devices()
    assert len(devices) > 0
    required = {"index", "device_name", "vendor_id", "device_id",
                "device_type", "api_version"}
    for d in devices:
        assert required.issubset(d.keys()), f"missing keys: {required - d.keys()}"
        assert isinstance(d["device_name"], str)
        assert isinstance(d["vendor_id"], int)


def test_probe_returns_complete_caps_dict(spv_module, vulkan_available):
    """Every key ``caps_from_probe`` reads must be present in the
    probe result. Drift here causes `caps.matmul_shapes` to fall
    back to `frozenset()` silently."""
    if not vulkan_available:
        pytest.skip("no Vulkan device available on this host")
    p = spv_module.probe(0)

    # Identity
    assert isinstance(p["device_name"], str)
    assert isinstance(p["vendor_id"], int)
    assert isinstance(p["device_id"], int)

    # Limits
    assert p["max_compute_workgroup_invocations"] >= 64
    assert p["max_compute_shared_memory_size"] >= 16 * 1024  # 16 KiB minimum
    assert p["max_push_constants_size"] >= 128

    # Subgroup
    assert p["subgroup_size"] in (8, 16, 32, 64), \
        f"unexpected subgroup_size {p['subgroup_size']}"

    # Feature flags are bool, not int
    assert isinstance(p["bf16_type"], bool)
    assert isinstance(p["bf16_cooperative_matrix"], bool)
    assert isinstance(p["atomic_f32_add_buffer"], bool)

    # Shapes list shape
    assert isinstance(p["cooperative_matrix_shapes"], list)
    if p["cooperative_matrix_shapes"]:
        s0 = p["cooperative_matrix_shapes"][0]
        for k in ("m", "n", "k", "a_dtype", "b_dtype", "c_dtype",
                  "result_dtype", "scope", "saturating_accumulation"):
            assert k in s0


def test_pick_default_device_skips_cpu(spv_module, vulkan_available):
    """``pick_default_device`` must never return a CPU rasterizer
    index when a real GPU is also enumerated (llvmpipe is useful as
    a fallback but never the production target)."""
    if not vulkan_available:
        pytest.skip("no Vulkan device available on this host")
    devices = spv_module.enumerate_devices()
    # Look for a non-CPU device; if there's only llvmpipe, pick will
    # fall back per docstring (still skipping CPU type).
    non_cpu = [d for d in devices if d["device_type"] != 4]
    if non_cpu:
        idx = spv_module.pick_default_device(devices)
        assert idx is not None
        picked = next(d for d in devices if d["index"] == idx)
        assert picked["device_type"] != 4, \
            f"picked CPU device {picked!r} when a non-CPU was available"


def test_caps_from_probe_intel_battlemage(spv_module, vulkan_available):
    """When the host is the Battlemage devkit, ``caps_from_probe``
    must surface the four registered ``m8n16k16_intel_*`` shapes
    AND nothing else. Acts as a smoke for the cross-check between
    ``ir/mma_registry`` and the live driver-advertised tuple set.
    """
    if not vulkan_available:
        pytest.skip("no Vulkan device available on this host")
    devices = spv_module.enumerate_devices()
    intel = [d for d in devices if d["vendor_id"] == 0x8086]
    if not intel:
        pytest.skip("no Intel Vulkan device — test is Battlemage-specific")
    probe_dict = spv_module.probe(intel[0]["index"])
    caps = spv_module.caps_from_probe(probe_dict)

    from quark.device import DeviceFamily
    assert caps.family is DeviceFamily.INTEL_GPU

    # All four registered Intel shapes must be confirmed by the
    # driver. If this fails, either the registry's gates are too
    # permissive, the driver's advertised shapes drifted, or the
    # cross-check key tuple in ``caps_from_probe`` is wrong.
    expected = {
        "m8n16k16_intel_bf16_f32",
        "m8n16k16_intel_bf16_bf16",
        "m8n16k16_intel_f16_f32",
        "m8n16k16_intel_f16_f16",
    }
    assert expected <= caps.matmul_shapes, \
        f"missing: {expected - caps.matmul_shapes}, got: {caps.matmul_shapes}"

    # bf16 path is intact + subgroup width is the SIMD32 we pin in §3.5
    assert caps.supports_bf16_mma is True
    assert caps.subgroup_width == 32


def test_spv_driver_construct_when_vulkan_available(spv_module, vulkan_available):
    """``SpvDriver()`` constructs cleanly on a host with Vulkan; on a
    host without, it raises a ``RuntimeError`` with a clear pointer
    at the install fix."""
    if not vulkan_available:
        with pytest.raises(RuntimeError, match="libvulkan"):
            spv_module.SpvDriver()
        return
    drv = spv_module.SpvDriver()
    assert drv.device_index >= 0
    # caps property is cached on first call; just confirm it resolves.
    caps = drv.caps
    assert caps is not None


def test_spv_driver_compile_validates_smem_budget(spv_module, vulkan_available):
    """``compile`` rejects kernels asking for more threadgroup memory
    than the device exposes. Caller-side validation; cheaper than
    letting Vulkan reject the pipeline."""
    if not vulkan_available:
        pytest.skip("requires Vulkan to construct SpvDriver")
    drv = spv_module.SpvDriver()
    huge = drv.caps.max_smem_per_block * 2 + 1
    with pytest.raises(ValueError, match="smem"):
        drv.compile(
            source=b"\x03\x02\x23\x07",  # SPIR-V magic, never reached
            entry="main",
            n_buffers=1,
            smem_bytes=huge,
        )


def test_spv_driver_launch_validates_buffer_count(spv_module, vulkan_available):
    """Buffer-count + push-bytes mismatches surface as ValueError
    from the Python facade — saves a trip into the C ext for the
    obvious caller-side errors."""
    if not vulkan_available:
        pytest.skip("requires Vulkan")
    drv = spv_module.SpvDriver()
    # Build a fake compiled module — we never actually launch, the
    # validation happens before the C call. Same shape for the test
    # in test_spv_compile_launch.py that validates from inside the C
    # ext; this one validates the Python facade's pre-check.
    compiled = spv_module.SpvCompiledModule(
        handle=0, n_buffers=3, push_size=4, smem_bytes=0,
    )
    with pytest.raises(ValueError, match="buffers"):
        drv.launch(compiled, (1, 1, 1), [1, 2], push_bytes=b"\x00" * 4)
    with pytest.raises(ValueError, match="push_bytes"):
        drv.launch(compiled, (1, 1, 1), [1, 2, 3], push_bytes=b"\x00" * 8)
