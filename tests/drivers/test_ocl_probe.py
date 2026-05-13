"""Smoke + sanity tests for the _ocl_dispatch C extension.

Skipped on non-Linux hosts (the extension is Linux-only — depends on
``libOpenCL`` + Intel's NEO ICD). Skipped further when no OpenCL GPU
device is reachable on the host.

These tests assert the contract between the C extension and
``drivers/ocl.py``: ``probe()`` returns the keys ``caps_from_probe``
consumes, the Intel SPV / coopmat extensions are present on Intel
iGPUs, USM allocations succeed, etc.
"""

from __future__ import annotations

import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="_ocl_dispatch is Linux-only (OpenCL + nanobind ext)",
)


@pytest.fixture(scope="module")
def ocl_module():
    from quark.drivers import ocl
    return ocl


@pytest.fixture(scope="module")
def ocl_available(ocl_module):
    """True iff the C extension imports AND at least one OpenCL GPU
    device is reachable on this host."""
    return ocl_module.is_available()


def test_module_imports(ocl_module):
    assert hasattr(ocl_module, "is_available")
    assert hasattr(ocl_module, "enumerate_devices")
    assert hasattr(ocl_module, "probe")
    assert hasattr(ocl_module, "caps_from_probe")
    assert hasattr(ocl_module, "OclDriver")


def test_enumerate_devices_smoke(ocl_module):
    """``enumerate_devices`` should return a list (possibly empty)
    without raising even when no OpenCL ICD is installed."""
    devices = ocl_module.enumerate_devices()
    assert isinstance(devices, list)


def test_probe_when_available(ocl_module, ocl_available):
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")
    p = ocl_module.probe(0)
    # Required keys for caps_from_probe to function.
    for key in [
        "device_name", "vendor_id", "device_id", "device_type",
        "max_compute_units", "max_compute_workgroup_invocations",
        "max_compute_shared_memory_size", "subgroup_size",
        "has_il_program", "has_usm",
    ]:
        assert key in p, f"probe() missing key {key!r}"


def test_intel_extensions_on_intel_device(ocl_module, ocl_available):
    """On an Intel iGPU/dGPU we *must* see the extensions the Phase 3
    lowerer will depend on, otherwise the whole OCL backend is moot."""
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")
    p = ocl_module.probe(0)
    if p.get("vendor_id") != 0x8086:
        pytest.skip("not an Intel OpenCL GPU")
    assert p["has_il_program"], "Intel GPU missing cl_khr_il_program (SPIR-V binary input)"
    assert p["has_usm"], "Intel GPU missing cl_intel_unified_shared_memory"
    assert p["has_subgroup_matrix_mma"], (
        "Intel GPU missing cl_intel_subgroup_matrix_multiply_accumulate "
        "(direct dpas access — Phase 3 lowerer requires it)"
    )


def test_caps_from_probe(ocl_module, ocl_available):
    """``caps_from_probe`` should produce a usable ``DeviceCaps``."""
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")
    from quark.device import DeviceFamily

    p = ocl_module.probe(0)
    caps = ocl_module.caps_from_probe(p)
    assert caps.family is DeviceFamily.INTEL_GPU
    assert caps.max_threads_per_block > 0
    assert caps.subgroup_width > 0
    # On the PTL devkit we expect 128 KB smem; minimum threshold here
    # is conservative (32 KB) so the test doesn't false-fail on
    # devices with smaller LDS.
    assert caps.max_smem_per_block >= 32 * 1024


def test_driver_construction(ocl_module, ocl_available):
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")
    drv = ocl_module.OclDriver()
    assert drv.device.caps.family.value == "intel_gpu"
    assert drv.caps.max_smem_per_block > 0


def test_pick_default_device_returns_intel_when_present(ocl_module, ocl_available):
    if not ocl_available:
        pytest.skip("no OpenCL GPU device reachable")
    idx = ocl_module.pick_default_device()
    assert idx is not None
    assert idx >= 0
