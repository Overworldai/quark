"""OclDriver — quark backend on Intel OpenCL (NEO runtime + IGC).

C side: ``src/quark/drivers/_ocl_dispatch.cpp`` (nanobind, links
libOpenCL.so). Pipeline:

  OpenClSpirVLowerer → bytes (SPIR-V binary, OpenCL dialect)
        ↓
  OclDriver.compile(blob, entry, n_buffers, local_size)
        ↓ clCreateProgramWithIL + clBuildProgram + clCreateKernel
        ↓
  OclCompiledKernel(handle, n_buffers, push_size, local_size)
        ↓
  OclDriver.launch(mod, grid, buffer_handles, push_bytes)
        ↓ clSetKernelArgMemPointerINTEL × n_buffers
        ↓ clEnqueueNDRangeKernel (global = grid × local)
        ↓ optionally clFinish

Surface: probe / bind_device / allocate_buffer / compile / launch /
sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quark.device import (
    ChipGeneration,
    Device,
    DeviceCaps,
    DeviceFamily,
    chip_gen_from_intel_info,
)


def _try_import_dispatch():
    """Import the ``_ocl_dispatch`` C extension lazily so callers on
    non-OCL hosts get a clear "backend unavailable" error rather than
    ImportError at module import time."""
    try:
        from quark.drivers import _ocl_dispatch  # noqa: PLC0415
    except ImportError:
        return None
    return _ocl_dispatch


def is_available() -> bool:
    """True iff the ``_ocl_dispatch`` extension imports AND an OpenCL
    GPU device is enumerable."""
    mod = _try_import_dispatch()
    if mod is None:
        return False
    try:
        devices = mod.enumerate_devices()
    except Exception:
        return False
    return len(devices) > 0


def enumerate_devices() -> list[dict]:
    """List every OpenCL GPU device the ICD loader can see."""
    mod = _try_import_dispatch()
    if mod is None:
        return []
    try:
        return list(mod.enumerate_devices())
    except Exception:
        return []


def probe(device_index: int = 0) -> dict[str, Any]:
    """Return the full OpenCL capability dict for the device at
    ``device_index``."""
    mod = _try_import_dispatch()
    if mod is None:
        raise RuntimeError(
            "quark.drivers._ocl_dispatch is not available on this host. "
            "Install with libOpenCL + intel-opencl-icd (Ubuntu/Debian) and "
            "rebuild quark via ``pip install -e .`` on a Linux host."
        )
    return dict(mod.probe(device_index))


_INTEL_VENDOR_ID = 0x8086
_VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2


def pick_default_device(devices: list[dict] | None = None) -> int | None:
    """Pick a default device index. Intel discrete > Intel iGPU > any
    non-CPU."""
    if devices is None:
        devices = enumerate_devices()
    if not devices:
        return None
    intel_discrete = [
        d for d in devices
        if d.get("vendor_id") == _INTEL_VENDOR_ID
        and d.get("device_type") == _VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    ]
    if intel_discrete:
        return int(intel_discrete[0]["index"])
    intel_igpu = [
        d for d in devices
        if d.get("vendor_id") == _INTEL_VENDOR_ID
        and d.get("device_type") == _VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    ]
    if intel_igpu:
        return int(intel_igpu[0]["index"])
    return int(devices[0]["index"])


def caps_from_probe(probe_dict: dict[str, Any]) -> DeviceCaps:
    """Build a ``DeviceCaps`` from an OCL probe result."""
    chip_gen = chip_gen_from_intel_info({
        "device_name": probe_dict.get("device_name", ""),
        "device_id": probe_dict.get("device_id", 0),
    })
    name = probe_dict.get("device_name", "Unknown OpenCL device")
    arch_tag = chip_gen.value if chip_gen != ChipGeneration.UNKNOWN else "unknown"

    matmul_shapes = _intel_matmul_shapes_for(probe_dict, chip_gen)

    return DeviceCaps(
        family=DeviceFamily.INTEL_GPU,
        name=name,
        compute_unit_count=int(probe_dict.get("max_compute_units", 0)),
        subgroup_width=int(probe_dict.get("subgroup_size", 32)),
        max_threads_per_block=int(probe_dict.get("max_compute_workgroup_invocations", 1024)),
        max_smem_per_block=int(probe_dict.get("max_compute_shared_memory_size", 0)),
        max_regs_per_thread=0,
        max_regs_per_block=0,
        arch_tag=arch_tag,
        compute_capability=None,
        # Phase 6 will flip supports_async_copy=True once we land
        # the secondary command queue + cl_event prefetch path.
        supports_async_copy=False,
        supports_graph_capture=False,
        supports_fp8_e4m3=False,
        supports_bf16_mma=bool(probe_dict.get("bf16_cooperative_matrix", False))
            or bool(probe_dict.get("has_bf16_conversions", False)),
        matmul_shapes=matmul_shapes,
        supported_dtypes=_dtypes_from_probe(probe_dict),
        cpu_features=frozenset(),
    )


def _intel_matmul_shapes_for(
    probe_dict: dict[str, Any], chip_gen: ChipGeneration,
) -> frozenset[str]:
    """Return the set of MMA shape IDs the OCL/IGC lowerer can emit
    for this device.

    Filters the registry's chip-supported shapes by what
    ``quark.lower.ocl.lower._INTEL_MMA_LAYOUTS`` knows how to emit.
    The probe's ``has_subgroup_matrix_mma`` flag gates the whole set —
    if the device doesn't advertise the
    ``cl_intel_subgroup_matrix_multiply_accumulate`` extension, no
    shapes are supported (kernel cohort falls back to scalar paths).
    """
    if not probe_dict.get("has_subgroup_matrix_mma", False):
        return frozenset()

    # Importing here keeps the bootstrap order clean — quark.ir
    # imports quark.drivers via the device registry, so the lowerer
    # module can't be imported at top of this file.
    from quark.lower.ocl.lower import _INTEL_MMA_LAYOUTS
    from quark.ir.mma_registry import _BY_SHAPE_ID, shapes_for_chip

    lowerer_supported_keys = set(_INTEL_MMA_LAYOUTS.keys())

    confirmed: set[str] = set()
    for shape_id in shapes_for_chip(chip_gen):
        cfg = _BY_SHAPE_ID.get(shape_id)
        if cfg is None:
            continue
        # Only Intel-tagged shapes belong on the OCL backend (the
        # Vulkan KHR shapes carry the same chip gate but live on the
        # INTEL_GPU family, not INTEL_GPU).
        if cfg.intel_gpu is None:
            continue
        # Layout keys are SG-independent now — the SG width is a
        # property of the layout, not a key. The Intel MMA extension
        # constrains each (dtype, m, n, k) tuple to a single SG (8
        # or 16); the lowerer picks the right SG per shape.
        key = (
            cfg.shape.a_dtype, cfg.shape.b_dtype, cfg.shape.acc_dtype,
            cfg.shape.m, cfg.shape.n, cfg.shape.k,
        )
        if key in lowerer_supported_keys:
            confirmed.add(shape_id)
    return frozenset(confirmed)


def _dtypes_from_probe(probe_dict: dict[str, Any]) -> frozenset[str]:
    """Derive supported_dtypes from the OCL probe."""
    out: set[str] = {"f32", "s32", "u32", "f16"}
    if probe_dict.get("has_bf16_conversions"):
        out.add("bf16")
    if probe_dict.get("has_subgroup_matrix_mma"):
        out.add("s8")
        out.add("u8")
    return frozenset(out)


@dataclass
class OclCompiledKernel:
    """Opaque handle for a SPIR-V kernel built into a cl_kernel.

    ``handle`` is the C-side kernel-table index from
    ``_ocl_dispatch.compile``. ``n_buffers`` / ``push_size`` /
    ``local_size`` cached here for caller-side validation.
    """
    handle: int
    n_buffers: int
    push_size: int = 0
    local_size: tuple[int, int, int] = (32, 1, 1)


@dataclass
class OclDriver:
    """Driver implementing quark's Driver protocol on top of OpenCL."""

    device_index: int = 0

    def __post_init__(self):
        mod = _try_import_dispatch()
        if mod is None:
            raise RuntimeError("OCL backend unavailable; cannot build OclDriver")
        if not enumerate_devices():
            raise RuntimeError("No OpenCL GPU devices found")
        mod.bind_device(self.device_index)
        self._mod = mod
        self._probe = probe(self.device_index)
        self._caps = caps_from_probe(self._probe)

    @property
    def device(self) -> Device:
        return Device(
            family=DeviceFamily.INTEL_GPU,
            index=self.device_index,
            caps=self._caps,
        )

    @property
    def caps(self) -> DeviceCaps:
        return self._caps

    def allocate_buffer(self, nbytes: int) -> tuple[int, int]:
        return self._mod.allocate_buffer(nbytes)

    def compile(
        self,
        spirv: bytes,
        entry: str = "main",
        n_buffers: int = 0,
        push_constants_size: int = 0,
        subgroup_size: int = 32,
        local_size: tuple[int, int, int] = (32, 1, 1),
    ) -> OclCompiledKernel:
        lx, ly, lz = local_size
        h = self._mod.compile(
            spirv, entry, n_buffers, push_constants_size,
            subgroup_size, lx, ly, lz,
        )
        return OclCompiledKernel(
            handle=h, n_buffers=n_buffers,
            push_size=push_constants_size, local_size=local_size,
        )

    def launch(
        self,
        kernel: OclCompiledKernel | int,
        grid: tuple[int, int, int],
        buffers: list[int],
        push_bytes: bytes = b"",
        sync: bool = True,
    ) -> None:
        h = kernel.handle if isinstance(kernel, OclCompiledKernel) else int(kernel)
        self._mod.launch(h, tuple(grid), buffers, push_bytes, sync)

    def sync(self) -> None:
        self._mod.sync()
