"""SpvDriver — implements the launcher's Driver protocol on top of Vulkan.

PORTABILITY_PLAN §3.1's Python surface — the C++ side lives in
``src/quark/drivers/_spv_dispatch.cpp``. Pipeline:

  SpirVLowerer → bytes (SPIR-V binary)
        ↓
  SpvDriver.compile(blob, entry, smem_bytes)
        ↓ (in follow-up commit)
        ↓ vkCreateShaderModule → vkCreateComputePipelines
        ↓
  SpvCompiledModule(pipeline, descriptor_set_layout, smem_bytes)
        ↓
  SpvDriver.launch(mod, grid, block, buffer_handles, scalar_args)
        ↓ (in follow-up commit)
        ↓ record vkCmdDispatch into the active VkCommandBuffer
        ↓ trigger submit on sync() / output read / ops threshold

This file's first cut wires probing only — ``probe()``,
``is_available()``, ``enumerate_devices()``, plus a ``DeviceCaps``
factory that mirrors the Metal driver's ``probe_device`` shape.
``compile`` and ``launch`` raise ``NotImplementedError`` with a
pointer at the plan; this matches the §3.0 stub strategy.
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
    """Import the ``_spv_dispatch`` C extension lazily so callers on
    non-Linux hosts get a clear "backend unavailable" error rather
    than ``ImportError`` at module import time.

    Returns the module on success, ``None`` if it's not built or if
    Vulkan/the loader is unavailable on this host.
    """
    try:
        from quark.drivers import _spv_dispatch  # noqa: PLC0415
    except ImportError:
        return None
    return _spv_dispatch


def is_available() -> bool:
    """Report whether the SPIR-V backend can run on this host.

    True iff the ``_spv_dispatch`` extension imports AND a Vulkan
    physical device is enumerable. The C side raises
    ``RuntimeError`` from ``vkCreateInstance`` if no ICD is installed
    — surface that as ``False`` here instead of propagating, so
    ``Engine.__new__``'s family detection can fall through cleanly.
    """
    mod = _try_import_dispatch()
    if mod is None:
        return False
    try:
        devices = mod.enumerate_devices()
    except Exception:
        return False
    return len(devices) > 0


def enumerate_devices() -> list[dict]:
    """List every Vulkan physical device the loader can see.

    Identity-only (deviceName, vendorID, deviceID, deviceType,
    apiVersion). Use ``probe(index)`` to get the full caps surface
    for a specific device.
    """
    mod = _try_import_dispatch()
    if mod is None:
        return []
    try:
        return list(mod.enumerate_devices())
    except Exception:
        return []


def probe(device_index: int = 0) -> dict[str, Any]:
    """Return the full Vulkan capability dict for the device at
    ``device_index``.

    Same data the standalone ``scripts/spirv/probe_coopmat.c`` dumps
    to stdout, but parsed into Python types — the
    ``cooperative_matrix_shapes`` entry is a list of dicts, the
    feature flags are bools, etc.

    Raises ``RuntimeError`` if the C extension can't init Vulkan;
    raises ``IndexError`` if ``device_index`` is out of range.
    """
    mod = _try_import_dispatch()
    if mod is None:
        raise RuntimeError(
            "quark.drivers._spv_dispatch is not available on this host. "
            "Install with libvulkan-dev (Ubuntu/Debian) and rebuild quark "
            "via ``pip install -e .`` on a Linux host."
        )
    return dict(mod.probe(device_index))


# Vulkan VkPhysicalDeviceType values — see
# https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceType.html
_VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
_VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
_VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
_VK_PHYSICAL_DEVICE_TYPE_CPU = 4

# Intel vendor ID — used by ``pick_default_device`` to prefer Intel
# discrete/iGPU over CPU rasterisers (llvmpipe) on a fresh Vulkan
# install.
_INTEL_VENDOR_ID = 0x8086


def pick_default_device(devices: list[dict] | None = None) -> int | None:
    """Pick a default Vulkan device index for the SPIR-V backend.

    Preference order:
      1. Intel discrete GPU (vendorID=0x8086, deviceType=DISCRETE)
      2. Intel integrated GPU (vendorID=0x8086, deviceType=INTEGRATED)
      3. Any non-CPU device
      4. None — caller must handle the no-device case

    Skips ``deviceType == CPU`` (llvmpipe) — useful as an integration-
    test fallback but never the production target.

    ``devices`` defaults to ``enumerate_devices()`` when not supplied
    so callers can override (e.g. tests injecting a synthetic list).
    """
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
    non_cpu = [
        d for d in devices
        if d.get("device_type") != _VK_PHYSICAL_DEVICE_TYPE_CPU
    ]
    if non_cpu:
        return int(non_cpu[0]["index"])
    return None


def caps_from_probe(probe_dict: dict[str, Any]) -> DeviceCaps:
    """Build a ``DeviceCaps`` from a ``probe()`` result.

    Translates the raw Vulkan capability dict into Quark's
    backend-agnostic caps surface:
      * ``family`` is always ``DeviceFamily.INTEL_GPU`` today (no AMD
        SPIR-V path yet — when that lands, this function gates on
        vendorID).
      * ``arch_tag`` is the chip-gen string (``"intel_xe3"`` etc.).
      * ``matmul_shapes`` is the cross-section of
        ``shapes_for_chip(chip_gen)`` AND the device-reported
        ``cooperative_matrix_shapes`` — the registry's gate ensures
        we don't claim shapes the driver doesn't expose.
      * ``supports_async_copy`` / ``supports_fp8_e4m3`` are False for
        Battlemage today; revisit when AMD or future Intel exposes
        an equivalent.

    Empty / partial caps when the probe can't resolve something
    (e.g. no Vulkan loader); call ``is_available()`` first if your
    code path requires probe success.
    """
    from quark.ir.mma_registry import shapes_for_chip  # noqa: PLC0415

    chip_gen = chip_gen_from_intel_info({
        "device_name": probe_dict.get("device_name", ""),
        "device_id": probe_dict.get("device_id", 0),
    })
    registry_shapes = shapes_for_chip(chip_gen)

    # Cross-check the registry's claim against what the driver
    # actually advertises. Each registered shape's (M, N, K, A, B,
    # C, R, scope) tuple must appear in
    # ``cooperative_matrix_shapes``; mismatches surface as a missing
    # entry in the final ``matmul_shapes`` set so kernels that
    # require them fail their ``is_valid_for(caps)`` check cleanly
    # rather than at dispatch time.
    driver_shapes = {
        (s["m"], s["n"], s["k"], s["a_dtype"], s["b_dtype"],
         s["c_dtype"], s["result_dtype"], s["scope"])
        for s in probe_dict.get("cooperative_matrix_shapes", [])
    }
    confirmed: set[str] = set()
    for shape_id in registry_shapes:
        cfg = _shape_id_to_config(shape_id)
        if cfg is None:
            continue
        key = (
            cfg.shape.m, cfg.shape.n, cfg.shape.k,
            cfg.shape.a_dtype.value, cfg.shape.b_dtype.value,
            cfg.shape.acc_dtype.value, cfg.shape.acc_dtype.value,
            "subgroup",
        )
        if key in driver_shapes:
            confirmed.add(shape_id)

    name = probe_dict.get("device_name", "Unknown Vulkan device")
    arch_tag = chip_gen.value if chip_gen != ChipGeneration.UNKNOWN else "unknown"

    return DeviceCaps(
        family=DeviceFamily.INTEL_GPU,
        name=name,
        compute_unit_count=0,  # VkPhysicalDeviceProperties doesn't expose this directly
        subgroup_width=int(probe_dict.get("subgroup_size", 32)),
        max_threads_per_block=int(probe_dict.get("max_compute_workgroup_invocations", 1024)),
        max_smem_per_block=int(probe_dict.get("max_compute_shared_memory_size", 0)),
        max_regs_per_thread=0,  # not exposed by Vulkan
        max_regs_per_block=0,  # not exposed by Vulkan
        arch_tag=arch_tag,
        compute_capability=None,  # CUDA-only field
        supports_async_copy=False,
        supports_graph_capture=False,  # VK_KHR_pipeline_executable_properties is read-only
        supports_fp8_e4m3=False,  # not on Battlemage
        supports_bf16_mma=bool(probe_dict.get("bf16_cooperative_matrix", False)),
        matmul_shapes=frozenset(confirmed),
        supported_dtypes=_dtypes_from_probe(probe_dict),
        cpu_features=frozenset(),
    )


def _shape_id_to_config(shape_id: str):
    """Local helper — find the ``MmaConfig`` for ``shape_id``.

    Avoids polluting ``ir/mma_registry``'s public surface with a
    one-off lookup; the dict is internal there.
    """
    from quark.ir import mma_registry  # noqa: PLC0415

    for cfg in mma_registry.ALL_SHAPES:
        if cfg.shape.name == shape_id:
            return cfg
    return None


def _dtypes_from_probe(probe_dict: dict[str, Any]) -> frozenset[str]:
    """Derive the ``supported_dtypes`` set from the probe result.

    f32 / f16 / bf16 are universal on Vulkan compute (storage +
    arithmetic at the SPIR-V level); we expose them based on the
    coopmat shapes the driver actually implements (so kernels that
    only need scalar f16 don't accidentally land on a device whose
    coopmat surface is f32-only). bf16 gates on
    ``shaderBFloat16Type`` since that's what controls SPIR-V
    ``BFloat16`` capability availability.
    """
    out: set[str] = {"f32", "s32", "u32"}  # always available
    if probe_dict.get("bf16_type"):
        out.add("bf16")
    # f16 is exposed by ``shaderFloat16``, but the probe doesn't yet
    # query that struct. Default-include for now since every Vulkan
    # 1.3 driver targeting compute exposes f16.
    out.add("f16")
    # int8 — included because the s8/u8 cooperative-matrix shapes
    # were observed on Battlemage. Matches the existing convention
    # in ``_default_dtypes_for``.
    out.add("u8")
    out.add("s8")
    return frozenset(out)


@dataclass
class SpvCompiledModule:
    """Opaque SPIR-V compiled handle.

    Mirrors ``CudaCompiledModule`` / Metal's compiled-pipeline
    dataclass. Filled in by ``SpvDriver.compile`` once that lands;
    today the dataclass exists so the type-only side of the launcher
    can reference it.

    ``pipeline`` is the ``VkPipeline`` handle (uint64 from C).
    ``layout`` is the ``VkPipelineLayout``.
    ``descriptor_set_layout`` is the ``VkDescriptorSetLayout``.
    ``smem_bytes`` is the kernel's static threadgroup-memory ask;
    Vulkan's ``maxComputeSharedMemorySize`` cap is enforced by
    ``compile``.
    """

    pipeline: int
    layout: int
    descriptor_set_layout: int
    smem_bytes: int


class SpvDriver:
    """SPIR-V backend driver.

    Probe-only in this first cut. Compile + launch land in subsequent
    commits per PORTABILITY_PLAN §3.1.
    """

    def __init__(self, device_index: int | None = None):
        if not is_available():
            raise RuntimeError(
                "SpvDriver: Vulkan unavailable. Install libvulkan + an "
                "ICD, then rebuild quark via ``pip install -e .`` on "
                "Linux."
            )
        if device_index is None:
            device_index = pick_default_device()
            if device_index is None:
                raise RuntimeError(
                    "SpvDriver: no non-CPU Vulkan device found. "
                    "Devices: " + repr(enumerate_devices())
                )
        self.device_index: int = device_index
        self._probe: dict[str, Any] | None = None

    @property
    def caps(self) -> DeviceCaps:
        """``DeviceCaps`` for the bound device. Cached after first call."""
        if self._probe is None:
            self._probe = probe(self.device_index)
        return caps_from_probe(self._probe)

    @property
    def device(self) -> Device:
        """Quark ``Device`` wrapper around the bound Vulkan physical device.

        ``device.caps`` carries the family / chip-gen / shape set the
        rest of the framework consumes.
        """
        return Device(family=DeviceFamily.INTEL_GPU, index=self.device_index, caps=self.caps)

    def compile(self, source: bytes, entry: str, smem_bytes: int) -> SpvCompiledModule:
        """Compile a SPIR-V binary blob to a dispatchable pipeline.

        Lands in PORTABILITY_PLAN §3.1's compile/launch follow-up
        commit. Today raises with a pointer at the plan so callers
        get a clear progress signal.
        """
        del source, entry, smem_bytes
        raise NotImplementedError(
            "SpvDriver.compile: lands in the next §3.1 commit. See "
            "PORTABILITY_PLAN §3.1 — runtime-choice section is done; "
            "vkCreateShaderModule + vkCreateComputePipelines wiring "
            "is the next step."
        )

    def launch(self, *args, **kwargs) -> None:
        """Record a dispatch into the active command buffer.

        Lands alongside ``compile`` in §3.1's compile/launch commit.
        """
        del args, kwargs
        raise NotImplementedError(
            "SpvDriver.launch: lands in the next §3.1 commit alongside "
            "``compile``. See PORTABILITY_PLAN §3.1."
        )
