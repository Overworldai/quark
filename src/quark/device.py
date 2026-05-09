"""Device and DeviceCaps abstraction.

EXEMPT FROM 500-LINE RULE: DeviceFamily / ChipGeneration / DeviceCaps /
Device form one tightly coupled surface — every probe path, test-
device helper, and capability key needs to see them together. Splitting
forces awkward cross-module cycles between the enum declarations and
the helper functions that dispatch on them.

The single typed structure every `Kernel.is_valid_for()` and the
launcher consult to ask "what hardware am I running on, and what does
it support?"

Bundle 2 ships the **data types** plus a CUDA-only `current_device()`
implementation. The actual `cuDeviceGetAttribute` probing is done by
the CUDA driver (Bundle 3); until that lands, `current_device()`
populates `DeviceCaps` from torch's CUDA properties (which is what
torch already exposes via `torch.cuda.get_device_properties`).

Other backends (ROCm, OpenCL, Metal, CPU) are wired into the
detection order but raise `NotImplementedError` when probed —
Bundle 3+ adds the CUDA implementation; later bundles add the rest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from typing import Optional


class DeviceFamily(Enum):
    """Backend / hardware family. Exactly one is active per process."""

    CUDA = "cuda"
    ROCM = "rocm"
    OPENCL = "opencl"
    METAL = "metal"
    # Intel GPUs (Arc discrete + Xe integrated). Distinct from OPENCL
    # because the target IR is SPIR-V compute shaders driven via
    # Vulkan / Level Zero, not CL's SPIR-V flavor — the driver picks,
    # the register table keys on this family. See
    # PORTABILITY_PLAN.md §3.
    INTEL_GPU = "intel_gpu"
    CPU = "cpu"


# Default subgroup (warp/simd/execution-lane) width assumed at kernel
# import time. The 5 elementwise/normalization kernels currently hardcode
# this into their ``is_valid()`` tile-size checks because ``is_valid()``
# runs before any device context is attached to the kernel. Once we plumb
# caps into ``is_valid_for``, each kernel can consult the device's actual
# ``caps.subgroup_width`` instead and this constant goes away.
#
# 32 is correct for every backend currently wired up (NV / Metal / Apple
# GPU / CDNA) and for the SPIR-V v1 plan (see PORTABILITY_PLAN.md §3.4 —
# Intel pinned to SIMD32 via ``reqd_sub_group_size(32)`` for v1). RDNA
# wave64 and Intel non-pinned builds are the failure modes the plan
# already calls out; they require the is_valid_for caps work first.
DEFAULT_SUBGROUP_WIDTH: int = 32


class ChipGeneration(Enum):
    """Finer-grained hardware generation than ``DeviceFamily``.

    A single ``arch_tag`` string (e.g. ``"sm_89"``, ``"metal3"``) can't
    distinguish chips with materially different MMA capabilities — M1
    vs M3 Ultra vs M5 (tensor-core) all report ``"metal3"``-class; 4090
    vs 5090 differ by one sm number but have different block-MMA
    support. This enum is the authoritative per-chip capability key.
    Per-descriptor ``min_*`` gates in ``quark.ir.mma_registry`` filter
    the shape set a given generation supports.
    """

    # CUDA (NVIDIA) — ordered by (major, minor).
    SM_75 = "sm_75"  # Turing
    SM_80 = "sm_80"  # Ampere A100
    SM_86 = "sm_86"  # Ampere consumer (RTX 30-series)
    SM_89 = "sm_89"  # Ada (RTX 40-series)
    SM_90 = "sm_90"  # Hopper (H100)
    SM_100 = "sm_100"  # Blackwell datacenter
    SM_120 = "sm_120"  # Blackwell consumer (RTX 50-series)
    # Apple Metal — ordered by tensor-compute generation.
    METAL_M1 = "metal_m1"  # A14 / M1 / M2 — simdgroup_matrix baseline
    METAL_M3 = "metal_m3"  # M3 / M3 Pro / M3 Max / M3 Ultra — bf16 MMA native
    METAL_M5 = "metal_m5"  # M5+ — direct tensor-core primitive (reserved)
    # Intel GPU (SPIR-V via Vulkan + VK_KHR_cooperative_matrix). Ordered
    # by Xe microarchitecture generation.
    INTEL_XE_LPG = "intel_xe_lpg"  # Meteor Lake iGPU
    INTEL_XE2 = "intel_xe2"  # Lunar Lake iGPU + discrete Arc B-series (Battlemage)
    INTEL_XE3 = "intel_xe3"  # Panther Lake iGPU + future discrete (Battlemage+)
    # Fallback.
    UNKNOWN = "unknown"

    @property
    def is_cuda(self) -> bool:
        return self.name.startswith("SM_")

    @property
    def is_metal(self) -> bool:
        return self.name.startswith("METAL_")

    @property
    def is_intel_gpu(self) -> bool:
        return self.name.startswith("INTEL_")

    def cuda_cc(self) -> tuple[int, int] | None:
        """CUDA compute capability (major, minor) for CUDA chips; None else."""
        if not self.is_cuda:
            return None
        n = self.value[3:]  # strip 'sm_'
        # sm_120 → (12, 0); sm_89 → (8, 9)
        if len(n) == 2:
            return (int(n[0]), int(n[1]))
        return (int(n[:-1]), int(n[-1]))


def chip_gen_from_cuda_cc(major: int, minor: int) -> ChipGeneration:
    """Map a CUDA compute capability to ``ChipGeneration``. Unknown
    triples (e.g. future sms) fall back to the nearest-older gen so
    kernels keep running — the shape registry will still filter to
    actually-supported shapes based on ``min_ptx_cc``.
    """
    tag = f"sm_{major}{minor}"
    for gen in ChipGeneration:
        if gen.value == tag:
            return gen
    # Unknown — pick the newest older generation we know of.
    known_cc: list[tuple[tuple[int, int], ChipGeneration]] = []
    for g in ChipGeneration:
        cc = g.cuda_cc() if g.is_cuda else None
        if cc is not None:
            known_cc.append((cc, g))
    known_cc.sort(key=lambda x: x[0])
    for cc, gen in reversed(known_cc):
        if cc <= (major, minor):
            return gen
    return ChipGeneration.UNKNOWN


def chip_gen_from_intel_info(info: dict) -> ChipGeneration:
    """Map Intel Vulkan device info to ``ChipGeneration``.

    ``info`` is a dict with ``device_name`` (e.g. ``"Intel(R) Graphics
    (PTL)"``), ``device_id`` (the Vulkan ``deviceID`` int — Intel
    encodes the Xe arch generation in the upper nibble), and optionally
    ``driver_info`` (Mesa version string, useful for forcing a known
    chip when ``deviceID`` is opaque on early hardware).

    Falls back to ``INTEL_XE2`` (the safe baseline that supports
    ``VK_KHR_cooperative_matrix``) when the device name doesn't match a
    known generation; UNKNOWN is reserved for non-Intel devices.

    Generation map (per Mesa's ``intel_device_info`` + Vulkan
    ``deviceID`` upper-nibble convention):
      * 0x9...  → Xe-LPG (Meteor Lake)
      * 0xa...  → Xe2 (Lunar Lake, Battlemage discrete)
      * 0xb...  → Xe3 (Panther Lake, Battlemage+)
    """
    name = str(info.get("device_name", "")).lower()
    device_id = info.get("device_id", 0)
    if isinstance(device_id, int):
        nibble = (device_id >> 12) & 0xF
    else:
        nibble = 0
    # Direct device-id nibble mapping — first source of truth.
    if nibble == 0xB:
        return ChipGeneration.INTEL_XE3
    if nibble == 0xA:
        return ChipGeneration.INTEL_XE2
    if nibble == 0x9:
        return ChipGeneration.INTEL_XE_LPG
    # Name-based fallback for hardware where the deviceID nibble
    # convention doesn't apply yet (e.g. early dev-kit firmware).
    if "(ptl)" in name or "panther lake" in name:
        return ChipGeneration.INTEL_XE3
    if "lunar lake" in name or "(lnl)" in name or "arc b" in name:
        return ChipGeneration.INTEL_XE2
    if "meteor lake" in name or "(mtl)" in name:
        return ChipGeneration.INTEL_XE_LPG
    # Unknown Intel chip — assume Xe2 (the lowest gen with KHR
    # cooperative_matrix exposed). Picking newer would silently skip
    # shape filters that older hardware might not support.
    return ChipGeneration.INTEL_XE2


def chip_gen_from_metal_info(info: dict) -> ChipGeneration:
    """Map Metal device info to ``ChipGeneration``.

    ``info`` is a dict with ``architecture`` (e.g. ``"applegpu_g17s"``)
    and ``device_name`` (e.g. ``"Apple M5 Max"``). We map known
    families; unknowns default to METAL_M3 (bf16 MMA baseline).
    """
    arch = str(info.get("architecture", "")).lower()
    name = str(info.get("device_name", "")).lower()
    hay = f"{arch} {name}"
    # M5 family: applegpu_g17{s,d,g} (s=Max, d=Ultra, g=base/Pro).
    # Also match "m5" in the device_name string as a fallback.
    if "m5" in hay or "applegpu_g17" in hay:
        return ChipGeneration.METAL_M5
    # M3 family and onwards through M4 — all share the bf16-MMA simdgroup
    # surface until M5 adds the new primitive.
    if any(k in hay for k in ("m3", "m4", "applegpu_g15", "g15d", "g15c", "g15s")):
        return ChipGeneration.METAL_M3
    if any(k in hay for k in ("m1", "m2", "applegpu_g13", "applegpu_g14")):
        return ChipGeneration.METAL_M1
    # Newer unknown Apple chip — assume at least the M3-class baseline.
    return ChipGeneration.METAL_M3


# ---------------------------------------------------------------------------
# DeviceCaps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceCaps:
    """Hardware capability snapshot. Frozen — populated once at probe
    time. Every kernel `is_valid_for(caps)` consults this to gate its
    config against the actual device.

    Fields ordered per the proposal §3.1. `None` means "not queryable
    on this backend" — kernels that need that field must guard the
    None case explicitly.
    """

    # Universal fields (every backend populates)
    family: DeviceFamily
    name: str
    compute_unit_count: int  # SMs / CUs / cores / logical CPUs
    # Execution-lane grouping width. 32 on NVIDIA/CDNA/Apple, 32-or-64 on
    # RDNA, 1 on CPU, 8/16/32 on Intel (driver-chosen). Vendor-neutral name
    # per the portability plan; see ``warp_size`` property below for the
    # legacy alias.
    subgroup_width: int
    max_threads_per_block: int
    max_smem_per_block: int  # bytes, dynamic
    max_regs_per_thread: Optional[int]
    max_regs_per_block: Optional[int]

    # Arch / version
    arch_tag: str  # "sm_120" / "gfx1100" / "metal3" / "x86_64-avx512bf16"
    compute_capability: Optional[tuple[int, int]]

    # Feature flags
    supports_async_copy: bool  # cp.async (Ampere+)
    supports_graph_capture: bool  # CUDA graphs / HIP graphs
    supports_fp8_e4m3: bool  # Hopper+, MI300+, M3+, SPR-AMX
    supports_bf16_mma: bool  # hardware bf16 matmul

    # Upper bound on total IR op count in a single compiled function.
    # A good proxy for the MSL source's LoC / SSA variable count / JIT
    # compile time — Metal's Apple-provided shader compiler scales
    # poorly on very large kernels (empirically >~3000 ops per function
    # stalls for seconds). `ptxas` handles big kernels fine so CUDA
    # leaves this `None`. Enforced by `Kernel.is_valid_for` by walking
    # the emitted Module and counting ops in every function body.
    max_ir_ops: Optional[int] = None

    # Upper bound on the number of distinct MMA accumulator chains in
    # a compiled function — equivalently, the per-warp simdgroup_matrix
    # accumulator grid size (mt × nt_per_warp for a gemm-style kernel).
    # Metal's shader compiler scales poorly with the number of live
    # simdgroup_matrix registers, so configs whose accumulator grid
    # blows up past ~4-8 tiles take seconds-to-minutes to JIT even
    # when the total op count is modest. CUDA: `None` (no limit).
    max_mma_accumulator_tiles: Optional[int] = None

    # The set of MmaShape `shape_id`s the backend lowerer can emit.
    # `Kernel.is_valid_for` walks the emitted Module and rejects any
    # MmaOp / LoadMatrixOp / StoreMatrixOp whose shape_id isn't here.
    matmul_shapes: frozenset[str] = field(default_factory=frozenset)

    # Scalar atomic-add dtypes. sm_89 has no sub-4-byte atomic;
    # Metal has only 32-bit atomic ops (f32 / s32 / u32 — no f16, no
    # bf16). Empty means "no atomic support"; filters every kernel
    # with an AtomicRmwOp.
    atomic_add_dtypes: frozenset = field(default_factory=frozenset)

    # Vector atomic-add ops: (dtype, lanes) pairs the hardware supports
    # natively. Examples:
    #   (F16, 2)  — ``red.add.noftz.f16x2``  on sm_70+ (Volta)
    #   (BF16, 2) — ``red.add.noftz.bf16x2`` on sm_90+ (Hopper)
    # Metal and older CUDA: empty. Lets epilogues autotune between
    # scalar and vector-atomic paths gated on real hardware support.
    atomic_add_vector: frozenset[tuple] = field(default_factory=frozenset)

    # Packed bf16×2 FMA (``fma.rn.bf16x2`` on CUDA sm_80+). Distinct
    # from the vector-atomic-add flag above: this gates the *compute*
    # op, which ships one generation earlier than the vector-atomic
    # red-op. Non-CUDA backends leave this False — the legalization
    # pass rewrites ``ArithOp(kind="fma_bf16x2")`` and
    # ``ArithOp(kind="cvt_rn_bf16x2_f32")`` into scalar F32 chains
    # when this is False. See PORTABILITY_PLAN.md §2.2.
    has_fma_bf16x2: bool = False

    # Packed bf16×2 atomic add (``red.add.noftz.bf16x2`` on CUDA
    # sm_90+). Convenience alias of ``(BF16, 2) in atomic_add_vector``
    # — kept as a named bool so the legalization pass call sites read
    # like ``if not caps.has_atomic_add_bf16x2:`` instead of
    # tuple-membership checks. Populated by the driver probe.
    has_atomic_add_bf16x2: bool = False

    # Metal 4 API family support (macOS 26+). Gates MSL 4.0 language
    # version at compile time (required for MetalPerformancePrimitives
    # includes). Non-Metal backends: False.
    supports_metal4: bool = False

    # Neural Accelerator (NAX) hardware available — Apple GPU gen >= 17
    # (Mac) or >= 18 (iPhone). Gates MPP matmul2d emission in the
    # lowerer. Implies supports_metal4. Non-Metal backends: False.
    supports_nax: bool = False

    # Native single-op subgroup reduction (``simd_sum`` / ``simd_max`` on
    # Metal; ``OpGroupNonUniformAdd`` on SPIR-V). CUDA's only primitive
    # is ``shfl.sync.bfly.b32``, so the PTX lowerer expands
    # :class:`SubgroupReduceOp` inline to a butterfly chain — that's
    # ``has_native_subgroup_reduce=False``. When a future backend with
    # a native reduce arrives, the legalization pass keeps the op as-is
    # and the lowerer emits the intrinsic; backends without the native
    # form get the butterfly expansion from the legalize pass. See
    # PORTABILITY_PLAN.md §2.2 / §3.
    has_native_subgroup_reduce: bool = False

    supported_dtypes: frozenset[str] = field(default_factory=frozenset)

    # CPU-specific ISA feature set (empty for GPU)
    cpu_features: frozenset[str] = field(default_factory=frozenset)

    # Finer-grained capability key. Drives ``matmul_shapes`` population
    # and any future per-generation kernel gates (Blackwell block-MMA,
    # M5 tensor-core primitive). Default UNKNOWN so hand-built test
    # DeviceCaps don't have to set it explicitly.
    chip_gen: ChipGeneration = ChipGeneration.UNKNOWN

    def has(self, feature: str) -> bool:
        """Check whether a CPU ISA feature flag is set."""
        return feature in self.cpu_features

    @property
    def warp_size(self) -> int:
        """Deprecated alias for ``subgroup_width``. Kept for one release
        so existing call sites (`caps.warp_size`) continue to work while
        downstream code migrates to the vendor-neutral name."""
        return self.subgroup_width


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    """A physical compute device — family + ordinal + populated caps."""

    family: DeviceFamily
    index: int
    caps: DeviceCaps

    def fingerprint(self) -> str:
        """Stable string used as the device portion of autotune cache
        keys. Two machines with the same family but different
        arch / CU count never share cache entries."""
        return f"{self.family.value}_{self.caps.arch_tag}_{self.caps.compute_unit_count}cu"


# ---------------------------------------------------------------------------
# Detection — current_device()
# ---------------------------------------------------------------------------


_FORCE_BACKEND_ENV = "QUARK_FORCE_BACKEND"


def _forced_family() -> DeviceFamily | None:
    """Honor `QUARK_FORCE_BACKEND` if set; raise on garbage values."""
    forced = os.environ.get(_FORCE_BACKEND_ENV)
    if forced is None:
        return None
    try:
        return DeviceFamily(forced.lower())
    except ValueError as e:
        raise ValueError(
            f"{_FORCE_BACKEND_ENV}={forced!r} is not a valid family. "
            f"Allowed: {[f.value for f in DeviceFamily]}"
        ) from e


def _detect_family() -> DeviceFamily:
    """Auto-detect the active backend.

    Detection order (first match wins):
      1. CUDA      (libcuda probe finds a device)
      2. Metal     (metal-cpp driver detects a device)
      3. INTEL_GPU (Vulkan / SPV probe finds a device — Linux iGPU)
      4. CPU       (always available as a fallback)

    Intel SPV is below CUDA in the auto-detect order so a CUDA box
    with both libcuda and Vulkan available stays on CUDA. Force
    via ``QUARK_FORCE_BACKEND=intel_gpu`` to override on dual-driver
    machines (or any other family).

    ROCm / OpenCL aren't auto-detected; force via ``QUARK_FORCE_BACKEND``
    and plug a dedicated probe when a user lands on that path.
    """
    forced = _forced_family()
    if forced is not None:
        return forced
    # Probe libcuda via ctypes — doesn't require torch to be installed.
    try:
        from quark.runtime.cuda import CudaRuntime

        if CudaRuntime.instance().device_count() > 0:
            return DeviceFamily.CUDA
    except Exception:
        pass
    try:
        from quark.drivers.metal import is_available

        if is_available():
            return DeviceFamily.METAL
    except Exception:
        pass
    try:
        from quark.drivers import spv as _spv_drivers

        if _spv_drivers.is_available():
            return DeviceFamily.INTEL_GPU
    except Exception:
        pass
    return DeviceFamily.CPU


@cache
def current_device() -> Device:
    """Detect the active backend and return a `Device` with populated
    capabilities. The result is cached per-process — repeated calls
    are free after the first.

    For Bundle 2 (this commit), only the CUDA path produces a real
    populated `DeviceCaps`. Other families raise `NotImplementedError`
    at probe time. Bundle 3 wires CUDA through the `CudaDriver.probe`
    method that talks to libcuda directly; until then we read what we
    can from torch's CUDA properties so kernels can still ask the
    questions they care about.
    """
    family = _detect_family()
    if family is DeviceFamily.CUDA:
        return _probe_cuda_via_libcuda(index=0)
    if family is DeviceFamily.METAL:
        return _probe_metal(index=0)
    if family is DeviceFamily.INTEL_GPU:
        # SpvDriver.__init__ runs ``ensure_device`` and populates a
        # full DeviceCaps via ``_spv_dispatch.probe`` — reuse it.
        from quark.drivers import spv as _spv_drivers

        return _spv_drivers.SpvDriver().device
    raise NotImplementedError(
        f"current_device(): family {family.value!r} not yet implemented. "
        f"Set {_FORCE_BACKEND_ENV}=cuda or {_FORCE_BACKEND_ENV}=metal."
    )


def _probe_cuda_via_libcuda(index: int) -> Device:
    """CUDA probe through libcuda ctypes — no torch dependency."""
    from quark.runtime.cuda import (
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        CudaRuntime,
    )

    rt = CudaRuntime.instance()
    if rt.device_count() == 0:
        raise RuntimeError("CUDA probe: no CUDA devices visible to libcuda")

    cc_major = rt.get_device_attribute(index, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    cc_minor = rt.get_device_attribute(index, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    arch_tag = f"sm_{cc_major}{cc_minor}"
    try:
        smem_optin = rt.get_device_attribute(
            index, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )
    except Exception:
        smem_optin = 49152
    caps = DeviceCaps(
        family=DeviceFamily.CUDA,
        name=rt.get_device_name(index),
        compute_unit_count=rt.get_device_attribute(index, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        subgroup_width=32,
        max_threads_per_block=rt.get_device_attribute(
            index, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        ),
        max_smem_per_block=smem_optin,
        max_regs_per_thread=255,
        max_regs_per_block=None,
        arch_tag=arch_tag,
        compute_capability=(cc_major, cc_minor),
        supports_async_copy=cc_major >= 8,  # Ampere+
        supports_graph_capture=True,
        supports_fp8_e4m3=cc_major >= 9 or (cc_major == 8 and cc_minor == 9),
        supports_bf16_mma=cc_major >= 8,
        matmul_shapes=_shapes_for_chip_gen(chip_gen_from_cuda_cc(cc_major, cc_minor)),
        atomic_add_dtypes=_atomic_add_dtypes_for_cuda(cc_major, cc_minor),
        atomic_add_vector=_atomic_add_vector_for_cuda(cc_major, cc_minor),
        # Compute / atomic packed-bf16 gates. Ampere+ (sm_80) exposes
        # ``fma.rn.bf16x2``; Hopper+ (sm_90) exposes the vector atomic.
        has_fma_bf16x2=cc_major >= 8,
        has_atomic_add_bf16x2=cc_major >= 9,
        supported_dtypes=_default_dtypes_for(arch_tag),
        cpu_features=frozenset(),
        chip_gen=chip_gen_from_cuda_cc(cc_major, cc_minor),
    )
    return Device(family=DeviceFamily.CUDA, index=index, caps=caps)


def _shapes_for_chip_gen(gen: ChipGeneration) -> frozenset[str]:
    """Lazy indirection to ``quark.ir.mma_registry.shapes_for_chip`` —
    avoids a hard import cycle (registry imports ChipGeneration from us).
    """
    from quark.ir.mma_registry import shapes_for_chip

    return shapes_for_chip(gen)


def _atomic_add_dtypes_for_cuda(cc_major: int, cc_minor: int) -> frozenset:
    """CUDA scalar atomic-add dtype support by compute capability."""
    from quark.ir import DType

    # f32 / s32 / u32 are atomic on every arch Quark targets (sm_60+).
    dts = {DType.F32, DType.S32, DType.U32}
    # f16 scalar atomic add lands in sm_70 (Volta) — same generation
    # that added f16x2 vector atomics.
    if cc_major >= 7:
        dts.add(DType.F16)
    # bf16 scalar atomic-add: Blackwell (sm_100+) only. Hopper (sm_90)
    # exposes bf16x2 red.add but not a clean scalar bf16 atomic — we
    # gate the whole dtype here so split-K GEMMs with bf16 output only
    # enumerate on arches where atomic accumulation is actually cheap.
    if cc_major >= 10:
        dts.add(DType.BF16)
    return frozenset(dts)


def _atomic_add_vector_for_cuda(cc_major: int, cc_minor: int) -> frozenset[tuple]:
    """CUDA vector atomic-add support. Each entry is (dtype, lanes)."""
    from quark.ir import DType

    caps: set[tuple] = set()
    if cc_major >= 7:
        # red.add.noftz.f16x2 — Volta+. Reliable way to atomic-accumulate
        # packed fp16 epilogues.
        caps.add((DType.F16, 2))
    if cc_major >= 9:
        # red.add.noftz.bf16x2 — Hopper+ (sm_90). The vector form ships
        # two generations later than f16x2: ptxas flatly rejects it on
        # sm_89/sm_80 ("requires sm_90 or higher"). Used by the paired
        # scatter path to accumulate split-K / MoE epilogues into bf16
        # output at half the atomic op count.
        caps.add((DType.BF16, 2))
    return frozenset(caps)


def _default_atomic_add_dtypes_for(family: DeviceFamily, arch_tag: str) -> frozenset:
    """atomic_add dtype set for make_test_device's defaults."""
    from quark.ir import DType

    if family is DeviceFamily.METAL:
        # Metal has only 32-bit atomic ops — f32 / s32 / u32.
        return frozenset({DType.F32, DType.S32, DType.U32})
    if family in (DeviceFamily.CUDA, DeviceFamily.ROCM):
        if arch_tag.startswith("sm_"):
            try:
                major = int(arch_tag[3])
                minor = int(arch_tag[4])
                return _atomic_add_dtypes_for_cuda(major, minor)
            except (ValueError, IndexError):
                pass
        return frozenset({DType.F32, DType.S32, DType.U32})
    return frozenset({DType.F32, DType.S32, DType.U32})


def _probe_metal(index: int) -> Device:
    """Metal probe via the native metal-cpp + nanobind driver."""
    from quark.drivers.metal import MetalDriver

    caps = MetalDriver().probe(index)
    return Device(family=DeviceFamily.METAL, index=index, caps=caps)


# ---------------------------------------------------------------------------
# Per-arch dtype table
# ---------------------------------------------------------------------------
#
# MMA shape tables moved to ``quark.ir.mma_registry``. One source of
# truth for descriptors + chip gates; ``matmul_shapes`` is computed via
# ``shapes_for_chip(chip_gen)`` at probe time.

_DTYPES_BY_ARCH: dict[str, frozenset[str]] = {
    "sm_75": frozenset({"f32", "f16", "s32", "s8", "u8"}),
    "sm_80": frozenset({"f32", "f16", "bf16", "s32", "s8", "u8"}),
    "sm_86": frozenset({"f32", "f16", "bf16", "s32", "s8", "u8"}),
    "sm_89": frozenset({"f32", "f16", "bf16", "e4m3", "e5m2", "s32", "s8", "u8"}),
    "sm_90": frozenset({"f32", "f16", "bf16", "e4m3", "e5m2", "s32", "s8", "u8"}),
    "sm_100": frozenset({"f32", "f16", "bf16", "e4m3", "e5m2", "s32", "s8", "u8"}),
    "sm_120": frozenset({"f32", "f16", "bf16", "e4m3", "e5m2", "s32", "s8", "u8"}),
}


def _default_dtypes_for(arch_tag: str) -> frozenset[str]:
    return _DTYPES_BY_ARCH.get(arch_tag, frozenset({"f32", "f16"}))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_test_device(
    *,
    family: DeviceFamily = DeviceFamily.CUDA,
    arch_tag: str = "sm_89",
    name: str = "test-device",
    compute_unit_count: int = 84,
    warp_size: int = 32,
    max_threads_per_block: int = 1024,
    max_smem_per_block: int = 100 * 1024,
    matmul_shapes: frozenset[str] | None = None,
    chip_gen: ChipGeneration | None = None,
) -> Device:
    """Build a `Device` with reasonable defaults for unit tests so
    kernel `is_valid_for(caps)` paths can be exercised without
    needing real hardware. Caller can override any field."""
    if chip_gen is None:
        if arch_tag.startswith("sm_"):
            try:
                cc_major = int(arch_tag[3])
                cc_minor = int(arch_tag[4])
                chip_gen = chip_gen_from_cuda_cc(cc_major, cc_minor)
            except (ValueError, IndexError):
                chip_gen = ChipGeneration.UNKNOWN
        elif arch_tag.startswith("metal"):
            chip_gen = ChipGeneration.METAL_M3
        else:
            chip_gen = ChipGeneration.UNKNOWN
    caps = DeviceCaps(
        family=family,
        name=name,
        compute_unit_count=compute_unit_count,
        subgroup_width=warp_size,
        max_threads_per_block=max_threads_per_block,
        max_smem_per_block=max_smem_per_block,
        max_regs_per_thread=255 if family is not DeviceFamily.METAL else None,
        max_regs_per_block=65536 if family is not DeviceFamily.METAL else None,
        arch_tag=arch_tag,
        compute_capability=(int(arch_tag[3]), int(arch_tag[4]))
        if arch_tag.startswith("sm_")
        else None,
        supports_async_copy=family is not DeviceFamily.METAL,
        supports_graph_capture=family is not DeviceFamily.METAL,
        supports_fp8_e4m3=arch_tag in ("sm_89", "sm_90", "sm_100", "sm_120"),
        supports_bf16_mma=True,
        max_ir_ops=1500 if family is DeviceFamily.METAL else None,
        max_mma_accumulator_tiles=8 if family is DeviceFamily.METAL else None,
        matmul_shapes=matmul_shapes
        if matmul_shapes is not None
        else _shapes_for_chip_gen(chip_gen),
        atomic_add_dtypes=_default_atomic_add_dtypes_for(family, arch_tag),
        supported_dtypes=_default_dtypes_for(arch_tag),
        cpu_features=frozenset(),
        chip_gen=chip_gen,
    )
    return Device(family=family, index=0, caps=caps)
