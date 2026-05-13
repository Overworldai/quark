"""MMA shape registry — the single source of truth.

EXEMPT FROM 500-LINE RULE: this file is the canonical per-shape
table — one ``MmaConfig`` row per MMA Quark can lower, carrying
layout offsets + chip gates + per-backend lowering payloads
(``cuda``, ``metal``, …). Splitting means "add a new MMA" returns
to a multi-file edit, which is the exact drift this file exists
to prevent. Hard cap 800 still applies.

Collects every MMA descriptor Quark can lower, along with the
per-chip gates that determine which hardware supports each shape.
Replaces two parallel tables that had drifted:

* ``kernels/gemm/mma_shapes._MMA_TABLE`` (descriptors + offset tables)
* ``device._MMA_SHAPES_BY_ARCH`` (per-arch shape-name sets)

Both are now derived from the list of descriptors below via
``shapes_for_chip(gen)``. Add a new MMA by writing one ``MmaConfig``
with its ``min_ptx_cc`` / ``min_metal_gen`` gates; everything else
(``lookup_mma``, ``DeviceCaps.matmul_shapes``, autotune enumeration)
picks it up automatically.

Per MMA_SHAPES proposal Stage M1.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from quark.device import ChipGeneration, DeviceFamily
from quark.ir.module import MmaShape
from quark.ir.types import DType


@dataclass(frozen=True)
class MmaConfig:
    """One MMA descriptor + its per-register lane metadata + per-backend
    gate & lowering payload.

    ``shape`` is the IR-level descriptor the lowerer consumes.
    ``a_offsets`` / ``b_offsets`` / ``cd_offsets`` are the per-register
    (row, col) offsets in the PTX fragment layout (see PTX ISA §9.7.14.5).
    ``lane_col_step`` is the elements-per-b32 count: 2 for bf16/fp16,
    4 for 8-bit.

    Backend fields come in **gate + payload pairs**, grouped by family.
    The payload field name equals ``DeviceFamily.<X>.value`` so
    ``payload_for(shape_id, family)`` can resolve it with ``getattr``.

    * ``min_cuda_cc`` + ``cuda``    — CUDA  (PTX mnemonic suffix)
    * ``min_metal_gen`` + ``metal`` — Metal (MSL tiling
      ``"<frag_dtype>:<mf>:<nf>:<kf>"`` consumed by ``_parse_msl_tiling``)
    * ``min_intel_gpu_gen`` + ``intel_gpu`` — Intel iGPU/Arc via OCL
      + IGC. The payload is a tag the OCL lowerer parses; today the
      only emitted form is ``"khr:subgroup"`` (subgroup-scope
      cooperative-matrix, mapping to ``cl_intel_subgroup_matrix_
      multiply_accumulate`` at IGC compile time).

    Gate semantics: ``min_*=None`` means the shape has no path on that
    backend at all. ``shapes_for_chip`` filters the descriptor list
    against the active device's ``ChipGeneration``: CUDA chips pass iff
    ``chip.cuda_cc() >= min_cuda_cc``; Metal chips pass iff
    ``chip >= min_metal_gen`` in the Metal-gen declaration order;
    Intel chips pass iff ``chip >= min_intel_gpu_gen`` in the
    Xe-gen declaration order. A descriptor with every gate ``None``
    is rejected by any chip (mis-registered).

    Payload semantics: the value is a backend-opaque string that the
    lowerer parses. ``None`` (default) means no lowering — if the gate
    passes but the payload is ``None``, lowering fails with a clear
    error (see ``payload_for``'s callers).

    ``min_ptx_cc`` is kept as a read-only property alias of
    ``min_cuda_cc`` for one release so downstream dashboards / registries
    that introspect the field keep working. Prefer the vendor-neutral
    name at new call sites.
    """

    shape: MmaShape
    a_offsets: tuple[tuple[int, int], ...]
    b_offsets: tuple[tuple[int, int], ...]
    cd_offsets: tuple[tuple[int, int], ...]
    lane_col_step: int  # elements per b32 reg — drives smem-base math

    # --- CUDA backend ---
    min_cuda_cc: Optional[tuple[int, int]] = None
    cuda: Optional[str] = None  # PTX mnemonic suffix after ``mma.sync.aligned.``

    # --- Metal backend ---
    min_metal_gen: Optional[ChipGeneration] = None
    metal: Optional[str] = None  # MSL tiling ``"<frag_dtype>:<mf>:<nf>:<kf>"``

    # --- Intel iGPU/Arc backend (OpenCL + IGC, cl_intel_subgroup_matrix_multiply_accumulate) ---
    min_intel_gpu_gen: Optional[ChipGeneration] = None
    intel_gpu: Optional[str] = None  # tag, e.g. ``"khr:subgroup"``

    @property
    def shape_id(self) -> str:
        return self.shape.name

    @property
    def mma_k(self) -> int:
        return self.shape.k

    @property
    def min_ptx_cc(self) -> Optional[tuple[int, int]]:
        """Deprecated alias for ``min_cuda_cc``."""
        return self.min_cuda_cc


# ---------------------------------------------------------------------------
# bf16 × bf16 → f32 (m8n8k8) — Metal-native direct simdgroup_matrix
#
# One simdgroup_matrix<bf16, 8, 8> multiply, no internal tiling. The
# existing MSL lowerer's m_frags*n_frags*k_frags iteration collapses
# to a single simdgroup_multiply_accumulate when all three are 1.
#
# Not emittable on PTX (ptx=None) — it would map to mma.m8n8 variants
# that Quark's PTX lowerer doesn't support. Metal-only by design;
# hypothesis (proposal §7.1): direct k=8 emission wins at large BK
# against the current k=16 path which two-tiles internally.
# ---------------------------------------------------------------------------

_BF16_M8N8K8 = MmaConfig(
    shape=MmaShape(
        name="m8n8k8_bf16",
        m=8,
        n=8,
        k=8,
        a_dtype=DType.BF16,
        b_dtype=DType.BF16,
        acc_dtype=DType.F32,
        a_regs=1,  # 1 bf16x2 b32 per lane × 32 lanes = 64 = 8m × 8k
        b_regs=1,  # 1 bf16x2 b32 per lane × 32 lanes = 64 = 8n × 8k
        c_regs=2,  # 2 f32 per lane × 32 lanes = 64 = 8m × 8n
    ),
    a_offsets=((0, 0),),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1)),  # Apple 8x8 acc: 2 elements per lane
    lane_col_step=2,
    min_cuda_cc=None,  # no PTX path
    cuda=None,
    min_metal_gen=ChipGeneration.METAL_M3,
    metal="bfloat:1:1:1",
)

# ---------------------------------------------------------------------------
# bf16 × bf16 → f32 (m16n8k8) — PTX-native, smaller-K variant
#
# Halves the k per mma.sync vs m16n8k16; callers iterate 2x more but
# with half the live a-fragment register pressure per inner loop.
# Worth an autotune sweep on sm_80+ where it's been available the
# whole time but quark never registered it.
# ---------------------------------------------------------------------------

_BF16_M16N8K8 = MmaConfig(
    shape=MmaShape(
        name="m16n8k8_bf16",
        m=16,
        n=8,
        k=8,
        a_dtype=DType.BF16,
        b_dtype=DType.BF16,
        acc_dtype=DType.F32,
        a_regs=2,  # 2 bf16x2 b32 = 4 bf16 per lane = 16m × 8k
        b_regs=1,  # 1 bf16x2 b32 = 2 bf16 per lane = 8n × 8k
        c_regs=4,  # same acc layout as all m16n8 shapes
    ),
    a_offsets=((0, 0), (8, 0)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,
    min_cuda_cc=(8, 0),  # Ampere+
    cuda="m16n8k8.row.col.f32.bf16.bf16.f32",
    # No MSL emission for m16n8 on Metal — use m8n8k8 or the
    # existing m16n8k16 (which internally tiles to 2x simdgroup_matrix).
    min_metal_gen=None,
)

# ---------------------------------------------------------------------------
# bf16 × bf16 → f32 (m16n8k16)
# ---------------------------------------------------------------------------

_BF16_K16 = MmaConfig(
    shape=MmaShape(
        name="m16n8k16_bf16",
        m=16,
        n=8,
        k=16,
        a_dtype=DType.BF16,
        b_dtype=DType.BF16,
        acc_dtype=DType.F32,
        a_regs=4,
        b_regs=2,
        c_regs=4,
    ),
    a_offsets=((0, 0), (8, 0), (0, 8), (8, 8)),
    b_offsets=((0, 0), (0, 8)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,
    min_cuda_cc=(8, 0),  # Ampere+
    cuda="m16n8k16.row.col.f32.bf16.bf16.f32",
    # No Metal in shapes_for_chip — m8n8k8 dominates on every M3 autotune
    # problem, so keeping m16n8k16 in Metal's shape set just wastes
    # autotune time. But the MSL lowerer still supports the shape if a
    # future kernel opts in explicitly via ``main_shape``, so we carry
    # the MSL payload for that path.
    min_metal_gen=None,
    metal="bfloat:2:1:2",
)

# ---------------------------------------------------------------------------
# fp16 × fp16 → f32
# ---------------------------------------------------------------------------

_F16_K16 = MmaConfig(
    shape=MmaShape(
        name="m16n8k16_f16",
        m=16,
        n=8,
        k=16,
        a_dtype=DType.F16,
        b_dtype=DType.F16,
        acc_dtype=DType.F32,
        a_regs=4,
        b_regs=2,
        c_regs=4,
    ),
    a_offsets=((0, 0), (8, 0), (0, 8), (8, 8)),
    b_offsets=((0, 0), (0, 8)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,
    min_cuda_cc=(7, 5),  # Turing+ (m16n8k16 f16 MMA)
    cuda="m16n8k16.row.col.f32.f16.f16.f32",
    # No Metal in shapes_for_chip — see m16n8k16_bf16 note.
    # m8n8k8_f16 would be the Metal-native choice; register when a
    # kernel actually wants fp16. Carry the MSL payload for
    # ``main_shape`` opt-in; the m_frags/n_frags/k_frags tuple is
    # identical to bf16 since only the frag-element type changes.
    min_metal_gen=None,
    metal="half:2:1:2",
)

# ---------------------------------------------------------------------------
# e4m3 × e4m3 → f32 (m16n8k16)
# ---------------------------------------------------------------------------

_E4M3_K16 = MmaConfig(
    shape=MmaShape(
        name="m16n8k16_e4m3",
        m=16,
        n=8,
        k=16,
        a_dtype=DType.E4M3,
        b_dtype=DType.E4M3,
        acc_dtype=DType.F32,
        a_regs=2,
        b_regs=1,
        c_regs=4,
    ),
    a_offsets=((0, 0), (8, 0)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    # m16n8k16 fp8 requires PTX ISA 8.7+ (Blackwell sm_120). Ada
    # (sm_89) only supports fp8 via m16n8k32 — see _E4M3_K32.
    min_cuda_cc=(12, 0),
    cuda="m16n8k16.row.col.f32.e4m3.e4m3.f32",
)

# ---------------------------------------------------------------------------
# e5m2 × e5m2 → f32 (m16n8k16)
# ---------------------------------------------------------------------------

_E5M2_K16 = MmaConfig(
    shape=MmaShape(
        name="m16n8k16_e5m2",
        m=16,
        n=8,
        k=16,
        a_dtype=DType.E5M2,
        b_dtype=DType.E5M2,
        acc_dtype=DType.F32,
        a_regs=2,
        b_regs=1,
        c_regs=4,
    ),
    a_offsets=((0, 0), (8, 0)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    min_cuda_cc=(12, 0),  # Blackwell — Ada has fp8 MMAs only at k=32
    cuda="m16n8k16.row.col.f32.e5m2.e5m2.f32",
)

# ---------------------------------------------------------------------------
# e4m3 × e4m3 → f32 (m16n8k32)
# ---------------------------------------------------------------------------

_E4M3_K32 = MmaConfig(
    shape=MmaShape(
        name="m16n8k32_e4m3",
        m=16,
        n=8,
        k=32,
        a_dtype=DType.E4M3,
        b_dtype=DType.E4M3,
        acc_dtype=DType.F32,
        a_regs=4,
        b_regs=2,
        c_regs=4,
    ),
    a_offsets=((0, 0), (8, 0), (0, 16), (8, 16)),
    b_offsets=((0, 0), (0, 16)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    min_cuda_cc=(8, 9),
    cuda="m16n8k32.row.col.f32.e4m3.e4m3.f32",
)

# ---------------------------------------------------------------------------
# e5m2 × e5m2 → f32 (m16n8k32)
# ---------------------------------------------------------------------------

_E5M2_K32 = MmaConfig(
    shape=MmaShape(
        name="m16n8k32_e5m2",
        m=16,
        n=8,
        k=32,
        a_dtype=DType.E5M2,
        b_dtype=DType.E5M2,
        acc_dtype=DType.F32,
        a_regs=4,
        b_regs=2,
        c_regs=4,
    ),
    a_offsets=((0, 0), (8, 0), (0, 16), (8, 16)),
    b_offsets=((0, 0), (0, 16)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    min_cuda_cc=(8, 9),
    cuda="m16n8k32.row.col.f32.e5m2.e5m2.f32",
)

# ---------------------------------------------------------------------------
# Cross-dtype: bf16 × e4m3 → f32 (mixed-precision)
# PTX mma.sync supports mixed A/B dtypes on sm_89+.
# ---------------------------------------------------------------------------

_BF16xE4M3_K16 = MmaConfig(
    shape=MmaShape(
        name="m16n8k16_bf16_e4m3",
        m=16,
        n=8,
        k=16,
        a_dtype=DType.BF16,
        b_dtype=DType.E4M3,
        acc_dtype=DType.F32,
        a_regs=4,
        b_regs=1,
        c_regs=4,
    ),
    a_offsets=((0, 0), (8, 0), (0, 8), (8, 8)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,  # bf16 A dictates the lane stride (wider)
    # Mixed bf16×e4m3 at m16n8k16 uses the kind::f8f6f4 encoding
    # added in PTX ISA 8.7 (sm_120+). Ada/Hopper have no native mixed
    # form here — a precast-to-fp8 path would use _E4M3_K32 on those.
    min_cuda_cc=(12, 0),
    cuda="m16n8k16.row.col.f32.bf16.e4m3.f32",
)


# ---------------------------------------------------------------------------
# Descriptor list + lookup table (derived)
# ---------------------------------------------------------------------------

# Every descriptor known to Quark. Adding a new MMA means adding one
# row here; ``shapes_for_chip`` / ``lookup_mma`` / ``_MMA_TABLE`` all
# read from this list.
# ---------------------------------------------------------------------------
# bf16 × bf16 → f32 (m16n32k16) — NAX MPP matmul2d on Metal 4+ (M5 Max etc.)
#
# Uses Apple's MetalPerformancePrimitives mpp::tensor_ops::matmul2d which
# computes 16×32×16 per simdgroup in one hardware instruction. 16× more
# compute per instruction than the 8×8×8 simdgroup_matrix path.
#
# Fragment layout: 16×16 per fragment, 8 elements per thread (2 rows × 4 cols),
# kElemRowsJump=8. See MLX's nax.h BaseNAXFrag for the lane map:
#   fm = ((qid & 4) | ((lane >> 1) & 3))
#   fn = ((qid & 2) | (lane & 1)) * 4
# Two 16×16 fragments tile the 16×32 output (TN=2).
#
# Not emittable on PTX — Metal-only (NAX hardware + Metal 4 API).
# ---------------------------------------------------------------------------

_BF16_M16N32K16_NAX = MmaConfig(
    shape=MmaShape(
        name="m16n32k16_nax_bf16",
        m=16,
        n=32,
        k=16,
        a_dtype=DType.BF16,
        b_dtype=DType.BF16,
        acc_dtype=DType.F32,
        # Per-thread registers: 8 elements per 16×16 fragment.
        # A is TM×TK fragments (1×1 for the base unit),
        # B is TK×TN fragments (1×2: 16×32 output needs 2 B-tiles of 16×16).
        a_regs=8,
        b_regs=16,  # 8 per 16×16 tile × 2 tiles for N=32
        c_regs=16,  # 8 per 16×16 output tile × 2 tiles for N=32
    ),
    # NAX fragment offsets — per-lane element coordinates within the
    # full MMA tile. The NAX MSL visitor uses its own coordinate map
    # (BaseNAXFrag fm/fn formulas) so these aren't consulted directly,
    # but ``LoadMatrixOp`` / ``StoreMatrixOp`` validators check that
    # ``len(reg_offsets) == fragment_width``. The d/c (16×32) and b
    # (16×32 transpose_b) fragments are 2 BaseNAXFrag sub-tiles wide;
    # spell out the per-lane offsets for both so callers can pass these
    # directly without the validator rejecting the 16-wide fragment.
    #
    # Per BaseNAXFrag: per lane covers 2 rows × 4 cols, rows jump by 8.
    # A is 1 sub-tile (16×16), so 8 per-lane positions.
    # B (transpose_b, [N, K]) is 2 sub-tiles split along N (row): frag 0 at
    # row 0..15, frag 1 at row 16..31; per-lane K-cols 0..3 in each.
    # C/D ([M, N]) are 2 sub-tiles split along N (col): frag 0 at col 0..15,
    # frag 1 at col 16..31; per-lane rows 0/8 within each.
    a_offsets=tuple((r * 8, c) for r in range(2) for c in range(4)),
    b_offsets=tuple(
        (frag_r * 16 + r * 8, c) for frag_r in range(2) for r in range(2) for c in range(4)
    ),
    cd_offsets=tuple(
        (r * 8, frag_c * 16 + c) for frag_c in range(2) for r in range(2) for c in range(4)
    ),
    lane_col_step=4,  # 4 cols per thread per fragment row
    min_cuda_cc=None,  # no PTX path
    cuda=None,
    min_metal_gen=ChipGeneration.METAL_M5,
    metal="nax:1:2:1",  # frag_dtype:TM:TN:TK — tells the MSL visitor to use MPP
)


# ---------------------------------------------------------------------------
# bf16 × bf16 → f32 (m32n32k16) — NAX MPP matmul2d, larger M-fragment.
#
# Apple's matmul2d_descriptor accepts arbitrary (M, N, K) at the threadgroup
# tile level — m16n32k16 is the per-MMA-instruction fragment, and bigger
# tiles get decomposed into multiple m16 fragments by Apple's compiler.
# This descriptor doubles the M dimension to 32. With single-simdgroup
# execution, Apple emits 2 stacked m=16 fragments → per-lane regs double:
#   A: 2 stacked 16×16 = 16 lane elements
#   B: same as m=16 (N unchanged) = 16 lane elements
#   C: 2 stacked × 2 N-tiled = 4 sub-fragments = 32 lane elements
#
# Per-lane fragment offsets follow the same BaseNAXFrag layout (fm, fn)
# repeated per sub-fragment, just with M offset 16 for the stacked half.
# Confirmed via probe in the world_engine MLX path (M32NAXFrag in nax_m32.h)
# — same Apple cooperative_tensor distribution rule applies here.
#
# Single-simdgroup; for execution_simdgroups<N> support add a separate
# entry with proportional output size + n_simdgroups field.
# ---------------------------------------------------------------------------

_BF16_M32N32K16_NAX = MmaConfig(
    shape=MmaShape(
        name="m32n32k16_nax_bf16",
        m=32,
        n=32,
        k=16,
        a_dtype=DType.BF16,
        b_dtype=DType.BF16,
        acc_dtype=DType.F32,
        a_regs=16,  # 2 stacked 16×16 = 2 vec<8>
        b_regs=16,  # unchanged: 16×32 split into 2 N-tiles
        c_regs=32,  # 2 M-stacked × 2 N-tiled × 8 = 4 vec<8>
    ),
    # Same per-fragment offsets as the m=16 entry — the multi-fragment
    # composition is handled by the nax visitor's _nax_frag_layout, not here.
    a_offsets=tuple((r * 8, c) for r in range(2) for c in range(4)),
    b_offsets=tuple((r * 8, c) for r in range(2) for c in range(4)),
    cd_offsets=tuple((r * 8, c) for r in range(2) for c in range(4)),
    lane_col_step=4,
    min_cuda_cc=None,
    cuda=None,
    min_metal_gen=ChipGeneration.METAL_M5,
    metal="nax:2:2:1",  # 2 M-frags × 2 N-frags × 1 K-frag
)


# ---------------------------------------------------------------------------
# Intel SPIR-V cooperative-matrix shapes (Battlemage / Xe2+).
#
# Captured from ``scripts/spirv/probe_coopmat`` on Intel Panther Lake
# (Xe3 iGPU, vendor=0x8086 device=0xb080, Mesa 26.0.3). All entries are
# ``subgroup`` scope — Battlemage doesn't expose workgroup-scope coopmat.
#
# Per-fragment shape is fixed at ``m=8, n=16, k=16`` for fp16/bf16 (Intel's
# matrix engine native tile). Compare to PTX m16n8k16 (transposed M/N
# convention) and Apple NAX m16n32k16 (different per-fragment dims). Tile
# multiples on Intel: ``BM % 8 == 0``, ``BN % 16 == 0``, ``BK % 16 == 0``.
#
# Two acc-dtype variants are registered for both fp16 and bf16:
#   * ``_F32`` — accumulate in f32 (the standard "compute_dtype=bf16,
#     acc=f32" path; what the autotuned configs assume).
#   * ``_BF16`` / ``_F16`` — accumulate in the input dtype (lossy but
#     exposed by Intel for memory-bound kernels that don't need full f32
#     accumulation; left registered so kernel authors can opt in via
#     ``compute_dtype=bf16, acc=bf16`` if a future kernel benefits).
#
# Per-register lane offsets and reg counts are placeholders (matching
# the PTX m16n8 layout) — the SPIR-V lowerer doesn't consume the
# per-register coordinate map (KHR cooperative_matrix's lane↔element
# mapping is driver-private; element access goes through
# ``OpCooperativeMatrixLengthKHR``). Kept non-empty so the MmaConfig
# dataclass invariants pass.
# ---------------------------------------------------------------------------

# Placeholder per-register layout for Intel SPIR-V coopmat shapes. The
# SPIR-V lowerer never consults these — the KHR cooperative_matrix
# mapping is driver-private; per-element access uses
# ``OpCooperativeMatrixLengthKHR`` indexing. The IR validator
# (``_validate_reg_offsets``) requires ``len(offsets) == fragment
# width`` though, so the placeholders are sized to match the
# corresponding ``MmaShape.{a,b,c}_regs`` field for each shape.
def _intel_offsets(n: int) -> tuple[tuple[int, int], ...]:
    return ((0, 0),) * n

_BF16_M8N16K16_INTEL_F32 = MmaConfig(
    shape=MmaShape(
        name="m8n16k16_intel_bf16_f32",
        m=8,
        n=16,
        k=16,
        a_dtype=DType.BF16,
        b_dtype=DType.BF16,
        acc_dtype=DType.F32,
        # Reg counts are the spec-canonical SG=16 per-lane widths for
        # the ``cl_intel_subgroup_matrix_multiply_accumulate`` extension
        # (the only supported subgroup sizes are 8 and 16 per Khronos;
        # SG=16 is Battlemage's native DPAS width). The OCL lowerer
        # reads them to size per-lane fragment vectors.
        a_regs=8,  # SG=16: lane L holds column k=L; 8 rows of A per lane
        b_regs=8,  # SG=16: lane L holds column n=L; 16 K-rows in 8 packed-pair i32
        c_regs=8,  # SG=16: lane L holds column n=L; 8 rows of C per lane
    ),
    a_offsets=_intel_offsets(8),
    b_offsets=_intel_offsets(8),
    cd_offsets=_intel_offsets(8),
    lane_col_step=2,  # bf16 in b32 carrier
    min_cuda_cc=None,
    cuda=None,
    min_metal_gen=None,
    metal=None,
    min_intel_gpu_gen=ChipGeneration.INTEL_XE2,
    intel_gpu="khr:subgroup",
)

_BF16_M8N16K16_INTEL_BF16 = MmaConfig(
    shape=MmaShape(
        name="m8n16k16_intel_bf16_bf16",
        m=8,
        n=16,
        k=16,
        a_dtype=DType.BF16,
        b_dtype=DType.BF16,
        acc_dtype=DType.BF16,
        a_regs=8,
        b_regs=8,
        c_regs=8,  # SG=16: 8 rows of C per lane (bf16 acc)
    ),
    a_offsets=_intel_offsets(8),
    b_offsets=_intel_offsets(8),
    cd_offsets=_intel_offsets(8),
    lane_col_step=2,
    min_cuda_cc=None,
    cuda=None,
    min_metal_gen=None,
    metal=None,
    min_intel_gpu_gen=ChipGeneration.INTEL_XE2,
    intel_gpu="khr:subgroup",
)

_F16_M8N16K16_INTEL_F32 = MmaConfig(
    shape=MmaShape(
        name="m8n16k16_intel_f16_f32",
        m=8,
        n=16,
        k=16,
        a_dtype=DType.F16,
        b_dtype=DType.F16,
        acc_dtype=DType.F32,
        a_regs=8,
        b_regs=8,
        c_regs=8,
    ),
    a_offsets=_intel_offsets(8),
    b_offsets=_intel_offsets(8),
    cd_offsets=_intel_offsets(8),
    lane_col_step=2,
    min_cuda_cc=None,
    cuda=None,
    min_metal_gen=None,
    metal=None,
    min_intel_gpu_gen=ChipGeneration.INTEL_XE2,
    intel_gpu="khr:subgroup",
)

_F16_M8N16K16_INTEL_F16 = MmaConfig(
    shape=MmaShape(
        name="m8n16k16_intel_f16_f16",
        m=8,
        n=16,
        k=16,
        a_dtype=DType.F16,
        b_dtype=DType.F16,
        acc_dtype=DType.F16,
        a_regs=8,
        b_regs=8,
        c_regs=8,
    ),
    a_offsets=_intel_offsets(8),
    b_offsets=_intel_offsets(8),
    cd_offsets=_intel_offsets(8),
    lane_col_step=2,
    min_cuda_cc=None,
    cuda=None,
    min_metal_gen=None,
    metal=None,
    min_intel_gpu_gen=ChipGeneration.INTEL_XE2,
    intel_gpu="khr:subgroup",
)

# Intel Xe2 int8 cooperative_matrix — m=8, n=16, k=32 (doubled inner
# dim vs bf16/f16). Each MMA does 8192 ops/cycle vs 4096 for bf16,
# unlocking 2× theoretical throughput on attention/GEMM.
_S8_M8N16K32_INTEL_S32 = MmaConfig(
    shape=MmaShape(
        name="m8n16k32_intel_s8_s32",
        m=8,
        n=16,
        k=32,
        a_dtype=DType.S8,
        b_dtype=DType.S8,
        acc_dtype=DType.S32,
        # Per-lane storage at SIMD32:
        #   a: m=8 × k=32 / 32 lanes = 8 s8 per lane = 2 b32 carriers
        #   b: n=16 × k=32 / 32 lanes = 16 s8 per lane = 4 b32 carriers
        #   c: m=8 × n=16 / 32 lanes = 4 s32 per lane = 4 b32 carriers
        a_regs=8,
        b_regs=16,
        c_regs=4,
    ),
    a_offsets=_intel_offsets(8),
    b_offsets=_intel_offsets(16),
    cd_offsets=_intel_offsets(4),
    lane_col_step=4,  # s8 in b32 carrier = 4 elements per b32 reg
    min_cuda_cc=None,
    cuda=None,
    min_metal_gen=None,
    metal=None,
    min_intel_gpu_gen=ChipGeneration.INTEL_XE2,
    intel_gpu="khr:subgroup",
)


ALL_SHAPES: tuple[MmaConfig, ...] = (
    _BF16_M8N8K8,  # Metal-native 8x8x8
    _BF16_M16N8K8,  # PTX m16n8k8 (Ampere+)
    _BF16_M16N32K16_NAX,  # Metal NAX 16x32x16 (M5+)
    _BF16_M32N32K16_NAX,  # Metal NAX 32x32x16 (M5+, larger M-fragment)
    _BF16_K16,
    _F16_K16,
    _E4M3_K16,
    _E5M2_K16,
    _E4M3_K32,
    _E5M2_K32,
    _BF16xE4M3_K16,
    # Intel SPIR-V — KHR cooperative_matrix (subgroup scope) on Xe2+.
    _BF16_M8N16K16_INTEL_F32,
    _BF16_M8N16K16_INTEL_BF16,
    _F16_M8N16K16_INTEL_F32,
    _F16_M8N16K16_INTEL_F16,
    _S8_M8N16K32_INTEL_S32,
)

# Reverse index for O(1) shape_id → config lookups. Built once at
# module import; never mutated (tests extend the override table
# below, not this map).
_BY_SHAPE_ID: dict[str, MmaConfig] = {cfg.shape_id: cfg for cfg in ALL_SHAPES}


# ---------------------------------------------------------------------------
# Per-backend lowering payload lookup
#
# Production payloads are declared on each ``MmaConfig`` as backend-
# named fields (``cuda``, ``metal``, ...). Field names are keyed on
# ``DeviceFamily.<X>.value`` so ``payload_for`` can resolve them
# generically via ``getattr(cfg, family.value)``. ``_SUPPORTED_FAMILIES``
# names exactly the families that have a field on ``MmaConfig``; a
# module-import assertion below verifies this up-front so a typo
# like ``min_cuda_cc=(8, 0)`` paired with ``cuba="..."`` cannot ship.
#
# ``_PAYLOAD_OVERRIDES`` is a thin test-time side table. It takes
# precedence over the ``MmaConfig`` field so tests can stand up an
# ad-hoc shape (``test_no_msl``, ``custom_s8_shape``) or exercise a
# non-shipped variant of an existing shape, without mutating frozen
# dataclass instances.
# ---------------------------------------------------------------------------

_SUPPORTED_FAMILIES: tuple[DeviceFamily, ...] = (
    DeviceFamily.CUDA,
    DeviceFamily.METAL,
    DeviceFamily.INTEL_GPU,
)

_MMA_CONFIG_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in dataclasses.fields(MmaConfig))
for _fam in _SUPPORTED_FAMILIES:
    assert _fam.value in _MMA_CONFIG_FIELD_NAMES, (
        f"MmaConfig has no payload field for DeviceFamily.{_fam.name} — "
        f"expected a field named {_fam.value!r} "
        f"(must equal ``DeviceFamily.{_fam.name}.value``)"
    )


_PAYLOAD_OVERRIDES: dict[tuple[str, DeviceFamily], str] = {}


def payload_for(shape_id: str, family: DeviceFamily) -> Optional[str]:
    """Backend lowering payload for ``(shape_id, family)``, or ``None``
    if the shape has no path on that backend.

    Precedence: ``_PAYLOAD_OVERRIDES`` (test-time) > ``MmaConfig``
    field (production). Backends call this to decide whether they
    can lower a kernel that declares ``shape_id``; ``None`` feeds
    ``is_valid_for(caps)`` rejection at the family's dispatch layer.
    """
    override = _PAYLOAD_OVERRIDES.get((shape_id, family))
    if override is not None:
        return override
    cfg = _BY_SHAPE_ID.get(shape_id)
    if cfg is None:
        return None
    return getattr(cfg, family.value, None)


def register_backend_payload(shape_id: str, family: DeviceFamily, payload: str) -> None:
    """Register or override the lowering payload for ``(shape_id, family)``.

    Used by tests that either construct ad-hoc ``MmaShape`` instances
    (shape not in ``ALL_SHAPES``) or want to exercise a non-default
    payload for an existing shape. Production code does NOT call this —
    production payloads are declared on the shipped ``MmaConfig`` rows.
    Idempotent; replaces any prior override silently.
    """
    _PAYLOAD_OVERRIDES[(shape_id, family)] = payload


def _dtype_key(d: DType) -> str:
    """Normalize a DType to its short string form used by kernel specs.
    ``DType.F16.value`` is ``"f16"`` — most kernels still also accept
    ``"fp16"`` for historical reasons, hence the alias map below."""
    return d.value


# Alias map: kernel spec strings that map to the same DType.
_DTYPE_ALIASES: dict[str, str] = {
    "fp16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
}


def _normalize_dtype_str(s: str) -> str:
    return _DTYPE_ALIASES.get(s.lower(), s.lower())


# Back-compat: the legacy ``_MMA_TABLE`` mapping (a_dtype, b_dtype, k) → MmaConfig.
# Derived from ALL_SHAPES so there's only one source. Deliberately does
# NOT populate entries for shapes that share the (a, b, k) key — the
# legacy API can't disambiguate m16n8k8 from m8n8k8 on just the k, so
# those are reachable only via the new ``ALL_SHAPES`` / ``shapes_for_chip``
# path (M3 migrates kernels to it). Kernels still calling
# ``lookup_mma(compute, compute, k)`` with k=8 will KeyError — that's
# intentional: today's kernels only request k ∈ {16, 32}.
_MMA_TABLE: dict[tuple[str, str, int], MmaConfig] = {}
_TABLE_COLLISIONS: set[tuple[str, str, int]] = set()
# NAX shapes (M5+) are only reachable via shapes_for_chip + main_shape;
# exclude from the legacy (a, b, k) table to avoid collisions with the
# same-dtype same-k simdgroup_matrix shapes that the fallback path uses.
# Intel SPIR-V shapes are similarly only reachable via shapes_for_chip
# (no PTX path, no MSL path) — exclude from the legacy table by
# checking ``min_intel_gpu_gen`` is set.
_LEGACY_EXCLUDE_GENS = {ChipGeneration.METAL_M5}
for _cfg in ALL_SHAPES:
    if _cfg.min_metal_gen in _LEGACY_EXCLUDE_GENS and _cfg.min_cuda_cc is None:
        continue
    if (
        _cfg.min_intel_gpu_gen is not None
        and _cfg.min_cuda_cc is None
        and _cfg.min_metal_gen is None
    ):
        # Intel-only shape — never reach the legacy (a, b, k) table.
        continue
    _a = _dtype_key(_cfg.shape.a_dtype)
    _b = _dtype_key(_cfg.shape.b_dtype)
    _k = _cfg.shape.k
    _key = (_a, _b, _k)
    if _key in _MMA_TABLE:
        # (a, b, k) collision → ambiguous; remove from legacy lookup.
        _TABLE_COLLISIONS.add(_key)
    else:
        _MMA_TABLE[_key] = _cfg
for _key in _TABLE_COLLISIONS:
    _MMA_TABLE.pop(_key, None)
# fp16 legacy alias so old kernels passing "fp16" still resolve.
for (_a, _b, _k), _cfg in list(_MMA_TABLE.items()):
    if _a == "f16" and _b == "f16":
        _MMA_TABLE[("fp16", "fp16", _k)] = _cfg


def lookup_mma(a_dtype: str, b_dtype: str, mma_k: int = 16) -> MmaConfig:
    """Look up the MmaConfig for a (a_dtype, b_dtype, mma_k) triple.

    Raises KeyError with a helpful message listing available combos.
    Back-compat with the pre-registry API — kernels that import this
    name keep working while M2–M4 migrate them to the chip-driven
    autotune path.
    """
    key = (_normalize_dtype_str(a_dtype), _normalize_dtype_str(b_dtype), mma_k)
    if key not in _MMA_TABLE:
        raise KeyError(
            f"No MMA shape registered for A={a_dtype} × B={b_dtype} × k={mma_k}. "
            f"Available: {sorted(_MMA_TABLE.keys())}"
        )
    return _MMA_TABLE[key]


# ---------------------------------------------------------------------------
# Chip → supported shape set
# ---------------------------------------------------------------------------

# Declaration order of METAL_* matters for ``min_metal_gen`` ordering.
_METAL_GEN_ORDER: tuple[ChipGeneration, ...] = (
    ChipGeneration.METAL_M1,
    ChipGeneration.METAL_M3,
    ChipGeneration.METAL_M5,
)


def _metal_gen_index(gen: ChipGeneration) -> int:
    return _METAL_GEN_ORDER.index(gen)


# Intel Xe gens, ordered ascending. Used for ``min_intel_gpu_gen`` filter.
_INTEL_GPU_GEN_ORDER: tuple[ChipGeneration, ...] = (
    ChipGeneration.INTEL_XE_LPG,  # Meteor Lake
    ChipGeneration.INTEL_XE2,  # Lunar Lake / Battlemage discrete
    ChipGeneration.INTEL_XE3,  # Panther Lake / Battlemage+
)


def _intel_gpu_gen_index(gen: ChipGeneration) -> int:
    return _INTEL_GPU_GEN_ORDER.index(gen)


def _supports(cfg: MmaConfig, gen: ChipGeneration) -> bool:
    if gen.is_cuda:
        if cfg.min_cuda_cc is None:
            return False
        cc = gen.cuda_cc()
        assert cc is not None  # invariant for is_cuda=True
        return cc >= cfg.min_cuda_cc
    if gen.is_metal:
        if cfg.min_metal_gen is None:
            return False
        try:
            return _metal_gen_index(gen) >= _metal_gen_index(cfg.min_metal_gen)
        except ValueError:
            return False
    if gen.is_intel_gpu:
        if cfg.min_intel_gpu_gen is None:
            return False
        try:
            return _intel_gpu_gen_index(gen) >= _intel_gpu_gen_index(cfg.min_intel_gpu_gen)
        except ValueError:
            return False
    return False  # UNKNOWN or other families — no shapes


def shapes_for_chip(gen: ChipGeneration) -> frozenset[str]:
    """Return the set of MMA shape names (``MmaConfig.shape_id``) the
    given chip generation supports. Populates
    ``DeviceCaps.matmul_shapes`` at probe time.

    Empty frozenset for ``ChipGeneration.UNKNOWN`` — kernels that need
    MMA will fail their ``is_valid_for`` check cleanly rather than
    silently emitting ops the backend can't lower.
    """
    return frozenset(cfg.shape_id for cfg in ALL_SHAPES if _supports(cfg, gen))


__all__ = (
    "ALL_SHAPES",
    "MmaConfig",
    "lookup_mma",
    "payload_for",
    "register_backend_payload",
    "shapes_for_chip",
)
