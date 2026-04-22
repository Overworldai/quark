"""MMA shape registry — the single source of truth.

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

from dataclasses import dataclass
from typing import Optional

from quark.device import ChipGeneration
from quark.ir.module import MmaShape
from quark.ir.types import DType


@dataclass(frozen=True)
class MmaConfig:
    """One MMA descriptor + its per-register lane metadata + chip gates.

    ``shape`` is the IR-level descriptor the lowerer consumes.
    ``a_offsets`` / ``b_offsets`` / ``cd_offsets`` are the per-register
    (row, col) offsets in the PTX fragment layout (see PTX ISA §9.7.14.5).
    ``lane_col_step`` is the elements-per-b32 count: 2 for bf16/fp16,
    4 for 8-bit.

    ``min_ptx_cc`` / ``min_metal_gen`` are the chip gates. ``None`` means
    "unrestricted on that backend". ``shapes_for_chip`` filters the
    descriptor list against the active device's ``ChipGeneration``:
    CUDA chips pass iff ``chip.cuda_cc() >= min_ptx_cc``; Metal chips
    pass iff ``chip >= min_metal_gen`` in the enum's declaration order.
    A descriptor with both fields ``None`` is rejected by any chip
    (mis-registered); a descriptor with ``min_ptx_cc=None`` passes on
    every Metal chip if its ``min_metal_gen`` passes, and vice versa.
    """

    shape: MmaShape
    a_offsets: tuple[tuple[int, int], ...]
    b_offsets: tuple[tuple[int, int], ...]
    cd_offsets: tuple[tuple[int, int], ...]
    lane_col_step: int  # elements per b32 reg — drives smem-base math
    # Per-backend minimum chip that supports this shape. None = shape
    # not available on that backend at all (e.g. fp8 MMAs on Metal).
    min_ptx_cc: Optional[tuple[int, int]] = None
    min_metal_gen: Optional[ChipGeneration] = None

    @property
    def shape_id(self) -> str:
        return self.shape.name

    @property
    def mma_k(self) -> int:
        return self.shape.k


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
        ptx=None,
        msl="bfloat16_t:1:1:1",
    ),
    a_offsets=((0, 0),),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1)),  # Apple 8x8 acc: 2 elements per lane
    lane_col_step=2,
    min_ptx_cc=None,  # no PTX path
    min_metal_gen=ChipGeneration.METAL_M3,
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
        ptx="m16n8k8.row.col.f32.bf16.bf16.f32",
    ),
    a_offsets=((0, 0), (8, 0)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,
    min_ptx_cc=(8, 0),  # Ampere+
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
        ptx="m16n8k16.row.col.f32.bf16.bf16.f32",
        msl="bfloat16_t:2:1:2",
    ),
    a_offsets=((0, 0), (8, 0), (0, 8), (8, 8)),
    b_offsets=((0, 0), (0, 8)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,
    min_ptx_cc=(8, 0),  # Ampere+
    # No Metal — m8n8k8 dominates on every M3 autotune problem, so
    # keeping m16n8k16 in Metal's shape set just wastes autotune time.
    # The MSL lowerer still supports the shape if a future kernel opts
    # in explicitly via ``main_shape``.
    min_metal_gen=None,
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
        ptx="m16n8k16.row.col.f32.f16.f16.f32",
        msl="half:2:1:2",
    ),
    a_offsets=((0, 0), (8, 0), (0, 8), (8, 8)),
    b_offsets=((0, 0), (0, 8)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,
    min_ptx_cc=(7, 5),  # Turing+ (m16n8k16 f16 MMA)
    # No Metal — see m16n8k16_bf16 note. m8n8k8_f16 would be the
    # Metal-native choice; register when a kernel actually wants fp16.
    min_metal_gen=None,
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
        ptx="m16n8k16.row.col.f32.e4m3.e4m3.f32",
    ),
    a_offsets=((0, 0), (8, 0)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    # m16n8k16 fp8 requires PTX ISA 8.7+ (Blackwell sm_120). Ada
    # (sm_89) only supports fp8 via m16n8k32 — see _E4M3_K32.
    min_ptx_cc=(12, 0),
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
        ptx="m16n8k16.row.col.f32.e5m2.e5m2.f32",
    ),
    a_offsets=((0, 0), (8, 0)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    min_ptx_cc=(12, 0),  # Blackwell — Ada has fp8 MMAs only at k=32
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
        ptx="m16n8k32.row.col.f32.e4m3.e4m3.f32",
    ),
    a_offsets=((0, 0), (8, 0), (0, 16), (8, 16)),
    b_offsets=((0, 0), (0, 16)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    min_ptx_cc=(8, 9),
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
        ptx="m16n8k32.row.col.f32.e5m2.e5m2.f32",
    ),
    a_offsets=((0, 0), (8, 0), (0, 16), (8, 16)),
    b_offsets=((0, 0), (0, 16)),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=4,
    min_ptx_cc=(8, 9),
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
        ptx="m16n8k16.row.col.f32.bf16.e4m3.f32",
    ),
    a_offsets=((0, 0), (8, 0), (0, 8), (8, 8)),
    b_offsets=((0, 0),),
    cd_offsets=((0, 0), (0, 1), (8, 0), (8, 1)),
    lane_col_step=2,  # bf16 A dictates the lane stride (wider)
    # Mixed bf16×e4m3 at m16n8k16 uses the kind::f8f6f4 encoding
    # added in PTX ISA 8.7 (sm_120+). Ada/Hopper have no native mixed
    # form here — a precast-to-fp8 path would use _E4M3_K32 on those.
    min_ptx_cc=(12, 0),
)


# ---------------------------------------------------------------------------
# Descriptor list + lookup table (derived)
# ---------------------------------------------------------------------------

# Every descriptor known to Quark. Adding a new MMA means adding one
# row here; ``shapes_for_chip`` / ``lookup_mma`` / ``_MMA_TABLE`` all
# read from this list.
ALL_SHAPES: tuple[MmaConfig, ...] = (
    _BF16_M8N8K8,  # Metal-native 8x8x8
    _BF16_M16N8K8,  # PTX m16n8k8 (Ampere+)
    _BF16_K16,
    _F16_K16,
    _E4M3_K16,
    _E5M2_K16,
    _E4M3_K32,
    _E5M2_K32,
    _BF16xE4M3_K16,
)


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
for _cfg in ALL_SHAPES:
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


def _supports(cfg: MmaConfig, gen: ChipGeneration) -> bool:
    if gen.is_cuda:
        if cfg.min_ptx_cc is None:
            return False
        cc = gen.cuda_cc()
        assert cc is not None  # invariant for is_cuda=True
        return cc >= cfg.min_ptx_cc
    if gen.is_metal:
        if cfg.min_metal_gen is None:
            return False
        try:
            return _metal_gen_index(gen) >= _metal_gen_index(cfg.min_metal_gen)
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
    "shapes_for_chip",
)
