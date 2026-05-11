"""Generate / verify golden lowerer outputs for PTX & MSL.

EXEMPT FROM 500-LINE RULE — this file is a flat enumeration of lowerer
test cases. Adding a case means adding a short function; splitting the
registry across files would hurt discoverability.

Usage
-----
Run once on each device to populate goldens, then commit:

    # On a Mac (MSL) — populates tests/lower/golden/msl/
    python tools/generate_test_cases.py --update

    # On a CUDA box (PTX) — populates tests/lower/golden/ptx/
    python tools/generate_test_cases.py --update

Other modes::

    --check   (default) diff emitted output against stored goldens, exit 1 on mismatch
    --list    print every registered case id and its supported backends
    --filter  glob to subset cases (e.g. `--filter "arith/*"`)
    --backend ptx|msl|auto   override platform detection (auto: darwin→msl else ptx)

The pytest suite in tests/lower/golden/test_goldens.py iterates the
same registry and verifies goldens stay pinned as the lowerers evolve.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from quark.device import DeviceFamily
from quark.ir import BufferType, Builder, DType, GlobalTensor, MmaShape
from quark.ir.mma_registry import register_backend_payload

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKENDS_ALL = ("ptx", "msl")
REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = REPO_ROOT / "tests" / "lower" / "golden"


@dataclass
class Case:
    id: str
    description: str
    builder: Callable[[Builder], None]
    backends: tuple[str, ...] = BACKENDS_ALL


CASES: list[Case] = []
_IDS_SEEN: set[str] = set()


def case(cid: str, description: str, *, backends: Sequence[str] = BACKENDS_ALL):
    def decorator(fn: Callable[[Builder], None]):
        if cid in _IDS_SEEN:
            raise ValueError(f"duplicate case id: {cid}")
        _IDS_SEEN.add(cid)
        CASES.append(Case(cid, description, fn, tuple(backends)))
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _g1d(b: Builder, name: str = "X", dtype: DType = DType.F32, n: int = 128) -> GlobalTensor:
    b.param(name, BufferType(dtype))
    return GlobalTensor(
        dtype=dtype, shape=(n,), stride=(1,), name=name, param=b.function.params[-1]
    )


def _g2d(
    b: Builder,
    name: str = "X",
    dtype: DType = DType.F32,
    shape: tuple[int, int] = (64, 64),
) -> GlobalTensor:
    b.param(name, BufferType(dtype))
    return GlobalTensor(
        dtype=dtype,
        shape=shape,
        stride=(shape[1], 1),
        name=name,
        param=b.function.params[-1],
    )


# MMA shapes — registered lazily per case. PTX names match shapes_for_chip.
_SHAPE_M16N8K16_BF16 = MmaShape(
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
)
register_backend_payload("m16n8k16_bf16", DeviceFamily.CUDA, "m16n8k16.row.col.f32.bf16.bf16.f32")

_SHAPE_M16N8K16_F16 = MmaShape(
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
)
register_backend_payload("m16n8k16_f16", DeviceFamily.CUDA, "m16n8k16.row.col.f32.f16.f16.f32")

_SHAPE_M16N8K8_BF16 = MmaShape(
    name="m16n8k8_bf16",
    m=16,
    n=8,
    k=8,
    a_dtype=DType.BF16,
    b_dtype=DType.BF16,
    acc_dtype=DType.F32,
    a_regs=2,
    b_regs=1,
    c_regs=4,
)
register_backend_payload("m16n8k8_bf16", DeviceFamily.CUDA, "m16n8k8.row.col.f32.bf16.bf16.f32")

_SHAPE_M16N8K16_E4M3 = MmaShape(
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
)
register_backend_payload("m16n8k16_e4m3", DeviceFamily.CUDA, "m16n8k16.row.col.f32.e4m3.e4m3.f32")

_SHAPE_M16N8K32_E4M3 = MmaShape(
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
)
register_backend_payload("m16n8k32_e4m3", DeviceFamily.CUDA, "m16n8k32.row.col.f32.e4m3.e4m3.f32")

_SHAPE_M16N8K16_E5M2 = MmaShape(
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
)
register_backend_payload("m16n8k16_e5m2", DeviceFamily.CUDA, "m16n8k16.row.col.f32.e5m2.e5m2.f32")


_A_OFFS_K16 = ((0, 0), (8, 0), (0, 8), (8, 8))
_B_OFFS_K16 = ((0, 0), (0, 8))
_CD_OFFS = ((0, 0), (0, 1), (8, 0), (8, 1))

_A_OFFS_K8 = ((0, 0), (8, 0))
_B_OFFS_K8 = ((0, 0),)

_A_OFFS_FP8_K16 = ((0, 0), (8, 0))
_B_OFFS_FP8_K16 = ((0, 0),)
_A_OFFS_FP8_K32 = ((0, 0), (8, 0), (0, 16), (8, 16))
_B_OFFS_FP8_K32 = ((0, 0), (0, 16))


# ---------------------------------------------------------------------------
# §1 arith / const
# ---------------------------------------------------------------------------


@case("arith/const_f32", "mov.f32 with hex-encoded constant")
def _(b):
    b.const(DType.F32, 1.0)


@case("arith/const_f64", "mov.f64 with hex-encoded constant", backends=("ptx",))
def _(b):
    b.const(DType.F64, 1.0)


@case("arith/const_u32", "decimal u32 const")
def _(b):
    b.const(DType.U32, 42)


@case("arith/const_s32", "negative signed const")
def _(b):
    b.const(DType.S32, -7)


@case("arith/const_bf16", "bf16 const")
def _(b):
    b.const(DType.BF16, 1.0)


@case("arith/const_f16", "f16 const")
def _(b):
    b.const(DType.F16, 1.0)


@case("arith/const_pred_true", "pred const true")
def _(b):
    b.const(DType.PRED, True)


@case("arith/const_pred_false", "pred const false")
def _(b):
    b.const(DType.PRED, False)


def _bin_f32(method):
    def _body(b):
        x = b.const(DType.F32, 1.0)
        y = b.const(DType.F32, 2.0)
        getattr(b, method)(x, y)

    return _body


for _m in ("add", "sub", "mul", "div", "min", "max"):
    case(f"arith/{_m}_f32", f"f32 {_m}")(_bin_f32(_m))


def _bin_u32(method):
    def _body(b):
        x = b.const(DType.U32, 3)
        y = b.const(DType.U32, 4)
        getattr(b, method)(x, y)

    return _body


for _m in ("add", "sub", "mul", "min", "max", "and_", "or_", "xor", "shl", "shr"):
    case(f"arith/{_m.rstrip('_')}_u32", f"u32 {_m.rstrip('_')}")(_bin_u32(_m))


@case("arith/fma_f32", "fused multiply-add")
def _(b):
    x = b.const(DType.F32, 1.0)
    y = b.const(DType.F32, 2.0)
    z = b.const(DType.F32, 3.0)
    b.fma(x, y, z)


@case("arith/cmp_lt_f32", "setp.lt.f32")
def _(b):
    x = b.const(DType.F32, 1.0)
    y = b.const(DType.F32, 2.0)
    b.cmp("lt", x, y)


@case("arith/cmp_eq_u32", "setp.eq.u32")
def _(b):
    x = b.const(DType.U32, 1)
    y = b.const(DType.U32, 2)
    b.cmp("eq", x, y)


@case("arith/cmp_ge_f32", "setp.ge.f32")
def _(b):
    x = b.const(DType.F32, 1.0)
    y = b.const(DType.F32, 2.0)
    b.cmp("ge", x, y)


@case("arith/select_f32", "selp.f32 on f32 values")
def _(b):
    x = b.const(DType.F32, 1.0)
    y = b.const(DType.F32, 2.0)
    p = b.cmp("lt", x, y)
    b.select(p, x, y)


@case("arith/and_pred", "and.pred combining two predicates")
def _(b):
    x = b.const(DType.F32, 1.0)
    y = b.const(DType.F32, 2.0)
    p1 = b.cmp("lt", x, y)
    p2 = b.cmp("eq", x, y)
    b.and_(p1, p2)


@case("arith/or_pred", "or.pred combining two predicates")
def _(b):
    x = b.const(DType.F32, 1.0)
    y = b.const(DType.F32, 2.0)
    p1 = b.cmp("lt", x, y)
    p2 = b.cmp("eq", x, y)
    b.or_(p1, p2)


# ---------------------------------------------------------------------------
# §2 math intrinsics
# ---------------------------------------------------------------------------


@case("math/ex2_approx_f32", "ex2.approx.f32")
def _(b):
    b.ex2_approx(b.const(DType.F32, 1.0))


@case("math/rcp_approx_f32", "rcp.approx.f32")
def _(b):
    b.rcp_approx(b.const(DType.F32, 2.0))


@case("math/rsqrt_approx_f32", "rsqrt.approx.f32")
def _(b):
    b.rsqrt_approx(b.const(DType.F32, 4.0))


@case("math/sqrt_f32", "sqrt.f32")
def _(b):
    b.sqrt(b.const(DType.F32, 4.0))


@case("math/exp2_f32", "exp2.f32")
def _(b):
    b.exp2(b.const(DType.F32, 1.0))


@case("math/log2_f32", "log2.f32")
def _(b):
    b.log2(b.const(DType.F32, 2.0))


@case("math/tanh_f32", "tanh.f32")
def _(b):
    b.tanh(b.const(DType.F32, 0.5))


# ---------------------------------------------------------------------------
# §3 convert / bitcast
# ---------------------------------------------------------------------------


@case("convert/f32_to_bf16", "cvt.rn.bf16.f32")
def _(b):
    b.convert(b.const(DType.F32, 1.0), DType.BF16)


@case("convert/bf16_to_f32", "cvt.f32.bf16 (widening, no rounding)")
def _(b):
    b.convert(b.const(DType.BF16, 0), DType.F32)


@case("convert/f32_to_f16", "cvt.rn.f16.f32")
def _(b):
    b.convert(b.const(DType.F32, 1.0), DType.F16)


@case("convert/f16_to_f32", "cvt.f32.f16 widening")
def _(b):
    b.convert(b.const(DType.F16, 1.0), DType.F32)


@case("convert/f32_to_s32", "cvt.rzi.s32.f32")
def _(b):
    b.convert(b.const(DType.F32, 1.5), DType.S32)


@case("convert/s32_to_f32", "cvt.rn.f32.s32")
def _(b):
    b.convert(b.const(DType.S32, 3), DType.F32)


@case("convert/u32_to_f32", "cvt.rn.f32.u32")
def _(b):
    b.convert(b.const(DType.U32, 3), DType.F32)


@case("convert/bitcast_f32_to_b32", "reinterpret f32 bits as b32")
def _(b):
    b.bitcast(b.const(DType.F32, 1.0), DType.B32)


@case(
    "convert/unpacked_convert_e4m3_to_bf16",
    "packed f16x2.e4m3x2 chain for fp8→bf16",
    backends=("ptx",),
)
def _(b):
    packed = b.const(DType.B16, 0)
    b.unpacked_convert(packed, src_dtype=DType.E4M3, dst_dtype=DType.BF16)


@case(
    "convert/unpacked_convert_e5m2_to_bf16",
    "packed f16x2.e5m2x2 chain for fp8→bf16",
    backends=("ptx",),
)
def _(b):
    packed = b.const(DType.B16, 0)
    b.unpacked_convert(packed, src_dtype=DType.E5M2, dst_dtype=DType.BF16)


@case(
    "convert/unpacked_convert_e4m3_to_f16",
    "packed f16x2.e4m3x2 for fp8→f16",
    backends=("ptx",),
)
def _(b):
    packed = b.const(DType.B16, 0)
    b.unpacked_convert(packed, src_dtype=DType.E4M3, dst_dtype=DType.F16)


# ---------------------------------------------------------------------------
# §4 memory
# ---------------------------------------------------------------------------


@case("memory/smem_alloc_1d", "single-row smem alloc, verify pool declaration")
def _(b):
    b.smem_alloc("A", DType.F32, (1, 64))


@case("memory/smem_alloc_2d", "2d tile alloc")
def _(b):
    b.smem_alloc("A", DType.F32, (8, 16))


@case("memory/smem_alloc_multiple", "multiple allocs share the pool")
def _(b):
    b.smem_alloc("A", DType.F32, (4, 4))
    b.smem_alloc("B", DType.F32, (4, 4))


@case("memory/smem_alloc_bf16", "bf16 smem alloc")
def _(b):
    b.smem_alloc("A", DType.BF16, (16, 16))


@case("memory/smem_scalar_load_static", "static-offset smem load folds to constant offset")
def _(b):
    A = b.smem_alloc("A", DType.F32, (8, 16))
    row = b.const(DType.U32, 2)
    col = b.const(DType.U32, 3)
    b.load(A, row, col)


@case("memory/smem_scalar_store_static", "static-offset smem store")
def _(b):
    A = b.smem_alloc("A", DType.F32, (4, 4))
    row = b.const(DType.U32, 1)
    col = b.const(DType.U32, 2)
    v = b.const(DType.F32, 3.0)
    b.store(A, v, row, col)


@case("memory/smem_scalar_load_dynamic", "dynamic row index widens to u64 offset math")
def _(b):
    A = b.smem_alloc("A", DType.F32, (16, 16))
    row = b.thread_idx("x")
    col = b.const(DType.U32, 0)
    b.load(A, row, col)


@case("memory/gmem_scalar_load", "basic global load")
def _(b):
    g = _g1d(b)
    idx = b.const(DType.U32, 0)
    b.load(g, idx)


@case("memory/gmem_scalar_store", "basic global store")
def _(b):
    g = _g1d(b)
    idx = b.const(DType.U32, 0)
    v = b.const(DType.F32, 1.0)
    b.store(g, v, idx)


@case("memory/gmem_load_dynamic_idx", "dynamic gmem load widens to u64")
def _(b):
    g = _g2d(b, dtype=DType.F32, shape=(64, 64))
    row = b.thread_idx("x")
    col = b.const(DType.U32, 0)
    b.load(g, row, col)


@case("memory/gmem_load_multi_param", "two buffer params, load from each")
def _(b):
    A = _g1d(b, name="A")
    B = _g1d(b, name="B")
    idx = b.const(DType.U32, 0)
    a = b.load(A, idx)
    c = b.load(B, idx)
    b.add(a, c)


# ---------------------------------------------------------------------------
# §5 vector memory
# ---------------------------------------------------------------------------


@case("vec_mem/vec2_load_f32", "v2 gmem load")
def _(b):
    g = _g2d(b, dtype=DType.F32, shape=(16, 32))
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.vec_load(g, row, col, width=2)


@case("vec_mem/vec4_load_f32", "v4 gmem load")
def _(b):
    g = _g2d(b, dtype=DType.F32, shape=(16, 32))
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.vec_load(g, row, col, width=4)


@case("vec_mem/vec4_load_bf16", "v4 bf16 gmem load (8 bytes)")
def _(b):
    g = _g2d(b, dtype=DType.BF16, shape=(16, 32))
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.vec_load(g, row, col, width=4)


@case("vec_mem/vec4_store_f32", "v4 gmem store (width inferred from vec.shape)")
def _(b):
    g = _g2d(b, dtype=DType.F32, shape=(16, 32))
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    v = b.vec_load(g, row, col, width=4)
    b.vec_store(g, v, row, col)


@case("vec_mem/vec4_load_dynamic_row", "v4 load with dynamic row")
def _(b):
    g = _g2d(b, dtype=DType.F32, shape=(64, 64))
    row = b.thread_idx("x")
    col = b.const(DType.U32, 0)
    b.vec_load(g, row, col, width=4)


# ---------------------------------------------------------------------------
# §6 predicated memory
# ---------------------------------------------------------------------------


@case("pred_mem/gmem_load", "predicated gmem scalar load")
def _(b):
    g = _g1d(b)
    idx = b.const(DType.U32, 0)
    a = b.const(DType.F32, 1.0)
    z = b.const(DType.F32, 0.0)
    p = b.cmp("lt", z, a)
    b.load(g, idx, pred=p)


@case("pred_mem/gmem_store", "predicated gmem scalar store")
def _(b):
    g = _g1d(b)
    idx = b.const(DType.U32, 0)
    a = b.const(DType.F32, 1.0)
    z = b.const(DType.F32, 0.0)
    p = b.cmp("lt", z, a)
    b.store(g, a, idx, pred=p)


@case("pred_mem/vec4_load", "predicated vec4 gmem load")
def _(b):
    g = _g2d(b, dtype=DType.F32, shape=(16, 32))
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    z = b.const(DType.F32, 0.0)
    one = b.const(DType.F32, 1.0)
    p = b.cmp("lt", z, one)
    b.vec_load(g, row, col, width=4, pred=p)


# ---------------------------------------------------------------------------
# §7 atomic rmw
# ---------------------------------------------------------------------------


@case("atomic/add_f32", "atom.global.add.f32", backends=("ptx",))
def _(b):
    g = _g1d(b)
    idx = b.const(DType.U32, 0)
    v = b.const(DType.F32, 1.0)
    b.atomic_rmw(g, "add", v, idx)


@case("atomic/add_u32", "atom.global.add.u32", backends=("ptx",))
def _(b):
    g = _g1d(b, dtype=DType.U32)
    idx = b.const(DType.U32, 0)
    v = b.const(DType.U32, 1)
    b.atomic_rmw(g, "add", v, idx)


@case("atomic/min_u32", "atom.global.min.u32", backends=("ptx",))
def _(b):
    g = _g1d(b, dtype=DType.U32)
    idx = b.const(DType.U32, 0)
    v = b.const(DType.U32, 5)
    b.atomic_rmw(g, "min", v, idx)


@case("atomic/max_u32", "atom.global.max.u32", backends=("ptx",))
def _(b):
    g = _g1d(b, dtype=DType.U32)
    idx = b.const(DType.U32, 0)
    v = b.const(DType.U32, 5)
    b.atomic_rmw(g, "max", v, idx)


# ---------------------------------------------------------------------------
# §8 control flow
# ---------------------------------------------------------------------------


@case("cf/empty_for_loop", "for loop with no body — label + backedge only")
def _(b):
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 4)
    step = b.const(DType.U32, 1)
    with b.for_loop(lo, hi, step, iv_name="k"):
        pass


@case("cf/for_loop_single_carry_f32", "f32 accumulator carried across iterations")
def _(b):
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 16)
    step = b.const(DType.U32, 1)
    acc0 = b.const(DType.F32, 0.0)
    with b.for_loop(lo, hi, step, iv_name="k", carried=[acc0]) as (_, (acc,)):
        one = b.const(DType.F32, 1.0)
        b.yield_(b.add(acc, one))


@case("cf/for_loop_multi_carry", "two f32 carried slots")
def _(b):
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 8)
    step = b.const(DType.U32, 1)
    a0 = b.const(DType.F32, 0.0)
    b0 = b.const(DType.F32, 1.0)
    with b.for_loop(lo, hi, step, iv_name="k", carried=[a0, b0]) as (_, (a, c)):
        b.yield_(b.add(a, c), c)


@case("cf/nested_for_loops", "outer + inner loop, two labels")
def _(b):
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 4)
    step = b.const(DType.U32, 1)
    with b.for_loop(lo, hi, step, iv_name="i"):
        with b.for_loop(lo, hi, step, iv_name="j"):
            pass


@case("cf/nested_3_deep_for_loops", "triple-nested loop")
def _(b):
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 2)
    step = b.const(DType.U32, 1)
    with b.for_loop(lo, hi, step, iv_name="i"):
        with b.for_loop(lo, hi, step, iv_name="j"):
            with b.for_loop(lo, hi, step, iv_name="k"):
                pass


@case("cf/simple_if_else", "if/else with empty arms")
def _(b):
    a = b.const(DType.F32, 1.0)
    z = b.const(DType.F32, 0.0)
    p = b.cmp("lt", z, a)
    with b.if_(p) as (_, _, arms):
        with arms.then_():
            b.const(DType.F32, 2.0)
        with arms.else_():
            b.const(DType.F32, 3.0)


@case("cf/if_with_carry", "if/else carrying one f32 slot")
def _(b):
    a = b.const(DType.F32, 1.0)
    z = b.const(DType.F32, 0.0)
    p = b.cmp("lt", z, a)
    with b.if_(p, carried=[a]) as (then_in, else_in, arms):
        with arms.then_():
            two = b.const(DType.F32, 2.0)
            b.yield_(b.add(then_in[0], two))
        with arms.else_():
            b.yield_(else_in[0])


@case("cf/if_in_loop", "if nested inside for")
def _(b):
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 4)
    step = b.const(DType.U32, 1)
    a = b.const(DType.F32, 1.0)
    z = b.const(DType.F32, 0.0)
    with b.for_loop(lo, hi, step, iv_name="k"):
        p = b.cmp("lt", z, a)
        with b.if_(p) as (_, _, arms):
            with arms.then_():
                b.const(DType.F32, 2.0)
            with arms.else_():
                b.const(DType.F32, 3.0)


@case("cf/loop_in_if", "for nested inside if")
def _(b):
    a = b.const(DType.F32, 1.0)
    z = b.const(DType.F32, 0.0)
    p = b.cmp("lt", z, a)
    lo = b.const(DType.U32, 0)
    hi = b.const(DType.U32, 4)
    step = b.const(DType.U32, 1)
    with b.if_(p) as (_, _, arms):
        with arms.then_():
            with b.for_loop(lo, hi, step, iv_name="k"):
                pass
        with arms.else_():
            pass


@case("cf/for_int_bounds_u32", "u32 bounds")
def _(b):
    lo = b.const(DType.U32, 4)
    hi = b.const(DType.U32, 32)
    step = b.const(DType.U32, 2)
    with b.for_loop(lo, hi, step, iv_name="k"):
        pass


# ---------------------------------------------------------------------------
# §9 thread identity + barriers
# ---------------------------------------------------------------------------


@case("thread/tid_x", "threadIdx.x read")
def _(b):
    b.thread_idx("x")


@case("thread/tid_y", "threadIdx.y read")
def _(b):
    b.thread_idx("y")


@case("thread/tid_z", "threadIdx.z read")
def _(b):
    b.thread_idx("z")


@case("thread/bid_x", "blockIdx.x read")
def _(b):
    b.block_idx("x")


@case("thread/ntid_x", "blockDim.x read")
def _(b):
    b.block_dim("x")


@case("thread/nctaid_x", "gridDim.x read")
def _(b):
    b.grid_dim("x")


@case("thread/lane_id", "laneid read")
def _(b):
    b.lane_id()


@case("thread/subgroup_id", "warpid / simdgroup id read")
def _(b):
    b.subgroup_id()


@case("thread/barrier_block", "block-level barrier (bar.sync / threadgroup_barrier)")
def _(b):
    b.barrier("block")


@case("thread/barrier_system", "system-level barrier", backends=("ptx",))
def _(b):
    b.barrier("system")


# ---------------------------------------------------------------------------
# §10 shuffles & subgroup ops
# ---------------------------------------------------------------------------


@case("shuffle/bfly", "butterfly shuffle (PTX spelling)", backends=("ptx",))
def _(b):
    x = b.const(DType.F32, 1.0)
    b.shuffle("bfly", x, 16)


@case("shuffle/xor", "xor shuffle (MSL-compatible)")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.shuffle("xor", x, 16)


@case("shuffle/up", "up shuffle")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.shuffle("up", x, 1)


@case("shuffle/down", "down shuffle")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.shuffle("down", x, 1)


@case("shuffle/idx", "indexed shuffle")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.shuffle("idx", x, 0)


@case("shuffle/subgroup_reduce_sum_f32", "subgroup-reduce sum f32")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.subgroup_reduce("sum", x)


@case("shuffle/subgroup_reduce_max_f32", "subgroup-reduce max f32")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.subgroup_reduce("max", x)


@case("shuffle/subgroup_reduce_min_f32", "subgroup-reduce min f32")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.subgroup_reduce("min", x)


@case("shuffle/subgroup_broadcast", "broadcast lane 0")
def _(b):
    x = b.const(DType.F32, 1.0)
    b.subgroup_broadcast(x, 0)


# ---------------------------------------------------------------------------
# §11 async copy (PTX only)
# ---------------------------------------------------------------------------


def _async_setup(b):
    b.param("X", BufferType(DType.BF16))
    g = GlobalTensor(
        dtype=DType.BF16,
        shape=(64, 64),
        stride=(64, 1),
        name="X",
        param=b.function.params[-1],
    )
    A = b.smem_alloc("A", DType.BF16, (64, 64))
    return g, A


@case("async/copy_16b", "cp.async 16-byte count", backends=("ptx",))
def _(b):
    g, A = _async_setup(b)
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=16)


@case("async/copy_8b", "cp.async 8-byte count", backends=("ptx",))
def _(b):
    g, A = _async_setup(b)
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=8)


@case("async/copy_4b", "cp.async 4-byte count", backends=("ptx",))
def _(b):
    g, A = _async_setup(b)
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    b.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=4)


@case("async/commit_and_wait", "commit_group + multiple wait_group", backends=("ptx",))
def _(b):
    b.async_commit()
    b.async_wait(0)
    b.async_wait(2)


@case("async/copy_dynamic_src", "dynamic src row widens through u64 math", backends=("ptx",))
def _(b):
    g, A = _async_setup(b)
    dyn_row = b.thread_idx("x")
    col = b.const(DType.U32, 0)
    dst_row = b.const(DType.U32, 0)
    b.async_copy(A, g, dst_idx=(dst_row, col), src_idx=(dyn_row, col), count=16)


@case("async/copy_predicated", "predicated cp.async", backends=("ptx",))
def _(b):
    g, A = _async_setup(b)
    row = b.const(DType.U32, 0)
    col = b.const(DType.U32, 0)
    z = b.const(DType.F32, 0.0)
    one = b.const(DType.F32, 1.0)
    p = b.cmp("lt", z, one)
    b.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=16, pred=p)


# ---------------------------------------------------------------------------
# §12 MMA + fragment loads / stores
# ---------------------------------------------------------------------------


@case("mma/load_matrix_a_m16n8k16_bf16", "load A fragment (bf16, k=16)", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_BF16)
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS_K16)


@case("mma/load_matrix_b_m16n8k16_bf16", "load B fragment (bf16, k=16)", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_BF16)
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS_K16)


@case("mma/load_matrix_c_m16n8k16_bf16", "load C fragment (f32 acc)", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_BF16)
    C = b.smem_alloc("C", DType.F32, (16, 8))
    b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)


@case("mma/store_matrix_m16n8k16_bf16", "store D (f32) fragment to smem", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_BF16)
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    D = b.smem_alloc("D", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS_K16)
    bb = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS_K16)
    c = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    d = b.mma("m16n8k16_bf16", a, bb, c)
    b.store_matrix(D, d, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)


@case("mma/m16n8k16_bf16", "mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_BF16)
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS_K16)
    bb = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS_K16)
    c = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_bf16", a, bb, c)


@case("mma/m16n8k16_f16", "mma f16.f16.f32", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_F16)
    A = b.smem_alloc("A", DType.F16, (16, 16))
    B = b.smem_alloc("B", DType.F16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_f16", which="a", reg_offsets=_A_OFFS_K16)
    bb = b.load_matrix(B, "m16n8k16_f16", which="b", reg_offsets=_B_OFFS_K16)
    c = b.load_matrix(C, "m16n8k16_f16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_f16", a, bb, c)


@case("mma/m16n8k8_bf16", "mma m16n8k8 bf16", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K8_BF16)
    A = b.smem_alloc("A", DType.BF16, (16, 8))
    B = b.smem_alloc("B", DType.BF16, (8, 8))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k8_bf16", which="a", reg_offsets=_A_OFFS_K8)
    bb = b.load_matrix(B, "m16n8k8_bf16", which="b", reg_offsets=_B_OFFS_K8)
    c = b.load_matrix(C, "m16n8k8_bf16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k8_bf16", a, bb, c)


@case("mma/m16n8k16_e4m3", "mma m16n8k16 e4m3 (sm_89+)", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_E4M3)
    A = b.smem_alloc("A", DType.E4M3, (16, 16))
    B = b.smem_alloc("B", DType.E4M3, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_e4m3", which="a", reg_offsets=_A_OFFS_FP8_K16)
    bb = b.load_matrix(B, "m16n8k16_e4m3", which="b", reg_offsets=_B_OFFS_FP8_K16)
    c = b.load_matrix(C, "m16n8k16_e4m3", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_e4m3", a, bb, c)


@case("mma/m16n8k32_e4m3", "mma m16n8k32 e4m3 (double-k fp8)", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K32_E4M3)
    A = b.smem_alloc("A", DType.E4M3, (16, 32))
    B = b.smem_alloc("B", DType.E4M3, (8, 32))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k32_e4m3", which="a", reg_offsets=_A_OFFS_FP8_K32)
    bb = b.load_matrix(B, "m16n8k32_e4m3", which="b", reg_offsets=_B_OFFS_FP8_K32)
    c = b.load_matrix(C, "m16n8k32_e4m3", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k32_e4m3", a, bb, c)


@case("mma/m16n8k16_e5m2", "mma m16n8k16 e5m2", backends=("ptx",))
def _(b):
    b.register_shape(_SHAPE_M16N8K16_E5M2)
    A = b.smem_alloc("A", DType.E5M2, (16, 16))
    B = b.smem_alloc("B", DType.E5M2, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_e5m2", which="a", reg_offsets=_A_OFFS_FP8_K16)
    bb = b.load_matrix(B, "m16n8k16_e5m2", which="b", reg_offsets=_B_OFFS_FP8_K16)
    c = b.load_matrix(C, "m16n8k16_e5m2", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_e5m2", a, bb, c)


# ---------------------------------------------------------------------------
# §13 fragment ops
# ---------------------------------------------------------------------------


def _mma_acc(b, shape=_SHAPE_M16N8K16_BF16):
    """Build an accumulator fragment (result of one MMA) so frag_ ops
    have an f32 fragment to consume."""
    b.register_shape(shape)
    A = b.smem_alloc("A", DType.BF16, (shape.m, shape.k))
    B = b.smem_alloc("B", DType.BF16, (shape.n, shape.k))
    C = b.smem_alloc("C", DType.F32, (shape.m, shape.n))
    a = b.load_matrix(A, shape.name, which="a", reg_offsets=_A_OFFS_K16)
    bb = b.load_matrix(B, shape.name, which="b", reg_offsets=_B_OFFS_K16)
    c = b.load_matrix(C, shape.name, which="c", reg_offsets=_CD_OFFS)
    return b.mma(shape.name, a, bb, c)


@case("frag/apply_scale", "frag_apply: scalar multiply per slot", backends=("ptx",))
def _(b):
    d = _mma_acc(b)
    scale = b.const(DType.F32, 2.0)
    b.frag_apply("m16n8k16_bf16", d, lambda x: b.mul(x, scale))


@case("frag/apply_add_bias", "frag_apply: add scalar bias", backends=("ptx",))
def _(b):
    d = _mma_acc(b)
    bias = b.const(DType.F32, 0.5)
    b.frag_apply("m16n8k16_bf16", d, lambda x: b.add(x, bias))


@case("frag/apply_relu", "frag_apply: max(x, 0)", backends=("ptx",))
def _(b):
    d = _mma_acc(b)
    zero = b.const(DType.F32, 0.0)
    b.frag_apply("m16n8k16_bf16", d, lambda x: b.max(x, zero))


@case("frag/reduce_row_max", "frag_reduce axis=row kind=max", backends=("ptx",))
def _(b):
    d = _mma_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="max", axis="row", cd_offsets=_CD_OFFS)


@case("frag/reduce_row_add", "frag_reduce axis=row kind=add", backends=("ptx",))
def _(b):
    d = _mma_acc(b)
    b.frag_reduce("m16n8k16_bf16", d, kind="add", axis="row", cd_offsets=_CD_OFFS)


@case("frag/convert_f32_to_bf16", "frag_convert: acc → a-frag bf16", backends=("ptx",))
def _(b):
    d = _mma_acc(b)
    b.frag_convert(
        "m16n8k16_bf16",
        [d],
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=_CD_OFFS,
    )


@case("frag/for_each_smem_store", "frag_for_each: store each slot to smem", backends=("ptx",))
def _(b):
    d = _mma_acc(b)
    Output = b.smem_alloc("O", DType.F32, (16, 8))

    def body(elem, row, col):
        b.store(Output, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, body, cd_offsets=_CD_OFFS)


# ---------------------------------------------------------------------------
# Driver — build + lower + diff
# ---------------------------------------------------------------------------


def _build_msl_caps():
    from quark.device import DeviceCaps, DeviceFamily

    return DeviceCaps(
        family=DeviceFamily.METAL,
        name="test-metal",
        compute_unit_count=32,
        subgroup_width=32,
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


def lower_case(case: Case, backend: str) -> str:
    """Build the case's IR and lower it to PTX/MSL text."""
    b = Builder("t")
    b.begin_function("f")
    case.builder(b)
    if b._fn is not None:
        b.end_function()

    if backend == "ptx":
        from quark.lower.ptx import PtxLowerer

        return PtxLowerer().lower_module(b.module).ptx
    elif backend == "msl":
        from quark.lower.msl import MslLowerer

        return MslLowerer(_build_msl_caps()).lower_module(b.module).source
    else:
        raise ValueError(f"unknown backend: {backend}")


def detect_backend() -> str:
    return "msl" if sys.platform == "darwin" else "ptx"


def golden_path(backend: str, case_id: str) -> Path:
    ext = "ptx" if backend == "ptx" else "msl"
    return GOLDEN_ROOT / backend / f"{case_id.replace('/', '__')}.{ext}"


def select_cases(backend: str, pattern: str | None) -> list[Case]:
    out = [c for c in CASES if backend in c.backends]
    if pattern:
        out = [c for c in out if fnmatch.fnmatch(c.id, pattern)]
    return out


def write_manifest(backend: str, cases: list[Case]) -> None:
    manifest = {
        "backend": backend,
        "count": len(cases),
        "cases": [
            {"id": c.id, "description": c.description, "backends": list(c.backends)} for c in cases
        ],
    }
    path = GOLDEN_ROOT / backend / "MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def cmd_list(backend: str, pattern: str | None) -> int:
    cases = select_cases(backend, pattern)
    print(f"{len(cases)} cases for backend={backend}:")
    for c in cases:
        marks = "/".join(c.backends)
        print(f"  {c.id:<45s}  [{marks}]  {c.description}")
    return 0


def cmd_update(backend: str, pattern: str | None) -> int:
    cases = select_cases(backend, pattern)
    (GOLDEN_ROOT / backend).mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    for c in cases:
        try:
            out = lower_case(c, backend)
        except Exception as e:
            print(_color(f"[FAIL] {c.id}: {e}", "31"))
            traceback.print_exc()
            fail += 1
            continue
        p = golden_path(backend, c.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out)
        ok += 1
        print(_color(f"[ok]   {c.id}  → {p.relative_to(REPO_ROOT)}", "32"))
    write_manifest(backend, cases)
    print(f"\nwrote {ok} goldens, {fail} failures, backend={backend}")
    return 1 if fail else 0


def cmd_check(backend: str, pattern: str | None) -> int:
    cases = select_cases(backend, pattern)
    missing = 0
    mismatch = 0
    errors = 0
    ok = 0
    for c in cases:
        p = golden_path(backend, c.id)
        if not p.exists():
            print(_color(f"[miss] {c.id}  (no golden at {p.relative_to(REPO_ROOT)})", "33"))
            missing += 1
            continue
        try:
            got = lower_case(c, backend)
        except Exception as e:
            print(_color(f"[err]  {c.id}: {e}", "31"))
            errors += 1
            continue
        want = p.read_text()
        if got != want:
            print(_color(f"[diff] {c.id}", "31"))
            diff = difflib.unified_diff(
                want.splitlines(keepends=True),
                got.splitlines(keepends=True),
                fromfile=str(p.relative_to(REPO_ROOT)),
                tofile="<emitted>",
                n=2,
            )
            sys.stdout.writelines(diff)
            print()
            mismatch += 1
        else:
            ok += 1
    print(
        f"\nchecked {len(cases)} cases: {ok} ok, "
        f"{mismatch} mismatched, {missing} missing, {errors} errored"
    )
    return 0 if (mismatch == 0 and errors == 0 and missing == 0) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("ptx", "msl", "auto"), default="auto")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--update", action="store_true", help="write/overwrite goldens")
    g.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--filter", help="glob to subset cases (e.g. 'arith/*')")
    args = ap.parse_args(argv)

    backend = detect_backend() if args.backend == "auto" else args.backend

    if args.list:
        return cmd_list(backend, args.filter)
    if args.update:
        return cmd_update(backend, args.filter)
    return cmd_check(backend, args.filter)


if __name__ == "__main__":
    raise SystemExit(main())
