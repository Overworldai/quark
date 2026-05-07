"""End-to-end correctness test for the m32n32k16_nax_bf16 shape.

The m=32 NAX shape is a parametric extension over the base
m16n32k16: matmul2d_descriptor(M, N, K) accepts arbitrary M/N/K and
Apple's compiler decomposes into m16 fragments internally. This test
proves that quark's IR plumbing (registry + lowerer + load/store) is
parametric over shape.m and that the per-lane data layout matches
what Apple's compiler produces — without those two assumptions, the
emitted MSL kernel produces garbage even though it compiles cleanly.

Test compiles a tiny GEMM kernel via the Builder API using the new
shape, dispatches it, and compares vs. numpy.

Skipped on non-Metal platforms.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="NAX is Apple Silicon Metal only")


def _f32_to_bf16(x: np.ndarray) -> np.ndarray:
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def _read_f32(addr: int, count: int) -> np.ndarray:
    arr = (ctypes.c_float * count).from_address(addr)
    return np.array(arr, dtype=np.float32, copy=True)


def _build_one_block_gemm(*, shape_id: str, M: int, K: int, N: int):
    """Builder a single-threadgroup, single-simdgroup GEMM kernel that
    computes the full M×K @ K×N for tile-sized (M, K, N). Works for
    any registered NAX shape since the load_matrix / mma / store_matrix
    ops are now parametric over shape.m / shape.n / shape.k.
    """
    from quark.ir import Builder, DType
    from quark.ir.mma_registry import _BY_SHAPE_ID
    from quark.ir.module import ParamAttrs
    from quark.ir.tensor import GlobalTensor
    from quark.ir.types import BufferType

    cfg = _BY_SHAPE_ID[shape_id]
    sh = cfg.shape
    assert M == sh.m and K == sh.k and N == sh.n, (
        f"This test only handles single-tile shapes; got M={M} K={K} N={N} "
        f"vs shape m={sh.m} k={sh.k} n={sh.n}"
    )

    b = Builder("nax_mm_test")
    b.register_shape(sh)
    fn = b.begin_function("nax_mm_test")

    def _decl_tensor(name: str, dtype, shape, *, readonly: bool):
        b.param(name, BufferType(dtype), attrs=ParamAttrs(readonly=readonly))
        stride = []
        prod = 1
        for s in reversed(shape):
            stride.append(prod)
            prod *= int(s)
        stride = tuple(reversed(stride))
        return GlobalTensor(
            dtype=dtype,
            shape=tuple(shape),
            stride=stride,
            name=name,
            param=fn.params[len(fn.params) - 1],
        )

    # Row-major: A[M, K], B[N, K] (transpose_b=true reads it as K-contiguous),
    # C[M, N].
    A = _decl_tensor("A", DType.BF16, (M, K), readonly=True)
    B = _decl_tensor("B", DType.BF16, (N, K), readonly=True)
    C = _decl_tensor("C", DType.F32, (M, N), readonly=False)

    zero_u = b.const(DType.U32, 0)

    # Initialize accumulator. The c_regs entry on the shape gives us
    # the right per-lane width.
    zero_f = b.const(DType.F32, 0.0)
    init_acc = b.vec_build([zero_f] * int(sh.c_regs))

    a = b.load_matrix(A, shape_id, "a", zero_u, zero_u)
    bf = b.load_matrix(B, shape_id, "b", zero_u, zero_u)
    out = b.mma(shape_id, a, bf, init_acc)
    b.store_matrix(C, out, shape_id, "d", zero_u, zero_u)
    b.end_function()
    return b.module


def _compile_and_dispatch(
    module, *, M: int, K: int, N: int, n_simdgroups: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    from quark.device import current_device
    from quark.drivers import _metal_dispatch as _md
    from quark.drivers.metal_harness import TensorParam, build_kernel_source
    from quark.lower.msl import MslLowerer
    from quark.runtime.tensor import QuarkTensor

    lowered = MslLowerer(current_device().caps).lower_module(module)
    full_source, _ = build_kernel_source(
        name=lowered.kernel_name,
        body=lowered.source,
        inputs=[TensorParam(name="A", dtype="bfloat"), TensorParam(name="B", dtype="bfloat")],
        outputs=[TensorParam(name="C", dtype="float")],
        scalars=[],
        header=lowered.header or "",
        max_threads_per_threadgroup=32 * n_simdgroups,
    )
    pipeline = _md.compile(full_source, lowered.kernel_name, 262144)

    rng = np.random.default_rng(0)
    A_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B_f32 = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    A_bf = _f32_to_bf16(A_f32)
    B_bf = _f32_to_bf16(B_f32)
    C_ref = _bf16_to_f32(A_bf) @ _bf16_to_f32(B_bf).T

    A_t = QuarkTensor.from_numpy(A_bf, dtype="bf16")
    B_t = QuarkTensor.from_numpy(B_bf, dtype="bf16")
    C_t = QuarkTensor.zeros(M, N, dtype="f32")

    plan = ((0, 0, 0), (0, 1, 1), (1, 0, 2))
    grid = (32 * n_simdgroups, 1, 1)
    tg = (32 * n_simdgroups, 1, 1)
    _md.queue_launch(
        pipeline,
        [A_t.metal_handle, B_t.metal_handle],
        [0, 0],
        [0, 0],
        C_t._storage.nbytes,
        list(plan),
        grid,
        tg,
        0,
        C_t.metal_handle,
    )
    if _md.has_lazy_pending():
        _md.eval_queue()
    from quark.functional._dispatch import launcher

    launcher().driver.sync(None)
    C_got = _read_f32(C_t._storage.ptr, M * N).reshape(M, N)
    return C_got, C_ref


def _has_nax() -> bool:
    try:
        from quark.device import current_device

        return current_device().caps.supports_nax
    except Exception:
        return False


@pytest.mark.skipif(not _has_nax(), reason="requires NAX (M5+)")
def test_m16n32k16_baseline_correctness():
    """The base m=16 shape produces matmul-correct output. Sanity check
    that the test harness itself works before checking the new shape."""
    module = _build_one_block_gemm(shape_id="m16n32k16_nax_bf16", M=16, K=16, N=32)
    C_got, C_ref = _compile_and_dispatch(module, M=16, K=16, N=32)
    err = float(np.max(np.abs(C_got - C_ref)))
    rel = err / (float(np.max(np.abs(C_ref))) + 1e-9)
    assert rel < 0.02, f"m=16 baseline correctness failed: rel_err={rel:.4f}"


@pytest.mark.skipif(not _has_nax(), reason="requires NAX (M5+)")
def test_m32n32k16_correctness():
    """The new m=32 shape produces matmul-correct output. Validates
    that:
      (1) the registry entry's per-lane regs counts (16/16/32) match
          Apple's actual cooperative_tensor distribution,
      (2) the lowerer's _nax_frag_layout returns the right per-fragment
          (off_r, off_c) pairs for the stacked-m16 composition, and
      (3) the inline matmul2d_descriptor(32, 32, 16, ...) emit + the
          per-lane vec<8> arr decls / loop bounds all stay in sync.
    """
    module = _build_one_block_gemm(shape_id="m32n32k16_nax_bf16", M=32, K=16, N=32)
    C_got, C_ref = _compile_and_dispatch(module, M=32, K=16, N=32)
    err = float(np.max(np.abs(C_got - C_ref)))
    rel = err / (float(np.max(np.abs(C_ref))) + 1e-9)
    assert rel < 0.02, (
        f"m=32 correctness failed: rel_err={rel:.4f} max_err={err:.3f}\n"
        f"got[0,:8]:\n{C_got[0, :8]}\nref[0,:8]:\n{C_ref[0, :8]}\n"
        f"got[16,:8]:\n{C_got[16, :8]}\nref[16,:8]:\n{C_ref[16, :8]}"
    )
