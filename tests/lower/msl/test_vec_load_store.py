"""Exhaustive vec_load / vec_store coverage for the MSL lowerer.

Exercises every width × dtype combination ``NormalizeAndStore`` /
``TileLoad`` / the ld.shared.v{N}.b32 coalesced epilogues rely on.
Each test builds a tiny kernel that:
  1. Copies an input gmem tensor into smem via per-thread scalar stores.
  2. Reads the smem via ``vec_load`` with a possibly *different* dtype
     (e.g. B32 vec_load from a bf16 buffer, reinterpreting 2 bf16 as
     1 b32 per vec element).
  3. Writes the loaded vector back to output via ``vec_store`` — same
     buffer-dtype / vec-dtype split.

The first bug we found in Metal attention was exactly this: a B32
width-4 vec_load from a bf16 staging buffer was advancing addresses
by 1 bf16 element per vec element instead of 2, transferring half
the data. These tests regression-lock that fix across every width
the kernels use (2, 4, 8) and every dtype mismatch that can occur.
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_METAL = mx.metal.is_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="no Metal device")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _build_copy_kernel(*, buf_dtype, vec_dtype, width, n_elems):
    """Build a kernel that:
      • declares In/Out as ``buf_dtype`` gmem tensors (shape = n_elems),
      • allocates ``buf_dtype`` smem of same size,
      • each thread copies one chunk into smem,
      • barrier,
      • uses ``vec_load(smem, dtype=vec_dtype, width=W)`` starting at
        per-thread offsets then ``vec_store`` to Out,
    so In → Out is an identity copy through the vec_load/vec_store path.

    Returns the kernel class. Threads = ``n_elems // width`` so each
    thread handles exactly one vec.
    """
    from dataclasses import dataclass

    from popcorn.ir import BufferType, Builder, DType
    from popcorn.ir.module import ParamAttrs
    from popcorn.ir.tensor import GlobalTensor
    from popcorn.kernels.base import Kernel
    from popcorn.launcher import ParamSpec

    # vec_dtype in BUFFER ELEMENT units — how many buf elems per vec elem.
    buf_bytes = buf_dtype.bytes
    vec_bytes = vec_dtype.bytes
    assert vec_bytes >= buf_bytes, "vec dtype must be ≥ buf dtype"
    assert vec_bytes % buf_bytes == 0
    buf_per_vec = vec_bytes // buf_bytes

    n_vecs = n_elems // (width * buf_per_vec)  # total vecs in the tensor
    # One thread per vec (must fit in a threadblock — cap at 1024).
    assert 1 <= n_vecs <= 1024, f"n_vecs={n_vecs} not in [1, 1024]"
    n_threads = n_vecs

    @dataclass(frozen=True)
    class _Spec:
        pass

    @dataclass(frozen=True)
    class _Cfg:
        n_warps: int = max(1, (n_threads + 31) // 32)

    class _CopyKernel(Kernel):
        NAME = "vec_copy"
        SPEC_CLS = _Spec
        CONFIG_CLS = _Cfg
        OUTPUT_IDX = -1

        def __init__(self, spec, config):
            self.spec = spec
            self.config = config
            self._compiled = None

        def is_valid(self):
            return True

        def smem_estimate(self):
            return n_elems * buf_bytes

        def grid(self):
            return (1, 1, 1)

        def block(self):
            # Round up to multiple of 32.
            return (((n_threads + 31) // 32) * 32, 1, 1)

        def entry_name(self):
            return "vec_copy"

        def flops(self):
            return 0

        def reference(self, *tensors):
            return tensors[0]

        def emit(self):
            bld = Builder("vec_copy_module")
            bld.begin_function("vec_copy")
            bld.param("In", BufferType(buf_dtype), attrs=ParamAttrs(readonly=True))
            bld.param("Out", BufferType(buf_dtype), attrs=ParamAttrs(readonly=False))

            In_gt = GlobalTensor(
                dtype=buf_dtype,
                shape=(n_elems,),
                stride=(1,),
                name="In",
                param=bld.function.params[0],
            )
            Out_gt = GlobalTensor(
                dtype=buf_dtype,
                shape=(n_elems,),
                stride=(1,),
                name="Out",
                param=bld.function.params[1],
            )

            # 1D smem scratch.
            smem = bld.smem_alloc("scratch", buf_dtype, (n_elems,))

            tid = bld.thread_idx("x")
            # Each thread copies (width * buf_per_vec) elements from In→smem.
            elems_per_thread = width * buf_per_vec
            base = bld.mul(tid, bld.const(DType.U32, elems_per_thread))
            for i in range(elems_per_thread):
                idx = bld.add(base, bld.const(DType.U32, i))
                bld.store(smem, bld.load(In_gt, idx), idx)

            bld.barrier("block")

            # Per-thread vec_load from smem, write to Out.
            # Use a scalar pred = (tid < n_vecs) to cover the case where
            # block size rounded up past n_vecs.
            in_range = bld.cmp("lt", tid, bld.const(DType.U32, n_vecs))
            # Load vec_dtype × width starting at `base` (in buf_dtype units).
            v = bld.vec_load(
                smem,
                base,
                width=width,
                dtype=vec_dtype,
                pred=in_range,
            )
            # Store it to Out.
            bld.vec_store(Out_gt, v, base, pred=in_range)

            bld.end_function()
            return bld.module

        def param_spec(self):
            return ParamSpec.from_function(self.emit().functions[0])

    return _CopyKernel


def _mxbf(t: torch.Tensor) -> mx.array:
    """Torch bf16 → mlx bf16 via uint16 view."""
    assert t.dtype == torch.bfloat16
    b = bytes(t.contiguous().view(torch.uint8).numpy())
    np_ = np.frombuffer(b, dtype=np.uint16).reshape(t.shape)
    return mx.array(np_).view(mx.bfloat16)


def _run_copy(K_cls, In_mx, n_elems):
    """Launch the kernel and return the output as an mlx.array."""
    from popcorn.launcher.launcher import Launcher

    if In_mx.dtype == mx.bfloat16:
        Out = mx.zeros((n_elems,), dtype=mx.bfloat16)
    elif In_mx.dtype == mx.float16:
        Out = mx.zeros((n_elems,), dtype=mx.float16)
    elif In_mx.dtype == mx.float32:
        Out = mx.zeros((n_elems,), dtype=mx.float32)
    elif In_mx.dtype == mx.uint32:
        Out = mx.zeros((n_elems,), dtype=mx.uint32)
    else:
        raise TypeError(f"unsupported dtype {In_mx.dtype}")

    launcher = Launcher()
    compiled = launcher.compile(K_cls, K_cls.SPEC_CLS(), K_cls.CONFIG_CLS())
    result = compiled.launch(buffers=[In_mx, Out])
    assert result is not None
    mx.eval(result[0])
    return result[0]


# ---------------------------------------------------------------------------
# bf16 buffer, B32 vec_load/vec_store — the NormalizeAndStore staged path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2, 4, 8])
def test_vec_bf16_buf_b32_vec(width):
    """vec_load/vec_store with a bf16 buffer reinterpreted as B32.
    Each vec elem covers 2 bf16 elements. Width 2/4/8 corresponds to
    4/8/16 bf16 elements per vec op — matches the coalesced v{N}.b32
    store widths the attn epilogue emits.

    This is THE path that failed before the fix (address advanced by
    1 bf16 per vec elem instead of 2).
    """
    from popcorn.ir import DType

    n_elems = max(width * 2, 64)  # in bf16 elements, at least 1 warp × 2
    # Round n_elems up so n_threads is a valid block.
    n_elems = (n_elems // (width * 2)) * (width * 2)

    K = _build_copy_kernel(
        buf_dtype=DType.BF16,
        vec_dtype=DType.B32,
        width=width,
        n_elems=n_elems,
    )
    g = torch.Generator().manual_seed(0)
    In_t = (torch.randn(n_elems, generator=g) * 2.0).to(torch.bfloat16)
    In_mx = _mxbf(In_t)
    Out_mx = _run_copy(K, In_mx, n_elems)
    Out_t = torch.from_numpy(
        np.frombuffer(
            bytes(np.array(Out_mx.view(mx.uint16))),
            dtype=np.uint16,
        ).copy()
    ).view(torch.bfloat16)

    assert torch.equal(Out_t, In_t), (
        f"bf16 buf × B32 vec (width={width}): copy is lossy.\n"
        f"In  [:16] = {In_t[:16].float().tolist()}\n"
        f"Out [:16] = {Out_t[:16].float().tolist()}\n"
        f"diff mask [:32] = {(Out_t != In_t)[:32].tolist()}"
    )


# ---------------------------------------------------------------------------
# Same-dtype vec_load/store: baseline for the simple path (no bitcast).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2, 4, 8])
def test_vec_bf16_buf_bf16_vec(width):
    """bf16 buf × bf16 vec — no dtype mismatch. Simple per-element
    load/store sequence; should have always worked."""
    from popcorn.ir import DType

    n_elems = max(width * 4, 128)
    n_elems = (n_elems // width) * width

    K = _build_copy_kernel(
        buf_dtype=DType.BF16,
        vec_dtype=DType.BF16,
        width=width,
        n_elems=n_elems,
    )
    g = torch.Generator().manual_seed(1)
    In_t = (torch.randn(n_elems, generator=g) * 2.0).to(torch.bfloat16)
    In_mx = _mxbf(In_t)
    Out_mx = _run_copy(K, In_mx, n_elems)
    Out_t = torch.from_numpy(
        np.frombuffer(
            bytes(np.array(Out_mx.view(mx.uint16))),
            dtype=np.uint16,
        ).copy()
    ).view(torch.bfloat16)

    assert torch.equal(Out_t, In_t), (
        f"bf16 buf × bf16 vec (width={width}): mismatch at "
        f"{(Out_t != In_t).nonzero().flatten()[:4].tolist()}"
    )


@pytest.mark.parametrize("width", [2, 4])
def test_vec_f32_buf_f32_vec(width):
    """f32 buf × f32 vec — simple path at f32."""
    from popcorn.ir import DType

    n_elems = max(width * 4, 64)
    n_elems = (n_elems // width) * width

    K = _build_copy_kernel(
        buf_dtype=DType.F32,
        vec_dtype=DType.F32,
        width=width,
        n_elems=n_elems,
    )
    g = torch.Generator().manual_seed(2)
    In_t = (torch.randn(n_elems, generator=g) * 2.0).float()
    In_mx = mx.array(In_t.numpy())
    Out_mx = _run_copy(K, In_mx, n_elems)
    Out_t = torch.from_numpy(np.array(Out_mx))

    assert torch.equal(Out_t, In_t), f"f32 buf × f32 vec (width={width}): mismatch."


# ---------------------------------------------------------------------------
# bf16 buffer, B64 vec — covers the "even wider" case (8 bf16 / vec elem).
# The attn kernels don't use this today but the generic path should handle it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2])
def test_vec_bf16_buf_b64_vec(width):
    """bf16 buf × B64 vec. Each vec elem covers 4 bf16 elements.
    Rarely used by attn but needed for future 16-byte-per-elem
    cooperative stores."""
    from popcorn.ir import DType

    buf_per_vec = 4  # B64 / BF16 = 8 / 2 = 4
    n_elems = max(width * buf_per_vec * 2, 64)
    n_elems = (n_elems // (width * buf_per_vec)) * (width * buf_per_vec)

    K = _build_copy_kernel(
        buf_dtype=DType.BF16,
        vec_dtype=DType.B64,
        width=width,
        n_elems=n_elems,
    )
    g = torch.Generator().manual_seed(3)
    In_t = (torch.randn(n_elems, generator=g) * 2.0).to(torch.bfloat16)
    In_mx = _mxbf(In_t)
    Out_mx = _run_copy(K, In_mx, n_elems)
    Out_t = torch.from_numpy(
        np.frombuffer(
            bytes(np.array(Out_mx.view(mx.uint16))),
            dtype=np.uint16,
        ).copy()
    ).view(torch.bfloat16)

    assert torch.equal(Out_t, In_t), f"bf16 buf × B64 vec (width={width}): copy is lossy."
