"""``quark.functional.gemm`` — C = A @ B^T over QuarkTensor.

Signature:

    C = pcf.gemm(A, B, *, out_dtype=None, compute_dtype=None, b_shuffled=False)

A: [M, K], B: [N, K] (weight, stored transposed so K is fast axis).
Returns C: [M, N] in ``out_dtype`` (defaults to A's dtype).

EXEMPT FROM 500-LINE RULE: the GEMM functional surface owns three
backend dispatch paths (cuBLAS shortcut, NAX fast path on Metal, and
the generic call_with_bindings fallback) plus the per-shape NAX tile
table; they need to live together so the dispatch order — eligibility
gate → cache lookup → fast path → fallback — reads top-to-bottom
without splitting state across modules.
"""

from __future__ import annotations

import os
import sys

from quark.functional._dispatch import call_with_bindings, launcher, make_autotune
from quark.kernels import get

_GemmCls = None
_IS_METAL = sys.platform == "darwin"

# Dtype combos where ``cublasLtMatmul`` gives us scale-free row-major
# A × B^T directly. fp8 A+B requires both operands fp8 (cublasLt's fp8
# matmul doesn't do mixed half/fp8), so Linear pre-casts x to e4m3 when
# the weight is fp8 to land here. Mixed input combos + pre-shuffled B +
# fused activations stay on the custom kernel.
_CUBLAS_COMBOS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("bf16", "bf16", "bf16"),
        ("bf16", "bf16", "f32"),
        ("f16", "f16", "f16"),
        ("f16", "f16", "f32"),
        ("e4m3", "e4m3", "bf16"),
        ("e4m3", "e4m3", "f16"),
        ("e4m3", "e4m3", "e4m3"),
        ("e4m3", "e4m3", "f32"),
    }
)


def _cls():
    """Lazy import so module import order doesn't matter."""
    global _GemmCls
    if _GemmCls is None:
        _GemmCls = get("gemm")
    return _GemmCls


def _dtype_str(t) -> str:
    dt = t.dtype
    return dt if isinstance(dt, str) else str(dt)


def _try_cublas(A, B, *, out_dtype, compute_dtype, b_shuffled, activation, bias, out):
    """Short-circuit to ``cublasLtMatmul`` when the combo supports it.

    Returns the output tensor on a hit, or ``None`` to tell the caller
    to fall through to the custom-kernel path. Kept deliberately
    conservative: anything the gate can't prove cuBLAS handles in one
    call (scales, fused epilogues, weight pre-shuffle, non-standard
    dtypes) falls back immediately.
    """
    if _IS_METAL:
        return None
    if os.environ.get("QUARK_DISABLE_CUBLAS") == "1":
        return None
    # Validation knob: bypass the shortcut to force cuBLAS-eligible
    # shapes through the autotune cache's cuBLAS candidate path instead
    # (kernels/gemm/cublas_dispatch.py). Used to A/B the two paths
    # before the shortcut is retired in a follow-up commit. Separate
    # from QUARK_DISABLE_CUBLAS so cuBLAS stays usable via autotune
    # when the shortcut is off.
    if os.environ.get("QUARK_DISABLE_CUBLAS_SHORTCUT") == "1":
        return None
    if b_shuffled or activation is not None:
        return None

    from quark.runtime.tensor import QuarkTensor

    if not isinstance(A, QuarkTensor) or not isinstance(B, QuarkTensor):
        return None
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        return None

    # cublasLt's BIAS epilogue requires a 1D [N] vector.
    if bias is not None:
        if not isinstance(bias, QuarkTensor):
            return None
        if bias.ndim != 1 or int(bias.shape[0]) != int(B.shape[0]):
            return None

    a_dt = _dtype_str(A)
    b_dt = _dtype_str(B)

    # compute_dtype: custom kernel uses it to down-cast A during the
    # smem load (bf16 → e4m3 for mixed-dtype fp8 MMA). cuBLAS has no
    # equivalent zero-cost path without scales, so any explicit
    # compute_dtype that differs from a_dt forces fallback.
    if compute_dtype is not None and str(compute_dtype) != a_dt:
        return None

    if out is not None:
        c_dt = _dtype_str(out)
    else:
        c_dt = str(out_dtype) if out_dtype is not None else a_dt

    if (a_dt, b_dt, c_dt) not in _CUBLAS_COMBOS:
        return None

    from quark.runtime.cublas import CublasRuntime

    if not CublasRuntime.is_available():
        return None

    M = int(A.shape[0])
    K = int(A.shape[1])
    N = int(B.shape[0])

    if out is None:
        out = QuarkTensor.empty(M, N, dtype=c_dt)
    elif tuple(out.shape) != (M, N):
        return None

    from quark.graph import active_stream

    stream = active_stream() or 0
    bias_ptr = bias.data_ptr() if bias is not None else 0
    bias_dt = _dtype_str(bias) if bias is not None else None
    if os.environ.get("QUARK_CUBLAS_VERBOSE") == "1":
        tag = f" bias={bias_dt}" if bias is not None else ""
        print(
            f"[cublas] matmul M={M} N={N} K={K} a={a_dt} b={b_dt} c={c_dt}{tag} stream={stream}",
            flush=True,
        )
    CublasRuntime.instance().matmul(
        a_ptr=A.data_ptr(),
        b_ptr=B.data_ptr(),
        c_ptr=out.data_ptr(),
        M=M,
        N=N,
        K=K,
        a_dtype=a_dt,
        b_dtype=b_dt,
        c_dtype=c_dt,
        bias_ptr=bias_ptr,
        bias_dtype=bias_dt,
        stream=stream,
    )
    return out


_NAX_FASTPATH_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────
# Per-shape NAX tile picker. Sourced from ``scripts/sweep_nax_tiles.py``
# microbench on M5 Max. Each entry is the fastest valid NAX config for
# that (M, K, N, activation) tuple; the sweep also reports a "universal"
# config (best across all shapes) used as the fallback for any shape
# not in the table.
#
# Why a hard-coded table over autotune: the launcher's autotune search
# kept landing on the slower m8n8k8 fallback for these shapes (we hit
# this in commit ``11b23bf``), and the candidate-list approach we used
# after that picked a single tile (``BM=64 BN=128 BK=64 nstages=2``)
# that's optimal for nothing in particular. The table is small (5
# shapes plus the universal fallback) and the sweep takes ~1 minute
# to regenerate; updating it is cheaper than teaching the autotuner.
# ─────────────────────────────────────────────────────────────────────


def _nax_cfg(BM, BN, BK, n_warps, n_stages, a_pad, b_pad):
    from quark.kernels.gemm.config import GemmConfig

    return GemmConfig(
        BM=BM,
        BN=BN,
        BK=BK,
        n_warps=n_warps,
        n_stages=n_stages,
        a_pad=a_pad,
        b_pad=b_pad,
        main_shape="m16n32k16_nax_bf16",
    )


# (M, K, N, activation) → preferred GemmConfig.
_NAX_PER_SHAPE: dict[tuple[int, int, int, str | None], object] = {}


def _build_per_shape_table():
    """Defer construction so ``GemmConfig`` is imported only on Metal.

    Empty at 360p — the universal config below dominates the
    ``M=128`` shapes end-to-end. Per-shape entries from the full sweep
    didn't beat it (each distinct config compiles a distinct MSL
    pipeline, and the resulting shader-cache / L2 churn outweighs the
    isolated-microbench gains). See ``scripts/sweep_qkv_proj.py``
    (2026-05-02): 9 candidates on the qkv_proj 360p shape all regressed.

    At 720p (``M=512``) two of the four hot GEMMs do beat the universal
    config — see ``scripts/sweep_fc2_720p.py`` and
    ``scripts/sweep_gemms_720p.py`` (2026-05-03, M5 Max). The shape
    keys for fc2 and out_proj are M=512-specific, so 360p shapes don't
    pick them up; the universal config still applies there.

      shape (M=512)     baseline sat   override sat   delta
      mlp.fc2+gate         262.7 ms      250.7 ms    -12.0 ms
      out_proj+gate        260.7 ms      244.0 ms    -16.7 ms (layered on fc2)

    The wider-M (BM=128) and finer K-pipelining (BK=32) win matches
    the shape geometry: M=512 has 8 row tiles at BM=64 vs 4 at BM=128,
    and the smaller BK gives more pipeline iterations across K=2048
    or K=8192 — both shapes have >>2× the K-iter count needed to hide
    n_stages=2 prefetch. fc1 (silu fused) and qkv_proj at M=512 still
    prefer the universal config — both reject every wider-tile
    candidate end-to-end despite isolated profiling suggesting headroom.

    Per-shape GEMM compare in per-call mode (M5 Max, TFLOPs/s — matches
    the chained world-model pattern where each gemm's output is
    consumed before the next call dispatches):

      shape       quark   MLX     ratio
      qkv_proj     9.81    9.86   tied (MLX 1.005×)
      out_proj     5.27    4.12   quark 1.28×
      mlp.fc1     16.10   16.49   tied (MLX 1.024×)
      mlp.fc2     14.40    8.62   quark 1.67×

    Aggregate quark ~17 % faster — consistent with the 60.7 ms vs
    70.3 ms end-to-end gap. The two engines are essentially tied on
    per-shape kernel throughput on the realistic chained workload.
    Don't trust batched-mode microbench results (N matmuls scheduled
    then one eval) — those measure MLX's lazy-graph compiler quality
    across independent ops, which is shape-dependent and produces
    misleading 2-3× swings that don't reflect a chained workload.
    """
    # 720p mlp.fc2+gate and out_proj+gate share the same per-shape pick:
    # BM=128 BK=32 (n_warps=8, n_stages=2). The lookup keys are M=512-
    # specific so 360p (M=128) shapes never pick these up, and the
    # universal fallback still serves every other shape.
    #
    # The wider-M m=32 NAX shape (``m32n32k16_nax_bf16``) was tested
    # here on fc2 (sweep_m32_720p.py, 2026-05-05 M5 Max). The layered
    # sweep showed a ~40ms saturated win, but a clean paired A/B on a
    # cool SoC (3 runs alternating) put the m=32 fc2 within ±2 ms of
    # the m=16 baseline — i.e. inside run-to-run noise. The IR plumbing
    # for m=32 is kept (registry entry + parametric _nax_frag_layout +
    # ValueShape width=32); not wired here because end-to-end there is
    # no win at this shape on this hardware.
    _wider_m_finer_k = _nax_cfg(128, 128, 32, 8, 2, 8, 8)
    _NAX_PER_SHAPE[(512, 8192, 2048, None)] = _wider_m_finer_k
    _NAX_PER_SHAPE[(512, 2048, 2048, None)] = _wider_m_finer_k


def _universal_nax_fallback() -> object:
    """Single NAX config used for every shape that satisfies the NAX
    MMA constraints.

    PICKED EMPIRICALLY FROM THE END-TO-END BENCH, NOT FROM THE
    ISOLATED-MICROBENCH SWEEP. The latter (see
    ``scripts/sweep_nax_tiles.py``) is unreliable for picking a
    real-workload config:

      * ``n_stages=1`` wins in isolated microbench because the same op
        runs back-to-back with B always L2-resident, so the
        double-buffered prefetch in ``n_stages=2`` has nothing to
        hide. In the real bench B is large and gets evicted between
        layers; ``n_stages=2`` reclaims its prefetch advantage and
        ``n_stages=1`` regresses by ~40 ms / frame.
      * Wider ``BN=256`` wins in isolation (less total threadgroups,
        better arithmetic intensity per dispatch) but in the real
        workload it churns L2 against the surrounding kv_cache /
        norm / attn ops and again regresses end-to-end.
      * Per-shape configs (different tile per (M, K, N) pair) compile
        4 distinct MSL pipelines for the bench's 4 hot GEMM shapes;
        the resulting shader-cache thrashing dominates the per-op
        speedup the microbench predicted.

    The current pick (``BM=64 BN=128 BK=64 n_warps=8 n_stages=2
    a_pad=8 b_pad=16``) is the same one that's been benchmark-stable
    at 58.5 ms / frame since commit ``11b23bf``. Don't change it
    without an end-to-end bench number to back the change.
    """
    return _nax_cfg(64, 128, 64, 8, 2, 8, 16)


def _pick_nax_config(M: int, K: int, N: int, activation: str | None):
    """Return the preferred NAX config for this shape, or None if no
    NAX config is valid. Caller still validates via
    ``cls(spec, config).is_valid()`` — different chips / chip variants
    (or M-not-divisible shapes like the ctrl-side ``M=16`` GEMMs) may
    reject the picked config; those fall back to the autotune-picked
    m8n8k8 path.
    """
    _build_per_shape_table()
    cfg = _NAX_PER_SHAPE.get((M, K, N, activation))
    if cfg is not None:
        return cfg
    return _universal_nax_fallback()


def _try_nax_gemm(
    A,
    B,
    *,
    out_dtype,
    compute_dtype,
    b_shuffled,
    activation,
    bias,
    out,
    gate=None,
    residual=None,
    gate_groups=1,
):
    """Short-circuit to the autotuned ``GemmKernel`` MSL emitted by
    ``GemmKernel.build_metal`` when on Apple Metal.

    Compiles ``GemmKernel`` with the autotuned config and dispatches via
    the lazy ``_md.queue_launch`` (zero-copy for ``QuarkTensor`` inputs).
    Bypasses ``call_with_bindings`` to avoid the eager-launch path's
    per-call ``np.asarray`` (which would force ``eval_queue`` on
    ``QuarkTensor`` inputs) and ~80 µs of dict / spec assembly.

    Originally NAX-only (``m16n32k16_nax_bf16``). Now extended to every
    autotuned MMA shape (m8n8k8 fallbacks etc.) — the kernel binding
    plan is identical regardless of MMA shape, and the surrounding
    Python wrapping cost is what the fast path saves. The function
    name is kept for backwards-compat with cache key naming.

    ``gate`` + ``residual`` (both required together): build a fused
    GemmSpec with ``has_gate_residual=True`` and bind the two extra
    tensors at queue_launch time. The compiled NAX kernel uses the
    ``store_matrix_gate_residual`` epilogue from ``_nax_store_accs``
    and writes ``Out = Residual + Gate_bcast * (A @ Bᵀ)`` directly
    from the accumulator — saves a standalone ``AdaGateResidualKernel``
    dispatch + the bf16 round-trip on the GEMM output.

    Returns the output array on hit, or ``None`` to fall through.
    Conservative: bf16 only, no shuffle, aligned shapes.
    """
    if not _IS_METAL:
        return None
    if os.environ.get("QUARK_DISABLE_NAX") == "1":
        return None
    if b_shuffled or bias is not None:
        return None
    if activation is not None and activation != "silu":
        return None
    if compute_dtype is not None:
        return None

    fused_gate = gate is not None and residual is not None

    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(B.shape[0])

    # Hot-path cache: skip spec construction + autotune lookup +
    # compile lookup + grid/block computation by keying on the shape
    # tuple, dtype, and activation. After warmup every block in the
    # transformer hits this path with N stable shapes (qkv_proj,
    # out_proj, fc1, fc2 — 4 per block, identical across blocks), so
    # this cache is small and the per-call overhead drops from
    # several hundred microseconds to a single dict lookup. Fused-
    # gate-residual gets its own cache slot keyed on G + the fused
    # marker so the plain and fused compiled kernels don't collide.
    cache_key = (M, K, N, out_dtype, activation, fused_gate, gate_groups)
    cached = _NAX_FASTPATH_CACHE.get(cache_key)
    if cached is not None:
        mod, threads_grid, block, bias_dummy_handle = cached
    else:
        from quark.ir import DType

        a_dt = DType.from_backend(A)
        b_dt = DType.from_backend(B)
        if a_dt is not DType.BF16 or b_dt is not DType.BF16:
            return None

        cls = _cls()
        spec = cls.spec_from_tensors(
            A,
            B,
            out_dtype=out_dtype,
            compute_dtype=None,
            b_shuffled=False,
            activation=activation,
            has_bias=False,
        )
        if fused_gate:
            from dataclasses import replace

            spec = replace(spec, has_gate_residual=True, G=int(gate_groups))

        lc = launcher()
        config = lc._autotune.lookup_or_search(cls, spec)

        # Override the autotune pick when the shape is NAX-compatible
        # but autotune landed on the m8n8k8 fallback. NAX
        # (m16n32k16_nax_bf16) is faster on M5+ for every shape the
        # waypoint bench hits; per-shape best tiles come from
        # ``scripts/sweep_nax_tiles.py`` on M5 Max (see commit message
        # for the full table — re-run the sweep when porting to a
        # different chip / memory-bandwidth profile).
        try:
            from quark.device import current_device

            nax_ok = (
                current_device().caps.supports_nax
                and config.main_shape.startswith("m8n8k8")
                and (M % 16 == 0)
                and (N % 32 == 0)
                and (K % 16 == 0)
            )
            if nax_ok:
                cand = _pick_nax_config(M, K, N, activation)
                if cand is not None:
                    try:
                        if cls(spec=spec, config=cand).is_valid():
                            config = cand
                    except Exception:
                        pass
        except Exception:
            pass

        compiled = lc.compile(cls, spec, config)
        mod = compiled.module
        grid = compiled.grid_fn(*compiled.grid_args)
        block = compiled.block_fn(*compiled.block_args)
        threads_grid = (grid[0] * block[0], grid[1] * block[1], grid[2] * block[2])

        # Pre-warm the bias dummy here so the cached entry can hold the
        # handle directly — saves the getattr probe + per-call alloc
        # branch on the hot path. Plain (non-fused) spec also reuses the
        # dummy for the Gate / Residual slots since the binding plan has
        # 5 input slots regardless of ``has_gate_residual``; the kernel
        # just doesn't read those slots when the spec flag is off.
        bias_dummy = getattr(compiled, "_metal_bias_dummy", None)
        if bias_dummy is None:
            from quark.runtime.tensor import QuarkTensor as _QT

            c_dt_str = str(out_dtype) if out_dtype is not None else "bf16"
            bias_dummy = _QT.zeros(1, dtype=c_dt_str)
            compiled._metal_bias_dummy = bias_dummy
        _h = bias_dummy.metal_handle
        assert _h is not None, "bias_dummy must be on Metal storage"
        bias_dummy_handle = int(_h)
        _NAX_FASTPATH_CACHE[cache_key] = (mod, threads_grid, block, bias_dummy_handle)

    import numpy as np

    # Resolve A, B: QuarkTensor metal_handle → zero copy; numpy → memcpy.
    # The kernel binding has 5 input slots (A, B, Bias, Gate, Residual).
    # Plain GEMM binds the (1,) bias_dummy to all three placeholder
    # slots; fused gate-residual binds the real Gate + Residual handles.
    n_inputs = 5
    input_handles: list[int] = [0, 0, bias_dummy_handle, bias_dummy_handle, bias_dummy_handle]
    input_ptrs: list[int] = [0] * n_inputs
    input_nbytes: list[int] = [0] * n_inputs
    for i, arr in enumerate((A, B)):
        metal_h = getattr(arr, "metal_handle", None)
        if metal_h is not None:
            input_handles[i] = metal_h
        else:
            input_handles[i] = -1
            narr = np.ascontiguousarray(arr)
            input_ptrs[i] = narr.ctypes.data
            input_nbytes[i] = narr.nbytes
    if fused_gate:
        # Slots 3 = Gate, 4 = Residual. Both must be QuarkTensors with
        # a metal_handle for the fast path; numpy carriers would need a
        # cache_wrap_input round-trip and aren't worth the extra branch.
        for slot, arr in ((3, gate), (4, residual)):
            metal_h = getattr(arr, "metal_handle", None)
            if metal_h is None:
                # Soft fall-through: caller passed a non-Metal carrier
                # for one of the fused inputs. Bail and let the slow
                # path handle it via call_with_bindings.
                return None
            input_handles[slot] = metal_h

    from quark.drivers import _metal_dispatch as _md

    out_nbytes = M * N * 2
    c_dt = str(out_dtype) if out_dtype is not None else "bf16"
    out_handle, out_ptr = _md.queue_launch(
        mod.pipeline,
        input_handles,
        input_ptrs,
        input_nbytes,
        out_nbytes,
        mod._binding_plan_int,
        threads_grid,
        block,
        mod.smem_bytes,
    )
    from quark.runtime.tensor import QuarkTensor, _contiguous_strides, _MetalStorage

    storage = _MetalStorage(out_handle, out_ptr, out_nbytes)
    # ``activation="silu"`` is now fused inside the NAX kernel epilogue
    # (see GemmKernel._build_metal_nax_*); no post-hoc silu kernel.
    return QuarkTensor(storage, (M, N), _contiguous_strides((M, N)), 0, c_dt)


def _gemm_impl(
    A,
    B,
    *,
    out_dtype=None,
    compute_dtype=None,
    b_shuffled=False,
    activation=None,
    bias=None,
    gate=None,
    residual=None,
    gate_groups=1,
    out=None,
):
    fused_gate = gate is not None and residual is not None
    nax_out = _try_nax_gemm(
        A,
        B,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        bias=bias,
        out=out,
        gate=gate,
        residual=residual,
        gate_groups=gate_groups,
    )
    if nax_out is not None:
        return nax_out

    if not fused_gate:
        cublas_out = _try_cublas(
            A,
            B,
            out_dtype=out_dtype,
            compute_dtype=compute_dtype,
            b_shuffled=b_shuffled,
            activation=activation,
            bias=bias,
            out=out,
        )
        if cublas_out is not None:
            return cublas_out

    cls = _cls()
    has_bias = bias is not None
    spec_kwargs = dict(
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        has_bias=has_bias,
    )
    spec = cls.spec_from_tensors(A, B, **spec_kwargs)
    if fused_gate:
        # Re-derive the spec with the gate fields set; spec_from_tensors
        # doesn't take has_gate_residual since most callers don't fuse.
        from dataclasses import replace

        spec = replace(spec, has_gate_residual=True, G=int(gate_groups))
    provided = {"A": A, "B": B}
    if has_bias:
        provided["Bias"] = bias
    if fused_gate:
        provided["Gate"] = gate
        provided["Residual"] = residual
    if out is not None:
        provided["Out"] = out
    auto_alloc_names: tuple[str, ...] = ()
    if not has_bias:
        auto_alloc_names = auto_alloc_names + ("Bias",)
    if not fused_gate:
        auto_alloc_names = auto_alloc_names + ("Gate", "Residual")
    if out is None:
        auto_alloc_names = auto_alloc_names + ("Out",)
    result = call_with_bindings(
        cls,
        spec,
        provided=provided,
        auto_alloc=auto_alloc_names,
        like=A,
    )
    return result["Out"]


def gemm(
    A,
    B,
    out_dtype=None,
    compute_dtype=None,
    b_shuffled=False,
    activation=None,
    bias=None,
    gate=None,
    residual=None,
    gate_groups=1,
    out=None,
):
    """Compute ``C = A @ B.T``.

    ``out``: optional pre-allocated output buffer to write into. When
    provided, skips the auto_alloc path — lets callers (e.g. Linear
    layers) reuse a cached buffer keyed by M and avoid per-call allocs.

    ``gate`` + ``residual`` (both required together): fuse an
    AdaGate-residual epilogue, ``Out = Residual + Gate_bcast * (A @ Bᵀ)``.
    Gate has shape ``[gate_groups, N]`` and broadcasts over each block
    of ``M // gate_groups`` consecutive rows; residual has shape
    ``[M, N]``. Saves one dispatch + one bf16 round-trip on the
    accumulator vs the standalone ``pcf.ada_gate_residual`` op.
    """
    return _gemm_impl(
        A,
        B,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        bias=bias,
        gate=gate,
        residual=residual,
        gate_groups=gate_groups,
        out=out,
    )


gemm.autotune = make_autotune(_gemm_impl, _cls)  # ty: ignore[unresolved-attribute]
