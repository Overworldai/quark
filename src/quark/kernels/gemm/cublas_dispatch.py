"""cuBLAS-as-autotune-candidate plumbing for the universal GEMM.

Treats ``cublasLtMatmul`` as one execution option among the PTX
kernel's tune-space configs. The autotune cache enumerates a single
``GemmConfig(impl="cublas")`` alongside the PTX cartesian product,
times all candidates under the same event-bracketed timing loop,
and picks the winner per ``(spec, device)``.

Why a separate module rather than folding into ``functional/gemm.py``:
the autotune cache, the launcher's timing path, and user-facing
``pcf.gemm`` dispatch all need the same eligibility + runner logic.
Keeping it in one module avoids three slightly-different copies of
"which specs can cuBLAS take" drifting apart.

Eligibility today matches ``_try_cublas`` in ``functional/gemm.py``
exactly — same ``_CUBLAS_COMBOS``, same "no activation / no shuffle
/ bias must be [N]" constraints. The mixed-dtype widening (where A
gets cast inside dispatch so bf16/f16 × e4m3 is cuBLAS-eligible)
lands in a follow-up commit along with the removal of
``Linear.forward``'s pre-cast.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# cuBLAS-handleable (a_dtype, b_dtype, c_dtype) triples. Kept in sync
# with ``functional/gemm.py::_CUBLAS_COMBOS``; will consolidate when
# ``_try_cublas`` is removed.
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


def is_cublas_forced() -> bool:
    """Return True iff cuBLAS should short-circuit past the autotuner.

    Defaults to ON — every cuBLAS-eligible GEMM spec runs through
    cublasLtMatmul without any PTX enumeration, cache read, or cache
    write. Empirically this is a large speedup on the bread-and-butter
    shapes (the autotuner's timing harness tends to mis-rank cuBLAS on
    mixed-dtype specs where the bf16 → e4m3 cast dominates the measured
    time, hiding cuBLAS's ~2× win on the matmul proper).

    Three-state knob:
      - ``QUARK_FORCE_CUBLAS=0`` — turn OFF the force. cuBLAS stays
        available as a normal autotune candidate, the PTX search runs,
        and the winner goes to cache like any other config.
      - ``QUARK_DISABLE_CUBLAS=1`` — turn OFF cuBLAS entirely. Force is
        a no-op because eligibility fails.
      - default or ``QUARK_FORCE_CUBLAS=1`` — force cuBLAS.

    The returned config is ephemeral — it does not persist to the
    autotune cache, so unsetting the force reverts immediately to
    whatever the cache already knows (or triggers a real search).
    """
    return os.environ.get("QUARK_FORCE_CUBLAS", "1") != "0"


def is_cublas_eligible(spec, caps: Any | None = None) -> bool:
    """Return True iff ``spec`` can be executed via cublasLtMatmul.

    Pure function of spec + env + runtime availability. ``caps`` is
    accepted for signature parity with the ``alt_configs(spec, caps)``
    hook but not currently used — cuBLAS availability is a process-
    level property, not a device-cap one, so it's looked up via
    ``CublasRuntime.is_available()``.
    """
    if sys.platform == "darwin":
        return False
    if os.environ.get("QUARK_DISABLE_CUBLAS") == "1":
        return False
    if getattr(spec, "b_shuffle", False):
        return False
    if getattr(spec, "activation", None) is not None:
        return False

    a_dt = _dtype_str(spec.a_dtype)
    b_dt = _dtype_str(spec.b_dtype)
    out_dt = _dtype_str(spec.out_dtype)

    # When A and B disagree on dtype and B is fp8, ``dispatch_cublas``
    # casts A→e4m3 inline so cublasLt can take both fp8 operands. The
    # autotune timing loop measures cast+matmul together, which is
    # apples-to-apples against the PTX kernel's in-smem compute_dtype
    # down-cast. Eligibility checks the post-cast dtype triple.
    post_a_dt = "e4m3" if (b_dt == "e4m3" and a_dt in ("bf16", "f16")) else a_dt
    if post_a_dt != b_dt:
        return False

    if (post_a_dt, b_dt, out_dt) not in _CUBLAS_COMBOS:
        return False

    try:
        from quark.runtime.cublas import CublasRuntime
    except ImportError:
        return False
    if not CublasRuntime.is_available():
        return False

    return True


def make_cublas_config():
    """Return the singleton-ish ``GemmConfig(impl="cublas")`` candidate.

    The PTX knobs are left at the dataclass defaults (they're ignored
    by the cuBLAS dispatch path, but must be valid values so the config
    round-trips through the cache serializer).
    """
    from quark.kernels.gemm.config import GemmConfig

    return GemmConfig(impl="cublas")


def dispatch_cublas(*, A, B, Out, Bias=None, stream: int | None = None):
    """Run cublasLtMatmul on the given buffers. Returns ``Out``.

    Shared by the autotune timing path and the user-facing dispatch
    shim — both want the same behavior so the autotune comparison is
    apples-to-apples with the PTX kernel.

    A / B / Out / Bias are ``QuarkTensor`` instances; caller owns their
    allocation. ``stream=None`` (default) auto-picks
    ``quark.graph.active_stream()`` so callers inside a graph capture
    automatically land on the capture stream without having to thread
    it through. Pass a non-None stream to override.

    When ``A.dtype`` is bf16/f16 and ``B.dtype`` is e4m3, A is cast
    to e4m3 inline via ``pcf.quantize_e4m3`` so cublasLt can take
    both fp8 operands. The cast runs on-stream (same stream as the
    matmul) so it's included in the kernel's ``_time_callable``
    timing — the autotuner sees the real "cast + matmul" cost of
    this path vs. the PTX kernel's in-smem cvt + matmul.
    """
    from quark.runtime.cublas import CublasRuntime

    if stream is None:
        from quark.graph import active_stream

        stream = active_stream() or 0

    a_dt = _dtype_str(A.dtype)
    b_dt = _dtype_str(B.dtype)
    c_dt = _dtype_str(Out.dtype)

    if b_dt == "e4m3" and a_dt in ("bf16", "f16"):
        import quark.functional as pcf

        A = pcf.quantize_e4m3(A)
        a_dt = "e4m3"

    M = int(A.shape[0])
    K = int(A.shape[1])
    N = int(B.shape[0])

    bias_ptr = Bias.data_ptr() if Bias is not None else 0
    bias_dt = _dtype_str(Bias.dtype) if Bias is not None else None

    CublasRuntime.instance().matmul(
        a_ptr=A.data_ptr(),
        b_ptr=B.data_ptr(),
        c_ptr=Out.data_ptr(),
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
    return Out


def _dtype_str(dt) -> str:
    """Render a DType / str / backend-dtype as the lowercase quark string."""
    from quark.ir import DType

    if isinstance(dt, str):
        return dt
    if isinstance(dt, DType):
        return str(dt)
    return DType.from_backend(dt) if dt is not None else ""
