"""``quark.functional.ada_gate_residual`` — ``out = x + sigmoid(gate) * y``."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.ir import DType
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("ada_gate_residual")
    return _Cls


def _impl(X, Y, gate, *, out=None):
    cls = _cls()
    orig_shape = tuple(X.shape)
    # Match all dtypes to X (the kernel converts to f32 internally).
    x_dtype = DType.from_backend(X)
    if DType.from_backend(Y) != x_dtype:
        Y = Y.astype(x_dtype if isinstance(x_dtype, str) else X.dtype)
    if DType.from_backend(gate) != x_dtype:
        gate = gate.astype(x_dtype if isinstance(x_dtype, str) else X.dtype)
    X2 = X.reshape(-1, orig_shape[-1])
    Y2 = Y.reshape(-1, orig_shape[-1])
    G2 = gate.reshape(-1, gate.shape[-1])
    spec = cls.spec_from_tensors(X2, Y2, G2)
    provided: dict = {"X": X2, "Y": Y2, "gate": G2}
    if out is not None:
        provided["Out"] = out
    result = call_with_bindings(
        cls, spec, provided=provided, auto_alloc=() if out is not None else ("Out",), like=X2
    )
    Out = result["Out"]
    if len(orig_shape) != 2:
        Out = Out.reshape(*orig_shape)
    return Out


_AGR_FASTPATH_CACHE: dict = {}


def ada_gate_residual(X, Y, gate, *, out=None):
    """Compute ``out = x + gate * y``.

    ``out``: optional pre-allocated output buffer. When provided, skips
    auto_alloc — lets callers (e.g. AdaGateResidual layers) reuse a
    cached buffer and avoid per-call allocations.
    """
    import sys

    # Honor caller-provided ``out=``: ``nn.AdaGateResidual.forward``
    # passes a cached buffer specifically to keep this op off the
    # ``queue_launch_ir`` fast path on Metal, which produces token-
    # uniform output for AGR/AdaRMSNorm in the current dispatcher
    # (see comment in ``nn.layers._maybe_cached_out``). Without this
    # gate, MoE blocks silently route through the broken fast path
    # while the dense path's fused ``store_matrix_gate_residual``
    # epilogue side-steps it entirely — which was the "MoE garbage
    # on Metal but dense fine" pattern bisected during the redesign.
    if out is not None:
        return _impl(X, Y, gate, out=out)

    if sys.platform == "darwin":
        from quark.functional._dispatch import queue_launch_ir
        from quark.ir import DType
        from quark.kernels.ada_gate_residual.config import AdaGateResidualConfig
        from quark.kernels.ada_gate_residual.spec import AdaGateResidualSpec

        orig_shape = tuple(X.shape)
        D = orig_shape[-1]
        B = 1
        for s in orig_shape[:-1]:
            B *= int(s)
        dtype_str = getattr(X, "quark_dtype", None) or (
            X.dtype if hasattr(X, "dtype") and isinstance(X.dtype, str) else "f32"
        )
        G = int(gate.shape[0])
        if B % G != 0:
            return _impl(X, Y, gate, out=out)
        M = B // G
        cache_key = (G, M, D, dtype_str)
        cached = _AGR_FASTPATH_CACHE.get(cache_key)
        if cached is None:
            spec = AdaGateResidualSpec(G=G, M=M, D=D, dtype=DType(dtype_str))
            config = None
            for nw, cd, direct in [
                (4, 512, False),
                (4, 1024, False),
                (4, 512, True),
                (2, 512, False),
            ]:
                cand = AdaGateResidualConfig(n_warps=nw, chunk_D=cd, direct=direct)
                if _cls()(spec=spec, config=cand).is_valid():
                    config = cand
                    break
            if config is not None:
                _AGR_FASTPATH_CACHE[cache_key] = (spec, config)
                cached = (spec, config)
        if cached is not None:
            spec, config = cached
            out_h = -1
            if out is not None:
                h = getattr(out, "metal_handle", None)
                if h is not None:
                    out_h = int(h)
            result = queue_launch_ir(
                _cls(),
                spec,
                config,
                inputs=[X, Y, gate],
                out_shape=(B, D),
                out_dtype=dtype_str,
                out_handle=out_h,
            )
            return result.reshape(*orig_shape) if len(orig_shape) != 2 else result
    return _impl(X, Y, gate, out=out)


ada_gate_residual.autotune = make_autotune(_impl, _cls)
