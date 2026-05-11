"""``quark.functional.unpatchify`` — scatter-GEMM to [B, C*H*W]."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("unpatchify")
    return _Cls


def unpatchify(
    X, weight, bias, *, B: int, C: int, H: int, W: int, ph: int = 2, pw: int = 2, out=None
):
    cls = _cls()
    from quark.ir import DType
    from quark.kernels.unpatchify.spec import UnpatchifySpec

    d_model = int(X.shape[1])
    has_bias = bias is not None
    spec = UnpatchifySpec(
        B=B,
        C=C,
        H=H,
        W=W,
        ph=ph,
        pw=pw,
        d_model=d_model,
        dtype=DType.from_backend(X),
        has_bias=has_bias,
    )
    provided = {"X": X, "W": weight, "Bias": bias} if has_bias else {"X": X, "W": weight}
    auto: tuple[str, ...] = ("Out",) if has_bias else ("Bias", "Out")
    if out is not None:
        provided["Out"] = out
        auto = tuple(n for n in auto if n != "Out")
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto, like=X)
    return result["Out"]
