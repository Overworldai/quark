"""``popcorn.functional.patchify`` — patchify kernel (strided A-tile GEMM).

    out = pcf.patchify(X_flat, W, B=1, C=32, H=32, W_spatial=64)

``X_flat``: [B, C*H*W] (flat image). ``W``: [d_model, C*ph*pw] (flat weight).
Returns [B*Hp*Wp, d_model]. No reshapes, no permutes — the kernel's
A-tile loader reads the strided (h_tok, w_tok, c, ph, pw) pattern directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from popcorn.functional._dispatch import call_with_bindings
from popcorn.kernels import get

if TYPE_CHECKING:
    pass

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("patchify")
    return _Cls


def patchify(X, W, *, B: int, C: int, H: int, W_spatial: int, ph: int = 2, pw: int = 2, out=None):
    cls = _cls()
    d_model = int(W.shape[0])
    from popcorn.ir import DType
    from popcorn.kernels.patchify.spec import PatchifySpec

    spec = PatchifySpec(
        B=B,
        C=C,
        H=H,
        W=W_spatial,
        ph=ph,
        pw=pw,
        d_model=d_model,
        dtype=DType.from_backend(X.dtype),
    )
    provided = {"X": X, "W": W}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=X)
    return result["Out"]


# Keep backward-compat alias for the old composition helper.
def patchify_2x2(X, W):
    """Legacy composition helper — delegates to the patchify kernel now."""
    if X.ndim == 4:
        B, C, H, Wd = (int(s) for s in X.shape)
    elif X.ndim == 2:
        B = int(X.shape[0])
        # Caller must pass the spatial dims explicitly; can't infer from flat.
        raise ValueError("patchify_2x2: use pcf.patchify() for flat [B, C*H*W] input")
    else:
        raise ValueError(f"patchify_2x2: unexpected X rank {X.ndim}")

    if W.ndim == 4:
        d_model = int(W.shape[0])
        W_flat = W.reshape(d_model, -1)
    else:
        W_flat = W

    # Flatten X to [B, C*H*W] for the kernel.
    X_flat = X.reshape(B, -1)
    return patchify(X_flat, W_flat, B=B, C=C, H=H, W_spatial=Wd)
