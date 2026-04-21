"""``popcorn.functional.randn`` — on-device standard-normal generator.

    out = pcf.randn(shape=(1, 65536), dtype="bf16", counter_offset=cnt,
                    out=preallocated_tensor, seed0=..., seed1=...)

Counter-based stateless Philox4x32-10 + Box-Muller — see
``popcorn.kernels.randn.kernel`` for the kernel body. One launch produces
``N = prod(shape)`` standard normals directly into the output buffer;
no host→device transfer, no Python-side RNG loop.

Per-call ``counter_offset`` (u32) is the per-draw salt the caller bumps
each frame to get uncorrelated noise. ``seed0``/``seed1`` are compile-
time key bits: varying them produces independent streams, but changing
them triggers a recompile, so pick once at model init.
"""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings
from popcorn.ir import DType
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("randn")
    return _Cls


def randn(
    *,
    shape: tuple[int, ...] | int,
    dtype: str = "bf16",
    counter_offset,
    out=None,
    seed0: int = 0xDEADBEEF,
    seed1: int = 0xBADC0FFE,
):
    """Generate standard-normal samples of ``shape`` in ``dtype``.

    ``counter_offset``: a 1-element u32 device tensor. Host bumps it per
    draw; changing it picks an uncorrelated region of the Philox stream
    without recompiling the kernel.
    ``out``: optional preallocated output tensor. When provided, the
    kernel writes into it directly (shape / dtype must match). This is
    the graph-capture path — lets the caller hold a persistent buffer
    and replay.
    """
    from popcorn.kernels.randn.spec import RandnSpec

    if isinstance(shape, int):
        shape = (shape,)
    N = 1
    for d in shape:
        N *= int(d)

    cls = _cls()
    dt = DType.coerce(dtype)
    assert dt is not None
    spec = RandnSpec(N=N, dtype=dt, seed0=seed0, seed1=seed1)

    provided = {"counter_offset": counter_offset}
    auto_alloc: tuple[str, ...] = ()
    if out is not None:
        provided["Out"] = out
    else:
        auto_alloc = ("Out",)

    result = call_with_bindings(
        cls,
        spec,
        provided=provided,
        auto_alloc=auto_alloc,
        like=counter_offset,
    )
    Out = result["Out"]
    if len(shape) != 1:
        Out = Out.reshape(*shape)
    return Out
