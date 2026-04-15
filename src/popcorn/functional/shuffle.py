"""Offline weight-shuffle helpers matched to the functional surface.

Pre-shuffle B ahead of time so ``pcf.gemm(A, B_shuf, b_shuffled=True)``
(and the MoE equivalents) stays a pure function — no hidden cache, no
per-call permutation, safe under ``torch.compile``.

Use once at load time:

    from popcorn.functional.shuffle import shuffle_b_for_gemm
    B_shuf = shuffle_b_for_gemm(A, B)
    # store B_shuf next to B in the module; pass it to pcf.gemm at call time
    C = pcf.gemm(A, B_shuf, b_shuffled=True)

The shuffle layout (K_CHUNK, mma_k, b_pad) is read from the same
resolved config ``pcf.gemm`` would use on this device — mismatched
shuffles would silently corrupt output.

"""

from __future__ import annotations

from typing import Any

from popcorn.functional._dispatch import launcher
from popcorn.kernels import get as _registry_get
from popcorn.weight_shuffle import ShuffledWeight, shuffle_b_for_frag_load


def get(name: str) -> Any:
    """Registry lookup typed as ``Any`` so call sites can use the
    kernel class's structural API (``spec_from_tensors``, ``CONFIG_CLS``
    etc.) without ty flagging unresolved attributes on ``type``."""
    return _registry_get(name)


def _mma_k_for(kernel_cls, spec, cfg) -> int:
    """Resolve the mma_k for the kernel's primary MMA site under
    ``(spec, cfg)``. The shuffle fragment layout is determined by
    this; mismatched mma_k silently corrupts."""
    kernel = kernel_cls(spec, cfg)
    return kernel._mma_cfg().mma_k


def shuffle_b_for_gemm(A, B, *, out_dtype=None, compute_dtype=None):
    """Offline-shuffle ``B`` so ``pcf.gemm(A, B_shuf, b_shuffled=True)``
    can fast-path the vectorized fragment load.

    ``A``, ``B`` are the plain tensors (caller's normal ``[M,K]`` /
    ``[N,K]`` layout). The returned tensor has the same dtype as ``B``
    but its K dimension may be padded (physical K =
    ``(K // BK) * (BK + b_pad)``) when the resolved config sets
    ``b_pad > 0`` — call sites should treat the return as opaque.
    """
    cls = get("gemm")
    spec = cls.spec_from_tensors(
        A, B, out_dtype=out_dtype, compute_dtype=compute_dtype, b_shuffled=True
    )
    cfg = launcher()._autotune.lookup_or_search(cls, spec)
    mma_k = _mma_k_for(cls, spec, cfg)
    if cfg.b_pad > 0:
        return ShuffledWeight.from_plain(B, kchunk=cfg.BK, bpad=cfg.b_pad, mma_k=mma_k).tensor
    return shuffle_b_for_frag_load(B, cfg.BK, mma_k)


def shuffle_b_for_moe_inproj(X, W_in, *, n_experts, top_k=2, out_dtype=None, compute_dtype=None):
    """Offline-shuffle ``W_in`` for ``pcf.moe_inproj``.

    Requires the resolved config's ``b_shuffle=True``; callers whose
    tuned config leaves ``b_shuffle=False`` should store the plain
    ``W_in`` instead and skip this helper.
    """
    cls = get("moe_inproj")
    # spec_from_tensors needs token_ids + work_list just for validation,
    # but they don't affect the shuffle layout. Synthesize placeholders.
    from popcorn.backend import PT

    total_slots = X.shape[0] * top_k
    token_ids = PT.zeros(total_slots, dtype=PT.int32)
    work_list = PT.zeros(2, dtype=PT.int32)
    spec = cls.spec_from_tensors(
        X,
        W_in,
        token_ids,
        work_list,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )
    cfg = launcher()._autotune.lookup_or_search(cls, spec)
    if not cfg.b_shuffle:
        raise ValueError(
            "shuffle_b_for_moe_inproj: resolved config has b_shuffle=False. "
            "Pass the plain W_in to pcf.moe_inproj, or autotune a config "
            "with b_shuffle=True for this spec first."
        )
    mma_k = _mma_k_for(cls, spec, cfg)
    if cfg.b_pad > 0:
        return ShuffledWeight.from_plain(W_in, kchunk=cfg.BK, bpad=cfg.b_pad, mma_k=mma_k).tensor
    return shuffle_b_for_frag_load(W_in, cfg.BK, mma_k)


def shuffle_b_for_moe_outproj(
    h_in, W_out, *, M, n_experts, top_k=2, out_dtype="bf16", compute_dtype=None
):
    """Offline-shuffle ``W_out`` for ``pcf.moe_outproj``."""
    cls = get("moe_outproj")
    from popcorn.backend import PT

    total_slots = M * top_k
    token_ids = PT.zeros(total_slots, dtype=PT.int32)
    slot_weights = PT.zeros(total_slots, dtype=PT.float32)
    work_list = PT.zeros(2, dtype=PT.int32)
    spec = cls.spec_from_tensors(
        h_in,
        W_out,
        token_ids,
        slot_weights,
        work_list,
        M=M,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )
    cfg = launcher()._autotune.lookup_or_search(cls, spec)
    if not cfg.b_shuffle:
        raise ValueError("shuffle_b_for_moe_outproj: resolved config has b_shuffle=False.")
    mma_k = _mma_k_for(cls, spec, cfg)
    if cfg.b_pad > 0:
        return ShuffledWeight.from_plain(W_out, kchunk=cfg.BK, bpad=cfg.b_pad, mma_k=mma_k).tensor
    return shuffle_b_for_frag_load(W_out, cfg.BK, mma_k)
