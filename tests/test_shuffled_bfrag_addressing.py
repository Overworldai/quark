"""Tests for the addressing scheme used by ``emit_shuffled_bfrag_load``.

The L0 emitter builds a per-lane smem address of the form::

    smem_base
      + warp_id * (BN_per_warp * bstride * elem_bytes)     # warp offset (dyn)
      + lane_id * FRAG_BYTES                               # lane offset  (dyn)
      + n_tile * 8 * bstride * elem_bytes                  # tile row      (static)
      + k_step * 32 * FRAG_BYTES                           # tile col      (static)

and issues a ``ld.shared.v{FRAG_REGS}.b32``. These tests don't exercise
the emitter itself (that requires a CUDA device); instead they verify
that the arithmetic matches what ``shuffle_b_for_frag_load`` produces,
by comparing the pure-Python address → lane-fragment-data mapping
against the explicit shuffle formula.

A failure here means either the emitter's offset formula or the
offline shuffle permutation has drifted, so the vectorized load will
read the wrong bytes. This is exactly the bug that BN ≥ 64 +
multi-warp configs used to hit.
"""

from __future__ import annotations

import numpy as np
import pytest

from quark.weight_shuffle import shuffle_b_for_frag_load


def _compute_frag_from_shuffled(
    W_shuf: np.ndarray,
    *,
    BN: int,
    BK: int,
    b_pad: int,
    warp_id: int,
    n_warps: int,
    lane_id: int,
    n_tile_local: int,
    k_step: int,
    frag_regs: int,
    elem_bytes: int,
) -> bytes:
    """Read ``FRAG_BYTES`` from the shuffled weight at exactly the offset
    the L0 emitter computes. Returns the raw bytes."""
    bstride = BK + b_pad
    BN_per_warp = (BN // 8) // n_warps * 8
    FRAG_BYTES = frag_regs * 4

    # Flatten (first BN rows, bstride cols) as a single contiguous byte block:
    # matches how smem sees the tile after cp.async.
    # Assume W_shuf[:BN, :BK] contains one B tile.
    tile = W_shuf[:BN, :BK].reshape(-1).view(np.uint8).reshape(BN, BK * elem_bytes)
    tile_bytes = tile.reshape(-1)

    # Emitter's byte offset:
    warp_off = warp_id * BN_per_warp * bstride * elem_bytes
    lane_off = lane_id * FRAG_BYTES
    static_row = n_tile_local * 8 * bstride * elem_bytes
    static_col = k_step * 32 * FRAG_BYTES
    off = warp_off + lane_off + static_row + static_col
    return bytes(tile_bytes[off : off + FRAG_BYTES])


def _expected_frag_from_plain(
    W_plain: np.ndarray,
    *,
    BN: int,
    BK: int,
    warp_id: int,
    n_warps: int,
    lane_id: int,
    n_tile_local: int,
    k_step: int,
    frag_regs: int,
    elem_bytes: int,
    mma_k: int,
) -> bytes:
    """Independently compute the bytes that lane ``lane_id`` in warp
    ``warp_id`` *should* receive as its B fragment for one (n_tile, k_step),
    derived directly from the PTX ISA formula on the PLAIN (un-shuffled)
    matrix:

        lane = 4 * gid + tid       (gid ∈ [0,8), tid ∈ [0,4))
        reg r reads bf16 elements at
            (row = tile_row_base + gid,
             col = k_step * mma_k + tid*2 + r*8)
    """
    # Global n-tile index within this B tile.
    NT_per_warp = (BN // 8) // n_warps
    n_tile_global = warp_id * NT_per_warp + n_tile_local
    tile_row_base = n_tile_global * 8

    gid = lane_id >> 2
    tid = lane_id & 3
    mma_k_bytes = mma_k * elem_bytes

    W_bytes_flat = W_plain.view(np.uint8).reshape(W_plain.shape[0], W_plain.shape[1] * elem_bytes)
    row = tile_row_base + gid
    # Each register r is 4 bytes: tid*4 base within the k_step's mma_k_bytes
    # block, then r*16 shift (one 8-element/16-byte stride between regs).
    out = bytearray()
    for r in range(frag_regs):
        col_byte = k_step * mma_k_bytes + tid * 4 + r * 16
        out.extend(bytes(W_bytes_flat[row, col_byte : col_byte + 4]))
    return bytes(out)


@pytest.mark.parametrize(
    "BN,BK,n_warps",
    [
        (64, 16, 4),
        (64, 32, 4),
        (64, 64, 4),
        (128, 16, 4),
        (128, 32, 4),
        (128, 64, 4),
        (256, 16, 8),
        (64, 16, 8),
    ],
)
def test_shuffled_addressing_matches_plain_frag_bf16(BN, BK, n_warps):
    """For every (warp, lane, n_tile, k_step), the bytes read by the
    emitter's address formula from the *shuffled* tensor must equal the
    bytes the PTX ISA formula would have read from the *plain* tensor.
    """
    # Sanity on config.
    assert (BN // 8) % n_warps == 0
    NT_per_warp = (BN // 8) // n_warps
    K_STEPS = BK // 16
    frag_regs = 2  # bf16 m16n8k16
    elem_bytes = 2  # bf16

    # Distinct-byte W so any single-byte mix-up shows up.
    rng = np.random.default_rng(0)
    W_np = rng.integers(1, 250, size=(BN, BK), dtype=np.uint16).astype(np.uint16)
    W = W_np.view(np.float16)  # treat bytes opaquely — interpret is irrelevant
    W_shuf = shuffle_b_for_frag_load(W, K_CHUNK=BK, mma_k=16)

    for warp_id in range(n_warps):
        for n_tile_local in range(NT_per_warp):
            for k_step in range(K_STEPS):
                for lane_id in range(32):
                    got = _compute_frag_from_shuffled(
                        W_shuf,
                        BN=BN,
                        BK=BK,
                        b_pad=0,
                        warp_id=warp_id,
                        n_warps=n_warps,
                        lane_id=lane_id,
                        n_tile_local=n_tile_local,
                        k_step=k_step,
                        frag_regs=frag_regs,
                        elem_bytes=elem_bytes,
                    )
                    want = _expected_frag_from_plain(
                        W,
                        BN=BN,
                        BK=BK,
                        warp_id=warp_id,
                        n_warps=n_warps,
                        lane_id=lane_id,
                        n_tile_local=n_tile_local,
                        k_step=k_step,
                        frag_regs=frag_regs,
                        elem_bytes=elem_bytes,
                        mma_k=16,
                    )
                    assert got == want, (
                        f"BN={BN} BK={BK} n_warps={n_warps} "
                        f"warp={warp_id} n_tile={n_tile_local} "
                        f"k_step={k_step} lane={lane_id}: "
                        f"shuffled-read {got.hex()} != plain-frag {want.hex()}"
                    )
