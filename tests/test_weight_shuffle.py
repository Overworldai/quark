"""Tests for the weight pre-shuffling.

The shuffle layout must match what quark.mma.bfrag_load_shuffled_e4m3_k16
expects to read. We verify by reconstructing each lane's expected fragment
data from the shuffled output and comparing to the original.
"""

import numpy as np
import pytest

from quark.weight_shuffle import shuffle_weights_e4m3_k16, verify_shuffled_layout


def _make_distinct_weights(N: int, K: int) -> np.ndarray:
    """Create a weight matrix where every byte has a unique trace value
    (mod 256). Useful to spot single-byte mismatches in the shuffle.
    """
    return (np.arange(N * K, dtype=np.int64) % 256).astype(np.uint8).reshape(N, K)


def test_shuffle_8x16_no_pad():
    """Smallest case: one (n=8, k=16) tile, one k_step. Verify each lane's
    4 bytes appear at the right shuffled offset.
    """
    N, K = 8, 16
    W = _make_distinct_weights(N, K)
    W_shuf = shuffle_weights_e4m3_k16(W, kchunk=16, bpad=0)

    assert W_shuf.shape == (8, 16)
    assert verify_shuffled_layout(W, W_shuf, kchunk=16, bpad=0)


def test_shuffle_16x64_no_pad():
    """Larger case with multiple n-blocks AND multiple k-steps per chunk."""
    N, K = 16, 64
    W = _make_distinct_weights(N, K)
    W_shuf = shuffle_weights_e4m3_k16(W, kchunk=64, bpad=0)

    assert W_shuf.shape == (16, 64)
    assert verify_shuffled_layout(W, W_shuf, kchunk=64, bpad=0)


def test_shuffle_round_trip_via_explicit_formula():
    """Reconstruct W from W_shuf using the formula and verify byte equality."""
    N, K = 32, 128
    kchunk = 64
    bstride = kchunk
    W = _make_distinct_weights(N, K)
    W_shuf = shuffle_weights_e4m3_k16(W, kchunk=kchunk, bpad=0)

    n_block_count = N // 8
    k_chunks = K // kchunk
    k_steps = kchunk // 16

    for n_block in range(n_block_count):
        n_base = n_block * 8
        for k_chunk in range(k_chunks):
            out_chunk_off = k_chunk * bstride
            for k_step in range(k_steps):
                kk = k_step * 16
                for lane in range(32):
                    gid = lane >> 2
                    tid = lane & 3
                    src_bytes = W[
                        n_base + gid,
                        k_chunk * kchunk + kk + tid * 4 : k_chunk * kchunk + kk + tid * 4 + 4,
                    ]
                    for byte in range(4):
                        frag_off = k_step * 32 * 4 + lane * 4 + byte
                        out_row = n_base + frag_off // bstride
                        out_col = out_chunk_off + (frag_off % bstride)
                        assert W_shuf[out_row, out_col] == src_bytes[byte], (
                            f"mismatch n_block={n_block} k_chunk={k_chunk} "
                            f"k_step={k_step} lane={lane} byte={byte}"
                        )


def test_shuffle_rejects_unaligned_dimensions():
    W = _make_distinct_weights(7, 16)
    with pytest.raises(AssertionError, match="multiple of 8"):
        shuffle_weights_e4m3_k16(W, kchunk=16)

    W = _make_distinct_weights(8, 17)
    with pytest.raises(AssertionError, match="multiple of kchunk"):
        shuffle_weights_e4m3_k16(W, kchunk=16)
