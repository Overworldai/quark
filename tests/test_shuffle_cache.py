"""Tests for ``cached_shuffle_b`` — the process-wide shuffle cache.

Two invariants worth testing in isolation:

1. The cache keys on Python object identity, not ``data_ptr()`` — a
   naive data_ptr-keyed cache would hand back stale shuffled bytes
   when an allocator recycled storage under a freed prior tensor.

2. Repeated calls with the same tensor return the exact same shuffled
   result (object identity), so autotuners exercising many (spec, config)
   pairs don't redundantly re-permute the weight matrix.

Numpy input path (runtime path for the shuffle cache in the
numpy-refs era — the device-tensor wrappers dispatch through the same
entry point).
"""

from __future__ import annotations

import numpy as np

from popcorn.weight_shuffle import cached_shuffle_b, shuffle_b_for_frag_load


def _make_bf16_u16(N: int, K: int, seed: int = 0) -> np.ndarray:
    """Deterministic bf16-carrier (u16) weight tensor for the tests."""
    rng = np.random.default_rng(seed)
    f32 = rng.standard_normal((N, K)).astype(np.float32)
    return (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def test_shuffle_cache_same_tensor_hits():
    W = _make_bf16_u16(32, 64, seed=1)
    s1 = cached_shuffle_b(W, K_CHUNK=16, mma_k=16)
    s2 = cached_shuffle_b(W, K_CHUNK=16, mma_k=16)
    assert s1 is s2, "expected cache hit on same W"


def test_shuffle_cache_different_params_miss():
    W = _make_bf16_u16(32, 64, seed=2)
    s16 = cached_shuffle_b(W, K_CHUNK=16, mma_k=16)
    s32 = cached_shuffle_b(W, K_CHUNK=32, mma_k=16)
    assert s16 is not s32, "different K_CHUNK must produce distinct cache entries"


def test_shuffle_cache_fresh_tensor_doesnt_return_stale():
    """A data_ptr-keyed cache would fail this test when the allocator
    recycles a freed tensor's storage. Verify the cache returns the
    right shuffle for each — not a stale shuffle from the previous
    tensor."""
    W1 = _make_bf16_u16(8, 16, seed=10)
    s1_of_W1 = cached_shuffle_b(W1, K_CHUNK=16, mma_k=16).copy()
    del W1

    W2 = _make_bf16_u16(8, 16, seed=11)
    s2 = cached_shuffle_b(W2, K_CHUNK=16, mma_k=16)
    expected = shuffle_b_for_frag_load(W2, K_CHUNK=16, mma_k=16)

    # s2 must equal the shuffle of W2 (not of W1).
    assert np.array_equal(s2, expected), (
        "cache returned stale shuffle for a fresh tensor "
        "(likely data_ptr collision with a freed prior tensor)"
    )
    # With seeds 10/11, the two shuffled results must differ.
    assert not np.array_equal(s2, s1_of_W1), "s2 and s1 coincidentally equal — tweak the seeds"


def test_shuffle_cache_result_matches_uncached():
    W = _make_bf16_u16(16, 32, seed=42)
    cached = cached_shuffle_b(W, K_CHUNK=32, mma_k=16)
    uncached = shuffle_b_for_frag_load(W, K_CHUNK=32, mma_k=16)
    assert np.array_equal(cached, uncached)
