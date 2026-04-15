"""Tests for ``cached_shuffle_b`` — the process-wide shuffle cache.

Two invariants worth testing in isolation:

1. The cache keys on Python object identity, not ``data_ptr()``. Torch's
   caching allocator freely reuses data_ptrs across freshly-allocated
   tensors; a naive data_ptr-keyed cache would hand back stale shuffled
   bytes when the plain tensor's storage was recycled.

2. Repeated calls with the same tensor return the exact same shuffled
   result (object identity), so autotuners exercising many (spec, config)
   pairs don't redundantly re-permute the weight matrix.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from popcorn.weight_shuffle import (  # noqa: E402
    cached_shuffle_b,
    shuffle_b_for_frag_load,
)


def _make_bf16(N: int, K: int, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((N, K), dtype=torch.bfloat16, generator=gen)


def test_shuffle_cache_same_tensor_hits():
    W = _make_bf16(32, 64, seed=1)
    s1 = cached_shuffle_b(W, K_CHUNK=16, mma_k=16)
    s2 = cached_shuffle_b(W, K_CHUNK=16, mma_k=16)
    assert s1 is s2, "expected cache hit on same W"


def test_shuffle_cache_different_params_miss():
    W = _make_bf16(32, 64, seed=2)
    s16 = cached_shuffle_b(W, K_CHUNK=16, mma_k=16)
    s32 = cached_shuffle_b(W, K_CHUNK=32, mma_k=16)
    assert s16 is not s32, "different K_CHUNK must produce distinct cache entries"


def test_shuffle_cache_fresh_tensor_doesnt_return_stale():
    """Torch reuses data_ptrs; a data_ptr-keyed cache would fail this test.
    Create two tensors with DIFFERENT random data (sharing storage after
    an allocator reuse is possible), and verify the cache returns the
    right shuffle for each — not a stale shuffle from the previous
    tensor that happened to share a data_ptr.
    """
    W1 = _make_bf16(8, 16, seed=10)
    s1_of_W1 = cached_shuffle_b(W1, K_CHUNK=16, mma_k=16).clone()
    del W1

    # Force the allocator to churn — hopefully recycle the ptr.
    W2 = _make_bf16(8, 16, seed=11)
    s2 = cached_shuffle_b(W2, K_CHUNK=16, mma_k=16)
    expected = shuffle_b_for_frag_load(W2, K_CHUNK=16, mma_k=16)

    # s2 must equal the shuffle of W2 (not of W1).
    assert torch.equal(s2.view(torch.uint8), expected.view(torch.uint8)), (
        "cache returned stale shuffle for a fresh tensor "
        "(likely data_ptr collision with a freed prior tensor)"
    )
    # And with seeds 10/11, the two shuffled results must differ.
    assert not torch.equal(s2.view(torch.uint8), s1_of_W1.view(torch.uint8)), (
        "s2 and s1 coincidentally equal — tweak the seeds"
    )


def test_shuffle_cache_result_matches_uncached():
    W = _make_bf16(16, 32, seed=42)
    cached = cached_shuffle_b(W, K_CHUNK=32, mma_k=16)
    uncached = shuffle_b_for_frag_load(W, K_CHUNK=32, mma_k=16)
    assert torch.equal(cached.view(torch.uint8), uncached.view(torch.uint8))
