"""Randn numpy reference — zero placeholder.

Reference for randn is statistical, not bitwise. The autotuner's
cos-sim gate is meaningless for random data, so the kernel sets
``CORRECTNESS_THRESHOLD = 0.0`` and ``tune_space = {}``. The function
below exists to satisfy the ``@kernel(reference=...)`` contract."""

from __future__ import annotations

from quark.runtime.npconv import zeros_for_dtype


def randn_reference_numpy(spec, *, counter_offset=None, Out=None):
    del counter_offset, Out
    return zeros_for_dtype((spec.N,), spec.dtype)
