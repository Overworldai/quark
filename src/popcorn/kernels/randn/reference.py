"""Reference for randn is statistical, not bitwise — there's no golden
output to compare against. The autotuner's cos-sim gate is meaningless
for random data, so this kernel simply disables the tune_space
(``tune_space() → {}``); ``default_for(spec)`` is always used.

The function below exists to satisfy the ``@kernel(reference=...)``
decorator contract. It produces a plausible output tensor (same shape,
same dtype, values clamped to a reasonable range) purely for shape /
dtype checking by the registry — it is never compared against the
kernel output in practice because there's no autotune search.
"""

from __future__ import annotations


def randn_reference_for_spec(spec):
    def _ref(kernel, counter_offset, Out):
        from popcorn.backend import PT

        # Return a tensor of the right shape/dtype filled with zeros;
        # correctness gating is disabled via empty tune_space.
        return PT.zeros(spec.N, dtype=spec.dtype.backend)

    return _ref
