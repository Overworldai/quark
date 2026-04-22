"""randn is on-demand only (no bench / fuzz problems). The offline
autotuner skips kernels with an empty problem list.
"""

from __future__ import annotations


def randn_problems():
    return []
