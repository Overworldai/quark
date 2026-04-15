"""Golden lowerer tests — pins emitted PTX/MSL to check for regressions.

Regenerate goldens after an intentional lowerer change:

    python tools/generate_test_cases.py --update

The case registry lives in `tools/generate_test_cases.py` so running
the diagnostic script and running pytest exercise exactly the same
cases.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import pytest

# Make the registry importable from tools/.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from generate_test_cases import (  # noqa: E402
    CASES,
    detect_backend,
    golden_path,
    lower_case,
)

BACKEND = detect_backend()


def _ids(cases):
    return [c.id for c in cases]


_CASES_FOR_BACKEND = [c for c in CASES if BACKEND in c.backends]


@pytest.mark.parametrize("case", _CASES_FOR_BACKEND, ids=_ids(_CASES_FOR_BACKEND))
def test_golden(case):
    p = golden_path(BACKEND, case.id)
    if not p.exists():
        pytest.skip(
            f"no golden at {p.relative_to(REPO_ROOT)} — "
            "run `python tools/generate_test_cases.py --update` on this device"
        )
    got = lower_case(case, BACKEND)
    want = p.read_text()
    if got == want:
        return
    diff = "".join(
        difflib.unified_diff(
            want.splitlines(keepends=True),
            got.splitlines(keepends=True),
            fromfile=str(p.relative_to(REPO_ROOT)),
            tofile="<emitted>",
            n=2,
        )
    )
    pytest.fail(f"golden mismatch for {case.id}:\n{diff}")
