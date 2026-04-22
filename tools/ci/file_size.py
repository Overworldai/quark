#!/usr/bin/env python
"""Fail if any .py file under src/ or tests/ exceeds the size limits.

DORMANT — see tools/ci/README.md.

Rules:
    * ≤500 lines: always fine
    * 501-800:    allowed only if the module docstring contains the
                  literal phrase 'EXEMPT FROM 500-LINE RULE'
    * >800:       banned unconditionally (exemption phrase ignored)

Designed to be run standalone with zero dependencies beyond stdlib:

    python tools/ci/file_size.py

Exit code 0 on pass, 1 on fail. Offending files printed to stdout.

Why these limits: AI coding assistants read files in bounded chunks.
Files under 500 lines fit comfortably in a single tool call; 500-800
forces a two-part read; over 800 forces offset guessing. See
docs/CONVENTIONS.md for the full rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOFT_LIMIT = 500
HARD_LIMIT = 800
ROOTS = ("src", "tests")
EXEMPTION_PHRASE = "EXEMPT FROM 500-LINE RULE"

# Files excluded from the size check entirely. The same legacy
# in-flight files ruff's `extend-exclude` skips, plus the
# IR/lower/op modules that are large by design (op catalog,
# builder, lowerer) and would fragment poorly.
EXCLUDED: frozenset[str] = frozenset(
    {
        # Legacy code, slated for migration post-universal-GEMM.
        "src/quark/compiler.py",
        "src/quark/program.py",
        "src/quark/cost_model.py",
        "src/quark/blocks/flash_attn.py",
        "src/quark/kernels/base.py",
        "src/quark/kernels/owl_attn.py",
        "src/quark/kernels/owl_attn_phase.py",
        "src/quark/kernels/row_stationary_megakernel.py",
        "src/quark/kernels/moe/inproj.py",
        "src/quark/kernels/moe/outproj.py",
        "src/quark/mma/frag.py",
        "src/quark/mma/store.py",
        # Authored as monolithic modules for cohesion. The op
        # catalog, builder, and PTX lowerer each have ~40 op
        # classes / visit methods that need to live next to each
        # other for the dispatch table to read top-to-bottom. The
        # split-by-dtype-family path the proposal sketches for
        # mma/frag.py would just fragment them.
        "src/quark/ir/builder.py",
        "src/quark/ir/op.py",
        "src/quark/lower/ptx/lower.py",
        "src/quark/autotune.py",
        # Test fixtures with offset tables and verbose parametrize
        # blocks that don't decompose well.
        "tests/lower/ptx/test_matmul.py",
    }
)


def check(path: Path) -> str | None:
    """Return an error message if `path` violates the rules, else None."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        return f"{path}: could not read ({e})"

    n = len(lines)
    if n <= SOFT_LIMIT:
        return None

    # Look for the exemption phrase in the first 40 lines (module docstring area).
    head = "\n".join(lines[:40])
    exempt = EXEMPTION_PHRASE in head

    if n > HARD_LIMIT:
        return (
            f"{path}: {n} lines > hard cap {HARD_LIMIT}. "
            f"Split the file; exemption does not apply at this size."
        )

    if not exempt:
        return (
            f"{path}: {n} lines > soft limit {SOFT_LIMIT}. "
            f"Add '{EXEMPTION_PHRASE}' to the module docstring with "
            f"a reason, or split the file. See docs/CONVENTIONS.md."
        )

    return None


def main() -> int:
    fails: list[str] = []
    for root in ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*.py")):
            # Skip vendored / generated directories.
            if any(part.startswith(".") for part in path.parts):
                continue
            if "__pycache__" in path.parts:
                continue
            # Skip files on the explicit exclusion list.
            if str(path) in EXCLUDED:
                continue
            err = check(path)
            if err:
                fails.append(err)

    if fails:
        print("file_size check FAILED:")
        for f in fails:
            print(f"  {f}")
        print()
        print(f"Soft limit: {SOFT_LIMIT} lines. Hard cap: {HARD_LIMIT} lines.")
        print("See docs/CONVENTIONS.md §file layout for decomposition advice.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
