#!/usr/bin/env python
"""Check import order and grouping in src/, tests/, tools/.

DORMANT — see tools/ci/README.md. Can be activated after the
rename so the 'from popcorn ...' patterns stabilize.

Rule: imports follow this grouping, with blank lines between groups:

    # 1. stdlib
    from __future__ import annotations    ← always first if present
    from dataclasses import dataclass
    from pathlib import Path

    # 2. third-party (only torch, really)
    import torch

    # 3. first-party absolute
    from popcorn.ir import DType, Module
    from popcorn.kernels.base import Kernel

No 'from X import *'. No 'import popcorn.ir as ir'. No private-name
cross-module imports (from X import _y).

This checker is intentionally simpler than ruff's 'I' (isort) rule —
it's the minimum form we can enforce with stdlib-only code. Once
the pre-commit hook runs 'uvx ruff check', ruff's isort rule takes
over and this script can be retired.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SCAN_ROOTS = ("src", "tests", "tools")

STDLIB_MODULES = frozenset(
    {
        # A short stdlib list is enough for detecting obvious violations.
        # Not exhaustive — but covers what popcorn uses.
        "abc",
        "argparse",
        "ast",
        "asyncio",
        "base64",
        "collections",
        "concurrent",
        "contextlib",
        "copy",
        "ctypes",
        "dataclasses",
        "datetime",
        "enum",
        "errno",
        "functools",
        "gc",
        "hashlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "queue",
        "random",
        "re",
        "shutil",
        "signal",
        "socket",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "traceback",
        "types",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
        "zlib",
        "__future__",
    }
)

THIRD_PARTY_KNOWN = frozenset({"torch", "numpy", "pytest"})

FIRST_PARTY_PREFIXES = ("popcorn", "popcorn")  # "popcorn" until rename


def classify(module: str) -> str:
    """Return 'stdlib', 'third_party', 'first_party', or 'unknown'."""
    top = module.split(".")[0]
    if top in STDLIB_MODULES:
        return "stdlib"
    if top in FIRST_PARTY_PREFIXES:
        return "first_party"
    if top in THIRD_PARTY_KNOWN:
        return "third_party"
    return "unknown"


def check_file(path: Path) -> list[str]:
    """Return a list of 'path:lineno: reason' strings."""
    fails: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    # Collect (lineno, category) for every top-level import.
    imports: list[tuple[int, str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, classify(alias.name), alias.name))
                # Ban 'import popcorn.ir as ir' (rename-style)
                if alias.asname and alias.name.split(".")[0] in FIRST_PARTY_PREFIXES:
                    fails.append(
                        f"{path.as_posix()}:{node.lineno}: "
                        f"use 'from {alias.name} import X' instead of "
                        f"'import {alias.name} as {alias.asname}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append((node.lineno, classify(module), module))
            # Ban 'from X import *'
            for alias in node.names:
                if alias.name == "*":
                    fails.append(
                        f"{path.as_posix()}:{node.lineno}: "
                        f"'from {module} import *' is banned — use explicit names"
                    )
                # Ban 'from X import _y' (private cross-module import)
                elif alias.name.startswith("_") and classify(module) == "first_party":
                    # Same-package imports are fine; cross-package aren't.
                    # We conservatively flag all private imports from popcorn.*
                    fails.append(
                        f"{path.as_posix()}:{node.lineno}: "
                        f"private import '{alias.name}' from '{module}' — "
                        f"make it public or restructure"
                    )

    # Check ordering: stdlib → third_party → first_party, no interleaving.
    order = {"stdlib": 0, "third_party": 1, "first_party": 2, "unknown": 3}
    prev_category = -1
    for lineno, cat, mod in imports:
        cur = order[cat]
        if cur < prev_category:
            fails.append(
                f"{path.as_posix()}:{lineno}: "
                f"import '{mod}' ({cat}) appears after a later-category import; "
                f"imports must be grouped stdlib → third_party → first_party"
            )
        prev_category = max(prev_category, cur)

    return fails


def main() -> int:
    fails: list[str] = []
    for root in SCAN_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            fails.extend(check_file(path))

    if fails:
        print("import_order check FAILED:")
        for f in fails[:50]:
            print(f"  {f}")
        if len(fails) > 50:
            print(f"  ... and {len(fails) - 50} more")
        print()
        print(
            "See docs/CONVENTIONS.md §imports for the required grouping. "
            "Once the pre-commit hook runs 'uvx ruff check', ruff's 'I' "
            "rule enforces this more precisely."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
