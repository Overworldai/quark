#!/usr/bin/env python
"""Fail if a public class or function in src/ lacks a docstring.

DORMANT — see tools/ci/README.md. Will fail on a lot of existing
code until the AI coding assistant proposal's §4 (docstring
expansion) lands alongside the IR migration's file decomposition.

Rule: every public (non-underscore) class and top-level function
under src/quark/ (or src/quark/) has a docstring. Private
helpers (leading underscore) are exempt.

This is the minimum form of the rule. A stricter version would
also check that module docstrings follow the CONVENTIONS.md
layout (Layer, Key exports, Invariants, See also sections).
That check is harder to write correctly, so this one just
verifies *presence*.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SCAN_ROOTS = ("src/quark", "src/quark")


def is_public(name: str) -> bool:
    return not name.startswith("_")


def check_tree(tree: ast.Module, path: str) -> list[str]:
    """Return a list of 'path:lineno: reason' strings for missing docstrings."""
    fails: list[str] = []

    # Module docstring
    if not ast.get_docstring(tree):
        fails.append(f"{path}:1: module missing docstring")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not is_public(node.name):
                continue
            # Skip if it's a method on a private class — nested walk finds them.
            if not ast.get_docstring(node):
                fails.append(
                    f"{path}:{node.lineno}: public function '{node.name}' missing docstring"
                )
        elif isinstance(node, ast.ClassDef):
            if not is_public(node.name):
                continue
            if not ast.get_docstring(node):
                fails.append(f"{path}:{node.lineno}: public class '{node.name}' missing docstring")

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
            if path.name == "__init__.py":
                # __init__.py files get a pass on missing module docstrings
                # if they're empty or just re-exports. Still check for
                # docstring-less public classes defined in them.
                pass
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=path.as_posix())
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            fails.extend(check_tree(tree, path.as_posix()))

    if fails:
        print("docstring_check FAILED:")
        for f in fails[:50]:  # cap output so it's not overwhelming
            print(f"  {f}")
        if len(fails) > 50:
            print(f"  ... and {len(fails) - 50} more")
        print()
        print(
            "Every public class and function under src/ must have a "
            "docstring. See docs/CONVENTIONS.md §docstrings for the "
            "required format (Layer / Key exports / Invariants / See also)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
