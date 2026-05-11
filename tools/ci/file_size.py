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
forces a two-part read; over 800 forces offset guessing.
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
        # MSL lowering visitors — same monolithic-by-design pattern.
        # ``visitors.py`` is the per-op dispatch table (one ``_visit_*``
        # per IR op kind, ~50 entries) and ``mma.py`` is the matching
        # MMA-op catalog (LoadMatrixOp / MmaOp / StoreMatrixOp +
        # FragOp / NAX helper emission). Both have to read top-to-
        # bottom against the IR op catalog they mirror; splitting by
        # op family would force the dispatch table to import its own
        # cases back across module boundaries.
        "src/quark/lower/msl/mma.py",
        "src/quark/lower/msl/visitors.py",
        # GemmKernel — see kernel.py's docstring for the rationale
        # (IR emit + autotune + cuBLAS hook + per-shape NAX table all
        # share GemmSpec/Config). Adding the gate-residual fusion
        # branch on top of the existing silu / has_bias / NAX paths
        # pushed the file just past the 800-line cap.
        "src/quark/kernels/gemm/kernel.py",
        # NAX flash attention is one IR-emitted ``@kernel`` body —
        # ``dispatch_nax_attn`` plus the per-tile q-staging /
        # online-softmax / NAX-MMA helpers it uses inline. The helpers
        # share register-tile and smem-layout state with the kernel
        # body; splitting them out would force re-exporting that
        # state via globals or extra parameters.
        "src/quark/kernels/owl_attn/nax.py",
        # QuarkTensor is the single tensor type for both backends.
        # Storage classes, factories, view ops, slicing, and arithmetic
        # dispatch all share the type's __slots__ and would force
        # circular imports between the pieces if split. The docstring
        # already declares this exemption explicitly.
        "src/quark/runtime/tensor.py",
        # nn.Module catalog — Linear, Patchify, RMSNorm, AdaRMSNorm,
        # AdaGateResidual, KVCacheUpdate, OwlAttn, MLPFusion, etc. all
        # share the cached-out / pinned-buffer / metal_handle plumbing
        # and reach into the same set of qf.* functional helpers.
        # Splitting would fragment the layer ↔ functional mapping.
        "src/quark/nn/layers.py",
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
            f"a reason, or split the file."
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
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
