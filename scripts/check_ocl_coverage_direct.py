"""OCL coverage check, direct-construct edition.

Complements ``scripts/check_ocl_coverage.py``: instead of replaying
a Waypoint forward and harvesting (kernel_cls, spec) tuples from
the launcher trace, this script directly instantiates each kernel
of interest with a representative spec, emits its IR (post-legalize),
and cross-references against ``quark.lower.ocl.lower._DISPATCH``.

Use cases:
  - kernels that aren't exercised by the smoke forward
    (``gemm_int``, ``owl_attn_int8``, ``kv_cache_quantize``)
  - kernels missed by Phase 1 because Mac's autotune crashed
    mid-forward (``AdaGateResidualKernel``, ``UnpatchifyKernel``,
    ``ValueResidualPackedKernel``)

Closes Phase 2C.1 (gemm_int visitor check) of
``docs/OCL_E2E_PLAN.md``. Output goes to stdout + appended to
``docs/ocl_kernel_status.md``.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from pathlib import Path
from types import SimpleNamespace


# Same OCL caps shape as check_ocl_coverage.py.
def _make_ocl_caps():
    from quark.device import DeviceFamily
    return SimpleNamespace(
        family=DeviceFamily.INTEL_GPU,
        subgroup_width=32,
        supports_async_copy=False,
        has_fma_bf16x2=False,
        has_native_subgroup_reduce=True,
        atomic_add_vector=frozenset(),
        matmul_shapes=frozenset(),
    )


def _walk_ops(region):
    seen: set[type] = set()
    for op in region.ops:
        seen.add(type(op))
        for r in op.regions:
            seen.update(_walk_ops(r))
    return seen


def _ops_in_module(module):
    out: set[type] = set()
    for fn in module.functions:
        out.update(_walk_ops(fn.body))
    return out


def _build_default_config(kernel_cls, spec):
    """Resolve a config: prefer ``CONFIG_CLS.default_for(spec)``,
    fill in a default Intel MMA shape if main_shape is empty."""
    config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
    if config_cls is None:
        return None
    try:
        config = config_cls.default_for(spec)
    except Exception:
        return None
    if hasattr(config, "main_shape") and getattr(config, "main_shape", "") == "":
        # Pick a shape that exists for the kernel's dtype combo. bf16/f32
        # for bf16 inputs; s8/s32 for s8 inputs.
        from quark.ir import DType
        a_dtype = getattr(spec, "a_dtype", DType.BF16)
        if isinstance(a_dtype, property):
            a_dtype = a_dtype.fget(spec)
        if a_dtype == DType.S8:
            shape = "m8n16k32_intel_s8_s32"
        else:
            shape = "m8n16k16_intel_bf16_f32"
        config = dataclasses.replace(config, main_shape=shape)
    return config


def _check_kernel(kernel_cls, spec, covered, caps):
    """Returns (status, n_ops, missing_or_err)."""
    config = _build_default_config(kernel_cls, spec)
    if config is None:
        return ("error", 0, "no CONFIG_CLS or default_for raised")
    try:
        kernel = kernel_cls(spec, config)
        module = kernel.emit()
        from quark.lower.legalize import legalize as _legalize
        _legalize(module, caps)
    except Exception as exc:
        return ("error", 0, f"{type(exc).__name__}: {str(exc)[:200]}")
    ops_used = _ops_in_module(module)
    missing = sorted(t.__name__ for t in ops_used - covered)
    if not missing:
        return ("full", len(ops_used), "")
    return ("partial", len(ops_used), f"missing: {', '.join(missing)}")


def _representative_specs():
    """Hand-picked production-ish specs for kernels that need
    direct coverage. Sized to match real Waypoint shapes so the
    emitted IR is representative."""
    # Lazy imports — these kernel modules pull in IR + lots of glue.
    from quark.ir import DType

    specs = []

    # ── Kernels missed by Phase 1's Mac autotune crash ──
    from quark.kernels.ada_gate_residual import AdaGateResidualKernel
    from quark.kernels.ada_gate_residual.spec import AdaGateResidualSpec
    specs.append((
        AdaGateResidualKernel,
        AdaGateResidualSpec(G=1, M=512, D=2048),
    ))

    from quark.kernels.unpatchify import UnpatchifyKernel
    from quark.kernels.unpatchify.spec import UnpatchifySpec
    specs.append((
        UnpatchifyKernel,
        UnpatchifySpec(B=1, C=32, H=32, W=64, d_model=2048),
    ))

    from quark.kernels.value_residual_packed import ValueResidualPackedKernel
    from quark.kernels.value_residual_packed.spec import ValueResidualPackedSpec
    specs.append((
        ValueResidualPackedKernel,
        ValueResidualPackedSpec(M=512, D_full=4096, v_col_offset=3072, v_width=1024),
    ))

    # ── int8 path (Phase 2C/D) ──
    from quark.kernels.gemm_int import GemmIntKernel, GemmIntSpec
    specs.append((
        GemmIntKernel,
        GemmIntSpec(M=512, N=2048, K=2048, out_dtype=DType.BF16),
    ))

    from quark.kernels.owl_attn_int8 import OwlAttnIntKernel, OwlAttnIntSpec
    specs.append((
        OwlAttnIntKernel,
        OwlAttnIntSpec(
            B=1, n_kv_heads=16, gqa_ratio=2,
            H_spatial=16, W_spatial=32,
            num_buckets=16, pinned_dilation=8,
            Dh=64,
        ),
    ))

    from quark.kernels.kv_cache_quantize import KVQuantizeKernel, KVQuantizeSpec
    specs.append((
        KVQuantizeKernel,
        KVQuantizeSpec(num_tokens=512, Dh=64),
    ))

    return specs


def main():
    os.environ.setdefault("QUARK_FORCE_ENGINE", "intel")
    os.environ.setdefault("QUARK_DISABLE_NAX", "1")

    from quark.lower.ocl.lower import _DISPATCH as OCL_DISPATCH
    covered = set(OCL_DISPATCH.keys())
    caps = _make_ocl_caps()
    print(f"→ OCL _DISPATCH covers {len(covered)} IR op types\n", flush=True)

    results: list[tuple[str, str, str, int, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kernel_cls, spec in _representative_specs():
            status, n_ops, msg = _check_kernel(kernel_cls, spec, covered, caps)
            spec_summary = f"{type(spec).__name__}({_short_repr(spec)})"
            line = f"  {kernel_cls.__name__}: {status}"
            if status == "full":
                line += f" ({n_ops} ops)"
            elif status in ("partial", "error"):
                line += f" — {msg}"
            print(line, flush=True)
            results.append((kernel_cls.__name__, spec_summary, status, n_ops, msg))

    # Append to docs/ocl_kernel_status.md.
    repo_root = Path(__file__).resolve().parent.parent
    status_path = repo_root / "docs" / "ocl_kernel_status.md"
    if status_path.exists():
        text = status_path.read_text()
        if "## Additional kernels (direct-construct check)" not in text:
            block = ["", "## Additional kernels (direct-construct check)", "",
                     "Kernels not exercised by the Phase 1 forward inventory —",
                     "checked by ``scripts/check_ocl_coverage_direct.py`` with",
                     "hand-picked production-shape specs.", "",
                     "| kernel | spec | OCL status | ops | notes |",
                     "|---|---|---|---|---|"]
            for kc, spec_str, status, n_ops, msg in results:
                ops_cell = str(n_ops) if status == "full" else ""
                block.append(
                    f"| `{kc}` | `{spec_str}` | `{status}` | {ops_cell} | "
                    f"{msg if status != 'full' else 'all visitors present'} |"
                )
            block.append("")
            status_path.write_text(text.rstrip() + "\n" + "\n".join(block))
            print(f"\n→ appended block to {status_path.relative_to(repo_root)}",
                  flush=True)
        else:
            print(f"\n(table already in {status_path.relative_to(repo_root)} — "
                  "not rewriting; rerun deletes the block manually if you want "
                  "fresh output)", flush=True)


def _short_repr(spec):
    """Short field list for the table cell."""
    fields = []
    for fname in ("M", "N", "K", "D", "Dh", "num_tokens", "n_kv_heads",
                  "H_spatial", "W_spatial", "out_dtype"):
        if hasattr(spec, fname):
            v = getattr(spec, fname)
            if not isinstance(v, (int, str, bool)) and not repr(v).startswith("DType."):
                continue
            fields.append(f"{fname}={v!r}")
        if len(fields) >= 4:
            break
    return ", ".join(fields)


if __name__ == "__main__":
    main()
