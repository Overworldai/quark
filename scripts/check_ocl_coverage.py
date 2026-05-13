"""Static OCL coverage check — for each kernel in the inventory,
emit its IR and cross-reference against ``quark.lower.ocl.lower
._DISPATCH``.

Drives Phase 1.2 of ``docs/OCL_E2E_PLAN.md``. Produces a
classification per kernel:

  - ``full``   : every IR op the kernel emits has an OCL visitor
  - ``partial``: some ops missing — listed
  - ``error``  : kernel.emit() raised — config / spec issue, not
                 a lowerer-coverage issue

Output goes into ``docs/ocl_kernel_status.md`` — the script reads
the existing table, fills in the ``OCL status`` and ``notes``
columns, and rewrites the file.

Usage::

    .venv/bin/python scripts/check_ocl_coverage.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path


# Same shape as ``scripts/dump_kernel_calls.py``. Production-size
# dims so per-kernel is_valid_for accepts production tile sizes.
_SMOKE_CFG = {
    "model_type": "test",
    "d_model": 2048,
    "n_layers": 2,
    "n_heads": 32,
    "n_kv_heads": 16,
    "fourier_dim": 512,
    "channels": 32,
    "patch": [2, 2],
    "height": 16,
    "width": 32,
    "local_window": 16,
    "global_window": 128,
    "global_pinned_dilation": 8,
    "global_attn_period": 1,
    "n_buttons": 256,
    "ctrl_conditioning": False,
    "scheduler_sigmas": [1.0, 0.9, 0.5, 0.0],
    "ae_uri": "dummy/dummy",
    "taehv_ae": True,
    "inference_fps": 8,
    "base_fps": 16,
    "temporal_compression": 4,
    "prompt_conditioning": None,
}


def _walk_ops(region) -> set[type]:
    """Return the set of Op types in this Region (recursing into
    nested regions on each op)."""
    types: set[type] = set()
    for op in region.ops:
        types.add(type(op))
        for r in op.regions:
            types.update(_walk_ops(r))
    return types


def _ops_in_module(module) -> set[type]:
    types: set[type] = set()
    for fn in module.functions:
        types.update(_walk_ops(fn.body))
    return types


def _patch_intel_main_shape(kernel_cls, spec, config):
    """Manufacture a useable config when autotune failed on Mac.

    Two cases:
      - ``config is None``: autotune raised. Try
        ``kernel_cls.config_cls.default_for(spec)`` and patch
        main_shape to the Intel bf16/f32 coopmat shape.
      - ``config.main_shape == ""``: autotune ran but couldn't pick
        an MMA shape. Patch in the Intel shape.

    Returns a possibly-new config or the original when no patch
    needed.
    """
    import dataclasses

    if config is None:
        config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
        if config_cls is None:
            return None  # let the caller report the error
        try:
            config = config_cls.default_for(spec)
        except Exception:
            return None
    if not hasattr(config, "main_shape"):
        return config
    if getattr(config, "main_shape", "") != "":
        return config
    return dataclasses.replace(config, main_shape="m8n16k16_intel_bf16_f32")


def _check_kernel(kernel_cls, spec, config, covered: set[type], caps) -> tuple[str, str]:
    """Returns ``(status, notes)``. ``status`` ∈ {full, partial, error}.

    Walks the IR after legalization — that's what the OCL lowerer
    actually sees in production (the launcher legalizes before
    handing off to ``lower_module``). Async-copy ops on
    ``supports_async_copy=False`` backends, for example, get
    rewritten to VecLoad/VecStore by the legalizer and never reach
    a visitor.
    """
    config = _patch_intel_main_shape(kernel_cls, spec, config)
    try:
        kernel = kernel_cls(spec, config)
        module = kernel.emit()
        from quark.lower.legalize import legalize as _legalize
        _legalize(module, caps)
    except Exception as exc:
        return ("error", f"{type(exc).__name__}: {exc}".replace("|", "/")[:120])
    ops_used = _ops_in_module(module)
    missing = sorted((t.__name__ for t in ops_used - covered))
    if not missing:
        return ("full", f"{len(ops_used)} op types, all visitors present (post-legalize)")
    return ("partial", f"missing visitors: {', '.join(missing)}")


def _kernel_cls_lookup() -> dict[str, type]:
    """Discover every kernel class registered via ``@register_kernel``."""
    from quark.kernels import get as _kernel_get
    from quark.kernels import _REGISTRY  # type: ignore[attr-defined]

    # Force the kernel package to fully load so every kernel is
    # registered. Importing quark.kernels.* triggers the
    # @register_kernel side-effects.
    import quark.kernels  # noqa: F401

    return {cls.__name__: cls for cls in _REGISTRY.values()}


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    status_path = repo_root / "docs" / "ocl_kernel_status.md"

    if not status_path.exists():
        raise SystemExit(
            f"{status_path.relative_to(repo_root)} missing — run "
            "scripts/dump_kernel_calls.py first."
        )

    # Engine env + stubs (mirrors ``dump_kernel_calls.py``).
    os.environ.setdefault("QUARK_FORCE_ENGINE", "intel")
    os.environ.setdefault("QUARK_SKIP_VAE", "1")
    os.environ.setdefault("QUARK_DISABLE_NAX", "1")

    # Patch at the source module so it works regardless of how the
    # engine imports (module-level on Mac, lazy-inside-__init__ on
    # devkit's older intel.py).
    import quark.engine.intel as _intel_mod
    import quark.models.config as _cfg_mod
    _stub_load = lambda path: dict(_SMOKE_CFG)
    _stub_resolve = lambda uri, **kw: uri
    _cfg_mod.load_yaml_config = _stub_load
    _cfg_mod._resolve_path = _stub_resolve
    if hasattr(_intel_mod, "load_yaml_config"):
        _intel_mod.load_yaml_config = _stub_load
    if hasattr(_intel_mod, "_resolve_path"):
        _intel_mod._resolve_path = _stub_resolve

    # Load _DISPATCH from the OCL lowerer.
    print("→ loading OCL _DISPATCH ...", flush=True)
    from quark.lower.ocl.lower import _DISPATCH as OCL_DISPATCH
    covered = set(OCL_DISPATCH.keys())
    print(f"  OCL covers {len(covered)} IR op types", flush=True)

    # Synthetic OCL caps — duck-typed; the legalization layer only
    # reads a handful of attrs (``supports_async_copy``,
    # ``has_fma_bf16x2``, ``atomic_add_vector``,
    # ``has_native_subgroup_reduce``, ``subgroup_width``). Real caps
    # come from ``quark.drivers.ocl.caps_from_probe`` on devkit.
    from types import SimpleNamespace

    from quark.device import DeviceFamily

    ocl_caps = SimpleNamespace(
        family=DeviceFamily.INTEL_GPU,
        subgroup_width=32,
        supports_async_copy=False,
        has_fma_bf16x2=False,
        has_native_subgroup_reduce=True,
        atomic_add_vector=frozenset(),
        matmul_shapes=frozenset(),
    )

    # Discover kernels by re-running the inventory dump (same
    # script, in-process). We need the actual (cls, spec, config)
    # tuples to instantiate — parsing them out of the markdown
    # would lose typed data.
    print("→ replaying inventory dump to collect (cls, spec, config) ...",
          flush=True)
    from quark.autotune import AutotuneCache
    from quark.launcher.launcher import Launcher

    real_compile = Launcher.compile
    real_lookup = AutotuneCache.lookup_or_search
    records: list[tuple[type, object, object]] = []

    def recording_compile(self, kernel_cls, spec, config=None):
        records.append((kernel_cls, spec, config))
        return real_compile(self, kernel_cls, spec, config)

    def recording_lookup(self, kernel_cls, spec):
        try:
            cfg = real_lookup(self, kernel_cls, spec)
            records.append((kernel_cls, spec, cfg))
            return cfg
        except Exception:
            records.append((kernel_cls, spec, None))
            raise

    Launcher.compile = recording_compile  # type: ignore[method-assign]
    AutotuneCache.lookup_or_search = recording_lookup  # type: ignore[method-assign]

    import warnings
    from quark.engine import Engine
    from quark.nn.module import _tensor as _qk_tensor
    from quark.runtime.tensor import QuarkTensor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine = Engine("dummy", load_weights=False)
    latent = QuarkTensor.zeros(*engine._flat_shape, dtype="bf16")
    ft = _qk_tensor([0], dtype="s32")
    try:
        engine.model(latent, sigma_idx=0, frame_t=ft, frozen=True)
    except Exception as exc:
        print(f"  forward stopped at: {type(exc).__name__}", flush=True)

    # Dedup on (cls.__name__, repr(spec)). For each unique pair,
    # prefer the record that has a non-None config.
    by_key: dict[tuple[str, str], tuple[type, object, object]] = {}
    for kc, spec, cfg in records:
        key = (kc.__name__, repr(spec))
        prev = by_key.get(key)
        if prev is None or (prev[2] is None and cfg is not None):
            by_key[key] = (kc, spec, cfg)
    unique = sorted(by_key.values(), key=lambda r: (r[0].__name__, repr(r[1])))
    print(f"  {len(unique)} unique kernels to coverage-check", flush=True)

    # Coverage check per kernel.
    print("→ checking OCL coverage ...", flush=True)
    coverage: dict[tuple[str, str], tuple[str, str]] = {}
    for kc, spec, cfg in unique:
        status, notes = _check_kernel(kc, spec, cfg, covered, ocl_caps)
        coverage[(kc.__name__, repr(spec))] = (status, notes)
        print(f"  {kc.__name__}: {status} — {notes}", flush=True)

    # Rewrite the table in docs/ocl_kernel_status.md. The first
    # column matches the unique-tuple key, the new columns fill in
    # status + notes.
    text = status_path.read_text()
    out_lines: list[str] = []
    in_table = False
    table_idx = 0
    for line in text.splitlines():
        if line.startswith("|---|"):
            in_table = True
            table_idx = 0
            out_lines.append(line)
            continue
        if in_table and line.startswith("|") and not line.startswith("| kernel "):
            # Data row. The kernel name + spec_summary are escaped
            # backticks; we re-derive the (name, repr(spec)) key
            # from the captured `unique` order.
            if table_idx < len(unique):
                kc, spec, cfg = unique[table_idx]
                key = (kc.__name__, repr(spec))
                status, notes = coverage.get(key, ("", ""))
                # Re-emit the row with status + notes filled in.
                # Use the same cell content the dumper produced for
                # the first three columns by parsing it out.
                cells = [c.strip() for c in line.strip("|").split("|")]
                while len(cells) < 5:
                    cells.append("")
                cells[3] = f"`{status}`" if status else ""
                cells[4] = notes
                out_lines.append("| " + " | ".join(cells) + " |")
                table_idx += 1
                continue
        if in_table and not line.startswith("|"):
            in_table = False
        out_lines.append(line)

    status_path.write_text("\n".join(out_lines) + "\n")
    print(f"→ updated {status_path.relative_to(repo_root)}", flush=True)


if __name__ == "__main__":
    main()
