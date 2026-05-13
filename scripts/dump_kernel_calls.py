"""Inventory every kernel Waypoint forward compiles.

Drives Phase 1.1 of `docs/OCL_E2E_PLAN.md`. Monkey-patches
``Launcher.compile`` to record every ``(kernel_cls, spec, config)``
the model emits during a single synthetic forward, then writes the
unique tuples to ``docs/ocl_kernel_status.md`` as a markdown table
with columns: kernel | spec_summary | config_summary | OCL status |
notes.

Backend-independent: the kernel CLASS set the model emits is the
same on Metal / OCL / CUDA — only the configs and lowered code
differ. We run on whichever backend is local (Mac → Metal; this is
fine, the inventory is for kernel-set coverage, not perf).

Usage::

    .venv/bin/python scripts/dump_kernel_calls.py

Writes to ``docs/ocl_kernel_status.md`` (overwrites). Run from
the quark repo root so the relative paths resolve.

The 'OCL status' and 'notes' columns are blank — Phase 1.2 fills
them in via static analysis of `quark/lower/ocl/lower.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Real Waypoint-1.5-1B 360p shapes (d_model / heads / patch / hwc)
# so per-kernel ``is_valid_for`` accepts production tile sizes.
# Drops ``n_layers`` to 2 (each block emits the same kernel set, so
# the inventory doesn't need 24 layers) and ``n_denoise`` is read
# from a short ``scheduler_sigmas`` — keeps the synthetic forward
# fast.
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
    "global_attn_period": 1,  # exercise both local + global attn in fewer layers
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


def _spec_summary(spec) -> str:
    """One-line summary of a KernelSpec — class name + a few fields."""
    if spec is None:
        return ""
    name = type(spec).__name__
    fields = []
    for fname in (
        "M", "N", "K", "Dh", "S", "D", "Bm", "Bn", "Bk",
        "n_heads", "n_kv_heads", "head_dim", "seq_len",
        "compute_dtype", "in_dtype", "out_dtype",
    ):
        if hasattr(spec, fname):
            v = getattr(spec, fname)
            if v is None:
                continue
            fields.append(f"{fname}={v!r}")
        if len(fields) >= 4:
            break
    return f"{name}({', '.join(fields)})" if fields else name


def _config_summary(config) -> str:
    """One-line summary of a KernelConfig — just class name + a couple fields."""
    if config is None:
        return "(autotune)"
    name = type(config).__name__
    fields = []
    for fname in ("BM", "BN", "BK", "n_warps", "n_stages", "subgroup_size"):
        if hasattr(config, fname):
            v = getattr(config, fname)
            if v is None:
                continue
            fields.append(f"{fname}={v!r}")
        if len(fields) >= 3:
            break
    return f"{name}({', '.join(fields)})" if fields else name


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "docs" / "ocl_kernel_status.md"

    # Engine env — pull through EngineIntel construction so the
    # monkeypatch sees the same call shape OCL will hit in production.
    os.environ.setdefault("QUARK_FORCE_ENGINE", "intel")
    os.environ.setdefault("QUARK_SKIP_VAE", "1")
    # Force Metal-specific fast paths off so kernels that production
    # OCL would dispatch via Launcher.compile actually compile here:
    #   - NAX (Apple-only fused-attn dispatch in owl_attn.py)
    # Without this, OwlAttn (the hottest kernel) is missing from the
    # inventory on Mac.
    os.environ.setdefault("QUARK_DISABLE_NAX", "1")

    # Patch BOTH compile (catches the kernels that reach the
    # launcher) and AutotuneCache.lookup_or_search (catches the ones
    # that fail at autotune before reaching it — common on Mac
    # without NAX for OwlAttn / int8 paths).
    from quark.autotune import AutotuneCache
    from quark.launcher.launcher import Launcher

    real_compile = Launcher.compile
    real_lookup = AutotuneCache.lookup_or_search

    records: list[tuple[str, object, object]] = []
    dispatch_failures: list[tuple[str, str]] = []

    def recording_compile(self, kernel_cls, spec, config=None):
        records.append((kernel_cls.__name__, spec, config))
        try:
            return real_compile(self, kernel_cls, spec, config)
        except Exception as exc:
            dispatch_failures.append(
                (kernel_cls.__name__, f"compile: {type(exc).__name__}: {exc}"))
            raise

    def recording_lookup(self, kernel_cls, spec):
        records.append((kernel_cls.__name__, spec, None))
        try:
            return real_lookup(self, kernel_cls, spec)
        except Exception as exc:
            dispatch_failures.append(
                (kernel_cls.__name__, f"autotune: {type(exc).__name__}: {exc}"))
            raise

    Launcher.compile = recording_compile  # type: ignore[method-assign]
    AutotuneCache.lookup_or_search = recording_lookup  # type: ignore[method-assign]

    # Stub config load + path resolve at the source module so the
    # patches work regardless of how the engine imports them (module
    # level on Mac's new intel.py; lazy-inside-__init__ on the
    # devkit's SPV-stub intel.py).
    import quark.engine.intel as _intel_mod
    import quark.models.config as _cfg_mod

    _stub_load = lambda path: dict(_SMOKE_CFG)
    _stub_resolve = lambda uri, **kw: uri
    _cfg_mod.load_yaml_config = _stub_load
    _cfg_mod._resolve_path = _stub_resolve
    # Patch the engine-module bindings too (Mac's new intel.py
    # imports them at module load and would otherwise miss).
    if hasattr(_intel_mod, "load_yaml_config"):
        _intel_mod.load_yaml_config = _stub_load
    if hasattr(_intel_mod, "_resolve_path"):
        _intel_mod._resolve_path = _stub_resolve

    # Build engine.
    print("→ constructing EngineIntel ...", flush=True)
    import warnings

    from quark.engine import Engine

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine = Engine("dummy", load_weights=False)
    print(f"  done. n_denoise={engine._n_denoise}", flush=True)

    # Synthetic latent + frame_t. Trigger one forward — same call
    # shape ``_compute_latent`` uses, but skipping the lazy/euler
    # loop to keep this script narrow.
    print("→ running one synthetic forward ...", flush=True)
    from quark.nn.module import _tensor as _qk_tensor
    from quark.runtime.tensor import QuarkTensor

    n_flat = 1
    for d in engine._flat_shape:
        n_flat *= d
    latent = QuarkTensor.zeros(*engine._flat_shape, dtype="bf16")
    ft = _qk_tensor([0], dtype="s32")

    try:
        engine.model(latent, sigma_idx=0, frame_t=ft, frozen=True)
    except Exception as exc:
        # The forward may fail mid-way on Mac because some kernels
        # have backend-specific compile paths that need real
        # weights or hit issues with the smoke shape. We still
        # care about the kernels that *did* compile before the
        # crash — keep going.
        print(
            f"  warning: forward raised "
            f"{type(exc).__name__}: {exc}\n  (recorded "
            f"{len(records)} compile calls before the crash)",
            flush=True,
        )

    # Dedup. Keyed on (kernel_cls_name, repr(spec)) — drop config
    # because autotune may pick different shapes per (cls, spec).
    seen: dict[tuple[str, str], tuple[str, object, object]] = {}
    for kc, spec, cfg in records:
        key = (kc, repr(spec))
        if key not in seen:
            seen[key] = (kc, spec, cfg)
    unique = sorted(seen.values(), key=lambda r: (r[0], _spec_summary(r[1])))

    print(f"→ {len(records)} compile calls, {len(unique)} unique (cls, spec)",
          flush=True)

    # Emit the markdown.
    lines = [
        "# OCL kernel coverage status",
        "",
        "Generated by ``scripts/dump_kernel_calls.py``. Inventory of every",
        "kernel ``Launcher.compile`` was asked to build during one synthetic",
        "Waypoint forward. Phase 1.2 of ``docs/OCL_E2E_PLAN.md`` fills in the",
        "``OCL status`` and ``notes`` columns via static analysis of",
        "``quark/lower/ocl/lower.py``.",
        "",
        "**Inventory caveats on Mac**: the script forces ``QUARK_DISABLE_NAX=1``",
        "so kernels normally NAX-dispatched (OwlAttn) route through",
        "``Launcher.compile`` and get recorded. Autotune may then fail to find",
        "a valid config for those kernels on Mac (no MMA shape that fits ",
        "Apple-without-NAX); the kernel is still listed below but the forward",
        "crashes there. A devkit run completes the inventory; cross-check",
        "with ``tests/lower/ocl/test_lower.py`` for the OCL-side coverage.",
        "",
        f"- Compile records captured: {len(records)}",
        f"- Unique (kernel, spec) tuples: {len(unique)}",
        f"- Dispatch failures on Mac: {len(dispatch_failures)}",
    ]
    if dispatch_failures:
        lines.extend([
            "",
            "### Dispatch-failed on this run (autotune / lower / launch)",
            "",
        ])
        for kc, msg in dispatch_failures:
            short = msg.split(". ", 1)[0]
            lines.append(f"- `{kc}` — {short}")
    lines.extend([
        "",
        "## Kernel set",
        "",
        "| kernel | spec_summary | config_summary | OCL status | notes |",
        "|---|---|---|---|---|",
    ])
    for kc, spec, cfg in unique:
        lines.append(
            f"| `{kc}` | `{_spec_summary(spec)}` | `{_config_summary(cfg)}` | | |"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"→ wrote {out_path.relative_to(repo_root)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
