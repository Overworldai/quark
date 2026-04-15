"""Kernel package — each subfolder is a self-contained kernel.

Auto-discovery imports every subdirectory under `kernels/` that has
an `__init__.py`. Module-level `@register("name")` side-effects
populate the registry so `all_kernels()` / `get(name)` find them.

Adding a new kernel touches one place: a new subfolder with spec,
config, kernel, and __init__.py.
"""

from popcorn.kernels.base import (
    Autotuner,
    AutotuneResult,
    Baseline,
    Kernel,
    KernelConfig,
    KernelSpec,
    Problem,
)
from popcorn.kernels.decorator import kernel
from popcorn.kernels.registry import (
    all_kernels,
    get,
    names,
    register,
    unregister,
)


def _autodiscover_folder_kernels() -> None:
    import importlib
    from pathlib import Path

    pkg_dir = Path(__file__).parent
    for entry in sorted(pkg_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue
        if not (entry / "__init__.py").exists():
            continue
        importlib.import_module(f"popcorn.kernels.{entry.name}")


_autodiscover_folder_kernels()

__all__ = [
    "AutotuneResult",
    "Autotuner",
    "Baseline",
    "Kernel",
    "KernelConfig",
    "KernelSpec",
    "Problem",
    "all_kernels",
    "get",
    "kernel",
    "names",
    "register",
    "unregister",
]
