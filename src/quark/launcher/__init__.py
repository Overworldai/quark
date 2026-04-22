"""quark.launcher — top-level entry point for kernel compilation
and launch.

Per quark launcher proposal §6. Bundle 3 ships the CUDA path; other
backends arrive in later bundles.
"""

from quark.launcher.launcher import (
    CompiledKernel,
    Launcher,
    _check_contiguous,
    _check_dtype_quark,
    _driver_for,
)
from quark.launcher.param_spec import (
    BufferSpec,
    ParamSpec,
    ProgramFootprint,
    ScalarSpec,
)

__all__ = [
    "BufferSpec",
    "CompiledKernel",
    "Launcher",
    "ParamSpec",
    "ProgramFootprint",
    "ScalarSpec",
    "_check_contiguous",
    "_check_dtype_quark",
    "_driver_for",
]
