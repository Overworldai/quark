"""popcorn.launcher — top-level entry point for kernel compilation
and launch.

Per popcorn launcher proposal §6. Bundle 3 ships the CUDA path; other
backends arrive in later bundles.
"""

from popcorn.launcher.launcher import (
    CompiledKernel,
    Launcher,
    _check_contiguous,
    _check_dtype,
    _driver_for,
)
from popcorn.launcher.param_spec import (
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
    "_check_dtype",
    "_driver_for",
]
