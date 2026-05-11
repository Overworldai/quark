"""Build configuration for the _metal_dispatch C++ extension.

Uses nanobind (matching MLX's binding stack) for low-overhead Python→C calls.
Only built on macOS (sys_platform == 'darwin').
"""

import os
import sys

from setuptools import setup

ext_modules = []

if sys.platform == "darwin":
    import nanobind
    from setuptools import Extension

    nb_inc = nanobind.include_dir()
    nb_src = os.path.join(os.path.dirname(nb_inc), "src")

    ext_modules.append(
        Extension(
            "quark.drivers._metal_dispatch",
            sources=[
                "src/quark/drivers/_metal_dispatch.cpp",
                # nanobind combined runtime
                os.path.join(nb_src, "nb_combined.cpp"),
            ],
            include_dirs=[
                "src/quark/drivers",
                nb_inc,
                # robin_map (small-set/map) used by nanobind internals
                os.path.join(os.path.dirname(nb_inc), "ext", "robin_map", "include"),
            ],
            extra_compile_args=[
                "-std=c++17",
                "-stdlib=libc++",
                "-xobjective-c++",
                "-fno-objc-arc",
                "-O2",
                "-fvisibility=hidden",
                "-DNB_COMPACT_ASSERTIONS",
                # Clang thread-safety analysis. Globals tagged with
                # ``TSA_GUARDED_BY`` (see _metal_dispatch.cpp) are then
                # checked at compile time — accessing one without
                # holding its mutex becomes a build error rather than
                # a SIGSEGV in production after a few hundred frames.
                # ``-Werror=thread-safety`` keeps the contract from
                # silently rotting.
                "-Wthread-safety",
                "-Werror=thread-safety",
            ],
            extra_link_args=[
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-framework",
                "CoreFoundation",
            ],
        )
    )

setup(ext_modules=ext_modules)
