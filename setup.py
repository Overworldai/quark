"""Build configuration for Quark's per-backend native dispatch extensions.

Uses nanobind (matching MLX's binding stack) for low-overhead Python→C calls.

* macOS (``sys_platform == 'darwin'``): builds ``_metal_dispatch`` —
  the metal-cpp + nanobind extension that drives Apple Metal directly
  (no MLX, no PyObjC). See ``src/quark/drivers/_metal_dispatch.cpp``.
* Linux (``sys_platform == 'linux'``): builds ``_spv_dispatch`` — the
  Vulkan + nanobind extension for the SPIR-V backend. See
  ``src/quark/drivers/_spv_dispatch.cpp`` and PORTABILITY_PLAN §3.1.
* Windows: no native ext today — the CUDA driver is pure ctypes
  (``drivers/cuda.py``).

Both extensions are optional at build time: missing system deps
(``libvulkan-dev`` on Linux, the macOS SDK on darwin) skip the
build with a warning rather than failing — letting CI on a host
without one toolchain still install the rest of the package.
"""

import os
import sys

from setuptools import setup

ext_modules = []


def _nanobind_paths():
    """Resolve nanobind's include + ``src/`` paths.

    Imported lazily so a build host without nanobind only fails when
    actually building the ext, not when ``setup.py`` is evaluated for
    ``sdist`` / metadata-only operations.
    """
    import nanobind  # noqa: PLC0415  (intentional lazy import)
    nb_inc = nanobind.include_dir()
    nb_src = os.path.join(os.path.dirname(nb_inc), "src")
    nb_robin = os.path.join(os.path.dirname(nb_inc), "ext", "robin_map", "include")
    return nb_inc, nb_src, nb_robin


if sys.platform == "darwin":
    from setuptools import Extension

    nb_inc, nb_src, nb_robin = _nanobind_paths()

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
                nb_robin,
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

if sys.platform == "linux":
    from setuptools import Extension

    nb_inc, nb_src, nb_robin = _nanobind_paths()

    # ``-lvulkan`` resolves to libvulkan.so via the system loader. The
    # build host needs ``libvulkan-dev`` (Ubuntu/Debian) /
    # ``vulkan-loader-devel`` (Fedora) for ``vulkan/vulkan.h``. At
    # runtime any vendor ICD (Mesa, Intel proprietary, NVIDIA) is
    # picked up — no SDK install required on the deploy box.
    ext_modules.append(
        Extension(
            "quark.drivers._spv_dispatch",
            sources=[
                "src/quark/drivers/_spv_dispatch.cpp",
                os.path.join(nb_src, "nb_combined.cpp"),
            ],
            include_dirs=[
                "src/quark/drivers",
                nb_inc,
                nb_robin,
            ],
            libraries=["vulkan"],
            extra_compile_args=[
                "-std=c++17",
                "-O2",
                "-fvisibility=hidden",
                "-DNB_COMPACT_ASSERTIONS",
                "-DVK_ENABLE_BETA_EXTENSIONS=1",
            ],
        )
    )

setup(ext_modules=ext_modules)
