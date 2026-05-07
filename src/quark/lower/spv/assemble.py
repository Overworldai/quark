"""Text → binary SPIR-V via the ``spirv-as`` CLI.

The lowerer emits SPIR-V text (``.spvasm``); the dispatch path
(``drivers/spv.py: SpvDriver.compile``) wants binary words. This
module bridges them.

External tool: ``spirv-as`` (Khronos SPIR-V Tools). Installed via
``apt install spirv-tools`` on Ubuntu / Debian, ``brew install
spirv-tools`` on macOS, or built from
https://github.com/KhronosGroup/SPIRV-Tools .

Direct binary emission (no external tool) is the eventual cleaner
shape; this shell-out is the prototype path that lets us iterate on
the visitor set without first writing a full SPIR-V binary builder.
See PORTABILITY_PLAN §3.2's "SPIR-V emission shape" decision.
"""

from __future__ import annotations

import shutil
import subprocess


class SpirvAsNotFound(RuntimeError):
    """Raised when the ``spirv-as`` CLI is missing."""


def text_to_binary(spirv_text: str, *, target_env: str = "vulkan1.3") -> bytes:
    """Run ``spirv-as`` on ``spirv_text``, return the binary words.

    ``target_env`` selects the validator profile — ``vulkan1.3``
    matches ``_spv_dispatch.cpp``'s ``apiVersion`` choice; mismatches
    surface as syntax errors when the assembler enforces the older
    profile's tighter rules.

    Errors:
      * ``SpirvAsNotFound`` — ``spirv-as`` not on PATH.
      * ``RuntimeError`` — assembler ran but rejected the input. The
        message includes ``spirv-as``'s stderr, which points at the
        offending line.
    """
    if shutil.which("spirv-as") is None:
        raise SpirvAsNotFound(
            "spirv-as not on PATH. Install via:\n"
            "    apt install spirv-tools         (Debian / Ubuntu)\n"
            "    brew install spirv-tools        (macOS)\n"
            "and retry."
        )
    proc = subprocess.run(
        ["spirv-as", "--target-env", target_env, "-o", "-", "-"],
        input=spirv_text.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "spirv-as failed:\n"
            f"  stderr: {proc.stderr.decode('utf-8', errors='replace')}\n"
            f"  source first 800 bytes:\n{spirv_text[:800]}"
        )
    return proc.stdout
