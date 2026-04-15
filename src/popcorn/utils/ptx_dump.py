"""Helpers for dumping PTX when a compile fails.

Used by the autotune + bench entrypoints: when `launcher.compile(...)`
raises (commonly because `cuModuleLoadData` rejects the PTX as
malformed / unsupported on the target SM), we want to keep the PTX
text on disk so a human can diff it against a working run instead of
staring at a three-word CUDA_ERROR_INVALID_PTX message.

The dump is best-effort: if re-lowering the module raises on the way
in, we return None and the caller should print the original error.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def dump_path_for(kernel_cls: type, spec: Any, config: Any) -> Path:
    """Deterministic ``/tmp/popcorn_ptx_dumps/{kernel}_{hash}.ptx`` path.

    The hash derives from (spec, config) so the same bad config always
    lands at the same filename — re-running a failing bench doesn't
    pile up stale dumps.
    """
    base = Path(os.environ.get("POPCORN_PTX_DUMP_DIR", "/tmp/popcorn_ptx_dumps"))
    key = f"{kernel_cls.__qualname__}|{spec}|{config}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    name = getattr(kernel_cls, "NAME", kernel_cls.__name__.lower())
    return base / f"{name}_{digest}.ptx"


def write_ptx_for(kernel_cls: type, spec: Any, config: Any, device_caps: Any) -> Path | None:
    """Re-lower the kernel and write the PTX to ``dump_path_for``.

    Returns the path on success, None if re-lowering itself raised
    (caller should surface the original exception instead). Uses the
    same PtxLowerer settings the launcher uses so the dump reflects
    what ``cuModuleLoadData`` actually rejected.
    """
    try:
        from popcorn.lower.ptx import PtxLowerer

        kernel = kernel_cls(spec, config)
        module = kernel.emit()
        cc = getattr(device_caps, "compute_capability", None)
        target_sm = (cc[0] * 10 + cc[1]) if cc is not None else 89
        lowered = PtxLowerer(target_sm=target_sm).lower_module(module)
        path = dump_path_for(kernel_cls, spec, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(lowered.ptx)
        return path
    except Exception:
        return None


def classify_compile_error(exc: BaseException) -> str:
    """Tag a compile exception as 'ptx' (driver rejected PTX) or 'other'.

    Used by callers to decide whether a PTX dump is worth producing.
    PTX-related errors usually surface as ``CudaError`` with
    ``code in {CUDA_ERROR_INVALID_PTX=218, CUDA_ERROR_INVALID_IMAGE=200,
    CUDA_ERROR_NO_BINARY_FOR_GPU=209}`` or
    ``CUDA_ERROR_INVALID_VALUE=1`` from module-load arg validation.
    """
    ptx_codes = {1, 200, 209, 218}
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in ptx_codes:
        return "ptx"
    name = getattr(exc, "name", "") or ""
    if "PTX" in name or "INVALID_VALUE" in name or "INVALID_IMAGE" in name:
        return "ptx"
    return "other"
