"""Reference-oracle cache for the autotune / fuzz correctness gate.

Every kernel declares a ``reference_numpy(spec, **inputs)`` method
and a ``make_tensors_numpy(problem, seed)`` method — both pure
numpy. The cache sits between those methods and the tools that
consume them so we only pay the reference cost ONCE per (kernel,
spec, seed) across a whole session:

    from popcorn.refs import ref_cache

    inputs_np, outputs_np = ref_cache().get(kernel_cls, problem)

Two storage tiers:

  * **In-memory** — ``dict[key, (inputs, outputs)]`` on the cache
    singleton, cleared when the process exits.
  * **On-disk** — ``~/.cache/popcorn/refs/<hash>.npz`` so subsequent
    sessions skip the reference compute entirely. First cold visit
    pays numpy matmul (seconds on big problems); every visit after
    is an mmap + shape decode.

Invalidation: none automatic. If you change the numpy reference for
a kernel, clear the cache manually (``rm -r ~/.cache/popcorn/refs``
or ``python -m popcorn.refs --clear``). Adding per-source-file hash
invalidation isn't worth the complexity — early-alpha users expect
to clear when they poke at the oracle.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_SEED = 0x5A1E_5EED
_DEFAULT_CACHE_SIZE_MB = 500


def _default_cache_dir() -> Path:
    """``~/.cache/popcorn/refs`` (or ``$POPCORN_CACHE_DIR/refs`` when
    set). Created on first write."""
    root = os.environ.get("POPCORN_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "popcorn"
    return base / "refs"


def _hash_key(kernel_name: str, problem_params: dict, seed: int) -> str:
    """Stable short key for disk cache filenames.

    ``problem_params`` is serialized as sorted JSON so equivalent
    dicts hash identically regardless of insertion order. DType enums
    and tuples survive the round-trip because we coerce to str /
    list at serialize time."""
    canon = json.dumps(_canonicalize(problem_params), sort_keys=True, separators=(",", ":"))
    payload = f"{kernel_name}|{canon}|{seed}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _canonicalize(obj: Any) -> Any:
    """Coerce dataclasses / enums / tuples to JSON-friendly primitives."""
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, (bool, int, float)):
        # Looks like an enum — use the string value so BF16 == "bf16".
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


class RefCache:
    """In-memory + on-disk reference cache.

    Thread-safe under a coarse lock — reference compute is not the
    contended path and the cache is hit once per spec, so a single
    lock around get() is fine.
    """

    def __init__(self, cache_dir: Path | None = None, size_cap_mb: int = _DEFAULT_CACHE_SIZE_MB):
        self._dir = cache_dir if cache_dir is not None else _default_cache_dir()
        self._mem: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self._size_cap_bytes = size_cap_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        kernel_cls,
        problem_params: dict,
        *,
        seed: int = _DEFAULT_SEED,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return ``(inputs_np, outputs_np)`` for ``(kernel_cls, problem_params)``.

        ``inputs_np`` is the full ``TensorDecl`` manifest as a
        ``{name: np.ndarray}`` dict — every kernel input plus
        zero-initialized stubs for role="out" entries (same shape the
        device buffer will have). ``outputs_np`` is the subset of
        ``TensorDecl`` entries with role="out", populated from the
        kernel's ``reference_numpy`` call.

        Fresh arrays on every call — cached values are deep-copied out
        so callers can mutate freely without poisoning the cache.
        """
        key = _hash_key(kernel_cls.NAME, problem_params, seed)
        with self._lock:
            if key in self._mem:
                return _deep_copy_pair(self._mem[key])

            disk = self._load_disk(key)
            if disk is not None:
                self._mem[key] = disk
                return _deep_copy_pair(disk)

            inputs_np, outputs_np = self._compute(kernel_cls, problem_params, seed)
            self._mem[key] = (inputs_np, outputs_np)
            self._save_disk(key, inputs_np, outputs_np)
            return _deep_copy_pair((inputs_np, outputs_np))

    def clear_memory(self) -> None:
        with self._lock:
            self._mem.clear()

    def clear_disk(self) -> None:
        """Remove every cached .npz under the cache dir. Does not
        touch in-memory entries."""
        if not self._dir.is_dir():
            return
        for f in self._dir.glob("*.npz"):
            with contextlib.suppress(OSError):
                f.unlink()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute(
        self, kernel_cls, problem_params: dict, seed: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Invoke ``make_tensors_numpy`` + ``reference_numpy``.

        Both methods are classmethods on the kernel. ``reference_numpy``
        may return a single ``np.ndarray`` (for single-output kernels);
        we wrap it in a dict keyed on the kernel's sole role="out"
        TensorDecl name for a uniform shape downstream.
        """
        make_tensors = getattr(kernel_cls, "make_tensors_numpy", None)
        reference = getattr(kernel_cls, "reference_numpy", None)
        if not (callable(make_tensors) and callable(reference)):
            raise NotImplementedError(
                f"RefCache: {kernel_cls.__name__} needs make_tensors_numpy + "
                f"reference_numpy — numpy-refs migration."
            )

        spec = kernel_cls.from_problem(problem_params).spec
        inputs_np = make_tensors(problem_params, seed=seed)
        out_dict = reference(spec, **dict(inputs_np))
        if isinstance(out_dict, np.ndarray):
            out_names = _output_names(kernel_cls)
            if len(out_names) != 1:
                raise TypeError(
                    f"RefCache: {kernel_cls.__name__}.reference_numpy returned a "
                    f"bare ndarray but the kernel has {len(out_names)} role='out' "
                    f"tensors; return a dict keyed by name."
                )
            out_dict = {out_names[0]: out_dict}
        elif not isinstance(out_dict, dict):
            raise TypeError(
                f"RefCache: {kernel_cls.__name__}.reference_numpy returned "
                f"{type(out_dict).__name__}; expected np.ndarray or dict[str, np.ndarray]"
            )
        return inputs_np, out_dict

    def _disk_path(self, key: str) -> Path:
        return self._dir / f"{key}.npz"

    def _load_disk(self, key: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
        path = self._disk_path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as npz:
                # .npz keys are prefixed "in__" / "out__" to disambiguate
                # the two halves in a flat namespace.
                inputs = {
                    k.removeprefix("in__"): npz[k].copy() for k in npz.files if k.startswith("in__")
                }
                outputs = {
                    k.removeprefix("out__"): npz[k].copy()
                    for k in npz.files
                    if k.startswith("out__")
                }
            return inputs, outputs
        except (OSError, ValueError):
            # Corrupt / truncated file — treat as miss.
            with contextlib.suppress(OSError):
                path.unlink()
            return None

    def _save_disk(
        self,
        key: str,
        inputs: dict[str, np.ndarray],
        outputs: dict[str, np.ndarray],
    ) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        for k, v in inputs.items():
            payload[f"in__{k}"] = np.ascontiguousarray(v)
        for k, v in outputs.items():
            payload[f"out__{k}"] = np.ascontiguousarray(v)
        # np.savez appends ``.npz`` if the path doesn't already end in
        # it, which bites the usual ``foo.tmp → foo`` atomic-replace
        # pattern. Stage under a ``.npz`` path that also carries
        # ``.tmp`` so savez accepts it verbatim, then atomic-replace.
        # The inline ignore silences a stub-artifact: numpy types
        # ``allow_pickle`` as a positional bool alongside
        # ``**kwds: ArrayLike``; ty can't tell our keys won't collide
        # with ``allow_pickle`` and flags every dict-kwargs expansion.
        path = self._disk_path(key)
        tmp = path.parent / (path.name + ".tmp.npz")
        np.savez(tmp, **payload)  # ty: ignore[invalid-argument-type]
        os.replace(tmp, path)
        self._evict_to_cap()

    def _evict_to_cap(self) -> None:
        """Trim the cache dir to ``self._size_cap_bytes`` via LRU-by-mtime.
        Called after every successful save."""
        if not self._dir.is_dir():
            return
        files = [(f, f.stat()) for f in self._dir.glob("*.npz")]
        total = sum(st.st_size for _, st in files)
        if total <= self._size_cap_bytes:
            return
        # Oldest first.
        files.sort(key=lambda x: x[1].st_mtime)
        for f, st in files:
            if total <= self._size_cap_bytes:
                break
            try:
                f.unlink()
                total -= st.st_size
            except OSError:
                pass


def _output_names(kernel_cls) -> list[str]:
    """TENSORS entries with role='out', in declaration order."""
    return [t.name for t in kernel_cls.TENSORS if getattr(t, "role", None) == "out"]


def _deep_copy_pair(
    pair: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    a, b = pair
    return {k: v.copy() for k, v in a.items()}, {k: v.copy() for k, v in b.items()}


# ---------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------

_SINGLETON: RefCache | None = None
_SINGLETON_LOCK = threading.Lock()


def ref_cache() -> RefCache:
    """Return the process-wide ``RefCache`` singleton."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = RefCache()
    return _SINGLETON


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="popcorn reference-oracle cache")
    parser.add_argument("--clear", action="store_true", help="delete every cached .npz")
    parser.add_argument("--list", action="store_true", help="list cached files with sizes")
    args = parser.parse_args()

    cache = ref_cache()
    if args.clear:
        cache.clear_disk()
        print(f"cleared {cache._dir}")
        return 0
    if args.list or not (args.clear):
        if not cache._dir.is_dir():
            print(f"(no cache at {cache._dir})")
            return 0
        total = 0
        for f in sorted(cache._dir.glob("*.npz")):
            size = f.stat().st_size
            total += size
            print(f"  {f.name}  {size / 1024:.1f} KB")
        print(f"total: {total / 1024 / 1024:.1f} MB in {cache._dir}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
