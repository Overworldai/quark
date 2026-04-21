"""Disk I/O + JSON serialization for ``AutotuneCache``.

Split from ``autotune.py`` so the search-engine file stays focused on
genetic search. All functions here are pure plumbing: spec fingerprint,
config ↔ JSON mapping, SHA256 source hashing, and atomic write/read of
cached-config records.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

_KILL_SWITCH_ENV = "POPCORN_DISABLE_AUTOTUNE"
_CACHE_DIR_ENV = "POPCORN_CACHE_DIR"


def default_cache_dir() -> Path:
    env = os.environ.get(_CACHE_DIR_ENV)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "popcorn"
    return Path.home() / ".cache" / "popcorn"


def default_bundled_dir() -> Path:
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        candidate = ancestor / "configs"
        if candidate.is_dir():
            return candidate
    return here.parent.parent / "configs"


def autotune_disabled() -> bool:
    val = os.environ.get(_KILL_SWITCH_ENV, "")
    return val.lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Spec fingerprinting / (de)serialization
# ---------------------------------------------------------------------------


def spec_fingerprint(spec: Any) -> tuple:
    if dataclasses.is_dataclass(spec):
        return tuple((f.name, getattr(spec, f.name)) for f in dataclasses.fields(spec))
    return ((spec.__class__.__name__, repr(spec)),)


def spec_to_json_dict(spec: Any) -> dict:
    if dataclasses.is_dataclass(spec):
        return dataclasses.asdict(spec)
    return {"_repr": repr(spec)}


def config_to_json_dict(config: Any) -> dict:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return {"_repr": repr(config)}


def config_from_json_dict(config_cls: type, payload: dict) -> Any:
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError(f"config_from_json_dict: {config_cls.__name__} is not a dataclass")
    field_names = {f.name for f in dataclasses.fields(config_cls)}
    kwargs = {k: v for k, v in payload.items() if k in field_names}
    return config_cls(**kwargs)


def format_spec_label(spec: Any) -> str:
    if not dataclasses.is_dataclass(spec):
        return hashlib.sha256(repr(spec).encode("utf-8")).hexdigest()[:16]
    fields = sorted(dataclasses.fields(spec), key=lambda f: f.name)
    return "_".join(f"{f.name}{getattr(spec, f.name)}" for f in fields)


_SOURCE_HASH_CACHE: dict[type, str] = {}


def source_hash(kernel_cls: type) -> str:
    # Process-lifetime memo: source files don't change mid-run and
    # ``inspect.getsource`` + SHA256 on a multi-KB kernel class shows up
    # as ~1ms per call on the hot path. Kernel classes are identity-
    # stable so keying by the class object is safe.
    cached = _SOURCE_HASH_CACHE.get(kernel_cls)
    if cached is not None:
        return cached
    try:
        src = inspect.getsource(kernel_cls)
    except (OSError, TypeError):
        src = kernel_cls.__qualname__
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    _SOURCE_HASH_CACHE[kernel_cls] = digest
    return digest


def resolve_config_cls(kernel_qualname: Optional[str]) -> Optional[type]:
    if not kernel_qualname:
        return None
    try:
        from popcorn.kernels.registry import all_kernels
    except ImportError:
        return None
    short = kernel_qualname.rsplit(".", 1)[-1]
    for cls in all_kernels():
        if cls.__qualname__ == kernel_qualname or cls.__name__ == short:
            return getattr(cls, "CONFIG_CLS", None)
    return None


# ---------------------------------------------------------------------------
# Key construction + filename
# ---------------------------------------------------------------------------


def make_key(kernel_cls, spec, device_fingerprint: str) -> tuple:
    return (
        kernel_cls.__qualname__,
        spec_fingerprint(spec),
        device_fingerprint,
        source_hash(kernel_cls),
    )


def key_to_filename(key: tuple) -> str:
    kernel_name = key[0]
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:24]
    return f"{kernel_name}__{digest}.json"


# ---------------------------------------------------------------------------
# Disk read / write
# ---------------------------------------------------------------------------


def load_from_disk(cache_dir: Path, key: tuple, device_fingerprint: str) -> Optional[Any]:
    path = cache_dir / key_to_filename(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("source_hash") != key[3]:
        return None
    if payload.get("device") != device_fingerprint:
        return None
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        return None
    config_cls = resolve_config_cls(payload.get("kernel"))
    if config_cls is None:
        return None
    try:
        return config_from_json_dict(config_cls, config_payload)
    except TypeError:
        return None


def save_to_disk(
    cache_dir: Path,
    key: tuple,
    kernel_cls,
    spec,
    config,
    device_fingerprint: str,
    *,
    runtime_us: Optional[float],
) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    record = {
        "kernel": kernel_cls.__qualname__,
        "spec": spec_to_json_dict(spec),
        "device": device_fingerprint,
        "source_hash": key[3],
        "config": config_to_json_dict(config),
        "runtime_us": runtime_us,
        "timestamp": int(time.time()),
    }
    path = cache_dir / key_to_filename(key)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(cache_dir),
            prefix=".popcorn_cache_",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(record, tmp)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    except OSError:
        return


def load_bundled_default(bundled_dir: Path, kernel_cls, spec) -> Optional[Any]:
    if not bundled_dir.exists():
        return None
    kernel_name = getattr(kernel_cls, "NAME", None) or kernel_cls.__name__.lower()
    spec_label = format_spec_label(spec)
    candidates = [
        bundled_dir / f"{kernel_name}_{spec_label}.json",
        bundled_dir / f"{kernel_cls.__qualname__}_{spec_label}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            continue
        config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
        if config_cls is None:
            continue
        try:
            return config_from_json_dict(config_cls, config_payload)
        except TypeError:
            continue
    return None


def load_warm_seeds(
    cache_dir: Path, bundled_dir: Path, kernel_cls, spec, device_fingerprint: str, caps
) -> list:
    """Return previously-saved configs that are still valid for ``spec``.

    Scans both the user cache and the bundled dir regardless of spec —
    any config that survives ``is_valid_for(caps)`` becomes a seed for
    the next search's initial population.
    """
    seeds: list = []
    seen: set = set()

    for d in (cache_dir, bundled_dir):
        if not d.exists():
            continue
        for path in d.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("kernel") not in (
                kernel_cls.__qualname__,
                getattr(kernel_cls, "NAME", None),
            ):
                continue
            if payload.get("device") != device_fingerprint:
                continue
            cfg_dict = payload.get("config")
            if not isinstance(cfg_dict, dict):
                continue
            config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
            if config_cls is None:
                continue
            try:
                cfg = config_from_json_dict(config_cls, cfg_dict)
            except (TypeError, KeyError):
                continue
            k = tuple(sorted(cfg_dict.items()))
            if k in seen:
                continue
            seen.add(k)
            try:
                if kernel_cls(spec, cfg).is_valid_for(caps):
                    seeds.append(cfg)
            except Exception:
                continue

    return seeds
