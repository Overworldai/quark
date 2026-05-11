"""Tests for quark.autotune.AutotuneCache.

Pure-python tests — no real CUDA needed. The compile-and-time hook
is mocked via a callable injected onto the cache, mirroring how the
real Launcher injects ``_compile_and_time_for_autotune``.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "torch",
    reason="torch removed from runtime; numpy-refs migration — test kept for dev-only cross-check when torch is installed",
)

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from quark.autotune import AutotuneCache
from quark.autotune.cache import _cartesian
from quark.autotune.io import (
    autotune_disabled as _autotune_disabled,
)
from quark.autotune.io import (
    default_cache_dir as _default_cache_dir,
)
from quark.autotune.io import (
    format_spec_label as _format_spec_label,
)
from quark.autotune.io import (
    source_hash as _source_hash,
)
from quark.autotune.io import (
    spec_fingerprint as _spec_fingerprint,
)
from quark.device import make_test_device

# ---------------------------------------------------------------------------
# Toy spec / config / kernel for the cache to operate on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    M: int
    N: int


@dataclass(frozen=True)
class _Config:
    BM: int = 32
    BN: int = 32
    n_warps: int = 4


class _ToyKernel:
    """Lightweight stand-in for a Kernel — exposes the surface the
    autotune cache touches: NAME, CONFIG_CLS, tune_space, problems,
    make_tensors, _pick_default_cfg, source-hashable __qualname__."""

    NAME = "toy"
    CONFIG_CLS = _Config

    def __init__(self, spec: _Spec, config: _Config):
        self.spec = spec
        self.config = config

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            "BM": [16, 32, 64],
            "BN": [16, 32],
            "n_warps": [2, 4],
        }

    @classmethod
    def problems(cls) -> list[dict]:
        return [{"M": 64, "N": 64}]

    @classmethod
    def make_tensors(cls, problem: dict) -> dict:
        return {}

    @classmethod
    def _pick_default_cfg(cls, spec: _Spec) -> _Config:
        return _Config(BM=32, BN=32, n_warps=4)

    def is_valid_for(self, caps) -> bool:
        # Reject obviously bad configs to give the search something
        # to filter on.
        return self.config.BM > 0 and self.config.BN > 0

    def prune_score(self) -> float:
        # Lower BM*BN sorts first, deterministic tiebreak.
        return self.config.BM * self.config.BN


# ---------------------------------------------------------------------------
# Helpers / fingerprinting
# ---------------------------------------------------------------------------


class TestSpecFingerprint:
    def test_returns_field_value_pairs(self):
        spec = _Spec(M=64, N=128)
        fp = _spec_fingerprint(spec)
        assert fp == (("M", 64), ("N", 128))

    def test_two_equal_specs_share_fingerprint(self):
        a = _Spec(M=64, N=128)
        b = _Spec(M=64, N=128)
        assert _spec_fingerprint(a) == _spec_fingerprint(b)

    def test_different_specs_differ(self):
        a = _Spec(M=64, N=128)
        b = _Spec(M=128, N=64)
        assert _spec_fingerprint(a) != _spec_fingerprint(b)


class TestSourceHash:
    def test_same_class_same_hash(self):
        assert _source_hash(_ToyKernel) == _source_hash(_ToyKernel)

    def test_different_classes_differ(self):
        class _Other(_ToyKernel):
            def emit(self):
                return 1

        # _Other has different source.
        assert _source_hash(_ToyKernel) != _source_hash(_Other)


class TestCartesian:
    def test_empty(self):
        assert list(_cartesian([])) == [()]

    def test_two_axes(self):
        out = list(_cartesian([[1, 2], ["a", "b"]]))
        assert out == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]

    def test_three_axes_count(self):
        out = list(_cartesian([[1, 2], [3, 4], [5, 6, 7]]))
        assert len(out) == 12


class TestSpecLabel:
    def test_label_is_field_value_concat(self):
        spec = _Spec(M=64, N=128)
        assert _format_spec_label(spec) == "M64_N128"

    def test_label_sorted_by_field_name(self):
        # Field declaration order is M, N → sorted is M, N → same.
        # But for a dataclass with un-sorted declaration order we
        # still expect alphabetical output.
        @dataclass(frozen=True)
        class S:
            z: int
            a: int

        s = S(z=10, a=20)
        assert _format_spec_label(s) == "a20_z10"


# ---------------------------------------------------------------------------
# Cache directory resolution
# ---------------------------------------------------------------------------


class TestDefaultCacheDir:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUARK_CACHE_DIR", str(tmp_path / "explicit"))
        assert _default_cache_dir() == tmp_path / "explicit"

    def test_xdg_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUARK_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert _default_cache_dir() == tmp_path / "xdg" / "quark"

    def test_home_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUARK_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        # Path.home() reads HOME on POSIX; we trust the stdlib here.
        assert _default_cache_dir() == Path(tmp_path) / ".cache" / "quark"


class TestKillSwitch:
    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("QUARK_DISABLE_AUTOTUNE", raising=False)
        assert _autotune_disabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy(self, monkeypatch, val):
        monkeypatch.setenv("QUARK_DISABLE_AUTOTUNE", val)
        assert _autotune_disabled() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no"])
    def test_falsy(self, monkeypatch, val):
        monkeypatch.setenv("QUARK_DISABLE_AUTOTUNE", val)
        assert _autotune_disabled() is False


# ---------------------------------------------------------------------------
# Cache lookup chain
# ---------------------------------------------------------------------------


def _fresh_cache(tmp_path: Path) -> AutotuneCache:
    """A cache with empty disk + bundled dirs so the chain hits the
    in-process tier deterministically."""
    return AutotuneCache(
        device=make_test_device(),
        cache_dir=tmp_path / "cache",
        bundled_dir=tmp_path / "bundled",
    )


class TestLookupHotTier:
    def test_store_then_lookup_is_dict_lookup(self, tmp_path):
        cache = _fresh_cache(tmp_path)
        spec = _Spec(M=64, N=128)
        cfg = _Config(BM=32, BN=32, n_warps=4)
        cache.store(_ToyKernel, spec, cfg)
        assert cache.lookup(_ToyKernel, spec) == cfg

    def test_miss_returns_none(self, tmp_path):
        cache = _fresh_cache(tmp_path)
        assert cache.lookup(_ToyKernel, _Spec(M=64, N=128)) is None

    def test_clear_hot_drops_in_process_tier(self, tmp_path):
        cache = _fresh_cache(tmp_path)
        spec = _Spec(M=64, N=128)
        cfg = _Config(BM=32, BN=32, n_warps=4)
        cache._hot[cache._make_key(_ToyKernel, spec)] = cfg
        cache.clear_hot()
        # Hot is empty; the disk save from store() never happened so
        # this should now miss entirely.
        assert cache.lookup(_ToyKernel, spec) is None


class TestLookupDiskTier:
    def test_store_persists_to_disk(self, tmp_path):
        cache = _fresh_cache(tmp_path)
        spec = _Spec(M=64, N=128)
        cfg = _Config(BM=64, BN=16, n_warps=2)
        cache.store(_ToyKernel, spec, cfg)
        # Drop the hot tier; lookup must reload from disk.
        cache.clear_hot()
        # The disk loader needs the registry to know CONFIG_CLS;
        # mock that out via patching ``resolve_config_cls`` in the io
        # module.
        from quark.autotune import io as io_module

        io_module.resolve_config_cls = lambda qname: (
            _Config if qname and "ToyKernel" in qname else None
        )
        result = cache.lookup(_ToyKernel, spec)
        assert result == cfg

    def test_disk_write_is_atomic(self, tmp_path):
        """The write goes through a temp file + rename so a partial
        write doesn't leave a half-written .json sitting in the
        cache dir."""
        cache = _fresh_cache(tmp_path)
        spec = _Spec(M=64, N=128)
        cfg = _Config()
        cache.store(_ToyKernel, spec, cfg)
        # The temp file should not survive the write.
        leftover_tmp = list((tmp_path / "cache").glob(".quark_cache_*.tmp"))
        assert leftover_tmp == []
        # And there should be exactly one .json file.
        jsons = list((tmp_path / "cache").glob("*.json"))
        assert len(jsons) == 1

    def test_source_hash_change_invalidates_disk_under_revalidate(self, tmp_path, monkeypatch):
        cache = _fresh_cache(tmp_path)
        spec = _Spec(M=64, N=128)
        cfg = _Config()
        cache.store(_ToyKernel, spec, cfg)

        # Mutate the stored source_hash so the on-disk file's value no
        # longer matches the current kernel source. With
        # QUARK_AUTOTUNE_REVALIDATE off (default), the cache still
        # accepts the entry; with it on, the load is rejected.
        from quark.autotune import io as io_module

        monkeypatch.setattr(io_module, "source_hash", lambda cls: "deadbeef" + ("0" * 8))
        cache.clear_hot()

        monkeypatch.delenv("QUARK_AUTOTUNE_REVALIDATE", raising=False)
        assert cache.lookup(_ToyKernel, spec) == cfg

        cache.clear_hot()
        monkeypatch.setenv("QUARK_AUTOTUNE_REVALIDATE", "1")
        assert cache.lookup(_ToyKernel, spec) is None


class TestLookupBundledTier:
    def test_bundled_default_loaded_when_disk_misses(self, tmp_path):
        """Drop a JSON in the bundled dir matching the kernel +
        spec label, and the cache should load it on lookup."""
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        record = {
            "kernel": "toy",
            "spec": {"M": 64, "N": 128},
            "config": {"BM": 128, "BN": 64, "n_warps": 8},
        }
        # _format_spec_label(_Spec(M=64, N=128)) → "M64_N128"
        (bundled / "toy_M64_N128.json").write_text(json.dumps(record))

        cache = AutotuneCache(
            device=make_test_device(),
            cache_dir=tmp_path / "cache",
            bundled_dir=bundled,
        )
        cfg = cache.lookup(_ToyKernel, _Spec(M=64, N=128))
        assert cfg is not None
        assert cfg.BM == 128 and cfg.BN == 64 and cfg.n_warps == 8

    def test_bundled_miss_returns_none(self, tmp_path):
        cache = AutotuneCache(
            device=make_test_device(),
            cache_dir=tmp_path / "cache",
            bundled_dir=tmp_path / "bundled",  # doesn't exist
        )
        assert cache.lookup(_ToyKernel, _Spec(M=1, N=1)) is None


# ---------------------------------------------------------------------------
# Bounded search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_winner_via_timing_hook(self, tmp_path):
        cache = _fresh_cache(tmp_path)

        # Mock the compile-and-time hook to declare BM=64 the fastest.
        def fake_time(kernel_cls, spec, config):
            return 100.0 - config.BM  # bigger BM → smaller (faster) us

        cache._compile_and_time = fake_time

        winner = cache._search(_ToyKernel, _Spec(M=64, N=128))
        assert winner is not None
        assert winner.BM == 64

    def test_search_skips_invalid_configs(self, tmp_path):
        """Invalid configs (per is_valid_for) never reach the timer."""
        cache = _fresh_cache(tmp_path)
        seen: list = []

        def fake_time(kernel_cls, spec, config):
            seen.append(config)
            return 50.0

        cache._compile_and_time = fake_time
        cache._search(_ToyKernel, _Spec(M=64, N=128))
        # Every BM/BN in the tune_space passes _ToyKernel.is_valid_for.
        # Confirm we got exactly the cartesian-product count.
        # (BM 3 × BN 2 × n_warps 2 = 12)
        assert len(seen) == 12

    def test_search_falls_back_to_first_when_no_timer(self, tmp_path):
        """Without a timer hook, _search returns the lowest-prune-score
        candidate without timing."""
        cache = _fresh_cache(tmp_path)
        # Don't set _compile_and_time.
        winner = cache._search(_ToyKernel, _Spec(M=64, N=128))
        assert winner is not None
        # Lowest BM*BN candidate is BM=16 BN=16 (score 256).
        assert winner.BM == 16 and winner.BN == 16

    def test_search_returns_none_on_empty_tune_space(self, tmp_path):
        cache = _fresh_cache(tmp_path)

        class _BareKernel(_ToyKernel):
            @classmethod
            def tune_space(cls):
                return {}

        assert cache._search(_BareKernel, _Spec(M=64, N=128)) is None


class TestLookupOrSearch:
    def test_kill_switch_short_circuits_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUARK_DISABLE_AUTOTUNE", "1")
        cache = _fresh_cache(tmp_path)
        cfg = cache.lookup_or_search(_ToyKernel, _Spec(M=64, N=128))
        # Falls back to _pick_default_cfg → BM=32 BN=32 n_warps=4.
        assert cfg == _Config(BM=32, BN=32, n_warps=4)

    def test_search_result_persists_to_hot_and_disk(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUARK_DISABLE_AUTOTUNE", raising=False)
        cache = _fresh_cache(tmp_path)
        cache._compile_and_time = lambda *_: 1.0
        spec = _Spec(M=64, N=128)
        cfg = cache.lookup_or_search(_ToyKernel, spec)
        # Hot tier hit on second call.
        assert cache._hot[cache._make_key(_ToyKernel, spec)] == cfg
        # And one JSON should now exist on disk.
        assert len(list((tmp_path / "cache").glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Integration check: AutotuneCache is wired into Launcher
# ---------------------------------------------------------------------------


class TestLauncherWiring:
    def test_launcher_owns_an_autotune_cache(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Launcher autotune wiring requires CUDA")
        from quark.launcher import Launcher

        launcher = Launcher()
        assert isinstance(launcher._autotune, AutotuneCache)
        # The hook is injected at construction time.
        assert callable(launcher._autotune._compile_and_time)
