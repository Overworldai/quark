"""Tests for popcorn.device — DeviceFamily / DeviceCaps / Device."""

import pytest

from popcorn.device import (
    ChipGeneration,
    DeviceCaps,
    DeviceFamily,
    _default_dtypes_for,
    _detect_family,
    _forced_family,
    chip_gen_from_cuda_cc,
    make_test_device,
)
from popcorn.ir.mma_registry import shapes_for_chip


class TestDeviceFamily:
    def test_all_families_are_strings(self):
        for f in DeviceFamily:
            assert isinstance(f.value, str)
            assert f.value.islower()

    def test_lookup_by_value(self):
        assert DeviceFamily("cuda") is DeviceFamily.CUDA
        assert DeviceFamily("metal") is DeviceFamily.METAL


class TestDeviceCaps:
    def _caps(self, **overrides):
        defaults = dict(
            family=DeviceFamily.CUDA,
            name="RTX 5090",
            compute_unit_count=170,
            warp_size=32,
            max_threads_per_block=1024,
            max_smem_per_block=228 * 1024,
            max_regs_per_thread=255,
            max_regs_per_block=65536,
            arch_tag="sm_120",
            compute_capability=(12, 0),
            supports_async_copy=True,
            supports_graph_capture=True,
            supports_fp8_e4m3=True,
            supports_bf16_mma=True,
            matmul_shapes=frozenset({"m16n8k16_bf16"}),
            supported_dtypes=frozenset({"f32", "bf16"}),
            cpu_features=frozenset(),
        )
        defaults.update(overrides)
        return DeviceCaps(**defaults)

    def test_frozen(self):
        caps = self._caps()
        with pytest.raises(Exception):  # FrozenInstanceError
            caps.name = "RTX 4090"  # type: ignore[misc]

    def test_has_cpu_feature(self):
        caps = self._caps(cpu_features=frozenset({"avx512bf16", "amx_bf16"}))
        assert caps.has("avx512bf16") is True
        assert caps.has("amx_bf16") is True
        assert caps.has("sve2") is False

    def test_matmul_shapes_for_known_arch(self):
        # Ada (sm_89): bf16/f16 at k=8/k=16 and fp8 at k=32 only.
        # m16n8k16 fp8 is a Blackwell-only (sm_120+) PTX 8.7 instruction.
        s = shapes_for_chip(ChipGeneration.SM_89)
        assert "m16n8k16_bf16" in s
        assert "m16n8k8_bf16" in s
        assert "m16n8k32_e4m3" in s
        assert "m16n8k16_e4m3" not in s  # Blackwell-only

    def test_matmul_shapes_blackwell_gains_fp8_k16(self):
        s = shapes_for_chip(ChipGeneration.SM_120)
        assert "m16n8k16_e4m3" in s
        assert "m16n8k16_e5m2" in s

    def test_matmul_shapes_for_unknown_chip_is_empty(self):
        assert shapes_for_chip(ChipGeneration.UNKNOWN) == frozenset()

    def test_chip_gen_from_cuda_cc_maps_newer_to_older_on_unknown(self):
        # sm_99 doesn't exist; falls back to nearest-older we know.
        assert chip_gen_from_cuda_cc(8, 9) is ChipGeneration.SM_89
        fallback = chip_gen_from_cuda_cc(9, 9)  # unknown — should be SM_90
        cc = fallback.cuda_cc()
        assert cc is not None and cc <= (9, 9)

    def test_default_dtypes_for_ada_includes_fp8(self):
        d = _default_dtypes_for("sm_89")
        assert "e4m3" in d
        assert "e5m2" in d
        assert "bf16" in d

    def test_default_dtypes_for_ampere_excludes_fp8(self):
        d = _default_dtypes_for("sm_80")
        assert "bf16" in d
        assert "e4m3" not in d


class TestDevice:
    def test_construction_via_helper(self):
        d = make_test_device(arch_tag="sm_89", name="test-4090")
        assert d.family is DeviceFamily.CUDA
        assert d.index == 0
        assert d.caps.name == "test-4090"
        assert d.caps.arch_tag == "sm_89"
        assert d.caps.supports_fp8_e4m3 is True
        assert d.caps.supports_bf16_mma is True

    def test_fingerprint_includes_arch_and_cu_count(self):
        d = make_test_device(arch_tag="sm_89", compute_unit_count=84)
        fp = d.fingerprint()
        assert "cuda" in fp
        assert "sm_89" in fp
        assert "84cu" in fp

    def test_two_devices_different_arch_have_different_fingerprints(self):
        a = make_test_device(arch_tag="sm_89")
        b = make_test_device(arch_tag="sm_120")
        assert a.fingerprint() != b.fingerprint()

    def test_two_devices_same_arch_different_cu_count_differ(self):
        a = make_test_device(arch_tag="sm_120", compute_unit_count=128)
        b = make_test_device(arch_tag="sm_120", compute_unit_count=170)
        assert a.fingerprint() != b.fingerprint()


class TestForceBackend:
    def test_no_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("POPCORN_FORCE_BACKEND", raising=False)
        assert _forced_family() is None

    def test_valid_value(self, monkeypatch):
        monkeypatch.setenv("POPCORN_FORCE_BACKEND", "cuda")
        assert _forced_family() is DeviceFamily.CUDA

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("POPCORN_FORCE_BACKEND", "METAL")
        assert _forced_family() is DeviceFamily.METAL

    def test_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("POPCORN_FORCE_BACKEND", "vulkan")
        with pytest.raises(ValueError, match="POPCORN_FORCE_BACKEND"):
            _forced_family()


class TestDetectFamily:
    def test_force_overrides_detection(self, monkeypatch):
        monkeypatch.setenv("POPCORN_FORCE_BACKEND", "cpu")
        assert _detect_family() is DeviceFamily.CPU

    def test_returns_cpu_when_no_torch(self, monkeypatch):
        # We can't truly remove torch, but the fallback path is tested
        # via a forced env override.
        monkeypatch.setenv("POPCORN_FORCE_BACKEND", "cpu")
        assert _detect_family() is DeviceFamily.CPU


class TestCurrentDevice:
    def test_metal_family_probes_successfully(self, monkeypatch):
        from popcorn import device

        # Metal is now a supported backend (MLX probe).
        monkeypatch.setenv("POPCORN_FORCE_BACKEND", "metal")
        device.current_device.cache_clear()
        d = device.current_device()
        assert d.family is device.DeviceFamily.METAL
        device.current_device.cache_clear()

    def test_cpu_force_path_raises_not_implemented(self, monkeypatch):
        from popcorn import device

        monkeypatch.setenv("POPCORN_FORCE_BACKEND", "cpu")
        device.current_device.cache_clear()
        with pytest.raises(NotImplementedError, match="cpu"):
            device.current_device()
        device.current_device.cache_clear()
