"""``_resolve_quant_for_family`` — per-family fp8/bf16 policy.

Metal and Intel iGPU/Arc both lack native fp8 (no e4m3 in MSL,
no fp8 dpas in IGC). Both force ``QuantConfig.all_bf16()``;
CUDA leaves the requested config untouched.
"""

from __future__ import annotations

import warnings

import pytest

from quark.device import DeviceFamily
from quark.engine.base import _resolve_quant_for_family
from quark.models.waypoint_15 import QuantConfig


class TestNoFp8Families:
    @pytest.mark.parametrize(
        "family",
        [DeviceFamily.METAL, DeviceFamily.INTEL_GPU],
    )
    def test_default_request_is_forced_to_bf16(self, family):
        """``quant=None`` resolves to fp8 by default; the no-fp8 family
        constraint rewrites it to all-bf16 and emits a warning."""
        with pytest.warns(RuntimeWarning, match="forcing QuantConfig.all_bf16"):
            out = _resolve_quant_for_family(None, family)
        assert out == QuantConfig.all_bf16()

    @pytest.mark.parametrize(
        "family",
        [DeviceFamily.METAL, DeviceFamily.INTEL_GPU],
    )
    def test_explicit_bf16_no_warning(self, family):
        """When the caller already passed ``"bf16"``, no rewrite + no
        warning — the policy should only chime in for surprising
        rewrites."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = _resolve_quant_for_family("bf16", family)
        assert out == QuantConfig.all_bf16()

    @pytest.mark.parametrize(
        "family",
        [DeviceFamily.METAL, DeviceFamily.INTEL_GPU],
    )
    def test_partial_quantconfig_gets_all_fields_forced(self, family):
        """A caller-supplied QuantConfig with mixed fp8 + bf16 still
        gets every fp8 field rewritten — partial is not enough on
        a backend with no fp8 path."""
        partial = QuantConfig(
            linear="bf16", kv_cache="fp8", attn_compute="fp8", moe="bf16",
        )
        with pytest.warns(RuntimeWarning):
            out = _resolve_quant_for_family(partial, family)
        assert out == QuantConfig.all_bf16()


class TestCudaPassThrough:
    def test_cuda_default_stays_fp8(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = _resolve_quant_for_family(None, DeviceFamily.CUDA)
        assert out == QuantConfig()

    def test_cuda_bf16_request_stays_bf16(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = _resolve_quant_for_family("bf16", DeviceFamily.CUDA)
        assert out == QuantConfig.all_bf16()

    def test_cuda_partial_quantconfig_preserved(self):
        partial = QuantConfig(
            linear="bf16", kv_cache="fp8", attn_compute="fp8", moe="bf16",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = _resolve_quant_for_family(partial, DeviceFamily.CUDA)
        assert out == partial
