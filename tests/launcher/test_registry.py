"""Tests for popcorn.kernels.registry.

Pure-python — no real kernels needed. We construct minimal Kernel
subclasses on the fly to exercise the registration validation
machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from popcorn.kernels.base import Kernel, KernelConfig, KernelSpec
from popcorn.kernels.registry import (
    _REGISTRY,
    _REQUIRED_OVERRIDES,
    all_kernels,
    get,
    names,
    register,
    unregister,
)

# ---------------------------------------------------------------------------
# Toy kernel that satisfies every required override
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ToySpec(KernelSpec):
    n: int = 16


@dataclass(frozen=True)
class _ToyConfig(KernelConfig):
    block: int = 16


class _CompleteKernel(Kernel):
    """Kernel subclass that overrides every required hook so it
    can be registered."""

    NAME = "_complete_test"
    SPEC_CLS = _ToySpec
    CONFIG_CLS = _ToyConfig

    def __init__(self, spec, config):
        self.spec = spec
        self.config = config
        self._compiled = None

    # Legacy hooks
    def is_valid(self) -> bool:
        return True

    def smem_estimate(self) -> int:
        return 0

    def emit(self):
        raise NotImplementedError

    def global_tensors(self):
        return []

    def grid(self):
        return (1, 1, 1)

    def reference(self, *tensors):
        return tensors[0] if tensors else None

    def flops(self) -> int:
        return 0

    # New required overrides
    @classmethod
    def problems(cls):
        return [{"n": 16}]

    @classmethod
    def make_tensors(cls, problem):
        return {}

    @classmethod
    def tune_space(cls):
        return {"block": [8, 16, 32]}

    def param_spec(self):
        from popcorn.launcher import ParamSpec

        return ParamSpec()


# ---------------------------------------------------------------------------
# Fixtures: keep the registry isolated per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot + restore so individual tests can register without
    polluting the rest of the suite."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterDecorator:
    def test_registers_under_explicit_name(self):
        @register("toy_explicit")
        class K(_CompleteKernel):
            NAME = "toy_explicit"

        assert get("toy_explicit") is K
        assert "toy_explicit" in names()

    def test_falls_back_to_NAME_attr(self):
        @register()
        class K(_CompleteKernel):
            NAME = "toy_from_name"

        assert get("toy_from_name") is K

    def test_returns_class_unchanged(self):
        @register("toy_passthrough")
        class K(_CompleteKernel):
            NAME = "toy_passthrough"

        # The decorator must return the original class.
        assert isinstance(K, type)
        assert K is get("toy_passthrough")

    def test_idempotent_reregister_of_same_class(self):
        @register("toy_idem")
        class K(_CompleteKernel):
            NAME = "toy_idem"

        # Re-applying the decorator to the same class is a no-op.
        register("toy_idem")(K)
        assert get("toy_idem") is K

    def test_duplicate_name_different_class_raises(self):
        @register("toy_dup")
        class K1(_CompleteKernel):
            NAME = "toy_dup"

        with pytest.raises(ValueError, match="already registered"):

            @register("toy_dup")
            class K2(_CompleteKernel):
                NAME = "toy_dup"


class TestRequiredOverrideValidation:
    def test_kernel_missing_problems_rejected(self):
        with pytest.raises(TypeError, match="problems"):

            @register("toy_no_problems")
            class K(_CompleteKernel):
                NAME = "toy_no_problems"
                problems = Kernel.problems  # un-override

    def test_kernel_missing_tune_space_rejected(self):
        with pytest.raises(TypeError, match="tune_space"):

            @register("toy_no_tune")
            class K(_CompleteKernel):
                NAME = "toy_no_tune"
                tune_space = Kernel.tune_space

    def test_required_overrides_constant_matches_proposal(self):
        """Sanity: the constant lists the hooks we validate via
        function-identity comparison. ``param_spec`` dropped as a
        required override — the base default (``ParamSpec.from_function(
        emit().functions[0])``) is the canonical implementation.
        ``make_tensors`` exists in the proposal but isn't validated
        here because the legacy instance-method shape collides with
        the classmethod contract — see the registry comment."""
        assert set(_REQUIRED_OVERRIDES) == {
            "problems",
            "tune_space",
        }

    def test_register_rejects_non_kernel_subclass(self):
        with pytest.raises(TypeError, match="not a subclass of Kernel"):

            @register("not_a_kernel")
            class NotAKernel:
                pass


class TestLookupAPI:
    def test_get_raises_keyerror_with_registered_listed(self):
        @register("toy_listed")
        class K(_CompleteKernel):
            NAME = "toy_listed"

        with pytest.raises(KeyError) as exc:
            get("missing")
        assert "missing" in str(exc.value)
        assert "toy_listed" in str(exc.value)

    def test_all_kernels_returns_list(self):
        @register("toy_a")
        class A(_CompleteKernel):
            NAME = "toy_a"

        @register("toy_b")
        class B(_CompleteKernel):
            NAME = "toy_b"

        kernels = all_kernels()
        assert A in kernels
        assert B in kernels

    def test_names_is_sorted(self):
        @register("zebra")
        class Z(_CompleteKernel):
            NAME = "zebra"

        @register("apple")
        class A(_CompleteKernel):
            NAME = "apple"

        n = names()
        assert "apple" in n
        assert "zebra" in n
        assert n == sorted(n)

    def test_unregister_drops_entry(self):
        @register("toy_dropme")
        class K(_CompleteKernel):
            NAME = "toy_dropme"

        unregister("toy_dropme")
        with pytest.raises(KeyError):
            get("toy_dropme")

    def test_unregister_missing_is_silent(self):
        # Doesn't raise.
        unregister("never_registered")
