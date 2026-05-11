"""Tests for the Kernel.is_valid_for(caps) shim added in Bundle 2.

The shim is a concrete method on the Kernel base class that gates on
`caps.max_smem_per_block` and delegates the rest of the legality
check to the kernel's legacy `is_valid()` method. Subclasses can
override it when they need richer cap-based gating.
"""

from dataclasses import dataclass

from quark.device import make_test_device
from quark.ir import Builder, DType
from quark.kernels.base import Kernel, KernelConfig, KernelSpec


@dataclass(frozen=True)
class _MiniSpec(KernelSpec):
    pass


@dataclass(frozen=True)
class _MiniConfig(KernelConfig):
    smem_request: int = 1024
    legal: bool = True


class _MiniKernel(Kernel):
    """Concrete Kernel just for exercising is_valid_for. Emits a single
    SmemAllocOp sized to ``config.smem_request`` so the smem_layout
    pass produces a matching total for the is_valid_for check."""

    def __init__(self, spec: _MiniSpec, config: _MiniConfig):
        self.spec = spec
        self.config = config
        self._compiled = None

    def is_valid(self) -> bool:
        return self.config.legal

    def emit(self):
        # Emit a single smem region sized to ``smem_request``. Using
        # u8 so element count == byte count.
        b = Builder("mini")
        b.begin_function("mini")
        b.smem_alloc("S", DType.U8, (self.config.smem_request,))
        b.end_function()
        return b.module

    def global_tensors(self):  # pragma: no cover
        raise NotImplementedError

    def grid(self):  # pragma: no cover
        raise NotImplementedError

    def reference(self, *tensors):  # pragma: no cover
        raise NotImplementedError

    def flops(self):  # pragma: no cover
        raise NotImplementedError


class TestIsValidForShim:
    def test_passes_when_smem_fits_and_legal(self):
        device = make_test_device(max_smem_per_block=64 * 1024)
        k = _MiniKernel(_MiniSpec(), _MiniConfig(smem_request=32 * 1024, legal=True))
        assert k.is_valid_for(device.caps) is True

    def test_fails_when_smem_overflows(self):
        device = make_test_device(max_smem_per_block=64 * 1024)
        k = _MiniKernel(_MiniSpec(), _MiniConfig(smem_request=128 * 1024, legal=True))
        assert k.is_valid_for(device.caps) is False

    def test_delegates_to_is_valid_when_smem_ok(self):
        """When smem fits but the legacy is_valid() says no, the shim
        must return False — it's not just a smem check."""
        device = make_test_device(max_smem_per_block=64 * 1024)
        k = _MiniKernel(_MiniSpec(), _MiniConfig(smem_request=32 * 1024, legal=False))
        assert k.is_valid_for(device.caps) is False

    def test_smem_check_uses_caps_value_not_global(self):
        """Two devices with different smem caps must produce different
        verdicts on the same kernel config."""
        small = make_test_device(max_smem_per_block=48 * 1024)
        big = make_test_device(max_smem_per_block=228 * 1024)
        k = _MiniKernel(_MiniSpec(), _MiniConfig(smem_request=100 * 1024, legal=True))
        assert k.is_valid_for(small.caps) is False
        assert k.is_valid_for(big.caps) is True

    def test_subclass_can_override_for_richer_caps_check(self):
        """A kernel that needs specific matmul-shape support should be
        able to override is_valid_for directly to read caps."""

        class _NeedsBf16Mma(_MiniKernel):
            def is_valid_for(self, caps) -> bool:
                if "m16n8k16_bf16" not in caps.matmul_shapes:
                    return False
                return super().is_valid_for(caps)

        k = _NeedsBf16Mma(_MiniSpec(), _MiniConfig())
        bf16_device = make_test_device(arch_tag="sm_89")
        assert k.is_valid_for(bf16_device.caps) is True

        bare_device = make_test_device(arch_tag="sm_89", matmul_shapes=frozenset())
        assert k.is_valid_for(bare_device.caps) is False
