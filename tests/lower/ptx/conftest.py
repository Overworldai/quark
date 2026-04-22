"""Helpers for PTX-lowerer tests."""

import re

import pytest

from quark.ir import Builder
from quark.lower.ptx import LoweredKernel, PtxLowerer


@pytest.fixture
def fresh_builder() -> Builder:
    b = Builder("t")
    b.begin_function("f")
    return b


def lower(b: Builder) -> str:
    """Close the function (if needed) and return the PTX text string.

    Most tests just want the PTX to grep for instruction shapes, so
    this helper drops the `LoweredKernel` wrapper. Use `lower_full()`
    when you also need `smem_bytes` / `kernel_name`.
    """
    if b._fn is not None:  # type: ignore[attr-defined]
        b.end_function()
    return PtxLowerer().lower_module(b.module).ptx


def lower_full(b: Builder) -> LoweredKernel:
    """Close the function and return the full `LoweredKernel`."""
    if b._fn is not None:  # type: ignore[attr-defined]
        b.end_function()
    return PtxLowerer().lower_module(b.module)


def body(ptx: str) -> str:
    """Return the instruction-body substring between the kernel's { and
    the closing }. Useful for assertions that don't care about the
    module header."""
    m = re.search(r"\.visible \.entry.*?\)\s*\{(.*?)\n\}", ptx, re.DOTALL)
    assert m is not None, "kernel body not found in PTX output"
    return m.group(1)
