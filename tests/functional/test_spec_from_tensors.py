"""Round-trip test for ``KernelCls.spec_from_tensors``.

For every production kernel, build tensors from ``make_tensors(problem.params)``,
run ``spec_from_tensors(*inputs, **scalar_kwargs)`` on them, and assert
the resulting spec matches what ``Spec(**problem.params)`` would produce.

This is the F1 contract: tensors + explicit scalar kwargs
reconstruct the spec. No device allocation needed — pure shape / dtype
inspection.
"""

from __future__ import annotations

import pytest

from popcorn.backend import IS_METAL
from popcorn.kernels import all_kernels

# Per-kernel: which fields from Problem.params are scalar kwargs (not
# derivable from tensor shapes) that ``spec_from_tensors`` needs. The
# positional tensor args are the kernel's TENSORS entries minus the
# "out" roles; we pull them off the make_tensors() dict in TENSORS
# declaration order.
_SCALAR_KWARGS: dict[str, tuple[str, ...]] = {
    "gemm": ("out_dtype", "compute_dtype", "b_shuffle"),
    "attn": ("B", "n_kv_heads", "gqa_ratio", "seq_len", "kv_len"),
    "owl_attn": (
        "B",
        "n_kv_heads",
        "gqa_ratio",
        "H_spatial",
        "W_spatial",
        "num_buckets",
        "pinned_dilation",
        "out_dtype",
        "compute_dtype",
        "max_segments",
    ),
    "kv_cache_update": (
        "B",
        "n_kv_heads",
        "H_spatial",
        "W_spatial",
        "num_buckets",
        "pinned_dilation",
        "kv_dtype",
        "max_segments",
    ),
    "moe_inproj": ("n_experts", "top_k", "out_dtype", "compute_dtype"),
    "moe_outproj": ("M", "n_experts", "top_k", "out_dtype", "compute_dtype"),
}

# Some kernels rename one kwarg in the functional API vs the spec field —
# e.g. GemmSpec's ``b_shuffle`` → ``spec_from_tensors(..., b_shuffled=)``.
_KWARG_ALIAS: dict[str, dict[str, str]] = {
    "gemm": {"b_shuffle": "b_shuffled"},
}

# TENSORS entry names that are the kernel's *inputs* for spec_from_tensors.
# By default: every TENSORS entry that isn't role="out". Override when the
# kernel's spec_from_tensors positional-arg order differs from TENSORS
# declaration order, or when a role="out" entry is in-out (caller
# provides it).
_INPUT_TENSOR_NAMES: dict[str, tuple[str, ...]] = {
    "gemm": ("A", "B"),
    "attn": ("Q", "K", "V_t"),
    "owl_attn": ("Q", "K_cache", "Vt_cache", "cos", "sin", "segments", "n_segments"),
    "kv_cache_update": (
        "K",
        "V",
        "cos",
        "sin",
        "frame_t",
        "Vt_cache",
        "segments",
        "n_segments",
        "K_cache",
    ),
    "moe_inproj": ("X", "W_in", "token_ids", "work_list"),
    "moe_outproj": ("h_in", "W_out", "token_ids", "slot_weights", "work_list"),
}


def _kernels_with_spec_from_tensors():
    return [cls for cls in all_kernels() if hasattr(cls, "spec_from_tensors")]


@pytest.mark.parametrize(
    "kernel_cls",
    _kernels_with_spec_from_tensors(),
    ids=lambda cls: cls.NAME,
)
def test_spec_from_tensors_round_trip(kernel_cls):
    name = kernel_cls.NAME
    if name not in _SCALAR_KWARGS:
        pytest.skip(f"{name}: no test config registered")

    problems = kernel_cls.problems()
    assert problems, f"{name}: no problems registered"

    # Pick the first problem whose dtypes round-trip through the active
    # backend. On Metal there's no native fp8, so fp8 problems can't
    # survive the dtype_to_str round-trip — we skip them rather than
    # guard per-kernel.
    def _dtypes_survive(params: dict) -> bool:
        if not IS_METAL:
            return True
        fp8_tags = {"e4m3", "e5m2"}
        return not any(isinstance(v, str) and v in fp8_tags for v in params.values())

    problem = next((p for p in problems if _dtypes_survive(p.params)), problems[0])

    try:
        tensors = kernel_cls.make_tensors(problem.params)
    except Exception as e:
        pytest.skip(f"{name}: make_tensors failed ({e!s:.60})")

    input_names = _INPUT_TENSOR_NAMES[name]
    inputs = [tensors[n] for n in input_names]

    alias = _KWARG_ALIAS.get(name, {})
    kwargs = {}
    for field in _SCALAR_KWARGS[name]:
        if field not in problem.params:
            continue
        target = alias.get(field, field)
        kwargs[target] = problem.params[field]

    derived = kernel_cls.spec_from_tensors(*inputs, **kwargs)
    expected = kernel_cls.SPEC_CLS(**problem.params)
    assert derived == expected, (
        f"{name}: spec_from_tensors mismatch\n  derived:  {derived}\n  expected: {expected}"
    )
