"""Unit tests for ``metal_harness.py`` — pure Python, runs on any platform.

Pins the exact generated source for representative kernel shapes so we
catch any drift against MLX's signature format.
"""

from __future__ import annotations

from quark.drivers.metal_harness import (
    _THREAD_ATTRIBUTES,
    ScalarParam,
    TensorParam,
    build_kernel_source,
)


class TestBasicSignature:
    def test_vec_add_minimal(self):
        body = "C[0] = A[0] + B[0];"
        src, layout = build_kernel_source(
            name="vec_add",
            body=body,
            inputs=[TensorParam("A", "float"), TensorParam("B", "float")],
            outputs=[TensorParam("C", "float")],
        )
        assert "[[kernel]] void vec_add(" in src
        assert "const device float* A [[buffer(0)]]" in src
        assert "const device float* B [[buffer(1)]]" in src
        assert "device float* C [[buffer(2)]]" in src
        assert src.rstrip().endswith("}")
        assert [s.kind for s in layout.slots] == ["input", "input", "output"]
        assert [s.name for s in layout.slots] == ["A", "B", "C"]

    def test_with_thread_position_in_grid(self):
        body = """
        uint idx = thread_position_in_grid.x;
        C[idx] = A[idx] + B[idx];
        """
        src, _ = build_kernel_source(
            name="vec_add_idx",
            body=body,
            inputs=[TensorParam("A", "float"), TensorParam("B", "float")],
            outputs=[TensorParam("C", "float")],
        )
        assert "uint3 thread_position_in_grid [[thread_position_in_grid]]" in src

    def test_attribute_only_injected_if_referenced(self):
        body = "C[0] = A[0];"
        src, _ = build_kernel_source(
            name="identity",
            body=body,
            inputs=[TensorParam("A", "float")],
            outputs=[TensorParam("C", "float")],
        )
        assert "thread_position_in_grid" not in src
        assert "threadgroup_position_in_grid" not in src

    def test_every_builtin_wirable(self):
        body = "\n".join(f"// use {attr}" for attr, _ in _THREAD_ATTRIBUTES)
        body += "\nC[0] = 0.0f;"
        src, _ = build_kernel_source(
            name="all_attrs",
            body=body,
            inputs=[],
            outputs=[TensorParam("C", "float")],
        )
        for attr, dtype in _THREAD_ATTRIBUTES:
            assert f"{dtype} {attr} [[{attr}]]" in src


class TestAutoInjectedShapeStrides:
    def test_shape_injection(self):
        body = "uint m = X_shape[0]; C[0] = X[0];"
        src, layout = build_kernel_source(
            name="use_shape",
            body=body,
            inputs=[TensorParam("X", "float")],
            outputs=[TensorParam("C", "float")],
        )
        assert "const device float* X [[buffer(0)]]" in src
        assert "const constant int* X_shape [[buffer(1)]]" in src
        assert "device float* C [[buffer(2)]]" in src
        assert [s.kind for s in layout.slots] == ["input", "shape", "output"]

    def test_strides_and_ndim_injection(self):
        body = "uint n = X_ndim + X_shape[0] + X_strides[0]; C[0] = X[0];"
        src, layout = build_kernel_source(
            name="use_all_meta",
            body=body,
            inputs=[TensorParam("X", "float")],
            outputs=[TensorParam("C", "float")],
        )
        assert "const constant int* X_shape [[buffer(1)]]" in src
        assert "const constant int64_t* X_strides [[buffer(2)]]" in src
        assert "const constant int& X_ndim [[buffer(3)]]" in src
        assert "device float* C [[buffer(4)]]" in src
        assert [s.kind for s in layout.slots] == ["input", "shape", "strides", "ndim", "output"]

    def test_shape_only_on_referenced_inputs(self):
        body = "uint m = X_shape[0]; C[0] = X[0] + Y[0];"
        src, layout = build_kernel_source(
            name="selective",
            body=body,
            inputs=[TensorParam("X", "float"), TensorParam("Y", "float")],
            outputs=[TensorParam("C", "float")],
        )
        assert "const constant int* X_shape [[buffer(1)]]" in src
        assert "Y_shape" not in src


class TestAtomicOutputs:
    def test_atomic_output_uses_atomic_wrapper(self):
        body = "atomic_fetch_add_explicit(&C[0], 1.0f, memory_order_relaxed);"
        src, _ = build_kernel_source(
            name="scatter_add",
            body=body,
            inputs=[],
            outputs=[TensorParam("C", "float", atomic=True)],
        )
        assert "device atomic<float>* C [[buffer(0)]]" in src
        assert "device float* C" not in src


class TestScalars:
    def test_scalar_emitted_as_device_pointer_mlx_compat(self):
        body = "C[0] = A[0] * scale[0];"
        src, layout = build_kernel_source(
            name="scaled",
            body=body,
            inputs=[TensorParam("A", "float")],
            outputs=[TensorParam("C", "float")],
            scalars=[ScalarParam("scale", "float")],
        )
        assert "const device float* scale [[buffer(2)]]" in src
        assert [s.kind for s in layout.slots] == ["input", "output", "scalar"]


class TestHeader:
    def test_header_emitted_before_signature(self):
        header = "typedef float myfloat;"
        src, _ = build_kernel_source(
            name="hdr_test",
            body="C[0] = 1.0;",
            inputs=[],
            outputs=[TensorParam("C", "float")],
            header=header,
        )
        assert src.index("typedef") < src.index("[[kernel]]")


class TestSnapshot:
    def test_vec_add_exact(self):
        body = "uint i = thread_position_in_grid.x; C[i] = A[i] + B[i];"
        src, _ = build_kernel_source(
            name="vec_add",
            body=body,
            inputs=[TensorParam("A", "float"), TensorParam("B", "float")],
            outputs=[TensorParam("C", "float")],
        )
        expected = (
            "[[kernel]] void vec_add(\n"
            "    const device float* A [[buffer(0)]],\n"
            "    const device float* B [[buffer(1)]],\n"
            "    device float* C [[buffer(2)]],\n"
            "    uint3 thread_position_in_grid [[thread_position_in_grid]]\n"
            ") {\n"
            "uint i = thread_position_in_grid.x; C[i] = A[i] + B[i];\n"
            "}"
        )
        assert src.strip() == expected
