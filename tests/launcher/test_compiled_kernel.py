"""Tests for popcorn.launcher.CompiledKernel.launch dispatch logic.

These exercise the validation / packing / dispatch path with a mock
driver — no real CUDA needed. End-to-end CUDA tests live in
test_cuda_e2e.py and skip when no GPU is present.
"""

import pytest
import torch

from popcorn.ir import DType
from popcorn.launcher import (
    BufferSpec,
    CompiledKernel,
    ParamSpec,
    ProgramFootprint,
    ScalarSpec,
    _check_contiguous,
    _check_dtype,
)

# ---------------------------------------------------------------------------
# A mock driver that records what launch was called with
# ---------------------------------------------------------------------------


class _RecordingDriver:
    def __init__(self):
        self.calls: list[dict] = []
        self.next_stream = 0xCAFE

    def launch(self, mod, *, grid, block, buffer_ptrs, scalar_args, stream):
        self.calls.append(
            dict(
                mod=mod,
                grid=grid,
                block=block,
                buffer_ptrs=list(buffer_ptrs),
                scalar_args=list(scalar_args),
                stream=stream,
            )
        )

    def current_torch_stream(self):
        return self.next_stream

    def stream_from_torch(self, s):
        return self.next_stream + 1


def _mock_kernel(buffers, scalars=()) -> CompiledKernel:
    spec = ParamSpec(buffers=tuple(buffers), scalars=tuple(scalars))
    return CompiledKernel(
        driver=_RecordingDriver(),
        module=object(),
        entry="k",
        grid_fn=lambda: (4, 1, 1),
        block_fn=lambda: (32, 1, 1),
        param_spec=spec,
        footprint=ProgramFootprint(smem_bytes=0),
    )


# ---------------------------------------------------------------------------
# _check_dtype + _check_contiguous
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors for the cuda check")
class TestCheckDtype:
    def test_f32_matches(self):
        t = torch.zeros(4, dtype=torch.float32, device="cuda")
        _check_dtype(t, DType.F32)  # no raise

    def test_bf16_matches(self):
        t = torch.zeros(4, dtype=torch.bfloat16, device="cuda")
        _check_dtype(t, DType.BF16)

    def test_mismatch_raises(self):
        t = torch.zeros(4, dtype=torch.float32, device="cuda")
        with pytest.raises(TypeError, match="buffer dtype mismatch"):
            _check_dtype(t, DType.BF16)


class TestCheckContiguous:
    def test_contiguous_passes(self):
        t = torch.zeros(8, 4, dtype=torch.float32)
        _check_contiguous(t, BufferSpec(name="X", dtype=DType.F32))

    def test_non_contiguous_raises(self):
        t = torch.zeros(8, 4, dtype=torch.float32).T  # transpose → non-contig
        with pytest.raises(ValueError, match="must be contiguous"):
            _check_contiguous(t, BufferSpec(name="X", dtype=DType.F32))


# ---------------------------------------------------------------------------
# CompiledKernel.launch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="launch path needs cuda tensors")
class TestCompiledKernelLaunch:
    def test_dispatches_through_driver(self):
        ck = _mock_kernel(buffers=[BufferSpec("X", DType.F32), BufferSpec("Y", DType.F32)])
        x = torch.zeros(8, dtype=torch.float32, device="cuda")
        y = torch.zeros(8, dtype=torch.float32, device="cuda")
        ck.launch(buffers=[x, y])
        calls = ck.driver.calls
        assert len(calls) == 1
        c = calls[0]
        assert c["grid"] == (4, 1, 1)
        assert c["block"] == (32, 1, 1)
        assert c["buffer_ptrs"] == [x.data_ptr(), y.data_ptr()]
        assert c["scalar_args"] == []
        assert c["stream"] == 0xCAFE

    def test_buffer_count_mismatch(self):
        ck = _mock_kernel(buffers=[BufferSpec("X", DType.F32), BufferSpec("Y", DType.F32)])
        x = torch.zeros(8, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="expected 2 buffers"):
            ck.launch(buffers=[x])

    def test_buffer_on_cpu_rejected(self):
        ck = _mock_kernel(buffers=[BufferSpec("X", DType.F32)])
        x = torch.zeros(8, dtype=torch.float32)  # CPU
        with pytest.raises(ValueError, match="not a GPU device"):
            ck.launch(buffers=[x])

    def test_dtype_mismatch_rejected(self):
        ck = _mock_kernel(buffers=[BufferSpec("X", DType.BF16)])
        x = torch.zeros(8, dtype=torch.float32, device="cuda")
        with pytest.raises(TypeError, match="buffer dtype mismatch"):
            ck.launch(buffers=[x])

    def test_non_contiguous_rejected(self):
        ck = _mock_kernel(buffers=[BufferSpec("X", DType.F32)])
        x = torch.zeros(8, 4, dtype=torch.float32, device="cuda").T
        with pytest.raises(ValueError, match="must be contiguous"):
            ck.launch(buffers=[x])

    def test_scalar_packing_passed_through(self):
        ck = _mock_kernel(
            buffers=[BufferSpec("X", DType.F32)],
            scalars=[ScalarSpec("N", DType.U32), ScalarSpec("scale", DType.F32)],
        )
        x = torch.zeros(8, dtype=torch.float32, device="cuda")
        ck.launch(buffers=[x], scalars=(64, 2.5))
        c = ck.driver.calls[0]
        assert len(c["scalar_args"]) == 2
        # Each blob is one scalar's bytes, in declaration order.
        import struct

        assert c["scalar_args"][0] == struct.pack("=I", 64)
        assert c["scalar_args"][1] == struct.pack("=f", 2.5)

    def test_int_stream_passed_through(self):
        ck = _mock_kernel(buffers=[BufferSpec("X", DType.F32)])
        x = torch.zeros(8, dtype=torch.float32, device="cuda")
        ck.launch(buffers=[x], stream=0xDEADBEEF)
        assert ck.driver.calls[0]["stream"] == 0xDEADBEEF
