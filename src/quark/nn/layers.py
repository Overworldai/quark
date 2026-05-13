"""
EXEMPT FROM 500-LINE RULE: all leaf module classes (Linear, Patchify,
KVCacheUpdate, OwlAttn, etc.) form a single cohesive API. Splitting
would scatter the module hierarchy across files with no readability
benefit.

"""

from __future__ import annotations

import sys

from quark.nn.module import (
    Module,
    Parameter,
    _quantize_to_e4m3,
    _tensor,
    _zeros,
)

import os as _os

_IS_METAL = sys.platform == "darwin"
# Use the runtime's auto-detect (OpenCL availability + env override),
# not just the env var — scripts that rely on auto-detect won't have
# ``QUARK_BACKEND`` set but still run on Intel.
from quark.runtime.sync import _IS_OCL  # noqa: E402
_IS_HOST_COHERENT = _IS_METAL or _IS_OCL  # mapped device pointers — memmove works in place


def _quark_dtype(t) -> str:
    """Return ``t``'s quark dtype string regardless of backend.

    QuarkTensor on CUDA exposes ``.dtype = "bf16"`` (string). Metal
    carriers are numpy arrays where ``.dtype`` is ``uint16`` etc., so
    fall back to the ``quark_dtype`` tag, then to a numpy-name map.
    """
    qd = getattr(t, "quark_dtype", None)
    if isinstance(qd, str):
        return qd
    raw = getattr(t, "dtype", None)
    if isinstance(raw, str):
        return raw
    name = getattr(raw, "name", str(raw))
    return {
        "uint16": "bf16",
        "float16": "f16",
        "float32": "f32",
        "int32": "s32",
        "int64": "s64",
        "uint8": "u8",
        "int8": "s8",
    }.get(name, name)


class _TaggedNdarray:
    """Lazy import of the ndarray subclass used to tag carriers on Metal."""

    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is None:
            import numpy as np

            class _Tagged(np.ndarray):
                def __array_finalize__(self, obj):
                    if obj is None:
                        return
                    self.quark_dtype = getattr(obj, "quark_dtype", None)

            cls._cls = _Tagged
        return cls._cls


def _tag_carrier(arr, dtype: str):
    """Tag a numpy carrier array with ``quark_dtype = dtype``."""
    out = arr.view(_TaggedNdarray.get())
    out.quark_dtype = dtype
    return out


def _set_s32(t, value: int) -> None:
    """Write a single-element s32 tensor to ``value`` (Metal- + OCL-aware)."""
    if _IS_HOST_COHERENT:
        # Mapped pointer — write directly via ctypes. Metal pool
        # buffers and OCL USM-shared memory both expose the device
        # pointer as a host-writable address.
        import ctypes

        bits = (int(value) & 0xFFFFFFFF).to_bytes(4, "little")
        if hasattr(t, "metal_handle") or hasattr(t, "ocl_handle") or hasattr(t, "data_ptr"):
            ctypes.memmove(t.data_ptr(), bits, 4)
        else:
            # Plain numpy carrier — overwrite in place.
            import numpy as np

            t[...] = np.frombuffer(bits, dtype=np.int32 if t.dtype == np.int32 else np.uint32)[0]
        return
    from quark.runtime.cuda import CudaRuntime

    CudaRuntime.instance().memset_d32(t.data_ptr(), int(value) & 0xFFFFFFFF, 1, stream=0)


def _pad_rows(x, target_M: int):
    """Pad ``x`` from ``M`` to ``target_M`` rows with zeros.

    Backend-agnostic: numpy on Metal (only when ``x`` is a host
    ndarray), ``QuarkTensor`` on CUDA. For Metal QuarkTensor inputs
    (the hot path — outputs of a previous kernel, possibly mid-lazy-
    queue), a Metal-side row-pad kernel is queued so the pad runs
    *after* the producer kernel without forcing an eval-queue flush.
    Earlier the helper called ``x.to_numpy()`` mid-forward, which
    flushed the entire pending lazy queue (~5 ms of GPU stall per
    call). Profile showed 40 calls per forward = 200 ms of pure
    stall.
    """
    M = int(x.shape[0])
    if target_M <= M:
        return x, M
    if _IS_METAL:
        import numpy as np

        qd = _quark_dtype(x)
        # Path A — caller passed a host numpy array. Cheap concat,
        # no GPU flush. (Used by ControllerInputEmbedding for the
        # always-host ctrl_input.)
        if not hasattr(x, "metal_handle"):
            carrier_dt = {
                "bf16": np.uint16,
                "f16": np.float16,
                "f32": np.float32,
                "s32": np.int32,
                "u8": np.uint8,
                "s8": np.int8,
                "e4m3": np.uint8,
                "e5m2": np.uint8,
                "u16": np.uint16,
            }.get(qd, np.uint16)
            x_np = np.asarray(x)
            pad = np.zeros((target_M - M, int(x.shape[1])), dtype=carrier_dt)
            out = np.concatenate([x_np, pad], axis=0)
            return _tag_carrier(out, qd), M
        # Path B — QuarkTensor from a prior kernel. Queue a
        # Metal-side pad-and-copy that runs in the lazy queue.
        return _metal_pad_rows(x, target_M, qd), M
    from quark.runtime.tensor import QuarkTensor

    pad = QuarkTensor.zeros(target_M - M, int(x.shape[1]), dtype=_quark_dtype(x))
    out = QuarkTensor.cat([x, pad], dim=0)
    return out, M


def _metal_pad_rows(x, target_M: int, qd: str):
    """Queue a Metal kernel that pads ``x`` (M, K) → (target_M, K) with
    zero rows, runs after ``x``'s producer in the lazy queue.

    Allocates a fresh pool buffer for the output. The kernel writes
    M valid rows + (target_M - M) zero rows. Caller never blocks on
    ``x``'s data.
    """
    import numpy as np

    from quark.drivers import _metal_dispatch as _md
    from quark.runtime.tensor import QuarkTensor, _contiguous_strides, _MetalStorage

    M = int(x.shape[0])
    K = int(x.shape[1])
    elem_size = {
        "bf16": 2,
        "f16": 2,
        "f32": 4,
        "s32": 4,
        "u8": 1,
        "s8": 1,
        "e4m3": 1,
        "e5m2": 1,
    }.get(qd, 2)

    # Compile the kernel once per elem_size, cache it.
    cache = _metal_pad_rows._cache  # type: ignore[attr-defined]
    pipeline = cache.get(elem_size)
    if pipeline is None:
        type_map = {1: "uchar", 2: "ushort", 4: "uint"}
        t = type_map[elem_size]
        name = f"pad_rows_{elem_size}"
        src = (
            "#include <metal_stdlib>\nusing namespace metal;\n"
            f"kernel void {name}(\n"
            f"    device const {t}* src   [[buffer(0)]],\n"
            f"    device {t}*       dst   [[buffer(1)]],\n"
            "    constant int* meta [[buffer(2)]],\n"
            "    uint tid [[thread_position_in_grid]])\n"
            "{\n"
            "    int M = meta[0];\n"
            "    int K = meta[1];\n"
            "    int target = meta[2];\n"
            "    int total = target * K;\n"
            "    if ((int)tid >= total) return;\n"
            "    int row = (int)tid / K;\n"
            "    int col = (int)tid % K;\n"
            f"    dst[tid] = (row < M) ? src[row * K + col] : ({t})0;\n"
            "}\n"
        )
        pipeline = _md.compile(src, name, 196610)
        cache[elem_size] = pipeline

    # Pack metadata. Keep the array alive until eval_queue runs.
    meta = np.array([M, K, target_M], dtype=np.int32)
    from quark.functional._dispatch import _copy_strided_meta

    _copy_strided_meta.append(meta)

    out_nbytes = target_M * K * elem_size
    plan = [(0, 0, 0), (1, 0, 1), (0, 1, 2)]
    total_threads = target_M * K
    tg_size = min(256, total_threads)
    grid = (total_threads, 1, 1)

    src_handle = int(x.metal_handle)
    out_handle, out_ptr = _md.queue_launch(
        pipeline,
        [src_handle, -1],  # input handles (src lazy + meta eager)
        [0, int(meta.ctypes.data)],  # input ptrs
        [0, int(meta.nbytes)],  # input nbytes
        out_nbytes,
        plan,
        grid,
        (tg_size, 1, 1),
        0,  # smem
    )

    storage = _MetalStorage(out_handle, out_ptr, out_nbytes)
    out_shape = (target_M, K)
    qt = QuarkTensor(storage, out_shape, _contiguous_strides(out_shape), 0, qd)
    # ``QuarkTensor`` doesn't carry the quark_dtype string in the same
    # attribute spot as numpy carriers, but its ``.dtype`` already returns
    # the short string "bf16" / etc., so callers' ``_quark_dtype`` resolves.
    return qt


_metal_pad_rows._cache = {}  # type: ignore[attr-defined]


def _cached_out_buf(module, shape, dtype):
    """Return a cached ``empty`` buffer of ``shape`` / ``dtype`` on ``module``.

    Per-call ``cuMemAllocAsync`` for output tensors shows up as
    ~150 µs/call on Ada/Blackwell, so any stateless ``nn.Module`` that
    wraps a ``pcf.*`` kernel needs to own its output buffer (same trick
    Linear uses in ``_out_buf``). Cache key is ``(shape, dtype)``: each
    distinct shape (e.g. attn x vs mlp x) gets its own buffer. Kernel
    writes every element, so we don't zero — ``empty`` is enough.

    """
    cache = getattr(module, "_out_buf_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(module, "_out_buf_cache", cache)
    key = (shape, dtype)
    buf = cache.get(key)
    if buf is None:
        from quark.runtime.tensor import QuarkTensor

        buf = QuarkTensor.empty(*shape, dtype=dtype)
        cache[key] = buf
    return buf


class Linear(Module):
    """``y = x @ weight.T [+ bias]`` via ``pcf.gemm``.

    Drop-in for ``torch.nn.Linear`` in inference-only models. Weights
    are ``[out_features, in_features]`` (same convention).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        out_dtype: str = "f16",
        fp8_skip: bool = False,
    ):
        self.weight = Parameter(_zeros(out_features, in_features, dtype="bf16"))
        self._has_bias = bias
        self._out_dtype = out_dtype
        # Opt this Linear out of the fp8 quantization step in
        # ``prepare(fp8=True)``. Set for (a) small / precision-sensitive
        # paths where quantization noise overwhelms the compute win,
        # and (b) shapes where K isn't divisible by any legal fp8
        # ``mma_k`` on the target device — e.g. Ada's only fp8 MMA is
        # ``m16n8k32_e4m3``, so K must be a multiple of 32. Callers
        # that mix fp8 and non-fp8 Linears flip this flag per-layer.
        self._fp8_skip = fp8_skip
        if bias:
            self.bias = Parameter(_zeros(out_features, dtype="bf16"))

    def prepare(self, *, shuffle: bool = False, fp8: bool = False, **kwargs) -> None:
        """Prepare weight for inference.

        ``fp8``: quantize bf16 weights to e4m3 for fp8 MMA compute.
        ``shuffle``: pre-shuffle weight for vectorized frag loads.

        Layers constructed with ``fp8_skip=True`` keep their bf16
        weights even when called with ``fp8=True``.

        Call after ``load_state_dict()``.
        """
        if fp8 and not self._fp8_skip and not getattr(self, "_fp8", False):
            self.weight.data = _quantize_to_e4m3(self.weight.data)
            self._fp8 = True

        if shuffle:
            raise RuntimeError(
                "b_shuffle is currently broken — the byte-level permutation "
                "depends on MMA fragment layout and dtype in ways that aren't "
                "correctly handled yet. Use shuffle=False for now."
            )

    _BM_MIN = 16  # minimum M for GEMM tile

    def _out_buf(self, M_eff: int):
        """Return a cached ``[M_eff, out_features]`` buffer.

        On CUDA we zero every call to pay for correctness under
        split-K winners (atomic-add epilogues accumulate onto existing
        contents). On Metal the NAX gemm fast path doesn't honor
        ``out=`` and writes to a fresh transient — Linear's cached
        buffer is unused on the hot path, so skip the per-call CPU
        memset (~5 µs / call × 96 Linear calls / forward).
        """
        cache = getattr(self, "_out_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_out_cache", cache)
        buf = cache.get(M_eff)
        if buf is None:
            from quark.runtime.tensor import QuarkTensor

            out_f = int(self.weight.data.shape[0])
            buf = QuarkTensor.zeros(M_eff, out_f, dtype=self._out_dtype)
            cache[M_eff] = buf
        elif not _IS_METAL:
            buf.zero_()
        return buf

    def forward(
        self,
        x,
        *,
        activation: str | None = None,
        bias=None,
        gate=None,
        residual=None,
        gate_groups: int = 1,
    ):
        """``y = activation(x @ weight.T + bias?)``.

        ``activation``: optional fused activation passed through to the
        GEMM kernel (``"silu"`` supported). Keeps all GEMMs owned by
        a Linear layer so output buffers stay cached here.

        ``bias``: optional call-time bias tensor shaped ``[out_features]``
        (or any shape that flattens to that — ``reshape(-1)`` at the
        caller). Used when the layer wasn't constructed with its own
        bias parameter but the caller wants to fold an external vector
        into the GEMM epilogue (e.g. ``MLPFusion`` adding ``cond @ Wc.T``
        into ``x @ Wx.T`` for free). Mutually exclusive with the
        constructor's ``bias=True`` path.

        ``gate`` + ``residual`` (both required together): fold an
        AdaGate-residual epilogue into the GEMM —
        ``y = residual + gate_bcast * (x @ weight.T)``. Mutually
        exclusive with ``bias`` (the GEMM kernel only fuses one
        epilogue at a time today). Skips the cached output buffer
        because the kernel writes the post-combine result directly,
        and skips the M-padding shortcut since the residual would
        also have to be padded — kernels using gate/residual fusion
        always have M ≥ the kernel's BM_MIN in the wired layers
        (Waypoint's tpf=128). NAX-only on Metal; the non-NAX
        ``store_acc`` epilogue handles other shapes.
        """
        import quark.functional as pcf

        fused_gate = gate is not None and residual is not None
        if fused_gate and bias is not None:
            raise ValueError("Linear.forward: gate/residual fusion is incompatible with bias")

        b_dtype = _quark_dtype(self.weight.data)

        # Activation dtype must already match the compute requirements —
        # silent casts used to live here but masked model-wiring bugs,
        # so we raise instead. The caller is responsible for feeding
        # Linear the dtype that matches its weight (or f16/bf16 for fp8
        # weights, which do the narrow-cast inside the MMA itself).
        a_dtype = _quark_dtype(x)
        is_fp8_weight = b_dtype in ("e4m3", "e5m2")

        # Linear accepts any half / fp8 activation. When a_dtype != b_dtype
        # the GEMM kernel does the cast during smem load (``compute_dtype
        # = b_dtype`` is set below). This keeps the "no silent cast" spirit
        # — the cast is explicit at the kernel boundary — while letting
        # the fp8 inference path carry e4m3 intermediaries end-to-end.
        _OK_ACTIVATION = ("f16", "bf16", "e4m3", "e5m2")
        if a_dtype not in _OK_ACTIVATION:
            raise TypeError(
                f"Linear: activation dtype {a_dtype!r} not in {_OK_ACTIVATION}. "
                f"Insert a Cast upstream."
            )

        # cuBLAS routing: the mixed-dtype bf16→e4m3 pre-cast used to
        # live here, gating on ``can_use_cublas``, to satisfy cublasLt's
        # "both operands fp8" constraint. It's now owned by
        # ``kernels/gemm/cublas_dispatch.py``: the autotuner treats
        # (bf16, e4m3) specs as cuBLAS-eligible and dispatch applies
        # the cast on-stream when cuBLAS wins the config race. PTX
        # configs still see the raw bf16 A and down-cast during the
        # smem load (``compute_dtype=b_dtype`` below).

        # Pad M when too small for the GEMM kernel's minimum tile.
        M = int(x.shape[0])
        needs_pad = M < self._BM_MIN
        if needs_pad:
            x, _ = _pad_rows(x, self._BM_MIN)
        M_eff = int(x.shape[0])

        kw = {
            "b_shuffled": getattr(self, "_shuffled", False),
            "out_dtype": self._out_dtype,
        }
        if activation is not None:
            kw["activation"] = activation
        if is_fp8_weight or a_dtype != b_dtype:
            kw["compute_dtype"] = b_dtype

        if fused_gate:
            # The fused-gate-residual epilogue writes the final post-
            # combine output, not the bare GEMM result, so we skip the
            # cached out=. M-padding is also skipped (residual must
            # match the unpadded M).
            if needs_pad:
                raise NotImplementedError(
                    "Linear.forward: gate/residual fusion does not yet "
                    "support M-padding (M < BM_MIN). Hits in production "
                    "would need a pad-aware residual layout."
                )
            return pcf.gemm(
                x,
                self.weight.data,
                gate=gate,
                residual=residual,
                gate_groups=gate_groups,
                **kw,
            )

        out = self._out_buf(M_eff)
        if self._has_bias and bias is not None:
            raise ValueError(
                "Linear: call-time bias= conflicts with a layer constructed with bias=True"
            )
        gemm_bias = self.bias.data if self._has_bias else bias
        if gemm_bias is not None:
            y = pcf.gemm(x, self.weight.data, bias=gemm_bias, out=out, **kw)
        else:
            y = pcf.gemm(x, self.weight.data, out=out, **kw)

        if needs_pad:
            return y[:M]
        return y

    def __repr__(self) -> str:
        out_f, in_f = (int(s) for s in self.weight.data.shape)
        shuffled = getattr(self, "_shuffled", False)
        return f"Linear(in={in_f}, out={out_f}, bias={self._has_bias}, shuffled={shuffled})"


class Cast(Module):
    """Stateless dtype cast with cached output + dummy-input buffers.

    The elementwise cast kernel takes ``(X, Y, Out)``; ``Y`` is a
    dummy input that the kernel ignores. Without caching, every call
    allocates a fresh ``Out`` tensor and a fresh 1-element ``Y``,
    churning the driver allocator and making each call's
    ``(X, Y, Out)`` triple a new ``input_key`` combination.

    This Module caches:
      * ``_out_cache[(shape, src_dtype, dst_dtype)]`` — output buffer
      * ``_dummy_cache[src_dtype]`` — 1-element ``Y`` of the input dtype

    Both live for the life of the Module so pointers stay stable
    across calls — the hot loop spends zero μs on cuMemAllocAsync.
    """

    def __init__(self, dtype: str):
        self._dtype = dtype

    def forward(self, x, dtype: str | None = None):
        target = dtype if dtype is not None else self._dtype
        if x.dtype == target:
            return x

        from quark.runtime.kernels import cast
        from quark.runtime.tensor import QuarkTensor

        out_cache = getattr(self, "_out_cache", None)
        if out_cache is None:
            out_cache = {}
            object.__setattr__(self, "_out_cache", out_cache)
        dummy_cache = getattr(self, "_dummy_cache", None)
        if dummy_cache is None:
            dummy_cache = {}
            object.__setattr__(self, "_dummy_cache", dummy_cache)

        key = (tuple(x.shape), x.dtype, target)
        out = out_cache.get(key)
        if out is None:
            out = QuarkTensor.empty(*x.shape, dtype=target)
            out_cache[key] = out

        dummy = dummy_cache.get(x.dtype)
        if dummy is None:
            dummy = QuarkTensor.zeros(1, dtype=x.dtype)
            dummy_cache[x.dtype] = dummy

        return cast(x, target, out=out, dummy_y=dummy)

    def __repr__(self) -> str:
        return f"Cast(dtype={self._dtype!r})"


class Patchify(Module):
    """``[B, C*H*W] → [B*Hp*Wp, d_model]`` via strided-GEMM kernel."""

    def __init__(self, C: int, d_model: int, H: int, W: int, ph: int = 2, pw: int = 2):
        self.weight = Parameter(_zeros(d_model, C * ph * pw, dtype="bf16"))
        self._C, self._H, self._W, self._ph, self._pw = C, H, W, ph, pw

    def forward(self, x):
        import quark.functional as pcf

        B = int(x.shape[0])
        # Output: [B * (H/ph) * (W/pw), d_model] in ``x.dtype``.
        d_model = int(self.weight.data.shape[0])
        out_rows = B * (self._H // self._ph) * (self._W // self._pw)
        out = _cached_out_buf(self, (out_rows, d_model), x.dtype)

        return pcf.patchify(
            x,
            self.weight.data,
            B=B,
            C=self._C,
            H=self._H,
            W_spatial=self._W,
            ph=self._ph,
            pw=self._pw,
            out=out,
        )


class Unpatchify(Module):
    """``[M, d_model] → [B, C*H*W]`` via scatter-GEMM + bias kernel."""

    def __init__(self, d_model: int, C: int, H: int, W: int, ph: int = 2, pw: int = 2):
        out_dim = C * ph * pw
        self.weight = Parameter(_zeros(out_dim, d_model, dtype="bf16"))
        self.bias = Parameter(_zeros(out_dim, dtype="bf16"))
        self._C, self._H, self._W, self._ph, self._pw = C, H, W, ph, pw

    def forward(self, x, B: int = 1):
        import quark.functional as pcf

        # Output: [B, C * H * W] in ``x.dtype``.
        out_shape = (B, self._C * self._H * self._W)
        out = _cached_out_buf(self, out_shape, x.dtype)
        return pcf.unpatchify(
            x,
            self.weight.data,
            self.bias.data,
            B=B,
            C=self._C,
            H=self._H,
            W=self._W,
            ph=self._ph,
            pw=self._pw,
            out=out,
        )


def _maybe_cached_out(module, shape, dtype):
    """Return a cached output buffer on every backend.

    Earlier this function returned ``None`` on Metal so the leaf
    modules (AdaRMSNorm / AdaGateResidual / RMSNorm / SiLU) would
    skip ``out=`` and let ``pcf.*`` route through ``queue_launch_ir``.
    That path was a measured ~3× speedup but produces token-uniform
    output for at least the AdaRMSNorm and AdaGateResidual fast
    paths in the current kernel + dispatcher combo on M5 Max
    (every spatial position converges to the same per-channel
    value). Bisected by selectively forcing classes back onto the
    cached-out path: forcing *either* AdaRMSNorm or AdaGateResidual
    alone is enough to restore correct output, which points at a
    shared queue_launch_ir bug rather than a per-kernel issue.

    Until that's tracked down, prefer correctness over throughput
    and use the cached-out / ``call_with_bindings`` path on every
    backend. ``QUARK_FORCE_FASTPATH=1`` re-enables the (broken)
    Metal fast path for benchmark experiments.
    """
    import os

    if _IS_METAL and os.environ.get("QUARK_FORCE_FASTPATH") == "1":
        return None
    return _cached_out_buf(module, shape, dtype)


class AdaRMSNorm(Module):
    """Stateless — no learnable params. Scale/bias are passed as arguments.

    y = ada_rmsnorm(x, scale, bias[, activation="silu"])
    """

    def forward(self, x, scale, bias, *, activation=None):
        import quark.functional as pcf

        out = _maybe_cached_out(self, tuple(x.shape), x.dtype)
        return pcf.ada_rmsnorm(x, scale, bias, activation=activation, out=out)


class AdaGateResidual(Module):
    """``out = x + gate * y`` with broadcast over groups. Stateless."""

    def forward(self, x, y, gate):
        import quark.functional as pcf

        out = _maybe_cached_out(self, tuple(x.shape), x.dtype)
        return pcf.ada_gate_residual(x, y, gate, out=out)


class HeadRMSNorm(Module):
    """Per-head RMSNorm on packed QKV tensor. Stateless (no params)."""

    def __init__(self, n_q_heads: int, n_kv_heads: int, Dh: int):
        self._nq, self._Hk, self._Dh = n_q_heads, n_kv_heads, Dh

    def forward(self, qkv):
        import quark.functional as pcf

        out = _maybe_cached_out(self, tuple(qkv.shape), qkv.dtype)
        return pcf.head_rmsnorm(qkv, n_q_heads=self._nq, n_kv_heads=self._Hk, Dh=self._Dh, out=out)


class RMSNorm(Module):
    """Plain RMSNorm (no learnable gain). Stateless; caches its output."""

    def forward(self, x):
        import quark.functional as pcf

        out = _maybe_cached_out(self, tuple(x.shape), x.dtype)
        return pcf.rmsnorm(x, out=out)


class Add(Module):
    """Elementwise ``out = x + y`` with a cached output buffer.

    Wraps the same-shape ``__add__`` path so the result lives in a
    persistent buffer owned by the Module — avoids the ~150 µs
    ``cuMemAllocAsync`` that ``x + y`` does by default. Also gives the
    op a name in the profile so it shows up as a leaf instead of being
    hidden inside a parent's timing.
    """

    def forward(self, x, y):
        from quark.runtime.kernels import elemwise_binary

        out = _maybe_cached_out(self, tuple(x.shape), x.dtype)
        return elemwise_binary("add", x, y, out=out)


class SiLU(Module):
    """Standalone elementwise SiLU with a cached output buffer.

    Prefer the fused ``Linear(..., activation="silu")`` form when
    applicable; this module covers the (rarer) case where SiLU stands
    alone between non-Linear ops.
    """

    def forward(self, x):
        import quark.functional as pcf

        out = _maybe_cached_out(self, tuple(x.shape), x.dtype)
        return pcf.silu(x, out=out)


class EulerStep(Module):
    """Fused ``out = x + dsig * v`` (single-element ``dsig`` tensor).

    One kernel launch with a cached ``[*shape, dtype]`` output buffer.
    """

    def forward(self, x, v, dsig):
        import quark.functional as pcf

        out = _maybe_cached_out(self, tuple(x.shape), x.dtype)
        return pcf.euler_step(x, v, dsig, out=out)


class MLP(Module):
    """Two-layer MLP: ``fc2(silu(fc1(x)))``.

    Metal: silu fused into fc1's NAX-GEMM epilogue (in-register on the
    F32 accumulators before the bf16 store). Saves one kernel launch
    plus the bf16-read/bf16-write of the standalone silu kernel
    (~256 KB per layer for the 128×8192 hidden activation). Other
    backends keep the unfused fc1 → silu → fc2 chain because
    cublasLtMatmul has no silu epilogue.

    ``hidden_out_dtype``: optional override for fc1's output dtype.
    Default tracks ``out_dtype``. fc2's output dtype is always
    ``out_dtype`` so the return value matches the residual stream.
    """

    def __init__(
        self,
        d_in: int,
        d_mid: int,
        d_out: int,
        out_dtype: str = "f16",
        *,
        hidden_out_dtype: str | None = None,
    ):
        mid_dt = hidden_out_dtype if hidden_out_dtype is not None else out_dtype
        self.fc1 = Linear(d_in, d_mid, out_dtype=mid_dt)
        self.fc2 = Linear(d_mid, d_out, out_dtype=out_dtype)
        # Kept for the cuBLAS path where Linear can't fuse silu.
        self.silu = SiLU()

    def forward(self, x, *, gate=None, residual=None, gate_groups: int = 1):
        """``y = fc2(silu(fc1(x)))``, optionally folding an
        AdaGate-residual epilogue into ``fc2``: when ``gate`` and
        ``residual`` are both provided, returns
        ``residual + gate_bcast * fc2(silu(fc1(x)))`` in one fc2 dispatch.
        """
        if _IS_METAL:
            return self.fc2(
                self.fc1(x, activation="silu"),
                gate=gate,
                residual=residual,
                gate_groups=gate_groups,
            )
        # Non-Metal: cuBLAS has no silu epilogue so the fc1→silu→fc2
        # chain stays unfused, but fc2's gate/residual still routes
        # through Linear → pcf.gemm → GemmKernel.build()'s
        # ``qk.store_acc(gate=, residual=)`` epilogue, which folds the
        # AdaGate-residual into the GEMM tail (scalar gmem stores —
        # the NAX-fused ``StoreMatrixGateResidualOp`` is Metal-only,
        # but the unfused ``store_acc`` path doesn't need it). Saves
        # the standalone ``AdaGateResidualKernel`` dispatch + the
        # bf16 round-trip on the post-MLP residual.
        return self.fc2(
            self.silu(self.fc1(x)),
            gate=gate,
            residual=residual,
            gate_groups=gate_groups,
        )


class ControllerInputEmbedding(Module):
    """``cat([mouse, button, scroll]) → MLP → [1, d_model]``."""

    def __init__(
        self,
        n_buttons: int,
        d_model: int,
        mlp_ratio: int = 4,
        out_dtype: str = "f16",
        *,
        fp8_skip: bool = False,
    ):
        d_mid = d_model * mlp_ratio
        raw_in = n_buttons + 3
        self._raw_in = raw_in
        # fc1's K is the padded-controller-input width, which is rounded
        # up to 16 — not guaranteed to be a multiple of 32. On Ada the
        # only fp8 MMA is ``m16n8k32_e4m3``, which needs K%32==0. Default
        # to skipping fp8 for this MLP (one tensor per frame, small
        # precision-sensitive path — the fp8 win wouldn't show up
        # anyway). Callers can override.
        self._padded_in = ((raw_in + 15) // 16) * 16
        self.fc1 = Linear(self._padded_in, d_mid, out_dtype=out_dtype, fp8_skip=fp8_skip)
        self.fc2 = Linear(d_mid, d_model, out_dtype=out_dtype, fp8_skip=fp8_skip)
        # Silu standalone so fc1 is a plain GEMM that cuBLAS takes. The
        # fused-silu custom kernel blocks cuBLAS routing entirely, and
        # cuBLAS wins by enough on every shape we care about that the
        # extra silu launch doesn't dent the budget.
        self.silu = SiLU()

    def forward(self, ctrl_input):
        """``ctrl_input``: [1, padded_in] tensor."""
        # Pad M to BM_MIN to avoid the fc1 Linear padding path.
        x, M = _pad_rows(ctrl_input, Linear._BM_MIN)
        h = self.silu(self.fc1(x))
        out = self.fc2(h)
        if M < Linear._BM_MIN:
            return out[:M]
        return out


class MLPFusion(Module):
    """Fuse conditioning into tokens: ``y = fc2(silu(x @ Wx + cond @ Wc))``."""

    def __init__(self, d_model: int, out_dtype: str = "f16"):
        self.fc1_x = Linear(d_model, d_model, out_dtype=out_dtype)
        self.fc1_c = Linear(d_model, d_model, out_dtype=out_dtype)
        self.fc2 = Linear(d_model, d_model, out_dtype=out_dtype)
        self.silu = SiLU()

    def forward(self, x, cond):
        """``x``: [M, d], ``cond``: [1, d] (row-broadcast as GEMM bias).

        Math: ``h = silu(x @ Wx.T + cond @ Wc.T); y = h @ fc2.T``.

        ``fc1_x`` folds the cond projection in as a ``[d]`` bias vector
        — cublasLt's BIAS epilogue row-broadcasts it for free — and
        silu runs as a standalone kernel after the GEMM. This keeps
        the three logical ops (fc1_x, add cond, silu) compressed to
        two launches while staying on the cuBLAS path.
        """
        h_c = self.fc1_c(cond).reshape(-1)  # [d]
        return self.fc2(self.silu(self.fc1_x(x, bias=h_c)))


class ValueResidualPacked(Module):
    """Lerp V columns of packed QKV against first-layer QKV."""

    def __init__(self, v_col_offset: int, v_width: int, lamb_init: float = 0.5):
        self.lamb = Parameter(_tensor([lamb_init], dtype="f32"))
        self._vco, self._vw = v_col_offset, v_width

    def forward(self, qkv_curr, qkv_first):
        import quark.functional as pcf

        out = _cached_out_buf(self, tuple(qkv_curr.shape), qkv_curr.dtype)
        return pcf.value_residual_packed(
            qkv_curr,
            qkv_first,
            self.lamb.data,
            v_col_offset=self._vco,
            v_width=self._vw,
            out=out,
        )


class KVCacheUpdate(Module):
    """Per-layer KV cache state + update kernel.

    Holds the cache buffers as non-parameter state. ``forward()``
    calls ``pcf.kv_cache_update`` with the right kwargs.
    """

    def __init__(
        self,
        *,
        B: int,
        n_kv_heads: int,
        n_q_heads: int,
        H_spatial: int,
        W_spatial: int,
        Dh: int,
        num_buckets: int,
        pinned_dilation: int,
        packed_qkv: bool,
        rope_n_frames: int = 1,  # kept for compat but unused (inline RoPE)
        dtype: str = "bf16",
        quilt_factor: int = 1,
        quilt_offset: int = 0,
    ):
        tpf = H_spatial * W_spatial
        if tpf % quilt_factor != 0:
            raise ValueError(
                f"KVCacheUpdate: tpf={tpf} not divisible by quilt_factor={quilt_factor}"
            )
        tpf_cached = tpf // quilt_factor
        cap = num_buckets * tpf_cached + tpf_cached
        L = num_buckets * tpf_cached

        self.K_cache = _zeros(B * n_kv_heads * cap, Dh, dtype=dtype)
        self.Vt_cache = _zeros(B * n_kv_heads * Dh, cap, dtype=dtype)
        self.segments = _tensor([0, 0, L, tpf_cached, 0, 0], dtype="s32")
        self.n_segments = _tensor([2], dtype="s32")
        self.frame_t = _zeros(1, dtype="s32")
        self.frozen = _zeros(1, dtype="s32")

        self._kw = dict(
            B=B,
            n_kv_heads=n_kv_heads,
            H_spatial=H_spatial,
            W_spatial=W_spatial,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            packed_qkv=packed_qkv,
            n_q_heads=n_q_heads,
            quilt_factor=quilt_factor,
            quilt_offset=quilt_offset,
        )

    def forward(self, qkv, frame_t=None, frozen=False):
        """``frozen``: skip ring writes (denoise mode)."""
        import quark.functional as pcf

        ft = frame_t if frame_t is not None else self.frame_t

        # Set frozen flag on device tensor.
        _set_s32(self.frozen, 1 if frozen else 0)
        frozen_buf = self.frozen

        # In-place RMW. On Metal the launcher routes role="out" tensors
        # back to the caller's pinned QuarkTensor handle when they appear
        # in ``provided`` (see ``persistent_outs`` in
        # ``functional._dispatch.call_with_bindings`` →
        # ``Launcher._launch_metal``), so the kernel's writes land on the
        # same buffer self.K_cache / self.Vt_cache / self.segments /
        # self.n_segments hold. No reassignment needed; the in-place
        # contract here mirrors CUDA's.
        pcf.kv_cache_update(
            qkv,
            qkv,
            ft,
            frozen_buf,
            self.Vt_cache,
            self.segments,
            self.n_segments,
            self.K_cache,
            **self._kw,
        )

    def set_frame_t(self, value: int, stream: int = 0) -> None:
        """Write a Python int into the frame_t device tensor.

        Call BEFORE graph replay (not during capture). Pass the shared
        graph stream so the write is ordered before the replay.
        """
        from quark.runtime.tensor import QuarkTensor

        if isinstance(self.frame_t, QuarkTensor):
            from quark.runtime.sync import _IS_OCL

            if _IS_METAL or _IS_OCL:
                # Host-coherent (Metal pool / OCL USM-shared) backing —
                # serialize with any pending kernel writes, then memmove
                # the s32 pattern. ``CudaRuntime`` is unavailable here.
                import ctypes

                from quark.runtime.sync import synchronize

                synchronize()
                bits = (int(value) & 0xFFFFFFFF).to_bytes(4, "little")
                ctypes.memmove(self.frame_t.data_ptr(), bits, 4)
                return
            from quark.runtime.cuda import CudaRuntime

            CudaRuntime.instance().memset_d32(
                self.frame_t.data_ptr(), value & 0xFFFFFFFF, 1, stream=stream
            )
        else:
            self.frame_t = _tensor([value], dtype="s32")

    def reset(self, stream: int = 0) -> None:
        """Zero the ring buffers and rewind frame_t. Async on ``stream``."""
        for buf in (self.K_cache, self.Vt_cache):
            if hasattr(buf, "zero_"):
                # CUDA QuarkTensor — async device-side memset.
                buf.zero_()
            else:
                # Metal numpy carrier (tagged ndarray). Fill in place
                # via numpy; the kernel-dispatch path picks it up on
                # the next forward without a buffer-cache invalidation.
                buf.fill(0)
        self.set_frame_t(0, stream=stream)


class OwlAttn(Module):
    """Owl attention module — carries per-layer config kwargs."""

    def __init__(
        self,
        *,
        B: int,
        n_kv_heads: int,
        gqa_ratio: int,
        H_spatial: int,
        W_spatial: int,
        num_buckets: int,
        pinned_dilation: int,
        packed_qkv: bool,
        rope_n_frames: int = 1,  # kept for compat but unused (inline RoPE)
        compute_dtype: str | None = "e4m3",
        quilt_factor: int = 1,
        quilt_offset: int = 0,
    ):
        self._kw = dict(
            B=B,
            n_kv_heads=n_kv_heads,
            gqa_ratio=gqa_ratio,
            H_spatial=H_spatial,
            W_spatial=W_spatial,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            packed_qkv=packed_qkv,
            quilt_factor=quilt_factor,
            quilt_offset=quilt_offset,
        )
        # MMA / smem compute dtype. ``"e4m3"`` drives both GEMMs in fp8
        # regardless of Q's arrival dtype (caller must also pin the KV
        # cache to e4m3 for this to work). ``None`` inherits from Q —
        # bf16 input → bf16 MMAs. Set to None by the bf16 fallback path
        # in Waypoint15 when ``cfg.quant.attn_compute = "bf16"``.
        self._compute_dtype = compute_dtype

    def forward(self, qkv, kv_cache, frame_t=None):
        """``kv_cache``: a ``KVCacheUpdate`` module (reads its buffers)."""
        import quark.functional as pcf

        # Derive the output shape from the qkv input so we can cache
        # the ``output`` buffer instead of auto-allocating it per call
        # (~150 µs ``cuMemAllocAsync`` on Blackwell). With packed_qkv:
        #   qkv  : [B*tpf, (n_q + 2*n_kv) * Dh]
        #   out  : [B*tpf, n_q * Dh]
        q_cols = int(qkv.shape[1])
        n_kv = self._kw["n_kv_heads"]
        gqa = self._kw["gqa_ratio"]
        n_q = n_kv * gqa
        Dh = q_cols // (n_q + 2 * n_kv)
        out_shape = (int(qkv.shape[0]), n_q * Dh)
        out = _cached_out_buf(self, out_shape, qkv.dtype)

        return pcf.owl_attn(
            qkv,
            kv_cache.K_cache,
            kv_cache.Vt_cache,
            kv_cache.segments,
            kv_cache.n_segments,
            frame_t=frame_t,
            out=out,
            compute_dtype=self._compute_dtype,
            **self._kw,
        )
