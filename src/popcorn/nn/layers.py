"""
EXEMPT FROM 500-LINE RULE: all leaf module classes (Linear, Patchify,
KVCacheUpdate, OwlAttn, etc.) form a single cohesive API. Splitting
would scatter the module hierarchy across files with no readability
benefit.

"""

from __future__ import annotations

from popcorn.nn.module import (
    Module,
    Parameter,
    _quantize_to_e4m3,
    _tensor,
    _zeros,
)


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
        from popcorn.runtime.tensor import PopcornTensor

        buf = PopcornTensor.empty(*shape, dtype=dtype)
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

    def _cast_to_fp8(self, x, fp8_dtype: str):
        """Narrow ``x`` to e4m3/e5m2 via ``pcf.quantize_e4m3``.

        Used by ``forward`` to land on the cuBLAS fp8 matmul path when
        the weight is fp8 and ``x`` arrives in bf16/f16. The custom
        GEMM kernel's in-kernel ``compute_dtype`` cast is still
        available as a fallback (and is what the b_shuffle / fused-
        activation paths use), but cuBLAS wants both operands fp8 so
        we materialize the cast here for the simple Linear path.
        """
        if x.dtype == fp8_dtype:
            return x
        import popcorn.functional as pcf

        return pcf.quantize_e4m3(x)

    def _out_buf(self, M_eff: int):
        """Return a cached, pre-zeroed ``[M_eff, out_features]`` buffer.

        Zeroing every call is a ~5 µs device memset that pays for
        correctness under any split-K winner (atomic-add epilogue
        accumulates onto existing contents). The real per-call cost
        we're avoiding here is the ``cuMemAllocAsync`` + free — the
        cache makes the alloc happen once per M and the zero is just
        cheap bookkeeping.
        """
        cache = getattr(self, "_out_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_out_cache", cache)
        buf = cache.get(M_eff)
        if buf is None:
            from popcorn.runtime.tensor import PopcornTensor

            out_f = int(self.weight.data.shape[0])
            buf = PopcornTensor.zeros(M_eff, out_f, dtype=self._out_dtype)
            cache[M_eff] = buf
        else:
            # Call the tensor method directly. ``PT.zero_`` routes
            # through ``PT._is_mx(x)`` which does ``import mlx.core``
            # every call — on CUDA (no mlx installed) the ImportError
            # raise + catch is ~140 µs per call. Tensor.zero_() is
            # the raw ``cuMemsetD8Async`` in ~4 µs.
            buf.zero_()
        return buf

    def forward(self, x, *, activation: str | None = None, bias=None):
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
        """
        import popcorn.functional as pcf

        b_dtype = (
            self.weight.data.dtype
            if isinstance(self.weight.data.dtype, str)
            else str(self.weight.data.dtype)
        )

        # Activation dtype must already match the compute requirements —
        # silent casts used to live here but masked model-wiring bugs,
        # so we raise instead. The caller is responsible for feeding
        # Linear the dtype that matches its weight (or f16/bf16 for fp8
        # weights, which do the narrow-cast inside the MMA itself).
        a_dtype = x.dtype if isinstance(x.dtype, str) else str(x.dtype)
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

        # cuBLAS routing: cublasLt's fp8 matmul only takes fp8 A *and*
        # B — mixed bf16×e4m3 stays on the custom kernel. To land the
        # fp8-weight Linear on cuBLAS, narrow the activation to e4m3
        # here (cached buffer, zero churn). Scalar shuffle / fused-
        # activation / bias paths skip the cast and keep the custom
        # kernel.
        can_use_cublas = (
            activation is None
            and bias is None
            and not self._has_bias
            and not getattr(self, "_shuffled", False)
        )
        if is_fp8_weight and a_dtype not in ("e4m3", "e5m2") and can_use_cublas:
            x = self._cast_to_fp8(x, b_dtype)
            a_dtype = b_dtype

        # Pad M when too small for the GEMM kernel's minimum tile.
        M = int(x.shape[0])
        needs_pad = M < self._BM_MIN
        if needs_pad:
            from popcorn.runtime.tensor import PopcornTensor

            pad = PopcornTensor.zeros(self._BM_MIN - M, int(x.shape[1]), dtype=x.dtype)
            x = PopcornTensor.cat([x, pad], dim=0)
        M_eff = int(x.shape[0])

        kw = {
            "b_shuffled": getattr(self, "_shuffled", False),
            "out_dtype": self._out_dtype,
        }
        if activation is not None:
            kw["activation"] = activation
        if is_fp8_weight or a_dtype != b_dtype:
            kw["compute_dtype"] = b_dtype

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

        from popcorn.runtime.kernels import cast
        from popcorn.runtime.tensor import PopcornTensor

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
            out = PopcornTensor.empty(*x.shape, dtype=target)
            out_cache[key] = out

        dummy = dummy_cache.get(x.dtype)
        if dummy is None:
            dummy = PopcornTensor.zeros(1, dtype=x.dtype)
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
        import popcorn.functional as pcf

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
        import popcorn.functional as pcf

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


class AdaRMSNorm(Module):
    """Stateless — no learnable params. Scale/bias are passed as arguments.

    y = ada_rmsnorm(x, scale, bias[, activation="silu"])
    """

    def forward(self, x, scale, bias, *, activation=None):
        import popcorn.functional as pcf

        out = _cached_out_buf(self, tuple(x.shape), x.dtype)
        return pcf.ada_rmsnorm(x, scale, bias, activation=activation, out=out)


class AdaGateResidual(Module):
    """``out = x + gate * y`` with broadcast over groups. Stateless."""

    def forward(self, x, y, gate):
        import popcorn.functional as pcf

        out = _cached_out_buf(self, tuple(x.shape), x.dtype)
        return pcf.ada_gate_residual(x, y, gate, out=out)


class HeadRMSNorm(Module):
    """Per-head RMSNorm on packed QKV tensor. Stateless (no params)."""

    def __init__(self, n_q_heads: int, n_kv_heads: int, Dh: int):
        self._nq, self._Hk, self._Dh = n_q_heads, n_kv_heads, Dh

    def forward(self, qkv):
        import popcorn.functional as pcf

        out = _cached_out_buf(self, tuple(qkv.shape), qkv.dtype)
        return pcf.head_rmsnorm(qkv, n_q_heads=self._nq, n_kv_heads=self._Hk, Dh=self._Dh, out=out)


class RMSNorm(Module):
    """Plain RMSNorm (no learnable gain). Stateless; caches its output."""

    def forward(self, x):
        import popcorn.functional as pcf

        out = _cached_out_buf(self, tuple(x.shape), x.dtype)
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
        from popcorn.runtime.kernels import elemwise_binary

        out = _cached_out_buf(self, tuple(x.shape), x.dtype)
        return elemwise_binary("add", x, y, out=out)


class SiLU(Module):
    """Standalone elementwise SiLU with a cached output buffer.

    Prefer the fused ``Linear(..., activation="silu")`` form when
    applicable; this module covers the (rarer) case where SiLU stands
    alone between non-Linear ops.
    """

    def forward(self, x):
        import popcorn.functional as pcf

        out = _cached_out_buf(self, tuple(x.shape), x.dtype)
        return pcf.silu(x, out=out)


class EulerStep(Module):
    """Fused ``out = x + dsig * v`` (single-element ``dsig`` tensor).

    One kernel launch with a cached ``[*shape, dtype]`` output buffer.
    """

    def forward(self, x, v, dsig):
        import popcorn.functional as pcf

        out = _cached_out_buf(self, tuple(x.shape), x.dtype)
        return pcf.euler_step(x, v, dsig, out=out)


class MLP(Module):
    """Two-layer MLP with fused SiLU on fc1: ``fc2(silu(fc1(x)))``.

    ``hidden_out_dtype``: optional override for fc1's output dtype, so
    the mid activation can stay in e4m3 when both weights are fp8. When
    ``None``, falls back to ``out_dtype`` (original behavior). fc2's
    output dtype is always ``out_dtype`` so the MLP's return value
    matches the residual stream.
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

    def forward(self, x):
        return self.fc2(self.fc1(x, activation="silu"))


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

    def forward(self, ctrl_input):
        """``ctrl_input``: [1, padded_in] tensor."""
        from popcorn.runtime.tensor import PopcornTensor

        # Pad M to BM_MIN to avoid the fc1 Linear padding path.
        M = int(ctrl_input.shape[0])
        BM_MIN = Linear._BM_MIN
        if M < BM_MIN:
            pad = PopcornTensor.zeros(BM_MIN - M, int(ctrl_input.shape[1]), dtype=ctrl_input.dtype)
            x = PopcornTensor.cat([ctrl_input, pad], dim=0)
        else:
            x = ctrl_input

        h = self.fc1(x, activation="silu")
        out = self.fc2(h)

        if M < BM_MIN:
            return out[:M]
        return out


class MLPFusion(Module):
    """Fuse conditioning into tokens: ``y = fc2(silu(x @ Wx + cond @ Wc))``."""

    def __init__(self, d_model: int, out_dtype: str = "f16"):
        self.fc1_x = Linear(d_model, d_model, out_dtype=out_dtype)
        self.fc1_c = Linear(d_model, d_model, out_dtype=out_dtype)
        self.fc2 = Linear(d_model, d_model, out_dtype=out_dtype)

    def forward(self, x, cond):
        """``x``: [M, d], ``cond``: [1, d] (broadcast-added via GEMM bias).

        Math: ``h = silu(x @ Wx.T + cond @ Wc.T); y = h @ fc2.T``.

        Fused so ``fc1_x``'s GEMM epilogue does ``silu(A @ B.T + bias)``
        with ``bias = cond @ Wc.T`` as the ``[d]`` bias vector — the
        broadcast-over-M is free inside the epilogue, and silu is fused
        too. Collapses what used to be three extra kernel launches
        (broadcast-cat + elementwise add + silu) into one.
        """
        h_c = self.fc1_c(cond).reshape(-1)  # [d]
        return self.fc2(self.fc1_x(x, bias=h_c, activation="silu"))


class ValueResidualPacked(Module):
    """Lerp V columns of packed QKV against first-layer QKV."""

    def __init__(self, v_col_offset: int, v_width: int, lamb_init: float = 0.5):
        self.lamb = Parameter(_tensor([lamb_init], dtype="f32"))
        self._vco, self._vw = v_col_offset, v_width

    def forward(self, qkv_curr, qkv_first):
        import popcorn.functional as pcf

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
    ):
        tpf = H_spatial * W_spatial
        cap = num_buckets * tpf + tpf
        L = num_buckets * tpf

        self.K_cache = _zeros(B * n_kv_heads * cap, Dh, dtype=dtype)
        self.Vt_cache = _zeros(B * n_kv_heads * Dh, cap, dtype=dtype)
        self.segments = _tensor([0, 0, L, tpf, 0, 0], dtype="s32")
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
        )

    def forward(self, qkv, frame_t=None, frozen=False):
        """``frozen``: skip ring writes (denoise mode)."""
        import popcorn.functional as pcf

        ft = frame_t if frame_t is not None else self.frame_t

        # Set frozen flag on device tensor.
        if frozen:
            from popcorn.runtime.cuda import CudaRuntime

            CudaRuntime.instance().memset_d32(self.frozen.data_ptr(), 1, 1, stream=0)
        else:
            from popcorn.runtime.cuda import CudaRuntime

            CudaRuntime.instance().memset_d32(self.frozen.data_ptr(), 0, 1, stream=0)

        # On CUDA the kernel mutates buffers in-place. Don't reassign
        # from the return values — they may be stale copies if
        # prepare_launch_tensors created intermediates.
        pcf.kv_cache_update(
            qkv,
            qkv,
            ft,
            self.frozen,
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
        from popcorn.runtime.tensor import PopcornTensor

        if isinstance(self.frame_t, PopcornTensor):
            from popcorn.runtime.cuda import CudaRuntime

            CudaRuntime.instance().memset_d32(
                self.frame_t.data_ptr(), value & 0xFFFFFFFF, 1, stream=stream
            )
        else:
            self.frame_t = _tensor([value], dtype="s32")


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
        )

    def forward(self, qkv, kv_cache, frame_t=None):
        """``kv_cache``: a ``KVCacheUpdate`` module (reads its buffers)."""
        import popcorn.functional as pcf

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
            compute_dtype="e4m3",
            **self._kw,
        )
