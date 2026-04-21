"""Isolate per-call overhead inside ``nn.Linear`` → ``pcf.gemm`` dispatch.

Autotune's bench timer says each ``gemm(M=512, N=2048, K=2048, bf16)``
takes ~20 µs on the GPU. The model profile says each Linear call takes
~400 µs. This 20× gap has to be host-side overhead somewhere on the
path:

    Linear.forward →
    pcf.gemm → _gemm_impl →
    call_with_bindings →
        cls.spec_from_tensors  (GemmSpec construction)
        _autotune.lookup_or_search
            _make_key
                spec_fingerprint (tuple of every field value)
                source_hash (cached)
                device.fingerprint() (str format)
            _hot dict lookup (hashes the key tuple)
        launcher.compile (dict lookup on (cls, spec, config))
        kernel.prepare_launch_tensors
        [b for b in pspec.buffers] (list-build)
        compiled.launch
            _check_dtype_popcorn × N
            _check_contiguous × N
            data_ptr() × N
            pack_scalars
            driver.launch

This script builds a Linear-equivalent workload on a Waypoint-shaped
problem and measures each layer by itself, so you can see which
breakdown explains the ~380 µs of lost time.

Usage:
    python scripts/debug_linear_overhead.py

Everything stays on device. CUDA events give ns-precision GPU time;
perf_counter gives host-side dispatch time. Run on a quiet GPU — a
busy stream will inflate the host-side numbers with driver-serialization
stalls.
"""

from __future__ import annotations

import gc
import time

import popcorn.functional as pcf
import popcorn.nn as nn
from popcorn.functional._dispatch import call_with_bindings, launcher
from popcorn.kernels import get as get_kernel
from popcorn.runtime.cuda import CudaRuntime
from popcorn.runtime.tensor import PopcornTensor


# ---------------------------------------------------------------
# Config — match a typical Waypoint transformer block Linear.
# ---------------------------------------------------------------
M = 512  # 16 spatial tokens × 32 = 512
K = 2048  # d_model
N = 2048  # out_features

N_ITERS = 2000
N_WARMUP = 200


def _sync():
    CudaRuntime.instance().stream_synchronize(0)


def _gpu_time_us(fn, *, iters: int = N_ITERS, warmup: int = N_WARMUP) -> float:
    """Time ``fn`` on the GPU stream. Returns microseconds per call."""
    rt = CudaRuntime.instance()
    for _ in range(warmup):
        fn()
    _sync()
    start = rt.event_create()
    end = rt.event_create()
    rt.event_record(start, 0)
    for _ in range(iters):
        fn()
    rt.event_record(end, 0)
    rt.event_synchronize(end)
    total_ms = rt.event_elapsed_time(start, end)
    rt.event_destroy(start)
    rt.event_destroy(end)
    return total_ms * 1000 / iters


def _cpu_time_us(fn, *, iters: int = N_ITERS, warmup: int = N_WARMUP) -> float:
    """Host-side wall-clock. Includes whatever time the call spends
    waiting on the driver before returning to Python."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed * 1e6 / iters


def _sec(title: str):
    print()
    print("─" * 60)
    print(title)
    print("─" * 60)


def main():
    print(f"Linear overhead probe — M={M} K={K} N={N}, bf16")
    print(f"{N_ITERS} iters per measurement ({N_WARMUP} warmup)")

    # ── Setup ──
    A = PopcornTensor.randn(M, K, dtype="bf16")
    B = PopcornTensor.randn(N, K, dtype="bf16")
    lin = nn.Linear(K, N, out_dtype="bf16")
    # Point Linear at real B to match the shape (weight is allocated
    # inside, but we can overwrite it).
    lin.weight.data = B
    out_buf = PopcornTensor.zeros(M, N, dtype="bf16")

    # Prime everything (autotune lookup + compile + first launch).
    _ = lin(A)
    _sync()

    # ── 1. Full Linear.forward ──
    _sec("1. Full Linear.forward  (baseline)")
    t_full = _cpu_time_us(lambda: lin(A))
    t_full_gpu = _gpu_time_us(lambda: lin(A))
    print(f"  host     : {t_full:7.1f} µs")
    print(f"  gpu event: {t_full_gpu:7.1f} µs")

    # ── 2. pcf.gemm direct (skip Linear Python) ──
    _sec("2. pcf.gemm(A, B, out=buf)  (skip Linear Python)")
    t_gemm = _cpu_time_us(lambda: pcf.gemm(A, B, out=out_buf, out_dtype="bf16"))
    t_gemm_gpu = _gpu_time_us(lambda: pcf.gemm(A, B, out=out_buf, out_dtype="bf16"))
    print(f"  host     : {t_gemm:7.1f} µs")
    print(f"  gpu event: {t_gemm_gpu:7.1f} µs")

    # ── 3. Prebuild everything → pure launch ──
    _sec("3. compiled.launch(buffers=[...])  (bare launch)")
    gemm_cls = get_kernel("gemm")
    spec = gemm_cls.spec_from_tensors(A, B, out_dtype="bf16")
    lc = launcher()
    config = lc._autotune.lookup_or_search(gemm_cls, spec)
    compiled = lc.compile(gemm_cls, spec, config)

    # Dummy bias for has_bias=False GEMM (shape from the kernel manifest).
    decls = {d.name: d for d in gemm_cls.TENSORS}
    from popcorn.functional._dispatch import alloc_from_decl

    bias_dummy = alloc_from_decl(decls["Bias"], spec, config, like=A)

    pspec = compiled.param_spec
    full = {"A": A, "B": B, "Bias": bias_dummy, "Out": out_buf}
    full = compiled.kernel.prepare_launch_tensors(full)
    prebuilt_buffers = [full[b.name] for b in pspec.buffers]

    t_launch = _cpu_time_us(lambda: compiled.launch(buffers=prebuilt_buffers))
    t_launch_gpu = _gpu_time_us(lambda: compiled.launch(buffers=prebuilt_buffers))
    print(f"  host     : {t_launch:7.1f} µs")
    print(f"  gpu event: {t_launch_gpu:7.1f} µs")

    # ── 4. Just the launcher.compile cache lookup ──
    _sec("4. launcher.compile(cls, spec, config)  (just the dict hit)")
    t_compile = _cpu_time_us(lambda: lc.compile(gemm_cls, spec, config))
    print(f"  host     : {t_compile:7.1f} µs")

    # ── 5. Just the autotune.lookup hot-dict path ──
    _sec("5. autotune.lookup(cls, spec)  (spec_fingerprint + dict lookup)")
    t_lookup = _cpu_time_us(lambda: lc._autotune.lookup(gemm_cls, spec))
    print(f"  host     : {t_lookup:7.1f} µs")

    # ── 6. spec_fingerprint + make_key alone ──
    _sec("6. make_key(cls, spec, dev_fp)  (fingerprinting only)")
    from popcorn.autotune.io import make_key

    dev_fp = lc.device.fingerprint()
    t_key = _cpu_time_us(lambda: make_key(gemm_cls, spec, dev_fp))
    print(f"  host     : {t_key:7.1f} µs")

    _sec("6b. device.fingerprint()")
    t_devfp = _cpu_time_us(lambda: lc.device.fingerprint())
    print(f"  host     : {t_devfp:7.1f} µs")

    # ── 7. GemmSpec construction ──
    _sec("7. GemmSpec.spec_from_tensors(A, B, ...)")
    t_spec = _cpu_time_us(lambda: gemm_cls.spec_from_tensors(A, B, out_dtype="bf16"))
    print(f"  host     : {t_spec:7.1f} µs")

    # ── 8. Full call_with_bindings ──
    _sec("8. call_with_bindings(...)  (the full dispatch wrapper)")
    provided = {"A": A, "B": B, "Out": out_buf}

    def _cwb():
        call_with_bindings(
            gemm_cls,
            spec,
            provided=provided,
            auto_alloc=("Bias",),
            like=A,
        )

    t_cwb = _cpu_time_us(_cwb)
    print(f"  host     : {t_cwb:7.1f} µs")

    # ── 9. Linear._out_buf cache hit + zero_ ──
    _sec("9. Linear._out_buf(M)  (zeros a [M, N] buffer via memset)")
    t_outbuf = _cpu_time_us(lambda: lin._out_buf(M))
    print(f"  host     : {t_outbuf:7.1f} µs")

    # ── 9a. Break _out_buf apart line by line ──
    _sec("9a. inside _out_buf — individual steps")
    # Prime the cache so we're always on the hit path.
    _ = lin._out_buf(M)
    cache = lin._out_cache
    buf = cache[M]

    t_getattr = _cpu_time_us(lambda: getattr(lin, "_out_cache", None))
    print(f"  getattr(self, '_out_cache')        : {t_getattr:7.1f} µs")

    t_cache_get = _cpu_time_us(lambda: cache.get(M))
    print(f"  cache.get(M)                       : {t_cache_get:7.1f} µs")

    from popcorn.backend import PT  # noqa — local reference
    t_pt_zero = _cpu_time_us(lambda: PT.zero_(buf))
    print(f"  PT.zero_(buf)                      : {t_pt_zero:7.1f} µs")

    t_buf_zero = _cpu_time_us(lambda: buf.zero_())
    print(f"  buf.zero_()  (skip PT dispatch)    : {t_buf_zero:7.1f} µs")

    rt = CudaRuntime.instance()
    ptr = buf.data_ptr()
    from popcorn.runtime.tensor import PC_BYTES
    nbytes = buf.numel() * PC_BYTES[buf.dtype]
    t_memset_raw = _cpu_time_us(lambda: rt.memset_d8(ptr, 0, nbytes))
    print(f"  rt.memset_d8(ptr, 0, {nbytes:>7}) : {t_memset_raw:7.1f} µs")

    # ── 9b. memset size sweep — is this fixed overhead or bandwidth? ──
    _sec("9b. raw cuMemsetD8Async at different sizes")
    for size in (128, 2 * 1024, 64 * 1024, 2 * 1024 * 1024):
        tmp = PopcornTensor.zeros(size, dtype="u8")
        tptr = tmp.data_ptr()
        tlen = size
        t = _cpu_time_us(lambda: rt.memset_d8(tptr, 0, tlen))
        print(f"  {size:>10} B : {t:7.1f} µs")

    # ── 9d. Manual inline of _out_buf body ──
    # If this matches 146 µs, the cost is in the body's stream/queue
    # behavior, not method dispatch. If it matches ~5 µs, method
    # resolution on the Module class is eating the time.
    _sec("9d. manual inline of _out_buf body (same statements, no method)")

    from popcorn.backend import PT as _PT_ref

    def _manual_out_buf():
        c = getattr(lin, "_out_cache", None)
        b = c.get(M)
        _PT_ref.zero_(b)
        return b

    t_manual = _cpu_time_us(_manual_out_buf)
    print(f"  manual_out_buf()                   : {t_manual:7.1f} µs")

    # Same body but without the fetch-from-cache (buf is closure-captured).
    _buf_closure = cache[M]

    def _only_zero():
        _PT_ref.zero_(_buf_closure)
        return _buf_closure

    t_only_zero = _cpu_time_us(_only_zero)
    print(f"  only PT.zero_(closure_buf)         : {t_only_zero:7.1f} µs")

    # And the tightest possible: buf.zero_() via captured reference.
    def _only_tensor_zero():
        _buf_closure.zero_()

    t_tensor_zero = _cpu_time_us(_only_tensor_zero)
    print(f"  only buf.zero_() (closure)         : {t_tensor_zero:7.1f} µs")

    # ── 9c. memset on a dedicated non-default stream ──
    _sec("9c. memset on a freshly-created non-default stream")
    stream = rt.stream_create()
    tmp = PopcornTensor.zeros(M, N, dtype="bf16")
    tptr = tmp.data_ptr()
    tlen = tmp.numel() * PC_BYTES[tmp.dtype]
    t_stream = _cpu_time_us(lambda: rt.memset_d8(tptr, 0, tlen, stream=stream))
    print(f"  stream={stream}  (2 MB memset)      : {t_stream:7.1f} µs")

    # ── 10. Per-buffer dtype/ptr extraction inside launch ──
    _sec("10. [buf.data_ptr() for buf in buffers]  (ptr extraction only)")
    t_ptrs = _cpu_time_us(lambda: [b.data_ptr() for b in prebuilt_buffers])
    print(f"  host     : {t_ptrs:7.1f} µs")

    # ── 11. Param-spec scalar pack ──
    _sec("11. param_spec.pack_scalars(())")
    t_pack = _cpu_time_us(lambda: pspec.pack_scalars(()))
    print(f"  host     : {t_pack:7.1f} µs")

    # ── Summary ──
    _sec("Summary")
    print(f"  Linear.forward wall-clock   : {t_full:7.1f} µs")
    print(f"  pcf.gemm wall-clock         : {t_gemm:7.1f} µs  ({t_full - t_gemm:+.1f} Linear overhead)")
    print(f"  compiled.launch wall-clock  : {t_launch:7.1f} µs  ({t_gemm - t_launch:+.1f} call_with_bindings overhead)")
    print(f"  gpu kernel time             : {t_launch_gpu:7.1f} µs  ({t_launch - t_launch_gpu:+.1f} driver/stream)")
    print()
    print(f"  autotune.lookup             : {t_lookup:7.1f} µs")
    print(f"  make_key alone              : {t_key:7.1f} µs")
    print(f"  spec_from_tensors           : {t_spec:7.1f} µs")
    print(f"  launcher.compile cache hit  : {t_compile:7.1f} µs")
    print()

    gc.collect()


if __name__ == "__main__":
    main()
