"""Quick SPV benchmark: run each kernel that smoke-passes and report
μs/call, GB/s, and dispatch count."""
import ctypes
import time
import numpy as np

from quark.ir import DType
from quark.kernels import all_kernels
from quark.launcher import Launcher
from quark.drivers import spv as _spv
from quark.drivers import _spv_dispatch as _sd

drv = _spv.SpvDriver()
device = drv.device
launcher = Launcher(device=device)

_DTYPE_KEYS = ("dtype", "in_dtype", "kv_dtype", "compute_dtype", "src_dtype", "out_dtype")
_OK = {DType.F32, DType.U32, DType.S32}


def _coerce_f32(params, spec_cls):
    p = dict(params)
    for k in _DTYPE_KEYS:
        if k in p:
            p[k] = DType.F32
    if spec_cls is not None:
        import inspect
        sig = inspect.signature(spec_cls)
        for k in _DTYPE_KEYS:
            if k in sig.parameters and k not in p:
                p[k] = DType.F32
    return p


def time_callable(fn, *, warmup_ms=20.0, bench_ms=300.0):
    t0 = time.perf_counter()
    n = 0
    while (time.perf_counter() - t0) * 1000 < warmup_ms:
        fn()
        n += 1
    t = time.perf_counter()
    n = 0
    while (time.perf_counter() - t) * 1000 < bench_ms:
        fn()
        n += 1
    return (time.perf_counter() - t) * 1e6 / max(n, 1)


for kcls in all_kernels():
    name = getattr(kcls, "NAME", kcls.__name__)
    problems = kcls.problems()
    if not problems:
        continue
    
    chosen = None
    for p in problems:
        params = _coerce_f32(p.params, kcls.SPEC_CLS)
        try:
            kernel = kcls.from_problem(params)
        except Exception:
            continue
        spec_dts = {getattr(kernel.spec, a) for a in _DTYPE_KEYS
                    if isinstance(getattr(kernel.spec, a, None), DType)}
        if (not spec_dts or spec_dts <= _OK) and kernel.is_valid_for(device.caps):
            chosen = (p, params, kernel)
            break
    if chosen is None:
        continue
    p, params, kernel = chosen
    
    if not hasattr(kcls, "make_tensors_numpy"):
        continue
    try:
        tensors = kcls.make_tensors_numpy(params)
    except Exception:
        continue
    if not isinstance(tensors, dict):
        continue
    
    try:
        compiled = launcher.compile(kcls, kernel.spec, kernel.config)
    except Exception:
        continue
    if compiled.module.n_buffers != len(compiled.param_spec.buffers):
        continue
    
    handles = []
    bytes_total = 0
    for buf in compiled.param_spec.buffers:
        arr = tensors[buf.name]
        h, ptr = _sd.allocate_buffer(arr.nbytes)
        ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)
        handles.append(h)
        bytes_total += arr.nbytes
    
    try:
        us = time_callable(lambda: compiled.launch(buffers=handles))
    except Exception as e:
        print(f"{name:20s}  bench failed: {e}")
        continue
    
    gbs = bytes_total / (us * 1e-6) / 1e9 if us > 0 else 0
    print(f"{name:20s}  {p.name:20s}  {us:8.2f} μs  {gbs:6.1f} GB/s")
