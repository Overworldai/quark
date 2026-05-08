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

_DTYPE_KEYS = (
    "dtype", "in_dtype", "kv_dtype", "compute_dtype",
    "src_dtype", "out_dtype", "partials_dtype",
    "a_dtype", "b_dtype",
)
_OK = {DType.F32, DType.U32, DType.S32, DType.BF16}


def _coerce_dtype(params, spec_cls, target):
    p = dict(params)
    for k in _DTYPE_KEYS:
        if k in p:
            p[k] = target
    if spec_cls is not None:
        import inspect
        sig = inspect.signature(spec_cls)
        for k in _DTYPE_KEYS:
            if k in sig.parameters and k not in p:
                p[k] = target
    return p


def time_callable(fn, *, warmup_ms=20.0, bench_ms=300.0, sync_each=True):
    """Time ``fn`` over a fixed wall budget.

    ``sync_each=False`` lets the bench batch many launches per fence
    wait, then ``_sd.sync()`` once at the end. Reveals the achievable
    GPU rate when the eager-fence-wait isn't on the critical path.
    """
    t0 = time.perf_counter()
    n = 0
    while (time.perf_counter() - t0) * 1000 < warmup_ms:
        fn()
        n += 1
    if not sync_each:
        _sd.sync()
    t = time.perf_counter()
    n = 0
    while (time.perf_counter() - t) * 1000 < bench_ms:
        fn()
        n += 1
    if not sync_each:
        _sd.sync()
    return (time.perf_counter() - t) * 1e6 / max(n, 1)


for kcls in all_kernels():
    name = getattr(kcls, "NAME", kcls.__name__)
    problems = kcls.problems()
    if not problems:
        continue
    
    # Pick the first problem that lowers + dispatches at the natural
    # dtype, falling back to F32 only when the natural path doesn't
    # compile (mirrors test_spv_smoke). This keeps the bench numbers
    # representative of production (mostly bf16 on Battlemage).
    chosen = None
    for p in problems:
        for target in (None, DType.F32):
            params = p.params if target is None else _coerce_dtype(
                p.params, kcls.SPEC_CLS, target,
            )
            try:
                kernel = kcls.from_problem(params)
            except Exception:
                continue
            spec_dts = {getattr(kernel.spec, a) for a in _DTYPE_KEYS
                        if isinstance(getattr(kernel.spec, a, None), DType)}
            if not (not spec_dts or spec_dts <= _OK):
                continue
            # Mirror test_spv_smoke: resolve empty ``config.main_shape``
            # via the device's caps before is_valid_for runs (the
            # default config carries ``main_shape=''`` for many kernels
            # — the launcher resolves it at compile time, but
            # is_valid_for sees the un-resolved config).
            resolved_cfg = launcher._maybe_resolve_main_shape(
                kcls, kernel.spec, kernel.config,
            )
            if resolved_cfg is not kernel.config:
                kernel = kcls(kernel.spec, resolved_cfg)
            if not kernel.is_valid_for(device.caps):
                continue
            try:
                launcher.compile(kcls, kernel.spec, kernel.config)
            except Exception:
                continue
            chosen = (p, params, kernel)
            break
        if chosen is not None:
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
    
    # Resolve the workgroup grid once; the driver path packs scalars
    # and queries grid_fn — neither matters for the dispatch tight
    # loop, so compute it up-front and call the driver directly.
    grid = compiled.grid_fn(*compiled.grid_args)
    try:
        us_sync = time_callable(
            lambda: _sd.launch(
                compiled.module.handle, grid, handles, b"", sync=True,
            ),
        )
        us_async = time_callable(
            lambda: _sd.launch(
                compiled.module.handle, grid, handles, b"", sync=False,
            ),
            sync_each=False,
        )
    except Exception as e:
        print(f"{name:20s}  bench failed: {e}")
        continue
    
    gbs_sync = bytes_total / (us_sync * 1e-6) / 1e9 if us_sync > 0 else 0
    gbs_async = bytes_total / (us_async * 1e-6) / 1e9 if us_async > 0 else 0
    spec_dt = next(
        (str(getattr(kernel.spec, k).value) for k in _DTYPE_KEYS
         if isinstance(getattr(kernel.spec, k, None), DType)),
        "—",
    )
    print(f"{name:22s} {p.name:22s} {spec_dt:5s} "
          f"sync {us_sync:8.2f}μs {gbs_sync:5.1f}GB/s | "
          f"async {us_async:8.2f}μs {gbs_async:5.1f}GB/s")
