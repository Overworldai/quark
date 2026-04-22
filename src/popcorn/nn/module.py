"""Minimal inference-only Module system.

EXEMPT FROM 500-LINE RULE: Module + all leaf module classes (Linear,
Patchify, KVCacheUpdate, OwlAttn, etc.) form a single cohesive API.
Splitting would scatter the module hierarchy across files with no
readability benefit.

No autograd, no training hooks, no optimizer state. Just:
- Named parameter storage (PopcornTensor on CUDA, mx.array on Metal)
- Nested sub-modules
- state_dict() / load_state_dict() with key remapping
- forward() convention

On CUDA, all tensors are ``PopcornTensor`` instances with no torch or
numpy dependency. On Metal (until Stage 2 migration), the existing
MLX path uses ``mx.array`` directly; the old ``popcorn.backend.PT``
polymorphic tensor was retired in the numpy-refs migration.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

_IS_METAL = sys.platform == "darwin"


_MX_DT_MAP = None


def _mx_dt(dtype: str):
    """Map popcorn short strings → mlx dtype. Metal only."""
    global _MX_DT_MAP
    if _MX_DT_MAP is None:
        import mlx.core as mx

        _MX_DT_MAP = {
            "bf16": mx.bfloat16,
            "f16": mx.float16,
            "f32": mx.float32,
            "s32": mx.int32,
        }
    return _MX_DT_MAP.get(dtype, _MX_DT_MAP["bf16"])


def _randn(*shape, dtype="bf16"):
    """Create a random normal tensor on the active backend."""
    if _IS_METAL:
        import mlx.core as mx

        return mx.random.normal(shape=shape).astype(_mx_dt(dtype))
    from popcorn.runtime.tensor import PopcornTensor

    return PopcornTensor.randn(*shape, dtype=dtype)


def _zeros(*shape, dtype="bf16"):
    if _IS_METAL:
        import mlx.core as mx

        return mx.zeros(shape, dtype=_mx_dt(dtype))
    from popcorn.runtime.tensor import PopcornTensor

    return PopcornTensor.zeros(*shape, dtype=dtype)


def _tensor(data, dtype="f32"):
    if _IS_METAL:
        import mlx.core as mx

        return mx.array(data).astype(_mx_dt(dtype))
    from popcorn.runtime.tensor import PopcornTensor

    return PopcornTensor.from_list(data, dtype=dtype)


def _astype(x, dtype: str):
    if _IS_METAL:
        return x.astype(_mx_dt(dtype))
    return x.astype(dtype)


def _quantize_to_e4m3(t):
    """Quantize bf16/f16 tensor to e4m3 via the quantize_e4m3 GPU kernel."""
    import popcorn.functional as pcf

    return pcf.quantize_e4m3(t)


def _profile_sync():
    """Sync the active backend so wall-clock time reflects GPU time."""
    from popcorn.runtime.sync import synchronize

    synchronize()


class Parameter:
    """A tensor that should appear in ``state_dict()``."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    def __repr__(self) -> str:
        shape = tuple(self.data.shape) if hasattr(self.data, "shape") else "?"
        dtype = getattr(self.data, "dtype", "?")
        return f"Parameter(shape={shape}, dtype={dtype})"


class Module:
    """Minimal inference module.

    Subclasses define ``__init__`` (register parameters and sub-modules
    via attribute assignment) and ``forward(*args, **kwargs)`` (the
    actual computation). Calling the module invokes ``forward``.

    Parameters are ``Parameter`` instances; sub-modules are ``Module``
    instances. Both are discovered by attribute name and participate
    in ``state_dict()`` / ``load_state_dict()`` and ``parameters()``.
    """

    # Profiling state (class-level so all modules share one timer dict).
    _profile_enabled: bool = False
    _profile_timings: dict[str, list[float]] = {}  # noqa: RUF012

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        graphs = getattr(self, "_graphs", None)
        if graphs and not Module._profile_enabled:
            return self._graph_replay(*args, **kwargs)
        kwargs.pop("graph_key", None)
        if Module._profile_enabled:
            return self._profiled_forward(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def _profiled_forward(self, *args: Any, **kwargs: Any) -> Any:
        import time

        name = getattr(self, "_profile_name", type(self).__name__)
        _profile_sync()
        t0 = time.perf_counter()
        result = self.forward(*args, **kwargs)
        _profile_sync()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        # A name of ``None`` marks the profile root — it's the sum of
        # every child so it always eclipses true leaves. Skip recording.
        if name is not None:
            Module._profile_timings.setdefault(name, []).append(elapsed_us)
        return result

    def _graph_replay(self, *args: Any, **kwargs: Any) -> Any:
        """Replay a captured graph. Selects the right graph by ``graph_key``
        kwarg (or the single graph if only one was captured)."""
        from popcorn.runtime.tensor import PopcornTensor

        graphs: dict = self._graphs  # type: ignore[attr-defined]
        key = kwargs.pop("graph_key", None)
        if key is None and len(graphs) == 1:
            key = next(iter(graphs))
        if key not in graphs:
            raise KeyError(f"No graph for key={key!r}. Available: {list(graphs.keys())}")
        g, input_bufs, outputs = graphs[key]

        # Copy tensor args into stable buffers.
        for buf, arg in zip(input_bufs, args, strict=False):
            if isinstance(arg, PopcornTensor) and buf is not None:
                buf.copy_from(arg)

        # Replay.
        g.replay()

        # Return cloned outputs.
        if isinstance(outputs, PopcornTensor):
            return outputs.clone()
        if isinstance(outputs, tuple):
            return tuple(o.clone() if isinstance(o, PopcornTensor) else o for o in outputs)
        return outputs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # ── parameter / module discovery ──

    def _own_members(self) -> Iterator[tuple[str, Any]]:
        for k, v in self.__dict__.items():
            if isinstance(v, (Parameter, Module)):
                yield k, v

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        for name, val in self._own_members():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            if isinstance(val, Parameter):
                yield full, val.data
            elif isinstance(val, Module):
                yield from val.named_parameters(prefix=full)

    def parameters(self) -> Iterator[Any]:
        for _, p in self.named_parameters():
            yield p

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        yield prefix, self
        for name, val in self._own_members():
            if isinstance(val, Module):
                full = f"{prefix}.{name}" if prefix else name
                yield from val.named_modules(prefix=full)

    # ── state dict ──

    def state_dict(self) -> OrderedDict[str, Any]:
        sd: OrderedDict[str, Any] = OrderedDict()
        for name, param in self.named_parameters():
            sd[name] = param
        return sd

    def load_state_dict(self, sd: dict[str, Any], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        missing = set(own) - set(sd)
        unexpected = set(sd) - set(own)
        if strict and missing:
            raise KeyError(f"Missing keys: {sorted(missing)}")
        if strict and unexpected:
            raise KeyError(f"Unexpected keys: {sorted(unexpected)}")
        self._assign_params(sd, prefix="")
        # If any Linear uses f16, cast all remaining bf16 params to f16.
        from popcorn.nn.layers import Linear

        has_f16 = any(
            isinstance(m, Linear) and m._out_dtype == "f16" for _, m in self.named_modules()
        )
        if has_f16:
            for _, val in self._own_members():
                if isinstance(val, Parameter) and str(val.data.dtype) == "bf16":
                    val.data = _astype(val.data, "f16")
                elif isinstance(val, Module):
                    self._cast_bf16_to_f16(val)

    @staticmethod
    def _cast_bf16_to_f16(mod: Module) -> None:
        """Recursively cast bf16 Parameters to f16 (skip f32)."""
        for _, val in mod._own_members():
            if isinstance(val, Parameter) and str(val.data.dtype) == "bf16":
                val.data = _astype(val.data, "f16")
            elif isinstance(val, Module):
                Module._cast_bf16_to_f16(val)

    def _assign_params(self, sd: dict[str, Any], prefix: str) -> None:
        for name, val in self.__dict__.items():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            if isinstance(val, Parameter) and full in sd:
                val.data = sd[full]
            elif isinstance(val, Module):
                val._assign_params(sd, prefix=full)

    # ── dtype cast (inference convenience) ──

    def to_dtype(self, dtype: str) -> Module:
        """Cast every parameter to ``dtype`` (a string: "bf16", "f32", etc.).
        Returns self."""
        for _, val in self.__dict__.items():
            if isinstance(val, Parameter):
                val.data = _astype(val.data, dtype)
            elif isinstance(val, Module):
                val.to_dtype(dtype)
        return self

    # ── prepare (recursive) ──

    def prepare(self, **kwargs) -> None:
        """Recursive prepare — calls ``prepare()`` on all child modules.

        Subclasses override this to add module-specific setup (e.g.
        ``Linear.prepare()`` pre-shuffles weights). The base
        implementation just recurses into children.
        """
        for _name, val in self._own_members():
            if isinstance(val, Module):
                val.prepare(**kwargs)

    # ── graph capture ──

    def graph(self, *example_args: Any, key: Any = 0, **example_kwargs: Any) -> None:
        """Capture this module's forward() as a replayable CUDA graph.

        ``key``: identifier for this graph variant. Use different keys
        to capture multiple graphs (e.g. one per sigma_idx). On replay,
        pass ``graph_key=key`` to select which graph to run.

        Creates stable input buffers from the tensor args. On replay,
        the caller's tensor args are D2D copied into these stable
        buffers, the graph replays, and outputs are returned as clones.
        """
        from popcorn.graph import capture_graph
        from popcorn.runtime.tensor import PopcornTensor

        if not hasattr(self, "_graphs"):
            self._graphs: dict = {}

        # Create stable input buffers from tensor args.
        input_bufs = []
        capture_args = []
        for arg in example_args:
            if isinstance(arg, PopcornTensor):
                buf = arg.clone()
                input_bufs.append(buf)
                capture_args.append(buf)
            else:
                input_bufs.append(None)
                capture_args.append(arg)

        with capture_graph() as g:
            outputs = self.forward(*capture_args, **example_kwargs)

        self._graphs[key] = (g, input_bufs, outputs)

    def ungraph(self) -> None:
        """Release all captured graphs, revert to eager execution."""
        if hasattr(self, "_graphs"):
            del self._graphs

    # ── profiling ──

    @staticmethod
    def profile(root: Module | None = None):
        """Context manager that times every Module.__call__ with GPU sync.

        Usage::

            with Module.profile(model):
                model(x, sigma_idx=0, frame_t=ft)
            Module.print_profile()

        Pass the root module to get hierarchical names (``blocks.0.attn``).
        Each sub-module's forward() is individually timed. Call
        ``Module.print_profile()`` afterwards to see the breakdown.
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            if root is not None:
                root._set_profile_names()
            Module._profile_timings.clear()
            Module._profile_enabled = True
            try:
                yield
            finally:
                Module._profile_enabled = False

        return _ctx()

    @staticmethod
    def print_profile(top_n: int = 0, leaves_only: bool = False) -> None:
        """Print profiling results.

        ``top_n=0`` means print all. ``leaves_only=True`` filters out
        parent modules whose children also appear (avoids double-counting).
        """
        from popcorn.utils.pretty import Table, print_header, style

        timings = Module._profile_timings
        if not timings:
            print("(no profile data)")
            return

        names = set(timings.keys())
        if leaves_only:
            # A name is a parent if any other name starts with "name.".
            parents = {n for n in names if any(o.startswith(n + ".") for o in names if o != n)}
            names = names - parents

        rows: list[tuple[str, int, float, float]] = []
        for name in names:
            times = timings[name]
            n = len(times)
            total = sum(times)
            median = sorted(times)[n // 2]
            rows.append((name, n, total, median))

        rows.sort(key=lambda r: r[2], reverse=True)
        if top_n > 0:
            rows = rows[:top_n]

        grand_total = sum(r[2] for r in rows)

        label = "Profile (leaves only)" if leaves_only else "Profile"
        print_header(label)
        t = Table()
        t.add_column("Module", align="left", min_width=30)
        t.add_column("Calls", align="right")
        t.add_column("Total (ms)", align="right")
        t.add_column("Median (us)", align="right")
        t.add_column("%", align="right")

        for name, n, total, median in rows:
            pct = total / grand_total * 100 if grand_total > 0 else 0
            t.add_row(name, str(n), f"{total / 1000:.2f}", f"{median:.1f}", f"{pct:.1f}")

        t.add_separator()
        t.add_row(style("total", "bold"), "", f"{grand_total / 1000:.2f}", "", "")
        t.print()

    def _set_profile_names(self, prefix: str = "") -> None:
        """Recursively assign ``_profile_name`` to every sub-module.

        The root's own name is set to ``None`` so its timing (which is
        the sum of every child) is excluded from the profile — otherwise
        the root eclipses the true leaves and ``leaves_only`` can't
        filter it out (children use flat names like ``blocks.0.attn``,
        not ``<Root>.blocks.0.attn``).
        """
        if not prefix:
            self._profile_name = None  # type: ignore[assignment]
        for name, val in self._own_members():
            if isinstance(val, Module):
                full = f"{prefix}.{name}" if prefix else name
                val._profile_name = full
                val._set_profile_names(prefix=full)

    # ── repr ──

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, val in self._own_members():
            lines.append(f"  ({name}): {val!r}")
        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    """Ordered list of sub-modules, indexed by position."""

    def __init__(self, modules: list[Module] | None = None):
        self._modules: list[Module] = list(modules or [])
        for i, m in enumerate(self._modules):
            setattr(self, str(i), m)

    def __getitem__(self, idx: int) -> Module:
        return self._modules[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def _own_members(self):
        for i, m in enumerate(self._modules):
            yield str(i), m

    def append(self, module: Module) -> None:
        i = len(self._modules)
        self._modules.append(module)
        setattr(self, str(i), module)
