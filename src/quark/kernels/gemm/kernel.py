"""Universal GEMM kernel — pipelined K, multi-dtype.

C[M, N] = A[M, K] @ B^T[N, K]^T

Architecture:
  - 2D grid: (N//BN, M//BM, 1); @kernel auto-publishes
    ``self.m_base = block_base("y", BM)`` / ``self.n_base =
    block_base("x", BN)`` before build() runs.
  - Pipelined K via ``run_pipeline``: n_stages=1 (synchronous)
    or n_stages=2 (double buffer).
  - ``SmemPlan.staged_pairs`` allocates n_stages of paired A/B
    smem with padding + b_shuffle-aware per-lane B views.
  - ``MmaBody(acc=, BK=)`` is callable as ``mma(ictx)`` for
    the K consume loop.
  - ``qk.store_acc`` epilogue, dtype-generic.

EXEMPT FROM 500-LINE RULE: the GEMM kernel ties together its IR-emit
``build_metal``/``build_ptx``, the autotune ``tune_space`` /
``alt_configs`` / ``force_alt_config`` triplet, and the cuBLAS-vs-PTX
``dispatch_alt_config`` hook. All four reach into the shared GemmSpec
+ GemmConfig dataclasses; splitting them would mean exporting that
state across module boundaries.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import (
    Accumulators,
    IterCtx,
    MmaBody,
    PipelineBody,
    SmemPlan,
    TensorDecl,
)
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.gemm.baselines import gemm_baselines
from quark.kernels.gemm.config import GemmConfig
from quark.kernels.gemm.problems import gemm_problems
from quark.kernels.gemm.reference import gemm_reference_numpy
from quark.kernels.gemm.spec import GemmSpec


@kernel(
    "gemm",
    spec=GemmSpec,
    config=GemmConfig,
    problems=gemm_problems,
    baselines=lambda kernel, tensors: gemm_baselines(tensors),
    reference=gemm_reference_numpy,
)
class GemmKernel(Kernel):
    # Parameter manifest — single source of truth for kernel tensor
    # shapes and dtypes (see quark.blocks.TensorDecl).
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A", dtype=lambda s, c: s.a_dtype, shape=lambda s, c: (s.M, s.K)),
        TensorDecl(
            "B",
            dtype=lambda s, c: s.b_dtype,
            shape=lambda s, c: (
                s.N,
                (s.K // c.BK) * (c.BK + c.b_pad) if s.b_shuffle and c.b_pad > 0 else s.K,
            ),
        ),
        TensorDecl(
            "Bias",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.N,) if s.has_bias else (1,),
        ),
        # Gate / Residual: only meaningful when has_gate_residual=True;
        # collapse to (1,) when off so the kernel doesn't pay the metadata
        # cost on the bias-only / plain-GEMM path. Mirrors the Bias slot.
        TensorDecl(
            "Gate",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.G, s.N) if s.has_gate_residual else (1,),
        ),
        TensorDecl(
            "Residual",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.M, s.N) if s.has_gate_residual else (1,),
        ),
        TensorDecl(
            "Out", dtype=lambda s, c: s.out_dtype, shape=lambda s, c: (s.M, s.N), role="out"
        ),
    ]

    spec: GemmSpec
    config: GemmConfig

    SHUFFLE_TENSOR = "B"

    # ── Validity ──

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        # cuBLAS-impl configs ignore the PTX tiling knobs entirely —
        # validity is decided by cublas_dispatch.is_cublas_eligible at
        # enumeration time, not by BM/BN/BK/main_shape here.
        if getattr(c, "impl", "ptx") == "cublas":
            return True
        if s.M % c.BM != 0 or s.N % c.BN != 0 or s.K % c.BK != 0:
            return False
        try:
            mma = self._mma_cfg()
        except KeyError:
            return False
        # NAX path: validates only the NAX-shaped tile constraints. The
        # default ``_validate_gemm_tile`` enforces simdgroup_matrix's
        # per-warp N-partition (``(BN//shape.n) % n_warps == 0``) which
        # doesn't match NAX's 2D warp layout (WM × WN). The NAX
        # ``build_metal`` only handles the explicit (n_warps, BM, BN, BK)
        # combos below; reject anything else.
        if mma.shape.name in ("m16n32k16_nax_bf16", "m32n32k16_nax_bf16"):
            if s.has_bias or c.split_k > 1:
                return False
            if s.activation is not None and s.activation != "silu":
                return False
            compute = s.compute_dtype_resolved
            if mma.shape.a_dtype is not compute or mma.shape.b_dtype is not compute:
                return False
            if c.n_warps not in (1, 8, 16):
                return False
            # Compute SM/SN for the chosen warp layout; the picked
            # main_shape must evenly tile into the per-simdgroup tile.
            n_sg = c.n_warps
            if n_sg == 16:
                WM, WN = 4, 4
            elif n_sg == 8:
                WM, WN = 2, 4
            elif n_sg == 1:
                WM, WN = 1, 1
            else:
                WM, WN = 1, n_sg
            if c.BM % WM != 0 or c.BN % WN != 0:
                return False
            SM = c.BM // WM
            SN = c.BN // WN
            M_frag = int(mma.shape.m)
            N_frag = int(mma.shape.n)
            if SM % M_frag != 0 or SN % N_frag != 0:
                return False
            if c.n_warps == 1:
                # Single-simdgroup path is wired only for the m=16 shape;
                # m=32 single-warp could be added but the multi-warp path
                # is the realistic 720p use case.
                if mma.shape.name != "m16n32k16_nax_bf16":
                    return False
                return (c.BM, c.BN, c.BK) == (16, 32, 16)
            # Multi-warp. BK ≤ 64 (tested up to 256 with
            # BaseNAXFrag::load — still degrades; Apple's runtime
            # compile of large bodies can't match MLX's pre-compiled
            # metallib at BK ≥ 128).
            if c.BM not in (64, 128) or c.BN not in (128, 256):
                return False
            return c.BK in (16, 32, 64) and s.K % c.BK == 0
        # The kernel's build() casts A/B on load to ``compute_dtype_resolved``
        # and lays out smem as that dtype. The MMA fragment loads then
        # read smem with the layout of ``mma_cfg.shape``. If the config
        # picks a ``main_shape`` whose dtype axes disagree with the spec's
        # compute dtype, the frag load uses the wrong stride / element
        # size on smem laid out for a different dtype — classic misaligned
        # address at launch. Reject the mismatch.
        compute = s.compute_dtype_resolved
        if mma.shape.a_dtype is not compute or mma.shape.b_dtype is not compute:
            return False
        if not self._validate_gemm_tile(mma):
            return False
        # K must split cleanly into BK-sized chunks. Without this the
        # kernel silently drops ``K % BK`` elements off the tail of
        # every row (cos_sim degrades and, because the tail read
        # straddles the real A/B boundary, some dtype / BK combos end
        # up dereferencing misaligned addresses at launch time).
        if s.K % c.BK != 0:
            return False
        # cp.async requires 16-byte aligned smem rows.
        compute_dt = s.compute_dtype_resolved
        elem_b = compute_dt.bytes
        if c.a_pad and ((c.BK + c.a_pad) * elem_b) % 16 != 0:
            return False
        if c.b_pad and ((c.BK + c.b_pad) * elem_b) % 16 != 0:
            return False
        # n_stages=2 uses a 2× unrolled loop → K-iteration count must be even.
        if c.n_stages == 2 and (s.K // c.BK) % 2 != 0:
            return False
        # split_k > 1 requires atomic add on the output dtype.
        # Dtype support is validated by is_valid_for(caps) which checks
        # AtomicRmwOp dtypes against device.atomic_add_dtypes.
        if c.split_k > 1:
            k_iters = s.K // c.BK
            if k_iters % c.split_k != 0:
                return False
            # Fused epilogue runs per-block BEFORE the atomic add, so
            # with split_k > 1 the bias gets added ``split_k`` times
            # (``out = sum_z (partial_z + bias)``) and the activation
            # is applied to each partial instead of the final sum
            # (``sum_z silu(partial_z) != silu(sum_z partial_z)`` —
            # silu is nonlinear). Reject the combo; the full-precision
            # fused path is only correct at split_k=1. A post-split
            # epilogue kernel would re-enable it but isn't wired up.
            if s.has_bias or s.activation is not None or s.has_gate_residual:
                return False
        return True

    def grid(self) -> tuple[int, int, int]:
        # Convention: x=N/BN (col_base), y=M/BM (row_base), z=split_k.
        s, c = self.spec, self.config
        return (s.N // c.BN, s.M // c.BM, c.split_k)

    def flops(self) -> int:
        return 2 * self.spec.M * self.spec.N * self.spec.K

    # ── Registry classmethods ──

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = GemmSpec(**problem)
        rng = np.random.default_rng(seed)
        if spec.has_bias:
            bias_np = astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.out_dtype)
        else:
            bias_np = zeros_for_dtype((1,), spec.out_dtype)
        if spec.has_gate_residual:
            gate_np = astype_numpy(
                rng.standard_normal((spec.G, spec.N)).astype(np.float32), spec.out_dtype
            )
            residual_np = astype_numpy(
                rng.standard_normal((spec.M, spec.N)).astype(np.float32), spec.out_dtype
            )
        else:
            gate_np = zeros_for_dtype((1,), spec.out_dtype)
            residual_np = zeros_for_dtype((1,), spec.out_dtype)
        return {
            "A": astype_numpy(
                rng.standard_normal((spec.M, spec.K)).astype(np.float32), spec.a_dtype
            ),
            "B": astype_numpy(
                rng.standard_normal((spec.N, spec.K)).astype(np.float32), spec.b_dtype
            ),
            "Bias": bias_np,
            "Gate": gate_np,
            "Residual": residual_np,
            "Out": zeros_for_dtype((spec.M, spec.N), spec.out_dtype),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        A,
        B,
        *,
        out_dtype: DType | str | None = None,
        compute_dtype: DType | str | None = None,
        b_shuffled: bool = False,
        activation: str | None = None,
        has_bias: bool = False,
    ) -> GemmSpec:
        """Derive a GemmSpec from live A, B tensors + scalar kwargs.

        A: [M, K] in ``a_dtype``. B: [N, K] in ``b_dtype`` (or preshuffled
        when ``b_shuffled=True`` — B is still [N, K*pad] in that case,
        but the functional layer accepts the raw preshuffled buffer).
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"qf.gemm: A, B must both be rank-2; got A={A.shape}, B={B.shape}")
        M, K = int(A.shape[0]), int(A.shape[1])
        N, K_b = int(B.shape[0]), int(B.shape[1])
        if not b_shuffled and K_b != K:
            raise ValueError(f"qf.gemm: A.shape[1] ({K}) != B.shape[1] ({K_b})")
        a_dt_str = DType.from_backend(A)
        b_dt_str = DType.from_backend(B)
        return GemmSpec(
            M=M,
            N=N,
            K=K,
            a_dtype=a_dt_str,
            b_dtype=b_dt_str,
            out_dtype=DType.coerce(out_dtype) or a_dt_str,
            compute_dtype=DType.coerce(compute_dtype),
            activation=activation,
            has_bias=has_bias,
            b_shuffle=b_shuffled,
        )

    @classmethod
    def alt_configs(cls, spec, caps=None) -> list:
        """Alternate-implementation configs outside ``tune_space()``.

        The autotune cache concatenates this with the PTX cartesian
        enumeration so cuBLAS competes on equal footing — one candidate
        among ~hundreds of PTX configs, timed under the same event
        bracket, cached by the same ``(kernel, spec, device)`` key.
        The cache also seeds alt_configs into its warm-seed slot so
        fast-search always times them (a single candidate in a pool
        of hundreds would be randomly-sampled out most of the time
        otherwise).

        Returns ``[]`` when cuBLAS is ineligible or unavailable; the
        caller treats both cases as "no alternate for this spec".
        """
        from quark.kernels.gemm.cublas_dispatch import is_cublas_eligible, make_cublas_config

        if not is_cublas_eligible(spec, caps):
            return []
        return [make_cublas_config()]

    @classmethod
    def force_alt_config(cls, spec, caps=None):
        """Return a forced-winner alt config, or ``None`` to skip the force path.

        When ``QUARK_FORCE_CUBLAS=1`` and the spec is cuBLAS-eligible,
        the autotune cache uses the returned config directly — no PTX
        search, no cache write. The forced config is ephemeral: it's
        returned in-memory only, so toggling the env off immediately
        reverts to whatever the cache already knows (or triggers a
        real search if it doesn't).
        """
        from quark.kernels.gemm.cublas_dispatch import (
            is_cublas_eligible,
            is_cublas_forced,
            make_cublas_config,
        )

        if is_cublas_forced() and is_cublas_eligible(spec, caps):
            return make_cublas_config()
        return None

    @classmethod
    def dispatch_alt_config(cls, spec, config, *, provided, auto_alloc, like=None) -> dict:
        """User-facing dispatch for an alt-impl config (cuBLAS).

        Mirrors ``call_with_bindings``'s PTX branch but only allocates
        auto-alloc buffers the alt path actually reads. For cuBLAS
        that's just output-role buffers (``Out``) — input dummies like
        the placeholder ``Bias`` when ``has_bias=False`` are *not*
        materialized: cublasLt takes ``bias_ptr=0`` for the no-bias
        case directly, so a fresh 1-element buffer every call would
        burn ``cuMemAllocAsync`` + ``cuMemsetD8Async`` per GEMM. Inside
        a captured CUDA graph that shows up as a per-frame replay cost
        (~100-500 µs per call × every fp8 Linear × every frame — the
        PR that introduced this path without the skip measured ~1.8×
        overall regression from this exact waste).

        The PTX path has no equivalent shortcut because the kernel's
        ParamSpec always includes ``Bias`` in its buffer list; it
        avoids the alloc cost instead by caching the dummy on
        ``CompiledKernel._input_dummies``. cuBLAS has no ParamSpec
        dependency, so we skip the mechanism entirely.
        """
        from quark.functional._dispatch import alloc_from_decl
        from quark.kernels.gemm.cublas_dispatch import dispatch_cublas

        if getattr(config, "impl", "ptx") != "cublas":
            raise ValueError(f"dispatch_alt_config: expected impl='cublas', got {config.impl!r}")

        decls_by_name = {d.name: d for d in cls.TENSORS}
        if like is None:
            like = next(iter(provided.values()))
        full: dict = dict(provided)
        for name in auto_alloc:
            decl = decls_by_name[name]
            # Only output-role buffers are materialized here — see
            # docstring for why we skip input-role dummies.
            if getattr(decl, "role", "in") != "out":
                continue
            full[name] = alloc_from_decl(decl, spec, config, like=like)

        Bias = full.get("Bias") if spec.has_bias else None

        # dispatch_cublas auto-picks active_stream() when stream=None,
        # so it lands on the graph-capture stream correctly here.
        dispatch_cublas(A=full["A"], B=full["B"], Out=full["Out"], Bias=Bias)
        return full

    @classmethod
    def check_alt_config(cls, spec, config, *, tensors, reference) -> tuple:
        """Correctness-check an alt-impl config against the reference.

        Runs ``dispatch_cublas`` against the pre-built autotune test
        tensors and compares the output to ``reference`` via
        ``check_correctness``. Returns ``(passed: bool, cos_sim: float,
        err: str | None)`` — same shape the autotune cache's
        ``_check_config`` / full-search's ``_evaluate_config``
        expect so the caller can branch on impl without duplicating
        the correctness-gate machinery.

        This hook is required because the default path in both searches
        calls ``launcher.compile(kernel_cls, spec, cfg)`` — which
        lowers *every* config through the PTX pipeline regardless of
        ``impl``. For cuBLAS configs that means the default PTX knobs
        (BM/BN/BK dataclass defaults) get used to "compile" the config;
        the correctness gate then checks the PTX output (not cuBLAS's)
        and the timing loop measures the PTX launch (not cuBLAS's).
        Without this hook cuBLAS never actually gets timed or checked
        as cuBLAS, and the autotune result's ``impl="cublas"`` tag
        reflects PTX-with-defaults performance — completely wrong.
        """
        from quark.correctness import check_correctness
        from quark.kernels.gemm.cublas_dispatch import dispatch_cublas
        from quark.runtime.sync import ir_dtype_of, synchronize, zero_buffer

        if getattr(config, "impl", "ptx") != "cublas":
            raise ValueError(f"check_alt_config: expected impl='cublas', got {config.impl!r}")

        A = tensors["A"]
        B = tensors["B"]
        Out_buf = tensors["Out"]
        Bias = tensors.get("Bias") if spec.has_bias else None

        try:
            Out_zeroed = zero_buffer(Out_buf)
            dispatch_cublas(A=A, B=B, Out=Out_zeroed, Bias=Bias)
            synchronize()
        except BaseException as e:
            return False, float("nan"), f"{type(e).__name__}: {e}"

        out_dtype = ir_dtype_of(Out_zeroed)
        cr = check_correctness(
            Out_zeroed,
            reference,
            out_dtype=out_dtype,
            threshold=cls.correctness_threshold(out_dtype),
        )
        return cr.passed, cr.cos_sim, None

    @classmethod
    def time_alt_config(cls, launcher, kernel, *, tensors=None) -> float:
        """Time an alt-impl config (cuBLAS) under the launcher's timing loop.

        The PTX path compiles a kernel then times ``ck.launch(buffers)``;
        this path skips compilation and times a ``dispatch_cublas``
        closure against the same event-bracketed ``_time_callable``.
        Tensors are materialized the same way as the PTX path so the
        comparison is apples-to-apples.
        """
        from quark.kernels.gemm.cublas_dispatch import dispatch_cublas
        from quark.launcher.launcher import _time_callable

        if getattr(kernel.config, "impl", "ptx") != "cublas":
            raise ValueError(f"time_alt_config: expected impl='cublas', got {kernel.config.impl!r}")

        if tensors is None:
            from quark.refs import ref_cache
            from quark.runtime.device_tensors import numpy_to_device_dict

            problems_fn = getattr(cls, "problems", None)
            problems = problems_fn() if callable(problems_fn) else []
            if not problems:
                raise ValueError("time_alt_config: kernel.problems() is empty")
            inputs_np, _ = ref_cache().get(cls, problems[0].params)
            tensors = numpy_to_device_dict(cls, kernel.spec, inputs_np)

        A = tensors["A"]
        B = tensors["B"]
        Out = tensors["Out"]
        Bias = tensors.get("Bias") if kernel.spec.has_bias else None

        for _ in range(launcher._autotune.warmup):
            dispatch_cublas(A=A, B=B, Out=Out, Bias=Bias)
        return _time_callable(
            lambda: dispatch_cublas(A=A, B=B, Out=Out, Bias=Bias),
            warmup_ms=10.0,
            bench_ms=50.0,
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # a_pad / b_pad includes both 8 (2-byte dtype granule) and 16
        # (1-byte dtype granule); is_valid() filters to the right one
        # per compute dtype. Backend-specific pruning (Metal unroll cap,
        # feature-flag gates like b_shuffle) happens in `is_valid_for`
        # via `DeviceCaps`, so this one search space serves every
        # backend — the autotuner just iterates the cartesian product
        # and drops invalid configs.
        return {
            "BM": [16, 32, 64, 128],
            "BN": [16, 32, 64, 128, 256],
            "BK": [16, 32, 64],
            "n_warps": [2, 4, 8],
            "n_stages": [1, 2],
            "a_pad": [0, 8, 16],
            "b_pad": [0, 8, 16],
            "split_k": [1, 2, 4, 8],
        }

    # ── build() ──

    def build_metal(self) -> None:
        """Metal NAX path — used when caps support NAX and the autotuner
        picks the ``m16n32k16_nax_bf16`` shape.

        Minimum-viable single-warp tile (BM=16, BN=32, BK=16): each
        block produces one 16×32 output tile via a runtime K-outer
        loop with a carried width-16 F32 accumulator. One NAX MMA
        per K-tile. Direct device loads + stores via the IR's NAX
        LoadMatrixOp / StoreMatrixOp paths.

        Falls through to the simdgroup_matrix ``build()`` for any
        spec/config combination this path doesn't cover (bias,
        activation, split-K, non-NAX shape).
        """
        s, c = self.spec, self.config
        try:
            mma = self._mma_cfg()
        except KeyError:
            self.build()
            return
        if mma.shape.name not in ("m16n32k16_nax_bf16", "m32n32k16_nax_bf16"):
            self.build()
            return
        # NAX path doesn't yet wire bias / split-K.
        if s.has_bias or c.split_k > 1:
            self.build()
            return
        if s.activation is not None and s.activation != "silu":
            self.build()
            return

        # NAX tile shapes wired:
        #   single-warp 16×32×16 (m=16 only) — one MMA per K-tile, simple.
        #   multi-warp  64×128×16+ — 8 simdgroups (WM=2 × WN=4); each
        #     simdgroup tiles its (SM, SN) sub-block with the picked
        #     per-fragment NAX shape (m16 default, m32 wider-M variant).
        #     m=32 halves the MMA dispatch count per K-iter at the cost
        #     of a wider cooperative_tensor (32 elems/lane vs 16).
        if (
            c.n_warps == 1
            and (c.BM, c.BN, c.BK) == (16, 32, 16)
            and mma.shape.name == "m16n32k16_nax_bf16"
        ):
            self._build_metal_nax_single_warp()
            return
        if (
            c.n_warps in (8, 16)
            and c.BM in (64, 128)
            and c.BN in (128, 256)
            and c.BK in (16, 32, 64)
        ):
            self._build_metal_nax_multi_warp()
            return
        self.build()

    def _build_metal_nax_single_warp(self) -> None:
        """Single-simdgroup NAX path: BM=16, BN=32, BK=16. One MMA per
        K-tile. One width-16 F32 accumulator. Used for small tiles where
        the multi-warp version's setup cost isn't worth it."""
        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx
        bld = self.bld
        m_base, n_base = self.m_base, self.n_base

        zero_f = bctx.c(0.0, dtype=DType.F32)
        init_acc = bld.vec_build([zero_f] * 16)

        K_outer = s.K // c.BK
        BK_c = bctx.c(c.BK, dtype=DType.U32)

        with qk.for_range(0, K_outer, 1, iv_name="ki", carried=(init_acc,)) as (
            ki,
            (acc_carried,),
        ):
            k_col = ki * BK_c
            a_frag = bld.load_matrix(g.A, "m16n32k16_nax_bf16", "a", m_base, k_col)
            b_frag = bld.load_matrix(g.B, "m16n32k16_nax_bf16", "b", n_base, k_col)
            new_acc = bld.mma("m16n32k16_nax_bf16", a_frag, b_frag, acc_carried)
            qk.yield_(new_acc)

        final_acc = bld.last_results[0]
        if s.activation == "silu":
            final_acc = qk.silu([final_acc])[0]
        if s.has_gate_residual:
            bld.store_matrix_gate_residual(
                g.Out,
                g.Residual,
                g.Gate,
                final_acc,
                "m16n32k16_nax_bf16",
                m_base,
                n_base,
                m_per_group=s.M // s.G,
            )
        else:
            bld.store_matrix(g.Out, final_acc, "m16n32k16_nax_bf16", "d", m_base, n_base)

    def _build_metal_nax_multi_warp(self) -> None:
        """Multi-simdgroup NAX gemm.

        8 simdgroups laid out as WM=2 × WN=4 cover the BM × BN output
        tile, each carrying TM × TN width-16 F32 accumulators (one per
        NAX 16×32 destination sub-tile). Direct gmem loads — Apple's
        L2 cache provides the data-reuse benefit and the NAX
        BaseNAXFrag lane→fragment coordinate map produces cache-friendly
        access patterns. Mirrors MLX's ``steel/gemm/gemm_nax.h``.

        K-loop: runtime loop over ``K / BK`` iterations, each
        Python-unrolling ``BK / 16`` inner K-tiles. BK=64 produces
        4 unrolled K-tiles per iter (sweet spot: tested BK={16..256},
        SK-style runtime inner loop, and kk=1 per-iter — BK=64
        Python-unroll consistently wins for the largest shapes).
        """
        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx
        bld = self.bld

        # Per-fragment dims come from the picked main_shape — m=16 stays
        # the legacy default, m=32 is the wider-fragment NAX shape that
        # halves the per-K-iter MMA dispatch count (1 m32 MMA ≡ 2 stacked
        # m16 MMAs in the M direction).
        mma = self._mma_cfg()
        shape_id = mma.shape.name
        M_FRAG = int(mma.shape.m)
        N_FRAG = int(mma.shape.n)
        K_FRAG = int(mma.shape.k)
        c_regs = int(mma.shape.c_regs)

        # Warp layout: prefer square (WM=WN) for balanced SM occupancy.
        # n_warps=8 → WM=2 WN=4; n_warps=16 → WM=4 WN=4.
        n_sg = c.n_warps
        if n_sg == 16:
            WM, WN = 4, 4
        elif n_sg == 8:
            WM, WN = 2, 4
        elif n_sg == 4:
            WM, WN = 2, 2
        else:
            WM, WN = 1, n_sg

        SM = c.BM // WM  # rows per simdgroup
        SN = c.BN // WN  # cols per simdgroup
        TM = SM // M_FRAG  # M-fragments per simdgroup
        TN = SN // N_FRAG  # N-fragments per simdgroup
        kk_per_iter = c.BK // K_FRAG
        assert c.BK % K_FRAG == 0

        K_iters = s.K // c.BK  # runtime loop count
        c_row_b, c_col_b, sg_row, sg_col = self._nax_block_origins(WN, SM, SN)
        am_offsets, tn_offsets, kk_offsets = self._nax_offset_consts(
            TM, TN, kk_per_iter, K_FRAG, N_FRAG, M_FRAG
        )
        zero_f = bctx.c(0.0, dtype=DType.F32)
        init_accs = tuple(bld.vec_build([zero_f] * c_regs) for _ in range(TM * TN))
        BK_c = bctx.c(c.BK, dtype=DType.U32)

        with qk.for_range(0, K_iters, 1, iv_name="ki", carried=init_accs) as (
            ki,
            carried_accs,
        ):
            k_col_base = ki * BK_c
            new_accs = self._nax_inner_mma(
                list(carried_accs),
                g.A,
                g.B,
                row_a=sg_row,
                col_a_base=k_col_base,
                row_b=sg_col,
                col_b_base=k_col_base,
                am_offsets=am_offsets,
                tn_offsets=tn_offsets,
                kk_offsets=kk_offsets,
                kk_per_iter=kk_per_iter,
                TM=TM,
                TN=TN,
                shape_id=shape_id,
            )
            qk.yield_(*new_accs)

        self._nax_store_accs(
            bld.last_results, sg_row, sg_col, am_offsets, tn_offsets, TM, TN, shape_id=shape_id
        )

    # ── NAX multi-warp helpers ──

    def _nax_block_origins(self, WN, SM, SN):
        """Return (c_row_b, c_col_b, sg_row, sg_col) — global block- and
        simdgroup-level origins. ``sg_row`` / ``sg_col`` are global
        (block_origin + simdgroup_offset)."""
        bctx = self.bctx
        c = self.config
        c_row_b = qk.block_idx("y") * bctx.c(c.BM, dtype=DType.U32)
        c_col_b = qk.block_idx("x") * bctx.c(c.BN, dtype=DType.U32)
        WN_c = bctx.c(WN, dtype=DType.U32)
        sg_m = bctx.warp_id // WN_c
        sg_n = bctx.warp_id % WN_c
        sg_row = c_row_b + sg_m * bctx.c(SM, dtype=DType.U32)
        sg_col = c_col_b + sg_n * bctx.c(SN, dtype=DType.U32)
        return c_row_b, c_col_b, sg_row, sg_col

    def _nax_offset_consts(self, TM, TN, kk_per_iter, K_FRAG, N_FRAG, M_FRAG=16):
        """Pre-build the small per-fragment offset constants used inside
        the inner MMA emission. Hoisted so each ``for_range`` body
        doesn't re-emit them. ``M_FRAG`` defaults to 16 for the legacy
        m=16 main_shape; m=32 callers pass M_FRAG=32."""
        bctx = self.bctx
        am_offsets = [bctx.c(am * M_FRAG, dtype=DType.U32) for am in range(TM)]
        tn_offsets = [bctx.c(tn * N_FRAG, dtype=DType.U32) for tn in range(TN)]
        kk_offsets = [bctx.c(kk * K_FRAG, dtype=DType.U32) for kk in range(kk_per_iter)]
        return am_offsets, tn_offsets, kk_offsets

    def _nax_inner_mma(
        self,
        accs,
        A_tensor,
        B_tensor,
        *,
        row_a,
        col_a_base,
        row_b,
        col_b_base,
        am_offsets,
        tn_offsets,
        kk_offsets,
        kk_per_iter,
        TM,
        TN,
        shape_id="m16n32k16_nax_bf16",
    ):
        """Emit ``kk_per_iter`` inner K-tile steps, each issuing all TN B
        frags + all TM A frags up front and then all TM × TN MMAs.
        Loads are issued before any MMA so Apple's scheduler can hoist
        them and overlap gmem latency with the MMA-compute phase.

        ``shape_id`` selects the per-fragment NAX shape — m16n32k16
        (legacy default, 1 MMA per (am, tn) cell) or m32n32k16 (wider M
        fragment, 1 MMA covers 32 rows so TM is half what it would be at
        m=16 for the same SM).
        """
        bld = self.bld
        for kk in range(kk_per_iter):
            k_col_a = col_a_base + kk_offsets[kk]
            k_col_b = col_b_base + kk_offsets[kk]
            b_frags = [
                bld.load_matrix(
                    B_tensor,
                    shape_id,
                    "b",
                    row_b + tn_offsets[tn],
                    k_col_b,
                )
                for tn in range(TN)
            ]
            a_frags = [
                bld.load_matrix(
                    A_tensor,
                    shape_id,
                    "a",
                    row_a + am_offsets[am],
                    k_col_a,
                )
                for am in range(TM)
            ]
            for am in range(TM):
                for tn in range(TN):
                    slot = am * TN + tn
                    accs[slot] = bld.mma(
                        shape_id,
                        a_frags[am],
                        b_frags[tn],
                        accs[slot],
                    )
        return accs

    def _nax_store_accs(
        self,
        accs,
        sg_row,
        sg_col,
        am_offsets,
        tn_offsets,
        TM,
        TN,
        *,
        shape_id="m16n32k16_nax_bf16",
    ):
        """Write the TM × TN destination sub-tiles to global Out.

        When ``self.spec.activation == "silu"``, fuse silu in-register
        on the F32 accumulators before each store — saves the downstream
        silu kernel + bf16 read/write roundtrip on the MLP fc1 path.

        When ``self.spec.has_gate_residual``, route each tile through
        ``store_matrix_gate_residual`` instead — emits a per-lane fused
        store that reads residual + gate from gmem (bf16), combines in
        F32, and writes the bf16 output. Saves the standalone
        ``AdaGateResidualKernel`` dispatch + the bf16 read/write of the
        accumulator on the post-attn / post-MLP residual.

        ``shape_id`` selects the per-fragment NAX shape (m=16 or m=32);
        the store machinery in the lowerer reads ``shape.m / .n`` for
        the per-fragment offsets, so the same call works for both.
        """
        bld = self.bld
        g = self.g
        s = self.spec
        if s.activation == "silu":
            accs = qk.silu(accs)
        m_per_group = (s.M // s.G) if s.has_gate_residual else 1
        for am in range(TM):
            a_row = sg_row + am_offsets[am]
            for tn in range(TN):
                slot = am * TN + tn
                a_col = sg_col + tn_offsets[tn]
                if s.has_gate_residual:
                    bld.store_matrix_gate_residual(
                        g.Out,
                        g.Residual,
                        g.Gate,
                        accs[slot],
                        shape_id,
                        a_row,
                        a_col,
                        m_per_group=m_per_group,
                    )
                else:
                    bld.store_matrix(
                        g.Out,
                        accs[slot],
                        shape_id,
                        "d",
                        a_row,
                        a_col,
                    )

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx, m_base, n_base = self.bctx, self.m_base, self.n_base
        compute_ir = s.compute_dtype_resolved
        mma_cfg = self._mma_cfg()
        a_cast = compute_ir if s.a_dtype is not compute_ir else None
        b_cast = compute_ir if s.b_dtype is not compute_ir else None

        stages = SmemPlan.staged_pairs(
            compute_ir,
            a_shape=(c.BM, c.BK),
            b_shape=(c.BN, c.BK),
            a_pad=c.a_pad,
            b_pad=c.b_pad,
            mma_cfg=mma_cfg,
            n_warps=c.n_warps,
            b_shuffled=s.b_shuffle,
            n_stages=c.n_stages,
        )
        acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
        mma = MmaBody(acc=acc, b_shuffled=s.b_shuffle)
        BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
        bk_stride_a = c.BK
        bk_stride_b = (c.BK + c.b_pad) if (s.b_shuffle and c.b_pad > 0) else c.BK
        K_outer = s.K // c.BK

        # Split-K: each block processes a slice of the K dimension.
        split_k = c.split_k
        if split_k > 1:
            k_iters_per_split = K_outer // split_k
            k_split_idx = qk.block_idx("z")
            k_start = k_split_idx * bctx.c(k_iters_per_split, dtype=DType.U32)
        else:
            k_iters_per_split = K_outer
            k_start = bctx.c(0, dtype=DType.U32)

        def produce(ictx: IterCtx) -> None:
            plan = ictx.stage
            # Offset the K iteration by the split-K start.
            k_iter = k_start + ictx.iter_idx
            k_col_a = k_iter * bk_stride_a
            k_col_b = k_col_a if bk_stride_b == bk_stride_a else k_iter * bk_stride_b
            plan.a.load_from(g.A, row=m_base, col=k_col_a, cast=a_cast)
            plan.b.load_from(g.B, row=n_base, col=k_col_b, cast=b_cast)

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=mma,
            carry=acc,
        ).run(n_iters=k_iters_per_split, n_stages=c.n_stages)

        qk.store_acc(
            g.Out,
            acc,
            row=m_base,
            col=n_base + bctx.warp_id * BN_per_warp,
            cast=s.out_dtype,
            activation=s.activation,
            bias=g.Bias if s.has_bias else None,
            gate=g.Gate if s.has_gate_residual else None,
            residual=g.Residual if s.has_gate_residual else None,
            gate_groups=s.G if s.has_gate_residual else 1,
            atomic=split_k > 1,
        )
