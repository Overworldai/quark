"""ctypes binding to libcublasLt — cuBLAS Lt (fp8-capable) matmul.

EXEMPT FROM 500-LINE RULE: mirrors ``runtime/cuda.py`` — one thin
wrapper per cuBLAS Lt symbol we actually need, plus a
``CublasRuntime.instance()`` process-wide singleton.

Used by ``popcorn.functional.gemm`` to short-circuit the custom kernel
path for dtype combos cuBLAS handles well with zero preprocessing:

    (bf16, bf16) → bf16 / f32
    (e4m3, e4m3) → bf16     (scale-free, fp8 TN kernel)

All matmuls are row-major ``C = A @ B^T`` with ``A:[M, K]``,
``B:[N, K]``, ``C:[M, N]``. Row-major is mapped to cublasLt's
col-major by swapping A/B and using ``opA=T, opB=N``. That happens to
be exactly the TN form cublasLt's fp8 kernel requires — the "always
A × B^T" convention popcorn uses for weights is free alignment here.

FP8 scales: cublasLt requires A_scale / B_scale pointers for fp8. We
allocate a 4-byte device scalar set to 1.0f once at singleton init and
point both at it — equivalent to "no scaling" while satisfying the API.

Availability: if ``libcublasLt.so.12`` isn't on the loader path, the
singleton's ``is_available()`` returns ``False`` and the functional
dispatch silently falls back to the custom kernel. No hard dep.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# cublasStatus_t constants
# ---------------------------------------------------------------------------

CUBLAS_STATUS_SUCCESS = 0

# cublasOperation_t
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

# cudaDataType_t (cuda.h)
CUDA_R_16F = 2
CUDA_R_32F = 0
CUDA_R_16BF = 14
CUDA_R_8F_E4M3 = 28
CUDA_R_8F_E5M2 = 29

# cublasComputeType_t (cublas_api.h)
CUBLAS_COMPUTE_32F = 68

# cublasLtMatmulDescAttributes_t (cublasLt.h)
CUBLASLT_MATMUL_DESC_TRANSA = 3
CUBLASLT_MATMUL_DESC_TRANSB = 4
CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17
CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18
CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19
CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20

# popcorn dtype string → cudaDataType_t
_DTYPE_TO_CUDA: dict[str, int] = {
    "bf16": CUDA_R_16BF,
    "f16": CUDA_R_16F,
    "f32": CUDA_R_32F,
    "e4m3": CUDA_R_8F_E4M3,
    "e5m2": CUDA_R_8F_E5M2,
}

# Workspace size cublasLt recommends (32 MiB covers every plan on
# Ada/Hopper/Blackwell per NVIDIA's guide).
_WORKSPACE_BYTES = 32 * 1024 * 1024


class CublasError(RuntimeError):
    """Raised when a cublasLt call returns nonzero status."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"cublasStatus({code}): {message}")
        self.code = code
        self.message = message


def _load_libcublaslt() -> Optional[ctypes.CDLL]:
    """Load ``libcublasLt.so.12`` / ``cublasLt64_12.dll``.

    Returns ``None`` (not raising) if not found — callers fall back
    to the custom kernel path. We try the CUDA 12 SONAME first since
    that's what ships with modern toolkits.
    """
    if sys.platform == "win32":
        for name in ("cublasLt64_12.dll", "cublasLt64_11.dll", "cublasLt"):
            try:
                return ctypes.WinDLL(name)  # type: ignore[attr-defined]
            except OSError:
                continue
        return None

    candidates = ["libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so"]
    found = ctypes.util.find_library("cublasLt")
    if found is not None:
        candidates.insert(0, found)
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class CublasRuntime:
    """ctypes wrapper over libcublasLt. Singleton.

    The handle, workspace buffer, and unit-scale device scalar are
    allocated once and reused across every ``matmul`` call.
    """

    _lock = threading.Lock()
    _instance: Optional[CublasRuntime] = None
    _unavailable: bool = False

    def __init__(self) -> None:
        lib = _load_libcublaslt()
        if lib is None:
            raise CublasError(0, "libcublasLt not found")
        self._lib = lib
        self._setup_signatures()

        # Singleton handle — lives for process lifetime.
        handle = ctypes.c_void_p(0)
        self._check(self._lib.cublasLtCreate(ctypes.byref(handle)))
        self._handle = handle.value or 0

        # Persistent workspace + unit-scale buffer via CudaRuntime.
        from popcorn.runtime.cuda import CudaRuntime

        rt = CudaRuntime.instance()
        self._workspace = rt.mem_alloc(_WORKSPACE_BYTES)
        self._workspace_size = _WORKSPACE_BYTES

        # One device float32 = 1.0, used as the A/B scale for fp8. We
        # allocate with the legacy (non-async) path so the scalar is
        # visible independent of any graph-capture stream state.
        one_f32 = ctypes.c_float(1.0)
        ptr = ctypes.c_uint64(0)
        rt._check(rt._lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(4)))
        self._unit_scale = int(ptr.value)
        rt.memcpy_htod(self._unit_scale, ctypes.addressof(one_f32), 4)

    # ---- singleton ----

    @classmethod
    def instance(cls) -> CublasRuntime:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """True iff libcublasLt loads and a handle can be created."""
        if cls._unavailable:
            return False
        try:
            cls.instance()
            return True
        except Exception:
            cls._unavailable = True
            return False

    # ---- signatures ----

    def _setup_signatures(self) -> None:
        L = self._lib

        L.cublasLtCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.cublasLtCreate.restype = ctypes.c_int

        L.cublasLtDestroy.argtypes = [ctypes.c_void_p]
        L.cublasLtDestroy.restype = ctypes.c_int

        L.cublasLtMatmulDescCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # cublasLtMatmulDesc_t*
            ctypes.c_int,  # cublasComputeType_t
            ctypes.c_int,  # cudaDataType_t (scale type)
        ]
        L.cublasLtMatmulDescCreate.restype = ctypes.c_int

        L.cublasLtMatmulDescDestroy.argtypes = [ctypes.c_void_p]
        L.cublasLtMatmulDescDestroy.restype = ctypes.c_int

        L.cublasLtMatmulDescSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        L.cublasLtMatmulDescSetAttribute.restype = ctypes.c_int

        L.cublasLtMatrixLayoutCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # cublasLtMatrixLayout_t*
            ctypes.c_int,  # cudaDataType
            ctypes.c_uint64,  # rows
            ctypes.c_uint64,  # cols
            ctypes.c_int64,  # ld
        ]
        L.cublasLtMatrixLayoutCreate.restype = ctypes.c_int

        L.cublasLtMatrixLayoutDestroy.argtypes = [ctypes.c_void_p]
        L.cublasLtMatrixLayoutDestroy.restype = ctypes.c_int

        L.cublasLtMatmul.argtypes = [
            ctypes.c_void_p,  # lightHandle
            ctypes.c_void_p,  # computeDesc
            ctypes.c_void_p,  # alpha (host)
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # Adesc
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # Bdesc
            ctypes.c_void_p,  # beta (host)
            ctypes.c_void_p,  # C
            ctypes.c_void_p,  # Cdesc
            ctypes.c_void_p,  # D
            ctypes.c_void_p,  # Ddesc
            ctypes.c_void_p,  # algo (NULL = heuristic)
            ctypes.c_void_p,  # workspace
            ctypes.c_size_t,  # workspaceSize
            ctypes.c_void_p,  # stream
        ]
        L.cublasLtMatmul.restype = ctypes.c_int

        # Optional: status-string helper. Available in CUDA 11.5+ but
        # not every libcublasLt exports it — guard the lookup.
        try:
            L.cublasLtGetStatusString.argtypes = [ctypes.c_int]
            L.cublasLtGetStatusString.restype = ctypes.c_char_p
            self._has_status_string = True
        except AttributeError:
            self._has_status_string = False

    # ---- error check ----

    def _check(self, res: int) -> None:
        if res == CUBLAS_STATUS_SUCCESS:
            return
        if self._has_status_string:
            msg_ptr = self._lib.cublasLtGetStatusString(res)
            msg = msg_ptr.decode() if msg_ptr else "<no status string>"
        else:
            msg = "<cublasLtGetStatusString unavailable>"
        raise CublasError(res, msg)

    # ---- matmul ----

    def matmul(
        self,
        *,
        a_ptr: int,
        b_ptr: int,
        c_ptr: int,
        M: int,
        N: int,
        K: int,
        a_dtype: str,
        b_dtype: str,
        c_dtype: str,
        stream: int = 0,
    ) -> None:
        """Compute row-major ``C[M, N] = A[M, K] @ B[N, K]^T``.

        Dispatched from ``popcorn.functional.gemm`` after the dtype
        gate; callers are responsible for ensuring the combo is one
        of the supported TN-shaped combos (bf16×bf16→bf16/f32 or
        e4m3×e4m3→bf16). alpha=1, beta=0.

        ``stream`` is a ``CUstream`` pointer (as int); 0 uses the
        legacy default stream.
        """
        a_cuda = _DTYPE_TO_CUDA[a_dtype]
        b_cuda = _DTYPE_TO_CUDA[b_dtype]
        c_cuda = _DTYPE_TO_CUDA[c_dtype]
        is_fp8_input = a_dtype in ("e4m3", "e5m2") or b_dtype in ("e4m3", "e5m2")
        is_fp8_output = c_dtype in ("e4m3", "e5m2")

        desc = ctypes.c_void_p(0)
        self._check(
            self._lib.cublasLtMatmulDescCreate(
                ctypes.byref(desc),
                ctypes.c_int(CUBLAS_COMPUTE_32F),
                ctypes.c_int(CUDA_R_32F),
            )
        )
        try:
            trans_t = ctypes.c_int(CUBLAS_OP_T)
            trans_n = ctypes.c_int(CUBLAS_OP_N)
            self._check(
                self._lib.cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_TRANSA,
                    ctypes.byref(trans_t),
                    ctypes.sizeof(trans_t),
                )
            )
            self._check(
                self._lib.cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_TRANSB,
                    ctypes.byref(trans_n),
                    ctypes.sizeof(trans_n),
                )
            )

            if is_fp8_input:
                scale_ptr = ctypes.c_uint64(self._unit_scale)
                for attr in (
                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                ):
                    self._check(
                        self._lib.cublasLtMatmulDescSetAttribute(
                            desc,
                            attr,
                            ctypes.byref(scale_ptr),
                            ctypes.sizeof(scale_ptr),
                        )
                    )
            if is_fp8_output:
                # cublasLt writes D = alpha * op(A) * op(B) * A_scale *
                # B_scale / D_scale. With a unit D_scale the output is
                # the raw product narrowed to fp8 — exactly what the
                # custom kernel produces (no extra scaling).
                d_scale_ptr = ctypes.c_uint64(self._unit_scale)
                self._check(
                    self._lib.cublasLtMatmulDescSetAttribute(
                        desc,
                        CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                        ctypes.byref(d_scale_ptr),
                        ctypes.sizeof(d_scale_ptr),
                    )
                )

            # Layouts: see module docstring for the row-major → col-major
            # mapping. cublasLt_A = our B (op=T, K×N, ld=K), cublasLt_B =
            # our A (op=N, K×M, ld=K), cublasLt_D = our C (N×M, ld=N).
            a_layout = ctypes.c_void_p(0)
            b_layout = ctypes.c_void_p(0)
            d_layout = ctypes.c_void_p(0)
            self._check(
                self._lib.cublasLtMatrixLayoutCreate(
                    ctypes.byref(a_layout),
                    ctypes.c_int(b_cuda),
                    ctypes.c_uint64(K),
                    ctypes.c_uint64(N),
                    ctypes.c_int64(K),
                )
            )
            self._check(
                self._lib.cublasLtMatrixLayoutCreate(
                    ctypes.byref(b_layout),
                    ctypes.c_int(a_cuda),
                    ctypes.c_uint64(K),
                    ctypes.c_uint64(M),
                    ctypes.c_int64(K),
                )
            )
            self._check(
                self._lib.cublasLtMatrixLayoutCreate(
                    ctypes.byref(d_layout),
                    ctypes.c_int(c_cuda),
                    ctypes.c_uint64(N),
                    ctypes.c_uint64(M),
                    ctypes.c_int64(N),
                )
            )
            try:
                alpha = ctypes.c_float(1.0)
                beta = ctypes.c_float(0.0)
                self._check(
                    self._lib.cublasLtMatmul(
                        ctypes.c_void_p(self._handle),
                        desc,
                        ctypes.addressof(alpha),
                        ctypes.c_void_p(b_ptr),  # our B as cublasLt A
                        a_layout,
                        ctypes.c_void_p(a_ptr),  # our A as cublasLt B
                        b_layout,
                        ctypes.addressof(beta),
                        ctypes.c_void_p(c_ptr),
                        d_layout,
                        ctypes.c_void_p(c_ptr),
                        d_layout,
                        None,  # algo = NULL → heuristic
                        ctypes.c_void_p(self._workspace),
                        ctypes.c_size_t(self._workspace_size),
                        ctypes.c_void_p(stream),
                    )
                )
            finally:
                self._lib.cublasLtMatrixLayoutDestroy(a_layout)
                self._lib.cublasLtMatrixLayoutDestroy(b_layout)
                self._lib.cublasLtMatrixLayoutDestroy(d_layout)
        finally:
            self._lib.cublasLtMatmulDescDestroy(desc)
