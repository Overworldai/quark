# Testing

Three layers:

| Layer | Command | Purpose |
|---|---|---|
| Unit | `make test` | IR construction, lowering regex, launcher dispatch, registry |
| Smoke | `make test KERNEL=name` | one `problems()[0]` end-to-end per kernel |
| Correctness sweep | `make fuzz [KERNEL=…] [TAG=…]` | every kernel × every problem |

The pre-commit hook runs `make test` (device-free subset) — it must
stay <5 seconds.

## Structure

```
tests/
    ir/              Builder API, ops, tensor types, validator
    lower/ptx/       PTX regex against lowered IR (CUDA-specific; skip on Metal)
    lower/msl/       MSL regex against lowered IR (Metal-specific; skip on CUDA)
    launcher/        registry, device, param spec, autotune cache
    kernels/
        test_smoke.py   auto-parameterized — one test per registered kernel
    test_correctness.py, test_weight_shuffle.py, ...
```

Tests are CUDA-available-aware. On Apple silicon the `tests/lower/ptx/`
suite runs fine (it just tests text generation); tests that actually
allocate on a CUDA device skip via
`pytest.mark.skipif(not torch.cuda.is_available())`. Torch is a dev
extra (`pip install .[dev]`), so it's present in the test env even
though the runtime inference path doesn't need it.

## Smoke tests (`tests/kernels/test_smoke.py`)

Auto-discovers every `@kernel`-registered class. For each:

1. `cls.problems()[0]` → first Problem.
2. `cls.from_problem(problem.params)` → kernel instance.
3. `kernel.is_valid_for(device.caps)` → skip if invalid.
4. `cls.make_tensors(problem.params)` → allocate.
5. Compile + launch.
6. `kernel.reference(*inputs)` → oracle.
7. `check_correctness(output, ref, out_dtype)` → cos_sim gate.

**Adding a kernel with `@kernel` automatically adds a smoke test.**
You never write a per-kernel test file.

## Fuzz (`tools/fuzz.py`)

Registry-driven correctness sweep. Walks every kernel × every
problem with the default (or tuned) config, checks cos_sim.

```bash
make fuzz                          # everything
make fuzz KERNEL=moe_inproj        # one kernel
make fuzz TAG=smoke                # fast gate
make fuzz TAG=production
```

Error output surfaces root-cause. `ERR: tensors: <Exception>`,
`ERR: launch: <Exception>`, `ERR: ref: <Exception>` so you can see
which stage broke.

## Correctness metric

**Cosine similarity only.** No `max_abs`, no `torch.allclose`.

`check_correctness(out, ref, out_dtype)`:
- Accepts `PopcornTensor`, raw torch, or mlx tensors (any device).
- `cos_sim = dot(out, ref) / (|out| * |ref|)`.
- Threshold from a `(out_dtype, accum_dtype)` table
  (`popcorn/correctness.py:_THRESHOLD_TABLE`).
- NaN in reference → `ValueError` (fix your reference).
- NaN / Inf in output → hard fail with flat index of first bad elem.

| `out_dtype` | `accum_dtype` | Threshold |
|---|---|---|
| F32 | F32 | 1 − 1e-5 |
| BF16 | F32 | 1 − 1e-3 |
| F16 | F32 | 1 − 1e-3 |
| F16 | F16 | 1 − 5e-3 |
| E4M3 | F32 | 1 − 5e-3 |
| S32 / U32 | S32 / U32 | 1 − 1e-9 |

Override per-kernel by setting `CORRECTNESS_THRESHOLD: float` on the
class (not as a getattr — the base `Kernel.correctness_threshold`
method consults the class attribute).

```python
class AttnKernel(Kernel):
    CORRECTNESS_THRESHOLD = 0.99    # bf16 MMA + online softmax drift
```

## What to test and how

| Layer | What | Pattern |
|---|---|---|
| IR | New ops, type guards | Build graph; assert shapes / dtypes (`tests/ir/`) |
| IR validator | Error paths | Hand-construct broken IR; verify rejection |
| PTX lowering | New op emission | Emit IR → `PtxLowerer.lower_module()` → regex (`tests/lower/ptx/`) |
| MSL lowering | Same | `tests/lower/msl/` |
| Launcher | Dispatch, caching | Mock driver, verify args (`tests/launcher/`) |
| Registry | Registration gates | Snapshot/restore registry per test via fixture |
| Kernels | Correctness | Automatic via `@kernel` — no manual test files |

## Don't

- Write per-kernel test files (smoke + fuzz handle kernel-level
  correctness).
- Use `torch.allclose` or `max_abs` as a pass/fail gate.
- Put timing in tests — lives in `tools/bench.py`.
- Write manual correctness sweeps — the fuzz harness walks every
  registered problem.
- Hit `device="cuda"` unconditionally — wrap in a CUDA-available
  skip.

See [DEBUGGING.md](DEBUGGING.md) for failure triage.
