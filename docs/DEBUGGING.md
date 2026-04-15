# Debugging

Symptom → recipe.

## Kernel fails correctness

```bash
make fuzz KERNEL=name PROBLEM=name     # reproduce
```

Bench / fuzz now print the root-cause exception type + traceback for
launch and tensor-setup failures; read it before starting recipes.

1. NaN / Inf? `PT.has_nan(out)`, `PT.all_finite(out)`,
   `PT.first_non_finite_index(out)` — single call tells you where.
2. IR inspect — `print_module(kernel.emit())` (ir/printer.py) —
   compare the structural shape to your mental model before going
   to the lowered text.
3. Dump lowered:
   ```bash
   make dump-ptx KERNEL=name    # CUDA
   # (MSL dump is via `kernel.emit()` through `MslLowerer(...).lower_module(...).msl`)
   ```
4. Element-level diff: `PT.abs_diff_stats(out, ref)` → mean / max.
   Large `max_abs` with tiny `mean_abs` → one corrupted tile; look
   at the first bad element's `(m_base, n_base, lane)` to trace
   back to which MMA tile or epilogue store.
5. Shrink to the minimum failing problem — bench's `PROBLEM=` flag
   restricts to one.

### Common causes

- Missing `barrier("block")` between smem write and read.
- Wrong smem padding → bank conflicts → wrong values.
- Wrong `cd_offsets` → fragment elements mapped to wrong positions.
- Off-by-one in tile base (`m_base`, `n_base`).
- Accumulator rounding — try F32 output to see if it's precision.
- FragReduceOp on a non-ACC layout — reductions only work on
  accumulator-layout tiles.
- Shuffle on MSL with the wrong xor distance — Apple's 2×2×2
  bit-swizzle is distinct from PTX's lane layout.
- **MMA shape mismatch after autotune** — a kernel's `_mma_cfg`
  reads `config.main_shape`; a stale tuned JSON from before the
  MMA_SHAPES migration will have `main_shape=""` and quietly fall
  back to the compute-dtype default. Re-run `make autotune
  KERNEL=name` to refresh.

## Kernel fails to compile

```bash
POPCORN_PRINT_FAILED_SOURCE=1 make fuzz KERNEL=name
```

Prints the lowered source (PTX or MSL) that the driver rejected.

Common errors:
- **PTX "type mismatch"** — register type doesn't match instruction.
  Check the IR op's DType.
- **PTX "unknown instruction"** — wrong PTX version for the target
  SM. Check `PtxLowerer(ptx_version, target_sm)`.
- **PTX "operand type mismatch in mma"** — fragment register count
  wrong. Check `MmaShape.a_regs / b_regs / c_regs` vs the ISA spec.
- **MSL "as_type cast from X to Y is not allowed"** — usually a
  vec_load / vec_store packing width mismatch. The MSL lowerer
  computes the buffer `uintN` type from the pack byte width — check
  `visitors.py::_visit_vec_load / _visit_vec_store`.
- **MSL exceeds `max_ir_ops`** — config explores too many
  unrolled MMA tiles for the MSL JIT. Bump `max_ir_ops` in
  `drivers/mlx.py` or reduce `MTiles × NCW` in the config.

## Kernel hangs

- **Divergent barrier** — some threads hit `barrier("block")`,
  others don't. Barriers must be unconditional across all threads
  of the block.
- **Infinite loop** — K-loop bound wrong; `K_outer_iters == 0` or
  `K // BK` computing as something huge.
- **cp.async without wait** — commit without matching
  `async_wait(n)` freezes subsequent barriers on some drivers.

## Kernel is slow

```bash
make bench KERNEL=name TAG=smoke
```

- Tuned vs default: `POPCORN_FORCE_DEFAULT_CONFIG=1 make bench`
  — if tuned is slower, re-autotune (kernel-emit refactors routinely
  invalidate old configs).
- Occupancy: too much smem → fewer blocks / SM.
  `kernel.smem_estimate()` runs the `smem_layout` pass and returns
  the post-aliasing byte count.
- Inner loop: MMA bottleneck vs tile-load bottleneck. Swap in a
  no-op MMA (`MmaBody` with a `.map(lambda x: x)` post-pass) to
  check.
- Wave tail — grid not a multiple of SM count / MLX dispatch groups.

## Wrong results only on some configs

Run `make fuzz KERNEL=name` with the full sweep. If only certain
(BM, BN, BK) fail:
- `is_valid()` should reject the config before it compiles.
- Tile load: `(rows * cols) % n_threads != 0` causes partial loads.
- Smem size: allocation pass should size correctly — if
  `smem_estimate()` under-counts for a config, the `Lifetime`
  probably isn't tight.

## Environment flags

| Variable | Effect |
|---|---|
| `POPCORN_DISABLE_PERF_WARNINGS=1` | silences validator `PerfWarning` |
| `POPCORN_DISABLE_CSE=1` | disables Builder CSE (debug only) |
| `POPCORN_FORCE_DEFAULT_CONFIG=1` | skips tuned JSON lookup in bench/fuzz |
| `POPCORN_PRINT_FAILED_SOURCE=1` | prints lowered text on compile failure |
| `POPCORN_CONFIGS_DIR` | override `configs/` path |

## Bench / fuzz error surfacing

Both tools print `type(e).__name__: str(e)` for exceptions thrown
during tensor setup / compile / launch. Truncated-to-30-char errors
were retired; you'll see the full traceback on launch failures via
`traceback.print_exc()`.

If you're adding a new failure path, the rule is **surface the root
cause**. Don't catch-and-return `"invalid"`.
