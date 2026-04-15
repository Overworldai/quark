.PHONY: help setup install install-hooks ci \
        test test-v test-x test-k test-file \
        fuzz bench autotune \
        probe-device dump-ptx \
        fmt lint repl clean cleanall

# ── Toolchain ──
# Detect platform: macOS uses Metal (no CUDA_PATH), Linux requires CUDA.
# On Linux we find the latest /usr/local/cuda-* instead of hardcoding.
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  _CUDA_ENV :=
else ifeq ($(UNAME_S),Linux)
  _CUDA_PATH := $(shell ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
  ifeq ($(_CUDA_PATH),)
    $(error No CUDA installation found under /usr/local/cuda-*. Install CUDA or run on macOS.)
  endif
  CUDA_PATH ?= $(_CUDA_PATH)
  _CUDA_ENV := CUDA_PATH=$(CUDA_PATH)
else
  $(error Unsupported platform: $(UNAME_S). Only macOS (Metal) and Linux (CUDA) are supported.)
endif

VENV       := .venv
ENV_PREFIX := $(_CUDA_ENV) VIRTUAL_ENV=$(VENV)
PY         := $(ENV_PREFIX) PYTHONPATH=src python
PYTEST     := $(ENV_PREFIX) $(VENV)/bin/pytest


help:
	@echo "Targets (post-Bundle 6 generic shape):"
	@echo ""
	@echo "  setup                 Full onboarding: venv + install + git hooks"
	@echo "  install               Create venv + install package + dev deps"
	@echo "  install-hooks         Symlink tools/hooks/pre-commit into .git/hooks/"
	@echo "  ci                    Run the pre-commit hook on demand (no commit)"
	@echo ""
	@echo "── Tests (unit + smoke per kernel) ──"
	@echo "  test                  Run pytest tests/ (quiet)"
	@echo "  test KERNEL=name      Restrict kernel smokes to one kernel"
	@echo "  test-v                ... verbose"
	@echo "  test-x                ... stop on first failure"
	@echo "  test-k K=...          Filter pytest by keyword K"
	@echo "  test-file F=...       Run a specific test file"
	@echo ""
	@echo "── Correctness sweeps (registry-driven) ──"
	@echo "  fuzz                  Sweep every registered kernel × every problem"
	@echo "  fuzz KERNEL=name      Restrict to one kernel"
	@echo "  fuzz KERNEL=name PROBLEM=0   Restrict to one (kernel, problem)"
	@echo ""
	@echo "── Perf (registry-driven) ──"
	@echo "  bench                 Bench every kernel against its baselines"
	@echo "  bench KERNEL=name     Restrict to one kernel"
	@echo "  bench KERNEL=name PROBLEM=0  Restrict to one (kernel, problem)"
	@echo ""
	@echo "── Autotune (registry-driven) ──"
	@echo "  autotune KERNEL=name           Tune one kernel across all problems"
	@echo "  autotune KERNEL=name PROBLEM=0 Tune one (kernel, problem)"
	@echo ""
	@echo "── Dev helpers ──"
	@echo "  probe-device          Print DeviceCaps for the current machine"
	@echo "  dump-ptx KERNEL=name [PROBLEM=0]  Print the lowered PTX"
	@echo "  fmt                   Format src/ + tests/ with ruff"
	@echo "  lint                  Lint src/ + tests/ with ruff"
	@echo "  clean                 Remove caches"
	@echo "  cleanall              Remove venv and caches"
	@echo ""
	@echo "Tunables (override on the command line):"
	$(if $(_CUDA_ENV),@echo "  CUDA_PATH=$(CUDA_PATH)")
	@echo ""
	@echo "Adding a new kernel: drop a folder under src/popcorn/kernels/"
	@echo "with a @register(\"name\")-decorated Kernel subclass. Zero"
	@echo "Makefile edits required — every target above is registry-driven."

# `make setup` is the one-shot onboarding command. After cloning,
# the contributor runs `make setup` once and gets a working venv,
# the package installed editable, and the git pre-commit hook
# wired up. From then on, every `git commit` runs the hook.
setup: install install-hooks
	@echo ""
	@echo "✓ popcorn is set up. Try:"
	@echo "    make probe-device   # confirm the GPU caps are sane"
	@echo "    make test           # run the unit + smoke suite"

install:
	uv venv $(VENV)
	VIRTUAL_ENV=$(VENV) uv pip install -e .
	VIRTUAL_ENV=$(VENV) uv pip install pytest

# Install the git pre-commit hook by symlinking. Pulling new
# commits to tools/hooks/pre-commit automatically updates the
# active hook for every contributor — no re-install needed.
install-hooks:
	@if [ ! -d .git ]; then \
		echo "✗ install-hooks: not a git repo (.git/ missing)"; \
		exit 1; \
	fi
	@mkdir -p .git/hooks
	@ln -sf ../../tools/hooks/pre-commit .git/hooks/pre-commit
	@chmod +x tools/hooks/pre-commit
	@echo "✓ installed .git/hooks/pre-commit → tools/hooks/pre-commit"

# Run the same checks the pre-commit hook runs, but on demand
# (without creating a commit). Useful in CI and for debugging.
ci:
	@bash tools/hooks/pre-commit

# ── Tests ──
# Generic targets only. KERNEL=<name> restricts the kernel-smoke
# parametrize to one kernel via `-k smoke_<name>`. Other test
# subdirectories (ir/, lower/, launcher/, etc.) always run.

test:
	$(PYTEST) tests/ $(if $(KERNEL),-k "smoke_$(KERNEL)")

test-v:
	$(PYTEST) tests/ -v $(if $(KERNEL),-k "smoke_$(KERNEL)")

test-x:
	$(PYTEST) tests/ -x -v $(if $(KERNEL),-k "smoke_$(KERNEL)")

test-k:
	$(PYTEST) tests/ -v -k "$(K)"

test-file:
	$(PYTEST) tests/$(F) -v

# ── Correctness sweeps ──
# Registry-driven. Adding a new kernel adds it to the sweep
# automatically — no Makefile edit needed.

fuzz:
	$(PY) tools/fuzz.py $(if $(KERNEL),--kernel $(KERNEL)) $(if $(PROBLEM),--problem $(PROBLEM)) $(if $(TAG),--tag $(TAG))

# ── Bench ──
# Registry-driven. The bench harness consults `popcorn.kernels
# .all_kernels()` and runs each kernel × every problem unless
# KERNEL= / PROBLEM= restrict the sweep.

bench:
	$(PY) tools/bench.py $(if $(KERNEL),--kernel $(KERNEL)) $(if $(PROBLEM),--problem $(PROBLEM)) $(if $(TAG),--tag $(TAG)) $(if $(TAG),,--exclude-tag smoke)

# ── Autotune ──
# Registry-driven. Writes results into the bundled configs/ dir.
# tools/autotune.py is the existing offline genetic search; the
# runtime autotune cache (popcorn.autotune.AutotuneCache) is what
# Launcher.compile() consults at runtime per Bundle 5.

autotune:
	$(PY) tools/autotune.py $(if $(KERNEL),--kernel $(KERNEL)) $(if $(PROBLEM),--problem $(PROBLEM)) $(if $(TAG),--tag $(TAG)) $(if $(TAG),,--exclude-tag smoke)

# ── Dev helpers ──

probe-device:
	$(PY) -c "from popcorn.device import current_device; d = current_device(); print(d.caps); print(d.fingerprint())"

dump-ptx:
	$(PY) -c "import sys; sys.exit(\"dump-ptx not yet wired (see Bundle 7)\")"

fmt:
	uvx ruff format src tests

lint:
	uvx ruff check src tests
	$(ENV_PREFIX) uvx ty check src tests

repl:
	$(PY) -m IPython -i -c "from popcorn import *; from popcorn.tensor import *"

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

cleanall: clean
	rm -rf $(VENV) build dist
