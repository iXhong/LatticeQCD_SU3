# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python project for SU(3) pure gauge lattice QCD experiments.

- `src/lattice_su3/` contains the package implementation: lattice geometry, SU(2)/SU(3) group helpers, update algorithms, observables, thermalization helpers, configuration I/O, autocorrelation utilities, and optional acceleration hooks.
- `scripts/` contains runnable analysis and generation workflows.
- `tests/` contains pytest tests for numerical SU(3) properties, lattice indexing, observables, update algorithms, configuration I/O, and scripts.
- `generate_conf.py` is a compatibility import shim for older entrypoints.
- `pyproject.toml` defines the Python version and dependencies.
- `uv.lock` pins dependency versions for reproducible local runs.
- `reference/` stores background material and notes. Do not treat these files as generated output.

Keep new source code close to the current module until the project grows enough to justify a package layout.

## Build, Test, and Development Commands

Use `uv` for dependency management and command execution.

```bash
uv sync
```

Installs the locked dependencies into the project environment.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Runs the test suite. The explicit cache directory is useful in restricted environments where the home cache may be read-only.

```bash
uv run python generate_conf.py
```

Runs the main module directly if executable examples or diagnostics are added later.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_chain.py
```

Runs the unified Markov-chain workflow, writing a manifest and observable history under `results/runs/`.

## Coding Style & Naming Conventions

Use Python 3.13 syntax and NumPy arrays for numerical work. Prefer clear functions with explicit inputs over hidden global state. Use:

- 4-space indentation.
- `snake_case` for functions, variables, and module names.
- `PascalCase` for classes such as `LatticeGeometry`.
- Type hints for public functions and class attributes where practical.

Every implemented function and method should have a short, clean docstring. Include a one-line purpose plus explicit `Inputs:` and `Outputs:` sections. Keep descriptions practical and avoid long derivations in docstrings; put detailed physics notes in `reference/` instead.

For every new or materially modified script or executable program, add a short module-level header that explains:

- What the file does.
- What inputs or run directory layout it expects.
- How to run it with `uv`, including any important script-level parameters to edit first.

`ruff` is listed as a development dependency; use it for linting once project rules are configured.

## Testing Guidelines

Tests use `pytest`. Name test files `test_*.py` and test functions `test_*`. For numerical checks, use `np.allclose(..., atol=...)` instead of exact equality.

Current tests should verify:

- SU(3) matrices satisfy `X.conj().T @ X ~= I`.
- `det(X) ~= 1`.
- Periodic lattice neighbors wrap correctly at boundaries.
- Precomputed neighbor tables match lookup methods.

## Commit & Pull Request Guidelines

The current history uses concise imperative commit messages, for example:

```text
Implement SU3 lattice geometry basics
```

Keep commits focused and include tests with behavior changes. Commit messages should include a concise imperative subject plus a short summary of the main changes when the commit is more than a trivial single-file edit. The summary should mention user-visible behavior, script/workflow changes, data format changes, and validation performed when applicable.

Pull requests should describe the numerical or algorithmic change, list validation commands, and mention any assumptions about lattice shape, boundary conditions, or random number generation.
