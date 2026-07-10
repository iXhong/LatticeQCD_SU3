# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Python project for SU(3) pure gauge lattice QCD experiments.

- `generate_conf.py` contains the current implementation: lattice geometry, periodic neighbor tables, and SU(2)/SU(3) matrix helpers.
- `test_su3.py` contains pytest tests for numerical SU(3) properties and lattice indexing.
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

## Coding Style & Naming Conventions

Use Python 3.13 syntax and NumPy arrays for numerical work. Prefer clear functions with explicit inputs over hidden global state. Use:

- 4-space indentation.
- `snake_case` for functions, variables, and module names.
- `PascalCase` for classes such as `LatticeGeometry`.
- Type hints for public functions and class attributes where practical.

Every implemented function and method should have a short, clean docstring. Include a one-line purpose plus explicit `Inputs:` and `Outputs:` sections. Keep descriptions practical and avoid long derivations in docstrings; put detailed physics notes in `reference/` instead.

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

Keep commits focused and include tests with behavior changes. Pull requests should describe the numerical or algorithmic change, list validation commands, and mention any assumptions about lattice shape, boundary conditions, or random number generation.
