# Scripts and Workflow API Refactoring Plan

This document records a proposed refactoring of the repository's executable
scripts. It is a design reference for future work, not a requirement to change
all scripts in one pass.

## Motivation

Several scripts currently use tracked module-level constants as experiment
parameters. A typical workflow is therefore:

1. edit a tracked Python file;
2. run the script;
3. leave an experiment-only Git diff behind;
4. edit the same file again for the next run.

This couples program source code to per-run configuration. It also makes it
harder to reproduce a command, prepare multiple experiments concurrently, and
distinguish algorithm changes from parameter changes during review.

The target design is:

```text
numerical kernels
    -> reusable workflow APIs
        -> CLI commands and thin compatibility scripts
```

Source code remains tracked and normally unchanged between experiments.
Experiment parameters are supplied through a command line or configuration
file, and every run records the final effective configuration with its output.

## Goals

- Allow normal experiments without editing tracked Python files.
- Make chain generation and analysis workflows callable from Python.
- Keep CLI and Python callers on the same implementation path.
- Preserve reproducibility through explicit inputs and recorded provenance.
- Keep numerical kernels independent from CLI parsing and repository paths.
- Make workflow behavior testable using temporary directories and small
  lattices.
- Preserve existing script commands during a gradual migration.

## Non-Goals

- Do not introduce Hydra or a workflow engine before the project needs one.
- Do not expose every helper as a stable top-level public API.
- Do not combine this refactor with changes to the physics algorithms.
- Do not migrate benchmarks merely to make `scripts/` small.
- Do not require a single large rewrite of all scripts.

## Current Issues

The current scripts commonly contain values such as `RUN_NAME`, `SHAPE`,
`BETA`, `SWEEPS`, `THERMALIZATION_SWEEPS`, and `OVERWRITE` at module scope.
This has several consequences:

- experiment-only changes dirty the working tree;
- the shell command does not describe the complete run;
- a stale run name or overwrite setting can affect a later experiment;
- scripts cannot easily be composed from notebooks or other Python code;
- tests may depend on mutable module globals;
- directory layout and numerical work are mixed in the same module;
- duplicated classes such as `SweepRunner` can diverge between scripts.

`scripts/run_chain.py`, for example, currently owns parameter validation,
initial-state loading, sweep dispatch, chain execution, configuration output,
observable output, and manifest generation. That is an application workflow,
not just a command-line wrapper.

## Target Architecture

### Layer 1: Numerical Kernels

Existing modules such as the following remain focused on calculations:

```text
src/lattice_su3/geometry.py
src/lattice_su3/group.py
src/lattice_su3/update.py
src/lattice_su3/observables.py
src/lattice_su3/autocorrelation.py
src/lattice_su3/thermalization.py
src/lattice_su3/configuration.py
```

Functions in this layer should generally:

- receive arrays, geometry objects, random generators, and numerical values;
- return arrays, scalars, or small result objects;
- avoid parsing command-line arguments;
- avoid assuming a repository root or run-directory layout;
- avoid printing as their primary result;
- make file side effects explicit in their names and signatures.

### Layer 2: Workflow APIs

Add a package for reusable experiment orchestration:

```text
src/lattice_su3/workflows/
    __init__.py
    chain.py
    configurations.py
    polyakov.py
    autocorrelation.py
    thermalization.py
```

Workflow modules may understand the run-directory format. They may load a
manifest, select configurations, call numerical kernels, write outputs, and
record analysis provenance.

Each substantial workflow should normally provide:

- an immutable configuration dataclass;
- validation that operates on the configuration object;
- one in-memory calculation API where useful;
- one end-to-end run API for standard file-based operation;
- a result dataclass describing created outputs and important counts;
- explicit `Path` inputs rather than hidden global paths.

Example API shape:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolyakovMeasurementConfig:
    """Configure a file-based Polyakov correlator measurement."""

    run_dir: Path
    thermalization_sweeps: int
    time_direction: int = -1
    config_glob: str = "*.npz"
    overwrite: bool = False


@dataclass(frozen=True)
class PolyakovMeasurementResult:
    """Summarize a completed Polyakov correlator measurement."""

    output_path: Path
    configuration_count: int
    first_sweep: int
    last_sweep: int
    correlator_shape: tuple[int, ...]


def measure_polyakov_run(
    config: PolyakovMeasurementConfig,
) -> PolyakovMeasurementResult:
    """Measure and save correlators for one existing run."""
    ...
```

The low-level and workflow APIs should remain separate:

```python
# In-memory calculation, useful in tests and notebooks.
correlators = measure_correlator_ensemble(paths, geometry, time_direction=-1)

# Standard run-directory workflow with recorded outputs.
result = measure_polyakov_run(config)
```

### Layer 3: CLI

Add a package for command-line parsing and presentation:

```text
src/lattice_su3/cli/
    __init__.py
    main.py
    chain.py
    polyakov.py
    autocorrelation.py
    thermalization.py
```

Register a console entry point in `pyproject.toml`:

```toml
[project.scripts]
lattice-su3 = "lattice_su3.cli.main:main"
```

The intended interface is based on subcommands:

```bash
uv run lattice-su3 run-chain --help
uv run lattice-su3 measure-polyakov --help
uv run lattice-su3 autocorrelation --help
uv run lattice-su3 analyze-thermalization --help
```

Example production measurement:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run lattice-su3 measure-polyakov \
    --run polyakov_correlator \
    --thermalization-sweeps 1000 \
    --time-direction -1
```

Use the standard-library `argparse` module initially. It is sufficient for the
current parameter count and avoids adding a runtime dependency.

### Layer 4: Thin Scripts

During migration, existing commands should remain available. A compatibility
script should only parse arguments or delegate to the CLI:

```python
from lattice_su3.cli.polyakov import main


if __name__ == "__main__":
    main()
```

Once users and documentation consistently use the installed CLI, redundant
compatibility wrappers may be deprecated separately.

## Configuration Strategy

### Command-Line Arguments

Use CLI arguments for values that commonly vary between runs:

- run name or run directory;
- source run and chain;
- lattice shape and beta for generation;
- sweep counts and measurement intervals;
- thermalization cutoff;
- time direction;
- random seed;
- backend and algorithm;
- output path;
- overwrite permission.

CLI defaults must be conservative. In particular, output overwriting should be
disabled unless the caller supplies `--force` or an equivalent explicit flag.

### Optional Configuration Files

Add TOML or JSON configuration only after the basic CLI is stable. TOML is a
reasonable project default because Python 3.13 includes `tomllib` for reading
it. A possible layout is:

```text
configs/
    defaults/
    experiments/
    local/
```

- `defaults/` contains tracked baseline configurations.
- `experiments/` contains tracked, named, reproducible experiment definitions.
- `local/` contains ignored personal or temporary configurations.

Parameter precedence should be explicit and tested:

```text
safe code defaults < configuration file < command-line overrides
```

Do not make configuration-file support a prerequisite for removing the
script-level constants; CLI arguments provide the first useful improvement.

## Provenance and Output Rules

The existing run `manifest.json` is a useful foundation. Analysis workflows
should additionally save their own effective configuration because an analysis
can be run multiple times against one configuration ensemble.

Suggested layout:

```text
results/runs/<run_name>/
    manifest.json
    observables.csv
    configurations/
    analyses/
        polyakov_<analysis_name>/
            manifest.json
            polyakov_vector_correlators.npz
```

An analysis manifest should record at least:

- workflow name and schema version;
- source run directory;
- selected configuration filenames or their sweep range;
- number of selected configurations;
- thermalization cutoff and time direction;
- effective CLI/config parameters;
- output filenames;
- package version and Git commit when available;
- start and completion timestamps;
- overwrite policy;
- validation or completion status.

The output writer should prefer creating a new analysis directory. If a fixed
output path is required, it should reject existing output by default. For
important artifacts, write a temporary file in the destination directory and
replace the final path only after a successful write.

## Script-by-Script Classification

### Move Most Logic Into Workflow APIs

#### `scripts/run_chain.py`

Move into `workflows/chain.py`:

- `SweepRunner`;
- `InitialState`;
- parameter validation;
- run-label and provenance construction;
- initial-state loading;
- one-chain execution;
- manifest and configuration metadata construction;
- the top-level multi-chain workflow.

Retain in CLI/script:

- argument definitions;
- conversion to `ChainConfig`;
- human-readable progress and final summary;
- process exit status.

Avoid duplicating `SweepRunner` in other scripts. It should become one shared
implementation or a lower-level update dispatcher.

#### `scripts/measure_polyakov_correlators.py`

Move into `workflows/polyakov.py`:

- manifest loading and geometry reconstruction;
- deterministic configuration discovery;
- sweep-based filtering;
- per-configuration and ensemble measurement;
- NPZ writing;
- analysis-manifest writing;
- end-to-end `measure_polyakov_run`.

The current in-memory measurement helpers are already close to useful APIs.
Their hidden defaults and repository-root assumptions should be removed.

#### `scripts/auto_correlation.py`

Move into `workflows/autocorrelation.py`:

- observable-history loading;
- chain and thermalization filtering;
- autocorrelation calculation orchestration;
- CSV writing;
- result metadata.

Reuse numerical functions already present in
`src/lattice_su3/autocorrelation.py` rather than duplicating calculations.

#### `scripts/thinning_autocorrelation.py`

Move reusable thinning and spacing logic into the autocorrelation workflow or
a dedicated analysis module. Keep plot presentation separate from the data
calculation when practical.

#### `scripts/analyze_thermalization.py`

Move data loading, chain grouping, and reusable analysis behavior into
`workflows/thermalization.py`. Plot construction can live in a plotting module
if it is shared by multiple commands.

#### `scripts/generate_configurations.py`

Move generation orchestration into `workflows/configurations.py`. Consolidate
its sweep dispatch with `workflows/chain.py`; do not retain a second
`SweepRunner` implementation.

#### `scripts/average_plaquette_gen.py` and `scripts/thermal_check.py`

Determine whether these are still distinct supported workflows after the chain
API is extracted. Prefer expressing their behavior through `run_chains` plus
analysis functions. Preserve a compatibility command only if it represents a
meaningfully different user workflow.

### Move Only Reusable Plot Functions

The following scripts can remain CLI entry points while reusable loading and
plotting functions move under a plotting or workflow module:

- `scripts/autocorrelation_plot.py`;
- `scripts/thermal_plot.py`.

Plotting APIs should accept data and an explicit output path. They should
return the created path and should not rely on module-level run names.

### Keep as Scripts

Benchmarks are executable development tools and do not need to become stable
public APIs:

- `scripts/benchmark_average_plaquette.py`;
- `scripts/benchmark_heatbath_acceleration.py`.

Shared timing or reporting helpers may move only if multiple benchmarks need
them. Benchmark parameters should still use CLI flags so benchmark runs do not
require source edits.

## Public API Policy

Moving a function under `src/` does not automatically make it a top-level API.
Use explicit module imports for workflow functionality:

```python
from lattice_su3.workflows.chain import ChainConfig, run_chains
from lattice_su3.workflows.polyakov import (
    PolyakovMeasurementConfig,
    measure_polyakov_run,
)
```

Keep `lattice_su3/__init__.py` focused on stable numerical primitives and a
small set of widely useful types. Internal helpers should use a leading
underscore where appropriate and should not be re-exported.

Public workflow functions and classes must have type hints and repository-style
docstrings. Configuration and result dataclasses should be immutable when
mutation is unnecessary.

## API Design Rules

### Prefer Configuration Objects

Avoid workflow signatures with many loosely related positional parameters:

```python
run_chain(shape, beta, sweeps, seed, algorithm, backend, start, ...)
```

Prefer:

```python
result = run_chains(ChainConfig(...))
```

Configuration objects are easier to validate, serialize, extend, and compare
with a saved manifest.

### Return Structured Results

Workflow APIs should return values rather than relying on printed output:

```python
result = run_chains(config)
print(result.run_dir)
```

The CLI owns user-facing printing. Tests and notebooks can inspect the result
object directly.

### Make Side Effects Explicit

Use names that distinguish calculation from output:

```text
compute_polyakov_correlators(...)
write_polyakov_correlators(...)
measure_polyakov_run(...)
```

Document which functions create directories, write files, or update arrays in
place.

### Inject Paths and Randomness

Workflow functions should accept `Path` objects instead of deriving everything
from a repository-global `ROOT`. Numerical functions should accept a random
generator or a clearly specified seed when randomness is involved.

### Validate Before Expensive Work

Before starting a long run, validate:

- parameter ranges and compatible choices;
- input run and manifest existence;
- source configuration compatibility;
- selected sweep range and configuration count;
- output collisions;
- backend availability;
- checkerboard geometry constraints where applicable.

### Support Dry Runs

Long-running or file-writing commands should eventually support `--dry-run`.
The command should report resolved inputs, selected configuration count,
effective parameters, and output targets without performing the computation.

## Testing Strategy

### Numerical Tests

Continue testing kernels independently with small arrays and lattices. Workflow
refactoring must not change established numerical tolerances.

### Workflow Tests

Use temporary directories and small configurations to test:

- configuration validation;
- manifest parsing and compatibility checks;
- deterministic file selection;
- thermalization filtering;
- overwrite refusal;
- output schema and metadata;
- returned result objects;
- failures with missing or malformed inputs.

### CLI Tests

Add focused integration tests for:

- every command's `--help` output;
- required arguments;
- invalid choices and useful errors;
- one small successful invocation;
- config-file precedence if configuration files are added;
- `--dry-run` and `--force` behavior.

### Compatibility Tests

While wrapper scripts remain, verify that invoking an old script and invoking
the corresponding CLI reach the same workflow implementation and produce the
same output schema.

## Incremental Migration Plan

### Phase 1: Establish the Pattern With Polyakov Measurement

- [ ] Add `src/lattice_su3/workflows/`.
- [ ] Add `PolyakovMeasurementConfig` and `PolyakovMeasurementResult`.
- [ ] Move file discovery, filtering, measurement, and writing from the script.
- [ ] Remove experiment-specific module constants from the implementation.
- [ ] Add an `argparse` CLI with safe defaults and `--force`.
- [ ] Keep `scripts/measure_polyakov_correlators.py` as a compatibility entry.
- [ ] Save a measurement manifest beside the correlator output.
- [ ] Move existing helper tests to workflow tests and add CLI coverage.

This is the preferred first migration because the workflow is bounded, already
tested, and was recently exercised with real production data.

### Phase 2: Autocorrelation and Thermalization Analysis

- [ ] Extract observable-history readers shared by analysis workflows.
- [ ] Move autocorrelation orchestration into a workflow module.
- [ ] Move thinning analysis into a reusable API.
- [ ] Separate reusable plots from CLI-specific path selection.
- [ ] Make run name, chain, cutoff, and maximum lag CLI parameters.
- [ ] Save analysis parameters and source observable metadata.

### Phase 3: Chain Generation

- [ ] Introduce `ChainConfig`, `ChainResult`, and shared initial-state types.
- [ ] Consolidate duplicated sweep dispatch.
- [ ] Move run-directory and manifest behavior into `workflows/chain.py`.
- [ ] Expose a tested `run_chains(config)` API.
- [ ] Add `lattice-su3 run-chain`.
- [ ] Convert `run_chain.py` to a thin wrapper.
- [ ] Re-evaluate `thermal_check.py`, `average_plaquette_gen.py`, and
      `generate_configurations.py` against the shared API.

### Phase 4: Unified CLI and Documentation

- [ ] Register the `lattice-su3` console command in `pyproject.toml`.
- [ ] Add consistent subcommands and shared path/verbosity options.
- [ ] Update script module headers and repository usage documentation.
- [ ] Add examples for shell, Python, and notebook use.
- [ ] Define a compatibility/deprecation policy for old script names.

### Phase 5: Optional Experiment Configuration

- [ ] Decide whether repeated parameter sets justify TOML configurations.
- [ ] Define and validate the configuration schema.
- [ ] Implement and test the precedence rules.
- [ ] Add tracked example experiment configurations.
- [ ] Ignore only the documented local-configuration directory.

### Phase 6: Optional Workflow Engine

Consider Hydra only when parameter composition and parameter sweeps become a
recurring burden. Consider Snakemake only when generation, autocorrelation,
measurement, binning, error estimation, fitting, and plotting form a stable
dependency graph that benefits from caching or cluster scheduling.

These tools should solve an observed scaling problem rather than being a
prerequisite for this refactor.

## Compatibility and Rollout

- Preserve output schemas during the first extraction unless a schema change
  is explicitly planned and versioned.
- Keep old script filenames operational through wrappers for at least one
  development cycle.
- Print a deprecation notice only after the unified CLI is documented and
  tested.
- Avoid changing physics defaults silently. If a default becomes unsafe, make
  the argument required or fail with a clear message.
- Refactor one workflow per focused commit and run the complete test suite after
  each extraction.
- Do not mix backend optimization, observable changes, and CLI refactoring in
  the same commit.

## Definition of Done for One Migrated Workflow

A workflow is considered migrated when:

- users can run it without modifying tracked source files;
- Python callers can invoke it with a small documented API;
- CLI and Python calls share the same implementation;
- configuration validation happens before expensive work;
- inputs and outputs are explicit;
- overwrite is disabled by default;
- the final effective configuration is stored with the result;
- core behavior is covered by workflow tests;
- the CLI has help and at least one small integration test;
- the compatibility script, if retained, contains no duplicated workflow
  logic;
- existing numerical behavior and output data are validated.

## Suggested Validation Commands

Run these after each migration stage:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check scripts src tests
UV_CACHE_DIR=/tmp/uv-cache uv run lattice-su3 <command> --help
```

For Polyakov measurement, also compare a migrated run with the current output
using the same saved configuration ensemble and require matching sweeps,
metadata, shapes, and correlator values within the established numerical
tolerance.

## Recommended First Implementation Slice

The first future refactoring task should be limited to
`measure_polyakov_correlators.py`:

1. create `workflows/polyakov.py`;
2. introduce configuration and result dataclasses;
3. move the existing tested helpers with minimal behavioral changes;
4. add explicit `Path` inputs and safe overwrite handling;
5. add an `argparse` entry point;
6. preserve the old script as a wrapper;
7. record a measurement manifest;
8. validate against the existing `polyakov_correlator` run output.

That slice establishes the architecture without coupling the work to the much
larger chain-generation refactor.
