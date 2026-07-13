# lattice_su3

SU(3) pure gauge lattice QCD experiments in Python. The code implements lattice
geometry helpers, SU(2)/SU(3) matrix utilities, Wilson-gauge observables,
Metropolis and Cabibbo-Marinari heatbath updates, optional Numba acceleration,
configuration I/O, and time-series analysis tools.

## Project Layout

```text
src/lattice_su3/
  geometry.py          periodic hypercubic lattice indexing and neighbors
  group.py             SU(2) and SU(3) matrix helpers
  update.py            Metropolis, heatbath, and checkerboard update routines
  accelerated.py       optional Numba-JIT heatbath sweep
  observables.py       plaquette, Wilson action, Polyakov loop, staples
  configuration.py     cold/hot starts and NPZ configuration I/O
  thermalization.py    reusable thermalization helpers
  autocorrelation.py   autocovariance, Gamma(t), tau_int helpers

scripts/
  run_chain.py              unified Markov-chain runner and observable writer
  analyze_thermalization.py thermalization plaquette plot from run observables
  auto_correlation.py       autocorrelation analysis from run observables
  autocorrelation_plot.py   plot autocorrelation CSV output
  benchmark_*.py            timing scripts
  *_legacy scripts          older workflows kept for comparison/migration

tests/
  pytest coverage for geometry, group properties, observables, updates,
  configuration I/O, autocorrelation helpers, and script data formats
```

`generate_conf.py` is currently a compatibility import shim for older entrypoints.

## Setup and Validation

Use `uv` for dependency management and command execution.

```bash
uv sync
```

Run the full test suite:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Run focused lint checks for the active scripts:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check scripts src tests
```

Install optional acceleration dependencies with the `acceleration` extra if you
want to use `lattice_su3.accelerated.heatbath_jit_sweep`.

## Unified Run Workflow

The preferred workflow is to generate one reusable run directory, then perform
thermalization and autocorrelation analysis from the same observable history.
This avoids incompatible CSV formats between separate scripts.

Edit script-level parameters in `scripts/run_chain.py`, especially:

```python
SHAPE = (16, 16, 16, 6)
BETA = 5.7
SWEEPS = 300
MEASURE_EVERY = 1
SAVE_CONFIG_EVERY = 0
ALGORITHM = "heatbath"
BACKEND = "jit"
STARTS = ("hot",)
RUN_NAME = ""
```

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_chain.py
```

Output is written to:

```text
results/runs/<run_name>/
  manifest.json
  observables.csv
  configurations/        only when SAVE_CONFIG_EVERY > 0
```

`observables.csv` contains one row per measured sweep:

```csv
chain,start,sweep,average_plaquette,acceptance_rate,accepted_links,attempted_links
```

`SAVE_CONFIG_EVERY = 0` is the default and does not save full gauge
configurations. This is appropriate for thermalization checks and
autocorrelation analysis, where the plaquette history is usually enough.

For production runs, first estimate thermalization and autocorrelation from an
observable-only run. Then set `SAVE_CONFIG_EVERY` to the chosen decorrelation
interval so complete configurations are saved periodically.

## Thermalization Analysis

For cold/hot comparison runs, set in `scripts/run_chain.py`:

```python
STARTS = ("cold", "hot")
SAVE_CONFIG_EVERY = 0
```

Then run the chain and plot plaquette histories:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_chain.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analyze_thermalization.py
```

`scripts/analyze_thermalization.py` reads
`results/runs/<run_name>/observables.csv` and saves a cold/hot plaquette history
plot as `thermalization_plaquette.png` in the same run directory.

## Autocorrelation Analysis

After choosing a thermalization cutoff, edit `scripts/auto_correlation.py`:

```python
RUN_NAME = "<run_name>"
CHAIN = 0
THERMALIZATION_SWEEPS = 100
MAX_LAG = 250
```

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/auto_correlation.py
```

The script discards measurements with `sweep <= THERMALIZATION_SWEEPS`, computes
the normalized autocorrelation `Gamma(t) = C(t) / C(0)`, estimates the running
integrated autocorrelation time, and prints a suggested sampling interval.

The autocorrelation CSV is saved next to the run observables:

```text
results/runs/<run_name>/autocorrelation_chainXX_afterNsweeps.csv
```

## Configuration I/O

Configurations are NumPy NPZ files containing:

- `links`: gauge links with shape `[site, direction, 3, 3]`
- scalar or tuple metadata fields such as shape, beta, sweep, backend, chain,
  start, and save interval

Use:

```python
from lattice_su3 import load_configuration, save_configuration
```

Full configurations are large. For a `16x16x16x6` lattice with four directions,
one complex128 configuration is roughly tens of MB, so avoid saving every sweep
unless that is explicitly required.

## Notes on Older Scripts

`scripts/thermal_check.py`, `scripts/average_plaquette_gen.py`, and
`scripts/generate_configurations.py` are older workflow scripts. They are useful
for comparison and existing tests, but new analysis work should prefer:

1. `scripts/run_chain.py`
2. `scripts/analyze_thermalization.py`
3. `scripts/auto_correlation.py`

New or materially modified scripts should include a short top-of-file
description and a usage command.
