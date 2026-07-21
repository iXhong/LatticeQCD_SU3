# lattice_su3

SU(3) pure gauge lattice QCD experiments in Python. The code implements lattice
geometry helpers, SU(2)/SU(3) matrix utilities, Wilson-gauge observables,
Metropolis and Cabibbo-Marinari heatbath updates, optional Numba acceleration,
configuration I/O, and time-series analysis tools.

## Project Layout

```text
src/lattice_su3/
  geometry.py             periodic hypercubic lattice indexing and neighbors
  group.py                SU(2) and SU(3) matrix helpers
  update.py               Metropolis, heatbath, and checkerboard update routines
  accelerated.py          optional Numba-JIT heatbath sweep
  observables.py          plaquette, Wilson action, Polyakov loops/correlators, staples
  configuration.py        cold/hot starts and NPZ configuration I/O
  thermalization.py       reusable thermalization helpers
  autocorrelation.py      autocovariance, Gamma(t), tau_int helpers
  correlator_analysis.py  spatial/radial/axis binning for Polyakov correlators
  resampling.py           blocking, jackknife, and bootstrap for correlated samples
  static_potential.py     static-potential extraction and correlated Cornell fits
  run_config.py           TOML configuration loaders for production workflows
  run_outputs.py          standard run artifact writers
  chain.py                single-chain update loop used by workflow scripts

scripts/
  workflows/                  TOML-driven thermalization and ensemble workflows
    run_server_chain.py       resumable single-chain production for scheduler tasks
  analysis/                   observable, Polyakov, resampling, and fit analysis
  plotting/                   plot-only or plot-focused postprocessing helpers
  benchmarks/                 timing scripts
  legacy/                     compatibility runners kept outside the main workflow

configs/
  thermalize_16x16x16x6.toml               example thermalization (16x16x16x6, b57)
  thermalize_static_12x12x12x12.toml        example thermalization (12x12x12x12, b57)
  thermalize_smoke_2x2x2x2.toml            smoke-test thermalization
  ensemble_16x16x16x6.toml                 example production ensemble (16x16x16x6)
  ensemble_static_12x12x12x12.toml          example production ensemble (12x12x12x12)
  ensemble_polyakov_hb_or2_server.toml     server-based multi-chain ensemble
  ensemble_polyakov_multichain_smoke_16x16x16x6.toml  multi-chain smoke test
  ensemble_smoke_2x2x2x2.toml              smoke-test ensemble

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

Optionally install Numba-based acceleration for JIT-compiled heatbath sweeps (which I strongly recommend):

```bash
uv sync --extra acceleration
```

Run the full test suite:

```bash
uv run pytest
```

Run focused lint checks for the active scripts:

```bash
uv run ruff check scripts src tests
```

## Configuration-Driven Production Workflow

The preferred static-potential workflow uses small TOML configuration files and
separate Unix-style scripts:

1. `scripts/workflows/thermalize.py` thermalizes one chain and saves checkpoint
   configurations.
2. `scripts/workflows/generate_ensemble.py` starts multiple independent
   production chains from a thermalized checkpoint.
3. `scripts/analysis/auto_correlation.py` analyzes plaquette histories.
4. `scripts/analysis/measure_polyakov_correlators.py` measures vector Polyakov
   loop correlators from saved configurations.

Run the example thermalization workflow:

```bash
uv run python scripts/workflows/thermalize.py \
  configs/thermalize_16x16x16x6.toml
```

Then run the example production ensemble workflow:

```bash
uv run python scripts/workflows/generate_ensemble.py \
  configs/ensemble_16x16x16x6.toml
```

For server or Slurm-array deployments, use `scripts/workflows/run_server_chain.py`
instead. It runs one chain per invocation, writes to isolated `chains/chainNNN`
directories, and supports `--resume` from the last saved checkpoint.

Both scripts write the standard run artifact layout:

```text
results/runs/<run_name>/
  manifest.json
  observables.csv
  configurations/
```

`observables.csv` contains one row per measured plaquette sweep:

```csv
chain,start,sweep,average_plaquette,acceptance_rate,accepted_links,attempted_links
```

Saved configurations are NPZ files under `configurations/` and include metadata
such as `shape`, `beta`, `chain`, `start`, `sweep`, and `run_name`.

### Thermalization Configuration

The thermalization config controls one chain:

```toml
[run]
name = "therm_16x16x16x6_b57_seed12345"
shape = [16, 16, 16, 6]
beta = 5.7
sweeps = 1000
seed = 12345
start = "hot"

[update]
algorithm = "heatbath"
backend = "jit_checkerboard"
overrelaxation_sweeps = 2

[measure]
plaquette_every = 10

[save]
config_every = 100
overwrite = false
```

### Ensemble Configuration

The ensemble config starts several chains from a thermalized checkpoint:

```toml
[run]
name = "prod_static_potential_b57"
shape = [16, 16, 16, 6]
beta = 5.7

[source]
config = "results/runs/therm_16x16x16x6_b57_seed12345/configurations/chain00_hot_sweep001000.npz"

[ensemble]
chains = 4
sweeps_per_chain = 1000
discard_sweeps = 200
seed_base = 20000
parallel = 4

[update]
algorithm = "heatbath"
backend = "jit_checkerboard"
overrelaxation_sweeps = 2

[measure]
plaquette_every = 10

[save]
config_every = 10
overwrite = false
```

The production script writes all chains into one ensemble run directory, with
filenames like `chain00_load_sweep001210.npz` and
`chain01_load_sweep001210.npz`. This keeps downstream analysis independent of
how the chains were launched.

## Legacy General Run Workflow

`scripts/legacy/run_chain.py` remains available as a general and compatibility runner.
It can still generate one reusable run directory and supports command-line
overrides, but new static-potential production work should prefer the TOML
workflow above.

Edit script-level parameters in `scripts/legacy/run_chain.py`, especially:

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
uv run python scripts/legacy/run_chain.py
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

For cold/hot comparison runs, set in `scripts/legacy/run_chain.py`:

```python
STARTS = ("cold", "hot")
SAVE_CONFIG_EVERY = 0
```

Then run the chain and plot plaquette histories:

```bash
uv run python scripts/legacy/run_chain.py
uv run python scripts/analysis/analyze_thermalization.py
```

`scripts/analysis/analyze_thermalization.py` reads
`results/runs/<run_name>/observables.csv` and saves a cold/hot plaquette history
plot as `thermalization_plaquette.png` in the same run directory.

## Autocorrelation Analysis

After choosing a thermalization cutoff, edit `scripts/analysis/auto_correlation.py`:

```python
RUN_NAME = "<run_name>"
CHAIN = 0
THERMALIZATION_SWEEPS = 100
MAX_LAG = 250
```

Run:

```bash
uv run python scripts/analysis/auto_correlation.py
```

The script discards measurements with `sweep <= THERMALIZATION_SWEEPS`, computes
the normalized autocorrelation `Gamma(t) = C(t) / C(0)`, estimates the running
integrated autocorrelation time, and prints a suggested sampling interval.

The autocorrelation CSV is saved next to the run observables:

```text
results/runs/<run_name>/autocorrelation_chainXX_afterNsweeps.csv
```

Two supplementary scripts compare autocorrelation behaviour across runs:

- `scripts/analysis/polyakov_autocorrelation.py` — compares Polyakov-loop
  autocorrelation from legacy runs with `MEASURE_POLYAKOV = True`.
- `scripts/analysis/thinning_autocorrelation.py` — visualises how different
  thinning intervals affect the plaquette autocorrelation function.

## Polyakov Loop Correlator

The observables module provides Polyakov loops and the translationally averaged
unconnected Polyakov loop correlator
`C(r) = mean_x P(x) conj(P(x + r))`. The implementation uses the existing
normalized `tr(W) / 3` Polyakov loop convention, so a cold-start configuration
has `C(r) = 1` for every spatial displacement.

```python
from lattice_su3 import (
    LatticeGeometry,
    load_configuration,
    polyakov_loop_correlator,
)

geometry = LatticeGeometry((16, 16, 16, 6))
links, metadata = load_configuration("config.npz")
correlator = polyakov_loop_correlator(links, geometry)
```

`correlator` has the spatial shape of the lattice, excluding the Euclidean time
direction. It is indexed by periodic displacement vectors. Use ensemble
averaging and any desired distance binning before extracting a static potential,
for example `a V(r) = -log(C(r)) / N_t` up to an additive constant.

For saved run directories, use:

```bash
uv run python scripts/analysis/measure_polyakov_correlators.py \
  --run-name <run_name> --thermalization-sweeps 0
```

The script default `RUN_NAME` is intentionally empty. Set it in the script or
pass `--run-name`; the selected run writes vector correlators under:

```text
results/runs/<run_name>/correlators/polyakov_vector_correlators.npz
```

Continue from the measured vector correlators with separate analysis stages:

```bash
uv run python scripts/analysis/bin_polyakov_correlators.py \
  results/runs/<run_name>/correlators/polyakov_vector_correlators.npz

uv run python scripts/analysis/resample_polyakov_correlators.py \
  results/runs/<run_name>/correlators/polyakov_binned_correlators.npz \
  --block-size 10 --bootstrap-samples 1000 --bootstrap-seed 12345

uv run python scripts/analysis/analyze_static_potential.py \
  results/runs/<run_name>/correlators/polyakov_resampled_correlators.npz \
  --binning axis --method jackknife --r-min 2 --r-max 7
```

The binning stage preserves per-configuration radial and axis correlators. The
resampling stage forms equal-size blocks independently within each chain and
writes delete-one-block jackknife and chain-stratified bootstrap means. The final
stage computes `aV(r)`, its covariance, a correlated
`A + B/r + (sigma*a^2)*r` fit, `r0/a`, and a fit-window scan. Supplying
`--r0-physical-fm` also reports the corresponding lattice spacing. The window
spread is a first fit-range diagnostic, not a substitute for continuum,
finite-volume, temperature, or improved-distance systematic studies.

Plotting helpers for the Polyakov analysis pipeline:

```bash
uv run python scripts/plotting/plot_polyakov_correlator_log.py \
  results/runs/<run_name>/correlators/polyakov_binned_correlators.npz --binning axis

uv run python scripts/plotting/plot_polyakov_loop_plane.py \
  results/runs/<run_name>

uv run python scripts/plotting/plot_static_potential.py \
  results/runs/<run_name>/correlators/static_axis_b05_jk_1_5.npz
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
one complex64 configuration is roughly tens of MB, so avoid saving every sweep
unless that is explicitly required.

