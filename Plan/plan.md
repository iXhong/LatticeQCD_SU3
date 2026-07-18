# Static Potential From Polyakov Correlators

This file records what is already available in the repository and what remains
to do before extracting the static quark-antiquark potential from Polyakov loop
correlators.

## Goal

Extract the lattice static potential from the ensemble-averaged Polyakov loop
correlator,

```text
C(r) = <P(x) P*(x + r)>
a V(r) = -log(C(r)) / N_t
```

up to an additive constant. The first target setup is the existing
`16^3 x 6`, `beta = 5.7` pure-gauge workflow.

## End-to-End Overview

The extraction will proceed from gauge-field generation to a fitted potential
in one reproducible chain:

1. Generate trial Markov chains from cold and hot starts, measuring the average
   plaquette after each sweep. Use these histories to decide when the system is
   thermalized.
2. Estimate autocorrelation from the post-thermalization plaquette history.
   Choose a configuration-saving interval large enough that saved
   configurations are approximately decorrelated.
3. Run a production Markov chain after those choices are fixed. Save full gauge
   configurations at the chosen interval instead of only saving plaquette
   measurements.
4. For each saved configuration, compute the Polyakov loop field `P(x)` over
   all spatial sites. Then compute the translationally averaged correlator
   `C(r_vec) = mean_x P(x) P*(x + r_vec)`.
5. Convert the vector correlator into radial data by folding periodic
   displacement vectors to minimal-image distances and binning all vectors with
   the same `r^2`.
6. Average the binned correlator over the configuration ensemble. Use jackknife
   samples to estimate uncertainties on both `C(r)` and
   `aV(r) = -log(C(r)) / N_t`.
7. Inspect the correlator and potential data. Remove invalid bins such as
   `r = 0`, non-positive correlators, and distances too close to the finite
   volume boundary.
8. Fit the usable potential data to
   `aV(r) = A + B / (r/a) + sigma_a2 * (r/a)`. The fit gives the string tension
   in lattice units, up to later scale setting.
9. Record the run parameters, thermalization cutoff, saving interval, analysis
   outputs, fit range, and validation commands in the run log below.

## Current Repository Status

Done:

- `src/lattice_su3/observables.py` computes normalized Polyakov loops,
  `tr(W) / 3`.
- `src/lattice_su3/observables.py` computes the translationally averaged
  Polyakov loop correlator over spatial displacement vectors.
- Tests cover cold-start values, gauge invariance, center invariance, direct
  correlator comparison, and non-default time directions.
- `scripts/run_chain.py` generates Markov chains and can save full gauge
  configurations when `SAVE_CONFIG_EVERY > 0`.
- Thermalization and autocorrelation scripts already exist for plaquette
  histories.
- Reference notes exist in `notes/polyakov_loop_and_correlator.md` and
  `reference/polyakov_loop_static_potential.md`.

Missing:

- No script currently reads saved configurations and produces Polyakov
  correlator data.
- No radial distance binning exists for spatial displacement vectors.
- No jackknife or bootstrap error analysis exists for the potential.
- No fit workflow exists for `aV(r) = A + B/r + sigma r`.
- Current saved run directories appear to contain plaquette histories, not
  saved configuration files.

## Workflow Plan

### 1. Thermalization Run

Use an observable-only run to estimate the thermalization cutoff.

Edit `scripts/run_chain.py`:

```python
STARTS = ("cold", "hot")
SAVE_CONFIG_EVERY = 0
RUN_NAME = "thermal_beta5p7_16x16x16x6"
```

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_chain.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analyze_thermalization.py
```

Decision to record:

- thermalization cutoff sweep:
- evidence or plot path:

### 2. Autocorrelation Estimate

Use the plaquette history to estimate a practical saving interval.

Edit `scripts/auto_correlation.py`:

```python
RUN_NAME = "thermal_beta5p7_16x16x16x6"
THERMALIZATION_SWEEPS = ...
MAX_LAG = ...
```

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/auto_correlation.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/thinning_autocorrelation.py
```

Decision to record:

- integrated autocorrelation time:
- chosen `SAVE_CONFIG_EVERY`:
- chosen number of saved configurations:

### 3. Production Configuration Run

Generate decorrelated saved configurations for Polyakov analysis.

Edit `scripts/run_chain.py`:

```python
STARTS = ("hot",)
SAVE_CONFIG_EVERY = ...
SWEEPS = ...
RUN_NAME = "polyakov_beta5p7_16x16x16x6"
```

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_chain.py
```

Expected output:

```text
results/runs/polyakov_beta5p7_16x16x16x6/
  manifest.json
  observables.csv
  configurations/*.npz
```

### 4. Implement Radial Correlator Analysis

Add `scripts/analyze_polyakov_static_potential.py`.

Required behavior:

- Load `manifest.json` and saved `configurations/*.npz`.
- Reconstruct `LatticeGeometry` from the run shape.
- For each configuration, compute `polyakov_loop_correlator`.
- Fold periodic spatial displacement vectors to minimal-image separations.
- Bin equivalent displacements by `r^2`.
- Ensemble-average the binned correlator.
- Use the real part of the correlator for the potential.
- Reject or flag bins where `C(r) <= 0`.
- Write a CSV and a plot under the run directory.

Suggested CSV columns:

```text
r_over_a,r2,degeneracy,C_mean,C_err,aV,aV_err,n_configs
```

### 5. Statistical Errors

Use jackknife as the first error method because the potential is nonlinear in
the correlator.

For each distance bin:

```text
C_jackknife[k] = mean over all configurations except k
aV_jackknife[k] = -log(C_jackknife[k]) / N_t
```

Then estimate the standard jackknife error from the leave-one-out samples.

Later optional extension:

- bootstrap resampling
- blocking before jackknife if autocorrelation remains visible

### 6. Fit Static Potential

After stable positive correlator bins exist, fit:

```text
aV(r) = A + B / (r/a) + sigma_a2 * (r/a)
```

Fit rules:

- exclude `r = 0`
- exclude bins with non-positive correlator
- avoid distances close to the finite-volume boundary
- compare on-axis and off-axis bins to check rotational symmetry restoration
- report `sigma_a2`; later convert to physical units only after setting scale

## Implementation Checklist

- [ ] Implement the C++/OpenMP checkerboard heatbath sweep plan below before
      long production runs.
- [ ] Add distance-binning helper in `src/lattice_su3/observables.py` or a new
      analysis module.
- [ ] Add tests for minimal-image displacement folding.
- [ ] Add tests for radial bin degeneracies on small spatial lattices.
- [ ] Add `scripts/analyze_polyakov_static_potential.py`.
- [ ] Add tests for script helper functions using synthetic correlator data.
- [ ] Run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`.
- [ ] Run `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check scripts src tests`.
- [ ] Generate first saved-configuration production run.
- [ ] Produce first `polyakov_static_potential.csv`.
- [ ] Produce first potential plot.
- [ ] Decide fit range and perform first potential fit.

## Physics And Analysis Notes

- The code uses normalized Polyakov loops, `tr(W) / 3`. This changes only the
  additive constant in `aV(r)`, not the force or string tension.
- On `16^3 x 6`, the Polyakov correlator is a finite-temperature static free
  energy. A cleaner zero-temperature static potential needs a larger temporal
  extent or Wilson-loop analysis.
- Polyakov correlators can become noisy quickly. Many saved configurations may
  be needed before large-distance bins are reliable.
- The connected correlator is not the default target here. Use the unconnected
  correlator unless there is a specific reason to subtract `|<P>|^2`.

## Performance Priority: C++/OpenMP Checkerboard Sweep

Recent benchmark results in
`notes/jit_checkerboard_thread_scaling_benchmark.md` show that production-style
runs are still dominated by the heatbath update kernel after JIT acceleration.
For `(16, 16, 16, 6)`, the best measured `jit_checkerboard` update time was
about `0.056 s/sweep` at 16 threads, compared with about `0.102 s/sweep` for
the serial JIT sweep. This is useful, but only about a `1.8x` improvement over
serial JIT and about `2.8x` from one checkerboard thread to sixteen threads.

This limited scaling makes the checkerboard heatbath update the next
optimization priority before expensive production configuration generation.

### Goal

Add a CPU C++/OpenMP backend for the checkerboard heatbath sweep while keeping
the existing Python workflow and script-level backend selection intact.

The first C++ target is the update sweep only. Plaquette acceleration remains a
second priority for dense thermalization diagnostics with `MEASURE_EVERY=1`.

### Public Interface

Keep `scripts/run_chain.py` usage unchanged except for a new backend name:

```python
BACKEND = "cpp_openmp"
```

Add a Python wrapper with the same high-level shape as the current JIT backend:

```python
heatbath_cpp_openmp_checkerboard_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    seed: int | None = None,
) -> UpdateStats
```

The wrapper should:

- validate `beta >= 0`
- require even lattice lengths for checkerboard updates
- pass `links`, neighbor tables, compact parity-site lists, `beta`, and `seed`
  into the C++ extension
- update `links` in place
- return `UpdateStats(attempted_links=volume * ndim, accepted_links=volume * ndim)`

### Implementation Plan

Add a C++ extension module such as `lattice_su3._cpp_openmp` with one low-level
in-place sweep function. Use the existing NumPy data layout:

```text
links[site, mu, row, col]  # complex128
```

The C++ sweep order must be:

```text
for mu in directions:
    for parity in (0, 1):
        OpenMP parallel for site in parity_sites[parity]:
            update link (site, mu)
```

Important implementation details:

- Reuse the current checkerboard physics and Cabibbo-Marinari SU(2) subgroup
  update logic.
- Hand-write the small `3 x 3` complex staple and row-update operations rather
  than calling BLAS for tiny matrices.
- Precompute compact parity-site arrays in Python so the C++ kernel does not
  scan the full volume for each parity.
- Use per-thread or counter-based random streams. Same backend, seed, and
  thread configuration should be reproducible; exact agreement with NumPy or
  Numba trajectories is not required.
- Keep the C++ backend optional. A missing extension should raise a clear
  `ImportError` when `BACKEND="cpp_openmp"` is selected.

### Validation

Required tests:

- C++ sweep reports the same attempted and accepted link counts as other
  heatbath sweeps.
- Updated links remain SU(3) within the existing tolerances.
- Average plaquette and Wilson action remain finite after several small-lattice
  sweeps.
- Odd lattice lengths are rejected for the C++ checkerboard backend.
- `scripts/run_chain.py` accepts `BACKEND="cpp_openmp"` and records it in the
  run manifest.

Required benchmark comparison:

```bash
OMP_NUM_THREADS=1  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_heatbath_acceleration.py
OMP_NUM_THREADS=2  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_heatbath_acceleration.py
OMP_NUM_THREADS=4  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_heatbath_acceleration.py
OMP_NUM_THREADS=8  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_heatbath_acceleration.py
OMP_NUM_THREADS=16 UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/benchmark_heatbath_acceleration.py
```

Benchmark acceptance target on `(16, 16, 16, 6)`:

- C++/OpenMP should beat the best measured `jit_checkerboard` update-only time
  of about `0.056 s/sweep`, or explain why the remaining bottleneck is memory,
  RNG, or measurement overhead.
- If `MEASURE_EVERY=1` is still dominated by `average_plaquette`, schedule a
  follow-up observable acceleration pass.

## Run Log

Use this section to record concrete decisions and outputs.

```text
Date:
Run name:
Shape:
Beta:
Thermalization cutoff:
SAVE_CONFIG_EVERY:
Number of saved configurations:
Analysis output:
Validation:
Notes:
```
