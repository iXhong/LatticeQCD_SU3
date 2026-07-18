# Polyakov Correlator Production Pilot Notes

Date: 2026-07-18

## Goal

Record the pilot tests used to decide how to generate gauge configurations for
Polyakov-loop correlator static-potential extraction.

The practical question was whether the newer TOML-based multi-chain production
workflow can reproduce the older single-chain `16^3 x 6` Polyakov correlator
behavior before committing to a larger production run.

## Relevant Workflows and Files

Existing single-chain reference run:

```text
results/runs/polyakov_correlator
```

Reference source configuration:

```text
results/runs/check_thermal/configurations/chain00_hot_sweep001000.npz
```

New multi-chain smoke-test run:

```text
results/runs/polyakov_multichain_smoke_16x16x16x6_b57
```

New `12^4` exploratory run:

```text
results/runs/prod_static_polyakov_12x12x12x12_b57
```

Important scripts used:

```text
scripts/generate_ensemble.py
scripts/measure_polyakov_correlators.py
scripts/bin_polyakov_correlators.py
scripts/resample_polyakov_correlators.py
scripts/analyze_static_potential.py
```

Important source helpers:

```text
src/lattice_su3/observables.py
src/lattice_su3/correlator_analysis.py
src/lattice_su3/resampling.py
src/lattice_su3/static_potential.py
src/lattice_su3/autocorrelation.py
```

## Precision Change

Gauge configurations and Polyakov correlator outputs were changed to use
`complex64` for storage. The goal was to reduce disk usage. The update leaves
many internal matrix calculations in their existing precision path while storing
large persistent arrays at single precision.

For `12^4`, saved configurations had:

```text
links shape: (20736, 4, 3, 3)
links dtype: complex64
```

For `16^3 x 6`, saved configurations had:

```text
links shape: (24576, 4, 3, 3)
links dtype: complex64
```

## `12^4` Polyakov-Correlator Pilot

### Thermalization

Configuration:

```text
configs/thermalize_static_12x12x12x12.toml
```

Parameters:

```toml
shape = [12, 12, 12, 12]
beta = 5.7
sweeps = 1500
algorithm = "heatbath"
backend = "jit_checkerboard"
overrelaxation_sweeps = 2
plaquette_every = 10
config_every = 150
```

Result:

```text
run: therm_12x12x12x12_b57_seed12345
final sweep: 1500
saved configurations: 10
elapsed: 188.8 s
```

Thermalization/autocorrelation check after discarding sweeps `<= 400`:

```text
sweeps analyzed: 410..1500
samples: 110
mean plaquette: 0.54911719
std plaquette: 0.00138340
window lag: 2
tau_int: 0.6179 measured intervals
```

Because plaquette was measured every 10 sweeps, this corresponds to roughly:

```text
tau_int ≈ 6.2 sweeps
suggested spacing ≈ 2 measured intervals ≈ 20 sweeps
```

### Production

Configuration:

```text
configs/ensemble_static_12x12x12x12.toml
```

Parameters:

```toml
shape = [12, 12, 12, 12]
beta = 5.7
chains = 4
sweeps_per_chain = 2000
discard_sweeps = 400
save.config_every = 20
measure.plaquette_every = 0
algorithm = "heatbath"
backend = "jit_checkerboard"
overrelaxation_sweeps = 2
```

Result:

```text
run: prod_static_polyakov_12x12x12x12_b57
total configurations: 320
per chain: 80 configurations
sweep range per chain: 1920..3500
spacing: 20 sweeps
elapsed: 900.8 s
directory size: 1.8G
```

The first saved production sweep is `1920` because the source sweep is `1500`,
`discard_sweeps = 400`, and `save_config_every = 20`.

### Analysis Results

Workflow:

```text
measure vector Polyakov correlators
bin axis/radial correlators
block by chain with block_size = 5
build jackknife/bootstrap samples
fit Cornell form A + B/r + sigma*r
```

Vector correlator output:

```text
polyakov_vector_correlators.npz
shape: (320, 12, 12, 12)
```

Resampling:

```text
block size: 5 configurations = 100 sweeps
blocks: 64
dropped tails: 0
```

Axis correlator signs showed a serious signal problem:

```text
r = 1: positive
r = 2: positive
r = 3: negative
r = 4: negative
r = 5: positive
r = 6: positive
```

Static-potential fits:

| Binning | Method | Fit range | `sigma*a^2` | `chi2/dof` | `r0/a` |
| --- | --- | ---: | ---: | ---: | ---: |
| axis | jackknife | `1..6` | `-0.0412` | `0.0796/1` | `nan` |
| radial | jackknife | `1..3` | `0.00735` | `1.764/4` | `13.82` |
| radial | jackknife | `1..4` | `-0.0195` | `1.903/6` | `nan` |

Interpretation:

- The workflow ran correctly.
- The `12^4` Polyakov correlator signal was too noisy for reliable static
  potential extraction.
- `Nt = 12` suppresses the correlator as `C(r) ~ exp[-V(r) Nt]`, making
  long-distance signals much harder than at `Nt = 6`.
- The positive `r0/a` from radial `r=1..3` is not physically useful because it
  is larger than the spatial box scale.

Diagnostic plots:

```text
results/runs/prod_static_polyakov_12x12x12x12_b57/correlators/plots/
```

Key files:

```text
static_potential_axis_jackknife_r1_6.png
static_potential_radial_jackknife_r1_3.png
polyakov_correlator_sign_overview.png
polyakov_correlator_abs_log_sign_excluding_r0.png
```

## `16^3 x 6` Reference and Multi-Chain Smoke Test

### Existing Single-Chain Reference

Run:

```text
results/runs/polyakov_correlator
```

Manifest summary:

```text
shape: [16, 16, 16, 6]
beta: 5.7
sweeps: 1000
measure_every: 10
save_config_every: 20
algorithm: heatbath
backend: jit_checkerboard
source: results/runs/check_thermal/configurations/chain00_hot_sweep001000.npz
```

The reference run is a single chain from sweep `1000` to `2000`, saving:

```text
50 configurations
sweeps: 1020..2000
```

Reference axis correlator central values:

| `r/a` | `Re C_axis(r)` |
| ---: | ---: |
| 0 | `1.110987e-01` |
| 1 | `5.374965e-03` |
| 2 | `9.617778e-04` |
| 3 | `1.400708e-04` |
| 4 | `1.334133e-04` |
| 5 | `1.623009e-04` |
| 6 | `-3.332227e-05` |
| 7 | `-7.855848e-06` |
| 8 | `-1.322685e-04` |

Reference axis Cornell fits over `r=1..5`:

| Block size | `sigma*a^2` | `chi2/dof` | `r0/a` |
| ---: | ---: | ---: | ---: |
| 2 | `0.106575` | `2.759/2` | `3.474` |
| 5 | `0.089893` | `2.127/2` | `3.746` |
| 10 | `0.058763` | `0.2009/2` | `5.897` |

### Check of `check_thermal` as Source

Run:

```text
results/runs/check_thermal
```

It contains both hot and cold thermalization chains to sweep `1000`.

The hot source:

```text
results/runs/check_thermal/configurations/chain00_hot_sweep001000.npz
```

is usable as the source for `16^3 x 6` Polyakov production. This is also the
same source recorded in the existing `polyakov_correlator` manifest.

### Multi-Chain Smoke Test

Configuration:

```text
configs/ensemble_polyakov_multichain_smoke_16x16x16x6.toml
```

Parameters:

```toml
shape = [16, 16, 16, 6]
beta = 5.7
source = "results/runs/check_thermal/configurations/chain00_hot_sweep001000.npz"
chains = 4
sweeps_per_chain = 400
discard_sweeps = 0
save.config_every = 20
measure.plaquette_every = 0
algorithm = "heatbath"
backend = "jit_checkerboard"
overrelaxation_sweeps = 0
```

Important: `overrelaxation_sweeps = 0` was required to match the old single-chain
reference. The old reference sweep attempted-link count corresponds to one
heatbath pass only.

Result:

```text
run: polyakov_multichain_smoke_16x16x16x6_b57
chains: 4
configurations: 80
per chain: 20
sweep range per chain: 1020..1400
spacing: 20 sweeps
elapsed: 85.2 s
directory size: 541M
```

### Multi-Chain Analysis

Workflow:

```text
measure vector Polyakov correlators
bin axis/radial correlators
resample with block_size = 2, 5, 10
fit axis Cornell form over r = 1..5
```

Axis Cornell fits:

| Run | Block size | `A` | `B` | `sigma*a^2` | `chi2/dof` | `r0/a` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single | 2 | `1.127346` | `-0.363814` | `0.106575` | `2.759/2` | `3.474` |
| single | 5 | `1.167918` | `-0.388578` | `0.089893` | `2.127/2` | `3.746` |
| single | 10 | `1.112942` | `0.393344` | `0.058763` | `0.2009/2` | `5.897` |
| multi | 2 | `0.838687` | `-0.166283` | `0.198855` | `0.4333/1` | `2.732` |
| multi | 5 | `0.842402` | `-0.167296` | `0.196364` | `0.2503/1` | `2.748` |
| multi | 10 | `0.865039` | `-0.182956` | `0.189319` | `0.1593/1` | `2.784` |

Axis correlator comparison:

| `r/a` | single `Re C` | multi `Re C` | ratio multi/single |
| ---: | ---: | ---: | ---: |
| 0 | `1.110987e-01` | `1.110670e-01` | `1.000` |
| 1 | `5.374965e-03` | `5.355733e-03` | `0.996` |
| 2 | `9.617778e-04` | `9.937925e-04` | `1.033` |
| 3 | `1.400708e-04` | `2.277684e-04` | `1.626` |
| 4 | `1.334133e-04` | `9.684826e-05` | `0.726` |
| 5 | `1.623009e-04` | `1.241428e-05` | `0.076` |
| 6 | `-3.332227e-05` | `1.814086e-05` | `-0.544` |
| 7 | `-7.855848e-06` | `-1.829789e-05` | `2.329` |
| 8 | `-1.322685e-04` | `5.314660e-05` | `-0.402` |

Interpretation:

- Short distances `r=1,2` agree well between the single-chain reference and the
  multi-chain smoke test.
- Distances `r=3,4` are still the same order but already show visible statistical
  fluctuation.
- Distances `r>=5` are noise dominated in both small runs.
- The multi-chain workflow gives a sensible rising potential and stable positive
  string tension across block sizes `2,5,10`.
- No evidence was found that multi-chain parallel generation is broken.
- The comparison is limited by statistics, especially at long distance.

Comparison plots:

```text
results/runs/polyakov_multichain_smoke_16x16x16x6_b57/correlators/plots/
```

Key files:

```text
axis_potential_single_vs_multichain_block05.png
axis_correlator_single_vs_multichain_log.png
```

## Autocorrelation of Polyakov Correlator Inputs

Since production did not measure plaquette, autocorrelation was evaluated on the
actual analysis inputs:

```text
Re C_axis(r), r = 1..5
```

using `src/lattice_su3/autocorrelation.py`.

Summary output:

```text
results/runs/polyakov_multichain_smoke_16x16x16x6_b57/correlators/axis_correlator_autocorrelation_summary.csv
```

Diagnostic plot:

```text
results/runs/polyakov_multichain_smoke_16x16x16x6_b57/correlators/plots/axis_correlator_autocorrelation_gamma.png
```

Multi-chain smoke-test integrated autocorrelation times, in saved-configuration
units:

| Observable | Mean `tau_int` | Median `tau_int` | Max `tau_int` | Suggested interval |
| --- | ---: | ---: | ---: | ---: |
| `axis_C_r1_re` | `0.707` | `0.675` | `0.977` | `1..2 configs` |
| `axis_C_r2_re` | `0.500` | `0.500` | `0.500` | `1 config` |
| `axis_C_r3_re` | `0.500` | `0.500` | `0.500` | `1 config` |
| `axis_C_r4_re` | `0.601` | `0.565` | `0.775` | `1..2 configs` |
| `axis_C_r5_re` | `0.669` | `0.680` | `0.817` | `1..2 configs` |

Because configurations are saved every 20 sweeps:

```text
tau_int ≈ 10..20 sweeps
suggested interval ≈ 20..40 sweeps
```

Interpretation:

- `save_config_every = 20` is already a reasonable spacing for the measured
  Polyakov correlator inputs.
- `block_size = 2` corresponds to 40 sweeps and is a plausible minimal blocking
  choice.
- `block_size = 5` corresponds to 100 sweeps and is conservative.
- The estimate is noisy because the smoke test has only 20 configurations per
  chain.

## Current Conclusions

1. The `12^4` Polyakov correlator route is substantially harder because `Nt=12`
   suppresses the correlator. The pilot data did not support a reliable static
   potential fit.

2. The `16^3 x 6` route gives much better signal for the current Polyakov
   correlator workflow, consistent with the earlier `dev/` and single-chain
   results.

3. `check_thermal` is a valid source for the `16^3 x 6` continuation runs.

4. The TOML-based multi-chain production workflow reproduces short-distance
   Polyakov correlator behavior and produces sensible static-potential fits in
   the smoke test.

5. Long-distance correlators remain statistics limited. Larger production should
   focus on increasing total configurations rather than changing the analysis
   pipeline first.

## Recommended Next Production Run

Use the `16^3 x 6` source:

```text
results/runs/check_thermal/configurations/chain00_hot_sweep001000.npz
```

Recommended starting point:

```toml
shape = [16, 16, 16, 6]
beta = 5.7
algorithm = "heatbath"
backend = "jit_checkerboard"
overrelaxation_sweeps = 0
chains = 4
sweeps_per_chain = 2000
discard_sweeps = 0
save.config_every = 20
measure.plaquette_every = 0
```

This would save:

```text
4 chains * (2000 / 20) = 400 configurations
```

For a stronger production run:

```toml
chains = 4
sweeps_per_chain = 3000
```

which would save:

```text
600 configurations
```

For analysis:

- Use axis binning first.
- Fit the primary window `r=1..5`.
- Compare block sizes `2`, `5`, and `10`.
- Re-evaluate autocorrelation after the larger production run, especially for
  `r=4,5`.

