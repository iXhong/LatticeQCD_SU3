# `prod_static_potential_b57_v2` Static-Potential Analysis

Date: 2026-07-20

## Input Data

Cluster run directory:

```text
/dssg/work/zhliu/projects/LatticeQCD_SU3/results/runs/prod_static_potential_b57_v2
```

Correlator directory:

```text
results/runs/prod_static_potential_b57_v2/correlators
```

Measured vector Polyakov correlator file:

```text
polyakov_vector_correlators.npz
```

The measured data contain:

```text
correlators: (2000, 16, 16, 16), complex64
shape: (16, 16, 16, 6)
beta: 5.7
time_direction: -1
chains: 8
configurations per chain: 250
sweeps per chain: 1220..6200
```

The run manifest has:

```text
workflow: generate_ensemble
backend: jit_checkerboard
overrelaxation_sweeps: 2
sweeps: 5200
discard_sweeps: 200
save_config_every: 20
```

## Analysis Steps Run

The radial and axis-only binning had already been produced:

```text
polyakov_binned_correlators.npz
```

Additional chain-aware resampling was run with block sizes 5 and 10:

```bash
.venv/bin/python scripts/resample_polyakov_correlators.py \
  results/runs/prod_static_potential_b57_v2/correlators/polyakov_binned_correlators.npz \
  --block-size 5 --bootstrap-samples 2000 --bootstrap-seed 24680 \
  --output results/runs/prod_static_potential_b57_v2/correlators/polyakov_resampled_block05.npz \
  --overwrite

.venv/bin/python scripts/resample_polyakov_correlators.py \
  results/runs/prod_static_potential_b57_v2/correlators/polyakov_binned_correlators.npz \
  --block-size 10 --bootstrap-samples 2000 --bootstrap-seed 24680 \
  --output results/runs/prod_static_potential_b57_v2/correlators/polyakov_resampled_block10.npz \
  --overwrite
```

Results:

```text
block_size = 5: 400 blocks, no dropped tail configurations
block_size = 10: 200 blocks, no dropped tail configurations
```

Static-potential fits were scanned over:

```text
binning: axis, radial
method: bootstrap, jackknife
block sizes: 5, 10
fit windows: 1-4, 1-5, 2-5, 2-6, 3-6
```

Full fit summary files:

```text
tmp_analysis/static_fit_grid_summary.csv
tmp_analysis/static_fit_grid_summary.md
```

## Main Findings

The existing `static_potential_analysis.npz` used:

```text
binning: axis
method: bootstrap
fit range: r = 3..6
block_size: 1
```

That fit produced a negative string tension:

```text
sigma a^2 = -0.02725
r0/a = nan
```

This is a large-distance instability. The `r=5` and `r=6` axis points have much
larger uncertainties, and the `r=6` central value bends downward relative to a
monotonic confining potential.

Axis-only fits are stable for windows using short and intermediate distances:

```text
axis bootstrap block 10, r=1..4:
sigma a^2 = 0.157300 +/- 0.0118
r0/a = 2.94796 +/- 0.0829
chi2/dof = 0.0181

axis bootstrap block 10, r=1..5:
sigma a^2 = 0.156095 +/- 0.0112
r0/a = 2.95657 +/- 0.0789
chi2/dof = 0.0583

axis bootstrap block 10, r=2..5:
sigma a^2 = 0.144262 +/- 0.0453
r0/a = 2.98236 +/- 0.151
chi2/dof = 0.0439
```

Jackknife gives consistent central values for the same windows:

```text
axis jackknife block 10, r=1..5:
sigma a^2 = 0.155603 +/- 0.0115
r0/a = 2.96013 +/- 0.00588
chi2/dof = 0.0788
```

The jackknife `r0/a` sample error appears much smaller than the bootstrap error
for the same central value. Use the bootstrap uncertainty as the more
conservative quoted statistical error unless a separate resampling validation is
performed.

Radial fits were not selected as the primary result. Their correlated
chi-squared values are large for the tested windows, for example block 10
bootstrap:

```text
r=1..4: chi2/dof = 178.2 / 11
r=1..5: chi2/dof = 196.4 / 19
r=2..5: chi2/dof = 34.29 / 16
```

This suggests the current radial shells contain direction-dependent lattice
artifacts that are not captured by the simple Cornell form and covariance model.

## Recommended Working Result

For this dataset, use the axis-only bootstrap fit with block size 10 and
`r=1..5` as the working result:

```text
sigma a^2 = 0.1561 +/- 0.0112
r0/a = 2.9566 +/- 0.0789
chi2/dof = 0.0583
```

Treat the fit-window spread across stable axis windows as a systematic check:

```text
r0/a roughly 2.95..2.98 for r=1..4, 1..5, 2..5
```

Do not use the `r=3..6` fit as a scale-setting result because it gives
`sigma a^2 < 0`.

## Output Files Added

Resampled correlators:

```text
results/runs/prod_static_potential_b57_v2/correlators/polyakov_resampled_block05.npz
results/runs/prod_static_potential_b57_v2/correlators/polyakov_resampled_block10.npz
```

Static fit grid:

```text
results/runs/prod_static_potential_b57_v2/correlators/static_*_b05_r*.npz
results/runs/prod_static_potential_b57_v2/correlators/static_*_b10_r*.npz
```

Recommended plots:

```text
results/runs/prod_static_potential_b57_v2/correlators/static_axis_bootstrap_b10_r1_5_potential.png
results/runs/prod_static_potential_b57_v2/correlators/static_axis_jackknife_b10_r1_5_potential.png
```

Temporary helper scripts and summaries:

```text
tmp_analysis/inspect_polyakov_analysis.py
tmp_analysis/summarize_static_fit_grid.py
tmp_analysis/static_fit_grid_summary.csv
tmp_analysis/static_fit_grid_summary.md
```

## Caveats

- This is a finite-temperature `16^3 x 6` Polyakov-correlator static free
  energy, not a zero-temperature Wilson-loop static potential.
- The current ensemble has 2000 saved configurations. More production
  configurations should reduce the large-distance noise and allow safer
  inclusion of larger `r`.
- If future analysis uses radial binning, inspect rotational-symmetry breaking
  and shell composition before fitting all radial shells with one Cornell curve.
