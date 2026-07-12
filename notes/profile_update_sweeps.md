# Profile Update Sweeps

Date: 2026-07-12

## Goal

Identify the most time-consuming parts of the current code before optimizing for
future `16 x 16 x 16 x 6` lattice runs.

## Commands

The profiling runs used `PYTHONPATH=src` with the locked project environment:

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python -c '...'
```

The benchmark compared initialization, observables, and one update sweep on
smaller 4D lattices with the same shape pattern.

## Timing Summary

Measured wall times:

| Shape | hot_start | average_plaquette | polyakov_loops | heatbath_sweep | metropolis_sweep |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(4, 4, 4, 6)` | 0.089535 s | 0.001136 s | 0.001940 s | 0.317607 s | 0.224981 s |
| `(6, 6, 6, 6)` | 0.279335 s | 0.003766 s | 0.006218 s | 1.069367 s | 0.753336 s |

The dominant runtime is the update sweep. Observable measurements are currently
small compared with either heatbath or Metropolis sweeps.

Approximate volume-scaled estimates for `(16, 16, 16, 6)`:

| Operation | Estimated time |
| --- | ---: |
| `hot_start` | ~5.3 s once |
| `average_plaquette` | ~0.07 s per measurement |
| `polyakov_loops` | ~0.12 s per measurement |
| `heatbath_sweep` | ~20 s per sweep |
| `metropolis_sweep` | ~14 s per sweep |

These estimates assume roughly linear scaling with lattice volume and should be
rechecked on the target lattice after each substantial optimization.

## Heatbath Profile

Profiled one `heatbath_sweep` on `(4, 4, 4, 6)`.

Top cumulative-time entries:

```text
532532 function calls in 0.466 seconds

ncalls  cumtime  function
     1    0.466  heatbath_sweep
  1536    0.465  heatbath_update_link
  4608    0.191  sample_su2_heatbath
  9216    0.081  np.ix_
  4608    0.079  embed_su2
  9216    0.069  su2_matrix_from_coefficients
  1536    0.061  staple
  4608    0.059  su2_effective_staple
  4608    0.034  np.linalg.det
 23040    0.021  dagger
```

Main heatbath hotspots:

- `sample_su2_heatbath`
- repeated tiny matrix construction in `su2_matrix_from_coefficients`
- repeated `np.ix_` and `embed_su2` allocations
- `staple` inside every link update

## Metropolis Profile

Profiled one `metropolis_sweep` on `(4, 4, 4, 6)`.

Top cumulative-time entries:

```text
331779 function calls in 0.317 seconds

ncalls  cumtime  function
     1    0.317  metropolis_sweep
  1536    0.316  metropolis_update_link
  1536    0.160  su3_metropolis_proposal
  3072    0.150  wilson_local_action
  3072    0.114  staple
  4608    0.082  random_su2_near_identity
  4608    0.069  embed_su2
  4608    0.040  np.ix_
 36864    0.029  dagger
```

Main Metropolis hotspots:

- `su3_metropolis_proposal`
- `wilson_local_action`
- `staple`
- repeated tiny SU(2)-to-SU(3) embedding allocations

## Initial Optimization Targets

The first optimization pass should focus on update internals, not measurement
observables:

1. Reduce repeated tiny array allocations in `embed_su2`, `np.ix_`, `np.eye`,
   and `su2_matrix_from_coefficients`.
2. Optimize `staple`, because it is shared by heatbath and Metropolis paths.
3. Add deterministic regression coverage before changing update logic, such as a
   fixed-seed short trajectory test that checks final plaquette/action values.
4. Re-profile after each change on the same small shapes, then confirm scaling
   on the target `(16, 16, 16, 6)` lattice.

## First Heatbath Optimization Pass

The first optimization pass kept the default heatbath sweep order unchanged and
reduced allocation-heavy work inside each Cabibbo-Marinari subgroup update:

- Replaced `np.ix_` active-block extraction with scalar 2x2 block entries.
- Replaced `embed_su2(... ) @ link` with direct active-row updates.
- Replaced `np.linalg.det` of SU(2)-form effective staples with coefficient
  norm calculation.
- Used neighbor tables directly inside `staple`.
- Added a separate checkerboard heatbath sweep for later parallel/GPU work.

Measured wall times after the optimization:

| Shape | hot_start | average_plaquette | polyakov_loops | heatbath_sweep | heatbath_checkerboard_sweep |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(4, 4, 4, 6)` | 0.091682 s | 0.001160 s | 0.002074 s | 0.185099 s | 0.184081 s |
| `(6, 6, 6, 6)` | 0.276220 s | 0.003710 s | 0.006475 s | 0.627163 s | 0.629341 s |

Default heatbath speedup versus the baseline timings:

| Shape | Baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| `(4, 4, 4, 6)` | 0.317607 s | 0.185099 s | 1.72x |
| `(6, 6, 6, 6)` | 1.069367 s | 0.627163 s | 1.70x |

Post-change profile for one `heatbath_sweep` on `(4, 4, 4, 6)`:

```text
197684 function calls in 0.246 seconds

ncalls  cumtime  function
     1    0.246  heatbath_sweep
  1536    0.246  heatbath_update_link
  4608    0.127  _sample_su2_heatbath_from_coefficients
  1536    0.049  staple
  4608    0.033  su2_matrix_from_coefficients
  4608    0.031  _active_block_effective_staple_coefficients
  4608    0.030  _left_multiply_active_rows
  4608    0.019  _su2_effective_staple_coefficients_from_entries
  4608    0.014  np.linalg.norm
```

The previous `np.ix_`, `embed_su2`, and `np.linalg.det` hotspots are no longer
top cumulative-time entries.

## Optional JIT Acceleration Pass

Numba was added as an optional acceleration dependency. This required changing
the project NumPy requirement from `numpy>=2.5.1` to `numpy>=2.4,<2.5`, because
current `numba` releases require `numpy<2.5`.

Environment used for the benchmark:

```text
numpy 2.4.6
numba 0.66.0
```

The JIT implementation lives in `lattice_su3.accelerated` and does not affect the
default NumPy heatbath path. It uses numba's random stream with an integer seed,
so it is statistically equivalent rather than fixed-seed identical to the NumPy
`Generator` implementation.

Small-lattice benchmark:

| Shape | NumPy heatbath | Checkerboard heatbath | JIT cached run | JIT speedup |
| --- | ---: | ---: | ---: | ---: |
| `(4, 4, 4, 6)` | 0.187284 s | 0.187255 s | 0.001720 s | 108.89x |
| `(6, 6, 6, 6)` | 0.626394 s | 0.632639 s | 0.005797 s | 108.05x |

Target-size benchmark:

| Shape | NumPy heatbath | JIT cached run | JIT speedup |
| --- | ---: | ---: | ---: |
| `(16, 16, 16, 6)` | 11.918980 s | 0.116877 s | 101.98x |

GPU status:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

No CUDA-capable GPU runtime was visible in this environment, so GPU benchmarking
was not possible here. The checkerboard heatbath sweep remains the natural
starting point for a future GPU kernel because each `(direction, parity)` batch
can be parallelized without direct link-update conflicts.
