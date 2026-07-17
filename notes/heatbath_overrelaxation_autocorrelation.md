# Heatbath + Overrelaxation Autocorrelation Notes

## Context

This note records the first small pilot tests comparing pure heatbath updates
against combined heatbath plus overrelaxation updates for SU(3) pure gauge
chains.

Implementation status:

- `src/lattice_su3/update.py` provides the NumPy reference overrelaxation path.
- `src/lattice_su3/accelerated.py` provides Numba-JIT overrelaxation sweeps.
- `scripts/run_chain.py` uses JIT overrelaxation automatically when
  `BACKEND = "jit"` or `BACKEND = "jit_checkerboard"`.
- No full gauge configurations were saved in these tests.

## Test Setup

Common parameters:

```python
BETA = 5.7
ALGORITHM = "heatbath"
BACKEND = "jit"
STARTS = ("hot",)
MEASURE_EVERY = 1
SAVE_CONFIG_EVERY = 0
SEED = 12345
```

Autocorrelation analysis used the average plaquette time series from
`scripts/run_chain.py` and `scripts/auto_correlation.py`.

## Results

### `4^4`, 1200 Sweeps

Analysis parameters:

```python
THERMALIZATION_SWEEPS = 200
MAX_LAG = 250
```

| Update | Runtime | Samples after cutoff | tau_int | Suggested interval | Effective samples |
|---|---:|---:|---:|---:|---:|
| HB | 2.2 s | 1000 | 2.7535 | 6 | 181.6 |
| HB + 2 OR | 3.4 s | 1000 | 2.4397 | 5 | 204.9 |

Observation: overrelaxation gave a small reduction in plaquette autocorrelation
on this very small lattice, but the improvement was modest.

### `6^4`, 800 Sweeps

Analysis parameters:

```python
THERMALIZATION_SWEEPS = 200
MAX_LAG = 200
```

| Update | Runtime | Samples after cutoff | tau_int | Suggested interval | Effective samples |
|---|---:|---:|---:|---:|---:|
| HB | 7.2 s | 600 | 13.7284 | 28 | 21.9 |
| HB + 2 OR | 11.0 s | 600 | 5.3332 | 11 | 56.3 |

Observation: on the larger pilot lattice, `HB + 2 OR` reduced the plaquette
integrated autocorrelation time by about 61 percent while increasing wall time
by about 53 percent. This is a useful early indication that overrelaxation is
more beneficial once pure heatbath autocorrelation becomes longer.

## Caveats

- These are single-seed pilot runs, not production statistics.
- The observable was only the average plaquette.
- The selected autocorrelation window is noisy for short chains.
- A recorded `HB + 2 OR` sweep contains three lattice passes, so comparisons
  should report both recorded-sweep tau and wall-clock cost.

## Next Tests

Suggested next steps:

1. Repeat `6^4` with several seeds.
2. Scan `OVERRELAXATION_SWEEPS = 1, 2, 3, 4`.
3. Run a short pilot at the target production geometry.
4. Compare autocorrelation for Polyakov loop observables, not only plaquette.
