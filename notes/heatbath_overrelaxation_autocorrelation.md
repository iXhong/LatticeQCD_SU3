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

## Polyakov Time-Series Pilot

The first Polyakov comparison used saved configurations from
`polyakov_correlator` and `polyakov_correlator_or2`, with one saved
configuration every 20 sweeps. Those saved samples were already too thinned to
resolve a useful autocorrelation difference in the Polyakov correlator:
representative `C(r)` series mostly selected window zero and
`tau_int = 0.5`.

To measure Polyakov autocorrelation directly, `scripts/run_chain.py` was
extended to write selected scalar Polyakov observables into `observables.csv`
without saving gauge configurations.

Common run parameters:

```python
SHAPE = (16, 16, 16, 6)
BETA = 5.7
STARTS = ("load",)
SOURCE_RUN_NAME = "check_thermal"
SOURCE_CHAIN = 0
SWEEPS = 1000
MEASURE_EVERY = 1
MEASURE_PLAQUETTE = False
MEASURE_POLYAKOV = True
POLYAKOV_TIME_DIRECTION = -1
POLYAKOV_OFFSETS = ((1, 0, 0), (2, 0, 0))
SAVE_CONFIG_EVERY = 0
BACKEND = "jit_checkerboard"
SEED = 12345
```

Compared runs:

| Run | `OVERRELAXATION_SWEEPS` | Wall time |
|---|---:|---:|
| `polyakov_timeseries_hb` | 0 | 174.4 s |
| `polyakov_timeseries_hb_or2` | 2 | 273.1 s |

Analysis parameters:

```python
THERMALIZATION_SWEEPS = 1100
MAX_LAG = 250
```

This leaves 900 high-frequency measurements from sweeps `1101..2000`.

| Observable | HB tau_int | HB + 2 OR tau_int | Comment |
|---|---:|---:|---|
| `abs(Pbar)` | 3.7741 | 1.5422 | clear improvement |
| `abs(Pbar)^2` | 3.9329 | 1.8055 | clear improvement |
| `C(1,0,0).real` | 0.7844 | 0.8239 | essentially unchanged |
| `C(2,0,0).real` | 0.6227 | 0.5991 | essentially unchanged |

Observation: `HB + 2 OR` reduced the integrated autocorrelation time for the
global Polyakov loop magnitude by roughly a factor of 2.1 to 2.4 while
increasing wall time by about a factor of 1.57. For the short-distance
Polyakov correlators tested here, the pure heatbath time series was already
close to the minimum measurable autocorrelation, so overrelaxation did not show
a meaningful improvement.

The summary table was written to:

```text
results/runs/polyakov_autocorrelation_summary.csv
```

No full gauge configurations were saved in the high-frequency time-series
runs.
