# Debian HB+OR2 Polyakov Production Run

This note records the planned standalone Debian production run for Polyakov
loop correlator measurements. It assumes the repository has been cloned on a
single Debian server without `uv` or a batch scheduler.

## Target Run

Configuration:

```text
configs/ensemble_polyakov_hb_or2_server.toml
```

Physics and workflow choices:

- Lattice: `16^3 x 6`
- Coupling: `beta = 5.7`
- Source: `results/runs/check_thermal/configurations/chain00_hot_sweep001000.npz`
- Update: heatbath plus two overrelaxation sweeps
- Production: 8 chains, 3200 sweeps per chain
- Initial per-chain adapter discard: 200 sweeps
- Save spacing: 20 sweeps
- Saved configurations: `8 * ((3200 - 200) / 20) = 1200`

The 200-sweep discard is included because the source configuration was produced
by the existing heatbath workflow and this run changes the Markov kernel to
HB+OR2.

## Debian Environment

Create a local virtual environment from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[acceleration]"
```

If Debian does not provide Python 3.13 directly, install it with the server's
preferred Python toolchain before creating the virtual environment.

## Production Command

For a 16-core server, use four concurrent chains with four Numba threads each:

```bash
source .venv/bin/activate
export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/generate_ensemble.py \
  configs/ensemble_polyakov_hb_or2_server.toml
```

The total CPU demand is approximately:

```text
parallel * NUMBA_NUM_THREADS
```

Adjust `parallel` in the TOML and `NUMBA_NUM_THREADS` together for the server.
For example, on 32 cores use `parallel = 4` and `NUMBA_NUM_THREADS=8`, or
`parallel = 8` and `NUMBA_NUM_THREADS=4`.

## Short Smoke Run

Before the full run, make a temporary copy of the TOML and reduce:

```toml
name = "smoke_polyakov_16x16x16x6_b57_hb_or2_server"
chains = 2
sweeps_per_chain = 600
discard_sweeps = 200
parallel = 2
```

Run the same command and check that configurations and `observables.csv` are
written under:

```text
results/runs/smoke_polyakov_16x16x16x6_b57_hb_or2_server/
```

## Follow-Up Analysis

After production finishes, measure and analyze Polyakov correlators with:

```bash
python scripts/measure_polyakov_correlators.py
python scripts/bin_polyakov_correlators.py \
  results/runs/prod_static_polyakov_16x16x16x6_b57_hb_or2_server/correlators/polyakov_vector_correlators.npz
python scripts/resample_polyakov_correlators.py \
  results/runs/prod_static_polyakov_16x16x16x6_b57_hb_or2_server/correlators/polyakov_binned_correlators.npz \
  --block-size 5 \
  --output results/runs/prod_static_polyakov_16x16x16x6_b57_hb_or2_server/correlators/polyakov_resampled_block05.npz
python scripts/analyze_static_potential.py \
  results/runs/prod_static_polyakov_16x16x16x6_b57_hb_or2_server/correlators/polyakov_resampled_block05.npz \
  --binning axis --method jackknife --r-min 1 --r-max 5 \
  --output results/runs/prod_static_polyakov_16x16x16x6_b57_hb_or2_server/correlators/static_b05_jk_1_5.npz
```

Before running `measure_polyakov_correlators.py`, set its `RUN_NAME` to:

```python
RUN_NAME = "prod_static_polyakov_16x16x16x6_b57_hb_or2_server"
THERMALIZATION_SWEEPS = 0
```
