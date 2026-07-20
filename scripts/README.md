# Script Layout

Use scripts from the repository root with `UV_CACHE_DIR=/tmp/uv-cache uv run python`.

## workflows/

Current production entrypoints:

- `thermalize.py`: thermalize one chain from a TOML config.
- `generate_ensemble.py`: generate a multi-chain ensemble from a thermalized source config.
- `run_server_chain.py`: run one resumable chain for scheduler/job-array use.

## analysis/

Postprocessing scripts that read standard run outputs and write analysis data:

- `auto_correlation.py`: plaquette autocorrelation from `observables.csv`.
- `analyze_thermalization.py`: cold/hot plaquette history plots.
- `measure_polyakov_correlators.py`: vector Polyakov correlators from saved configs.
- `bin_polyakov_correlators.py`: radial and axis binning for vector correlators.
- `resample_polyakov_correlators.py`: chain-aware blocking, jackknife, and bootstrap.
- `analyze_static_potential.py`: static-potential extraction and Cornell fits.
- `polyakov_autocorrelation.py`: Polyakov scalar autocorrelation comparison.
- `thinning_autocorrelation.py`: thinning diagnostics for plaquette histories.

## plotting/

Plot-focused helpers for existing analysis outputs:

- `plot_polyakov_correlator_log.py`
- `plot_polyakov_loop_plane.py`
- `plot_static_potential.py`

## benchmarks/

Performance diagnostics:

- `benchmark_average_plaquette.py`
- `benchmark_heatbath_acceleration.py`

## legacy/

Compatibility scripts kept outside the current TOML workflow:

- `run_chain.py`
