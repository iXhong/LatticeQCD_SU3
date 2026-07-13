"""
Compute integrated autocorrelation time from a run observable history.

This script reads results/runs/<run_name>/observables.csv produced by
scripts/run_chain.py, selects one chain, discards the configured thermalization
cutoff, and writes an autocorrelation CSV in the same run directory.

Usage:
    Edit RUN_NAME, CHAIN, THERMALIZATION_SWEEPS, and MAX_LAG below, then run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/auto_correlation.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

SHAPE = (16, 16, 16, 6)
BETA = 5.7
ALGORITHM = "heatbath"
BACKEND = "jit"
STARTS = ("hot",)
SWEEPS = 300
SEED = 12345
RUN_NAME = ""
CHAIN = 0
THERMALIZATION_SWEEPS = 0
MAX_LAG = 250

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def shape_label(shape: tuple[int, ...]) -> str:
    """Format a lattice shape for filenames.

    Inputs:
        shape: Lattice size in each direction.
    Outputs:
        Shape label with dimensions joined by x.
    """
    return "x".join(str(length) for length in shape)


def run_label() -> str:
    """Build the default run directory label.

    Inputs:
        None.
    Outputs:
        Stable label matching scripts/run_chain.py defaults.
    """
    if RUN_NAME:
        return RUN_NAME

    starts_label = "-".join(STARTS)
    return (
        f"{ALGORITHM}_{BACKEND}_{starts_label}_{shape_label(SHAPE)}_"
        f"beta{BETA}_{SWEEPS}sweeps_seed{SEED}"
    )


def input_observables_path() -> Path:
    """Build the input path for run observables.

    Inputs:
        None.
    Outputs:
        CSV path produced by scripts/run_chain.py.
    """
    return ROOT / "results" / "runs" / run_label() / "observables.csv"


def output_autocorrelation_path() -> Path:
    """Build the output path for autocorrelation data.

    Inputs:
        None.
    Outputs:
        CSV path for autocorrelation measurements.
    """
    filename = f"autocorrelation_chain{CHAIN:02d}_after{THERMALIZATION_SWEEPS}sweeps.csv"
    return input_observables_path().with_name(filename)


def load_observable_history(
    path: Path,
    chain: int,
    thermalization_sweeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one post-thermalization plaquette history.

    Inputs:
        path: Observable CSV path produced by scripts/run_chain.py.
        chain: Chain index to analyze.
        thermalization_sweeps: Initial sweeps to exclude from analysis.
    Outputs:
        Sweep numbers and average plaquette values.
    """
    if thermalization_sweeps < 0:
        raise ValueError("THERMALIZATION_SWEEPS must be non-negative")
    if not path.exists():
        raise FileNotFoundError(f"observable history not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.asarray([data])

    required_fields = {"chain", "sweep", "average_plaquette"}
    missing_fields = required_fields.difference(data.dtype.names)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"observable history is missing required fields: {missing}")

    chains = np.asarray(data["chain"], dtype=np.int64)
    sweeps = np.asarray(data["sweep"], dtype=np.int64)
    plaquettes = np.asarray(data["average_plaquette"], dtype=np.float64)
    mask = (chains == chain) & (sweeps > thermalization_sweeps)
    return sweeps[mask], plaquettes[mask]


def write_autocorrelation_csv(
    path: Path,
    covariance: np.ndarray,
    gamma: np.ndarray,
    tau_int: np.ndarray,
) -> None:
    """Write autocorrelation data to CSV.

    Inputs:
        path: Output CSV path.
        covariance: Autocovariance values.
        gamma: Normalized autocorrelation values.
        tau_int: Running integrated autocorrelation time.
    Outputs:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lag",
                "autocovariance",
                "gamma",
                "tau_int_running",
            ],
        )
        writer.writeheader()
        for lag in range(len(gamma)):
            writer.writerow(
                {
                    "lag": lag,
                    "autocovariance": covariance[lag],
                    "gamma": gamma[lag],
                    "tau_int_running": tau_int[lag],
                }
            )


def main() -> None:
    """Run autocorrelation analysis for one run chain.

    Inputs:
        None.
    Outputs:
        None.
    """
    from lattice_su3 import (
        autocovariance,
        choose_window,
        integrated_autocorrelation,
        normalized_autocorrelation,
        suggested_interval,
    )

    observables_path = input_observables_path()
    sweeps, series = load_observable_history(
        observables_path,
        CHAIN,
        THERMALIZATION_SWEEPS,
    )
    if len(series) <= 1:
        raise ValueError(
            "observable history must contain at least two measurements after "
            "thermalization cutoff"
        )

    max_lag = min(MAX_LAG, len(series) - 1)
    covariance = autocovariance(series, max_lag)
    gamma = normalized_autocorrelation(covariance)
    tau_int = integrated_autocorrelation(gamma)
    window = choose_window(gamma)
    selected_tau = float(tau_int[window])
    effective_samples = len(series) / (2.0 * selected_tau)
    interval = suggested_interval(selected_tau)

    autocorr_path = output_autocorrelation_path()
    write_autocorrelation_csv(autocorr_path, covariance, gamma, tau_int)

    print(f"Loaded observables: {observables_path}")
    print(f"chain: {CHAIN}")
    print(
        f"thermalization cutoff: discard sweeps <= {THERMALIZATION_SWEEPS}"
    )
    print(f"sweeps analyzed: {sweeps[0]}..{sweeps[-1]} ({len(series)} values)")
    print(f"mean plaquette: {np.mean(series):.8f}")
    print(f"std plaquette: {np.std(series, ddof=1):.8f}")
    print(f"window lag: {window}")
    print(f"tau_int: {selected_tau:.4f}")
    print(f"effective samples: {effective_samples:.1f} / {len(series)}")
    print(f"suggested sampling interval: {interval} sweeps")
    print(f"Autocorrelation saved to {autocorr_path}")


if __name__ == "__main__":
    main()
