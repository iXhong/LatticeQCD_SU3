"""
Compare Polyakov-loop autocorrelation from run observable histories.

This script reads high-frequency Polyakov scalar measurements written by
scripts/legacy/run_chain.py. It expects runs with MEASURE_POLYAKOV = True and compares
the configured observable columns after a thermalization cutoff.

Usage:
    Edit RUN_NAMES, THERMALIZATION_SWEEPS, and MAX_LAG below, then run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analysis/polyakov_autocorrelation.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

RUN_NAMES = {
    "HB": "polyakov_timeseries_hb",
    "HB+OR2": "polyakov_timeseries_hb_or2",
}
THERMALIZATION_SWEEPS = 1100
MAX_LAG = 250
OBSERVABLE_COLUMNS = (
    "polyakov_abs",
    "polyakov_abs2",
    "polyakov_c_1_0_0_re",
    "polyakov_c_2_0_0_re",
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import (  # noqa: E402
    autocovariance,
    choose_window,
    integrated_autocorrelation,
    normalized_autocorrelation,
    suggested_interval,
)


def run_directory(run_name: str) -> Path:
    """Build a run directory path.

    Inputs:
        run_name: Run directory name under results/runs.
    Outputs:
        Path to the selected run directory.
    """
    if not run_name:
        raise ValueError("run name must be non-empty")
    return ROOT / "results" / "runs" / run_name


def load_manifest(run_dir: Path) -> dict[str, object]:
    """Load one run manifest.

    Inputs:
        run_dir: Run directory path.
    Outputs:
        Parsed manifest dictionary.
    """
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_observable_columns(
    path: Path,
    columns: tuple[str, ...],
    thermalization_sweeps: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load selected observable columns after a sweep cutoff.

    Inputs:
        path: Observable CSV path.
        columns: Observable columns to load.
        thermalization_sweeps: Discard rows with sweep <= this value.
    Outputs:
        Sweep numbers and selected column arrays.
    """
    if thermalization_sweeps < 0:
        raise ValueError("THERMALIZATION_SWEEPS must be non-negative")
    if not path.exists():
        raise FileNotFoundError(f"observable history not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.asarray([data])
    if data.dtype.names is None:
        raise ValueError(f"{path} has no CSV header")

    required_fields = {"sweep", *columns}
    missing_fields = required_fields.difference(data.dtype.names)
    if missing_fields:
        raise ValueError(f"{path} is missing required fields: {sorted(missing_fields)}")

    sweeps = np.asarray(data["sweep"], dtype=np.int64)
    mask = sweeps > thermalization_sweeps
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            "at least two measurements are required after the thermalization cutoff"
        )

    series = {
        column: np.asarray(data[column], dtype=np.float64)[mask] for column in columns
    }
    return sweeps[mask], series


def autocorrelation_summary(
    values: np.ndarray,
    max_lag: int,
) -> dict[str, float | int]:
    """Summarize one autocorrelation time series.

    Inputs:
        values: Scalar observable history.
        max_lag: Maximum lag to include in autocovariance.
    Outputs:
        Summary fields for mean, standard deviation, tau, and interval.
    """
    if max_lag <= 0:
        raise ValueError("MAX_LAG must be positive")

    values = np.asarray(values, dtype=np.float64)
    selected_max_lag = min(max_lag, len(values) - 1)
    covariance = autocovariance(values, selected_max_lag)
    gamma = normalized_autocorrelation(covariance)
    tau_int = integrated_autocorrelation(gamma)
    window = choose_window(gamma)
    selected_tau = float(tau_int[window])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "window": int(window),
        "tau_int": selected_tau,
        "effective_samples": float(len(values) / (2.0 * selected_tau)),
        "suggested_interval": int(suggested_interval(selected_tau, 2.0)),
    }


def write_summary_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write autocorrelation comparison rows to CSV.

    Inputs:
        path: Output CSV path.
        rows: Summary row dictionaries.
    Outputs:
        None.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the configured Polyakov autocorrelation comparison.

    Inputs:
        None.
    Outputs:
        None.
    """
    rows = []
    for label, run_name in RUN_NAMES.items():
        run_dir = run_directory(run_name)
        manifest = load_manifest(run_dir)
        sweeps, series = load_observable_columns(
            run_dir / "observables.csv",
            OBSERVABLE_COLUMNS,
            THERMALIZATION_SWEEPS,
        )
        elapsed = manifest.get("elapsed_seconds", "")
        for observable, values in series.items():
            summary = autocorrelation_summary(values, MAX_LAG)
            rows.append(
                {
                    "run_label": label,
                    "run_name": run_name,
                    "observable": observable,
                    "samples": len(values),
                    "sweep_start": int(sweeps[0]),
                    "sweep_stop": int(sweeps[-1]),
                    "elapsed_seconds": elapsed,
                    **summary,
                }
            )

    output_path = ROOT / "results" / "runs" / "polyakov_autocorrelation_summary.csv"
    write_summary_csv(output_path, rows)

    print("Polyakov autocorrelation comparison")
    print(f"thermalization cutoff: discard sweeps <= {THERMALIZATION_SWEEPS}")
    print(f"max lag: {MAX_LAG}")
    print(
        "run, observable, samples, elapsed_s, mean, tau_int, "
        "effective_samples, suggested_interval"
    )
    for row in rows:
        print(
            f"{row['run_label']}, {row['observable']}, {row['samples']}, "
            f"{row['elapsed_seconds']}, {row['mean']:.8e}, "
            f"{row['tau_int']:.4f}, {row['effective_samples']:.1f}, "
            f"{row['suggested_interval']}"
        )
    print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    main()
