"""
Compare autocorrelation under different thinning intervals for a given run.

This script reads results/runs/<run_name>/observables.csv, sub-samples the
plaquette time series at multiple sweep intervals, and plots Gamma(t) for each
to visually assess whether thinning reduces autocorrelation.

Usage:
    Edit RUN_NAME, CHAIN, and THERMALIZATION_SWEEPS below, then run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analysis/thinning_autocorrelation.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

RUN_NAME = "check_thermal"
CHAIN = 0
THERMALIZATION_SWEEPS = 100
MAX_LAG = 100
THIN_INTERVALS = (1, 5, 10, 20)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def input_observables_path() -> Path:
    """Build the input path for run observables.

    Inputs:
        None.
    Outputs:
        CSV path produced by scripts/legacy/run_chain.py.
    """
    return ROOT / "results" / "runs" / RUN_NAME / "observables.csv"


def load_plaquette_series(
    path: Path,
    chain: int,
    thermalization_sweeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one post-thermalization plaquette series.

    Inputs:
        path: Observable CSV path produced by scripts/legacy/run_chain.py.
        chain: Chain index to analyze.
        thermalization_sweeps: Initial sweeps to exclude.
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


def validate_parameters() -> None:
    """Validate script-level thinning parameters.

    Inputs:
        None.
    Outputs:
        None.
    """
    if CHAIN < 0:
        raise ValueError("CHAIN must be non-negative")
    if THERMALIZATION_SWEEPS < 0:
        raise ValueError("THERMALIZATION_SWEEPS must be non-negative")
    if MAX_LAG <= 0:
        raise ValueError("MAX_LAG must be positive")
    if not THIN_INTERVALS:
        raise ValueError("THIN_INTERVALS must contain at least one interval")
    if any(interval <= 0 for interval in THIN_INTERVALS):
        raise ValueError("THIN_INTERVALS entries must be positive")


def thin_series(series: np.ndarray, interval: int) -> np.ndarray:
    """Sub-sample a series at a fixed sweep interval.

    Inputs:
        series: Full time series.
        interval: Take every interval-th element.
    Outputs:
        Thinned series.
    """
    return series[::interval]


def sweep_spacing(sweeps: np.ndarray) -> int:
    """Infer the sweep spacing between adjacent measurements.

    Inputs:
        sweeps: Sweep numbers for the analyzed measurements.
    Outputs:
        Positive sweep spacing.
    """
    if len(sweeps) <= 1:
        raise ValueError("at least two sweep values are required")

    spacing = np.diff(sweeps)
    if np.any(spacing <= 0):
        raise ValueError("sweep numbers must be strictly increasing")
    if not np.all(spacing == spacing[0]):
        raise ValueError("observable history must use a fixed measurement spacing")
    return int(spacing[0])


def output_plot_path() -> Path:
    """Build the output path for the thinning comparison plot.

    Inputs:
        None.
    Outputs:
        PNG path.
    """
    return (
        ROOT
        / "results"
        / "runs"
        / RUN_NAME
        / f"thinning_comparison_chain{CHAIN:02d}_after{THERMALIZATION_SWEEPS}sweeps.png"
    )


def main() -> None:
    """Compute autocorrelation for each thinning interval and plot.

    Inputs:
        None.
    Outputs:
        None.
    """
    validate_parameters()

    from lattice_su3 import (
        autocovariance,
        choose_window,
        integrated_autocorrelation,
        normalized_autocorrelation,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observables_path = input_observables_path()
    full_sweeps, full_series = load_plaquette_series(
        observables_path,
        CHAIN,
        THERMALIZATION_SWEEPS,
    )
    if len(full_series) <= 1:
        raise ValueError(
            "observable history must contain at least two measurements after "
            "thermalization cutoff"
        )
    base_sweep_spacing = sweep_spacing(full_sweeps)

    print(f"Observables: {observables_path}")
    print(f"Chain: {CHAIN}, discarding sweeps <= {THERMALIZATION_SWEEPS}")
    print(f"Post-thermalization samples: {len(full_series)}")
    print(f"Measurement spacing: {base_sweep_spacing} sweeps")
    print(f"Thinning intervals: {THIN_INTERVALS}")
    print()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {1: "C0", 2: "C1", 5: "C2", 10: "C3", 20: "C4"}
    markers = {1: "+", 2: "x", 5: "s", 10: "d", 20: "^"}

    for interval in THIN_INTERVALS:
        series = thin_series(full_series, interval)
        sweep_interval = interval * base_sweep_spacing
        max_lag = min(MAX_LAG, len(series) - 1)
        if max_lag < 1:
            print(f"  interval={interval}: too few samples ({len(series)}), skipping")
            continue

        covariance = autocovariance(series, max_lag)
        gamma = normalized_autocorrelation(covariance)
        tau = integrated_autocorrelation(gamma)
        window = choose_window(gamma)
        selected_tau = float(tau[window])
        tau_sweeps = selected_tau * sweep_interval

        lag = np.arange(len(gamma)) * sweep_interval
        effective_samples = len(series) / (2.0 * selected_tau)

        color = colors.get(interval, "gray")
        marker = markers.get(interval, ".")
        ax.plot(lag, gamma, "-", color=color, linewidth=0.8, alpha=0.4)
        ax.scatter(
            lag, gamma, marker=marker, color=color, s=40, zorder=3,
            label=(
                rf"$\Delta t={sweep_interval}$ sweeps "
                rf"($\tau_{{\mathrm{{int}}}}={tau_sweeps:.1f}$ sweeps, "
                rf"$N_{{\mathrm{{eff}}}}={effective_samples:.0f}$)"
            ),
        )

        print(
            f"  interval={interval} measurements ({sweep_interval} sweeps): "
            f"{len(series)} samples, "
            f"tau_int={selected_tau:.2f} thinned steps "
            f"({tau_sweeps:.2f} sweeps), "
            f"W={window}, "
            f"neff={effective_samples:.0f}"
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Sweep lag $t$")
    ax.set_ylabel(r"$\Gamma(t)$")
    ax.set_title(f"Autocorrelation vs thinning interval — {RUN_NAME} (chain {CHAIN})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = output_plot_path()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"\nPlot saved: {png_path}")


if __name__ == "__main__":
    main()
