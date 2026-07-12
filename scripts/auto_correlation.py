"""
README: this is the script for computing integrated autocorrelation time.
- for the definition of the integrated autocorrelation time, check reference/gattringer_sec4.5_analyzing_data.md
- 之前已经实现了组态的更新，我们需要通过计算积分自关联时间来确定一个合理的采样的间隔
- 在这个脚本中，请你参考相关定义，实现相关的计算功能
- 实现 Gamma(t) = C(t) / C(0)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SHAPE = (4, 4, 4, 4)
BETA = 5.7
ALGORITHM = "heatbath"
START = "hot"
MEASUREMENT_SWEEPS = 250
MAX_LAG = 250

ROOT = Path(__file__).resolve().parents[1]


def input_series_path() -> Path:
    """Build the input path for the measured plaquette series.

    Inputs:
        None.
    Outputs:
        CSV path produced by scripts/average_plaquette_gen.py.
    """
    shape_label = "x".join(str(length) for length in SHAPE)
    return (
        ROOT
        / "results"
        / "autocorrelation"
        / f"{ALGORITHM}_{START}_{shape_label}_beta{BETA}_n{MEASUREMENT_SWEEPS}_series.csv"
    )


def output_autocorrelation_path() -> Path:
    """Build the output path for autocorrelation data.

    Inputs:
        None.
    Outputs:
        CSV path for autocorrelation measurements.
    """
    series_path = input_series_path()
    return series_path.with_name(series_path.stem.replace("_series", "_autocorrelation") + ".csv")


def load_series_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load measured plaquette time series from CSV.

    Inputs:
        path: CSV path produced by scripts/average_plaquette_gen.py.
    Outputs:
        Measurement indices and average plaquette values.
    """
    if not path.exists():
        raise FileNotFoundError(f"plaquette series file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.asarray([data])
    return (
        np.asarray(data["measurement"], dtype=np.int64),
        np.asarray(data["average_plaquette"], dtype=np.float64),
    )


def autocovariance(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute the autocovariance function C(t).

    Inputs:
        series: One-dimensional observable time series.
        max_lag: Maximum lag to compute.
    Outputs:
        Autocovariance values from lag zero through max_lag.
    """
    mean = float(np.mean(series))
    centered = series - mean
    covariance = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        covariance[lag] = float(np.mean(centered[: len(series) - lag] * centered[lag:]))
    return covariance


def normalized_autocorrelation(covariance: np.ndarray) -> np.ndarray:
    """Compute Gamma(t) = C(t) / C(0).

    Inputs:
        covariance: Autocovariance values starting at lag zero.
    Outputs:
        Normalized autocorrelation values.
    """
    if covariance[0] <= 0.0:
        raise ValueError("C(0) must be positive to normalize autocorrelation")
    return covariance / covariance[0]


def integrated_autocorrelation(gamma: np.ndarray) -> np.ndarray:
    """Compute running integrated autocorrelation time.

    Inputs:
        gamma: Normalized autocorrelation values starting at lag zero.
    Outputs:
        Running tau_int values for each lag.
    """
    tau_int = np.empty_like(gamma)
    tau_int[0] = 0.5
    for lag in range(1, len(gamma)):
        tau_int[lag] = tau_int[lag - 1] + gamma[lag]
    return tau_int


def choose_window(gamma: np.ndarray) -> int:
    """Choose a conservative summation window.

    Inputs:
        gamma: Normalized autocorrelation values starting at lag zero.
    Outputs:
        Lag index where tau_int should be read.
    """
    for lag in range(1, len(gamma)):
        if gamma[lag] <= 0.0:
            return lag - 1
    return len(gamma) - 1


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
    """Run autocorrelation analysis for average plaquette measurements.

    Inputs:
        None.
    Outputs:
        None.
    """
    series_path = input_series_path()
    measurements, series = load_series_csv(series_path)
    if len(series) <= 1:
        raise ValueError("plaquette series must contain at least two measurements")

    max_lag = min(MAX_LAG, len(series) - 1)
    covariance = autocovariance(series, max_lag)
    gamma = normalized_autocorrelation(covariance)
    tau_int = integrated_autocorrelation(gamma)
    window = choose_window(gamma)
    selected_tau = tau_int[window]
    effective_samples = len(series) / (2.0 * selected_tau)
    suggested_interval = max(1, int(np.ceil(2.0 * selected_tau)))

    autocorr_path = output_autocorrelation_path()
    write_autocorrelation_csv(autocorr_path, covariance, gamma, tau_int)

    print(f"Loaded series: {series_path}")
    print(f"measurements: {measurements[0]}..{measurements[-1]} ({len(series)} values)")
    print(f"mean plaquette: {np.mean(series):.8f}")
    print(f"std plaquette: {np.std(series, ddof=1):.8f}")
    print(f"window lag: {window}")
    print(f"tau_int: {selected_tau:.4f}")
    print(f"effective samples: {effective_samples:.1f} / {len(series)}")
    print(f"suggested sampling interval: {suggested_interval} measurements")
    print(f"Autocorrelation saved to {autocorr_path}")


if __name__ == "__main__":
    main()
