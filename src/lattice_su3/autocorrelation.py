"""Autocorrelation analysis helpers for measured time series."""

import numpy as np


def autocovariance(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute the autocovariance function C(t).

    Inputs:
        series: One-dimensional observable time series.
        max_lag: Maximum lag to compute.
    Outputs:
        Autocovariance values from lag zero through max_lag.
    """
    if series.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if max_lag < 0 or max_lag >= len(series):
        raise ValueError("max_lag must be between zero and len(series) - 1")

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
    """Choose the first non-positive autocorrelation summation window.

    Inputs:
        gamma: Normalized autocorrelation values starting at lag zero.
    Outputs:
        Lag index where tau_int should be read.
    """
    for lag in range(1, len(gamma)):
        if gamma[lag] <= 0.0:
            return lag - 1
    return len(gamma) - 1


def suggested_interval(tau_int: float, factor: float = 2.0) -> int:
    """Compute a configuration spacing from integrated autocorrelation time.

    Inputs:
        tau_int: Integrated autocorrelation time in measured sweeps.
        factor: Multiplicative spacing factor.
    Outputs:
        Positive integer number of sweeps between saved configurations.
    """
    if tau_int <= 0.0:
        raise ValueError("tau_int must be positive")
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    return max(1, int(np.ceil(factor * tau_int)))
