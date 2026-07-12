import numpy as np
import pytest

from lattice_su3 import (
    autocovariance,
    choose_window,
    integrated_autocorrelation,
    normalized_autocorrelation,
    suggested_interval,
)


def test_autocorrelation_helpers_compute_expected_running_tau():
    gamma = np.asarray([1.0, 0.4, 0.1, -0.05], dtype=np.float64)

    tau_int = integrated_autocorrelation(gamma)

    assert np.allclose(tau_int, [0.5, 0.9, 1.0, 0.95], atol=1e-12)
    assert choose_window(gamma) == 2
    assert suggested_interval(tau_int[2], factor=2.0) == 2


def test_autocovariance_and_normalized_autocorrelation_shapes():
    series = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)

    covariance = autocovariance(series, max_lag=3)
    gamma = normalized_autocorrelation(covariance)

    assert covariance.shape == (4,)
    assert gamma.shape == (4,)
    assert np.isclose(gamma[0], 1.0)


def test_suggested_interval_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        suggested_interval(0.0)
    with pytest.raises(ValueError):
        suggested_interval(1.0, factor=0.0)
