import numpy as np
import pytest

from lattice_su3 import (
    bootstrap_mean_covariance,
    fit_cornell_correlated,
    fit_cornell_samples,
    jackknife_mean_covariance,
    potential_from_correlators,
    sommer_scale_r0_over_a,
)


def test_potential_from_correlators_transforms_positive_real_part():
    correlators = np.asarray([[1.0, np.exp(-4.0), -0.1]], dtype=np.complex128)

    potential = potential_from_correlators(correlators, nt=2)

    assert np.allclose(potential[0, :2], [0.0, 2.0], atol=1e-12)
    assert np.isnan(potential[0, 2])


def test_jackknife_mean_covariance_uses_delete_one_normalization():
    samples = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    mean, covariance = jackknife_mean_covariance(samples)

    centered = samples - samples.mean(axis=0)
    expected = 2.0 / 3.0 * centered.T @ centered
    assert np.allclose(mean, [2.0, 4.0], atol=1e-12)
    assert np.allclose(covariance, expected, atol=1e-12)


def test_bootstrap_mean_covariance_matches_numpy_sample_covariance():
    samples = np.asarray([[1.0, 2.0], [2.0, 5.0], [4.0, 8.0]])

    mean, covariance = bootstrap_mean_covariance(samples)

    assert np.allclose(mean, samples.mean(axis=0), atol=1e-12)
    assert np.allclose(covariance, np.cov(samples, rowvar=False, ddof=1), atol=1e-12)


def test_correlated_cornell_fit_recovers_synthetic_parameters():
    radii = np.arange(1.0, 7.0)
    expected = np.asarray([0.4, -0.3, 0.12])
    potential = expected[0] + expected[1] / radii + expected[2] * radii
    covariance = np.diag(np.linspace(0.01, 0.02, len(radii)) ** 2)

    fit = fit_cornell_correlated(radii, potential, covariance, 1.0, 6.0)

    assert np.allclose(fit.parameters, expected, atol=1e-10)
    assert np.isclose(fit.chi_squared, 0.0, atol=1e-20)
    assert fit.degrees_of_freedom == 3


def test_fit_cornell_samples_and_sommer_scale():
    radii = np.arange(1.0, 7.0)
    parameters = np.asarray([[0.4, -0.3, 0.12], [0.5, -0.2, 0.10]])
    samples = np.asarray(
        [a + b / radii + sigma * radii for a, b, sigma in parameters]
    )
    covariance = np.eye(len(radii)) * 0.01

    fitted = fit_cornell_samples(radii, samples, covariance, 1.0, 6.0)
    scales = sommer_scale_r0_over_a(fitted)

    assert np.allclose(fitted, parameters, atol=1e-10)
    assert np.allclose(
        scales,
        np.sqrt((1.65 + parameters[:, 1]) / parameters[:, 2]),
        atol=1e-10,
    )


def test_correlated_cornell_fit_requires_four_distances():
    with pytest.raises(ValueError, match="at least four"):
        fit_cornell_correlated(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 2.0, 3.0]),
            np.eye(3),
            1.0,
            3.0,
        )
