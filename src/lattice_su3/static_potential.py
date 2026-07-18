"""Static-potential extraction, correlated fitting, and scale helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CornellFit:
    """Store one correlated Cornell-potential fit.

    Inputs:
        parameters: Parameters ``(A, B, sigma_a2)``.
        covariance: Parameter covariance matrix.
        chi_squared: Correlated chi-squared at the optimum.
        degrees_of_freedom: Number of fitted points minus three.
        mask: Boolean mask selecting fitted distances.
    Outputs:
        Immutable correlated-fit result.
    """

    parameters: np.ndarray
    covariance: np.ndarray
    chi_squared: float
    degrees_of_freedom: int
    mask: np.ndarray


def potential_from_correlators(correlators: np.ndarray, nt: int) -> np.ndarray:
    """Transform correlator samples to lattice static-potential samples.

    Inputs:
        correlators: Correlator samples with a leading sample dimension.
        nt: Positive temporal lattice extent.
    Outputs:
        Real ``-log(Re C) / Nt`` values, with NaN for non-positive correlators.
    """
    correlators = np.asarray(correlators)
    if correlators.ndim < 1 or correlators.shape[0] == 0:
        raise ValueError("correlators must contain at least one sample")
    if nt <= 0:
        raise ValueError("nt must be positive")
    real = np.real(correlators)
    potential = np.full(real.shape, np.nan, dtype=np.float64)
    positive = real > 0.0
    potential[positive] = -np.log(real[positive]) / nt
    return potential


def jackknife_mean_covariance(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean and covariance from delete-one-block samples.

    Inputs:
        samples: Finite jackknife estimates with shape ``(n_blocks, n_values)``.
    Outputs:
        Jackknife mean and covariance matrix.
    """
    samples = _finite_sample_matrix(samples, "jackknife")
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = (len(samples) - 1) / len(samples) * (centered.T @ centered)
    return mean, covariance


def bootstrap_mean_covariance(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean and sample covariance from bootstrap estimates.

    Inputs:
        samples: Finite bootstrap estimates with shape ``(n_samples, n_values)``.
    Outputs:
        Bootstrap mean and covariance matrix.
    """
    samples = _finite_sample_matrix(samples, "bootstrap")
    return samples.mean(axis=0), np.cov(samples, rowvar=False, ddof=1)


def _finite_sample_matrix(samples: np.ndarray, label: str) -> np.ndarray:
    """Validate a finite two-dimensional resampling matrix.

    Inputs:
        samples: Candidate sample matrix.
        label: Resampling method name for errors.
    Outputs:
        Float sample matrix.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError(f"{label} samples must have shape (n_samples >= 2, n_values)")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{label} samples must be finite")
    return samples


def fit_cornell_correlated(
    radii: np.ndarray,
    potential: np.ndarray,
    covariance: np.ndarray,
    r_min: float,
    r_max: float,
) -> CornellFit:
    """Fit ``A + B/r + sigma_a2*r`` using a correlated linear fit.

    Inputs:
        radii: Positive distances in lattice units.
        potential: Static potential values in lattice units.
        covariance: Covariance matrix for all supplied distances.
        r_min: Inclusive lower fit distance.
        r_max: Inclusive upper fit distance.
    Outputs:
        Correlated Cornell fit result.
    """
    radii = np.asarray(radii, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if radii.ndim != 1 or potential.shape != radii.shape:
        raise ValueError("radii and potential must be matching one-dimensional arrays")
    if covariance.shape != (len(radii), len(radii)):
        raise ValueError("covariance shape must match the distance count")
    if r_min > r_max:
        raise ValueError("r_min cannot exceed r_max")
    mask = (radii >= r_min) & (radii <= r_max) & (radii > 0.0)
    mask &= np.isfinite(potential)
    if np.count_nonzero(mask) < 4:
        raise ValueError("Cornell fit requires at least four selected distances")

    selected_r = radii[mask]
    selected_v = potential[mask]
    selected_covariance = covariance[np.ix_(mask, mask)]
    inverse_covariance = np.linalg.pinv(selected_covariance, hermitian=True)
    design = np.column_stack(
        [np.ones_like(selected_r), 1.0 / selected_r, selected_r]
    )
    normal = design.T @ inverse_covariance @ design
    parameter_covariance = np.linalg.pinv(normal, hermitian=True)
    parameters = parameter_covariance @ design.T @ inverse_covariance @ selected_v
    residual = selected_v - design @ parameters
    chi_squared = float(residual @ inverse_covariance @ residual)
    return CornellFit(
        parameters=parameters,
        covariance=parameter_covariance,
        chi_squared=chi_squared,
        degrees_of_freedom=len(selected_r) - 3,
        mask=mask,
    )


def fit_cornell_samples(
    radii: np.ndarray,
    potential_samples: np.ndarray,
    covariance: np.ndarray,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """Fit every finite resampled potential using one covariance matrix.

    Inputs:
        radii: Distances in lattice units.
        potential_samples: Potential estimates with shape ``(n_samples, n_r)``.
        covariance: Covariance matrix used for every correlated fit.
        r_min: Inclusive lower fit distance.
        r_max: Inclusive upper fit distance.
    Outputs:
        Fitted ``(A, B, sigma_a2)`` parameters for every sample.
    """
    potential_samples = np.asarray(potential_samples, dtype=np.float64)
    if potential_samples.ndim != 2 or potential_samples.shape[1] != len(radii):
        raise ValueError("potential_samples must have shape (n_samples, n_radii)")
    parameters = []
    for sample in potential_samples:
        parameters.append(
            fit_cornell_correlated(radii, sample, covariance, r_min, r_max).parameters
        )
    return np.asarray(parameters)


def sommer_scale_r0_over_a(parameters: np.ndarray, force_constant: float = 1.65) -> np.ndarray:
    """Compute ``r0/a`` from Cornell parameters and the Sommer condition.

    Inputs:
        parameters: Cornell parameters with final dimension ``(A, B, sigma_a2)``.
        force_constant: Dimensionless Sommer force condition, conventionally 1.65.
    Outputs:
        ``r0/a = sqrt((force_constant + B) / sigma_a2)`` or NaN if invalid.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape[-1] != 3:
        raise ValueError("parameters must have final dimension three")
    numerator = force_constant + parameters[..., 1]
    denominator = parameters[..., 2]
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    valid = (numerator > 0.0) & (denominator > 0.0)
    result[valid] = np.sqrt(numerator[valid] / denominator[valid])
    return result
