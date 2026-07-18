"""Spatial binning helpers for Polyakov loop correlators."""

import numpy as np


def periodic_displacement_squared(spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Compute minimum-image squared distances for all periodic displacements.

    Inputs:
        spatial_shape: Positive spatial lattice lengths.
    Outputs:
        Integer array of squared distances with shape ``spatial_shape``.
    """
    if not spatial_shape or any(length <= 0 for length in spatial_shape):
        raise ValueError("spatial_shape must contain positive lattice lengths")

    squared = np.zeros(spatial_shape, dtype=np.int64)
    for axis, length in enumerate(spatial_shape):
        coordinates = np.arange(length, dtype=np.int64)
        minimum_image = np.minimum(coordinates, length - coordinates)
        reshape = [1] * len(spatial_shape)
        reshape[axis] = length
        squared += minimum_image.reshape(reshape) ** 2
    return squared


def radial_average_correlators(
    correlators: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average vector correlators over equal minimum-image squared distances.

    Inputs:
        correlators: Complex array with shape ``(n_cfg, *spatial_shape)``.
    Outputs:
        Squared radii, displacement degeneracies, and per-configuration radial
        correlators with shape ``(n_cfg, n_radii)``.
    """
    correlators = np.asarray(correlators)
    if correlators.ndim < 2 or correlators.shape[0] == 0:
        raise ValueError("correlators must have shape (n_cfg, *spatial_shape)")
    if any(length <= 0 for length in correlators.shape[1:]):
        raise ValueError("correlator spatial lengths must be positive")

    distance_grid = periodic_displacement_squared(correlators.shape[1:])
    r_squared, inverse, degeneracies = np.unique(
        distance_grid.reshape(-1), return_inverse=True, return_counts=True
    )
    flattened = correlators.reshape(correlators.shape[0], -1)
    radial = np.empty((correlators.shape[0], len(r_squared)), dtype=np.complex64)
    for bin_index in range(len(r_squared)):
        radial[:, bin_index] = flattened[:, inverse == bin_index].mean(axis=1)

    return r_squared, degeneracies.astype(np.int64), radial


def axis_average_correlators(
    correlators: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average vector correlators over positive and negative axis displacements.

    Inputs:
        correlators: Complex array with shape ``(n_cfg, *spatial_shape)``.
    Outputs:
        Integer axis distances, displacement degeneracies, and per-configuration
        axis correlators with shape ``(n_cfg, n_distances)``.
    """
    correlators = np.asarray(correlators)
    if correlators.ndim < 2 or correlators.shape[0] == 0:
        raise ValueError("correlators must have shape (n_cfg, *spatial_shape)")

    spatial_shape = correlators.shape[1:]
    max_distance = max(length // 2 for length in spatial_shape)
    distances: list[int] = []
    degeneracies: list[int] = []
    averages: list[np.ndarray] = []

    for distance in range(max_distance + 1):
        points: list[tuple[int, ...]] = []
        if distance == 0:
            points.append((0,) * len(spatial_shape))
        else:
            for axis, length in enumerate(spatial_shape):
                if distance > length // 2:
                    continue
                forward = [0] * len(spatial_shape)
                forward[axis] = distance
                points.append(tuple(forward))
                backward_coordinate = (-distance) % length
                if backward_coordinate != distance:
                    backward = [0] * len(spatial_shape)
                    backward[axis] = backward_coordinate
                    points.append(tuple(backward))
        if points:
            values = np.stack(
                [correlators[(slice(None), *point)] for point in points], axis=1
            )
            distances.append(distance)
            degeneracies.append(len(points))
            averages.append(values.mean(axis=1))

    return (
        np.asarray(distances, dtype=np.int64),
        np.asarray(degeneracies, dtype=np.int64),
        np.stack(averages, axis=1).astype(np.complex64, copy=False),
    )
