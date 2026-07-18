import numpy as np
import pytest

from lattice_su3 import (
    axis_average_correlators,
    periodic_displacement_squared,
    radial_average_correlators,
)


def test_periodic_displacement_squared_uses_minimum_image_on_rectangular_lattice():
    squared = periodic_displacement_squared((4, 6))

    assert squared.shape == (4, 6)
    assert squared[0, 0] == 0
    assert squared[3, 5] == 2
    assert squared[2, 3] == 13
    assert np.array_equal(squared[:, 0], [0, 1, 4, 1])


def test_radial_average_groups_all_periodic_displacements_by_integer_r_squared():
    squared = periodic_displacement_squared((4, 4))
    correlators = np.stack(
        [squared.astype(np.complex128), 2.0 * squared.astype(np.complex128)]
    )

    r_squared, degeneracies, radial = radial_average_correlators(correlators)

    assert np.array_equal(r_squared, [0, 1, 2, 4, 5, 8])
    assert np.array_equal(degeneracies, [1, 4, 4, 2, 4, 1])
    assert degeneracies.sum() == 16
    assert np.allclose(radial[0], r_squared, atol=1e-12)
    assert np.allclose(radial[1], 2.0 * r_squared, atol=1e-12)


def test_axis_average_does_not_double_count_half_extent_displacements():
    correlators = np.ones((3, 4, 4), dtype=np.complex128)

    distances, degeneracies, axis = axis_average_correlators(correlators)

    assert np.array_equal(distances, [0, 1, 2])
    assert np.array_equal(degeneracies, [1, 4, 2])
    assert np.allclose(axis, 1.0 + 0.0j, atol=1e-12)


@pytest.mark.parametrize("shape", [(), (0, 2), (-1, 2)])
def test_periodic_displacement_squared_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        periodic_displacement_squared(shape)


def test_correlator_averages_reject_missing_configuration_axis():
    with pytest.raises(ValueError):
        radial_average_correlators(np.asarray([], dtype=np.complex128))
    with pytest.raises(ValueError):
        axis_average_correlators(np.empty((0, 2, 2), dtype=np.complex128))
