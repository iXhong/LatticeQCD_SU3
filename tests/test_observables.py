import numpy as np
import pytest

from lattice_su3 import (
    LatticeGeometry,
    cold_start,
    hot_start,
    plaquette,
    polyakov_loop,
    polyakov_loop_correlator,
    polyakov_loop_correlator_from_loops,
    polyakov_loops,
    random_su3,
    staple,
    wilson_gauge_action,
    wilson_local_action,
)


def direct_polyakov_loop_correlator(loops: np.ndarray) -> np.ndarray:
    """Compute the Polyakov loop correlator by explicit periodic shifts.

    Inputs:
        loops: Complex Polyakov loop field over the spatial lattice.
    Outputs:
        Complex correlator C(r) = mean_x P(x) conj(P(x + r)).
    """
    correlator = np.empty_like(loops, dtype=np.complex128)
    axes = tuple(range(loops.ndim))
    for displacement in np.ndindex(loops.shape):
        shifted = np.roll(
            loops,
            shift=tuple(-offset for offset in displacement),
            axis=axes,
        )
        correlator[displacement] = np.mean(loops * shifted.conj())
    return correlator


def test_cold_start_has_zero_wilson_gauge_action():
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    assert np.isclose(wilson_gauge_action(links, geometry, beta=5.5), 0.0)


def test_plaquette_and_staple_shapes():
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    assert plaquette(links, geometry, site=0, mu=0, nu=1).shape == (3, 3)
    assert staple(links, geometry, site=0, mu=0).shape == (3, 3)


def test_cold_start_polyakov_loops_are_one():
    geometry = LatticeGeometry((2, 3, 4, 5))
    links = cold_start(geometry)

    loops = polyakov_loops(links, geometry)

    assert loops.shape == (2, 3, 4)
    assert np.allclose(loops, 1.0 + 0.0j, atol=1e-12)
    assert np.allclose(
        polyakov_loop(links, geometry, spatial_coords=(1, 2, 3)),
        loops[1, 2, 3],
        atol=1e-12,
    )


def test_cold_start_polyakov_loop_correlator_is_one():
    geometry = LatticeGeometry((2, 3, 4, 5))
    links = cold_start(geometry)

    correlator = polyakov_loop_correlator(links, geometry)

    assert correlator.shape == (2, 3, 4)
    assert np.allclose(correlator, 1.0 + 0.0j, atol=1e-12)


def test_polyakov_loop_correlator_matches_direct_periodic_sum():
    loops = np.asarray(
        [
            [[1.0 + 0.5j, -0.25 + 0.75j], [0.4 - 0.2j, -1.0 + 0.1j]],
            [[0.8 + 0.3j, -0.6 - 0.4j], [0.2 + 1.1j, -0.7 + 0.9j]],
        ],
        dtype=np.complex128,
    )

    correlator = polyakov_loop_correlator_from_loops(loops)
    expected = direct_polyakov_loop_correlator(loops)

    assert np.allclose(correlator, expected, atol=1e-12)


def test_polyakov_loop_matches_hand_constructed_temporal_product():
    geometry = LatticeGeometry((2, 2, 3))
    links = cold_start(geometry)
    phase = np.exp(2j * np.pi / 9.0)
    temporal_link = np.diag(
        np.asarray([phase, phase, phase.conjugate() ** 2], dtype=np.complex128)
    )
    expected_product = np.linalg.matrix_power(temporal_link, geometry.shape[-1])

    for time_coord in range(geometry.shape[-1]):
        site = geometry.index_from_coord((1, 0, time_coord))
        links[site, -1] = temporal_link

    assert np.allclose(
        polyakov_loop(links, geometry, spatial_coords=(1, 0)),
        np.trace(expected_product) / 3.0,
        atol=1e-12,
    )


def test_polyakov_loop_supports_non_default_time_direction():
    geometry = LatticeGeometry((3, 2, 2))
    links = cold_start(geometry)
    phase = np.exp(2j * np.pi / 9.0)
    temporal_link = np.diag(
        np.asarray([phase, phase, phase.conjugate() ** 2], dtype=np.complex128)
    )
    expected_product = np.linalg.matrix_power(temporal_link, geometry.shape[0])

    for time_coord in range(geometry.shape[0]):
        site = geometry.index_from_coord((time_coord, 1, 0))
        links[site, 0] = temporal_link

    loops = polyakov_loops(links, geometry, time_direction=0)

    assert loops.shape == (2, 2)
    assert np.allclose(loops[1, 0], np.trace(expected_product) / 3.0, atol=1e-12)
    assert np.allclose(
        polyakov_loop(links, geometry, spatial_coords=(1, 0), time_direction=0),
        loops[1, 0],
        atol=1e-12,
    )


def test_polyakov_loop_correlator_supports_non_default_time_direction():
    geometry = LatticeGeometry((3, 2, 2))
    links = cold_start(geometry)

    correlator = polyakov_loop_correlator(links, geometry, time_direction=0)

    assert correlator.shape == (2, 2)
    assert np.allclose(correlator, 1.0 + 0.0j, atol=1e-12)


def test_polyakov_loops_are_gauge_invariant():
    geometry = LatticeGeometry((2, 2, 2, 3))
    rng = np.random.default_rng(918)
    links = hot_start(geometry, rng)
    gauge_matrices = np.asarray(
        [random_su3(rng) for _ in range(geometry.volume)], dtype=np.complex128
    )
    transformed_links = np.empty_like(links)

    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            transformed_links[site, mu] = (
                gauge_matrices[site]
                @ links[site, mu]
                @ gauge_matrices[geometry.forward(site, mu)].conj().T
            )

    assert np.allclose(
        polyakov_loops(transformed_links, geometry),
        polyakov_loops(links, geometry),
        atol=1e-12,
    )


def test_polyakov_loop_correlator_is_gauge_invariant():
    geometry = LatticeGeometry((2, 2, 2, 3))
    rng = np.random.default_rng(917)
    links = hot_start(geometry, rng)
    gauge_matrices = np.asarray(
        [random_su3(rng) for _ in range(geometry.volume)], dtype=np.complex128
    )
    transformed_links = np.empty_like(links)

    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            transformed_links[site, mu] = (
                gauge_matrices[site]
                @ links[site, mu]
                @ gauge_matrices[geometry.forward(site, mu)].conj().T
            )

    assert np.allclose(
        polyakov_loop_correlator(transformed_links, geometry),
        polyakov_loop_correlator(links, geometry),
        atol=1e-12,
    )


def test_polyakov_loop_correlator_is_z3_center_invariant():
    geometry = LatticeGeometry((2, 2, 2, 3))
    rng = np.random.default_rng(916)
    links = hot_start(geometry, rng)
    transformed_links = links.copy()
    center_phase = np.exp(2j * np.pi / 3.0)

    for spatial_coords in np.ndindex(geometry.shape[:-1]):
        site = geometry.index_from_coord(spatial_coords + (0,))
        transformed_links[site, -1] = center_phase * transformed_links[site, -1]

    assert np.allclose(
        polyakov_loops(transformed_links, geometry),
        center_phase * polyakov_loops(links, geometry),
        atol=1e-12,
    )
    assert np.allclose(
        polyakov_loop_correlator(transformed_links, geometry),
        polyakov_loop_correlator(links, geometry),
        atol=1e-12,
    )


def test_polyakov_loop_rejects_invalid_inputs():
    links = cold_start(LatticeGeometry((2, 2, 2)))

    with pytest.raises(ValueError):
        polyakov_loops(links, LatticeGeometry((2,)), time_direction=0)
    with pytest.raises(ValueError):
        polyakov_loop(links, LatticeGeometry((2, 2, 2)), (0, 0), time_direction=3)
    with pytest.raises(ValueError):
        polyakov_loop(links, LatticeGeometry((2, 2, 2)), (0,), time_direction=-1)
    with pytest.raises(ValueError):
        polyakov_loop(links, LatticeGeometry((2, 2, 2)), (0, 2), time_direction=-1)
    with pytest.raises(ValueError):
        polyakov_loop_correlator_from_loops(np.asarray([], dtype=np.complex128))


def test_wilson_local_action_delta_matches_full_action_delta():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(123)
    links = hot_start(geometry, rng)
    beta = 5.7
    site = geometry.index_from_coord((1, 0, 1, 0))
    mu = 2
    new_link = random_su3(rng)

    old_full_action = wilson_gauge_action(links, geometry, beta)
    old_local_action = wilson_local_action(links, geometry, site, mu, beta)
    new_local_action = wilson_local_action(
        links, geometry, site, mu, beta, link_matrix=new_link
    )

    new_links = links.copy()
    new_links[site, mu] = new_link
    new_full_action = wilson_gauge_action(new_links, geometry, beta)

    assert np.isclose(
        new_local_action - old_local_action,
        new_full_action - old_full_action,
        atol=1e-4,
    )
