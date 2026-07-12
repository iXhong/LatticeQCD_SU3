import numpy as np
import pytest

import lattice_su3.update as update
from lattice_su3 import (
    LatticeGeometry,
    cold_start,
    heatbath_sweep,
    heatbath_update_link,
    hot_start,
    is_su3,
    metropolis_sweep,
    metropolis_update_link,
    plaquette,
    polyakov_loop,
    polyakov_loops,
    random_su3,
    random_su2,
    sample_su2_heatbath,
    staple,
    su2_effective_staple,
    su3_metropolis_proposal,
    wilson_gauge_action,
    wilson_local_action,
)


def test_lattice_geometry_index_coordinate_conversion():
    geometry = LatticeGeometry((2, 3, 4, 5))

    assert geometry.ndim == 4
    assert geometry.volume == 120
    assert geometry.index_from_coord((0, 0, 0, 0)) == 0
    assert geometry.index_from_coord((1, 2, 3, 4)) == 119
    assert geometry.coord_from_index(0) == (0, 0, 0, 0)
    assert geometry.coord_from_index(119) == (1, 2, 3, 4)


def test_lattice_geometry_periodic_neighbors_wrap_at_boundaries():
    geometry = LatticeGeometry((2, 3, 4, 5))

    site = geometry.index_from_coord((0, 0, 0, 0))
    assert geometry.coord_from_index(geometry.forward(site, 0)) == (1, 0, 0, 0)
    assert geometry.coord_from_index(geometry.backward(site, 0)) == (1, 0, 0, 0)
    assert geometry.coord_from_index(geometry.forward(site, 1)) == (0, 1, 0, 0)
    assert geometry.coord_from_index(geometry.backward(site, 1)) == (0, 2, 0, 0)
    assert geometry.coord_from_index(geometry.forward(site, 2)) == (0, 0, 1, 0)
    assert geometry.coord_from_index(geometry.backward(site, 2)) == (0, 0, 3, 0)
    assert geometry.coord_from_index(geometry.forward(site, 3)) == (0, 0, 0, 1)
    assert geometry.coord_from_index(geometry.backward(site, 3)) == (0, 0, 0, 4)

    corner = geometry.index_from_coord((1, 2, 3, 4))
    assert geometry.coord_from_index(geometry.forward(corner, 0)) == (0, 2, 3, 4)
    assert geometry.coord_from_index(geometry.backward(corner, 0)) == (0, 2, 3, 4)
    assert geometry.coord_from_index(geometry.forward(corner, 1)) == (1, 0, 3, 4)
    assert geometry.coord_from_index(geometry.backward(corner, 1)) == (1, 1, 3, 4)
    assert geometry.coord_from_index(geometry.forward(corner, 2)) == (1, 2, 0, 4)
    assert geometry.coord_from_index(geometry.backward(corner, 2)) == (1, 2, 2, 4)
    assert geometry.coord_from_index(geometry.forward(corner, 3)) == (1, 2, 3, 0)
    assert geometry.coord_from_index(geometry.backward(corner, 3)) == (1, 2, 3, 3)


def test_lattice_geometry_precomputed_neighbor_tables_match_methods():
    geometry = LatticeGeometry((3, 4, 5, 6))

    for site in (0, 1, 17, geometry.volume - 1):
        for mu in range(geometry.ndim):
            assert geometry.forward(site, mu) == geometry.forward_neighbors[site, mu]
            assert geometry.backward(site, mu) == geometry.backward_neighbors[site, mu]


def test_lattice_geometry_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        LatticeGeometry(())
    with pytest.raises(ValueError):
        LatticeGeometry((4, 0, 4, 4))
    with pytest.raises(ValueError):
        LatticeGeometry((4, 4)).index_from_coord((4, 0))
    with pytest.raises(ValueError):
        LatticeGeometry((4, 4)).coord_from_index(16)


def test_random_su3_is_unitary_and_has_unit_determinant():
    rng = np.random.default_rng(12345)
    identity = np.eye(3, dtype=np.complex128)

    for _ in range(100):
        x = random_su3(rng)

        assert x.shape == (3, 3)
        assert np.allclose(x.conj().T @ x, identity, atol=1e-12)
        assert np.allclose(np.linalg.det(x), 1.0 + 0.0j, atol=1e-12)


def test_is_su3_accepts_generated_su3_matrices():
    rng = np.random.default_rng(2024)

    for _ in range(100):
        assert is_su3(random_su3(rng))


def test_is_su3_rejects_non_su3_matrices():
    assert not is_su3(np.eye(2, dtype=np.complex128))
    assert not is_su3(2.0 * np.eye(3, dtype=np.complex128))


def test_hot_start_creates_random_su3_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(7)

    links = hot_start(geometry, rng)

    assert links.shape == (geometry.volume, geometry.ndim, 3, 3)
    assert links.dtype == np.complex128
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu])


def test_cold_start_creates_identity_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    identity = np.eye(3, dtype=np.complex128)

    links = cold_start(geometry)

    assert links.shape == (geometry.volume, geometry.ndim, 3, 3)
    assert links.dtype == np.complex128
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert np.allclose(links[site, mu], identity)
            assert is_su3(links[site, mu])


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
        atol=1e-12,
    )


def test_su3_metropolis_proposal_is_su3():
    rng = np.random.default_rng(4321)

    for _ in range(100):
        proposal = su3_metropolis_proposal(step_size=0.25, rng=rng)

        assert proposal.shape == (3, 3)
        assert is_su3(proposal)


def test_su3_metropolis_proposal_rejects_invalid_step_size():
    rng = np.random.default_rng(4321)

    with pytest.raises(ValueError):
        su3_metropolis_proposal(step_size=-0.1, rng=rng)
    with pytest.raises(ValueError):
        su3_metropolis_proposal(step_size=1.1, rng=rng)


def test_metropolis_update_link_preserves_su3_link():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(91)
    links = hot_start(geometry, rng)

    metropolis_update_link(
        links,
        geometry,
        site=geometry.index_from_coord((1, 0, 1, 0)),
        mu=2,
        beta=5.7,
        step_size=0.2,
        rng=rng,
    )

    assert is_su3(links[geometry.index_from_coord((1, 0, 1, 0)), 2])


def test_metropolis_sweep_step_size_zero_has_unit_acceptance_rate():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(314)
    links = hot_start(geometry, rng)
    old_links = links.copy()

    stats = metropolis_sweep(links, geometry, beta=5.7, step_size=0.0, rng=rng)

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    assert np.allclose(links, old_links, atol=1e-12)


def test_metropolis_sweep_reports_acceptance_rate():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(2718)
    links = hot_start(geometry, rng)

    stats = metropolis_sweep(links, geometry, beta=5.7, step_size=0.2, rng=rng)

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert 0 <= stats.accepted_links <= stats.attempted_links
    assert stats.acceptance_rate == stats.accepted_links / stats.attempted_links
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu], atol=1e-11)


def test_metropolis_acceptance_rate_for_unit_action_increase(monkeypatch):
    class GridRng:
        """Return deterministic uniform samples.

        Inputs:
            count: Number of samples spanning [0, 1).
        Outputs:
            Random-like object with a random method.
        """

        def __init__(self, count: int) -> None:
            """Initialize the deterministic random stream.

            Inputs:
                count: Number of samples spanning [0, 1).
            Outputs:
                None.
            """
            self.values = (np.arange(count) + 0.5) / count
            self.index = 0

        def random(self) -> float:
            """Return the next deterministic uniform sample.

            Inputs:
                None.
            Outputs:
                Uniform sample as a float.
            """
            value = float(self.values[self.index])
            self.index += 1
            return value

    geometry = LatticeGeometry((100, 100))
    links = cold_start(geometry)
    target_rate = float(np.exp(-1.0))

    monkeypatch.setattr(
        update,
        "su3_metropolis_proposal",
        lambda step_size, rng: np.eye(3, dtype=np.complex128),
    )

    def local_action(
        links, geometry, site, mu, beta, link_matrix=None
    ):
        """Return local actions with delta action equal to one.

        Inputs:
            links: Gauge links U[site, direction].
            geometry: Lattice geometry object.
            site: Flat site index of the link.
            mu: Direction index of the link.
            beta: Wilson gauge coupling parameter.
            link_matrix: Optional replacement matrix for U[site, mu].
        Outputs:
            Local action value as a float.
        """
        return 1.0 if link_matrix is not None else 0.0

    monkeypatch.setattr(update, "wilson_local_action", local_action)

    stats = metropolis_sweep(
        links,
        geometry,
        beta=5.7,
        step_size=0.1,
        rng=GridRng(geometry.volume * geometry.ndim),
    )

    assert np.isclose(stats.acceptance_rate, target_rate, atol=1 / stats.attempted_links)


def test_su2_effective_staple_preserves_subgroup_weight():
    rng = np.random.default_rng(802)
    block = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    effective_staple = su2_effective_staple(block)

    for _ in range(20):
        subgroup = random_su2(rng)

        assert np.isclose(
            np.real(np.trace(subgroup @ block)),
            np.real(np.trace(subgroup @ effective_staple)),
            atol=1e-12,
        )


def test_sample_su2_heatbath_returns_su2_matrix():
    rng = np.random.default_rng(803)
    effective_staple = su2_effective_staple(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    )
    identity = np.eye(2, dtype=np.complex128)

    for _ in range(20):
        sample = sample_su2_heatbath(effective_staple, beta_over_n=5.7 / 3.0, rng=rng)

        assert sample.shape == (2, 2)
        assert np.allclose(sample.conj().T @ sample, identity, atol=1e-12)
        assert np.allclose(np.linalg.det(sample), 1.0 + 0.0j, atol=1e-12)


def test_heatbath_update_link_preserves_su3_link():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(804)
    links = hot_start(geometry, rng)
    site = geometry.index_from_coord((1, 0, 1, 0))
    mu = 2

    heatbath_update_link(links, geometry, site, mu, beta=5.7, rng=rng)

    assert is_su3(links[site, mu], atol=1e-11)


def test_heatbath_sweep_reports_unit_acceptance_rate_and_preserves_su3_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(805)
    links = hot_start(geometry, rng)

    stats = heatbath_sweep(links, geometry, beta=5.7, rng=rng)

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu], atol=1e-11)


def test_heatbath_rejects_negative_beta():
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    with pytest.raises(ValueError):
        heatbath_update_link(links, geometry, site=0, mu=0, beta=-1.0)
