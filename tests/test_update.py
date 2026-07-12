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
    random_su2,
    sample_su2_heatbath,
    su2_effective_staple,
    su3_metropolis_proposal,
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

    def local_action(links, geometry, site, mu, beta, link_matrix=None):
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
