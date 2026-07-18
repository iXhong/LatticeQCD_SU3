import numpy as np
import pytest

import lattice_su3.update as update
from lattice_su3 import (
    LatticeGeometry,
    average_plaquette,
    cold_start,
    dagger,
    embed_su2,
    heatbath_checkerboard_sweep,
    heatbath_sweep,
    heatbath_update_link,
    hot_start,
    is_su3,
    metropolis_sweep,
    metropolis_update_link,
    overrelaxation_sweep,
    overrelaxation_update_link,
    random_su2,
    sample_su2_heatbath,
    staple,
    su2_effective_staple,
    su3_metropolis_proposal,
    wilson_gauge_action,
    wilson_local_action,
)


SINGLE_PRECISION_ATOL = 1e-5


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
            assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


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


def reference_sample_su2_heatbath(
    effective_staple: np.ndarray,
    beta_over_n: float,
    rng,
) -> np.ndarray:
    """Sample one SU(2) heatbath matrix with the baseline implementation.

    Inputs:
        effective_staple: Complex 2x2 SU(2)-form effective staple.
        beta_over_n: Wilson beta divided by the gauge group size.
        rng: NumPy random generator.
    Outputs:
        Complex 2x2 SU(2) heatbath update matrix.
    """
    determinant = float(np.real(np.linalg.det(effective_staple)))
    staple_norm = np.sqrt(max(determinant, 0.0))
    if staple_norm <= 1e-14 or beta_over_n == 0.0:
        return random_su2(rng)

    alpha = 2.0 * beta_over_n * staple_norm
    while True:
        r1 = max(rng.random(), np.finfo(float).tiny)
        r2 = rng.random()
        r3 = max(rng.random(), np.finfo(float).tiny)
        cos_term = np.cos(2.0 * np.pi * r2)
        lambda_squared = -(np.log(r1) + cos_term * cos_term * np.log(r3)) / (
            2.0 * alpha
        )
        if lambda_squared <= 1.0 and rng.random() ** 2 <= 1.0 - lambda_squared:
            break

    x0 = 1.0 - 2.0 * lambda_squared
    vector_norm = np.sqrt(max(1.0 - x0 * x0, 0.0))
    direction = rng.normal(size=3)
    direction_norm = np.linalg.norm(direction)
    while direction_norm == 0.0:
        direction = rng.normal(size=3)
        direction_norm = np.linalg.norm(direction)
    x1, x2, x3 = vector_norm * direction / direction_norm

    x_matrix = update.su2_matrix_from_coefficients(x0, x1, x2, x3)
    normalized_staple = effective_staple / staple_norm
    return x_matrix @ dagger(normalized_staple)


def reference_heatbath_update_link(
    links: np.ndarray,
    geometry: LatticeGeometry,
    site: int,
    mu: int,
    beta: float,
    rng,
) -> None:
    """Run one baseline Cabibbo-Marinari heatbath link update.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        site: Flat site index of the link.
        mu: Direction index of the link.
        beta: Wilson gauge coupling parameter.
        rng: NumPy random generator.
    Outputs:
        None.
    """
    link_matrix = links[site, mu]
    staple_matrix = staple(links, geometry, site, mu)
    for pair in ((0, 1), (0, 2), (1, 2)):
        active = np.ix_(pair, pair)
        block = (link_matrix @ staple_matrix)[active]
        subgroup_update = reference_sample_su2_heatbath(
            su2_effective_staple(block), beta / 3.0, rng
        )
        link_matrix = embed_su2(subgroup_update, pair[0], pair[1]) @ link_matrix

    links[site, mu] = link_matrix


def reference_heatbath_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    rng,
) -> None:
    """Run one baseline heatbath sweep over all links.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        rng: NumPy random generator.
    Outputs:
        None.
    """
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            reference_heatbath_update_link(links, geometry, site, mu, beta, rng)


def test_heatbath_sweep_matches_baseline_implementation_for_fixed_seed():
    geometry = LatticeGeometry((2, 2, 2, 2))
    initial_rng = np.random.default_rng(806)
    links = hot_start(geometry, initial_rng)
    optimized_links = links.copy()
    reference_links = links.copy()

    heatbath_sweep(optimized_links, geometry, beta=5.7, rng=np.random.default_rng(807))
    reference_heatbath_sweep(
        reference_links, geometry, beta=5.7, rng=np.random.default_rng(807)
    )

    assert np.allclose(optimized_links, reference_links, atol=1e-12)


def test_heatbath_update_link_preserves_su3_link():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(804)
    links = hot_start(geometry, rng)
    site = geometry.index_from_coord((1, 0, 1, 0))
    mu = 2

    heatbath_update_link(links, geometry, site, mu, beta=5.7, rng=rng)

    assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


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
            assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_overrelaxation_update_link_preserves_local_action_and_su3_link():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(811)
    links = hot_start(geometry, rng)
    site = geometry.index_from_coord((1, 0, 1, 0))
    mu = 2
    old_action = wilson_local_action(links, geometry, site, mu, beta=5.7)

    overrelaxation_update_link(links, geometry, site, mu, rng=rng)

    new_action = wilson_local_action(links, geometry, site, mu, beta=5.7)
    assert np.isclose(new_action, old_action, atol=SINGLE_PRECISION_ATOL)
    assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_overrelaxation_sweep_reports_stats_and_preserves_su3_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(812)
    links = hot_start(geometry, rng)

    stats = overrelaxation_sweep(links, geometry, rng=rng)

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    assert np.isfinite(average_plaquette(links, geometry))
    assert np.isfinite(wilson_gauge_action(links, geometry, beta=5.7))
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_overrelaxation_jit_sweep_matches_numpy_implementation():
    accelerated = pytest.importorskip("lattice_su3.accelerated")
    pytest.importorskip("numba")
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(813)
    links = hot_start(geometry, rng)
    numpy_links = links.copy()
    jit_links = links.copy()

    overrelaxation_sweep(numpy_links, geometry, rng=np.random.default_rng(814))
    stats = accelerated.overrelaxation_jit_sweep(jit_links, geometry, seed=814)

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    assert np.isfinite(average_plaquette(numpy_links, geometry))
    assert np.isfinite(average_plaquette(jit_links, geometry))
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(numpy_links[site, mu], atol=SINGLE_PRECISION_ATOL)
            assert is_su3(jit_links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_overrelaxation_checkerboard_jit_sweep_preserves_su3_links():
    accelerated = pytest.importorskip("lattice_su3.accelerated")
    pytest.importorskip("numba")
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(815)
    links = hot_start(geometry, rng)

    stats = accelerated.overrelaxation_checkerboard_jit_sweep(
        links, geometry, seed=816
    )

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    assert np.isfinite(average_plaquette(links, geometry))
    assert np.isfinite(wilson_gauge_action(links, geometry, beta=5.7))
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_overrelaxation_checkerboard_jit_sweep_rejects_odd_lattice_lengths():
    accelerated = pytest.importorskip("lattice_su3.accelerated")
    pytest.importorskip("numba")
    geometry = LatticeGeometry((3, 2, 2, 2))
    links = cold_start(geometry)

    with pytest.raises(ValueError):
        accelerated.overrelaxation_checkerboard_jit_sweep(links, geometry)


def test_heatbath_checkerboard_sweep_reports_stats_and_preserves_su3_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(806)
    links = hot_start(geometry, rng)

    stats = heatbath_checkerboard_sweep(links, geometry, beta=5.7, rng=rng)

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    assert np.isfinite(average_plaquette(links, geometry))
    assert np.isfinite(wilson_gauge_action(links, geometry, beta=5.7))
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_heatbath_checkerboard_sweep_rejects_odd_lattice_lengths():
    geometry = LatticeGeometry((3, 2, 2, 2))
    links = cold_start(geometry)

    with pytest.raises(ValueError):
        heatbath_checkerboard_sweep(links, geometry, beta=5.7)


def test_heatbath_jit_sweep_reports_stats_and_preserves_su3_links():
    accelerated = pytest.importorskip("lattice_su3.accelerated")
    pytest.importorskip("numba")
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(807)
    links = hot_start(geometry, rng)

    stats = accelerated.heatbath_jit_sweep(links, geometry, beta=5.7, seed=808)

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    assert np.isfinite(average_plaquette(links, geometry))
    assert np.isfinite(wilson_gauge_action(links, geometry, beta=5.7))
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_heatbath_checkerboard_jit_sweep_reports_stats_and_preserves_su3_links():
    accelerated = pytest.importorskip("lattice_su3.accelerated")
    pytest.importorskip("numba")
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(809)
    links = hot_start(geometry, rng)

    stats = accelerated.heatbath_checkerboard_jit_sweep(
        links, geometry, beta=5.7, seed=810
    )

    assert stats.attempted_links == geometry.volume * geometry.ndim
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0
    assert np.isfinite(average_plaquette(links, geometry))
    assert np.isfinite(wilson_gauge_action(links, geometry, beta=5.7))
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu], atol=SINGLE_PRECISION_ATOL)


def test_heatbath_checkerboard_jit_sweep_rejects_odd_lattice_lengths():
    accelerated = pytest.importorskip("lattice_su3.accelerated")
    pytest.importorskip("numba")
    geometry = LatticeGeometry((3, 2, 2, 2))
    links = cold_start(geometry)

    with pytest.raises(ValueError):
        accelerated.heatbath_checkerboard_jit_sweep(links, geometry, beta=5.7)


def test_heatbath_rejects_negative_beta():
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    with pytest.raises(ValueError):
        heatbath_update_link(links, geometry, site=0, mu=0, beta=-1.0)
