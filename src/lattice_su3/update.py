"""Gauge-link update algorithms."""

from dataclasses import dataclass

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.group import dagger, embed_su2, random_su2, random_su2_near_identity
from lattice_su3.observables import staple, wilson_local_action


@dataclass(frozen=True)
class UpdateStats:
    """Track accepted and attempted link updates.

    Inputs:
        attempted_links: Number of proposed link updates.
        accepted_links: Number of accepted link updates.
    Outputs:
        Immutable update accounting object.
    """

    attempted_links: int
    accepted_links: int

    @property
    def acceptance_rate(self) -> float:
        """Compute the accepted fraction of attempted updates.

        Inputs:
            None.
        Outputs:
            Acceptance rate as a float.
        """
        if self.attempted_links == 0:
            return 0.0
        return self.accepted_links / self.attempted_links


def su3_metropolis_proposal(
    step_size: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a near-identity SU(3) Metropolis proposal.

    Inputs:
        step_size: Maximum SU(2) subgroup proposal radius in [0, 1].
        rng: Optional NumPy random generator.
    Outputs:
        Complex 3x3 SU(3) proposal matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    r_matrix = embed_su2(random_su2_near_identity(step_size, rng), 0, 1)
    s_matrix = embed_su2(random_su2_near_identity(step_size, rng), 0, 2)
    t_matrix = embed_su2(random_su2_near_identity(step_size, rng), 1, 2)
    return r_matrix @ s_matrix @ t_matrix


def su2_matrix_from_coefficients(
    x0: float, x1: float, x2: float, x3: float
) -> np.ndarray:
    """Construct an SU(2)-form matrix from real coefficients.

    Inputs:
        x0: Real identity coefficient.
        x1: Real first Pauli-vector coefficient.
        x2: Real second Pauli-vector coefficient.
        x3: Real third Pauli-vector coefficient.
    Outputs:
        Complex 2x2 matrix in SU(2) quaternion form.
    """
    return np.array(
        [
            [x0 + 1j * x3, x2 + 1j * x1],
            [-x2 + 1j * x1, x0 - 1j * x3],
        ],
        dtype=np.complex128,
    )


def su2_effective_staple(block: np.ndarray) -> np.ndarray:
    """Project a 2x2 block onto the SU(2) heatbath form.

    Inputs:
        block: Complex 2x2 active block from U times the staple.
    Outputs:
        Complex 2x2 matrix with the same SU(2) subgroup weight.
    """
    if block.shape != (2, 2):
        raise ValueError("block must have shape (2, 2)")

    p, q = block[0, 0], block[0, 1]
    r, s = block[1, 0], block[1, 1]
    x0 = 0.5 * np.real(p + s)
    x1 = 0.5 * np.imag(r + q)
    x2 = 0.5 * np.real(q - r)
    x3 = 0.5 * (np.imag(p) - np.imag(s))
    return su2_matrix_from_coefficients(x0, x1, x2, x3)


def sample_su2_heatbath(
    effective_staple: np.ndarray,
    beta_over_n: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one SU(2) heatbath matrix.

    Inputs:
        effective_staple: Complex 2x2 SU(2)-form effective staple.
        beta_over_n: Wilson beta divided by the gauge group size.
        rng: Optional NumPy random generator.
    Outputs:
        Complex 2x2 SU(2) heatbath update matrix.
    """
    if beta_over_n < 0.0:
        raise ValueError("beta_over_n must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

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

    x_matrix = su2_matrix_from_coefficients(x0, x1, x2, x3)
    normalized_staple = effective_staple / staple_norm
    return x_matrix @ dagger(normalized_staple)


def heatbath_update_link(
    links: np.ndarray,
    geometry: LatticeGeometry,
    site: int,
    mu: int,
    beta: float,
    rng: np.random.Generator | None = None,
) -> None:
    """Run one in-place Cabibbo-Marinari heatbath link update.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        site: Flat site index of the link.
        mu: Direction index of the link.
        beta: Wilson gauge coupling parameter.
        rng: Optional NumPy random generator.
    Outputs:
        None.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    link_matrix = links[site, mu]
    staple_matrix = staple(links, geometry, site, mu)
    for pair in ((0, 1), (0, 2), (1, 2)):
        active = np.ix_(pair, pair)
        block = (link_matrix @ staple_matrix)[active]
        subgroup_update = sample_su2_heatbath(
            su2_effective_staple(block), beta / 3.0, rng
        )
        link_matrix = embed_su2(subgroup_update, pair[0], pair[1]) @ link_matrix

    links[site, mu] = link_matrix


def heatbath_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    rng: np.random.Generator | None = None,
) -> UpdateStats:
    """Run one in-place heatbath sweep over all links.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        rng: Optional NumPy random generator.
    Outputs:
        UpdateStats with attempted links, accepted links, and acceptance rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    attempted_links = geometry.volume * geometry.ndim
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            heatbath_update_link(links, geometry, site, mu, beta, rng)

    return UpdateStats(
        attempted_links=attempted_links,
        accepted_links=attempted_links,
    )


def metropolis_update_link(
    links: np.ndarray,
    geometry: LatticeGeometry,
    site: int,
    mu: int,
    beta: float,
    step_size: float,
    rng: np.random.Generator | None = None,
) -> bool:
    """Attempt one in-place Metropolis link update.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        site: Flat site index of the link.
        mu: Direction index of the link.
        beta: Wilson gauge coupling parameter.
        step_size: Maximum SU(2) subgroup proposal radius in [0, 1].
        rng: Optional NumPy random generator.
    Outputs:
        True if the proposed link was accepted.
    """
    if rng is None:
        rng = np.random.default_rng()

    old_link = links[site, mu]
    new_link = su3_metropolis_proposal(step_size, rng) @ old_link

    old_action = wilson_local_action(links, geometry, site, mu, beta)
    new_action = wilson_local_action(
        links, geometry, site, mu, beta, link_matrix=new_link
    )
    delta_action = new_action - old_action

    if delta_action <= 0.0 or rng.random() < np.exp(-delta_action):
        links[site, mu] = new_link
        return True
    return False


def metropolis_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    step_size: float,
    rng: np.random.Generator | None = None,
) -> UpdateStats:
    """Run one in-place Metropolis sweep over all links.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        step_size: Maximum SU(2) subgroup proposal radius in [0, 1].
        rng: Optional NumPy random generator.
    Outputs:
        UpdateStats with attempted links, accepted links, and acceptance rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    accepted_links = 0
    attempted_links = geometry.volume * geometry.ndim
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            if metropolis_update_link(links, geometry, site, mu, beta, step_size, rng):
                accepted_links += 1

    return UpdateStats(
        attempted_links=attempted_links,
        accepted_links=accepted_links,
    )
