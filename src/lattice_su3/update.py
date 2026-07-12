"""Gauge-link update algorithms."""

from dataclasses import dataclass

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.group import embed_su2, random_su2, random_su2_near_identity
from lattice_su3.observables import staple, wilson_local_action


SU2_SUBGROUP_PAIRS = ((0, 1), (0, 2), (1, 2))


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


def su2_effective_staple_coefficients(
    block: np.ndarray,
) -> tuple[float, float, float, float]:
    """Project a 2x2 block onto SU(2)-form coefficients.

    Inputs:
        block: Complex 2x2 active block from U times the staple.
    Outputs:
        Real coefficients x0, x1, x2, x3 for the SU(2)-form matrix.
    """
    if block.shape != (2, 2):
        raise ValueError("block must have shape (2, 2)")

    return _su2_effective_staple_coefficients_from_entries(
        block[0, 0], block[0, 1], block[1, 0], block[1, 1]
    )


def _su2_effective_staple_coefficients_from_entries(
    p: complex,
    q: complex,
    r: complex,
    s: complex,
) -> tuple[float, float, float, float]:
    """Project 2x2 entries onto SU(2)-form coefficients.

    Inputs:
        p: Upper-left block entry.
        q: Upper-right block entry.
        r: Lower-left block entry.
        s: Lower-right block entry.
    Outputs:
        Real coefficients x0, x1, x2, x3 for the SU(2)-form matrix.
    """
    x0 = 0.5 * np.real(p + s)
    x1 = 0.5 * np.imag(r + q)
    x2 = 0.5 * np.real(q - r)
    x3 = 0.5 * (np.imag(p) - np.imag(s))
    return float(x0), float(x1), float(x2), float(x3)


def _sample_su2_heatbath_from_coefficients(
    x0: float,
    x1: float,
    x2: float,
    x3: float,
    beta_over_n: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one SU(2) heatbath matrix from staple coefficients.

    Inputs:
        x0: Real identity coefficient of the effective staple.
        x1: Real first Pauli-vector coefficient of the effective staple.
        x2: Real second Pauli-vector coefficient of the effective staple.
        x3: Real third Pauli-vector coefficient of the effective staple.
        beta_over_n: Wilson beta divided by the gauge group size.
        rng: NumPy random generator.
    Outputs:
        Complex 2x2 SU(2) heatbath update matrix.
    """
    staple_norm = np.sqrt(max(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3, 0.0))
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

    y0 = 1.0 - 2.0 * lambda_squared
    vector_norm = np.sqrt(max(1.0 - y0 * y0, 0.0))
    direction = rng.normal(size=3)
    direction_norm = np.linalg.norm(direction)
    while direction_norm == 0.0:
        direction = rng.normal(size=3)
        direction_norm = np.linalg.norm(direction)
    y1, y2, y3 = vector_norm * direction / direction_norm

    x_matrix = su2_matrix_from_coefficients(y0, y1, y2, y3)
    inv_staple_norm = 1.0 / staple_norm
    normalized_staple_dagger = np.array(
        [
            [(x0 - 1j * x3) * inv_staple_norm, (-x2 - 1j * x1) * inv_staple_norm],
            [(x2 - 1j * x1) * inv_staple_norm, (x0 + 1j * x3) * inv_staple_norm],
        ],
        dtype=np.complex128,
    )
    return x_matrix @ normalized_staple_dagger


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

    return _sample_su2_heatbath_from_coefficients(
        *su2_effective_staple_coefficients(effective_staple), beta_over_n, rng
    )


def _active_block_effective_staple_coefficients(
    link_matrix: np.ndarray,
    staple_matrix: np.ndarray,
    i: int,
    j: int,
) -> tuple[float, float, float, float]:
    """Compute effective staple coefficients for an active SU(2) block.

    Inputs:
        link_matrix: Current SU(3) link matrix.
        staple_matrix: SU(3) staple matrix for the link.
        i: First active row/column index.
        j: Second active row/column index.
    Outputs:
        Real coefficients x0, x1, x2, x3 for the SU(2)-form matrix.
    """
    p = link_matrix[i, 0] * staple_matrix[0, i]
    p += link_matrix[i, 1] * staple_matrix[1, i]
    p += link_matrix[i, 2] * staple_matrix[2, i]

    q = link_matrix[i, 0] * staple_matrix[0, j]
    q += link_matrix[i, 1] * staple_matrix[1, j]
    q += link_matrix[i, 2] * staple_matrix[2, j]

    r = link_matrix[j, 0] * staple_matrix[0, i]
    r += link_matrix[j, 1] * staple_matrix[1, i]
    r += link_matrix[j, 2] * staple_matrix[2, i]

    s = link_matrix[j, 0] * staple_matrix[0, j]
    s += link_matrix[j, 1] * staple_matrix[1, j]
    s += link_matrix[j, 2] * staple_matrix[2, j]

    return _su2_effective_staple_coefficients_from_entries(p, q, r, s)


def _left_multiply_active_rows(
    link_matrix: np.ndarray,
    subgroup_update: np.ndarray,
    i: int,
    j: int,
) -> None:
    """Left-multiply two active rows by an SU(2) update in place.

    Inputs:
        link_matrix: SU(3) link matrix to update.
        subgroup_update: Complex 2x2 SU(2) update matrix.
        i: First active row index.
        j: Second active row index.
    Outputs:
        None.
    """
    row_i = link_matrix[i].copy()
    row_j = link_matrix[j].copy()
    link_matrix[i] = subgroup_update[0, 0] * row_i + subgroup_update[0, 1] * row_j
    link_matrix[j] = subgroup_update[1, 0] * row_i + subgroup_update[1, 1] * row_j


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

    link_matrix = links[site, mu].copy()
    staple_matrix = staple(links, geometry, site, mu)
    beta_over_n = beta / 3.0
    for i, j in SU2_SUBGROUP_PAIRS:
        subgroup_update = _sample_su2_heatbath_from_coefficients(
            *_active_block_effective_staple_coefficients(link_matrix, staple_matrix, i, j),
            beta_over_n,
            rng,
        )
        _left_multiply_active_rows(link_matrix, subgroup_update, i, j)

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


def _site_parities(geometry: LatticeGeometry) -> np.ndarray:
    """Compute checkerboard parity for each lattice site.

    Inputs:
        geometry: Lattice geometry object.
    Outputs:
        Integer array with parity zero or one for each flat site.
    """
    parities = np.empty(geometry.volume, dtype=np.int8)
    for site in range(geometry.volume):
        parities[site] = sum(geometry.coord_from_index(site)) % 2
    return parities


def heatbath_checkerboard_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    rng: np.random.Generator | None = None,
) -> UpdateStats:
    """Run one checkerboard heatbath sweep over all links.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        rng: Optional NumPy random generator.
    Outputs:
        UpdateStats with attempted links, accepted links, and acceptance rate.
    """
    if any(length % 2 != 0 for length in geometry.shape):
        raise ValueError("checkerboard sweep requires even lattice lengths")
    if rng is None:
        rng = np.random.default_rng()

    parities = _site_parities(geometry)
    attempted_links = geometry.volume * geometry.ndim
    for mu in range(geometry.ndim):
        for parity in (0, 1):
            for site in range(geometry.volume):
                if parities[site] == parity:
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
