"""Gauge-link update algorithms."""

from dataclasses import dataclass

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.group import embed_su2, random_su2_near_identity
from lattice_su3.observables import wilson_local_action


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
