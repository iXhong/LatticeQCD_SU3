"""Plaquette, staple, and Wilson action calculations."""

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.group import dagger


def plaquette(
    links: np.ndarray, geometry: LatticeGeometry, site: int, mu: int, nu: int
) -> np.ndarray:
    """Compute one plaquette matrix.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        site: Starting flat site index.
        mu: First direction.
        nu: Second direction.
    Outputs:
        Complex 3x3 plaquette matrix U_mu,nu(site).
    """
    site_plus_mu = geometry.forward(site, mu)
    site_plus_nu = geometry.forward(site, nu)

    return (
        links[site, mu]
        @ links[site_plus_mu, nu]
        @ dagger(links[site_plus_nu, mu])
        @ dagger(links[site, nu])
    )


def average_plaquette(links: np.ndarray, geometry: LatticeGeometry) -> float:
    """Compute the average normalized plaquette.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
    Outputs:
        Average real trace plaquette divided by three.
    """
    plaquette_sum = 0.0
    plaquette_count = 0

    for mu in range(geometry.ndim):
        for nu in range(mu + 1, geometry.ndim):
            site_plus_mu = geometry.forward_neighbors[:, mu]
            site_plus_nu = geometry.forward_neighbors[:, nu]
            plaquettes = (
                links[:, mu]
                @ links[site_plus_mu, nu]
                @ links[site_plus_nu, mu].swapaxes(-1, -2).conj()
                @ links[:, nu].swapaxes(-1, -2).conj()
            )
            plaquette_sum += (
                np.real(np.trace(plaquettes, axis1=-2, axis2=-1)).sum() / 3.0
            )
            plaquette_count += geometry.volume

    if plaquette_count == 0:
        raise ValueError("geometry must have at least two dimensions")
    return float(plaquette_sum / plaquette_count)


def staple(links: np.ndarray, geometry: LatticeGeometry, site: int, mu: int) -> np.ndarray:
    """Compute the staple sum around one link.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        site: Flat site index of the link.
        mu: Direction index of the link.
    Outputs:
        Complex 3x3 staple sum.
    """
    staple_sum = np.zeros((3, 3), dtype=np.complex128)

    for nu in range(geometry.ndim):
        if nu == mu:
            continue

        site_plus_mu = geometry.forward(site, mu)
        site_plus_nu = geometry.forward(site, nu)
        site_minus_nu = geometry.backward(site, nu)
        site_plus_mu_minus_nu = geometry.backward(site_plus_mu, nu)

        forward_staple = (
            links[site_plus_mu, nu]
            @ dagger(links[site_plus_nu, mu])
            @ dagger(links[site, nu])
        )
        backward_staple = (
            dagger(links[site_plus_mu_minus_nu, nu])
            @ dagger(links[site_minus_nu, mu])
            @ links[site_minus_nu, nu]
        )
        staple_sum += forward_staple + backward_staple

    return staple_sum


def wilson_local_action(
    links: np.ndarray,
    geometry: LatticeGeometry,
    site: int,
    mu: int,
    beta: float,
    link_matrix: np.ndarray | None = None,
) -> float:
    """Compute the local Wilson action for one link.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        site: Flat site index of the link.
        mu: Direction index of the link.
        beta: Wilson gauge coupling parameter.
        link_matrix: Optional replacement matrix for U[site, mu].
    Outputs:
        Local Wilson action as a float.
    """
    if link_matrix is None:
        link_matrix = links[site, mu]

    constant = 2 * (geometry.ndim - 1)
    local_matrix = constant * np.eye(3, dtype=np.complex128) - link_matrix @ staple(
        links, geometry, site, mu
    )
    return float(beta / 3.0 * np.real(np.trace(local_matrix)))


def wilson_gauge_action(
    links: np.ndarray, geometry: LatticeGeometry, beta: float
) -> float:
    """Compute the full Wilson gauge action.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
    Outputs:
        Full Wilson gauge action as a float.
    """
    action = 0.0
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            for nu in range(mu + 1, geometry.ndim):
                action += beta * (
                    1.0 - np.real(np.trace(plaquette(links, geometry, site, mu, nu))) / 3.0
                )
    return float(action)
