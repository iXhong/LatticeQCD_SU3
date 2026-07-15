"""Plaquette, staple, and Wilson action calculations."""

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.group import dagger


def _normalize_direction(geometry: LatticeGeometry, direction: int) -> int:
    """Normalize a possibly negative direction index.

    Inputs:
        geometry: Lattice geometry object.
        direction: Direction index, allowing negative Python indexing.
    Outputs:
        Non-negative direction index.
    """
    normalized_direction = direction % geometry.ndim
    if direction < -geometry.ndim or direction >= geometry.ndim:
        raise ValueError("direction is outside the lattice dimensions")
    return normalized_direction


def _full_coord_from_spatial_coord(
    geometry: LatticeGeometry,
    spatial_coords: tuple[int, ...],
    time_direction: int,
    time_coord: int,
) -> tuple[int, ...]:
    """Insert a time coordinate into a spatial coordinate tuple.

    Inputs:
        geometry: Lattice geometry object.
        spatial_coords: Coordinates excluding the time direction.
        time_direction: Non-negative time direction index.
        time_coord: Coordinate in the time direction.
    Outputs:
        Full lattice coordinate tuple.
    """
    if len(spatial_coords) != geometry.ndim - 1:
        raise ValueError("spatial_coords must exclude exactly one time direction")

    spatial_shape = (
        geometry.shape[:time_direction] + geometry.shape[time_direction + 1 :]
    )
    if any(
        coord < 0 or coord >= length
        for coord, length in zip(spatial_coords, spatial_shape)
    ):
        raise ValueError("spatial_coords are outside the spatial lattice")

    return (
        spatial_coords[:time_direction]
        + (time_coord,)
        + spatial_coords[time_direction:]
    )


def polyakov_loop(
    links: np.ndarray,
    geometry: LatticeGeometry,
    spatial_coords: tuple[int, ...],
    time_direction: int = -1,
) -> complex:
    """Compute one normalized Polyakov loop.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        spatial_coords: Coordinates excluding the time direction.
        time_direction: Direction used as Euclidean time.
    Outputs:
        Complex trace of the temporal Wilson line divided by three.
    """
    if geometry.ndim < 2:
        raise ValueError("geometry must have at least two dimensions")
    time_direction = _normalize_direction(geometry, time_direction)
    temporal_extent = geometry.shape[time_direction]
    wilson_line = np.eye(3, dtype=np.complex128)

    for time_coord in range(temporal_extent):
        full_coords = _full_coord_from_spatial_coord(
            geometry, spatial_coords, time_direction, time_coord
        )
        site = geometry.index_from_coord(full_coords)
        wilson_line = wilson_line @ links[site, time_direction]

    return complex(np.trace(wilson_line) / 3.0)


def polyakov_loops(
    links: np.ndarray,
    geometry: LatticeGeometry,
    time_direction: int = -1,
) -> np.ndarray:
    """Compute normalized Polyakov loops for all spatial sites.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        time_direction: Direction used as Euclidean time.
    Outputs:
        Complex array over the spatial lattice.
    """
    if geometry.ndim < 2:
        raise ValueError("geometry must have at least two dimensions")
    time_direction = _normalize_direction(geometry, time_direction)
    spatial_shape = (
        geometry.shape[:time_direction] + geometry.shape[time_direction + 1 :]
    )
    loops = np.empty(spatial_shape, dtype=np.complex128)

    for spatial_coords in np.ndindex(spatial_shape):
        loops[spatial_coords] = polyakov_loop(
            links, geometry, spatial_coords, time_direction
        )

    return loops


def polyakov_loop_correlator_from_loops(loops: np.ndarray) -> np.ndarray:
    """Compute the translationally averaged Polyakov loop correlator.

    Inputs:
        loops: Complex Polyakov loop field over the spatial lattice.
    Outputs:
        Complex correlator C(r) = mean_x P(x) conj(P(x + r)).
    """
    loops = np.asarray(loops, dtype=np.complex128)
    if loops.size == 0:
        raise ValueError("loops must contain at least one spatial site")

    loop_fft = np.fft.fftn(loops)
    opposite_orientation = np.fft.ifftn(loop_fft * loop_fft.conj())
    return opposite_orientation.conj() / loops.size


def polyakov_loop_correlator(
    links: np.ndarray,
    geometry: LatticeGeometry,
    time_direction: int = -1,
) -> np.ndarray:
    """Compute the Polyakov loop correlator from gauge links.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        time_direction: Direction used as Euclidean time.
    Outputs:
        Complex correlator over spatial displacement vectors.
    """
    loops = polyakov_loops(links, geometry, time_direction)
    return polyakov_loop_correlator_from_loops(loops)


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
    forward_neighbors = geometry.forward_neighbors
    backward_neighbors = geometry.backward_neighbors
    site_plus_mu = forward_neighbors[site, mu]

    for nu in range(geometry.ndim):
        if nu == mu:
            continue

        site_plus_nu = forward_neighbors[site, nu]
        site_minus_nu = backward_neighbors[site, nu]
        site_plus_mu_minus_nu = backward_neighbors[site_plus_mu, nu]

        forward_staple = (
            links[site_plus_mu, nu]
            @ links[site_plus_nu, mu].conj().T
            @ links[site, nu].conj().T
        )
        backward_staple = (
            links[site_plus_mu_minus_nu, nu].conj().T
            @ links[site_minus_nu, mu].conj().T
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
