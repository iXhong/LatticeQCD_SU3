"""
ReadMe:
this is the program for generating configurations of SU(3) pure gauge lattice qcd simulation. 
this file should contain parts listed below:
- lattice geometry
    周期边界和邻居索引

- SU(3) operations
    乘法、共轭转置、trace、随机 SU(3)

- gauge configuration
    U[x, mu]

- plaquette
    计算单个和平均 plaquette

- staple
    计算某条链周围六个 staple

- action
    Wilson gauge action

- update
    单链 Metropolis 更新
    完整 sweep

- thermalization
    hot/cold start、plaquette history

- configuration I/O
    保存和读取组态

"""

from dataclasses import dataclass

import numpy as np


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


class LatticeGeometry:
    """Periodic hypercubic lattice.

    Inputs:
        shape: Lattice size in each direction.
    Outputs:
        Object with volume, strides, and precomputed neighbor tables.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize lattice geometry.

        Inputs:
            shape: Lattice size in each direction.
        Outputs:
            None.
        """
        shape = tuple(shape)
        if len(shape) == 0:
            raise ValueError("shape must contain at least one lattice direction")
        if any(length <= 0 for length in shape):
            raise ValueError("all lattice lengths must be positive")

        self.shape = shape
        self.ndim = len(shape)
        self.volume = int(np.prod(shape))
        self.strides = self._compute_strides()
        self.forward_neighbors, self.backward_neighbors = self._compute_neighbors()

    def _compute_strides(self) -> np.ndarray:
        """Compute row-major strides for flat site indexing.

        Inputs:
            None.
        Outputs:
            Integer array with one stride per direction.
        """
        strides = np.empty(self.ndim, dtype=np.int64)
        stride = 1
        for mu in range(self.ndim - 1, -1, -1):
            strides[mu] = stride
            stride *= self.shape[mu]
        return strides

    def _compute_neighbors(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute periodic forward and backward neighbor tables.

        Inputs:
            None.
        Outputs:
            Two arrays indexed as [site, direction].
        """
        forward_neighbors = np.empty((self.volume, self.ndim), dtype=np.int64)
        backward_neighbors = np.empty((self.volume, self.ndim), dtype=np.int64)

        for site in range(self.volume):
            coords = self.coord_from_index(site)
            for mu, length in enumerate(self.shape):
                forward_coords = list(coords)
                backward_coords = list(coords)
                forward_coords[mu] = 0 if coords[mu] == length - 1 else coords[mu] + 1
                backward_coords[mu] = length - 1 if coords[mu] == 0 else coords[mu] - 1

                forward_neighbors[site, mu] = int(np.dot(forward_coords, self.strides))
                backward_neighbors[site, mu] = int(np.dot(backward_coords, self.strides))

        return forward_neighbors, backward_neighbors

    def index_from_coord(self, coords: tuple[int, ...]) -> int:
        """Convert coordinates to a flat site index.

        Inputs:
            coords: Coordinate tuple with one entry per direction.
        Outputs:
            Integer flat site index.
        """
        if len(coords) != len(self.shape):
            raise ValueError("coords must have one entry per lattice direction")
        if any(coord < 0 or coord >= length for coord, length in zip(coords, self.shape)):
            raise ValueError("coords are outside the lattice")

        return int(np.dot(coords, self.strides))

    def coord_from_index(self, site: int) -> tuple[int, ...]:
        """Convert a flat site index to coordinates.

        Inputs:
            site: Integer flat site index.
        Outputs:
            Coordinate tuple with one entry per direction.
        """
        if site < 0 or site >= self.volume:
            raise ValueError("site is outside the lattice")

        coords = []
        remainder = site
        for stride, length in zip(self.strides, self.shape):
            coord = remainder // stride
            coords.append(int(coord))
            remainder %= stride
            if coord >= length:
                raise ValueError("site is outside the lattice")
        return tuple(coords)

    def forward(self, site: int, mu: int) -> int:
        """Look up the forward periodic neighbor.

        Inputs:
            site: Flat site index.
            mu: Direction index.
        Outputs:
            Flat index of site + mu.
        """
        return int(self.forward_neighbors[site, mu])

    def backward(self, site: int, mu: int) -> int:
        """Look up the backward periodic neighbor.

        Inputs:
            site: Flat site index.
            mu: Direction index.
        Outputs:
            Flat index of site - mu.
        """
        return int(self.backward_neighbors[site, mu])


def random_su2(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random SU(2) matrix.

    Inputs:
        rng: Optional NumPy random generator.
    Outputs:
        Complex 2x2 SU(2) matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    a0, a1, a2, a3 = rng.normal(size=4)
    norm = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
    a0, a1, a2, a3 = a0 / norm, a1 / norm, a2 / norm, a3 / norm

    return np.array(
        [
            [a0 + 1j * a3, a2 + 1j * a1],
            [-a2 + 1j * a1, a0 - 1j * a3],
        ],
        dtype=np.complex128,
    )


def embed_su2(su2_matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    """Embed SU(2) into an SU(3) subgroup.

    Inputs:
        su2_matrix: Complex 2x2 SU(2) matrix.
        i: First SU(3) row/column index.
        j: Second SU(3) row/column index.
    Outputs:
        Complex 3x3 SU(3) matrix.
    """
    if su2_matrix.shape != (2, 2):
        raise ValueError("su2_matrix must have shape (2, 2)")
    if not (0 <= i < 3 and 0 <= j < 3 and i != j):
        raise ValueError("i and j must be distinct SU(3) indices in [0, 2]")

    su3_matrix = np.eye(3, dtype=np.complex128)
    su3_matrix[np.ix_([i, j], [i, j])] = su2_matrix
    return su3_matrix


def random_su3(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random SU(3) matrix.

    Inputs:
        rng: Optional NumPy random generator.
    Outputs:
        Complex 3x3 SU(3) matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    r_matrix = embed_su2(random_su2(rng), 0, 1)
    s_matrix = embed_su2(random_su2(rng), 0, 2)
    t_matrix = embed_su2(random_su2(rng), 1, 2)
    return r_matrix @ s_matrix @ t_matrix


def random_su2_near_identity(
    step_size: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a random SU(2) matrix near identity.

    Inputs:
        step_size: Maximum radius of the SU(2) vector part in [0, 1].
        rng: Optional NumPy random generator.
    Outputs:
        Complex 2x2 SU(2) proposal matrix.
    """
    if step_size < 0.0 or step_size > 1.0:
        raise ValueError("step_size must be in [0, 1]")
    if rng is None:
        rng = np.random.default_rng()
    if step_size == 0.0:
        return np.eye(2, dtype=np.complex128)

    direction = rng.normal(size=3)
    direction_norm = np.linalg.norm(direction)
    while direction_norm == 0.0:
        direction = rng.normal(size=3)
        direction_norm = np.linalg.norm(direction)

    radius = step_size * rng.random()
    a1, a2, a3 = radius * direction / direction_norm
    a0 = np.sqrt(1.0 - radius * radius)

    return np.array(
        [
            [a0 + 1j * a3, a2 + 1j * a1],
            [-a2 + 1j * a1, a0 - 1j * a3],
        ],
        dtype=np.complex128,
    )


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


def hot_start(geometry: LatticeGeometry, rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a random gauge configuration.

    Inputs:
        geometry: Lattice geometry object.
        rng: Optional NumPy random generator.
    Outputs:
        Link array U[site, direction] with random SU(3) matrices.
    """
    if rng is None:
        rng = np.random.default_rng()

    links = np.empty((geometry.volume, geometry.ndim, 3, 3), dtype=np.complex128)
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            links[site, mu] = random_su3(rng)
    return links


def cold_start(geometry: LatticeGeometry) -> np.ndarray:
    """Create an identity gauge configuration.

    Inputs:
        geometry: Lattice geometry object.
    Outputs:
        Link array U[site, direction] with identity matrices.
    """
    links = np.empty((geometry.volume, geometry.ndim, 3, 3), dtype=np.complex128)
    identity = np.eye(3, dtype=np.complex128)
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            links[site, mu] = identity
    return links


def dagger(matrix: np.ndarray) -> np.ndarray:
    """Return the conjugate transpose.

    Inputs:
        matrix: Complex matrix.
    Outputs:
        Conjugate-transposed matrix.
    """
    return matrix.conj().T


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


def is_su3(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """Check whether a matrix is numerically in SU(3).

    Inputs:
        matrix: Complex matrix to check.
        atol: Absolute tolerance for numerical comparisons.
    Outputs:
        True if matrix is 3x3, unitary, and has determinant one.
    """
    if matrix.shape != (3, 3):
        return False

    identity = np.eye(3, dtype=np.complex128)
    return np.allclose(matrix.conj().T @ matrix, identity, atol=atol) and np.allclose(
        np.linalg.det(matrix), 1.0 + 0.0j, atol=atol
    )
