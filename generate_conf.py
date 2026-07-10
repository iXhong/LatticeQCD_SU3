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
class LatticeGeometry:
    """Periodic hypercubic lattice with precomputed neighbor index tables."""

    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if len(shape) == 0:
            raise ValueError("shape must contain at least one lattice direction")
        if any(length <= 0 for length in shape):
            raise ValueError("all lattice lengths must be positive")

        ndim = len(shape)
        volume = int(np.prod(shape))
        strides = self._compute_strides(shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "strides", strides)

        forward_neighbors, backward_neighbors = self._compute_neighbors(strides, volume)
        object.__setattr__(self, "forward_neighbors", forward_neighbors)
        object.__setattr__(self, "backward_neighbors", backward_neighbors)

    @staticmethod
    def _compute_strides(shape: tuple[int, ...]) -> np.ndarray:
        strides = np.empty(len(shape), dtype=np.int64)
        stride = 1
        for mu in range(len(shape) - 1, -1, -1):
            strides[mu] = stride
            stride *= shape[mu]
        return strides

    def _compute_neighbors(
        self, strides: np.ndarray, volume: int
    ) -> tuple[np.ndarray, np.ndarray]:
        forward_neighbors = np.empty((volume, len(self.shape)), dtype=np.int64)
        backward_neighbors = np.empty((volume, len(self.shape)), dtype=np.int64)

        for site in range(volume):
            coords = self.coord_from_index(site)
            for mu, length in enumerate(self.shape):
                forward_coords = list(coords)
                backward_coords = list(coords)
                forward_coords[mu] = 0 if coords[mu] == length - 1 else coords[mu] + 1
                backward_coords[mu] = length - 1 if coords[mu] == 0 else coords[mu] - 1

                forward_neighbors[site, mu] = int(np.dot(forward_coords, strides))
                backward_neighbors[site, mu] = int(np.dot(backward_coords, strides))

        return forward_neighbors, backward_neighbors

    def index_from_coord(self, coords: tuple[int, ...]) -> int:
        """Convert lattice coordinates to the flattened site index."""
        if len(coords) != len(self.shape):
            raise ValueError("coords must have one entry per lattice direction")
        if any(coord < 0 or coord >= length for coord, length in zip(coords, self.shape)):
            raise ValueError("coords are outside the lattice")

        return int(np.dot(coords, self.strides))

    def coord_from_index(self, site: int) -> tuple[int, ...]:
        """Convert a flattened site index to lattice coordinates."""
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
        """Return the precomputed forward neighbor of site in direction mu."""
        return int(self.forward_neighbors[site, mu])

    def backward(self, site: int, mu: int) -> int:
        """Return the precomputed backward neighbor of site in direction mu."""
        return int(self.backward_neighbors[site, mu])


def random_su2(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a Haar-random SU(2) matrix from a unit quaternion."""
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
    """Embed a 2x2 SU(2) matrix in the (i, j) subgroup of SU(3)."""
    if su2_matrix.shape != (2, 2):
        raise ValueError("su2_matrix must have shape (2, 2)")
    if not (0 <= i < 3 and 0 <= j < 3 and i != j):
        raise ValueError("i and j must be distinct SU(3) indices in [0, 2]")

    su3_matrix = np.eye(3, dtype=np.complex128)
    su3_matrix[np.ix_([i, j], [i, j])] = su2_matrix
    return su3_matrix


def random_su3(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate an SU(3) matrix as X = R S T from embedded SU(2) matrices."""
    if rng is None:
        rng = np.random.default_rng()

    r_matrix = embed_su2(random_su2(rng), 0, 1)
    s_matrix = embed_su2(random_su2(rng), 0, 2)
    t_matrix = embed_su2(random_su2(rng), 1, 2)
    return r_matrix @ s_matrix @ t_matrix


def is_su3(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True when matrix is unitary with determinant one."""
    if matrix.shape != (3, 3):
        return False

    identity = np.eye(3, dtype=np.complex128)
    return np.allclose(matrix.conj().T @ matrix, identity, atol=atol) and np.allclose(
        np.linalg.det(matrix), 1.0 + 0.0j, atol=atol
    )
