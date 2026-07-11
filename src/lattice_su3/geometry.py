"""Lattice geometry helpers."""

import numpy as np


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
