"""Gauge configuration initializers."""

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.group import random_su3


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
