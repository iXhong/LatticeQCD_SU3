"""Gauge configuration initializers and file I/O."""

from pathlib import Path

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


def save_configuration(
    path: Path | str,
    links: np.ndarray,
    metadata: dict[str, object],
) -> None:
    """Save one gauge configuration and metadata to an NPZ file.

    Inputs:
        path: Output NPZ path.
        links: Gauge links U[site, direction].
        metadata: Scalar or tuple metadata values to store with the links.
    Outputs:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"links": links}
    for key, value in metadata.items():
        arrays[key] = np.asarray(value)
    np.savez(path, **arrays)


def load_configuration(path: Path | str) -> tuple[np.ndarray, dict[str, object]]:
    """Load one gauge configuration and metadata from an NPZ file.

    Inputs:
        path: Input NPZ path.
    Outputs:
        Gauge links and metadata dictionary.
    """
    with np.load(path, allow_pickle=False) as data:
        links = np.asarray(data["links"])
        metadata = {}
        for key in data.files:
            if key == "links":
                continue
            value = data[key]
            metadata[key] = value.item() if value.shape == () else value.tolist()
    return links, metadata
