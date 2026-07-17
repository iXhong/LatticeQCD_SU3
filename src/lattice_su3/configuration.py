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


def load_start(
    path: Path | str,
    geometry: LatticeGeometry,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and validate a gauge configuration for use as a chain start.

    Inputs:
        path: Input NPZ configuration path.
        geometry: Expected lattice geometry.
    Outputs:
        Writable complex128 gauge links and source metadata.
    """
    links, metadata = load_configuration(path)
    expected_shape = (geometry.volume, geometry.ndim, 3, 3)
    if links.shape != expected_shape:
        raise ValueError(
            f"loaded links have shape {links.shape}, expected {expected_shape} "
            f"for geometry shape {geometry.shape}"
        )
    if not np.issubdtype(links.dtype, np.complexfloating):
        raise ValueError("loaded links must have a complex floating-point dtype")
    if not np.all(np.isfinite(links)):
        raise ValueError("loaded links contain non-finite values")

    source_shape = metadata.get("shape")
    if (
        source_shape is not None
        and tuple(int(length) for length in source_shape) != geometry.shape
    ):
        raise ValueError(
            f"loaded metadata shape {tuple(source_shape)} does not match "
            f"geometry shape {geometry.shape}"
        )

    return np.array(links, dtype=np.complex128, copy=True), metadata


def latest_configuration_path(
    config_dir: Path | str,
    chain: int | None = None,
) -> Path:
    """Find the saved configuration with the greatest sweep number.

    Inputs:
        config_dir: Directory containing saved NPZ configurations.
        chain: Optional chain index used to filter configurations.
    Outputs:
        Unique configuration path with the greatest metadata sweep.
    """
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"configuration directory not found: {config_dir}")

    candidates: list[tuple[int, Path]] = []
    for path in sorted(config_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "sweep" not in data.files:
                raise ValueError(f"{path} is missing required metadata field: sweep")
            if chain is not None:
                if "chain" not in data.files or int(data["chain"].item()) != chain:
                    continue
            sweep = int(data["sweep"].item())
        if sweep < 0:
            raise ValueError(f"{path} has a negative sweep number: {sweep}")
        candidates.append((sweep, path))

    if not candidates:
        chain_label = "" if chain is None else f" for chain {chain}"
        raise FileNotFoundError(
            f"no saved configurations found in {config_dir}{chain_label}"
        )

    latest_sweep = max(sweep for sweep, _ in candidates)
    latest_paths = [path for sweep, path in candidates if sweep == latest_sweep]
    if len(latest_paths) != 1:
        paths = ", ".join(str(path) for path in latest_paths)
        raise ValueError(
            f"multiple configurations have latest sweep {latest_sweep}: {paths}"
        )
    return latest_paths[0]
