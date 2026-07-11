"""Thermalization helpers for gauge configurations."""

import numpy as np

from lattice_su3.configuration import cold_start, hot_start
from lattice_su3.geometry import LatticeGeometry
from lattice_su3.update import UpdateStats, metropolis_sweep


def thermalize(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    step_size: float,
    sweeps: int,
    rng: np.random.Generator | None = None,
) -> list[UpdateStats]:
    """Run Metropolis thermalization sweeps in place.

    Inputs:
        links: Gauge links U[site, direction] to update in place.
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        step_size: Maximum SU(2) subgroup proposal radius in [0, 1].
        sweeps: Number of full-lattice sweeps to run.
        rng: Optional NumPy random generator.
    Outputs:
        List of UpdateStats objects, one per sweep.
    """
    if sweeps < 0:
        raise ValueError("sweeps must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    history = []
    for _ in range(sweeps):
        history.append(metropolis_sweep(links, geometry, beta, step_size, rng))
    return history


def thermalize_cold_start(
    geometry: LatticeGeometry,
    beta: float,
    step_size: float,
    sweeps: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[UpdateStats]]:
    """Create and thermalize a cold-start gauge configuration.

    Inputs:
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        step_size: Maximum SU(2) subgroup proposal radius in [0, 1].
        sweeps: Number of full-lattice sweeps to run.
        rng: Optional NumPy random generator.
    Outputs:
        Thermalized links and per-sweep UpdateStats history.
    """
    links = cold_start(geometry)
    history = thermalize(links, geometry, beta, step_size, sweeps, rng)
    return links, history


def thermalize_hot_start(
    geometry: LatticeGeometry,
    beta: float,
    step_size: float,
    sweeps: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[UpdateStats]]:
    """Create and thermalize a hot-start gauge configuration.

    Inputs:
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        step_size: Maximum SU(2) subgroup proposal radius in [0, 1].
        sweeps: Number of full-lattice sweeps to run.
        rng: Optional NumPy random generator.
    Outputs:
        Thermalized links and per-sweep UpdateStats history.
    """
    if rng is None:
        rng = np.random.default_rng()

    links = hot_start(geometry, rng)
    history = thermalize(links, geometry, beta, step_size, sweeps, rng)
    return links, history
