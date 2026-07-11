"""Benchmark average plaquette implementations."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

SHAPE = (4, 4, 4, 4)
BETA = 5.7
STEP_SIZE = 0.2
SEED = 12345
REPEATS = 10

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import (  # noqa: E402
    LatticeGeometry,
    average_plaquette,
    hot_start,
    metropolis_sweep,
    plaquette,
)


def average_plaquette_loop(links: np.ndarray, geometry: LatticeGeometry) -> float:
    """Compute average plaquette with explicit Python loops.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
    Outputs:
        Average real trace plaquette divided by three.
    """
    plaquette_sum = 0.0
    plaquette_count = 0
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            for nu in range(mu + 1, geometry.ndim):
                plaquette_sum += (
                    np.real(np.trace(plaquette(links, geometry, site, mu, nu))) / 3.0
                )
                plaquette_count += 1

    return float(plaquette_sum / plaquette_count)


def time_function(function, links: np.ndarray, geometry: LatticeGeometry) -> float:
    """Measure the average runtime of one plaquette function.

    Inputs:
        function: Callable average plaquette implementation.
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
    Outputs:
        Mean runtime in seconds.
    """
    start = perf_counter()
    for _ in range(REPEATS):
        function(links, geometry)
    return (perf_counter() - start) / REPEATS


def main() -> None:
    """Run the benchmark.

    Inputs:
        None.
    Outputs:
        None.
    """
    geometry = LatticeGeometry(SHAPE)
    rng = np.random.default_rng(SEED)
    links = hot_start(geometry, rng)
    metropolis_sweep(links, geometry, BETA, STEP_SIZE, rng)

    loop_value = average_plaquette_loop(links, geometry)
    vectorized_value = average_plaquette(links, geometry)
    loop_time = time_function(average_plaquette_loop, links, geometry)
    vectorized_time = time_function(average_plaquette, links, geometry)

    print(f"shape: {SHAPE}")
    print(f"volume: {geometry.volume}")
    print(f"repeats: {REPEATS}")
    print(f"loop average plaquette: {loop_value:.16f}")
    print(f"vectorized average plaquette: {vectorized_value:.16f}")
    print(f"absolute difference: {abs(loop_value - vectorized_value):.3e}")
    print(f"loop time: {loop_time:.6f} s")
    print(f"vectorized time: {vectorized_time:.6f} s")
    print(f"speedup: {loop_time / vectorized_time:.2f}x")


if __name__ == "__main__":
    main()
