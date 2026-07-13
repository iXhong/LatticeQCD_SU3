"""
README:
this is for a statistics checking.
请你利用src/中的实现，在相同的beta,体积和参数设置下，分别从cold start 和hot start开始，记录每个sweep的平均plaquette。

Usage:
    Set ALGORITHM = "heatbath" and BACKEND = "jit" to use the numba heatbath
    implementation. Set BACKEND = "numpy" to use the pure NumPy heatbath path.
    Metropolis runs ignore JIT acceleration and require BACKEND = "numpy".

    Run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/thermal_check.py
"""

from __future__ import annotations

import csv
# from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

SHAPE = (16, 16, 16, 6)
BETA = 5.7
STEP_SIZE = 0.4
SWEEPS = 300
SEED = 12345
ALGORITHM = "heatbath"
BACKEND = "jit"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import (  # noqa: E402
    LatticeGeometry,
    average_plaquette,
    cold_start,
    heatbath_sweep,
    hot_start,
    metropolis_sweep,
)


@dataclass
class SweepRunner:
    """Run configured update sweeps.

    Inputs:
        algorithm: Update algorithm name.
        backend: Heatbath backend name.
        seed: Seed for the first JIT sweep.
        rng: NumPy random generator for NumPy updates.
    Outputs:
        Stateful sweep runner.
    """

    algorithm: str
    backend: str
    seed: int
    rng: np.random.Generator
    jit_seeded: bool = False

    def sweep(
        self,
        links: np.ndarray,
        geometry: LatticeGeometry,
        beta: float,
        step_size: float,
    ):
        """Run one configured update sweep.

        Inputs:
            links: Gauge links U[site, direction].
            geometry: Lattice geometry object.
            beta: Wilson gauge coupling parameter.
            step_size: Maximum SU(2) subgroup proposal radius for Metropolis.
        Outputs:
            UpdateStats object from the configured sweep.
        """
        if self.algorithm == "metropolis":
            return metropolis_sweep(links, geometry, beta, step_size, self.rng)

        if self.backend == "numpy":
            return heatbath_sweep(links, geometry, beta, self.rng)

        if self.backend != "jit":
            raise ValueError("BACKEND must be 'jit' or 'numpy'")

        from lattice_su3.accelerated import heatbath_jit_sweep

        jit_seed = self.seed if not self.jit_seeded else None
        stats = heatbath_jit_sweep(links, geometry, beta, seed=jit_seed)
        self.jit_seeded = True
        return stats


def validate_parameters() -> None:
    """Validate script-level simulation parameters.

    Inputs:
        None.
    Outputs:
        None.
    """
    if SWEEPS < 0:
        raise ValueError("SWEEPS must be non-negative")
    if len(SHAPE) < 2:
        raise ValueError("SHAPE must contain at least two lattice directions")
    if ALGORITHM not in {"heatbath", "metropolis"}:
        raise ValueError("ALGORITHM must be 'heatbath' or 'metropolis'")
    if BACKEND not in {"jit", "numpy"}:
        raise ValueError("BACKEND must be 'jit' or 'numpy'")
    if ALGORITHM != "heatbath" and BACKEND == "jit":
        raise ValueError("BACKEND='jit' is only available for ALGORITHM='heatbath'")


def write_history(
    writer: csv.DictWriter,
    start_name: str,
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    step_size: float,
    sweeps: int,
    runner: SweepRunner,
) -> None:
    """Write one thermalization history to CSV.

    Inputs:
        writer: CSV dict writer.
        start_name: Name of the initial condition.
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        step_size: Maximum SU(2) subgroup proposal radius in [0, 1].
        sweeps: Number of full-lattice sweeps to run.
        runner: Stateful sweep runner.
    Outputs:
        None.
    """
    writer.writerow(
        {
            "start": start_name,
            "sweep": 0,
            "average_plaquette": average_plaquette(links, geometry),
            "acceptance_rate": "",
            "accepted_links": "",
            "attempted_links": "",
        }
    )

    for sweep in range(1, sweeps + 1):
        stats = runner.sweep(links, geometry, beta, step_size)
        if sweep % 10 == 0:
            plaq = average_plaquette(links, geometry)
            print(f"  [{start_name}] sweep {sweep}/{sweeps} — plaq={plaq:.6f}, acc={stats.acceptance_rate:.3f}")
        writer.writerow(
            {
                "start": start_name,
                "sweep": sweep,
                "average_plaquette": average_plaquette(links, geometry),
                "acceptance_rate": stats.acceptance_rate,
                "accepted_links": stats.accepted_links,
                "attempted_links": stats.attempted_links,
            }
        )


def main() -> None:
    """Run cold-start and hot-start thermalization checks.

    Inputs:
        None.
    Outputs:
        None.
    """
    validate_parameters()

    print(
        f"Lattice: {SHAPE}, beta={BETA}, step_size={STEP_SIZE}, "
        f"sweeps={SWEEPS}, algorithm={ALGORITHM}, backend={BACKEND}"
    )

    geometry = LatticeGeometry(SHAPE)
    cold_rng_seed, hot_rng_seed, cold_jit_seed, hot_jit_seed = np.random.SeedSequence(
        SEED
    ).spawn(4)
    cold_rng = np.random.default_rng(cold_rng_seed)
    hot_rng = np.random.default_rng(hot_rng_seed)
    cold_runner = SweepRunner(
        ALGORITHM,
        BACKEND,
        int(cold_jit_seed.generate_state(1)[0]),
        cold_rng,
    )
    hot_runner = SweepRunner(
        ALGORITHM,
        BACKEND,
        int(hot_jit_seed.generate_state(1)[0]),
        hot_rng,
    )

    fieldnames = [
        "start",
        "sweep",
        "average_plaquette",
        "acceptance_rate",
        "accepted_links",
        "attempted_links",
    ]

    shape_label = "x".join(str(length) for length in SHAPE)
    filename = (
        ROOT
        / "results"
        / "thermalization"
        / f"thermal_check_{ALGORITHM}_{BACKEND}_{shape_label}_{SWEEPS}sweeps.csv"
    )
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        write_history(
            writer,
            "cold",
            cold_start(geometry),
            geometry,
            BETA,
            STEP_SIZE,
            SWEEPS,
            cold_runner,
        )
        write_history(
            writer,
            "hot",
            hot_start(geometry, hot_rng),
            geometry,
            BETA,
            STEP_SIZE,
            SWEEPS,
            hot_runner,
        )

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
