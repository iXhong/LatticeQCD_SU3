"""
README:
this is for a statistics checking.
请你利用src/中的实现，在相同的beta,体积和参数设置下，分别从cold start 和hot start开始，记录每个sweep的平均plaquette。
"""

from __future__ import annotations

import csv
# from datetime import datetime
from pathlib import Path
import sys

import numpy as np

SHAPE = (4, 4, 4, 4)
BETA = 5.7
STEP_SIZE = 0.4
SWEEPS = 200
SEED = 12345
ALGORITHM = "heatbath"

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


def update_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    step_size: float,
    rng: np.random.Generator,
):
    """Run one configured update sweep.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        step_size: Maximum SU(2) subgroup proposal radius for Metropolis.
        rng: NumPy random generator.
    Outputs:
        UpdateStats object from the configured sweep.
    """
    if ALGORITHM == "heatbath":
        return heatbath_sweep(links, geometry, beta, rng)
    return metropolis_sweep(links, geometry, beta, step_size, rng)


def write_history(
    writer: csv.DictWriter,
    start_name: str,
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    step_size: float,
    sweeps: int,
    rng: np.random.Generator,
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
        rng: NumPy random generator.
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
        stats = update_sweep(links, geometry, beta, step_size, rng)
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
        f"sweeps={SWEEPS}, algorithm={ALGORITHM}"
    )

    geometry = LatticeGeometry(SHAPE)
    cold_rng_seed, hot_rng_seed = np.random.SeedSequence(SEED).spawn(2)
    cold_rng = np.random.default_rng(cold_rng_seed)
    hot_rng = np.random.default_rng(hot_rng_seed)

    fieldnames = [
        "start",
        "sweep",
        "average_plaquette",
        "acceptance_rate",
        "accepted_links",
        "attempted_links",
    ]

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = ROOT / "results" / "thermalization" / f"thermal_check_{ALGORITHM}_{SWEEPS}sweeps.csv"

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
            cold_rng,
        )
        write_history(
            writer,
            "hot",
            hot_start(geometry, hot_rng),
            geometry,
            BETA,
            STEP_SIZE,
            SWEEPS,
            hot_rng,
        )

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
