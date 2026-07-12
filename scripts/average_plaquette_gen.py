"""
README:
Generate average-plaquette time series for autocorrelation analysis.

The generated CSV is saved under results/autocorrelation/ and can be consumed by
scripts/auto_correlation.py.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

SHAPE = (4, 4, 4, 4)
BETA = 5.7
STEP_SIZE = 0.4
ALGORITHM = "heatbath"
START = "hot"
THERMALIZATION_SWEEPS = 50
MEASUREMENT_SWEEPS = 250
MEASURE_EVERY = 1
SEED = 12345

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
    """Validate script-level series generation parameters.

    Inputs:
        None.
    Outputs:
        None.
    """
    if len(SHAPE) < 2:
        raise ValueError("SHAPE must contain at least two lattice directions")
    if ALGORITHM not in {"heatbath", "metropolis"}:
        raise ValueError("ALGORITHM must be 'heatbath' or 'metropolis'")
    if START not in {"cold", "hot"}:
        raise ValueError("START must be 'cold' or 'hot'")
    if THERMALIZATION_SWEEPS < 0:
        raise ValueError("THERMALIZATION_SWEEPS must be non-negative")
    if MEASUREMENT_SWEEPS <= 1:
        raise ValueError("MEASUREMENT_SWEEPS must be greater than one")
    if MEASURE_EVERY <= 0:
        raise ValueError("MEASURE_EVERY must be positive")


def update_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    rng: np.random.Generator,
) -> None:
    """Run one configured update sweep in place.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        rng: NumPy random generator.
    Outputs:
        None.
    """
    if ALGORITHM == "heatbath":
        heatbath_sweep(links, geometry, BETA, rng)
    else:
        metropolis_sweep(links, geometry, BETA, STEP_SIZE, rng)


def initial_links(geometry: LatticeGeometry, rng: np.random.Generator) -> np.ndarray:
    """Create the configured starting gauge configuration.

    Inputs:
        geometry: Lattice geometry object.
        rng: NumPy random generator.
    Outputs:
        Gauge links U[site, direction].
    """
    if START == "cold":
        return cold_start(geometry)
    return hot_start(geometry, rng)


def collect_plaquette_series(
    geometry: LatticeGeometry,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a plaquette time series after thermalization.

    Inputs:
        geometry: Lattice geometry object.
        rng: NumPy random generator.
    Outputs:
        Measurement indices and plaquette values.
    """
    links = initial_links(geometry, rng)

    for sweep in range(1, THERMALIZATION_SWEEPS + 1):
        update_sweep(links, geometry, rng)
        if sweep % 50 == 0:
            print(f"  thermalization {sweep}/{THERMALIZATION_SWEEPS}")

    measurements = []
    plaquettes = []
    for sweep in range(1, MEASUREMENT_SWEEPS + 1):
        update_sweep(links, geometry, rng)
        if sweep % MEASURE_EVERY == 0:
            measurements.append(sweep)
            plaquettes.append(average_plaquette(links, geometry))
        if sweep % 100 == 0:
            print(f"  measurement {sweep}/{MEASUREMENT_SWEEPS}")

    return (
        np.asarray(measurements, dtype=np.int64),
        np.asarray(plaquettes, dtype=np.float64),
    )


def output_path() -> Path:
    """Build the output path for this run.

    Inputs:
        None.
    Outputs:
        CSV path for the plaquette time series.
    """
    shape_label = "x".join(str(length) for length in SHAPE)
    return (
        ROOT
        / "results"
        / "autocorrelation"
        / f"{ALGORITHM}_{START}_{shape_label}_beta{BETA}_n{MEASUREMENT_SWEEPS}_series.csv"
    )


def write_series_csv(
    path: Path,
    measurements: np.ndarray,
    plaquettes: np.ndarray,
) -> None:
    """Write the measured plaquette time series to CSV.

    Inputs:
        path: Output CSV path.
        measurements: Sweep numbers where measurements were made.
        plaquettes: Average plaquette measurements.
    Outputs:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "measurement",
                "sweep",
                "average_plaquette",
            ],
        )
        writer.writeheader()
        for measurement, (sweep, plaquette) in enumerate(
            zip(measurements, plaquettes, strict=True)
        ):
            writer.writerow(
                {
                    "measurement": measurement,
                    "sweep": int(sweep),
                    "average_plaquette": plaquette,
                }
            )


def main() -> None:
    """Generate and save average plaquette measurements.

    Inputs:
        None.
    Outputs:
        None.
    """
    validate_parameters()
    geometry = LatticeGeometry(SHAPE)
    rng = np.random.default_rng(SEED)

    print(
        f"Lattice: {SHAPE}, beta={BETA}, algorithm={ALGORITHM}, start={START}, "
        f"thermalization={THERMALIZATION_SWEEPS}, measurements={MEASUREMENT_SWEEPS}"
    )
    measurements, plaquettes = collect_plaquette_series(geometry, rng)
    path = output_path()
    write_series_csv(path, measurements, plaquettes)

    print(f"mean plaquette: {np.mean(plaquettes):.8f}")
    print(f"std plaquette: {np.std(plaquettes, ddof=1):.8f}")
    print(f"Series saved to {path}")


if __name__ == "__main__":
    main()
