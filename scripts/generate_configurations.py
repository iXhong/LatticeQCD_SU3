"""Generate autocorrelation-spaced gauge configurations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

SHAPE = (16, 16, 16, 6)
BETA = 5.7
START = "hot"
BACKEND = "jit"
THERMALIZATION_SWEEPS = 1000
PILOT_SWEEPS = 5000
MAX_AUTOCORR_LAG = 1000
CONFIGURATION_COUNT = 100
AUTOCORR_INTERVAL_FACTOR = 2.0
SEED = 12345
OVERWRITE = False

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import (  # noqa: E402
    LatticeGeometry,
    autocovariance,
    average_plaquette,
    choose_window,
    cold_start,
    heatbath_sweep,
    hot_start,
    integrated_autocorrelation,
    normalized_autocorrelation,
    save_configuration,
    suggested_interval,
)


@dataclass
class SweepRunner:
    """Run configured heatbath sweeps.

    Inputs:
        backend: Update backend name.
        seed: Seed for the first JIT sweep.
        rng: NumPy random generator for NumPy heatbath.
    Outputs:
        Stateful sweep runner.
    """

    backend: str
    seed: int
    rng: np.random.Generator
    jit_seeded: bool = False

    def sweep(self, links: np.ndarray, geometry: LatticeGeometry, beta: float) -> None:
        """Run one configured heatbath sweep.

        Inputs:
            links: Gauge links U[site, direction].
            geometry: Lattice geometry object.
            beta: Wilson gauge coupling parameter.
        Outputs:
            None.
        """
        if self.backend == "numpy":
            heatbath_sweep(links, geometry, beta, self.rng)
            return

        if self.backend != "jit":
            raise ValueError("BACKEND must be 'jit' or 'numpy'")

        from lattice_su3.accelerated import heatbath_jit_sweep

        jit_seed = self.seed if not self.jit_seeded else None
        heatbath_jit_sweep(links, geometry, beta, seed=jit_seed)
        self.jit_seeded = True


def validate_parameters() -> None:
    """Validate script-level generation parameters.

    Inputs:
        None.
    Outputs:
        None.
    """
    if len(SHAPE) < 2:
        raise ValueError("SHAPE must contain at least two lattice directions")
    if BETA < 0.0:
        raise ValueError("BETA must be non-negative")
    if START not in {"cold", "hot"}:
        raise ValueError("START must be 'cold' or 'hot'")
    if BACKEND not in {"jit", "numpy"}:
        raise ValueError("BACKEND must be 'jit' or 'numpy'")
    if THERMALIZATION_SWEEPS < 0:
        raise ValueError("THERMALIZATION_SWEEPS must be non-negative")
    if PILOT_SWEEPS <= 1:
        raise ValueError("PILOT_SWEEPS must be greater than one")
    if MAX_AUTOCORR_LAG <= 0:
        raise ValueError("MAX_AUTOCORR_LAG must be positive")
    if CONFIGURATION_COUNT <= 0:
        raise ValueError("CONFIGURATION_COUNT must be positive")
    if AUTOCORR_INTERVAL_FACTOR <= 0.0:
        raise ValueError("AUTOCORR_INTERVAL_FACTOR must be positive")


def shape_label(shape: tuple[int, ...]) -> str:
    """Format a lattice shape for filenames.

    Inputs:
        shape: Lattice size in each direction.
    Outputs:
        Shape label with dimensions joined by x.
    """
    return "x".join(str(length) for length in shape)


def output_directory() -> Path:
    """Build the output directory for this generation run.

    Inputs:
        None.
    Outputs:
        Directory path for generated configurations and metadata.
    """
    label = shape_label(SHAPE)
    return ROOT / "results" / "configurations" / (
        f"heatbath_{BACKEND}_{label}_beta{BETA}_seed{SEED}"
    )


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


def write_plaquette_series_csv(
    path: Path,
    sweeps: list[int],
    plaquettes: list[float],
) -> None:
    """Write pilot plaquette measurements to CSV.

    Inputs:
        path: Output CSV path.
        sweeps: Absolute sweep numbers.
        plaquettes: Average plaquette values.
    Outputs:
        None.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["measurement", "sweep", "average_plaquette"])
        writer.writeheader()
        for measurement, (sweep, plaquette) in enumerate(
            zip(sweeps, plaquettes, strict=True)
        ):
            writer.writerow(
                {
                    "measurement": measurement,
                    "sweep": sweep,
                    "average_plaquette": plaquette,
                }
            )


def write_autocorrelation_csv(
    path: Path,
    covariance: np.ndarray,
    gamma: np.ndarray,
    tau_int: np.ndarray,
) -> None:
    """Write pilot autocorrelation analysis to CSV.

    Inputs:
        path: Output CSV path.
        covariance: Autocovariance values.
        gamma: Normalized autocorrelation values.
        tau_int: Running integrated autocorrelation time.
    Outputs:
        None.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lag", "autocovariance", "gamma", "tau_int_running"],
        )
        writer.writeheader()
        for lag in range(len(gamma)):
            writer.writerow(
                {
                    "lag": lag,
                    "autocovariance": covariance[lag],
                    "gamma": gamma[lag],
                    "tau_int_running": tau_int[lag],
                }
            )


def autocorrelation_interval(
    plaquettes: np.ndarray,
) -> tuple[int, float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate autocorrelation and configuration save interval.

    Inputs:
        plaquettes: Pilot average plaquette time series.
    Outputs:
        Interval, selected tau_int, window, covariance, gamma, running tau_int.
    """
    max_lag = min(MAX_AUTOCORR_LAG, len(plaquettes) - 1)
    covariance = autocovariance(plaquettes, max_lag)
    gamma = normalized_autocorrelation(covariance)
    tau_int = integrated_autocorrelation(gamma)
    window = choose_window(gamma)
    selected_tau = float(tau_int[window])
    interval = suggested_interval(selected_tau, AUTOCORR_INTERVAL_FACTOR)
    return interval, selected_tau, window, covariance, gamma, tau_int


def manifest_row(
    filename: str,
    config_index: int,
    sweep: int,
    plaquette: float,
    tau_int: float,
    sweeps_between_configs: int,
) -> dict[str, object]:
    """Build one manifest row for a saved configuration.

    Inputs:
        filename: Configuration filename.
        config_index: Saved configuration index.
        sweep: Absolute sweep number.
        plaquette: Average plaquette value at save time.
        tau_int: Selected integrated autocorrelation time.
        sweeps_between_configs: Saved configuration spacing.
    Outputs:
        Manifest row dictionary.
    """
    return {
        "filename": filename,
        "config_index": config_index,
        "sweep": sweep,
        "average_plaquette": plaquette,
        "backend": BACKEND,
        "beta": BETA,
        "shape": shape_label(SHAPE),
        "seed": SEED,
        "tau_int": tau_int,
        "sweeps_between_configs": sweeps_between_configs,
    }


def configuration_metadata(
    config_index: int,
    sweep: int,
    plaquette: float,
    tau_int: float,
    sweeps_between_configs: int,
) -> dict[str, object]:
    """Build metadata for one saved configuration.

    Inputs:
        config_index: Saved configuration index.
        sweep: Absolute sweep number.
        plaquette: Average plaquette value at save time.
        tau_int: Selected integrated autocorrelation time.
        sweeps_between_configs: Saved configuration spacing.
    Outputs:
        Metadata dictionary.
    """
    return {
        "shape": SHAPE,
        "beta": BETA,
        "sweep": sweep,
        "config_index": config_index,
        "backend": BACKEND,
        "seed": SEED,
        "average_plaquette": plaquette,
        "tau_int": tau_int,
        "sweeps_between_configs": sweeps_between_configs,
    }


def main() -> None:
    """Generate autocorrelation-spaced gauge configurations.

    Inputs:
        None.
    Outputs:
        None.
    """
    validate_parameters()
    out_dir = output_directory()
    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists() and not OVERWRITE:
        raise FileExistsError(f"refusing to overwrite existing run: {manifest_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = LatticeGeometry(SHAPE)
    rng = np.random.default_rng(SEED)
    runner = SweepRunner(BACKEND, SEED, rng)
    links = initial_links(geometry, rng)
    sweep_count = 0
    start_time = perf_counter()

    print(
        f"Lattice: {SHAPE}, beta={BETA}, backend={BACKEND}, start={START}, "
        f"thermalization={THERMALIZATION_SWEEPS}, pilot={PILOT_SWEEPS}"
    )

    for sweep in range(1, THERMALIZATION_SWEEPS + 1):
        runner.sweep(links, geometry, BETA)
        sweep_count += 1
        if sweep % 100 == 0:
            print(f"  thermalization {sweep}/{THERMALIZATION_SWEEPS}")

    pilot_sweeps = []
    pilot_plaquettes = []
    for sweep in range(1, PILOT_SWEEPS + 1):
        runner.sweep(links, geometry, BETA)
        sweep_count += 1
        plaquette = average_plaquette(links, geometry)
        pilot_sweeps.append(sweep_count)
        pilot_plaquettes.append(plaquette)
        if sweep % 500 == 0:
            print(f"  pilot {sweep}/{PILOT_SWEEPS}, plaquette={plaquette:.8f}")

    series_path = out_dir / "pilot_plaquette_series.csv"
    write_plaquette_series_csv(series_path, pilot_sweeps, pilot_plaquettes)

    interval, tau_int, window, covariance, gamma, running_tau = autocorrelation_interval(
        np.asarray(pilot_plaquettes, dtype=np.float64)
    )
    autocorr_path = out_dir / "pilot_autocorrelation.csv"
    write_autocorrelation_csv(autocorr_path, covariance, gamma, running_tau)
    print(
        f"  tau_int={tau_int:.4f}, window={window}, "
        f"sweeps_between_configs={interval}"
    )

    fieldnames = list(
        manifest_row("", 0, 0, 0.0, tau_int, interval).keys()
    )
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config_index in range(CONFIGURATION_COUNT):
            for _ in range(interval):
                runner.sweep(links, geometry, BETA)
                sweep_count += 1

            plaquette = average_plaquette(links, geometry)
            filename = f"config_{config_index:06d}.npz"
            metadata = configuration_metadata(
                config_index, sweep_count, plaquette, tau_int, interval
            )
            save_configuration(out_dir / filename, links, metadata)
            writer.writerow(
                manifest_row(filename, config_index, sweep_count, plaquette, tau_int, interval)
            )
            print(
                f"  saved {filename}: sweep={sweep_count}, "
                f"plaquette={plaquette:.8f}"
            )

    elapsed = perf_counter() - start_time
    print(f"Pilot plaquette series saved to {series_path}")
    print(f"Pilot autocorrelation saved to {autocorr_path}")
    print(f"Manifest saved to {manifest_path}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
