"""
Measure vector Polyakov loop correlators from saved gauge configurations.

This script reads an existing run directory with manifest.json and
configurations/*.npz files produced by scripts/run_chain.py. It computes the
full translationally averaged Polyakov loop correlator C(r_vec) for each saved
configuration after the configured thermalization cutoff and writes the raw
vector correlator ensemble under the run correlators/ directory. Radial binning
and static-potential extraction are intentionally left for later analysis steps
in the same directory.

Usage:
    Edit RUN_NAME and THERMALIZATION_SWEEPS below, then run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/measure_polyakov_correlators.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

RUN_NAME = ""
CONFIG_GLOB = "*.npz"
TIME_DIRECTION = -1
THERMALIZATION_SWEEPS = 400
OUTPUT_DIRNAME = "correlators"
OUTPUT_FILENAME = "polyakov_vector_correlators.npz"
OVERWRITE = True

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import (  # noqa: E402
    LatticeGeometry,
    load_configuration,
    polyakov_loop_correlator,
)


def run_directory(run_name: str = RUN_NAME) -> Path:
    """Build the configured run directory path.

    Inputs:
        run_name: Run directory name under results/runs.
    Outputs:
        Path to results/runs/<run_name>.
    """
    if not run_name:
        raise ValueError("run name must be non-empty")
    return ROOT / "results" / "runs" / run_name


def output_correlator_path(run_dir: Path) -> Path:
    """Build the output path for vector correlator measurements.

    Inputs:
        run_dir: Run directory under results/runs.
    Outputs:
        NPZ output path under the run correlators directory.
    """
    return run_dir / OUTPUT_DIRNAME / OUTPUT_FILENAME


def load_manifest(path: Path) -> dict[str, object]:
    """Load run metadata from manifest.json.

    Inputs:
        path: Manifest JSON path.
    Outputs:
        Parsed manifest dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path) as f:
        return json.load(f)


def configuration_paths(config_dir: Path, pattern: str = CONFIG_GLOB) -> list[Path]:
    """List saved gauge configurations in deterministic order.

    Inputs:
        config_dir: Directory containing saved NPZ configurations.
        pattern: Glob pattern for configuration files.
    Outputs:
        Sorted list of configuration paths.
    """
    if not config_dir.exists():
        raise FileNotFoundError(f"configuration directory not found: {config_dir}")
    paths = sorted(config_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no configurations matching {pattern!r} in {config_dir}")
    return paths


def configuration_sweep(path: Path) -> int:
    """Read the saved sweep number from one configuration file.

    Inputs:
        path: Saved configuration NPZ path.
    Outputs:
        Full-lattice sweep number stored in configuration metadata.
    """
    with np.load(path, allow_pickle=False) as data:
        if "sweep" not in data.files:
            raise ValueError(f"{path} is missing required metadata field: sweep")
        return int(data["sweep"].item())


def filter_thermalized_paths(
    paths: list[Path],
    thermalization_sweeps: int,
) -> list[Path]:
    """Discard configurations at or before the thermalization cutoff.

    Inputs:
        paths: Saved configuration NPZ paths.
        thermalization_sweeps: Discard configurations with sweep <= this value.
    Outputs:
        Configuration paths after the cutoff.
    """
    if thermalization_sweeps < 0:
        raise ValueError("THERMALIZATION_SWEEPS must be non-negative")

    filtered_paths = [
        path for path in paths if configuration_sweep(path) > thermalization_sweeps
    ]
    if not filtered_paths:
        raise ValueError(
            "no configurations remain after thermalization cutoff "
            f"{thermalization_sweeps}"
        )
    return filtered_paths


def geometry_from_manifest(manifest: dict[str, object]) -> LatticeGeometry:
    """Reconstruct lattice geometry from run metadata.

    Inputs:
        manifest: Parsed run manifest.
    Outputs:
        Lattice geometry object.
    """
    if "shape" not in manifest:
        raise ValueError("manifest is missing required field: shape")
    return LatticeGeometry(tuple(int(length) for length in manifest["shape"]))


def measure_configuration(
    path: Path,
    geometry: LatticeGeometry,
    time_direction: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute one vector Polyakov correlator from a saved configuration.

    Inputs:
        path: Saved configuration NPZ path.
        geometry: Lattice geometry object for this run.
        time_direction: Direction used as Euclidean time.
    Outputs:
        Complex correlator array and loaded configuration metadata.
    """
    links, metadata = load_configuration(path)
    expected_shape = (geometry.volume, geometry.ndim, 3, 3)
    if links.shape != expected_shape:
        raise ValueError(
            f"{path} has links shape {links.shape}, expected {expected_shape}"
        )
    return polyakov_loop_correlator(links, geometry, time_direction), metadata


def measure_correlator_ensemble(
    paths: list[Path],
    geometry: LatticeGeometry,
    time_direction: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure vector Polyakov correlators for an ensemble.

    Inputs:
        paths: Saved configuration NPZ paths.
        geometry: Lattice geometry object for this run.
        time_direction: Direction used as Euclidean time.
    Outputs:
        Correlator stack, sweeps, chains, starts, and filenames.
    """
    correlators = []
    sweeps = []
    chains = []
    starts = []
    filenames = []

    for path in paths:
        correlator, metadata = measure_configuration(path, geometry, time_direction)
        correlators.append(correlator)
        sweeps.append(int(metadata.get("sweep", -1)))
        chains.append(int(metadata.get("chain", -1)))
        starts.append(str(metadata.get("start", "")))
        filenames.append(path.name)

    return (
        np.asarray(correlators, dtype=np.complex64),
        np.asarray(sweeps, dtype=np.int64),
        np.asarray(chains, dtype=np.int64),
        np.asarray(starts, dtype=np.str_),
        np.asarray(filenames, dtype=np.str_),
    )


def write_correlators(
    path: Path,
    correlators: np.ndarray,
    sweeps: np.ndarray,
    chains: np.ndarray,
    starts: np.ndarray,
    filenames: np.ndarray,
    manifest: dict[str, object],
    time_direction: int,
    thermalization_sweeps: int,
) -> None:
    """Write measured vector correlators and metadata to NPZ.

    Inputs:
        path: Output NPZ path.
        correlators: Complex correlator stack with one entry per configuration.
        sweeps: Sweep number for each measured configuration.
        chains: Chain index for each measured configuration.
        starts: Start label for each measured configuration.
        filenames: Source filename for each measured configuration.
        manifest: Parsed run manifest.
        time_direction: Direction used as Euclidean time.
        thermalization_sweeps: Thermalization cutoff applied before measuring.
    Outputs:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        correlators=correlators,
        sweeps=sweeps,
        chains=chains,
        starts=starts,
        filenames=filenames,
        shape=np.asarray(manifest["shape"], dtype=np.int64),
        beta=np.asarray(manifest.get("beta", np.nan), dtype=np.float64),
        time_direction=np.asarray(time_direction, dtype=np.int64),
        thermalization_sweeps=np.asarray(thermalization_sweeps, dtype=np.int64),
        run_name=np.asarray(manifest.get("run_name", RUN_NAME)),
    )


def main() -> None:
    """Measure vector Polyakov correlators for the configured run.

    Inputs:
        None.
    Outputs:
        None.
    """
    run_dir = run_directory()
    output_path = output_correlator_path(run_dir)
    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    manifest = load_manifest(run_dir / "manifest.json")
    geometry = geometry_from_manifest(manifest)
    paths = configuration_paths(run_dir / "configurations")
    measured_paths = filter_thermalized_paths(paths, THERMALIZATION_SWEEPS)
    print(f"Run: {run_dir}")
    print(f"Configurations: {len(paths)} total, {len(measured_paths)} after cutoff")
    print(f"Thermalization cutoff: discard sweeps <= {THERMALIZATION_SWEEPS}")
    print(f"Lattice shape: {geometry.shape}, time_direction={TIME_DIRECTION}")

    correlators, sweeps, chains, starts, filenames = measure_correlator_ensemble(
        measured_paths,
        geometry,
        TIME_DIRECTION,
    )
    write_correlators(
        output_path,
        correlators,
        sweeps,
        chains,
        starts,
        filenames,
        manifest,
        TIME_DIRECTION,
        THERMALIZATION_SWEEPS,
    )

    print(f"Correlator shape: {correlators.shape}")
    print(f"Sweeps: {sweeps.min()}..{sweeps.max()}")
    print(f"Saved vector correlators to {output_path}")


if __name__ == "__main__":
    main()
