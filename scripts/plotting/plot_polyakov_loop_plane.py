"""
Measure and plot raw spatially averaged Polyakov loops in the complex plane.

This script reads a run directory containing manifest.json and saved
configurations/*.npz files. For every raw configuration it computes the
spatial average of the normalized Polyakov loop, P = mean_x Tr prod_t U_t(x,t)/3,
then writes both the raw measurements and a complex-plane scatter plot.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plotting/plot_polyakov_loop_plane.py \
        results/runs/prod_static_potential_16x16x16x6_b57_hb_2or
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import LatticeGeometry, load_configuration  # noqa: E402

OUTPUT_DIRNAME = "polyakov_loop"
NPZ_FILENAME = "spatial_average_polyakov_loop.npz"
CSV_FILENAME = "spatial_average_polyakov_loop.csv"
PLOT_FILENAME = "spatial_average_polyakov_loop_plane.png"
PLOT_AXIS_LIMIT = 0.25


def load_manifest(path: Path) -> dict[str, object]:
    """Load run metadata from manifest.json.

    Inputs:
        path: Manifest JSON path.
    Outputs:
        Parsed manifest dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path) as handle:
        return json.load(handle)


def geometry_from_manifest(manifest: dict[str, object]) -> LatticeGeometry:
    """Build lattice geometry from run metadata.

    Inputs:
        manifest: Parsed run manifest.
    Outputs:
        Lattice geometry object.
    """
    if "shape" not in manifest:
        raise ValueError("manifest is missing required field: shape")
    return LatticeGeometry(tuple(int(length) for length in manifest["shape"]))


def configuration_paths(run_dir: Path) -> list[Path]:
    """List saved configurations from one run directory.

    Inputs:
        run_dir: Run directory containing configurations/.
    Outputs:
        Sorted saved configuration paths.
    """
    config_dir = run_dir / "configurations"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"configuration directory not found: {config_dir}")
    paths = sorted(config_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no configurations found in {config_dir}")
    return paths


def spatial_polyakov_field(
    links: np.ndarray,
    geometry: LatticeGeometry,
    time_direction: int,
) -> np.ndarray:
    """Compute normalized Polyakov loops over all spatial sites.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        time_direction: Direction used as Euclidean time.
    Outputs:
        Complex Polyakov loop field over the spatial lattice.
    """
    normalized_time_direction = time_direction % geometry.ndim
    if time_direction < -geometry.ndim or time_direction >= geometry.ndim:
        raise ValueError("time direction is outside the lattice dimensions")

    expected_shape = (geometry.volume, geometry.ndim, 3, 3)
    if links.shape != expected_shape:
        raise ValueError(f"links have shape {links.shape}, expected {expected_shape}")

    link_field = links.reshape(*geometry.shape, geometry.ndim, 3, 3)
    temporal_links = np.take(link_field, normalized_time_direction, axis=geometry.ndim)
    temporal_links = np.moveaxis(temporal_links, normalized_time_direction, -3)

    spatial_shape = temporal_links.shape[:-3]
    wilson_lines = np.broadcast_to(
        np.eye(3, dtype=np.complex128),
        (*spatial_shape, 3, 3),
    ).copy()
    for time_index in range(geometry.shape[normalized_time_direction]):
        wilson_lines = wilson_lines @ temporal_links[..., time_index, :, :]

    return np.trace(wilson_lines, axis1=-2, axis2=-1) / 3.0


def measure_spatial_averages(
    paths: list[Path],
    geometry: LatticeGeometry,
    time_direction: int,
) -> dict[str, np.ndarray]:
    """Measure raw spatially averaged Polyakov loops.

    Inputs:
        paths: Saved configuration NPZ paths.
        geometry: Lattice geometry object.
        time_direction: Direction used as Euclidean time.
    Outputs:
        Dictionary of chains, sweeps, filenames, and complex averages.
    """
    chains = []
    sweeps = []
    starts = []
    filenames = []
    values = []

    for index, path in enumerate(paths, start=1):
        links, metadata = load_configuration(path)
        loops = spatial_polyakov_field(links, geometry, time_direction)
        chains.append(int(metadata.get("chain", -1)))
        sweeps.append(int(metadata.get("sweep", -1)))
        starts.append(str(metadata.get("start", "")))
        filenames.append(path.name)
        values.append(complex(np.mean(loops)))
        print(f"[{index:4d}/{len(paths)}] {path.name}")

    return {
        "chains": np.asarray(chains, dtype=np.int64),
        "sweeps": np.asarray(sweeps, dtype=np.int64),
        "starts": np.asarray(starts, dtype=np.str_),
        "filenames": np.asarray(filenames, dtype=np.str_),
        "polyakov": np.asarray(values, dtype=np.complex128),
    }


def write_measurements(
    output_dir: Path,
    measurements: dict[str, np.ndarray],
    manifest: dict[str, object],
    time_direction: int,
) -> tuple[Path, Path]:
    """Write raw measurements as NPZ and CSV files.

    Inputs:
        output_dir: Directory for Polyakov-loop analysis outputs.
        measurements: Raw measurement arrays.
        manifest: Parsed run manifest.
        time_direction: Direction used as Euclidean time.
    Outputs:
        NPZ path and CSV path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / NPZ_FILENAME
    csv_path = output_dir / CSV_FILENAME
    polyakov = measurements["polyakov"]

    np.savez(
        npz_path,
        **measurements,
        real=polyakov.real,
        imag=polyakov.imag,
        abs=np.abs(polyakov),
        phase=np.angle(polyakov),
        time_direction=np.asarray(time_direction, dtype=np.int64),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )

    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chain", "start", "sweep", "filename", "real", "imag", "abs", "phase"])
        for chain, start, sweep, filename, value in zip(
            measurements["chains"],
            measurements["starts"],
            measurements["sweeps"],
            measurements["filenames"],
            polyakov,
        ):
            writer.writerow(
                [
                    int(chain),
                    str(start),
                    int(sweep),
                    str(filename),
                    f"{value.real:.16e}",
                    f"{value.imag:.16e}",
                    f"{abs(value):.16e}",
                    f"{np.angle(value):.16e}",
                ]
            )

    return npz_path, csv_path


def plot_complex_plane(
    output_dir: Path,
    measurements: dict[str, np.ndarray],
    run_name: str,
) -> Path:
    """Plot spatially averaged Polyakov loops on the complex plane.

    Inputs:
        output_dir: Directory for Polyakov-loop analysis outputs.
        measurements: Raw measurement arrays.
        run_name: Run name for the plot title.
    Outputs:
        Saved PNG path.
    """
    plot_path = output_dir / PLOT_FILENAME
    polyakov = measurements["polyakov"]
    chains = measurements["chains"]
    max_radius = PLOT_AXIS_LIMIT

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for chain in np.unique(chains):
        mask = chains == chain
        ax.scatter(
            polyakov.real[mask],
            polyakov.imag[mask],
            s=26,
            alpha=0.72,
            label=f"chain {int(chain)}",
        )

    angles = [0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0]
    for angle in angles:
        ax.plot(
            [0.0, max_radius * np.cos(angle)],
            [0.0, max_radius * np.sin(angle)],
            color="0.45",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )

    ax.axhline(0.0, color="0.25", linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color="0.25", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_xlabel("Re <P>")
    ax.set_ylabel("Im <P>")
    ax.set_title(f"Raw spatially averaged Polyakov loop\n{run_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Inputs:
        None.
    Outputs:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory under results/runs")
    parser.add_argument(
        "--time-direction",
        type=int,
        default=-1,
        help="Euclidean time direction, default: -1",
    )
    return parser.parse_args()


def main() -> None:
    """Run the raw Polyakov-loop measurement and plotting workflow.

    Inputs:
        None.
    Outputs:
        None.
    """
    args = parse_args()
    run_dir = args.run_dir
    manifest = load_manifest(run_dir / "manifest.json")
    geometry = geometry_from_manifest(manifest)
    paths = configuration_paths(run_dir)
    output_dir = run_dir / OUTPUT_DIRNAME

    measurements = measure_spatial_averages(paths, geometry, args.time_direction)
    npz_path, csv_path = write_measurements(
        output_dir,
        measurements,
        manifest,
        args.time_direction,
    )
    plot_path = plot_complex_plane(
        output_dir,
        measurements,
        str(manifest.get("run_name", run_dir.name)),
    )

    polyakov = measurements["polyakov"]
    print(f"Measured configurations: {len(polyakov)}")
    print(f"Mean raw <P>: {np.mean(polyakov):.8e}")
    print(f"Mean raw |<P>|: {np.mean(np.abs(polyakov)):.8e}")
    print(f"Max raw |<P>|: {np.max(np.abs(polyakov)):.8e}")
    print(f"Wrote: {npz_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
