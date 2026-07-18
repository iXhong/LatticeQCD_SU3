"""
Bin measured Polyakov correlators by periodic spatial distance.

This script reads ``polyakov_vector_correlators.npz`` written by
``scripts/measure_polyakov_correlators.py``. It writes per-configuration radial
and axis-only correlators beside the input file without performing resampling or
potential extraction.

Run from the repository root with:

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/bin_polyakov_correlators.py \
        results/runs/<run_name>/correlators/polyakov_vector_correlators.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import axis_average_correlators, radial_average_correlators  # noqa: E402


REQUIRED_FIELDS = {
    "correlators",
    "sweeps",
    "chains",
    "starts",
    "filenames",
    "shape",
    "beta",
    "time_direction",
    "thermalization_sweeps",
    "run_name",
}


def load_vector_correlators(path: Path) -> dict[str, np.ndarray]:
    """Load and validate vector correlators from the measurement workflow.

    Inputs:
        path: Polyakov vector correlator NPZ path.
    Outputs:
        Dictionary containing correlators and their measurement metadata.
    """
    if not path.exists():
        raise FileNotFoundError(f"correlator input not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_FIELDS.difference(data.files)
        if missing:
            raise ValueError(f"input is missing required fields: {sorted(missing)}")
        loaded = {name: np.asarray(data[name]) for name in REQUIRED_FIELDS}

    correlators = np.asarray(loaded["correlators"], dtype=np.complex128)
    lattice_shape = tuple(int(value) for value in loaded["shape"])
    if not lattice_shape:
        raise ValueError("shape metadata must contain at least one dimension")
    time_direction = int(loaded["time_direction"])
    if time_direction < -len(lattice_shape) or time_direction >= len(lattice_shape):
        raise ValueError("time_direction is outside shape metadata")
    normalized_time = time_direction % len(lattice_shape)
    spatial_shape = lattice_shape[:normalized_time] + lattice_shape[normalized_time + 1 :]
    expected_shape = (len(loaded["sweeps"]), *spatial_shape)
    if correlators.shape != expected_shape:
        raise ValueError(
            f"correlators have shape {correlators.shape}, expected {expected_shape}"
        )
    for field in ("chains", "starts", "filenames"):
        if len(loaded[field]) != correlators.shape[0]:
            raise ValueError(f"{field} length does not match correlator count")
    loaded["correlators"] = correlators
    return loaded


def binned_output_path(input_path: Path) -> Path:
    """Build the default output path beside a vector correlator file.

    Inputs:
        input_path: Polyakov vector correlator NPZ path.
    Outputs:
        Default radial correlator NPZ path.
    """
    return input_path.with_name("polyakov_binned_correlators.npz")


def write_binned_correlators(
    path: Path,
    source: dict[str, np.ndarray],
) -> None:
    """Bin vector correlators and write results with source metadata.

    Inputs:
        path: Output NPZ path.
        source: Validated measurement data from ``load_vector_correlators``.
    Outputs:
        None.
    """
    radial_r_squared, radial_degeneracies, radial = radial_average_correlators(
        source["correlators"]
    )
    axis_r, axis_degeneracies, axis = axis_average_correlators(source["correlators"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        radial_r_squared=radial_r_squared,
        radial_r=np.sqrt(radial_r_squared.astype(np.float64)),
        radial_degeneracies=radial_degeneracies,
        radial_correlators=radial,
        axis_r=axis_r,
        axis_degeneracies=axis_degeneracies,
        axis_correlators=axis,
        **{name: source[name] for name in REQUIRED_FIELDS if name != "correlators"},
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for correlator binning.

    Inputs:
        None.
    Outputs:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="vector correlator NPZ file")
    parser.add_argument("--output", type=Path, help="output NPZ path")
    parser.add_argument("--overwrite", action="store_true", help="replace output")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run radial and axis-only correlator binning.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = build_argument_parser().parse_args(argv)
    output = args.output or binned_output_path(args.input)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source = load_vector_correlators(args.input)
    write_binned_correlators(output, source)
    print(f"Configurations: {source['correlators'].shape[0]}")
    print(f"Vector correlator shape: {source['correlators'].shape}")
    print(f"Saved binned correlators to {output}")


if __name__ == "__main__":
    main()
