"""
Block and jackknife binned Polyakov correlators within Markov chains.

This script reads ``polyakov_binned_correlators.npz`` written by
``scripts/analysis/bin_polyakov_correlators.py``. It sorts configurations by sweep within
each chain, forms equal-size blocks without crossing chains, and writes block
means plus delete-one-block jackknife correlators.

Run from the repository root with:

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analysis/resample_polyakov_correlators.py \
        results/runs/<run_name>/correlators/polyakov_binned_correlators.npz \
        --block-size 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import (  # noqa: E402
    block_by_chain,
    bootstrap_by_chain,
    jackknife_delete_one,
)


REQUIRED_FIELDS = {
    "radial_r_squared",
    "radial_r",
    "radial_degeneracies",
    "radial_correlators",
    "axis_r",
    "axis_degeneracies",
    "axis_correlators",
    "sweeps",
    "chains",
    "shape",
    "beta",
    "time_direction",
    "run_name",
}


def load_binned_correlators(path: Path) -> dict[str, np.ndarray]:
    """Load and validate binned correlators and chain metadata.

    Inputs:
        path: Binned Polyakov correlator NPZ path.
    Outputs:
        Dictionary containing binned correlators and metadata.
    """
    if not path.exists():
        raise FileNotFoundError(f"binned correlator input not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_FIELDS.difference(data.files)
        if missing:
            raise ValueError(f"input is missing required fields: {sorted(missing)}")
        loaded = {name: np.asarray(data[name]) for name in data.files}

    n_cfg = len(loaded["sweeps"])
    if loaded["chains"].shape != (n_cfg,):
        raise ValueError("chains must be one-dimensional and match sweeps")
    for field in ("radial_correlators", "axis_correlators"):
        if loaded[field].ndim != 2 or loaded[field].shape[0] != n_cfg:
            raise ValueError(f"{field} must have shape (n_cfg, n_bins)")
    return loaded


def resampled_output_path(input_path: Path) -> Path:
    """Build the default resampled correlator output path.

    Inputs:
        input_path: Binned Polyakov correlator NPZ path.
    Outputs:
        Default blocked and jackknife correlator NPZ path.
    """
    return input_path.with_name("polyakov_resampled_correlators.npz")


def write_resampled_correlators(
    path: Path,
    source: dict[str, np.ndarray],
    block_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    """Block radial and axis correlators and write jackknife samples.

    Inputs:
        path: Output NPZ path.
        source: Validated binned correlator data.
        block_size: Number of configurations per chain-local block.
        bootstrap_samples: Number of stratified bootstrap ensemble means.
        bootstrap_seed: Random seed for bootstrap reproducibility.
    Outputs:
        None.
    """
    radial_blocks = block_by_chain(
        source["radial_correlators"], source["chains"], source["sweeps"], block_size
    )
    axis_blocks = block_by_chain(
        source["axis_correlators"], source["chains"], source["sweeps"], block_size
    )
    if not np.array_equal(radial_blocks.chains, axis_blocks.chains):
        raise RuntimeError("radial and axis block provenance does not match")

    metadata_fields = (
        "radial_r_squared",
        "radial_r",
        "radial_degeneracies",
        "axis_r",
        "axis_degeneracies",
        "shape",
        "beta",
        "time_direction",
        "run_name",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        radial_block_correlators=radial_blocks.values,
        radial_jackknife_correlators=jackknife_delete_one(radial_blocks.values),
        radial_bootstrap_correlators=bootstrap_by_chain(
            radial_blocks.values,
            radial_blocks.chains,
            bootstrap_samples,
            np.random.default_rng(bootstrap_seed),
        ),
        axis_block_correlators=axis_blocks.values,
        axis_jackknife_correlators=jackknife_delete_one(axis_blocks.values),
        axis_bootstrap_correlators=bootstrap_by_chain(
            axis_blocks.values,
            axis_blocks.chains,
            bootstrap_samples,
            np.random.default_rng(bootstrap_seed),
        ),
        block_chains=radial_blocks.chains,
        block_sweep_start=radial_blocks.sweep_start,
        block_sweep_stop=radial_blocks.sweep_stop,
        block_size=np.asarray(block_size, dtype=np.int64),
        bootstrap_samples=np.asarray(bootstrap_samples, dtype=np.int64),
        bootstrap_seed=np.asarray(bootstrap_seed, dtype=np.int64),
        dropped_per_chain=radial_blocks.dropped_per_chain,
        **{name: source[name] for name in metadata_fields},
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for chain-aware resampling.

    Inputs:
        None.
    Outputs:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="binned correlator NPZ file")
    parser.add_argument("--block-size", required=True, type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, help="output NPZ path")
    parser.add_argument("--overwrite", action="store_true", help="replace output")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run chain-aware blocking and jackknife construction.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = build_argument_parser().parse_args(argv)
    output = args.output or resampled_output_path(args.input)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source = load_binned_correlators(args.input)
    write_resampled_correlators(
        output,
        source,
        args.block_size,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    with np.load(output, allow_pickle=False) as result:
        print(f"Configurations: {len(source['sweeps'])}")
        print(f"Blocks: {len(result['block_chains'])}, block size: {args.block_size}")
        print(f"Dropped tails [chain, count]: {result['dropped_per_chain'].tolist()}")
    print(f"Saved resampled correlators to {output}")


if __name__ == "__main__":
    main()
