"""
Plot Polyakov-loop correlators on a logarithmic y-axis.

This script expects ``polyakov_binned_correlators.npz`` written by
scripts/bin_polyakov_correlators.py. It plots the real part of the ensemble
mean correlator versus r/a for either axis or radial binning. Non-positive mean
values are omitted from the log plot and reported on stdout.

Run from the repository root with:

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot_polyakov_correlator_log.py \
        results/runs/prod_static_potential_16x16x16x6_b57_hb_2or/correlators/polyakov_binned_correlators.npz \
        --binning axis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def default_output_path(input_path: Path, binning: str) -> Path:
    """Build the default output path.

    Inputs:
        input_path: Binned correlator NPZ path.
        binning: Binning choice, axis or radial.
    Outputs:
        PNG output path beside the input file.
    """
    return input_path.with_name(f"polyakov_{binning}_correlator_log.png")


def load_binned_correlators(path: Path, binning: str) -> tuple[np.ndarray, np.ndarray]:
    """Load distances and binned correlator samples.

    Inputs:
        path: Binned correlator NPZ path.
        binning: Binning choice, axis or radial.
    Outputs:
        Distances r/a and correlator samples with shape (n_configs, n_distances).
    """
    if not path.exists():
        raise FileNotFoundError(f"binned correlator input not found: {path}")
    if binning not in {"axis", "radial"}:
        raise ValueError("binning must be 'axis' or 'radial'")

    with np.load(path, allow_pickle=False) as data:
        radius_field = "axis_r" if binning == "axis" else "radial_r"
        correlator_field = f"{binning}_correlators"
        missing = {radius_field, correlator_field}.difference(data.files)
        if missing:
            raise ValueError(f"input is missing required fields: {sorted(missing)}")
        radii = np.asarray(data[radius_field], dtype=np.float64)
        correlators = np.asarray(data[correlator_field], dtype=np.complex128)

    if correlators.ndim != 2 or correlators.shape[1] != len(radii):
        raise ValueError("correlators must have shape (n_configs, n_distances)")
    return radii, correlators


def correlator_mean_error(correlators: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute naive ensemble mean and standard error.

    Inputs:
        correlators: Complex correlator samples.
    Outputs:
        Real-part mean and standard error.
    """
    values = correlators.real
    mean = values.mean(axis=0)
    error = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, error


def plot_correlator_log(
    input_path: Path,
    output_path: Path,
    binning: str,
    r_max: float | None,
) -> None:
    """Plot the real Polyakov correlator on a logarithmic y-axis.

    Inputs:
        input_path: Binned correlator NPZ path.
        output_path: PNG output path.
        binning: Binning choice, axis or radial.
        r_max: Optional maximum r/a to include.
    Outputs:
        None.
    """
    radii, correlators = load_binned_correlators(input_path, binning)
    mean, error = correlator_mean_error(correlators)

    selected = np.isfinite(radii) & np.isfinite(mean) & np.isfinite(error)
    if r_max is not None:
        selected &= radii <= r_max
    positive = selected & (mean > 0.0)
    omitted = int(np.count_nonzero(selected & ~positive))
    if np.count_nonzero(positive) == 0:
        raise ValueError("no positive correlator means are available for log plotting")

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=180)
    ax.errorbar(
        radii[positive],
        mean[positive],
        yerr=error[positive],
        fmt="o",
        color="#1f5a9d",
        ecolor="#6f8fb5",
        elinewidth=1.1,
        capsize=3,
        markersize=4.5,
        label=f"{binning} mean Re C(r)",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$\mathrm{Re}\,C(r)$")
    ax.set_title("Polyakov-loop correlator")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    if omitted:
        ax.text(
            0.98,
            0.05,
            f"omitted non-positive means: {omitted}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#c5c9d3",
                "alpha": 0.9,
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Positive points plotted: {np.count_nonzero(positive)}")
    print(f"Non-positive means omitted: {omitted}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Inputs:
        None.
    Outputs:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="polyakov_binned_correlators.npz")
    parser.add_argument("--binning", choices=("axis", "radial"), default="axis")
    parser.add_argument("--r-max", type=float, help="maximum r/a to include")
    parser.add_argument("--output", type=Path, help="PNG output path")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run log-scale correlator plotting.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = build_argument_parser().parse_args(argv)
    output_path = args.output or default_output_path(args.input, args.binning)
    plot_correlator_log(args.input, output_path, args.binning, args.r_max)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
