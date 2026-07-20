"""
Plot static-potential points and the Cornell fit from an analysis NPZ file.

This script expects an output file written by scripts/analyze_static_potential.py,
for example static_axis_b05_jk_1_5.npz under a run correlators/ directory. It
plots aV(r) against r/a with resampling errors and overlays the fitted Cornell
form.

Run from the repository root with:

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot_static_potential.py \
        results/runs/prod_static_potential_16x16x16x6_b57_hb_2or/correlators/static_axis_b05_jk_1_5.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def cornell_potential(radii: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate the Cornell potential form.

    Inputs:
        radii: Distances r/a.
        parameters: Cornell parameters [A, B, sigma*a^2].
    Outputs:
        Potential values A + B/r + sigma*r.
    """
    constant, coulomb, string_tension = parameters
    return constant + coulomb / radii + string_tension * radii


def default_output_path(input_path: Path) -> Path:
    """Build the default PNG path beside an analysis NPZ file.

    Inputs:
        input_path: Static-potential analysis NPZ path.
    Outputs:
        PNG output path.
    """
    return input_path.with_name(f"{input_path.stem}_potential.png")


def load_analysis(path: Path) -> dict[str, np.ndarray]:
    """Load static-potential analysis arrays.

    Inputs:
        path: Analysis NPZ path from scripts/analyze_static_potential.py.
    Outputs:
        Dictionary of arrays needed for plotting.
    """
    if not path.exists():
        raise FileNotFoundError(f"analysis input not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "radii",
            "potential",
            "potential_error",
            "fit_parameters",
            "fit_r_min",
            "fit_r_max",
            "fit_chi_squared",
            "fit_degrees_of_freedom",
            "r0_over_a",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"input is missing required fields: {sorted(missing)}")
        return {name: np.asarray(data[name]) for name in data.files}


def plot_static_potential(input_path: Path, output_path: Path) -> None:
    """Plot static-potential data and the Cornell fit.

    Inputs:
        input_path: Analysis NPZ path from scripts/analyze_static_potential.py.
        output_path: PNG output path.
    Outputs:
        None.
    """
    analysis = load_analysis(input_path)
    radii = analysis["radii"].astype(float)
    potential = analysis["potential"].astype(float)
    error = analysis["potential_error"].astype(float)
    parameters = analysis["fit_parameters"].astype(float)
    r_min = float(analysis["fit_r_min"])
    r_max = float(analysis["fit_r_max"])
    chi2 = float(analysis["fit_chi_squared"])
    dof = int(analysis["fit_degrees_of_freedom"])
    r0_over_a = float(analysis["r0_over_a"])

    finite = np.isfinite(radii) & np.isfinite(potential) & np.isfinite(error)
    fitted = finite & (radii >= r_min) & (radii <= r_max)
    if np.count_nonzero(fitted) < 2:
        raise ValueError("fewer than two finite fitted points are available")

    fit_radii = np.linspace(radii[fitted].min(), radii[fitted].max(), 300)
    fit_potential = cornell_potential(fit_radii, parameters)
    string_tension = parameters[2]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=180)
    ax.errorbar(
        radii[finite],
        potential[finite],
        yerr=error[finite],
        fmt="o",
        color="#1f5a9d",
        ecolor="#6f8fb5",
        elinewidth=1.1,
        capsize=3,
        markersize=4.5,
        label="Potential data",
    )
    ax.plot(fit_radii, fit_potential, color="#b23a30", linewidth=1.8, label="Cornell fit")
    ax.axvspan(r_min, r_max, color="#d8dde6", alpha=0.28, linewidth=0)
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$aV(r)$")
    ax.set_title("Static potential from Polyakov correlators")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, loc="best")

    summary = "\n".join(
        [
            rf"$\sigma a^2={string_tension:.4f}$",
            rf"$r_0/a={r0_over_a:.3f}$",
            rf"$\chi^2/\mathrm{{dof}}={chi2:.3g}/{dof}$",
        ]
    )
    ax.text(
        0.98,
        0.05,
        summary,
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Inputs:
        None.
    Outputs:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="static-potential analysis NPZ")
    parser.add_argument("--output", type=Path, help="PNG output path")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run static-potential plotting.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = build_argument_parser().parse_args(argv)
    output_path = args.output or default_output_path(args.input)
    plot_static_potential(args.input, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
