"""
Extract and fit a static potential from resampled Polyakov correlators.

This script reads ``polyakov_resampled_correlators.npz``, transforms either the
axis or radial correlator into ``aV(r)``, constructs a jackknife or bootstrap
covariance, performs a correlated Cornell fit, scans nested fit windows, and
optionally converts ``r0/a`` to a lattice spacing using a supplied physical r0.

Run from the repository root with:

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analyze_static_potential.py \
        results/runs/<run_name>/correlators/polyakov_resampled_correlators.npz \
        --binning axis --method jackknife --r-min 2 --r-max 7
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

from lattice_su3 import (  # noqa: E402
    bootstrap_mean_covariance,
    fit_cornell_correlated,
    fit_cornell_samples,
    jackknife_mean_covariance,
    potential_from_correlators,
    sommer_scale_r0_over_a,
)


def load_analysis_input(
    path: Path, binning: str, method: str
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load distances, block correlators, and selected resampling samples.

    Inputs:
        path: Resampled Polyakov correlator NPZ path.
        binning: ``axis`` or ``radial``.
        method: ``jackknife`` or ``bootstrap``.
    Outputs:
        Metadata dictionary, radii, block correlators, and resampled correlators.
    """
    if not path.exists():
        raise FileNotFoundError(f"resampled correlator input not found: {path}")
    if binning not in {"axis", "radial"}:
        raise ValueError("binning must be 'axis' or 'radial'")
    if method not in {"jackknife", "bootstrap"}:
        raise ValueError("method must be 'jackknife' or 'bootstrap'")
    with np.load(path, allow_pickle=False) as data:
        radius_field = "axis_r" if binning == "axis" else "radial_r"
        block_field = f"{binning}_block_correlators"
        sample_field = f"{binning}_{method}_correlators"
        required = {radius_field, block_field, sample_field, "shape", "time_direction"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"input is missing required fields: {sorted(missing)}")
        metadata = {name: np.asarray(data[name]) for name in data.files}
    return (
        metadata,
        np.asarray(metadata[radius_field], dtype=np.float64),
        np.asarray(metadata[block_field], dtype=np.complex128),
        np.asarray(metadata[sample_field], dtype=np.complex128),
    )


def scan_fit_windows(
    radii: np.ndarray,
    potential: np.ndarray,
    covariance: np.ndarray,
    minimum_points: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit all contiguous distance windows containing enough points.

    Inputs:
        radii: Positive fitted distances.
        potential: Potential central values.
        covariance: Potential covariance matrix.
        minimum_points: Minimum number of points in each window.
    Outputs:
        Window bounds, Cornell parameters, and ``(chi2, dof)`` diagnostics.
    """
    windows = []
    parameters = []
    diagnostics = []
    for start in range(len(radii)):
        for stop in range(start + minimum_points - 1, len(radii)):
            try:
                fit = fit_cornell_correlated(
                    radii, potential, covariance, radii[start], radii[stop]
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            windows.append((radii[start], radii[stop]))
            parameters.append(fit.parameters)
            diagnostics.append((fit.chi_squared, fit.degrees_of_freedom))
    return (
        np.asarray(windows, dtype=np.float64).reshape(-1, 2),
        np.asarray(parameters, dtype=np.float64).reshape(-1, 3),
        np.asarray(diagnostics, dtype=np.float64).reshape(-1, 2),
    )


def analyze(
    input_path: Path,
    output_path: Path,
    binning: str,
    method: str,
    r_min: float,
    r_max: float,
    r0_physical_fm: float | None,
) -> None:
    """Extract, fit, and save one static-potential analysis.

    Inputs:
        input_path: Resampled correlator NPZ path.
        output_path: Analysis NPZ output path.
        binning: ``axis`` or ``radial`` correlators.
        method: Covariance method, ``jackknife`` or ``bootstrap``.
        r_min: Inclusive primary fit lower bound.
        r_max: Inclusive primary fit upper bound.
        r0_physical_fm: Optional physical Sommer scale in femtometers.
    Outputs:
        None.
    """
    metadata, radii, block_correlators, sample_correlators = load_analysis_input(
        input_path, binning, method
    )
    lattice_shape = tuple(int(value) for value in metadata["shape"])
    time_direction = int(metadata["time_direction"]) % len(lattice_shape)
    nt = lattice_shape[time_direction]
    full_potential = potential_from_correlators(block_correlators.mean(axis=0)[None], nt)[0]
    potential_samples = potential_from_correlators(sample_correlators, nt)
    valid_fraction = np.mean(np.isfinite(potential_samples), axis=0)
    required_fraction = 1.0 if method == "jackknife" else 0.95
    valid = np.isfinite(full_potential) & (valid_fraction >= required_fraction)
    valid &= radii > 0.0
    if np.count_nonzero(valid) < 4:
        raise ValueError(
            "fewer than four distances meet the positive-correlator sample threshold"
        )

    selected_radii = radii[valid]
    selected_full = full_potential[valid]
    selected_valid_fraction = valid_fraction[valid]
    selected_samples = potential_samples[:, valid]
    finite_sample_rows = np.all(np.isfinite(selected_samples), axis=1)
    selected_samples = selected_samples[finite_sample_rows]
    if len(selected_samples) < 2:
        raise ValueError("fewer than two resamples remain after positivity filtering")
    if method == "jackknife":
        sample_mean, covariance = jackknife_mean_covariance(selected_samples)
    else:
        sample_mean, covariance = bootstrap_mean_covariance(selected_samples)

    fit = fit_cornell_correlated(
        selected_radii, selected_full, covariance, r_min, r_max
    )
    parameter_samples = fit_cornell_samples(
        selected_radii, selected_samples, covariance, r_min, r_max
    )
    if method == "jackknife":
        _, parameter_covariance = jackknife_mean_covariance(parameter_samples)
    else:
        _, parameter_covariance = bootstrap_mean_covariance(parameter_samples)
    r0_over_a = float(sommer_scale_r0_over_a(fit.parameters))
    r0_samples = sommer_scale_r0_over_a(parameter_samples)
    lattice_spacing_fm = (
        np.nan if r0_physical_fm is None else r0_physical_fm / r0_over_a
    )

    windows, window_parameters, window_diagnostics = scan_fit_windows(
        selected_radii, selected_full, covariance
    )
    window_r0 = sommer_scale_r0_over_a(window_parameters)
    acceptable = (
        (window_diagnostics[:, 1] > 0)
        & (window_diagnostics[:, 0] / window_diagnostics[:, 1] <= 2.0)
        & np.isfinite(window_r0)
    )
    systematic_r0_std = float(np.std(window_r0[acceptable], ddof=1)) if acceptable.sum() > 1 else np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        radii=selected_radii,
        potential=selected_full,
        resample_potential_mean=sample_mean,
        potential_covariance=covariance,
        potential_error=np.sqrt(np.diag(covariance)),
        positive_sample_fraction=selected_valid_fraction,
        resample_count_input=np.asarray(len(potential_samples)),
        resample_count_used=np.asarray(len(selected_samples)),
        required_positive_fraction=np.asarray(required_fraction),
        fit_parameters=fit.parameters,
        fit_parameter_covariance=parameter_covariance,
        fit_chi_squared=np.asarray(fit.chi_squared),
        fit_degrees_of_freedom=np.asarray(fit.degrees_of_freedom),
        fit_r_min=np.asarray(r_min),
        fit_r_max=np.asarray(r_max),
        parameter_samples=parameter_samples,
        r0_over_a=np.asarray(r0_over_a),
        r0_over_a_samples=r0_samples,
        r0_physical_fm=np.asarray(np.nan if r0_physical_fm is None else r0_physical_fm),
        lattice_spacing_fm=np.asarray(lattice_spacing_fm),
        window_bounds=windows,
        window_parameters=window_parameters,
        window_chi_squared_dof=window_diagnostics,
        window_r0_over_a=window_r0,
        window_acceptable=acceptable,
        systematic_r0_std=np.asarray(systematic_r0_std),
        binning=np.asarray(binning),
        method=np.asarray(method),
        nt=np.asarray(nt),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the static-potential analysis command-line parser.

    Inputs:
        None.
    Outputs:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binning", choices=("axis", "radial"), default="axis")
    parser.add_argument("--method", choices=("jackknife", "bootstrap"), default="jackknife")
    parser.add_argument("--r-min", type=float, required=True)
    parser.add_argument("--r-max", type=float, required=True)
    parser.add_argument("--r0-physical-fm", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run static-potential extraction and correlated fitting.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = build_argument_parser().parse_args(argv)
    output = args.output or args.input.with_name("static_potential_analysis.npz")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    analyze(
        args.input,
        output,
        args.binning,
        args.method,
        args.r_min,
        args.r_max,
        args.r0_physical_fm,
    )
    with np.load(output, allow_pickle=False) as result:
        params = result["fit_parameters"]
        print(f"Cornell parameters [A, B, sigma*a^2]: {params.tolist()}")
        print(
            f"chi2/dof: {result['fit_chi_squared'].item():.4g}/"
            f"{result['fit_degrees_of_freedom'].item()}"
        )
        print(f"r0/a: {result['r0_over_a'].item():.6g}")
        print(f"fit-window systematic std(r0/a): {result['systematic_r0_std'].item():.6g}")
    print(f"Saved static-potential analysis to {output}")


if __name__ == "__main__":
    main()
