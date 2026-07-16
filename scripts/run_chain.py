"""
Run gauge-update chains and write reusable run outputs.

This script writes a manifest and an observable history under results/runs/.
Set MEASURE_EVERY = 0 to skip plaquette measurements while generating saved
configurations. Set SAVE_CONFIG_EVERY = 0 for analysis runs that only need
plaquette histories, or set SAVE_CONFIG_EVERY > 0 to save full gauge
configurations periodically.

Usage:
    Edit the script-level parameters below, then run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_chain.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

SHAPE = (16, 16, 16, 6)
BETA = 5.7
STEP_SIZE = 0.4
SWEEPS = 300
MEASURE_EVERY = 1
SAVE_CONFIG_EVERY = 0
SEED = 12345
ALGORITHM = "heatbath"
BACKEND = "jit"
STARTS = ("hot",)
RUN_NAME = ""
OVERWRITE = False

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
    save_configuration,
)


@dataclass
class SweepRunner:
    """Run configured update sweeps.

    Inputs:
        algorithm: Update algorithm name.
        backend: Heatbath backend name.
        seed: Seed for the first JIT sweep.
        rng: NumPy random generator for NumPy updates.
    Outputs:
        Stateful sweep runner.
    """

    algorithm: str
    backend: str
    seed: int
    rng: np.random.Generator
    jit_seeded: bool = False

    def sweep(
        self,
        links: np.ndarray,
        geometry: LatticeGeometry,
        beta: float,
        step_size: float,
    ):
        """Run one configured update sweep.

        Inputs:
            links: Gauge links U[site, direction].
            geometry: Lattice geometry object.
            beta: Wilson gauge coupling parameter.
            step_size: Maximum SU(2) subgroup proposal radius for Metropolis.
        Outputs:
            UpdateStats object from the configured sweep.
        """
        if self.algorithm == "metropolis":
            return metropolis_sweep(links, geometry, beta, step_size, self.rng)

        if self.backend == "numpy":
            return heatbath_sweep(links, geometry, beta, self.rng)

        if self.backend == "jit_checkerboard":
            from lattice_su3.accelerated import heatbath_checkerboard_jit_sweep

            jit_seed = self.seed if not self.jit_seeded else None
            stats = heatbath_checkerboard_jit_sweep(
                links, geometry, beta, seed=jit_seed
            )
            self.jit_seeded = True
            return stats

        if self.backend != "jit":
            raise ValueError("BACKEND must be 'jit', 'jit_checkerboard', or 'numpy'")

        from lattice_su3.accelerated import heatbath_jit_sweep

        jit_seed = self.seed if not self.jit_seeded else None
        stats = heatbath_jit_sweep(links, geometry, beta, seed=jit_seed)
        self.jit_seeded = True
        return stats


def validate_parameters() -> None:
    """Validate script-level run parameters.

    Inputs:
        None.
    Outputs:
        None.
    """
    if len(SHAPE) < 2:
        raise ValueError("SHAPE must contain at least two lattice directions")
    if BETA < 0.0:
        raise ValueError("BETA must be non-negative")
    if SWEEPS < 0:
        raise ValueError("SWEEPS must be non-negative")
    if MEASURE_EVERY < 0:
        raise ValueError("MEASURE_EVERY must be non-negative")
    if SAVE_CONFIG_EVERY < 0:
        raise ValueError("SAVE_CONFIG_EVERY must be non-negative")
    if ALGORITHM not in {"heatbath", "metropolis"}:
        raise ValueError("ALGORITHM must be 'heatbath' or 'metropolis'")
    if BACKEND not in {"jit", "jit_checkerboard", "numpy"}:
        raise ValueError("BACKEND must be 'jit', 'jit_checkerboard', or 'numpy'")
    if ALGORITHM != "heatbath" and BACKEND in {"jit", "jit_checkerboard"}:
        raise ValueError(
            "BACKEND='jit' and BACKEND='jit_checkerboard' are only available "
            "for ALGORITHM='heatbath'"
        )
    if not STARTS:
        raise ValueError("STARTS must contain at least one start type")
    if any(start not in {"cold", "hot"} for start in STARTS):
        raise ValueError("STARTS entries must be 'cold' or 'hot'")


def shape_label(shape: tuple[int, ...]) -> str:
    """Format a lattice shape for filenames.

    Inputs:
        shape: Lattice size in each direction.
    Outputs:
        Shape label with dimensions joined by x.
    """
    return "x".join(str(length) for length in shape)


def run_label() -> str:
    """Build the run directory label.

    Inputs:
        None.
    Outputs:
        Stable label for this configured run.
    """
    if RUN_NAME:
        return RUN_NAME

    starts_label = "-".join(STARTS)
    return (
        f"{ALGORITHM}_{BACKEND}_{starts_label}_{shape_label(SHAPE)}_"
        f"beta{BETA}_{SWEEPS}sweeps_seed{SEED}"
    )


def output_directory() -> Path:
    """Build the output directory for this run.

    Inputs:
        None.
    Outputs:
        Directory path containing manifest, observables, and configurations.
    """
    return ROOT / "results" / "runs" / run_label()


def initial_links(
    start: str,
    geometry: LatticeGeometry,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create the configured starting gauge configuration.

    Inputs:
        start: Initial condition name.
        geometry: Lattice geometry object.
        rng: NumPy random generator.
    Outputs:
        Gauge links U[site, direction].
    """
    if start == "cold":
        return cold_start(geometry)
    if start == "hot":
        return hot_start(geometry, rng)
    raise ValueError("start must be 'cold' or 'hot'")


def manifest_data() -> dict[str, object]:
    """Build metadata for this run.

    Inputs:
        None.
    Outputs:
        JSON-serializable run metadata dictionary.
    """
    return {
        "shape": list(SHAPE),
        "beta": BETA,
        "step_size": STEP_SIZE,
        "sweeps": SWEEPS,
        "measure_every": MEASURE_EVERY,
        "save_config_every": SAVE_CONFIG_EVERY,
        "seed": SEED,
        "algorithm": ALGORITHM,
        "backend": BACKEND,
        "starts": list(STARTS),
        "run_name": run_label(),
    }


def write_manifest(path: Path) -> None:
    """Write run metadata as JSON.

    Inputs:
        path: Manifest JSON path.
    Outputs:
        None.
    """
    with open(path, "w") as f:
        json.dump(manifest_data(), f, indent=2)
        f.write("\n")


def observable_row(
    chain: int,
    start: str,
    sweep: int,
    plaquette: float,
    acceptance_rate: float | str,
    accepted_links: int | str,
    attempted_links: int | str,
) -> dict[str, object]:
    """Build one observable CSV row.

    Inputs:
        chain: Chain index within this run.
        start: Initial condition name.
        sweep: Full-lattice sweep number.
        plaquette: Average plaquette measurement.
        acceptance_rate: Last sweep acceptance rate, or blank for sweep zero.
        accepted_links: Last sweep accepted link count, or blank for sweep zero.
        attempted_links: Last sweep attempted link count, or blank for sweep zero.
    Outputs:
        Observable row dictionary.
    """
    return {
        "chain": chain,
        "start": start,
        "sweep": sweep,
        "average_plaquette": plaquette,
        "acceptance_rate": acceptance_rate,
        "accepted_links": accepted_links,
        "attempted_links": attempted_links,
    }


def configuration_metadata(
    chain: int,
    start: str,
    sweep: int,
    plaquette: float | None,
) -> dict[str, object]:
    """Build metadata for one saved configuration.

    Inputs:
        chain: Chain index within this run.
        start: Initial condition name.
        sweep: Full-lattice sweep number.
        plaquette: Average plaquette value at save time, or None when no
            measurement was made for this sweep.
    Outputs:
        Metadata dictionary.
    """
    metadata = manifest_data()
    metadata.update(
        {
            "chain": chain,
            "start": start,
            "sweep": sweep,
        }
    )
    if plaquette is not None:
        metadata["average_plaquette"] = plaquette
    return metadata


def maybe_save_configuration(
    out_dir: Path,
    links: np.ndarray,
    chain: int,
    start: str,
    sweep: int,
    plaquette: float | None,
) -> Path | None:
    """Save a configuration when the configured interval is reached.

    Inputs:
        out_dir: Run output directory.
        links: Gauge links U[site, direction].
        chain: Chain index within this run.
        start: Initial condition name.
        sweep: Full-lattice sweep number.
        plaquette: Average plaquette value at save time, or None when saving
            without measuring.
    Outputs:
        Saved path, or None when saving is disabled for this sweep.
    """
    if SAVE_CONFIG_EVERY == 0 or sweep == 0 or sweep % SAVE_CONFIG_EVERY != 0:
        return None

    config_dir = out_dir / "configurations"
    filename = f"chain{chain:02d}_{start}_sweep{sweep:06d}.npz"
    path = config_dir / filename
    save_configuration(
        path,
        links,
        configuration_metadata(chain, start, sweep, plaquette),
    )
    return path


def run_one_chain(
    writer: csv.DictWriter,
    out_dir: Path,
    chain: int,
    start: str,
    seed_sequence: np.random.SeedSequence,
) -> None:
    """Run one Markov chain and write measurements.

    Inputs:
        writer: Observable CSV writer.
        out_dir: Run output directory.
        chain: Chain index within this run.
        start: Initial condition name.
        seed_sequence: Seed sequence for this chain.
    Outputs:
        None.
    """
    geometry = LatticeGeometry(SHAPE)
    rng_seed, jit_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(rng_seed)
    runner = SweepRunner(ALGORITHM, BACKEND, int(jit_seed.generate_state(1)[0]), rng)
    links = initial_links(start, geometry, rng)

    plaquette = None
    if MEASURE_EVERY > 0:
        plaquette = average_plaquette(links, geometry)
        writer.writerow(observable_row(chain, start, 0, plaquette, "", "", ""))
    maybe_save_configuration(out_dir, links, chain, start, 0, plaquette)

    for sweep in range(1, SWEEPS + 1):
        stats = runner.sweep(links, geometry, BETA, STEP_SIZE)
        if MEASURE_EVERY > 0 and sweep % MEASURE_EVERY == 0:
            plaquette = average_plaquette(links, geometry)
            writer.writerow(
                observable_row(
                    chain,
                    start,
                    sweep,
                    plaquette,
                    stats.acceptance_rate,
                    stats.accepted_links,
                    stats.attempted_links,
                )
            )
        maybe_save_configuration(out_dir, links, chain, start, sweep, plaquette)

        if sweep % 100 == 0:
            print(f"  [{start}] sweep {sweep}/{SWEEPS}")


def main() -> None:
    """Run configured Markov chains and save reusable outputs.

    Inputs:
        None.
    Outputs:
        None.
    """
    validate_parameters()
    out_dir = output_directory()
    manifest_path = out_dir / "manifest.json"
    observables_path = out_dir / "observables.csv"
    if observables_path.exists() and not OVERWRITE:
        raise FileExistsError(f"refusing to overwrite existing run: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path)
    start_time = perf_counter()

    print(
        f"Lattice: {SHAPE}, beta={BETA}, sweeps={SWEEPS}, "
        f"algorithm={ALGORITHM}, backend={BACKEND}, starts={STARTS}, "
        f"measure_every={MEASURE_EVERY}, save_config_every={SAVE_CONFIG_EVERY}"
    )

    fieldnames = list(observable_row(0, "hot", 0, 0.0, "", "", "").keys())
    chain_seeds = np.random.SeedSequence(SEED).spawn(len(STARTS))
    with open(observables_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chain, (start, seed_sequence) in enumerate(
            zip(STARTS, chain_seeds, strict=True)
        ):
            run_one_chain(writer, out_dir, chain, start, seed_sequence)

    elapsed = perf_counter() - start_time
    print(f"Manifest saved to {manifest_path}")
    print(f"Observables saved to {observables_path}")
    if SAVE_CONFIG_EVERY > 0:
        print(f"Configurations saved under {out_dir / 'configurations'}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
