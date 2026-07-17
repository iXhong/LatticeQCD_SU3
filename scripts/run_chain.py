"""
Run gauge-update chains and write reusable run outputs.

This script writes a manifest and an observable history under results/runs/.
Set MEASURE_EVERY = 0 to skip plaquette measurements while generating saved
configurations. Set SAVE_CONFIG_EVERY = 0 for analysis runs that only need
plaquette histories, or set SAVE_CONFIG_EVERY > 0 to save full gauge
configurations periodically. Set STARTS = ("load",), SOURCE_RUN_NAME, and
SOURCE_CHAIN to continue from the latest saved configuration in an existing
run. SWEEPS then means the number of additional sweeps to run.

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
MEASURE_PLAQUETTE = True
MEASURE_POLYAKOV = False
POLYAKOV_TIME_DIRECTION = -1
POLYAKOV_OFFSETS = ((1, 0, 0), (2, 0, 0))
SAVE_CONFIG_EVERY = 0
SEED = 12345
ALGORITHM = "heatbath"
BACKEND = "jit"
OVERRELAXATION_SWEEPS = 0
STARTS = ("hot",)
SOURCE_RUN_NAME = ""
SOURCE_CHAIN: int | None = 0
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
    latest_configuration_path,
    load_start,
    metropolis_sweep,
    overrelaxation_sweep,
    polyakov_loops,
    save_configuration,
)


@dataclass
class SweepRunner:
    """Run configured update sweeps.

    Inputs:
        algorithm: Update algorithm name.
        backend: Heatbath backend name.
        overrelaxation_sweeps: Number of overrelaxation sweeps after heatbath.
        seed: Seed for the first JIT sweep.
        rng: NumPy random generator for NumPy updates.
    Outputs:
        Stateful sweep runner.
    """

    algorithm: str
    backend: str
    overrelaxation_sweeps: int
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
            stats = heatbath_sweep(links, geometry, beta, self.rng)
            return self._run_overrelaxation_sweeps(links, geometry, stats, "numpy")

        if self.backend == "jit_checkerboard":
            from lattice_su3.accelerated import heatbath_checkerboard_jit_sweep

            jit_seed = self.seed if not self.jit_seeded else None
            stats = heatbath_checkerboard_jit_sweep(
                links, geometry, beta, seed=jit_seed
            )
            self.jit_seeded = True
            return self._run_overrelaxation_sweeps(
                links, geometry, stats, "jit_checkerboard"
            )

        if self.backend != "jit":
            raise ValueError("BACKEND must be 'jit', 'jit_checkerboard', or 'numpy'")

        from lattice_su3.accelerated import heatbath_jit_sweep

        jit_seed = self.seed if not self.jit_seeded else None
        stats = heatbath_jit_sweep(links, geometry, beta, seed=jit_seed)
        self.jit_seeded = True
        return self._run_overrelaxation_sweeps(links, geometry, stats, "jit")

    def _run_overrelaxation_sweeps(
        self,
        links: np.ndarray,
        geometry: LatticeGeometry,
        stats,
        backend: str,
    ):
        """Run configured overrelaxation sweeps after heatbath.

        Inputs:
            links: Gauge links U[site, direction].
            geometry: Lattice geometry object.
            stats: UpdateStats object from the heatbath sweep.
            backend: Backend name to use for overrelaxation sweeps.
        Outputs:
            Combined UpdateStats object.
        """
        attempted_links = stats.attempted_links
        accepted_links = stats.accepted_links
        for _ in range(self.overrelaxation_sweeps):
            if backend == "jit":
                from lattice_su3.accelerated import overrelaxation_jit_sweep

                overrelaxation_stats = overrelaxation_jit_sweep(links, geometry)
            elif backend == "jit_checkerboard":
                from lattice_su3.accelerated import (
                    overrelaxation_checkerboard_jit_sweep,
                )

                overrelaxation_stats = overrelaxation_checkerboard_jit_sweep(
                    links, geometry
                )
            else:
                overrelaxation_stats = overrelaxation_sweep(links, geometry, self.rng)
            attempted_links += overrelaxation_stats.attempted_links
            accepted_links += overrelaxation_stats.accepted_links
        return type(stats)(
            attempted_links=attempted_links,
            accepted_links=accepted_links,
        )


@dataclass
class InitialState:
    """Describe one chain's initial gauge configuration.

    Inputs:
        links: Gauge links U[site, direction].
        initial_sweep: Accumulated sweep number of the initial configuration.
        source_path: Loaded configuration path, or None for hot/cold starts.
        source_metadata: Metadata read from the loaded configuration.
    Outputs:
        Initial configuration and provenance information.
    """

    links: np.ndarray
    initial_sweep: int = 0
    source_path: Path | None = None
    source_metadata: dict[str, object] | None = None


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
    if MEASURE_EVERY > 0 and not MEASURE_PLAQUETTE and not MEASURE_POLYAKOV:
        raise ValueError("at least one measurement type must be enabled")
    if MEASURE_POLYAKOV:
        expected_offset_length = len(SHAPE) - 1
        for offset in POLYAKOV_OFFSETS:
            if len(offset) != expected_offset_length:
                raise ValueError(
                    "POLYAKOV_OFFSETS entries must have one component per "
                    "spatial lattice direction"
                )
    if SAVE_CONFIG_EVERY < 0:
        raise ValueError("SAVE_CONFIG_EVERY must be non-negative")
    if OVERRELAXATION_SWEEPS < 0:
        raise ValueError("OVERRELAXATION_SWEEPS must be non-negative")
    if ALGORITHM not in {"heatbath", "metropolis"}:
        raise ValueError("ALGORITHM must be 'heatbath' or 'metropolis'")
    if OVERRELAXATION_SWEEPS > 0 and ALGORITHM != "heatbath":
        raise ValueError("OVERRELAXATION_SWEEPS requires ALGORITHM='heatbath'")
    if BACKEND not in {"jit", "jit_checkerboard", "numpy"}:
        raise ValueError("BACKEND must be 'jit', 'jit_checkerboard', or 'numpy'")
    if ALGORITHM != "heatbath" and BACKEND in {"jit", "jit_checkerboard"}:
        raise ValueError(
            "BACKEND='jit' and BACKEND='jit_checkerboard' are only available "
            "for ALGORITHM='heatbath'"
        )
    if not STARTS:
        raise ValueError("STARTS must contain at least one start type")
    if any(start not in {"cold", "hot", "load"} for start in STARTS):
        raise ValueError("STARTS entries must be 'cold', 'hot', or 'load'")
    if "load" in STARTS and STARTS != ("load",):
        raise ValueError("load mode currently requires STARTS = ('load',)")
    if "load" in STARTS and not SOURCE_RUN_NAME:
        raise ValueError("SOURCE_RUN_NAME is required when STARTS = ('load',)")
    if "load" not in STARTS and SOURCE_RUN_NAME:
        raise ValueError("SOURCE_RUN_NAME requires STARTS = ('load',)")
    if SOURCE_CHAIN is not None and SOURCE_CHAIN < 0:
        raise ValueError("SOURCE_CHAIN must be non-negative or None")
    if STARTS == ("load",) and RUN_NAME and RUN_NAME == SOURCE_RUN_NAME:
        raise ValueError("RUN_NAME must differ from SOURCE_RUN_NAME in load mode")


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
    if STARTS == ("load",) and SOURCE_RUN_NAME:
        starts_label = f"load-{SOURCE_RUN_NAME}"
    algorithm_label = ALGORITHM
    if OVERRELAXATION_SWEEPS > 0:
        algorithm_label = f"{ALGORITHM}_or{OVERRELAXATION_SWEEPS}"
    return (
        f"{algorithm_label}_{BACKEND}_{starts_label}_{shape_label(SHAPE)}_"
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


def source_run_directory(run_name: str | None = None) -> Path:
    """Build the source run directory for a load start.

    Inputs:
        run_name: Existing run directory name, or None to use SOURCE_RUN_NAME.
    Outputs:
        Path to the existing source run directory.
    """
    selected_name = SOURCE_RUN_NAME if run_name is None else run_name
    if not selected_name:
        raise ValueError("source run name must be non-empty")
    if Path(selected_name).name != selected_name:
        raise ValueError("source run name must be a single directory name")
    return ROOT / "results" / "runs" / selected_name


def source_path_label(path: Path) -> str:
    """Format a source path for portable run metadata.

    Inputs:
        path: Loaded configuration path.
    Outputs:
        Repository-relative path when possible, otherwise an absolute path.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def initial_state(
    start: str,
    geometry: LatticeGeometry,
    rng: np.random.Generator,
) -> InitialState:
    """Create the configured starting gauge configuration.

    Inputs:
        start: Initial condition name.
        geometry: Lattice geometry object.
        rng: NumPy random generator.
    Outputs:
        Initial configuration and provenance information.
    """
    if start == "cold":
        return InitialState(cold_start(geometry))
    if start == "hot":
        return InitialState(hot_start(geometry, rng))
    if start != "load":
        raise ValueError("start must be 'cold', 'hot', or 'load'")
    source_path = latest_configuration_path(
        source_run_directory() / "configurations",
        chain=SOURCE_CHAIN,
    )
    links, metadata = load_start(source_path, geometry)
    if "sweep" not in metadata:
        raise ValueError(f"{source_path} is missing required metadata field: sweep")
    source_sweep = int(metadata["sweep"])
    if source_sweep < 0:
        raise ValueError("loaded configuration sweep must be non-negative")

    source_beta = metadata.get("beta")
    if source_beta is not None and not np.isclose(float(source_beta), BETA):
        print(
            f"Warning: loaded configuration beta={source_beta} differs from "
            f"the continuation beta={BETA}"
        )

    return InitialState(
        links,
        initial_sweep=source_sweep,
        source_path=source_path,
        source_metadata=metadata,
    )


def manifest_data(
    state: InitialState | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    """Build metadata for this run.

    Inputs:
        state: Optional loaded initial state with source provenance.
    Outputs:
        JSON-serializable run metadata dictionary.
    """
    metadata: dict[str, object] = {
        "shape": list(SHAPE),
        "beta": BETA,
        "step_size": STEP_SIZE,
        "sweeps": SWEEPS,
        "measure_every": MEASURE_EVERY,
        "measure_plaquette": MEASURE_PLAQUETTE,
        "measure_polyakov": MEASURE_POLYAKOV,
        "polyakov_time_direction": POLYAKOV_TIME_DIRECTION,
        "polyakov_offsets": [list(offset) for offset in POLYAKOV_OFFSETS],
        "save_config_every": SAVE_CONFIG_EVERY,
        "seed": SEED,
        "algorithm": ALGORITHM,
        "backend": BACKEND,
        "overrelaxation_sweeps": OVERRELAXATION_SWEEPS,
        "starts": list(STARTS),
        "run_name": run_label(),
    }
    if state is not None and state.source_path is not None:
        source_metadata = state.source_metadata or {}
        metadata.update(
            {
                "source_configuration": source_path_label(state.source_path),
                "source_run_name": SOURCE_RUN_NAME,
                "source_chain": source_metadata.get("chain", SOURCE_CHAIN),
                "source_sweep": state.initial_sweep,
            }
        )
        if "start" in source_metadata:
            metadata["source_start"] = source_metadata["start"]
        if "beta" in source_metadata:
            metadata["source_beta"] = source_metadata["beta"]
    if elapsed_seconds is not None:
        metadata["elapsed_seconds"] = elapsed_seconds
    return metadata


def write_manifest(
    path: Path,
    state: InitialState | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Write run metadata as JSON.

    Inputs:
        path: Manifest JSON path.
        state: Optional loaded initial state with source provenance.
        elapsed_seconds: Optional completed run wall time.
    Outputs:
        None.
    """
    with open(path, "w") as f:
        json.dump(manifest_data(state, elapsed_seconds), f, indent=2)
        f.write("\n")


def _polyakov_offset_label(offset: tuple[int, ...]) -> str:
    """Format one Polyakov correlator offset for a CSV column name.

    Inputs:
        offset: Spatial displacement vector.
    Outputs:
        Underscore-separated displacement label.
    """
    return "_".join(str(component).replace("-", "m") for component in offset)


def polyakov_fieldnames() -> list[str]:
    """Build Polyakov observable CSV field names.

    Inputs:
        None.
    Outputs:
        Field names for enabled Polyakov scalar measurements.
    """
    if not MEASURE_POLYAKOV:
        return []

    fieldnames = [
        "polyakov_re",
        "polyakov_im",
        "polyakov_abs",
        "polyakov_abs2",
    ]
    for offset in POLYAKOV_OFFSETS:
        label = _polyakov_offset_label(tuple(offset))
        fieldnames.append(f"polyakov_c_{label}_re")
        fieldnames.append(f"polyakov_c_{label}_im")
    return fieldnames


def observable_fieldnames() -> list[str]:
    """Build observable CSV field names for the configured measurements.

    Inputs:
        None.
    Outputs:
        Ordered CSV field names.
    """
    fieldnames = [
        "chain",
        "start",
        "sweep",
        "average_plaquette",
        "acceptance_rate",
        "accepted_links",
        "attempted_links",
    ]
    return fieldnames + polyakov_fieldnames()


def polyakov_measurements(
    links: np.ndarray,
    geometry: LatticeGeometry,
) -> dict[str, float]:
    """Compute selected scalar Polyakov loop observables.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
    Outputs:
        Mapping of Polyakov observable names to scalar values.
    """
    loops = polyakov_loops(links, geometry, POLYAKOV_TIME_DIRECTION)
    pbar = complex(np.mean(loops))
    measurements = {
        "polyakov_re": float(np.real(pbar)),
        "polyakov_im": float(np.imag(pbar)),
        "polyakov_abs": float(abs(pbar)),
        "polyakov_abs2": float(abs(pbar) ** 2),
    }
    axes = tuple(range(loops.ndim))
    for offset in POLYAKOV_OFFSETS:
        offset = tuple(int(component) for component in offset)
        shift = tuple(-component for component in offset)
        shifted = np.roll(loops, shift=shift, axis=axes)
        correlator = complex(np.mean(loops * shifted.conj()))
        label = _polyakov_offset_label(offset)
        measurements[f"polyakov_c_{label}_re"] = float(np.real(correlator))
        measurements[f"polyakov_c_{label}_im"] = float(np.imag(correlator))
    return measurements


def measure_observables(
    links: np.ndarray,
    geometry: LatticeGeometry,
) -> tuple[float | None, dict[str, float]]:
    """Measure configured scalar observables.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
    Outputs:
        Average plaquette or None, and additional scalar measurements.
    """
    plaquette = average_plaquette(links, geometry) if MEASURE_PLAQUETTE else None
    measurements = polyakov_measurements(links, geometry) if MEASURE_POLYAKOV else {}
    return plaquette, measurements


def observable_row(
    chain: int,
    start: str,
    sweep: int,
    plaquette: float | str | None,
    acceptance_rate: float | str,
    accepted_links: int | str,
    attempted_links: int | str,
    measurements: dict[str, float] | None = None,
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
        measurements: Optional extra scalar observable measurements.
    Outputs:
        Observable row dictionary.
    """
    row = {
        "chain": chain,
        "start": start,
        "sweep": sweep,
        "average_plaquette": "" if plaquette is None else plaquette,
        "acceptance_rate": acceptance_rate,
        "accepted_links": accepted_links,
        "attempted_links": attempted_links,
    }
    if measurements:
        row.update(measurements)
    return row


def configuration_metadata(
    chain: int,
    start: str,
    sweep: int,
    plaquette: float | None,
    state: InitialState | None = None,
) -> dict[str, object]:
    """Build metadata for one saved configuration.

    Inputs:
        chain: Chain index within this run.
        start: Initial condition name.
        sweep: Full-lattice sweep number.
        plaquette: Average plaquette value at save time, or None when no
            measurement was made for this sweep.
        state: Optional initial state with source provenance.
    Outputs:
        Metadata dictionary.
    """
    metadata = manifest_data(state)
    metadata.update(
        {
            "chain": chain,
            "start": start,
            "sweep": sweep,
            "initial_sweep": state.initial_sweep if state is not None else 0,
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
    segment_sweep: int | None = None,
    state: InitialState | None = None,
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
        segment_sweep: Sweep number within this run segment. Defaults to sweep.
        state: Optional initial state with source provenance.
    Outputs:
        Saved path, or None when saving is disabled for this sweep.
    """
    save_counter = sweep if segment_sweep is None else segment_sweep
    if (
        SAVE_CONFIG_EVERY == 0
        or save_counter == 0
        or save_counter % SAVE_CONFIG_EVERY != 0
    ):
        return None

    config_dir = out_dir / "configurations"
    filename = f"chain{chain:02d}_{start}_sweep{sweep:06d}.npz"
    path = config_dir / filename
    save_configuration(
        path,
        links,
        configuration_metadata(chain, start, sweep, plaquette, state),
    )
    return path


def run_one_chain(
    writer: csv.DictWriter,
    out_dir: Path,
    chain: int,
    start: str,
    seed_sequence: np.random.SeedSequence,
    prepared_state: InitialState | None = None,
) -> None:
    """Run one Markov chain and write measurements.

    Inputs:
        writer: Observable CSV writer.
        out_dir: Run output directory.
        chain: Chain index within this run.
        start: Initial condition name.
        seed_sequence: Seed sequence for this chain.
        prepared_state: Preloaded initial state for load mode.
    Outputs:
        None.
    """
    geometry = LatticeGeometry(SHAPE)
    rng_seed, jit_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(rng_seed)
    runner = SweepRunner(
        ALGORITHM,
        BACKEND,
        OVERRELAXATION_SWEEPS,
        int(jit_seed.generate_state(1)[0]),
        rng,
    )
    state = prepared_state or initial_state(start, geometry, rng)
    links = state.links
    initial_sweep = state.initial_sweep

    plaquette = None
    measurements: dict[str, float] = {}
    if MEASURE_EVERY > 0:
        plaquette, measurements = measure_observables(links, geometry)
        writer.writerow(
            observable_row(
                chain,
                start,
                initial_sweep,
                plaquette,
                "",
                "",
                "",
                measurements,
            )
        )
    maybe_save_configuration(
        out_dir,
        links,
        chain,
        start,
        initial_sweep,
        plaquette,
        segment_sweep=0,
        state=state,
    )

    for segment_sweep in range(1, SWEEPS + 1):
        sweep = initial_sweep + segment_sweep
        stats = runner.sweep(links, geometry, BETA, STEP_SIZE)
        if MEASURE_EVERY > 0 and segment_sweep % MEASURE_EVERY == 0:
            plaquette, measurements = measure_observables(links, geometry)
            writer.writerow(
                observable_row(
                    chain,
                    start,
                    sweep,
                    plaquette,
                    stats.acceptance_rate,
                    stats.accepted_links,
                    stats.attempted_links,
                    measurements,
                )
            )
        maybe_save_configuration(
            out_dir,
            links,
            chain,
            start,
            sweep,
            plaquette,
            segment_sweep=segment_sweep,
            state=state,
        )

        if segment_sweep % 100 == 0:
            print(
                f"  [{start}] segment sweep {segment_sweep}/{SWEEPS}, "
                f"accumulated sweep {sweep}"
            )


def main() -> None:
    """Run configured Markov chains and save reusable outputs.

    Inputs:
        None.
    Outputs:
        None.
    """
    validate_parameters()
    prepared_state = None
    if STARTS == ("load",):
        geometry = LatticeGeometry(SHAPE)
        prepared_state = initial_state(
            "load", geometry, np.random.default_rng(SEED)
        )
    out_dir = output_directory()
    manifest_path = out_dir / "manifest.json"
    observables_path = out_dir / "observables.csv"
    if observables_path.exists() and not OVERWRITE:
        raise FileExistsError(f"refusing to overwrite existing run: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, prepared_state)
    start_time = perf_counter()

    print(
        f"Lattice: {SHAPE}, beta={BETA}, sweeps={SWEEPS}, "
        f"algorithm={ALGORITHM}, backend={BACKEND}, starts={STARTS}, "
        f"overrelaxation_sweeps={OVERRELAXATION_SWEEPS}, "
        f"measure_every={MEASURE_EVERY}, save_config_every={SAVE_CONFIG_EVERY}"
    )

    fieldnames = observable_fieldnames()
    chain_seeds = np.random.SeedSequence(SEED).spawn(len(STARTS))
    with open(observables_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chain, (start, seed_sequence) in enumerate(
            zip(STARTS, chain_seeds, strict=True)
        ):
            run_one_chain(
                writer,
                out_dir,
                chain,
                start,
                seed_sequence,
                prepared_state=prepared_state,
            )

    elapsed = perf_counter() - start_time
    write_manifest(manifest_path, prepared_state, elapsed_seconds=elapsed)
    print(f"Manifest saved to {manifest_path}")
    print(f"Observables saved to {observables_path}")
    if SAVE_CONFIG_EVERY > 0:
        print(f"Configurations saved under {out_dir / 'configurations'}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
