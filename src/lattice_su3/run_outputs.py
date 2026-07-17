"""Standard run artifact writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from lattice_su3.configuration import save_configuration
from lattice_su3.run_config import MeasureConfig, SaveConfig, UpdateConfig


OBSERVABLE_FIELDNAMES = [
    "chain",
    "start",
    "sweep",
    "average_plaquette",
    "acceptance_rate",
    "accepted_links",
    "attempted_links",
]


def run_directory(root: Path, run_name: str) -> Path:
    """Build a standard run directory.

    Inputs:
        root: Repository root.
        run_name: Run directory name.
    Outputs:
        Path to results/runs/<run_name>.
    """
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be a single non-empty directory name")
    return root / "results" / "runs" / run_name


def manifest_data(
    *,
    run_name: str,
    shape: tuple[int, ...],
    beta: float,
    sweeps: int,
    seed: int | None,
    starts: Iterable[str],
    update: UpdateConfig,
    measure: MeasureConfig,
    save: SaveConfig,
    workflow: str,
    chains: int = 1,
    source_config: str | None = None,
    discard_sweeps: int | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    """Build standard run metadata.

    Inputs:
        run_name: Run directory name.
        shape: Lattice shape.
        beta: Wilson gauge coupling.
        sweeps: Number of sweeps per chain or segment.
        seed: Seed or seed base.
        starts: Start labels present in this run.
        update: Update settings.
        measure: Measurement settings.
        save: Save settings.
        workflow: Workflow label.
        chains: Number of chains in the run.
        source_config: Optional source configuration path.
        discard_sweeps: Optional production discard count.
        elapsed_seconds: Optional completed wall time.
    Outputs:
        JSON-serializable metadata dictionary.
    """
    metadata: dict[str, object] = {
        "shape": list(shape),
        "beta": beta,
        "step_size": update.step_size,
        "sweeps": sweeps,
        "measure_every": measure.plaquette_every,
        "measure_plaquette": measure.plaquette_every > 0,
        "measure_polyakov": False,
        "save_config_every": save.config_every,
        "seed": seed,
        "algorithm": update.algorithm,
        "backend": update.backend,
        "overrelaxation_sweeps": update.overrelaxation_sweeps,
        "starts": list(starts),
        "run_name": run_name,
        "workflow": workflow,
        "chains": chains,
    }
    if source_config is not None:
        metadata["source_configuration"] = source_config
    if discard_sweeps is not None:
        metadata["discard_sweeps"] = discard_sweeps
    if elapsed_seconds is not None:
        metadata["elapsed_seconds"] = elapsed_seconds
    return metadata


def write_manifest(path: Path, metadata: dict[str, object]) -> None:
    """Write run metadata as JSON.

    Inputs:
        path: Output manifest path.
        metadata: JSON-serializable metadata.
    Outputs:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def prepare_run_directory(out_dir: Path, overwrite: bool) -> None:
    """Create a run directory and guard against accidental overwrite.

    Inputs:
        out_dir: Run output directory.
        overwrite: Whether existing standard outputs may be replaced.
    Outputs:
        None.
    """
    manifest_path = out_dir / "manifest.json"
    observables_path = out_dir / "observables.csv"
    if not overwrite and (manifest_path.exists() or observables_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def open_observable_writer(path: Path) -> tuple[object, csv.DictWriter]:
    """Open a standard observable CSV writer.

    Inputs:
        path: Output CSV path.
    Outputs:
        Open file object and initialized CSV writer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=OBSERVABLE_FIELDNAMES)
    writer.writeheader()
    return handle, writer


def observable_row(
    *,
    chain: int,
    start: str,
    sweep: int,
    plaquette: float | None,
    acceptance_rate: float | str,
    accepted_links: int | str,
    attempted_links: int | str,
) -> dict[str, object]:
    """Build one standard observable row.

    Inputs:
        chain: Chain index.
        start: Start label.
        sweep: Full sweep number.
        plaquette: Average plaquette or None.
        acceptance_rate: Last sweep acceptance rate or blank.
        accepted_links: Last sweep accepted link count or blank.
        attempted_links: Last sweep attempted link count or blank.
    Outputs:
        Observable row dictionary.
    """
    return {
        "chain": chain,
        "start": start,
        "sweep": sweep,
        "average_plaquette": "" if plaquette is None else plaquette,
        "acceptance_rate": acceptance_rate,
        "accepted_links": accepted_links,
        "attempted_links": attempted_links,
    }


def configuration_metadata(
    *,
    manifest: dict[str, object],
    chain: int,
    start: str,
    sweep: int,
    initial_sweep: int,
    plaquette: float | None,
) -> dict[str, object]:
    """Build metadata for one saved configuration.

    Inputs:
        manifest: Run-level manifest metadata.
        chain: Chain index.
        start: Start label.
        sweep: Full sweep number.
        initial_sweep: Full sweep number of the initial configuration.
        plaquette: Average plaquette or None.
    Outputs:
        Configuration metadata dictionary.
    """
    metadata = dict(manifest)
    metadata.update(
        {
            "chain": chain,
            "start": start,
            "sweep": sweep,
            "initial_sweep": initial_sweep,
        }
    )
    if plaquette is not None:
        metadata["average_plaquette"] = plaquette
    return metadata


def configuration_path(out_dir: Path, chain: int, start: str, sweep: int) -> Path:
    """Build a standard saved-configuration path.

    Inputs:
        out_dir: Run output directory.
        chain: Chain index.
        start: Start label.
        sweep: Full sweep number.
    Outputs:
        Path under configurations/.
    """
    filename = f"chain{chain:02d}_{start}_sweep{sweep:06d}.npz"
    return out_dir / "configurations" / filename


def maybe_save_configuration(
    *,
    out_dir: Path,
    links: np.ndarray,
    manifest: dict[str, object],
    chain: int,
    start: str,
    sweep: int,
    initial_sweep: int,
    segment_sweep: int,
    plaquette: float | None,
    save_every: int,
    save_after_sweep: int = 0,
) -> Path | None:
    """Save a configuration when interval and discard conditions are met.

    Inputs:
        out_dir: Run output directory.
        links: Gauge links U[site, direction].
        manifest: Run-level manifest metadata.
        chain: Chain index.
        start: Start label.
        sweep: Full sweep number.
        initial_sweep: Initial full sweep number.
        segment_sweep: Sweep number within this segment.
        plaquette: Average plaquette or None.
        save_every: Save interval; zero disables saving.
        save_after_sweep: Do not save segment sweeps at or below this value.
    Outputs:
        Saved path, or None.
    """
    if (
        save_every == 0
        or segment_sweep == 0
        or segment_sweep <= save_after_sweep
        or segment_sweep % save_every != 0
    ):
        return None

    path = configuration_path(out_dir, chain, start, sweep)
    save_configuration(
        path,
        links,
        configuration_metadata(
            manifest=manifest,
            chain=chain,
            start=start,
            sweep=sweep,
            initial_sweep=initial_sweep,
            plaquette=plaquette,
        ),
    )
    return path
