"""
Run one resumable production chain for a server scheduler task.

This script reads the existing ensemble TOML format but runs exactly one chain.
It writes under ``<results-root>/<run.name>/chains/chainNNN`` so independent job
array tasks never share mutable files. A saved configuration is a restart
checkpoint; ``--resume`` continues from the greatest saved sweep while preserving
the original production discard, measurement, and save cadence.

Run from the repository root with:

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/workflows/run_server_chain.py \
        configs/ensemble_16x16x16x6.toml --chain 0 \
        --results-root "$SCRATCH/lattice_su3/runs" --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3.chain import ChainSpec, run_chain_segment  # noqa: E402
from lattice_su3.configuration import latest_configuration_path, load_start  # noqa: E402
from lattice_su3.geometry import LatticeGeometry  # noqa: E402
from lattice_su3.run_config import EnsembleConfig, load_ensemble_config  # noqa: E402
from lattice_su3.run_outputs import (  # noqa: E402
    OBSERVABLE_FIELDNAMES,
    manifest_data,
    write_manifest,
)


def default_results_root() -> Path:
    """Select the server run root from the environment or repository.

    Inputs:
        None.
    Outputs:
        Configured run root directory.
    """
    configured = os.environ.get("LATTICE_SU3_RESULTS_ROOT")
    return Path(configured) if configured else ROOT / "results" / "runs"


def resolve_source_config(path: Path) -> Path:
    """Resolve an absolute or repository-relative source configuration.

    Inputs:
        path: Source configuration path from the ensemble TOML.
    Outputs:
        Existing source configuration path.
    """
    selected = path if path.is_absolute() else ROOT / path
    if not selected.exists():
        raise FileNotFoundError(f"source configuration not found: {selected}")
    return selected.resolve()


def chain_directory(results_root: Path, run_name: str, chain: int) -> Path:
    """Build the scheduler-safe output directory for one chain.

    Inputs:
        results_root: Root containing named server runs.
        run_name: Ensemble run name.
        chain: Non-negative chain index.
    Outputs:
        Per-chain output directory.
    """
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run name must be a single non-empty directory name")
    if chain < 0:
        raise ValueError("chain must be non-negative")
    return results_root / run_name / "chains" / f"chain{chain:03d}"


def build_chain_manifest(
    config: EnsembleConfig, source_config: Path, chain: int
) -> dict[str, object]:
    """Build immutable and progress metadata for one server chain.

    Inputs:
        config: Ensemble workflow configuration.
        source_config: Resolved source configuration path.
        chain: Chain index.
    Outputs:
        JSON-serializable per-chain manifest.
    """
    manifest = manifest_data(
        run_name=config.name,
        shape=config.shape,
        beta=config.beta,
        sweeps=config.sweeps_per_chain,
        seed=config.seed_base + chain,
        starts=("load",),
        update=config.update,
        measure=config.measure,
        save=config.save,
        workflow="run_server_chain",
        chains=config.chains,
        source_config=str(source_config),
        discard_sweeps=config.discard_sweeps,
    )
    manifest.update(
        {
            "chain": chain,
            "status": "pending",
            "completed_segment_sweeps": 0,
            "elapsed_seconds": 0.0,
        }
    )
    return manifest


def validate_resume_manifest(
    existing: dict[str, object], expected: dict[str, object]
) -> None:
    """Reject resume when immutable chain settings have changed.

    Inputs:
        existing: Manifest stored by the earlier server task.
        expected: Manifest built from the current TOML and chain index.
    Outputs:
        None.
    """
    immutable = (
        "run_name",
        "shape",
        "beta",
        "sweeps",
        "seed",
        "algorithm",
        "backend",
        "overrelaxation_sweeps",
        "save_config_every",
        "discard_sweeps",
        "chain",
        "source_configuration",
    )
    changed = [key for key in immutable if existing.get(key) != expected.get(key)]
    if changed:
        raise ValueError(f"cannot resume after settings changed: {changed}")


def load_observable_rows(path: Path) -> list[dict[str, object]]:
    """Load existing per-chain observable rows for resume.

    Inputs:
        path: Per-chain observable CSV path.
    Outputs:
        Existing rows as dictionaries.
    """
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_observable_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Atomically replace the per-chain observable CSV.

    Inputs:
        path: Per-chain observable CSV path.
        rows: Complete rows to write.
    Outputs:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVABLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def segment_seed(base_seed: int, completed_sweeps: int) -> int:
    """Derive a deterministic random seed for one restart segment.

    Inputs:
        base_seed: Chain-specific seed from the ensemble configuration.
        completed_sweeps: Production sweeps completed before the segment.
    Outputs:
        Deterministic unsigned 32-bit compatible seed.
    """
    sequence = np.random.SeedSequence([base_seed, completed_sweeps])
    return int(sequence.generate_state(1)[0])


def run_server_chain(
    config: EnsembleConfig,
    chain: int,
    results_root: Path,
    resume: bool,
) -> Path:
    """Run or resume one production chain in checkpoint-sized segments.

    Inputs:
        config: Ensemble workflow configuration.
        chain: Chain index selected by the scheduler task.
        results_root: Root containing named server runs.
        resume: Whether an existing compatible chain may be continued.
    Outputs:
        Completed per-chain output directory.
    """
    if chain < 0 or chain >= config.chains:
        raise ValueError(f"chain must be between 0 and {config.chains - 1}")
    if config.save.config_every <= 0:
        raise ValueError("server runs require save.config_every > 0 for checkpoints")
    if config.sweeps_per_chain % config.save.config_every != 0:
        raise ValueError("sweeps_per_chain must be divisible by save.config_every")

    source_config = resolve_source_config(config.source_config)
    geometry = LatticeGeometry(config.shape)
    source_links, source_metadata = load_start(source_config, geometry)
    source_sweep = int(source_metadata.get("sweep", 0))
    out_dir = chain_directory(results_root, config.name, chain)
    manifest_path = out_dir / "manifest.json"
    observables_path = out_dir / "observables.csv"
    expected_manifest = build_chain_manifest(config, source_config, chain)

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"chain already exists; use --resume: {out_dir}")
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        validate_resume_manifest(manifest, expected_manifest)
        try:
            checkpoint = latest_configuration_path(
                out_dir / "configurations", chain=chain
            )
        except FileNotFoundError:
            links = source_links
            completed = 0
        else:
            links, checkpoint_metadata = load_start(checkpoint, geometry)
            checkpoint_sweep = int(checkpoint_metadata["sweep"])
            completed = checkpoint_sweep - source_sweep
        if completed < 0 or completed > config.sweeps_per_chain:
            raise ValueError("checkpoint sweep is outside the configured production range")
        rows = load_observable_rows(observables_path)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        manifest = expected_manifest
        links = source_links
        completed = 0
        rows = []

    elapsed_total = float(manifest.get("elapsed_seconds", 0.0))
    manifest.update(status="running", completed_segment_sweeps=completed)
    write_manifest(manifest_path, manifest)

    while completed < config.sweeps_per_chain:
        segment_length = min(
            config.save.config_every,
            config.sweeps_per_chain - completed,
        )
        initial_sweep = source_sweep + completed
        spec = ChainSpec(
            shape=config.shape,
            beta=config.beta,
            sweeps=segment_length,
            seed=segment_seed(config.seed_base + chain, completed),
            chain=chain,
            start="load",
            initial_sweep=initial_sweep,
            update=config.update,
            measure=config.measure,
            save=config.save,
            save_after_sweep=config.discard_sweeps,
            segment_sweep_offset=completed,
            record_initial=False,
        )
        started = perf_counter()
        result = run_chain_segment(
            links=links,
            spec=spec,
            out_dir=out_dir,
            manifest=manifest,
        )
        elapsed_total += perf_counter() - started
        completed += segment_length
        rows.extend(result.rows)
        write_observable_rows(observables_path, rows)
        manifest.update(
            status="running",
            completed_segment_sweeps=completed,
            final_sweep=source_sweep + completed,
            elapsed_seconds=elapsed_total,
        )
        write_manifest(manifest_path, manifest)

    manifest["status"] = "complete"
    write_manifest(manifest_path, manifest)
    return out_dir


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the server-chain command-line parser.

    Inputs:
        None.
    Outputs:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="ensemble TOML configuration")
    parser.add_argument("--chain", required=True, type=int)
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one scheduler-selected production chain.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = build_argument_parser().parse_args(argv)
    config = load_ensemble_config(args.config)
    out_dir = run_server_chain(config, args.chain, args.results_root, args.resume)
    print(f"Server chain complete: {out_dir}")


if __name__ == "__main__":
    main()
