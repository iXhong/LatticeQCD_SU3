"""
Generate a production ensemble from a thermalized source configuration.

This script reads one TOML configuration, starts several independent chains from
the same source configuration with different seeds, and writes a single standard
run directory containing all chains. It keeps the output layout compatible with
scripts/analysis/auto_correlation.py and scripts/analysis/measure_polyakov_correlators.py.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/workflows/generate_ensemble.py \
        configs/ensemble_16x16x16x6.toml
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from time import perf_counter
from threading import Lock


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3.chain import ChainResult, ChainSpec, run_chain_segment  # noqa: E402
from lattice_su3.configuration import load_start  # noqa: E402
from lattice_su3.geometry import LatticeGeometry  # noqa: E402
from lattice_su3.run_config import EnsembleConfig, load_ensemble_config  # noqa: E402
from lattice_su3.run_outputs import (  # noqa: E402
    manifest_data,
    open_observable_writer,
    prepare_run_directory,
    run_directory,
    write_manifest,
)


PRINT_LOCK = Lock()


def resolve_source_config(path: Path) -> Path:
    """Resolve a source configuration path.

    Inputs:
        path: Absolute path or repository-relative path.
    Outputs:
        Resolved path.
    """
    selected = path if path.is_absolute() else ROOT / path
    if not selected.exists():
        raise FileNotFoundError(f"source configuration not found: {selected}")
    return selected


def source_path_label(path: Path) -> str:
    """Format a source path for manifest metadata.

    Inputs:
        path: Source configuration path.
    Outputs:
        Repository-relative path when possible, otherwise absolute path.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def run_one_production_chain(
    *,
    config: EnsembleConfig,
    source_config: Path,
    out_dir: Path,
    manifest: dict[str, object],
    chain: int,
    progress_every: int,
) -> ChainResult:
    """Run one production chain from the configured source.

    Inputs:
        config: Ensemble workflow configuration.
        source_config: Resolved source configuration path.
        out_dir: Standard run output directory.
        manifest: Run-level manifest metadata.
        chain: Chain index.
        progress_every: Segment sweep interval for progress output.
    Outputs:
        Completed chain result.
    """
    geometry = LatticeGeometry(config.shape)
    links, metadata = load_start(source_config, geometry)
    initial_sweep = int(metadata.get("sweep", 0))
    spec = ChainSpec(
        shape=config.shape,
        beta=config.beta,
        sweeps=config.sweeps_per_chain,
        seed=config.seed_base + chain,
        chain=chain,
        start="load",
        initial_sweep=initial_sweep,
        update=config.update,
        measure=config.measure,
        save=config.save,
        save_after_sweep=config.discard_sweeps,
        record_initial=False,
    )

    def report_progress(
        chain: int,
        local_sweep: int,
        total_sweeps: int,
        saved_count: int,
    ) -> None:
        """Print one compact progress line for a production chain.

        Inputs:
            chain: Chain index.
            local_sweep: Completed segment sweeps.
            total_sweeps: Total segment sweeps for this chain.
            saved_count: Number of configurations saved so far.
        Outputs:
            None.
        """
        percent = 100.0 * local_sweep / total_sweeps if total_sweeps else 100.0
        with PRINT_LOCK:
            print(
                f"[chain {chain:02d}] sweep {local_sweep}/{total_sweeps} "
                f"({percent:5.1f}%), saved {saved_count}",
                flush=True,
            )

    return run_chain_segment(
        links=links,
        spec=spec,
        out_dir=out_dir,
        manifest=manifest,
        progress_every=progress_every,
        progress_callback=report_progress,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the configured ensemble generation workflow.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: generate_ensemble.py CONFIG.toml")

    config = load_ensemble_config(args[0])
    source_config = resolve_source_config(config.source_config)
    out_dir = run_directory(ROOT, config.name)
    prepare_run_directory(out_dir, config.save.overwrite)
    manifest = manifest_data(
        run_name=config.name,
        shape=config.shape,
        beta=config.beta,
        sweeps=config.sweeps_per_chain,
        seed=config.seed_base,
        starts=("load",),
        update=config.update,
        measure=config.measure,
        save=config.save,
        workflow="generate_ensemble",
        chains=config.chains,
        source_config=source_path_label(source_config),
        discard_sweeps=config.discard_sweeps,
    )
    write_manifest(out_dir / "manifest.json", manifest)

    start_time = perf_counter()
    parallel = min(config.parallel, config.chains)
    progress_every = max(100, config.sweeps_per_chain // 10)
    results: list[ChainResult] = []
    print(
        f"Starting ensemble run: {config.name}, chains={config.chains}, "
        f"parallel={parallel}, progress_every={progress_every} sweeps",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(
                run_one_production_chain,
                config=config,
                source_config=source_config,
                out_dir=out_dir,
                manifest=manifest,
                chain=chain,
                progress_every=progress_every,
            ): chain
            for chain in range(config.chains)
        }
        for future in as_completed(futures):
            chain = futures[future]
            result = future.result()
            print(
                f"[chain {chain:02d}] final sweep {result.final_sweep}, "
                f"saved {len(result.saved_paths)} configurations"
            )
            results.append(result)

    elapsed = perf_counter() - start_time
    manifest["elapsed_seconds"] = elapsed
    write_manifest(out_dir / "manifest.json", manifest)

    rows = [
        row
        for result in sorted(results, key=lambda item: item.chain)
        for row in result.rows
    ]
    handle, writer = open_observable_writer(out_dir / "observables.csv")
    try:
        writer.writerows(rows)
    finally:
        handle.close()

    saved_count = sum(len(result.saved_paths) for result in results)
    print(f"Ensemble run: {config.name}")
    print(f"Chains: {config.chains}, parallel={parallel}")
    print(f"Saved configurations: {saved_count}")
    print(f"Run directory: {out_dir}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
