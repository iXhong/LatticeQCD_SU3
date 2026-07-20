"""
Thermalize one SU(3) gauge chain from a TOML configuration.

This script only performs thermalization and writes standard run artifacts under
results/runs/<run.name>/. It expects a TOML file with [run], [update], [measure],
and [save] tables. The output manifest, observables.csv, and configurations/
layout are compatible with the standard analysis scripts.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/workflows/thermalize.py \
        configs/thermalize_16x16x16x6.toml
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3.chain import ChainSpec, run_chain_segment  # noqa: E402
from lattice_su3.configuration import cold_start, hot_start  # noqa: E402
from lattice_su3.geometry import LatticeGeometry  # noqa: E402
from lattice_su3.run_config import load_thermalize_config  # noqa: E402
from lattice_su3.run_outputs import (  # noqa: E402
    manifest_data,
    open_observable_writer,
    prepare_run_directory,
    run_directory,
    write_manifest,
)


def main(argv: list[str] | None = None) -> None:
    """Run one thermalization workflow.

    Inputs:
        argv: Optional command-line arguments excluding executable name.
    Outputs:
        None.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: thermalize.py CONFIG.toml")

    config = load_thermalize_config(args[0])
    out_dir = run_directory(ROOT, config.name)
    prepare_run_directory(out_dir, config.save.overwrite)
    manifest = manifest_data(
        run_name=config.name,
        shape=config.shape,
        beta=config.beta,
        sweeps=config.sweeps,
        seed=config.seed,
        starts=(config.start,),
        update=config.update,
        measure=config.measure,
        save=config.save,
        workflow="thermalize",
    )
    write_manifest(out_dir / "manifest.json", manifest)

    geometry = LatticeGeometry(config.shape)
    rng = np.random.default_rng(config.seed)
    links = hot_start(geometry, rng) if config.start == "hot" else cold_start(geometry)
    spec = ChainSpec(
        shape=config.shape,
        beta=config.beta,
        sweeps=config.sweeps,
        seed=config.seed,
        chain=0,
        start=config.start,
        initial_sweep=0,
        update=config.update,
        measure=config.measure,
        save=config.save,
        record_initial=True,
    )

    start_time = perf_counter()
    result = run_chain_segment(
        links=links,
        spec=spec,
        out_dir=out_dir,
        manifest=manifest,
    )
    elapsed = perf_counter() - start_time
    manifest["elapsed_seconds"] = elapsed
    write_manifest(out_dir / "manifest.json", manifest)

    handle, writer = open_observable_writer(out_dir / "observables.csv")
    try:
        writer.writerows(result.rows)
    finally:
        handle.close()

    print(f"Thermalization run: {config.name}")
    print(f"Final sweep: {result.final_sweep}")
    print(f"Saved configurations: {len(result.saved_paths)}")
    print(f"Run directory: {out_dir}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
