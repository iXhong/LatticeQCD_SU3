"""Single-chain update loop for production workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.observables import average_plaquette
from lattice_su3.run_config import MeasureConfig, SaveConfig, UpdateConfig
from lattice_su3.run_outputs import maybe_save_configuration, observable_row
from lattice_su3.update import UpdateStats, heatbath_sweep, overrelaxation_sweep


@dataclass
class ChainSpec:
    """Parameters for one chain segment.

    Inputs:
        shape: Lattice shape.
        beta: Wilson gauge coupling.
        sweeps: Number of segment sweeps to run.
        seed: Random seed.
        chain: Chain index.
        start: Start label.
        initial_sweep: Full sweep number of the initial links.
        update: Update settings.
        measure: Measurement settings.
        save: Save settings.
        save_after_sweep: Do not save or record segment sweeps at or below this.
        segment_sweep_offset: Completed production sweeps before this segment.
        record_initial: Whether to record the initial configuration observable.
    Outputs:
        Mutable chain specification.
    """

    shape: tuple[int, ...]
    beta: float
    sweeps: int
    seed: int
    chain: int
    start: str
    initial_sweep: int
    update: UpdateConfig
    measure: MeasureConfig
    save: SaveConfig
    save_after_sweep: int = 0
    segment_sweep_offset: int = 0
    record_initial: bool = False


@dataclass
class ChainResult:
    """Result from one chain segment.

    Inputs:
        chain: Chain index.
        rows: Observable CSV rows.
        saved_paths: Saved configuration paths.
        final_sweep: Final full sweep number.
    Outputs:
        Completed chain result.
    """

    chain: int
    rows: list[dict[str, object]]
    saved_paths: list[Path]
    final_sweep: int


class SweepRunner:
    """Run configured heatbath and overrelaxation sweeps.

    Inputs:
        update: Update settings.
        seed: Seed for JIT random state.
        rng: NumPy random generator for NumPy updates.
    Outputs:
        Stateful sweep runner.
    """

    def __init__(
        self,
        update: UpdateConfig,
        seed: int,
        rng: np.random.Generator,
    ) -> None:
        self.update = update
        self.seed = seed
        self.rng = rng
        self.jit_seeded = False

    def sweep(self, links: np.ndarray, geometry: LatticeGeometry, beta: float) -> UpdateStats:
        """Run one compound sweep.

        Inputs:
            links: Gauge links U[site, direction].
            geometry: Lattice geometry.
            beta: Wilson gauge coupling.
        Outputs:
            Combined update statistics.
        """
        if self.update.backend == "numpy":
            stats = heatbath_sweep(links, geometry, beta, self.rng)
            return self._run_overrelaxation(links, geometry, stats, "numpy")

        if self.update.backend == "jit_checkerboard":
            from lattice_su3.accelerated import heatbath_checkerboard_jit_sweep

            jit_seed = self.seed if not self.jit_seeded else None
            stats = heatbath_checkerboard_jit_sweep(links, geometry, beta, seed=jit_seed)
            self.jit_seeded = True
            return self._run_overrelaxation(links, geometry, stats, "jit_checkerboard")

        if self.update.backend != "jit":
            raise ValueError("backend must be 'jit', 'jit_checkerboard', or 'numpy'")

        from lattice_su3.accelerated import heatbath_jit_sweep

        jit_seed = self.seed if not self.jit_seeded else None
        stats = heatbath_jit_sweep(links, geometry, beta, seed=jit_seed)
        self.jit_seeded = True
        return self._run_overrelaxation(links, geometry, stats, "jit")

    def _run_overrelaxation(
        self,
        links: np.ndarray,
        geometry: LatticeGeometry,
        stats: UpdateStats,
        backend: str,
    ) -> UpdateStats:
        """Run configured overrelaxation sweeps after heatbath.

        Inputs:
            links: Gauge links U[site, direction].
            geometry: Lattice geometry.
            stats: Initial heatbath update statistics.
            backend: Backend family name.
        Outputs:
            Combined update statistics.
        """
        attempted = stats.attempted_links
        accepted = stats.accepted_links
        for _ in range(self.update.overrelaxation_sweeps):
            if backend == "numpy":
                overrelaxation_stats = overrelaxation_sweep(links, geometry, self.rng)
            elif backend == "jit_checkerboard":
                from lattice_su3.accelerated import overrelaxation_checkerboard_jit_sweep

                overrelaxation_stats = overrelaxation_checkerboard_jit_sweep(
                    links, geometry
                )
            else:
                from lattice_su3.accelerated import overrelaxation_jit_sweep

                overrelaxation_stats = overrelaxation_jit_sweep(links, geometry)
            attempted += overrelaxation_stats.attempted_links
            accepted += overrelaxation_stats.accepted_links
        return UpdateStats(attempted_links=attempted, accepted_links=accepted)


def run_chain_segment(
    *,
    links: np.ndarray,
    spec: ChainSpec,
    out_dir: Path,
    manifest: dict[str, object],
    progress_every: int = 0,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> ChainResult:
    """Run one chain segment and write saved configurations.

    Inputs:
        links: Mutable initial gauge links.
        spec: Chain segment specification.
        out_dir: Standard run output directory.
        manifest: Run-level manifest metadata for configuration metadata.
        progress_every: Segment sweep interval for optional progress reports.
        progress_callback: Optional callback receiving chain, local sweep,
            total sweeps, and saved configuration count.
    Outputs:
        Completed chain result with observable rows and saved paths.
    """
    geometry = LatticeGeometry(spec.shape)
    rng_seed, jit_seed = np.random.SeedSequence(spec.seed).spawn(2)
    rng = np.random.default_rng(rng_seed)
    runner = SweepRunner(spec.update, int(jit_seed.generate_state(1)[0]), rng)
    rows: list[dict[str, object]] = []
    saved_paths: list[Path] = []

    if spec.record_initial and spec.measure.plaquette_every > 0:
        plaquette = average_plaquette(links, geometry)
        rows.append(
            observable_row(
                chain=spec.chain,
                start=spec.start,
                sweep=spec.initial_sweep,
                plaquette=plaquette,
                acceptance_rate="",
                accepted_links="",
                attempted_links="",
            )
        )

    last_plaquette: float | None = None
    for local_sweep in range(1, spec.sweeps + 1):
        segment_sweep = spec.segment_sweep_offset + local_sweep
        sweep = spec.initial_sweep + local_sweep
        stats = runner.sweep(links, geometry, spec.beta)
        measured = (
            spec.measure.plaquette_every > 0
            and segment_sweep > spec.save_after_sweep
            and segment_sweep % spec.measure.plaquette_every == 0
        )
        if measured:
            last_plaquette = average_plaquette(links, geometry)
            rows.append(
                observable_row(
                    chain=spec.chain,
                    start=spec.start,
                    sweep=sweep,
                    plaquette=last_plaquette,
                    acceptance_rate=stats.acceptance_rate,
                    accepted_links=stats.accepted_links,
                    attempted_links=stats.attempted_links,
                )
            )

        saved_path = maybe_save_configuration(
            out_dir=out_dir,
            links=links,
            manifest=manifest,
            chain=spec.chain,
            start=spec.start,
            sweep=sweep,
            initial_sweep=spec.initial_sweep,
            segment_sweep=segment_sweep,
            plaquette=last_plaquette if measured else None,
            save_every=spec.save.config_every,
            save_after_sweep=spec.save_after_sweep,
        )
        if saved_path is not None:
            saved_paths.append(saved_path)

        if (
            progress_callback is not None
            and progress_every > 0
            and (local_sweep % progress_every == 0 or local_sweep == spec.sweeps)
        ):
            progress_callback(
                spec.chain,
                local_sweep,
                spec.sweeps,
                len(saved_paths),
            )

    return ChainResult(
        chain=spec.chain,
        rows=rows,
        saved_paths=saved_paths,
        final_sweep=spec.initial_sweep + spec.sweeps,
    )
