"""
Plot thermalization histories from a run observable CSV.

This script reads results/runs/<run_name>/observables.csv produced by
scripts/run_chain.py and plots average plaquette versus sweep for cold and hot
chains. Use it to visually inspect whether the two starts approach the same
equilibrium region.

Usage:
    Edit RUN_NAME below, then run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analyze_thermalization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SHAPE = (16, 16, 16, 6)
BETA = 5.7
ALGORITHM = "heatbath"
BACKEND = "jit"
OVERRELAXATION_SWEEPS = 0
STARTS = ("cold", "hot")
SWEEPS = 300
SEED = 12345
RUN_NAME = ""

ROOT = Path(__file__).resolve().parents[1]


def shape_label(shape: tuple[int, ...]) -> str:
    """Format a lattice shape for filenames.

    Inputs:
        shape: Lattice size in each direction.
    Outputs:
        Shape label with dimensions joined by x.
    """
    return "x".join(str(length) for length in shape)


def run_label() -> str:
    """Build the default run directory label.

    Inputs:
        None.
    Outputs:
        Stable label matching scripts/run_chain.py defaults.
    """
    if RUN_NAME:
        return RUN_NAME

    starts_label = "-".join(STARTS)
    algorithm_label = ALGORITHM
    if OVERRELAXATION_SWEEPS > 0:
        algorithm_label = f"{ALGORITHM}_or{OVERRELAXATION_SWEEPS}"
    return (
        f"{algorithm_label}_{BACKEND}_{starts_label}_{shape_label(SHAPE)}_"
        f"beta{BETA}_{SWEEPS}sweeps_seed{SEED}"
    )


def input_observables_path() -> Path:
    """Build the input path for run observables.

    Inputs:
        None.
    Outputs:
        CSV path produced by scripts/run_chain.py.
    """
    return ROOT / "results" / "runs" / run_label() / "observables.csv"


def output_plot_path() -> Path:
    """Build the output path for the thermalization plot.

    Inputs:
        None.
    Outputs:
        PNG path for the plaquette history plot.
    """
    return input_observables_path().with_name("thermalization_plaquette.png")


def load_plaquette_histories(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load plaquette histories grouped by start type.

    Inputs:
        path: Observable CSV path produced by scripts/run_chain.py.
    Outputs:
        Mapping from start name to sweep and plaquette arrays.
    """
    if not path.exists():
        raise FileNotFoundError(f"observable history not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.asarray([data])

    required_fields = {"start", "sweep", "average_plaquette"}
    missing_fields = required_fields.difference(data.dtype.names)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"observable history is missing required fields: {missing}")

    histories = {}
    for start in ("cold", "hot"):
        mask = np.asarray(data["start"]) == start
        histories[start] = {
            "sweep": np.asarray(data["sweep"][mask], dtype=np.int64),
            "plaquette": np.asarray(data["average_plaquette"][mask], dtype=np.float64),
        }
    return histories


def plot_plaquette_histories(
    histories: dict[str, dict[str, np.ndarray]],
    observables_path: Path,
    output_path: Path,
) -> Path:
    """Plot plaquette histories for cold and hot starts.

    Inputs:
        histories: Mapping from start name to sweep and plaquette arrays.
        observables_path: Source observable CSV path.
        output_path: PNG path to write.
    Outputs:
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for start, color, marker, label in [
        ("cold", "C0", "+", "Cold start"),
        ("hot", "C1", "x", "Hot start"),
    ]:
        history = histories[start]
        if len(history["sweep"]) == 0:
            continue
        ax.scatter(
            history["sweep"],
            history["plaquette"],
            color=color,
            marker=marker,
            label=label,
            s=30,
            alpha=0.7,
        )

    ax.set_xlabel("Sweep")
    ax.set_ylabel("Average plaquette")
    ax.set_title(f"Thermalization check - {observables_path.parent.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    """Plot cold/hot plaquette histories for one run.

    Inputs:
        None.
    Outputs:
        None.
    """
    observables_path = input_observables_path()
    histories = load_plaquette_histories(observables_path)
    plot_path = plot_plaquette_histories(
        histories,
        observables_path,
        output_plot_path(),
    )

    print(f"Loaded observables: {observables_path}")
    for start in ("cold", "hot"):
        history = histories[start]
        if len(history["sweep"]) == 0:
            print(f"  [{start}] no measurements")
            continue
        print(
            f"  [{start}] {len(history['sweep'])} measurements, "
            f"final plaquette: {history['plaquette'][-1]:.6f}"
        )
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
