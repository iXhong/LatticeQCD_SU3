"""
Summarize thermalization histories from a run observable CSV.

This script reads results/runs/<run_name>/observables.csv produced by
scripts/run_chain.py and summarizes the tail plaquette distribution for each
chain. For two-chain cold/hot runs, compare the reported tail means.

Usage:
    Edit RUN_NAME and TAIL_SWEEPS below, then run:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/analyze_thermalization.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SHAPE = (16, 16, 16, 6)
BETA = 5.7
ALGORITHM = "heatbath"
BACKEND = "jit"
STARTS = ("cold", "hot")
SWEEPS = 300
SEED = 12345
RUN_NAME = ""
TAIL_SWEEPS = 50

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
    return (
        f"{ALGORITHM}_{BACKEND}_{starts_label}_{shape_label(SHAPE)}_"
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


def output_summary_path() -> Path:
    """Build the output path for thermalization summary data.

    Inputs:
        None.
    Outputs:
        CSV path for thermalization summary rows.
    """
    return input_observables_path().with_name("thermalization_summary.csv")


def load_observables(path: Path) -> np.ndarray:
    """Load the run observable table.

    Inputs:
        path: Observable CSV path produced by scripts/run_chain.py.
    Outputs:
        Structured NumPy array of observable rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"observable history not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.asarray([data])

    required_fields = {"chain", "start", "sweep", "average_plaquette"}
    missing_fields = required_fields.difference(data.dtype.names)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"observable history is missing required fields: {missing}")
    return data


def summarize_chain(data: np.ndarray, chain: int, tail_sweeps: int) -> dict[str, object]:
    """Summarize the tail of one chain.

    Inputs:
        data: Structured observable rows.
        chain: Chain index to summarize.
        tail_sweeps: Number of final sweeps to include.
    Outputs:
        Summary dictionary for one chain.
    """
    if tail_sweeps <= 0:
        raise ValueError("TAIL_SWEEPS must be positive")

    chain_rows = data[np.asarray(data["chain"], dtype=np.int64) == chain]
    if len(chain_rows) == 0:
        raise ValueError(f"chain {chain} is not present in observable history")

    sweeps = np.asarray(chain_rows["sweep"], dtype=np.int64)
    plaquettes = np.asarray(chain_rows["average_plaquette"], dtype=np.float64)
    start = str(chain_rows["start"][0])
    cutoff = max(int(sweeps[-1]) - tail_sweeps + 1, int(sweeps[0]))
    tail = plaquettes[sweeps >= cutoff]
    return {
        "chain": chain,
        "start": start,
        "tail_start_sweep": cutoff,
        "tail_end_sweep": int(sweeps[-1]),
        "tail_measurements": len(tail),
        "tail_mean_plaquette": float(np.mean(tail)),
        "tail_std_plaquette": float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0,
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Write thermalization summary rows.

    Inputs:
        path: Output CSV path.
        rows: Summary row dictionaries.
    Outputs:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chain",
        "start",
        "tail_start_sweep",
        "tail_end_sweep",
        "tail_measurements",
        "tail_mean_plaquette",
        "tail_std_plaquette",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Summarize thermalization tails for all chains in one run.

    Inputs:
        None.
    Outputs:
        None.
    """
    observables_path = input_observables_path()
    data = load_observables(observables_path)
    chains = sorted(set(np.asarray(data["chain"], dtype=np.int64)))
    rows = [summarize_chain(data, int(chain), TAIL_SWEEPS) for chain in chains]

    summary_path = output_summary_path()
    write_summary(summary_path, rows)

    print(f"Loaded observables: {observables_path}")
    print(f"tail sweeps: {TAIL_SWEEPS}")
    for row in rows:
        print(
            f"chain {row['chain']} ({row['start']}): "
            f"tail mean={row['tail_mean_plaquette']:.8f}, "
            f"std={row['tail_std_plaquette']:.8f}"
        )
    if len(rows) == 2:
        diff = abs(rows[0]["tail_mean_plaquette"] - rows[1]["tail_mean_plaquette"])
        print(f"two-chain tail mean difference: {diff:.8f}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
