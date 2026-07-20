import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from lattice_su3 import LatticeGeometry, cold_start, save_configuration
from lattice_su3.run_config import load_ensemble_config


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "workflows"
    / "run_server_chain.py"
)
SPEC = importlib.util.spec_from_file_location("run_server_chain", SCRIPT_PATH)
server_chain = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = server_chain
SPEC.loader.exec_module(server_chain)


def write_server_fixture(tmp_path: Path, sweeps: int = 4) -> Path:
    """Write a small source configuration and ensemble TOML.

    Inputs:
        tmp_path: Temporary test directory.
        sweeps: Production sweeps per chain.
    Outputs:
        Ensemble TOML path.
    """
    source = tmp_path / "source.npz"
    geometry = LatticeGeometry((2, 2, 2, 2))
    save_configuration(
        source,
        cold_start(geometry),
        {"shape": geometry.shape, "sweep": 10, "beta": 5.7},
    )
    config = tmp_path / "ensemble.toml"
    config.write_text(
        f"""
[run]
name = "server_test"
shape = [2, 2, 2, 2]
beta = 5.7

[source]
config = "{source}"

[ensemble]
chains = 2
sweeps_per_chain = {sweeps}
discard_sweeps = 1
seed_base = 700
parallel = 1

[update]
backend = "numpy"
overrelaxation_sweeps = 0

[measure]
plaquette_every = 1

[save]
config_every = 2
overwrite = false
""",
        encoding="utf-8",
    )
    return config


def test_server_chain_writes_isolated_resumable_artifacts(tmp_path):
    config = load_ensemble_config(write_server_fixture(tmp_path))
    results_root = tmp_path / "server-results"

    out_dir = server_chain.run_server_chain(config, 1, results_root, resume=True)

    assert out_dir == results_root / "server_test" / "chains" / "chain001"
    with open(out_dir / "manifest.json") as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "complete"
    assert manifest["completed_segment_sweeps"] == 4
    assert manifest["final_sweep"] == 14
    paths = sorted((out_dir / "configurations").glob("*.npz"))
    assert [path.name for path in paths] == [
        "chain01_load_sweep000012.npz",
        "chain01_load_sweep000014.npz",
    ]
    with open(out_dir / "observables.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["sweep"]) for row in rows] == [12, 13, 14]

    resumed = server_chain.run_server_chain(config, 1, results_root, resume=True)
    assert resumed == out_dir
    with open(out_dir / "observables.csv", newline="") as handle:
        resumed_rows = list(csv.DictReader(handle))
    assert resumed_rows == rows


def test_server_chain_refuses_existing_run_without_resume(tmp_path):
    config = load_ensemble_config(write_server_fixture(tmp_path))
    results_root = tmp_path / "server-results"
    server_chain.run_server_chain(config, 0, results_root, resume=False)

    with pytest.raises(FileExistsError, match="--resume"):
        server_chain.run_server_chain(config, 0, results_root, resume=False)


def test_server_chain_requires_target_sweeps_to_align_with_checkpoints(tmp_path):
    config = load_ensemble_config(write_server_fixture(tmp_path, sweeps=3))

    with pytest.raises(ValueError, match="divisible"):
        server_chain.run_server_chain(config, 0, tmp_path / "results", resume=False)


def test_segment_seed_depends_on_chain_progress():
    assert server_chain.segment_seed(123, 0) == server_chain.segment_seed(123, 0)
    assert server_chain.segment_seed(123, 0) != server_chain.segment_seed(123, 20)
