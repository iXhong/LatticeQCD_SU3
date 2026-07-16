import importlib.util
from pathlib import Path
import sys

import numpy as np

from lattice_su3 import LatticeGeometry, cold_start, load_configuration


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_chain.py"
SPEC = importlib.util.spec_from_file_location("run_chain", SCRIPT_PATH)
run_chain = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = run_chain
SPEC.loader.exec_module(run_chain)


def test_run_label_and_output_directory_are_deterministic(monkeypatch):
    monkeypatch.setattr(run_chain, "STARTS", ("hot",))
    monkeypatch.setattr(run_chain, "RUN_NAME", "")
    path = run_chain.output_directory()

    assert run_chain.shape_label((16, 16, 16, 6)) == "16x16x16x6"
    assert path.name == "heatbath_jit_hot_16x16x16x6_beta5.7_300sweeps_seed12345"
    assert path.parent.name == "runs"


def test_manifest_data_contains_run_parameters(monkeypatch):
    monkeypatch.setattr(run_chain, "STARTS", ("hot",))
    monkeypatch.setattr(run_chain, "RUN_NAME", "")
    manifest = run_chain.manifest_data()

    assert manifest["shape"] == [16, 16, 16, 6]
    assert manifest["beta"] == 5.7
    assert manifest["sweeps"] == 300
    assert manifest["measure_every"] == 1
    assert manifest["save_config_every"] == 0
    assert manifest["algorithm"] == "heatbath"
    assert manifest["backend"] == "jit"
    assert manifest["starts"] == ["hot"]


def test_observable_row_contains_expected_fields():
    row = run_chain.observable_row(
        chain=1,
        start="cold",
        sweep=20,
        plaquette=0.55,
        acceptance_rate=1.0,
        accepted_links=128,
        attempted_links=128,
    )

    assert row == {
        "chain": 1,
        "start": "cold",
        "sweep": 20,
        "average_plaquette": 0.55,
        "acceptance_rate": 1.0,
        "accepted_links": 128,
        "attempted_links": 128,
    }


def test_maybe_save_configuration_is_disabled_by_default(tmp_path):
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    path = run_chain.maybe_save_configuration(
        tmp_path,
        links,
        chain=0,
        start="cold",
        sweep=10,
        plaquette=1.0,
    )

    assert path is None
    assert not (tmp_path / "configurations").exists()


def test_maybe_save_configuration_uses_configured_interval(tmp_path, monkeypatch):
    monkeypatch.setattr(run_chain, "SAVE_CONFIG_EVERY", 5)
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    skipped_path = run_chain.maybe_save_configuration(
        tmp_path,
        links,
        chain=0,
        start="cold",
        sweep=4,
        plaquette=1.0,
    )
    saved_path = run_chain.maybe_save_configuration(
        tmp_path,
        links,
        chain=0,
        start="cold",
        sweep=5,
        plaquette=1.0,
    )

    assert skipped_path is None
    assert saved_path == tmp_path / "configurations" / "chain00_cold_sweep000005.npz"
    saved_links, metadata = load_configuration(saved_path)
    assert np.allclose(saved_links, links, atol=1e-12)
    assert metadata["chain"] == 0
    assert metadata["start"] == "cold"
    assert metadata["sweep"] == 5
    assert metadata["save_config_every"] == 5
    assert metadata["average_plaquette"] == 1.0


def test_maybe_save_configuration_can_skip_plaquette_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(run_chain, "SAVE_CONFIG_EVERY", 5)
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    saved_path = run_chain.maybe_save_configuration(
        tmp_path,
        links,
        chain=0,
        start="cold",
        sweep=5,
        plaquette=None,
    )

    assert saved_path == tmp_path / "configurations" / "chain00_cold_sweep000005.npz"
    _, metadata = load_configuration(saved_path)
    assert "average_plaquette" not in metadata
