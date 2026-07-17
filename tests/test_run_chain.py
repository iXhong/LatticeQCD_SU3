import csv
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lattice_su3 import (
    LatticeGeometry,
    cold_start,
    load_configuration,
    save_configuration,
)


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
    assert manifest["measure_plaquette"] is True
    assert manifest["measure_polyakov"] is False
    assert manifest["save_config_every"] == 0
    assert manifest["algorithm"] == "heatbath"
    assert manifest["backend"] == "jit"
    assert manifest["overrelaxation_sweeps"] == 0
    assert manifest["starts"] == ["hot"]


def test_run_label_records_nonzero_overrelaxation_sweeps(monkeypatch):
    monkeypatch.setattr(run_chain, "STARTS", ("hot",))
    monkeypatch.setattr(run_chain, "RUN_NAME", "")
    monkeypatch.setattr(run_chain, "OVERRELAXATION_SWEEPS", 2)

    assert run_chain.output_directory().name == (
        "heatbath_or2_jit_hot_16x16x16x6_beta5.7_300sweeps_seed12345"
    )


def test_apply_arguments_overrides_chain_parameters():
    original_values = {
        name: getattr(run_chain, name)
        for name in [
            "SHAPE",
            "RUN_NAME",
            "SEED",
            "SWEEPS",
            "MEASURE_EVERY",
            "SAVE_CONFIG_EVERY",
            "BACKEND",
            "OVERRELAXATION_SWEEPS",
            "STARTS",
            "MEASURE_POLYAKOV",
        ]
    }
    parser = run_chain.build_argument_parser()
    args = parser.parse_args(
        [
            "--shape",
            "4x4x4x6",
            "--run-name",
            "cli_run",
            "--seed",
            "123",
            "--sweeps",
            "20",
            "--measure-every",
            "5",
            "--save-config-every",
            "10",
            "--backend",
            "jit_checkerboard",
            "--overrelaxation-sweeps",
            "2",
            "--start",
            "hot",
            "--no-measure-polyakov",
        ]
    )

    try:
        run_chain.apply_arguments(args)

        assert run_chain.SHAPE == (4, 4, 4, 6)
        assert run_chain.RUN_NAME == "cli_run"
        assert run_chain.SEED == 123
        assert run_chain.SWEEPS == 20
        assert run_chain.MEASURE_EVERY == 5
        assert run_chain.SAVE_CONFIG_EVERY == 10
        assert run_chain.BACKEND == "jit_checkerboard"
        assert run_chain.OVERRELAXATION_SWEEPS == 2
        assert run_chain.STARTS == ("hot",)
        assert run_chain.MEASURE_POLYAKOV is False
    finally:
        for name, value in original_values.items():
            setattr(run_chain, name, value)


def test_validate_parameters_requires_heatbath_for_overrelaxation(monkeypatch):
    monkeypatch.setattr(run_chain, "ALGORITHM", "metropolis")
    monkeypatch.setattr(run_chain, "BACKEND", "numpy")
    monkeypatch.setattr(run_chain, "OVERRELAXATION_SWEEPS", 1)

    with pytest.raises(ValueError, match="requires ALGORITHM='heatbath'"):
        run_chain.validate_parameters()


def test_sweep_runner_combines_heatbath_and_overrelaxation_stats(monkeypatch):
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    runner = run_chain.SweepRunner(
        "heatbath",
        "numpy",
        2,
        123,
        np.random.default_rng(123),
    )

    stats = runner.sweep(links, geometry, beta=5.7, step_size=0.4)

    links_per_sweep = geometry.volume * geometry.ndim
    assert stats.attempted_links == 3 * links_per_sweep
    assert stats.accepted_links == stats.attempted_links
    assert stats.acceptance_rate == 1.0


def test_validate_parameters_requires_source_run_for_load(monkeypatch):
    monkeypatch.setattr(run_chain, "STARTS", ("load",))
    monkeypatch.setattr(run_chain, "SOURCE_RUN_NAME", "")

    with pytest.raises(ValueError, match="SOURCE_RUN_NAME is required"):
        run_chain.validate_parameters()


def test_initial_state_loads_latest_configuration_from_source_run(
    tmp_path, monkeypatch
):
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    source_dir = tmp_path / "results" / "runs" / "source_run" / "configurations"
    earlier_path = source_dir / "later_name.npz"
    source_path = source_dir / "earlier_name.npz"
    save_configuration(
        earlier_path,
        links,
        {"shape": geometry.shape, "chain": 0, "sweep": 20},
    )
    save_configuration(
        source_path,
        links,
        {
            "shape": geometry.shape,
            "beta": 5.7,
            "chain": 0,
            "sweep": 30,
            "start": "hot",
        },
    )
    save_configuration(
        source_dir / "other_chain.npz",
        links,
        {"shape": geometry.shape, "chain": 1, "sweep": 40},
    )
    monkeypatch.setattr(run_chain, "ROOT", tmp_path)
    monkeypatch.setattr(run_chain, "SHAPE", geometry.shape)
    monkeypatch.setattr(run_chain, "SOURCE_RUN_NAME", "source_run")
    monkeypatch.setattr(run_chain, "SOURCE_CHAIN", 0)

    state = run_chain.initial_state(
        "load", geometry, np.random.default_rng(123)
    )
    manifest = run_chain.manifest_data(state)

    assert np.allclose(state.links, links, atol=0.0)
    assert state.links is not links
    assert state.initial_sweep == 30
    assert state.source_path == source_path
    assert manifest["source_configuration"] == (
        "results/runs/source_run/configurations/earlier_name.npz"
    )
    assert manifest["source_run_name"] == "source_run"
    assert manifest["source_chain"] == 0
    assert manifest["source_sweep"] == 30
    assert manifest["source_start"] == "hot"
    assert manifest["source_beta"] == 5.7


def test_load_mode_requires_distinct_output_run_name(monkeypatch):
    monkeypatch.setattr(run_chain, "STARTS", ("load",))
    monkeypatch.setattr(run_chain, "SOURCE_RUN_NAME", "existing_run")
    monkeypatch.setattr(run_chain, "RUN_NAME", "existing_run")

    with pytest.raises(ValueError, match="must differ"):
        run_chain.validate_parameters()


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


def test_observable_fieldnames_include_polyakov_columns_when_enabled(monkeypatch):
    monkeypatch.setattr(run_chain, "MEASURE_POLYAKOV", True)
    monkeypatch.setattr(run_chain, "POLYAKOV_OFFSETS", ((1, 0, 0),))

    assert run_chain.observable_fieldnames() == [
        "chain",
        "start",
        "sweep",
        "average_plaquette",
        "acceptance_rate",
        "accepted_links",
        "attempted_links",
        "polyakov_re",
        "polyakov_im",
        "polyakov_abs",
        "polyakov_abs2",
        "polyakov_c_1_0_0_re",
        "polyakov_c_1_0_0_im",
    ]


def test_polyakov_measurements_from_cold_start_are_unit_values(monkeypatch):
    monkeypatch.setattr(run_chain, "POLYAKOV_OFFSETS", ((1, 0, 0), (0, 1, 0)))
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)

    measurements = run_chain.polyakov_measurements(links, geometry)

    assert measurements["polyakov_re"] == 1.0
    assert measurements["polyakov_im"] == 0.0
    assert measurements["polyakov_abs"] == 1.0
    assert measurements["polyakov_abs2"] == 1.0
    assert measurements["polyakov_c_1_0_0_re"] == 1.0
    assert measurements["polyakov_c_1_0_0_im"] == 0.0
    assert measurements["polyakov_c_0_1_0_re"] == 1.0
    assert measurements["polyakov_c_0_1_0_im"] == 0.0


def test_validate_parameters_checks_polyakov_offset_shape(monkeypatch):
    monkeypatch.setattr(run_chain, "MEASURE_POLYAKOV", True)
    monkeypatch.setattr(run_chain, "POLYAKOV_OFFSETS", ((1, 0),))

    with pytest.raises(ValueError, match="POLYAKOV_OFFSETS"):
        run_chain.validate_parameters()


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


def test_loaded_save_interval_uses_segment_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(run_chain, "SAVE_CONFIG_EVERY", 5)
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    source_path = tmp_path / "source.npz"
    state = run_chain.InitialState(
        links,
        initial_sweep=300,
        source_path=source_path,
        source_metadata={"sweep": 300, "start": "hot", "beta": 5.7},
    )

    path = run_chain.maybe_save_configuration(
        tmp_path,
        links,
        chain=0,
        start="load",
        sweep=305,
        plaquette=1.0,
        segment_sweep=5,
        state=state,
    )

    assert path == tmp_path / "configurations" / "chain00_load_sweep000305.npz"
    _, metadata = load_configuration(path)
    assert metadata["sweep"] == 305
    assert metadata["initial_sweep"] == 300
    assert metadata["source_sweep"] == 300
    assert metadata["source_start"] == "hot"


def test_run_one_chain_continues_accumulated_sweep_numbers(
    tmp_path, monkeypatch
):
    geometry = LatticeGeometry((2, 2, 2, 2))
    state = run_chain.InitialState(cold_start(geometry), initial_sweep=30)
    monkeypatch.setattr(run_chain, "SHAPE", geometry.shape)
    monkeypatch.setattr(run_chain, "SWEEPS", 2)
    monkeypatch.setattr(run_chain, "MEASURE_EVERY", 1)
    monkeypatch.setattr(run_chain, "SAVE_CONFIG_EVERY", 0)

    stats = SimpleNamespace(
        acceptance_rate=1.0,
        accepted_links=geometry.volume * geometry.ndim,
        attempted_links=geometry.volume * geometry.ndim,
    )
    monkeypatch.setattr(
        run_chain.SweepRunner,
        "sweep",
        lambda self, links, geometry, beta, step_size: stats,
    )
    stream = io.StringIO()
    fieldnames = list(
        run_chain.observable_row(0, "load", 0, 0.0, "", "", "").keys()
    )
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()

    run_chain.run_one_chain(
        writer,
        tmp_path,
        chain=0,
        start="load",
        seed_sequence=np.random.SeedSequence(123),
        prepared_state=state,
    )

    stream.seek(0)
    rows = list(csv.DictReader(stream))
    assert [int(row["sweep"]) for row in rows] == [30, 31, 32]
