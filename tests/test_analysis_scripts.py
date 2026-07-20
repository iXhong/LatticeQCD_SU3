import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from lattice_su3 import LatticeGeometry, cold_start, save_configuration


AUTO_CORR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "auto_correlation.py"
)
AUTO_CORR_SPEC = importlib.util.spec_from_file_location("auto_correlation", AUTO_CORR_PATH)
auto_corr = importlib.util.module_from_spec(AUTO_CORR_SPEC)
assert AUTO_CORR_SPEC.loader is not None
sys.modules[AUTO_CORR_SPEC.name] = auto_corr
AUTO_CORR_SPEC.loader.exec_module(auto_corr)


THERMAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "analyze_thermalization.py"
)
THERMAL_SPEC = importlib.util.spec_from_file_location(
    "analyze_thermalization", THERMAL_PATH
)
thermal = importlib.util.module_from_spec(THERMAL_SPEC)
assert THERMAL_SPEC.loader is not None
sys.modules[THERMAL_SPEC.name] = thermal
THERMAL_SPEC.loader.exec_module(thermal)


THINNING_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "thinning_autocorrelation.py"
)
THINNING_SPEC = importlib.util.spec_from_file_location(
    "thinning_autocorrelation", THINNING_PATH
)
thinning = importlib.util.module_from_spec(THINNING_SPEC)
assert THINNING_SPEC.loader is not None
sys.modules[THINNING_SPEC.name] = thinning
THINNING_SPEC.loader.exec_module(thinning)


POLYAKOV_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "measure_polyakov_correlators.py"
)
POLYAKOV_SPEC = importlib.util.spec_from_file_location(
    "measure_polyakov_correlators", POLYAKOV_PATH
)
polyakov_measure = importlib.util.module_from_spec(POLYAKOV_SPEC)
assert POLYAKOV_SPEC.loader is not None
sys.modules[POLYAKOV_SPEC.name] = polyakov_measure
POLYAKOV_SPEC.loader.exec_module(polyakov_measure)


BIN_POLYAKOV_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "bin_polyakov_correlators.py"
)
BIN_POLYAKOV_SPEC = importlib.util.spec_from_file_location(
    "bin_polyakov_correlators", BIN_POLYAKOV_PATH
)
bin_polyakov = importlib.util.module_from_spec(BIN_POLYAKOV_SPEC)
assert BIN_POLYAKOV_SPEC.loader is not None
sys.modules[BIN_POLYAKOV_SPEC.name] = bin_polyakov
BIN_POLYAKOV_SPEC.loader.exec_module(bin_polyakov)


RESAMPLE_POLYAKOV_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "resample_polyakov_correlators.py"
)
RESAMPLE_POLYAKOV_SPEC = importlib.util.spec_from_file_location(
    "resample_polyakov_correlators", RESAMPLE_POLYAKOV_PATH
)
resample_polyakov = importlib.util.module_from_spec(RESAMPLE_POLYAKOV_SPEC)
assert RESAMPLE_POLYAKOV_SPEC.loader is not None
sys.modules[RESAMPLE_POLYAKOV_SPEC.name] = resample_polyakov
RESAMPLE_POLYAKOV_SPEC.loader.exec_module(resample_polyakov)


POLYAKOV_AUTO_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "polyakov_autocorrelation.py"
)
POLYAKOV_AUTO_SPEC = importlib.util.spec_from_file_location(
    "polyakov_autocorrelation", POLYAKOV_AUTO_PATH
)
polyakov_auto = importlib.util.module_from_spec(POLYAKOV_AUTO_SPEC)
assert POLYAKOV_AUTO_SPEC.loader is not None
sys.modules[POLYAKOV_AUTO_SPEC.name] = polyakov_auto
POLYAKOV_AUTO_SPEC.loader.exec_module(polyakov_auto)


def test_load_observable_history_filters_chain_and_thermalization(tmp_path):
    path = tmp_path / "observables.csv"
    path.write_text(
        "chain,start,sweep,average_plaquette,acceptance_rate,accepted_links,attempted_links\n"
        "0,hot,0,0.40,,,\n"
        "0,hot,10,0.45,1.0,8,8\n"
        "0,hot,20,0.50,1.0,8,8\n"
        "1,cold,20,0.60,1.0,8,8\n",
        encoding="utf-8",
    )

    sweeps, series = auto_corr.load_observable_history(
        path,
        chain=0,
        thermalization_sweeps=10,
    )

    assert np.array_equal(sweeps, [20])
    assert np.allclose(series, [0.50], atol=1e-12)


def test_polyakov_autocorrelation_loads_selected_columns(tmp_path):
    path = tmp_path / "observables.csv"
    path.write_text(
        "chain,start,sweep,average_plaquette,acceptance_rate,accepted_links,"
        "attempted_links,polyakov_abs,polyakov_abs2\n"
        "0,load,100,0.50,1.0,8,8,0.10,0.01\n"
        "0,load,110,0.51,1.0,8,8,0.20,0.04\n"
        "0,load,120,0.52,1.0,8,8,0.30,0.09\n",
        encoding="utf-8",
    )

    sweeps, series = polyakov_auto.load_observable_columns(
        path,
        ("polyakov_abs", "polyakov_abs2"),
        thermalization_sweeps=100,
    )

    assert np.array_equal(sweeps, [110, 120])
    assert np.allclose(series["polyakov_abs"], [0.20, 0.30], atol=1e-12)
    assert np.allclose(series["polyakov_abs2"], [0.04, 0.09], atol=1e-12)


def test_polyakov_autocorrelation_summary_contains_tau_fields():
    summary = polyakov_auto.autocorrelation_summary(
        np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        max_lag=2,
    )

    assert set(summary) == {
        "mean",
        "std",
        "window",
        "tau_int",
        "effective_samples",
        "suggested_interval",
    }
    assert summary["tau_int"] > 0.0


def test_auto_correlation_run_label_records_overrelaxation(monkeypatch):
    monkeypatch.setattr(auto_corr, "RUN_NAME", "")
    monkeypatch.setattr(auto_corr, "OVERRELAXATION_SWEEPS", 2)

    assert auto_corr.run_label() == (
        "heatbath_or2_jit_hot_16x16x16x6_beta5.7_300sweeps_seed12345"
    )


def test_thermalization_run_label_records_overrelaxation(monkeypatch):
    monkeypatch.setattr(thermal, "RUN_NAME", "")
    monkeypatch.setattr(thermal, "OVERRELAXATION_SWEEPS", 2)

    assert thermal.run_label() == (
        "heatbath_or2_jit_cold-hot_16x16x16x6_beta5.7_300sweeps_seed12345"
    )


def test_thinning_load_plaquette_series_filters_chain_and_thermalization(tmp_path):
    path = tmp_path / "observables.csv"
    path.write_text(
        "chain,start,sweep,average_plaquette,acceptance_rate,accepted_links,attempted_links\n"
        "0,hot,0,0.40,,,\n"
        "0,hot,10,0.45,1.0,8,8\n"
        "0,hot,20,0.50,1.0,8,8\n"
        "1,cold,20,0.60,1.0,8,8\n",
        encoding="utf-8",
    )

    sweeps, series = thinning.load_plaquette_series(
        path,
        chain=0,
        thermalization_sweeps=10,
    )

    assert np.array_equal(sweeps, [20])
    assert np.allclose(series, [0.50], atol=1e-12)


def test_thin_series_and_sweep_spacing():
    sweeps = np.asarray([10, 15, 20, 25, 30], dtype=np.int64)
    series = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    assert thinning.sweep_spacing(sweeps) == 5
    assert np.allclose(thinning.thin_series(series, 2), [1.0, 3.0, 5.0], atol=1e-12)


def test_load_plaquette_histories_groups_cold_and_hot_starts(tmp_path):
    path = tmp_path / "observables.csv"
    path.write_text(
        "chain,start,sweep,average_plaquette,acceptance_rate,accepted_links,attempted_links\n"
        "0,cold,0,1.00,,,\n"
        "0,cold,10,0.60,1.0,8,8\n"
        "1,hot,0,0.10,,,\n"
        "1,hot,10,0.50,1.0,8,8\n",
        encoding="utf-8",
    )

    histories = thermal.load_plaquette_histories(path)

    assert np.array_equal(histories["cold"]["sweep"], [0, 10])
    assert np.allclose(histories["cold"]["plaquette"], [1.00, 0.60], atol=1e-12)
    assert np.array_equal(histories["hot"]["sweep"], [0, 10])
    assert np.allclose(histories["hot"]["plaquette"], [0.10, 0.50], atol=1e-12)


def test_plot_plaquette_histories_writes_png(tmp_path):
    histories = {
        "cold": {
            "sweep": np.asarray([0, 10], dtype=np.int64),
            "plaquette": np.asarray([1.00, 0.60], dtype=np.float64),
        },
        "hot": {
            "sweep": np.asarray([0, 10], dtype=np.int64),
            "plaquette": np.asarray([0.10, 0.50], dtype=np.float64),
        },
    }
    observables_path = tmp_path / "observables.csv"
    output_path = tmp_path / "thermalization_plaquette.png"

    plot_path = thermal.plot_plaquette_histories(
        histories,
        observables_path,
        output_path,
    )

    assert plot_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_measure_polyakov_correlator_ensemble_from_cold_configs(tmp_path):
    config_dir = tmp_path / "configurations"
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    manifest = {"shape": [2, 2, 2, 2], "beta": 5.7, "run_name": "test_run"}

    for sweep in (5, 10):
        save_configuration(
            config_dir / f"chain00_hot_sweep{sweep:06d}.npz",
            links,
            {
                "shape": geometry.shape,
                "chain": 0,
                "start": "hot",
                "sweep": sweep,
            },
        )

    paths = polyakov_measure.configuration_paths(config_dir)
    assert polyakov_measure.output_correlator_path(tmp_path) == (
        tmp_path / "correlators" / "polyakov_vector_correlators.npz"
    )
    filtered_paths = polyakov_measure.filter_thermalized_paths(
        paths,
        thermalization_sweeps=5,
    )
    correlators, sweeps, chains, starts, filenames = (
        polyakov_measure.measure_correlator_ensemble(
            filtered_paths,
            geometry,
            time_direction=-1,
        )
    )

    assert correlators.shape == (1, 2, 2, 2)
    assert np.allclose(correlators, 1.0 + 0.0j, atol=1e-12)
    assert np.array_equal(sweeps, [10])
    assert np.array_equal(chains, [0])
    assert np.array_equal(starts, ["hot"])
    assert filenames[0] == "chain00_hot_sweep000010.npz"

    output_path = tmp_path / "polyakov_vector_correlators.npz"
    polyakov_measure.write_correlators(
        output_path,
        correlators,
        sweeps,
        chains,
        starts,
        filenames,
        manifest,
        time_direction=-1,
        thermalization_sweeps=5,
        run_name="test_run",
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert np.allclose(data["correlators"], correlators, atol=1e-12)
        assert np.array_equal(data["shape"], [2, 2, 2, 2])
        assert data["beta"].item() == 5.7
        assert data["time_direction"].item() == -1
        assert data["thermalization_sweeps"].item() == 5


def test_measure_polyakov_defaults_to_empty_run_name():
    parser = polyakov_measure.build_argument_parser()

    args = parser.parse_args([])

    assert args.run_name == ""
    with pytest.raises(ValueError, match="run name must be non-empty"):
        polyakov_measure.run_directory(args.run_name)


def test_bin_polyakov_correlators_reads_measurement_output(tmp_path):
    input_path = tmp_path / "polyakov_vector_correlators.npz"
    output_path = bin_polyakov.binned_output_path(input_path)
    correlators = np.ones((2, 4, 4, 4), dtype=np.complex128)
    manifest = {"shape": [4, 4, 4, 6], "beta": 5.7, "run_name": "test_run"}
    polyakov_measure.write_correlators(
        input_path,
        correlators,
        np.asarray([1010, 1020]),
        np.asarray([0, 1]),
        np.asarray(["load", "load"]),
        np.asarray(["cfg0.npz", "cfg1.npz"]),
        manifest,
        time_direction=-1,
        thermalization_sweeps=1000,
        run_name="test_run",
    )

    source = bin_polyakov.load_vector_correlators(input_path)
    bin_polyakov.write_binned_correlators(output_path, source)

    with np.load(output_path, allow_pickle=False) as data:
        assert data["radial_correlators"].shape == (2, 10)
        assert data["axis_correlators"].shape == (2, 3)
        assert data["radial_degeneracies"].sum() == 4**3
        assert np.allclose(data["radial_correlators"], 1.0, atol=1e-12)
        assert np.allclose(data["axis_correlators"], 1.0, atol=1e-12)
        assert np.array_equal(data["sweeps"], [1010, 1020])
        assert np.array_equal(data["chains"], [0, 1])
        assert np.array_equal(data["shape"], [4, 4, 4, 6])
        assert data["time_direction"].item() == -1


def test_resample_polyakov_correlators_preserves_chain_block_provenance(tmp_path):
    input_path = tmp_path / "polyakov_binned_correlators.npz"
    output_path = resample_polyakov.resampled_output_path(input_path)
    n_cfg = 10
    np.savez_compressed(
        input_path,
        radial_r_squared=np.asarray([0, 1]),
        radial_r=np.asarray([0.0, 1.0]),
        radial_degeneracies=np.asarray([1, 6]),
        radial_correlators=np.arange(n_cfg * 2).reshape(n_cfg, 2),
        axis_r=np.asarray([0, 1]),
        axis_degeneracies=np.asarray([1, 6]),
        axis_correlators=np.arange(n_cfg * 2).reshape(n_cfg, 2),
        sweeps=np.asarray([30, 10, 20, 40, 50, 20, 10, 30, 50, 40]),
        chains=np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        shape=np.asarray([4, 4, 4, 6]),
        beta=np.asarray(5.7),
        time_direction=np.asarray(-1),
        run_name=np.asarray("test_run"),
    )

    source = resample_polyakov.load_binned_correlators(input_path)
    resample_polyakov.write_resampled_correlators(
        output_path,
        source,
        block_size=2,
        bootstrap_samples=20,
        bootstrap_seed=123,
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert data["radial_block_correlators"].shape == (4, 2)
        assert data["radial_jackknife_correlators"].shape == (4, 2)
        assert data["radial_bootstrap_correlators"].shape == (20, 2)
        assert data["axis_block_correlators"].shape == (4, 2)
        assert np.array_equal(data["block_chains"], [0, 0, 1, 1])
        assert np.array_equal(data["block_sweep_start"], [10, 30, 10, 30])
        assert np.array_equal(data["block_sweep_stop"], [20, 40, 20, 40])
        assert np.array_equal(data["dropped_per_chain"], [[0, 1], [1, 1]])
