import importlib.util
from pathlib import Path
import sys

import numpy as np

from lattice_su3 import LatticeGeometry, cold_start, save_configuration


AUTO_CORR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "auto_correlation.py"
AUTO_CORR_SPEC = importlib.util.spec_from_file_location("auto_correlation", AUTO_CORR_PATH)
auto_corr = importlib.util.module_from_spec(AUTO_CORR_SPEC)
assert AUTO_CORR_SPEC.loader is not None
sys.modules[AUTO_CORR_SPEC.name] = auto_corr
AUTO_CORR_SPEC.loader.exec_module(auto_corr)


THERMAL_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_thermalization.py"
)
THERMAL_SPEC = importlib.util.spec_from_file_location(
    "analyze_thermalization", THERMAL_PATH
)
thermal = importlib.util.module_from_spec(THERMAL_SPEC)
assert THERMAL_SPEC.loader is not None
sys.modules[THERMAL_SPEC.name] = thermal
THERMAL_SPEC.loader.exec_module(thermal)


THINNING_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "thinning_autocorrelation.py"
)
THINNING_SPEC = importlib.util.spec_from_file_location(
    "thinning_autocorrelation", THINNING_PATH
)
thinning = importlib.util.module_from_spec(THINNING_SPEC)
assert THINNING_SPEC.loader is not None
sys.modules[THINNING_SPEC.name] = thinning
THINNING_SPEC.loader.exec_module(thinning)


POLYAKOV_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "measure_polyakov_correlators.py"
)
POLYAKOV_SPEC = importlib.util.spec_from_file_location(
    "measure_polyakov_correlators", POLYAKOV_PATH
)
polyakov_measure = importlib.util.module_from_spec(POLYAKOV_SPEC)
assert POLYAKOV_SPEC.loader is not None
sys.modules[POLYAKOV_SPEC.name] = polyakov_measure
POLYAKOV_SPEC.loader.exec_module(polyakov_measure)


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
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert np.allclose(data["correlators"], correlators, atol=1e-12)
        assert np.array_equal(data["shape"], [2, 2, 2, 2])
        assert data["beta"].item() == 5.7
        assert data["time_direction"].item() == -1
        assert data["thermalization_sweeps"].item() == 5
