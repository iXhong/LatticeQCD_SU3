import importlib.util
from pathlib import Path
import sys

import numpy as np


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
