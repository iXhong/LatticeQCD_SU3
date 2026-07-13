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


def test_summarize_chain_uses_tail_sweeps(tmp_path):
    path = tmp_path / "observables.csv"
    path.write_text(
        "chain,start,sweep,average_plaquette,acceptance_rate,accepted_links,attempted_links\n"
        "0,hot,0,0.40,,,\n"
        "0,hot,10,0.45,1.0,8,8\n"
        "0,hot,20,0.55,1.0,8,8\n"
        "0,hot,30,0.65,1.0,8,8\n",
        encoding="utf-8",
    )
    data = thermal.load_observables(path)

    summary = thermal.summarize_chain(data, chain=0, tail_sweeps=15)

    assert summary["chain"] == 0
    assert summary["start"] == "hot"
    assert summary["tail_start_sweep"] == 16
    assert summary["tail_end_sweep"] == 30
    assert summary["tail_measurements"] == 2
    assert np.isclose(summary["tail_mean_plaquette"], 0.60)

