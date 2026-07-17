import importlib.util
from pathlib import Path
import sys

import numpy as np

from lattice_su3.configuration import load_configuration
from lattice_su3.run_config import load_ensemble_config, load_thermalize_config


THERMALIZE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "thermalize.py"
THERMALIZE_SPEC = importlib.util.spec_from_file_location("thermalize", THERMALIZE_PATH)
thermalize = importlib.util.module_from_spec(THERMALIZE_SPEC)
assert THERMALIZE_SPEC.loader is not None
sys.modules[THERMALIZE_SPEC.name] = thermalize
THERMALIZE_SPEC.loader.exec_module(thermalize)


ENSEMBLE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_ensemble.py"
ENSEMBLE_SPEC = importlib.util.spec_from_file_location(
    "generate_ensemble", ENSEMBLE_PATH
)
generate_ensemble = importlib.util.module_from_spec(ENSEMBLE_SPEC)
assert ENSEMBLE_SPEC.loader is not None
sys.modules[ENSEMBLE_SPEC.name] = generate_ensemble
ENSEMBLE_SPEC.loader.exec_module(generate_ensemble)


def test_load_thermalize_config_reads_toml(tmp_path):
    path = tmp_path / "thermalize.toml"
    path.write_text(
        """
[run]
name = "therm_test"
shape = [2, 2, 2, 2]
beta = 5.7
sweeps = 2
seed = 123
start = "hot"

[update]
backend = "numpy"
overrelaxation_sweeps = 0

[measure]
plaquette_every = 1

[save]
config_every = 1
overwrite = true
""",
        encoding="utf-8",
    )

    config = load_thermalize_config(path)

    assert config.name == "therm_test"
    assert config.shape == (2, 2, 2, 2)
    assert config.update.backend == "numpy"
    assert config.save.config_every == 1


def test_load_ensemble_config_reads_toml(tmp_path):
    path = tmp_path / "ensemble.toml"
    path.write_text(
        """
[run]
name = "prod_test"
shape = [2, 2, 2, 2]
beta = 5.7

[source]
config = "source.npz"

[ensemble]
chains = 2
sweeps_per_chain = 2
discard_sweeps = 0
seed_base = 100
parallel = 1

[update]
backend = "numpy"
overrelaxation_sweeps = 0
""",
        encoding="utf-8",
    )

    config = load_ensemble_config(path)

    assert config.name == "prod_test"
    assert config.source_config == Path("source.npz")
    assert config.chains == 2
    assert config.parallel == 1


def test_thermalize_and_generate_ensemble_write_standard_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(thermalize, "ROOT", tmp_path)
    monkeypatch.setattr(generate_ensemble, "ROOT", tmp_path)
    thermal_config = tmp_path / "thermalize.toml"
    thermal_config.write_text(
        """
[run]
name = "therm_test"
shape = [2, 2, 2, 2]
beta = 5.7
sweeps = 2
seed = 123
start = "hot"

[update]
backend = "numpy"
overrelaxation_sweeps = 0

[measure]
plaquette_every = 1

[save]
config_every = 1
overwrite = true
""",
        encoding="utf-8",
    )

    thermalize.main([str(thermal_config)])
    source_config = (
        tmp_path
        / "results"
        / "runs"
        / "therm_test"
        / "configurations"
        / "chain00_hot_sweep000002.npz"
    )
    assert source_config.exists()

    ensemble_config = tmp_path / "ensemble.toml"
    ensemble_config.write_text(
        f"""
[run]
name = "prod_test"
shape = [2, 2, 2, 2]
beta = 5.7

[source]
config = "{source_config}"

[ensemble]
chains = 2
sweeps_per_chain = 2
discard_sweeps = 0
seed_base = 200
parallel = 1

[update]
backend = "numpy"
overrelaxation_sweeps = 0

[measure]
plaquette_every = 1

[save]
config_every = 1
overwrite = true
""",
        encoding="utf-8",
    )

    generate_ensemble.main([str(ensemble_config)])
    run_dir = tmp_path / "results" / "runs" / "prod_test"
    observables = np.genfromtxt(
        run_dir / "observables.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    config_paths = sorted((run_dir / "configurations").glob("*.npz"))

    assert (run_dir / "manifest.json").exists()
    assert set(np.asarray(observables["chain"], dtype=np.int64)) == {0, 1}
    assert len(config_paths) == 4
    _, metadata = load_configuration(config_paths[-1])
    assert metadata["shape"] == [2, 2, 2, 2]
    assert metadata["run_name"] == "prod_test"
    assert metadata["start"] == "load"
    assert metadata["sweep"] == 4
