import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_configurations.py"
SPEC = importlib.util.spec_from_file_location("generate_configurations", SCRIPT_PATH)
gen = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gen
SPEC.loader.exec_module(gen)


def test_shape_label_and_output_directory_are_deterministic():
    path = gen.output_directory()

    assert gen.shape_label((16, 16, 16, 6)) == "16x16x16x6"
    assert path.name == "heatbath_jit_16x16x16x6_beta5.7_seed12345"
    assert path.parent.name == "configurations"


def test_manifest_row_contains_expected_fields():
    row = gen.manifest_row(
        filename="config_000000.npz",
        config_index=0,
        sweep=6002,
        plaquette=0.55,
        tau_int=3.2,
        sweeps_between_configs=7,
    )

    assert row == {
        "filename": "config_000000.npz",
        "config_index": 0,
        "sweep": 6002,
        "average_plaquette": 0.55,
        "backend": "jit",
        "beta": 5.7,
        "shape": "16x16x16x6",
        "seed": 12345,
        "tau_int": 3.2,
        "sweeps_between_configs": 7,
    }


def test_configuration_metadata_contains_expected_fields():
    metadata = gen.configuration_metadata(
        config_index=2,
        sweep=6200,
        plaquette=0.56,
        tau_int=4.0,
        sweeps_between_configs=8,
    )

    assert metadata["shape"] == (16, 16, 16, 6)
    assert metadata["beta"] == 5.7
    assert metadata["sweep"] == 6200
    assert metadata["config_index"] == 2
    assert metadata["backend"] == "jit"
    assert metadata["seed"] == 12345
    assert metadata["average_plaquette"] == 0.56
    assert metadata["tau_int"] == 4.0
    assert metadata["sweeps_between_configs"] == 8
