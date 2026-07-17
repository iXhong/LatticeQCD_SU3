import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_ensemble.py"
SPEC = importlib.util.spec_from_file_location("run_ensemble", SCRIPT_PATH)
run_ensemble = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = run_ensemble
SPEC.loader.exec_module(run_ensemble)


def test_chain_command_assigns_distinct_run_name_and_seed():
    parser = run_ensemble.build_argument_parser()
    args = parser.parse_args(
        [
            "--chains",
            "4",
            "--parallel",
            "2",
            "--run-prefix",
            "prod",
            "--seed-base",
            "100",
            "--sweeps",
            "1000",
            "--save-config-every",
            "10",
            "--measure-every",
            "10",
            "--backend",
            "jit_checkerboard",
            "--overrelaxation-sweeps",
            "2",
        ]
    )

    command = run_ensemble.chain_command(args, 3)

    assert "--run-name" in command
    assert command[command.index("--run-name") + 1] == "prod_chain03_seed103"
    assert command[command.index("--seed") + 1] == "103"
    assert command[command.index("--sweeps") + 1] == "1000"
    assert command[command.index("--backend") + 1] == "jit_checkerboard"
    assert command[command.index("--overrelaxation-sweeps") + 1] == "2"
    assert "--measure-plaquette" in command
    assert "--no-measure-polyakov" in command
