"""
Launch several independent gauge-update chains in parallel.

This script starts multiple scripts/run_chain.py subprocesses with different
seeds and run names. Each chain writes its own results/runs/<run-name>
directory, so no observable CSV or configuration file is shared between
processes. It expects the same repository layout as scripts/run_chain.py.

Usage:
    Run 4 independent HB+2OR chains with the project environment:
        UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_ensemble.py \
            --chains 4 --parallel 4 --run-prefix prod_hb_or2 \
            --sweeps 1000 --save-config-every 10 --measure-every 10 \
            --backend jit_checkerboard --overrelaxation-sweeps 2
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN_CHAIN = ROOT / "scripts" / "run_chain.py"


def positive_int(value: str) -> int:
    """Parse a positive integer command-line value.

    Inputs:
        value: String command-line value.
    Outputs:
        Positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer command-line value.

    Inputs:
        value: String command-line value.
    Outputs:
        Non-negative integer.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the ensemble launcher argument parser.

    Inputs:
        None.
    Outputs:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Launch independent run_chain.py jobs in parallel."
    )
    parser.add_argument("--chains", type=positive_int, default=4)
    parser.add_argument("--parallel", type=positive_int, default=4)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--shape")
    parser.add_argument("--beta", type=float)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--sweeps", type=nonnegative_int, required=True)
    parser.add_argument("--measure-every", type=nonnegative_int, default=10)
    parser.add_argument("--save-config-every", type=nonnegative_int, default=10)
    parser.add_argument(
        "--measure-plaquette",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--measure-polyakov",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--algorithm",
        choices=("heatbath", "metropolis"),
        default="heatbath",
    )
    parser.add_argument(
        "--backend",
        choices=("jit", "jit_checkerboard", "numpy"),
        default="jit_checkerboard",
    )
    parser.add_argument("--overrelaxation-sweeps", type=nonnegative_int, default=2)
    parser.add_argument("--start", choices=("cold", "hot", "load"), default="hot")
    parser.add_argument("--source-run-name")
    parser.add_argument("--source-chain", type=nonnegative_int)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def optional_flag(command: list[str], name: str, value: object | None) -> None:
    """Append one optional command-line flag.

    Inputs:
        command: Command list being constructed.
        name: Flag name.
        value: Optional flag value.
    Outputs:
        None.
    """
    if value is not None:
        command.extend([name, str(value)])


def chain_command(args: argparse.Namespace, chain: int) -> list[str]:
    """Build the run_chain.py command for one chain.

    Inputs:
        args: Parsed launcher arguments.
        chain: Zero-based chain index.
    Outputs:
        Subprocess command argument list.
    """
    seed = args.seed_base + chain
    run_name = f"{args.run_prefix}_chain{chain:02d}_seed{seed}"
    command = [
        sys.executable,
        str(RUN_CHAIN),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--sweeps",
        str(args.sweeps),
        "--measure-every",
        str(args.measure_every),
        "--save-config-every",
        str(args.save_config_every),
        "--algorithm",
        args.algorithm,
        "--backend",
        args.backend,
        "--overrelaxation-sweeps",
        str(args.overrelaxation_sweeps),
        "--start",
        args.start,
    ]
    optional_flag(command, "--shape", args.shape)
    optional_flag(command, "--beta", args.beta)
    optional_flag(command, "--step-size", args.step_size)
    optional_flag(command, "--source-run-name", args.source_run_name)
    optional_flag(command, "--source-chain", args.source_chain)
    command.append("--measure-plaquette" if args.measure_plaquette else "--no-measure-plaquette")
    command.append("--measure-polyakov" if args.measure_polyakov else "--no-measure-polyakov")
    if args.overwrite:
        command.append("--overwrite")
    return command


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one chain subprocess.

    Inputs:
        command: Subprocess command argument list.
    Outputs:
        Completed subprocess result.
    """
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> None:
    """Launch configured independent chains.

    Inputs:
        argv: Optional command-line arguments excluding the executable name.
    Outputs:
        None.
    """
    args = build_argument_parser().parse_args(argv)
    commands = [chain_command(args, chain) for chain in range(args.chains)]
    parallel = min(args.parallel, args.chains)
    print(f"Launching {args.chains} chains with parallel={parallel}")

    failures = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures: dict[Future[subprocess.CompletedProcess[str]], int] = {
            executor.submit(run_command, command): chain
            for chain, command in enumerate(commands)
        }
        for future in as_completed(futures):
            chain = futures[future]
            result = future.result()
            run_name = f"{args.run_prefix}_chain{chain:02d}_seed{args.seed_base + chain}"
            if result.returncode == 0:
                print(f"[chain {chain:02d}] completed: {run_name}")
                continue
            failures.append((chain, result))
            print(f"[chain {chain:02d}] failed: {run_name}")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

    if failures:
        failed = ", ".join(f"{chain:02d}" for chain, _ in failures)
        raise SystemExit(f"failed chains: {failed}")

    print("All chains completed.")


if __name__ == "__main__":
    main()
