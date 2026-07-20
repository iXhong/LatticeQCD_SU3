"""Benchmark NumPy and optional JIT heatbath sweep implementations."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

SHAPES = ((4, 4, 4, 6), (6, 6, 6, 6))
BETA = 5.7
SEED = 12345

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lattice_su3 import (  # noqa: E402
    LatticeGeometry,
    heatbath_checkerboard_sweep,
    heatbath_sweep,
    hot_start,
)

try:
    from lattice_su3.accelerated import (  # noqa: E402
        heatbath_checkerboard_jit_sweep,
        heatbath_jit_sweep,
    )
except ImportError:
    heatbath_checkerboard_jit_sweep = None
    heatbath_jit_sweep = None


def time_call(function) -> float:
    """Measure one function call.

    Inputs:
        function: Zero-argument callable to run once.
    Outputs:
        Elapsed wall time in seconds.
    """
    start = perf_counter()
    function()
    return perf_counter() - start


def benchmark_shape(shape: tuple[int, ...]) -> None:
    """Benchmark heatbath implementations for one lattice shape.

    Inputs:
        shape: Lattice size in each direction.
    Outputs:
        None.
    """
    geometry = LatticeGeometry(shape)
    rng = np.random.default_rng(SEED)
    links = hot_start(geometry, rng)

    numpy_links = links.copy()
    checkerboard_links = links.copy()
    print(f"shape={shape}, volume={geometry.volume}")
    numpy_time = time_call(lambda: heatbath_sweep(numpy_links, geometry, BETA, rng))
    print(f"  heatbath_sweep: {numpy_time:.6f} s")

    checkerboard_time = time_call(
        lambda: heatbath_checkerboard_sweep(checkerboard_links, geometry, BETA, rng)
    )
    print(f"  heatbath_checkerboard_sweep: {checkerboard_time:.6f} s")

    if heatbath_jit_sweep is None:
        print("  heatbath_jit_sweep: unavailable")
        return

    compile_links = links.copy()
    compile_time = time_call(
        lambda: heatbath_jit_sweep(compile_links, geometry, BETA, seed=SEED)
    )
    jit_links = links.copy()
    jit_time = time_call(lambda: heatbath_jit_sweep(jit_links, geometry, BETA, seed=SEED))
    print(f"  heatbath_jit_sweep compile+run: {compile_time:.6f} s")
    print(f"  heatbath_jit_sweep cached run: {jit_time:.6f} s")
    print(f"  jit speedup vs numpy: {numpy_time / jit_time:.2f}x")

    checkerboard_jit_links = links.copy()
    checkerboard_jit_compile_time = time_call(
        lambda: heatbath_checkerboard_jit_sweep(
            checkerboard_jit_links, geometry, BETA, seed=SEED
        )
    )
    checkerboard_jit_links = links.copy()
    checkerboard_jit_time = time_call(
        lambda: heatbath_checkerboard_jit_sweep(
            checkerboard_jit_links, geometry, BETA, seed=SEED
        )
    )
    print(
        "  heatbath_checkerboard_jit_sweep compile+run: "
        f"{checkerboard_jit_compile_time:.6f} s"
    )
    print(
        "  heatbath_checkerboard_jit_sweep cached run: "
        f"{checkerboard_jit_time:.6f} s"
    )
    print(
        "  checkerboard jit speedup vs numpy: "
        f"{numpy_time / checkerboard_jit_time:.2f}x"
    )


def main() -> None:
    """Run heatbath acceleration benchmarks.

    Inputs:
        None.
    Outputs:
        None.
    """
    for shape in SHAPES:
        benchmark_shape(shape)


if __name__ == "__main__":
    main()
