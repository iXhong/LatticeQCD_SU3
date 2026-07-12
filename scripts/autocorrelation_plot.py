"""
readme: this is a script for plotting
- Gamma(t) vs t (scatter)
- Gamma(t) vs t (scatter style) and exp(-t/tau_int) vs t (line style) in the window
- and use lines between to guide the eyes
- use + , x as the marker
"""

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path("/tmp/matplotlib-cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402


def load_autocorrelation_csv(path: Path) -> dict[str, np.ndarray]:
    """Load autocorrelation data from CSV.

    Inputs:
        path: Path to the autocorrelation CSV file.
    Outputs:
        Dict with keys: lag, autocovariance, gamma, tau_int_running.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {
        "lag": np.asarray(data["lag"], dtype=np.int64),
        "gamma": np.asarray(data["gamma"], dtype=np.float64),
        "tau_int": np.asarray(data["tau_int_running"], dtype=np.float64),
    }


def choose_window(gamma: np.ndarray) -> int:
    """Find the lag where gamma first goes non-positive.

    Inputs:
        gamma: Normalized autocorrelation values.
    Outputs:
        Window lag index.
    """
    for lag in range(1, len(gamma)):
        if gamma[lag] <= 0.0:
            return lag - 1
    return len(gamma) - 1


def plot_autocorrelation(data: dict[str, np.ndarray], csv_path: Path) -> Path:
    """Generate autocorrelation plots.

    Inputs:
        data: Autocorrelation data with lag, gamma, tau_int.
        csv_path: Source CSV path (used to derive the PNG path).
    Outputs:
        Path to the saved PNG.
    """
    png_path = csv_path.with_suffix(".png")
    lag = data["lag"]
    gamma = data["gamma"]
    tau_int = data["tau_int"]
    window = choose_window(gamma)
    selected_tau = tau_int[window]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: Gamma(t) vs t ---
    ax1.plot(lag, gamma, "-", color="gray", linewidth=0.8, alpha=0.6)
    ax1.scatter(lag, gamma, marker="+", color="C0", s=40, zorder=3)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(r"$\Gamma(t)$")
    ax1.set_title("Normalized autocorrelation")
    ax1.legend([r"$\Gamma(t)$"], loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Right panel: Gamma(t) vs t (scatter) and exp(-t/tau_int) (line) in window ---
    decay = np.exp(-lag / selected_tau)

    ax2.plot(lag[: window + 1], gamma[: window + 1], "-", color="gray", linewidth=0.8, alpha=0.6)
    ax2.scatter(
        lag[: window + 1],
        gamma[: window + 1],
        marker="+",
        color="C0",
        s=40,
        zorder=3,
        label=r"$\Gamma(t)$",
    )
    ax2.plot(
        lag[: window + 1],
        decay[: window + 1],
        "-",
        color="C1",
        linewidth=1.5,
        label=r"$e^{-t/\tau_{\mathrm{int}}}$",
    )
    ax2.axvline(window, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"$\Gamma(t)$")
    ax2.set_title(f"Window $W = {window}$, $\\tau_{{\\mathrm{{int}}}} = {selected_tau:.2f}$")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Autocorrelation analysis — {csv_path.stem}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <autocorrelation_csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = load_autocorrelation_csv(csv_path)
    png_path = plot_autocorrelation(data, csv_path)
    print(f"Plot saved: {png_path}")


if __name__ == "__main__":
    main()
