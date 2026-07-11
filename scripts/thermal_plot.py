"""
README: 这个脚本实现从热化的计算结果，也就是cold/ hot start下的每一个sweep之后得到的结果中，
- 画图，查看最终是否达到了稳定。
- 丢弃热化区间之后，稳定部分的平均值在统计误差内应该一样。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_data(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    starts = {}
    for start_name in ("cold", "hot"):
        mask = data["start"] == start_name
        starts[start_name] = {
            "sweep": data["sweep"][mask].astype(int),
            "plaquette": data["average_plaquette"][mask].astype(float),
        }
    return starts


def plot(data: dict[str, np.ndarray], csv_path: Path) -> Path:
    png_path = csv_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10, 6))

    for (start_name, color, marker, label) in [
        ("cold", "C0", "+", "Cold start"),
        ("hot", "C1", "x", "Hot start"),
    ]:
        d = data[start_name]
        ax.scatter(d["sweep"], d["plaquette"], color=color, marker=marker, label=label, s=30, alpha=0.7)

    ax.set_xlabel("Sweep")
    ax.set_ylabel("Average plaquette")
    ax.set_title(f"Thermalization check — {csv_path.stem}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = load_data(csv_path)

    for start_name in ("cold", "hot"):
        d = data[start_name]
        print(f"  [{start_name}] {len(d['sweep'])} sweeps, "
              f"final plaquette: {d['plaquette'][-1]:.6f}")

    png_path = plot(data, csv_path)
    print(f"Plot saved: {png_path}")


if __name__ == "__main__":
    main()
