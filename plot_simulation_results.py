"""Plot MPC/FAADP simulation metrics from CSV logs produced by farl.simulation.

Example:
    python plot_simulation_results.py \
        --sim-dir Results_dir/refNum9/.../simulationReal/sine \
        --metrics y-x:"X [m]":"Y [m]" distance-error-t-cum:"Time [s]":"Cumu. distance error" \
        --algorithm "ADP:" \
        --algorithm "MPC-9 w/o TC:-MPC-9_wo_TC" \
        --algorithm "MPC-9 w/ 1-step TC:-MPC-9_w_1-step_TC" \
        --algorithm "MPC-9 w/ 9-step TC:-MPC-9_w_9-step_TC"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METRICS = {
    "y-x": ("X [m]", "Y [m]"),
    "distance-error-t-cum": ("Time [s]", "Cumu. distance error"),
    "phi-error-t-cum": ("Time [s]", "Cumu. angle error"),
    "utility-t-cum": ("Time [s]", "Cost"),
}

DEFAULT_ALGORITHMS = {
    "FAADP": "",
    "MPC-9 w/o TC": "-MPC-9_wo_TC",
    "MPC-9 w/ 1-step TC": "-MPC-9_w_1-step_TC",
    "MPC-9 w/ 9-step TC": "-MPC-9_w_9-step_TC",
}


def _parse_metrics(values: Iterable[str] | None) -> Dict[str, Tuple[str, str]]:
    if not values:
        return DEFAULT_METRICS
    metrics: Dict[str, Tuple[str, str]] = {}
    for item in values:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Metric specification must be 'name:xlabel:ylabel' (got %s)" % item
            )
        metrics[parts[0]] = (parts[1], parts[2])
    return metrics


def _parse_algorithms(values: Iterable[str] | None) -> Dict[str, str]:
    if not values:
        return DEFAULT_ALGORITHMS
    mapping: Dict[str, str] = {}
    for item in values:
        if ":" not in item:
            raise ValueError(
                "Algorithm specification must be 'label:suffix' (got %s)" % item
            )
        label, suffix = item.split(":", 1)
        mapping[label] = suffix
    return mapping


def _read_curve(path: Path) -> Tuple[pd.Series, pd.Series]:
    data = pd.read_csv(path, header=None, skiprows=1)
    return data.iloc[0, :], data.iloc[1, :]


def plot_simulation_results(args: argparse.Namespace) -> None:
    sim_dirs = [Path(p).expanduser().resolve() for p in args.sim_dir]
    metrics = _parse_metrics(args.metrics)
    algorithms = _parse_algorithms(args.algorithm)
    for sim_dir in sim_dirs:
        if not sim_dir.exists():
            raise FileNotFoundError(sim_dir)
        for metric, (xlabel, ylabel) in metrics.items():
            fig, ax = plt.subplots(figsize=tuple(args.figsize))
            color_cycle = iter(["red", "darkorange", "limegreen", "blue", "purple", "black"])
            linestyle_cycle = iter(["-", "--", "-.", ":"])
            for label, suffix in algorithms.items():
                csv_path = sim_dir / f"{metric}{suffix}.csv"
                if not csv_path.exists():
                    print(f"[WARN] Missing file {csv_path}, skipping")
                    continue
                x, y = _read_curve(csv_path)
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=next(color_cycle, "black"),
                    linestyle=next(linestyle_cycle, "-"),
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(framealpha=0.4)
            ax.grid(True, linestyle="--", alpha=0.4)
            output = sim_dir / f"{metric}.pdf"
            plt.savefig(output, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Saved {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot simulation metrics")
    parser.add_argument(
        "--sim-dir",
        nargs="+",
        required=True,
        help="One or more simulation output folders (e.g. Results_dir/.../simulationReal/sine)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        help="Optional overrides of metric plots: name:xlabel:ylabel",
    )
    parser.add_argument(
        "--algorithm",
        nargs="*",
        help="Optional overrides for algorithm suffixes: label:suffix",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(6.0, 4.0),
        metavar=("W", "H"),
        help="Figure size in inches",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plot_simulation_results(args)


if __name__ == "__main__":
    main()
