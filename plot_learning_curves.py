"""Plot FAADP training curves from TensorBoard event files.

Example:
    python plot_learning_curves.py \
        --tag "DLC cost" \
        --runs N=1:Results_dir/refNum1/.../events.out.tfevents... \
        --runs N=1:Results_dir/refNum1/.../events.out.tfevents... \
        --runs N=9:Results_dir/refNum9/.../events.out.tfevents...

Each ``--runs`` argument has the form ``label:path`` and you may repeat labels
multiple times to aggregate several seeds for the same preview horizon.
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


def _parse_runs(values: Iterable[str]) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = defaultdict(list)
    for item in values:
        if ":" not in item:
            raise ValueError(f"Run specification '{item}' must use label:path format")
        label, path = item.split(":", 1)
        run_path = Path(path).expanduser().resolve()
        if not run_path.exists():
            raise FileNotFoundError(f"Event file not found: {run_path}")
        mapping[label].append(run_path)
    if not mapping:
        raise ValueError("At least one --runs entry is required")
    return mapping


def _smooth(values: List[float], weight: float) -> List[float]:
    if not values or weight <= 0:
        return values
    smoothed = []
    prev = values[0]
    for v in values:
        prev = prev * weight + v * (1 - weight)
        smoothed.append(prev)
    return smoothed


def _load_scalars(path: Path, tag: str, max_step: int | None) -> List[tuple[int, float]]:
    ea = event_accumulator.EventAccumulator(str(path))
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        raise KeyError(f"Tag '{tag}' not found in {path}")
    data = ea.Scalars(tag)
    return [
        (int(item.step), float(item.value))
        for item in data
        if max_step is None or item.step <= max_step
    ]


def plot_learning_curves(args: argparse.Namespace) -> None:
    runs = _parse_runs(args.runs)
    records = []
    for label, paths in runs.items():
        for path in paths:
            series = _load_scalars(path, args.tag, args.max_step)
            if not series:
                continue
            steps, values = zip(*series)
            steps = [s / args.step_scale for s in steps]
            if args.smoothing > 0:
                values = _smooth(list(values), args.smoothing)
            for s, v in zip(steps, values):
                records.append({"step": s, "value": v, "label": label})
    if not records:
        raise RuntimeError("No scalar data collected. Check tag and event paths.")

    df = pd.DataFrame.from_records(records)
    sns.set_style("darkgrid")
    plt.figure(figsize=args.figsize)
    sns.lineplot(data=df, x="step", y="value", hue="label")
    plt.xlabel(args.x_label)
    plt.ylabel(args.tag)
    if args.y_limits:
        plt.ylim(args.y_limits)
    plt.legend(title=None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FAADP learning curves")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of label:path specifications for TensorBoard event files",
    )
    parser.add_argument(
        "--tag",
        default="DLC cost",
        help="Scalar tag to extract from the event files (default: %(default)s)",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="Maximum training step to include (default: full length)",
    )
    parser.add_argument(
        "--step_scale",
        type=float,
        default=1000.0,
        help="Divide raw steps by this factor when plotting (default: %(default)s)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Exponential smoothing weight (0 disables smoothing)",
    )
    parser.add_argument(
        "--output",
        default="learning_curve.png",
        help="Output image path (PNG/PDF)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(6.0, 4.0),
        metavar=("W", "H"),
        help="Figure size in inches (default: 6 4)",
    )
    parser.add_argument(
        "--x-label",
        dest="x_label",
        default="Thousand Iterations",
        help="Label for the x-axis",
    )
    parser.add_argument(
        "--y-limits",
        dest="y_limits",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional y-axis limits",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plot_learning_curves(args)


if __name__ == "__main__":
    main()
