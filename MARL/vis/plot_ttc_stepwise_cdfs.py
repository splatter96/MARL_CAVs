#!/usr/bin/env python3
"""
Plot step-wise TTC CDFs from safety_margin_metrics.npz files.

This script uses the raw per-step TTC arrays saved by the evaluation script:
    - ttc_current_front
    - ttc_target_front
    - ttc_target_rear

It produces a 2x2 figure with empirical CDFs for:
    1) current-lane front TTC
    2) target-lane front TTC
    3) target-lane rear TTC
    4) per-step minimum TTC across the three interaction partners

Important:
The CDFs are normalized by the TOTAL number of simulation steps, including
steps for which the TTC is +inf (no closing conflict). Since +inf cannot be
shown on a finite x-axis, those steps are not plotted explicitly, but they are
kept in the denominator. Consequently, the CDF value at a threshold x equals
the proportion of all simulation steps with TTC <= x.

This means that, for the "Per-step minimum TTC" subplot, the CDF value at the
TTC threshold directly corresponds to the table metric
"Steps Below T_TTC [%]" (up to <= vs. < at the threshold).

Example:
    python plot_ttc_stepwise_cdfs.py \
        --inputs sacd/safety_margin_metrics.npz \
                 ppo/safety_margin_metrics.npz \
                 idm/safety_margin_metrics.npz \
        --labels SACD PPO "IDM+MOBIL" \
        --output ttc_stepwise_cdfs.png \
        --ttc-threshold 1.5 \
        --x-max 10.0
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot step-wise TTC CDFs from safety_margin_metrics.npz files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more safety_margin_metrics.npz files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels corresponding to the input NPZ files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ttc_stepwise_cdfs.png",
        help="Output image path (default: ttc_stepwise_cdfs.png).",
    )
    parser.add_argument(
        "--ttc-threshold",
        type=float,
        default=1.5,
        help="Critical TTC threshold shown as a vertical reference line (default: 1.5 s).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=10.0,
        help="Maximum x-axis value in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def load_ttc_arrays(npz_path):
    data = np.load(npz_path)

    required_keys = [
        "ttc_current_front",
        "ttc_target_front",
        "ttc_target_rear",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {', '.join(missing)}")

    current_front = np.asarray(data["ttc_current_front"], dtype=float)
    target_front = np.asarray(data["ttc_target_front"], dtype=float)
    target_rear = np.asarray(data["ttc_target_rear"], dtype=float)

    if not (current_front.shape == target_front.shape == target_rear.shape):
        raise ValueError(
            f"The TTC arrays in {npz_path} do not have matching shapes."
        )

    return current_front, target_front, target_rear


def cdf_over_all_steps(values, x_max=None):
    """
    Return x/y arrays for a CDF normalized by the total number of valid steps.

    NaN values are excluded entirely.
    +inf values remain in the denominator but are not explicitly plotted.
    Thus, the final y value may be below 1.0 if some steps have infinite TTC.
    """
    values = np.asarray(values, dtype=float)

    valid_mask = ~np.isnan(values)
    values = values[valid_mask]
    total = values.size

    if total == 0:
        return np.array([]), np.array([]), 0, 0, 0

    finite_values = values[np.isfinite(values)]
    finite_values = np.sort(finite_values)

    n_finite = finite_values.size
    n_infinite = int(np.isinf(values).sum())

    if n_finite == 0:
        return np.array([]), np.array([]), total, n_finite, n_infinite

    if x_max is not None:
        finite_values = finite_values[finite_values <= x_max]

    n_shown = finite_values.size
    if n_shown == 0:
        return np.array([]), np.array([]), total, n_finite, n_infinite

    y = np.arange(1, n_shown + 1, dtype=float) / total
    # y = np.arange(1, n_shown + 1, dtype=float) / n_finite
    return finite_values, y, total, n_finite, n_infinite


def plot_metric(ax, all_series, labels, title, threshold, x_max):
    for values, label in zip(all_series, labels):
        x, y, total, n_finite, n_infinite = cdf_over_all_steps(values, x_max=x_max)

        if total == 0:
            print(f"{label} | {title}: no valid TTC values.")
            continue

        print(
            f"{label} | {title}: total steps={total}, "
            f"finite={n_finite}, infinite={n_infinite}"
        )

        if x.size == 0:
            continue

        ax.step(x, y, where="post", label=label)

    ax.axvline(
        x=threshold,
        color="0.4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("TTC [s]")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(0.0, x_max)
    # ax.set_ylim(0.0, 1.0)
    ax.set_ylim(0.0, 0.16)
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("The number of --inputs and --labels must match.")

    current_front_series = []
    target_front_series = []
    target_rear_series = []
    overall_min_series = []

    for npz_path in args.inputs:
        current_front, target_front, target_rear = load_ttc_arrays(npz_path)

        current_front_series.append(current_front)
        target_front_series.append(target_front)
        target_rear_series.append(target_rear)
        overall_min_series.append(
            np.minimum.reduce([current_front, target_front, target_rear])
        )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()

    plot_metric(
        axes[0],
        current_front_series,
        args.labels,
        "Current-lane front TTC",
        args.ttc_threshold,
        args.x_max,
    )
    plot_metric(
        axes[1],
        target_front_series,
        args.labels,
        "Target-lane front TTC",
        args.ttc_threshold,
        args.x_max,
    )
    plot_metric(
        axes[2],
        target_rear_series,
        args.labels,
        "Target-lane rear TTC",
        args.ttc_threshold,
        args.x_max,
    )
    plot_metric(
        axes[3],
        overall_min_series,
        args.labels,
        "Per-step minimum TTC",
        args.ttc_threshold,
        args.x_max,
    )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    threshold_handle = plt.Line2D(
        [0], [0], color="0.4", linestyle="--", linewidth=1.2, label="TTC threshold"
    )
    handles.append(threshold_handle)
    legend_labels.append("TTC threshold")
    axes[0].legend(handles, legend_labels)

    plt.tight_layout()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
