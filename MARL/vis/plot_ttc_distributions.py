#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("ttc_current_front.npy", "TTC to Current Front Vehicle", "ttc_current_front_distribution"),
    ("ttc_target_front.npy", "TTC to Target-Lane Front Vehicle", "ttc_target_front_distribution"),
    ("ttc_target_rear.npy", "TTC to Target-Lane Rear Vehicle", "ttc_target_rear_distribution"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TTC distributions from NumPy files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing the TTC .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        default="ttc_plots",
        help="Directory where the plots will be saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--max-ttc",
        type=float,
        default=None,
        help=(
            "Optional upper TTC limit for plotting. Values above this limit are "
            "excluded from the histogram. If omitted, all finite values are used."
        ),
    )
    parser.add_argument(
        "--min-ttc",
        type=float,
        default=None,
        help=(
            "Optional lower TTC limit for plotting. Values below this limit are "
            "excluded from the histogram. If omitted, all finite values are used."
        ),
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot normalized probability densities instead of raw counts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def prepare_values(values: np.ndarray, max_ttc=None, min_ttc=None):
    values = np.asarray(values).reshape(-1)
    finite_values = values[np.isfinite(values)]

    if max_ttc is not None:
        finite_values = finite_values[finite_values <= max_ttc]

    if min_ttc is not None:
        finite_values = finite_values[finite_values >= min_ttc]

    return finite_values


def plot_distribution(values, title, output_base: Path, bins=100, density=False, show=False):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    if values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite TTC values available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_xlabel("TTC [s]")
        ax.set_ylabel("Density" if density else "Count")
    else:
        ax.hist(values, bins=bins, density=density, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("TTC [s]")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, alpha=0.3)

        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f} s")
        ax.axvline(median_val, linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f} s")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, title, stem in METRICS:
        file_path = input_dir / filename
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        values = np.load(file_path, allow_pickle=False)
        finite_values = prepare_values(values, max_ttc=args.max_ttc, min_ttc=args.min_ttc)

        total_values = np.asarray(values).size
        finite_count = np.isfinite(values).sum()
        inf_count = np.isinf(values).sum()

        print(f"{filename}:")
        print(f"  total values:   {total_values}")
        print(f"  finite values:  {finite_count}")
        print(f"  infinite values:{inf_count}")
        if finite_values.size > 0:
            print(f"  mean:           {np.mean(finite_values):.3f} s")
            print(f"  median:         {np.median(finite_values):.3f} s")
            print(f"  std:            {np.std(finite_values):.3f} s")
            print(f"  min/max:        {np.min(finite_values):.3f} / {np.max(finite_values):.3f} s")
        print()

        plot_distribution(
            values=finite_values,
            title=title,
            output_base=output_dir / stem,
            bins=args.bins,
            density=args.density,
            show=args.show,
        )

    print(f"Saved TTC plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
