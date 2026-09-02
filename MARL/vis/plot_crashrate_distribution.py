#!/usr/bin/env python3
"""
Plot the distribution of the final crash‑rate (third‑to‑last line)
across all evaluation output files in ``eval_outputs``.
"""

import os
import re
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Folder that contains the *.txt output files produced by the evaluation scripts
OUTPUT_DIR = Path(__file__).parent / "eval_outputs"

# Pattern used in the output files, e.g.
#   "Crashrate 0.0 Mergerate 0.5 other_crashes 0.0"
CRASHRATE_REGEX = re.compile(r"Crashrate\s+([0-9]*\.?[0-9]+)")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def extract_crashrate_from_file(filepath: Path) -> float:
    """Return the crash‑rate value from the third‑to‑last line of *filepath*.

    Raises
    ------
    ValueError
        If the file does not contain enough lines or the pattern cannot be found.
    """
    with filepath.open("r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"File '{filepath}' has fewer than 3 lines.")

    # third‑to‑last line (Python negative indexing)
    target_line = lines[-3].strip()
    match = CRASHRATE_REGEX.search(target_line)
    if not match:
        raise ValueError(f"Could not find crashrate in line: '{target_line}' (file: {filepath})")
    return float(match.group(1))


def collect_crashrates(folder: Path) -> List[float]:
    """Iterate over all ``*.txt`` files in *folder* and return a list of crash‑rates."""
    crashrates = []
    for txt_file in sorted(folder.glob("*.txt")):
        try:
            cr = extract_crashrate_from_file(txt_file)
            crashrates.append(cr)
        except Exception as e:
            print(f"[WARN] Skipping '{txt_file.name}': {e}", file=sys.stderr)
    return crashrates


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main() -> None:
    if not OUTPUT_DIR.is_dir():
        sys.exit(f"⚠️  Directory not found: {OUTPUT_DIR}")

    crashrates = collect_crashrates(OUTPUT_DIR)

    if not crashrates:
        sys.exit("⚠️  No crash‑rate values could be extracted.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    sns.histplot(crashrates, kde=True, bins=20, color="#4c72b0", edgecolor="k")
    plt.title("Distribution of final Crashrate (3rd‑to‑last line) across runs")
    plt.xlabel("Crashrate")
    plt.ylabel("Count")
    plt.tight_layout()

    # Save the figure next to the script for convenience
    out_path = Path(__file__).parent / "crashrate_distribution.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Plot saved to: {out_path}")

    # Optionally also show the figure (useful during interactive work)
    plt.show()


if __name__ == "__main__":
    main()