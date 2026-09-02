#!/usr/bin/env python3
"""Run evaluation for multiple trained models.

This script searches for model zip files whose path starts with
"results/Aug_29_09" and executes ``eval.py`` for each of them.
It mirrors the manual command you normally run:

    python3 eval.py <model_path> --num-runs 100 --render

Adjust ``NUM_RUNS`` or add additional arguments as needed.
"""

import subprocess
import sys
import os
from pathlib import Path
import concurrent.futures

# Configuration
NUM_RUNS = "100"
RENDER = "--no-render"
# Pattern to locate model zip files. Adjust if your directory structure changes.
MODEL_PATTERN = "results/Aug_29_09*/models/*.zip"

def find_model_paths():
    """Return a sorted list of model .zip paths matching the pattern."""
    base = Path(__file__).parent
    # Use glob to find matching files
    matches = list(base.glob(MODEL_PATTERN))
    # Sort for deterministic ordering
    matches.sort()
    return matches

def run_eval(model_path: Path):
    """Run eval.py on a single model path and capture its output.

    The stdout and stderr of ``eval.py`` are written to a file inside the
    ``run_outputs`` directory. The filename is derived from the model zip name
    (e.g. ``model_weaving_seed_test_seed_9.txt``).
    """
    # Ensure the output directory exists
    output_dir = Path(__file__).parent / "run_outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir / f"{model_path.stem}.txt"

    cmd = [
        sys.executable,  # uses the current python interpreter (python3)
        "eval.py",
        str(model_path),
        "--num-runs", NUM_RUNS,
        RENDER,
    ]
    print(f"Running: {' '.join(cmd)} -> {output_file}")
    # Run the command, redirecting stdout+stderr to the file
    with open(output_file, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"❌ Evaluation failed for {model_path} with exit code {result.returncode}")
    else:
        print(f"✅ Completed {model_path}, output saved to {output_file}")

def main():
    model_paths = find_model_paths()
    if not model_paths:
        print("No model files found matching pattern.")
        sys.exit(1)
    # Limit to first 10 runs if more are found
    selected_paths = model_paths[:10]
    max_workers = os.cpu_count() or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_eval, mp): mp for mp in selected_paths}
        for future in concurrent.futures.as_completed(futures):
            # Exceptions are already printed inside run_eval; we just ensure they propagate
            future.result()

if __name__ == "__main__":
    main()
