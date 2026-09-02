#!/usr/bin/env python3
"""Batch-evaluate all collision/off-ramp reward sweep agents.

The script searches a results directory for run folders whose names match a
pattern (by default ``*collision_offramp_sweep*``). For every run it:

1. reads ``configs/args`` to recover the W&B sweep overrides;
2. extracts ``env.collision_reward`` and ``env.offramp_reward`` from those args;
3. optionally reads ``configs/config_sacd.yml`` for metadata such as the training seed;
4. finds the trained ``models/*.zip`` file;
5. evaluates each model repeatedly with several evaluation seeds, using the
   same scenario seeds for every model within each repetition; and
6. aggregates the step-wise TTC/induced-braking safety metrics; and
7. writes one row per (run, evaluation seed) pair to a single CSV file.

Example:
    python eval_collision_offramp_batch.py \
        --results-dir results \
        --run-pattern '*collision_offramp_sweep*' \
        --num-runs 1000 \
        --eval-seed 12345 \
        --num-eval-seeds 5 \
        --num-workers 4 \
        --output collision_offramp_evaluation.csv
"""

import argparse
import csv
import multiprocessing as mp
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import yaml
from tqdm import tqdm


# Keep the path handling from the original evaluation script, but make it safe
# when one of these paths is not currently present in sys.path.
for path in (
    "/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env",
    "/home/paul/Documents/PhD/RL/highway_env_commonroad/highway-env",
):
    if path in sys.path:
        sys.path.remove(path)

sys.path.append("../highway-env")
import highway_env  # noqa: F401, E402
from sb3_contrib import SACD  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all SACD models from a collision/off-ramp reward sweep."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing the individual training-run folders.",
    )
    parser.add_argument(
        "--run-pattern",
        type=str,
        default="*collision_offramp_sweep*",
        help="Glob pattern used to select run folders inside --results-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("collision_offramp_evaluation.csv"),
        help="Output CSV file.",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=3,
        help="Traffic-density/difficulty used during evaluation.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=200,
        help="Number of evaluation episodes per trained model.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=32,
        help=(
            "Base seed for the first evaluation repetition. For repetition r, "
            "the base scenario seed is eval_seed + r * num_runs, so the "
            "episode-seed blocks do not overlap. The same seed blocks are "
            "used for every model."
        ),
    )
    parser.add_argument(
        "--num-eval-seeds",
        type=int,
        default=5,
        help=(
            "Number of independent evaluation repetitions per trained model. "
            "Each repetition produces one CSV row. Default: 5."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of trained models to evaluate in parallel. "
            "Use 1 for sequential evaluation."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device passed to SACD.load in each worker, e.g. 'cpu', 'cuda', "
            "or 'auto'. For many parallel workers, CPU is often preferable."
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes. Disabled by default for batch evaluation.",
    )
    parser.add_argument(
        "--merging",
        action="store_true",
        help="Use the merging environment instead of the weaving environment.",
    )
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Dict:
    """Load the SACD YAML config stored in one result directory.

    The W&B sweep overrides are not expected to be reflected in this file; it
    is used only for metadata such as the training seed.
    """
    config_path = run_dir / "configs" / "config_sacd.yml"
    if not config_path.exists():
        return {}

    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")

    return config


def load_sweep_args(run_dir: Path) -> Dict[str, object]:
    """Parse the per-run ``configs/args`` file written by the W&B sweep.

    Example file::

        train_new.py env.collision_reward=0 env.offramp_reward=25

    The first token is the training program. Remaining ``key=value`` tokens are
    parsed into a dictionary. ``yaml.safe_load`` is used for the value so that
    numbers, booleans, and strings retain sensible Python types.
    """
    args_path = run_dir / "configs" / "args"
    if not args_path.exists():
        raise FileNotFoundError(f"Sweep args file not found: {args_path}")

    text = args_path.read_text().strip()
    if not text:
        raise ValueError(f"Sweep args file is empty: {args_path}")

    tokens = shlex.split(text)
    overrides: Dict[str, object] = {}

    # The first token is normally the training script name. Ignore every token
    # without '=' so this also works if the command contains extra flags.
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        overrides[key] = yaml.safe_load(value)

    return overrides


def find_model(run_dir: Path) -> Path:
    """Find the trained model ZIP in a run directory."""
    model_dir = run_dir / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_files = sorted(model_dir.glob("*.zip"))

    if len(model_files) == 1:
        return model_files[0]

    # Prefer the naming convention described for the sweep models.
    named_models = sorted(model_dir.glob("model_*.zip"))
    if len(named_models) == 1:
        return named_models[0]

    if not model_files:
        raise FileNotFoundError(f"No .zip model found in {model_dir}")

    raise RuntimeError(
        f"Found multiple model ZIP files in {model_dir}: "
        + ", ".join(path.name for path in model_files)
    )


def make_env(args):
    """Create the evaluation environment."""
    config = {
        "screen_height": 300,
        "screen_width": 2800,
        "safety_guarantee": False,
        "traffic_density": args.difficulty,
    }

    if args.merging:
        config["use_weaving"] = False

    return gym.make("merge-single-agent-v0", config=config)


def evaluate_model(model, env, args, evaluation_seed: int) -> Dict[str, float]:
    """Evaluate one loaded model for one evaluation repetition.

    ``evaluation_seed`` is the base seed for this repetition. Episode i uses
    ``evaluation_seed + i``. Every trained model receives the same seed blocks.
    """
    # deterministic=True is used below, but setting the model seed also keeps
    # any remaining internal stochasticity reproducible.
    model.set_random_seed(evaluation_seed)

    crashes = 0
    other_crashes = 0
    successful_merges = 0
    ego_speed_sum = 0.0
    network_speed_sum = 0.0
    total_steps = 0

    # Safety metrics are reported by the custom environment at every step.
    # TTC is np.inf when there is no closing conflict, and induced braking is
    # np.nan when no applicable target-lane rear vehicle exists.  Therefore,
    # only finite values are included in the evaluation means.
    safety_metric_names = (
        "ttc_current_front",
        "ttc_target_front",
        "ttc_target_rear",
        "target_rear_induced_braking",
    )
    safety_metric_sums = {key: 0.0 for key in safety_metric_names}
    safety_metric_counts = {key: 0 for key in safety_metric_names}

    # TTC violation metrics. Percentages are calculated over *all* evaluated
    # simulation steps. np.inf therefore naturally represents a safe/non-closing
    # situation and does not count as a violation.
    ttc_threshold = 1.5  # [s]
    ttc_metric_names = (
        "ttc_current_front",
        "ttc_target_front",
        "ttc_target_rear",
    )
    ttc_violation_counts = {key: 0 for key in ttc_metric_names}
    ttc_any_violation_count = 0

    start = time.time()

    for episode_idx in range(args.num_runs):
        done = False
        truncated = False

        # Every model is evaluated on the same sequence of traffic scenarios.
        episode_seed = evaluation_seed + episode_idx
        obs, info = env.reset(seed=episode_seed)

        if args.render:
            env.render()

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            ego_speed_sum += float(info["average_speed"])
            network_speed_sum += float(info["average_road_speed"])
            total_steps += 1

            # Collect the new safety metrics. Fail explicitly if the running
            # environment is not the updated version that reports them.
            missing_safety_metrics = [
                key for key in safety_metric_names if key not in info
            ]
            if missing_safety_metrics:
                raise KeyError(
                    "Environment info dict is missing safety metrics: "
                    + ", ".join(missing_safety_metrics)
                )

            # Infinite TTC values represent states in which the corresponding
            # vehicle pair is not closing and are intentionally excluded from
            # the mean. NaN braking values are excluded in the same way.
            step_has_ttc_violation = False
            for metric_name in safety_metric_names:
                metric_value = info[metric_name]
                try:
                    metric_value = float(metric_value)
                except (TypeError, ValueError):
                    continue

                if np.isfinite(metric_value):
                    safety_metric_sums[metric_name] += metric_value
                    safety_metric_counts[metric_name] += 1

                    # Count TTC threshold violations for the three TTC metrics.
                    # The denominator is total_steps, not only finite TTC samples,
                    # so this represents the percentage of driving time spent in
                    # a critical TTC state.
                    if (
                        metric_name in ttc_violation_counts
                        and metric_value < ttc_threshold
                    ):
                        ttc_violation_counts[metric_name] += 1
                        step_has_ttc_violation = True

            if step_has_ttc_violation:
                ttc_any_violation_count += 1

            if args.render:
                env.render()

            # Preserve the behavior of the original evaluator: stop processing
            # this episode if a background vehicle crashes while the ego has not.
            if info.get("other_crashes", False) and not info.get("crashed", False):
                break

        crashed = bool(info.get("crashed", False))
        other_crashed = bool(info.get("other_crashes", False)) and not crashed
        merged = bool(info.get("merged", False))

        crashes += int(crashed)
        other_crashes += int(other_crashed)
        successful_merges += int(merged)

    elapsed = time.time() - start

    if total_steps == 0:
        raise RuntimeError("No simulation steps recorded while evaluating model")

    safety_metric_means = {
        metric_name: (
            safety_metric_sums[metric_name] / safety_metric_counts[metric_name]
            if safety_metric_counts[metric_name] > 0
            else np.nan
        )
        for metric_name in safety_metric_sums
    }

    return {
        "crashrate": crashes / args.num_runs,
        "mergerate": successful_merges / args.num_runs,
        "ego_speed": ego_speed_sum / total_steps,
        "network_speed": network_speed_sum / total_steps,
        "other_crashrate": other_crashes / args.num_runs,
        "ttc_current_front": safety_metric_means["ttc_current_front"],
        "ttc_target_front": safety_metric_means["ttc_target_front"],
        "ttc_target_rear": safety_metric_means["ttc_target_rear"],
        "target_rear_induced_braking": safety_metric_means[
            "target_rear_induced_braking"
        ],
        "ttc_current_front_below_1_5_pct": (
            100.0 * ttc_violation_counts["ttc_current_front"] / total_steps
        ),
        "ttc_target_front_below_1_5_pct": (
            100.0 * ttc_violation_counts["ttc_target_front"] / total_steps
        ),
        "ttc_target_rear_below_1_5_pct": (
            100.0 * ttc_violation_counts["ttc_target_rear"] / total_steps
        ),
        "ttc_any_below_1_5_pct": (
            100.0 * ttc_any_violation_count / total_steps
        ),
        "evaluation_seed": evaluation_seed,
        "evaluation_episodes": args.num_runs,
        "evaluation_seconds": elapsed,
    }


def extract_reward_values(sweep_args: Dict[str, object], args_path: Path):
    """Extract collision/off-ramp reward values from W&B sweep overrides."""
    collision_key = "env.collision_reward"
    offramp_key = "env.offramp_reward"

    if collision_key not in sweep_args:
        raise KeyError(f"{collision_key} missing in {args_path}")
    if offramp_key not in sweep_args:
        raise KeyError(f"{offramp_key} missing in {args_path}")

    return sweep_args[collision_key], sweep_args[offramp_key]


def extract_training_seed(config: Dict) -> Optional[int]:
    """Extract the training seed from the YAML metadata if available."""
    seed = config.get("seed")

    # Support both plain YAML values and W&B-style {value: ...} entries.
    if isinstance(seed, dict) and "value" in seed:
        seed = seed["value"]

    return seed


def evaluate_job(job: Dict) -> List[Dict]:
    """Evaluate one prepared run for all requested evaluation seeds.

    A worker loads the model once, then evaluates all repetitions sequentially.
    Parallelism therefore remains at the model level and avoids repeatedly
    loading the same model for each seed.
    """
    # Prevent every worker from spawning a large number of PyTorch CPU threads.
    torch.set_num_threads(1)

    eval_args = SimpleNamespace(**job["eval_options"])
    model_path = Path(job["model_path"])

    # Load the model and construct the environment once per worker/model.
    model = SACD.load(str(model_path), device=eval_args.device)
    env = make_env(eval_args)

    rows = []
    try:
        for evaluation_seed in job["evaluation_seeds"]:
            metrics = evaluate_model(model, env, eval_args, evaluation_seed)

            rows.append(
                {
                    "run_name": job["run_name"],
                    "training_seed": job["training_seed"],
                    "collision_reward": job["collision_reward"],
                    "offramp_reward": job["offramp_reward"],
                    "evaluation_seed": metrics["evaluation_seed"],
                    "crashrate": metrics["crashrate"],
                    "mergerate": metrics["mergerate"],
                    "ego_speed": metrics["ego_speed"],
                    "network_speed": metrics["network_speed"],
                    "other_crashrate": metrics["other_crashrate"],
                    "ttc_current_front": metrics["ttc_current_front"],
                    "ttc_target_front": metrics["ttc_target_front"],
                    "ttc_target_rear": metrics["ttc_target_rear"],
                    "target_rear_induced_braking": metrics[
                        "target_rear_induced_braking"
                    ],
                    "ttc_current_front_below_1_5_pct": metrics[
                        "ttc_current_front_below_1_5_pct"
                    ],
                    "ttc_target_front_below_1_5_pct": metrics[
                        "ttc_target_front_below_1_5_pct"
                    ],
                    "ttc_target_rear_below_1_5_pct": metrics[
                        "ttc_target_rear_below_1_5_pct"
                    ],
                    "ttc_any_below_1_5_pct": metrics[
                        "ttc_any_below_1_5_pct"
                    ],
                    "evaluation_episodes": metrics["evaluation_episodes"],
                    "evaluation_seconds": metrics["evaluation_seconds"],
                    "model_path": str(model_path),
                }
            )
    finally:
        env.close()

    return rows


def main():
    args = parse_args()

    run_dirs = sorted(
        path
        for path in args.results_dir.glob(args.run_pattern)
        if path.is_dir()
    )

    if not run_dirs:
        raise FileNotFoundError(
            f"No run folders matching '{args.run_pattern}' found in "
            f"{args.results_dir.resolve()}"
        )

    print(f"Found {len(run_dirs)} matching run folders.")
    print(
        f"Evaluating {args.num_eval_seeds} repetitions x {args.num_runs} "
        f"episodes per model."
    )

    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.num_eval_seeds < 1:
        raise ValueError("--num-eval-seeds must be at least 1")

    evaluation_seeds = [
        args.eval_seed + repetition * args.num_runs
        for repetition in range(args.num_eval_seeds)
    ]
    print(f"Evaluation repetition base seeds: {evaluation_seeds}")
    print(
        "For each repetition, episode i uses base_seed + i; the same seed "
        "blocks are used for every model."
    )
    print()

    eval_options = {
        "difficulty": args.difficulty,
        "num_runs": args.num_runs,
        "render": args.render,
        "merging": args.merging,
        "device": args.device,
    }

    jobs = []
    for run_idx, run_dir in enumerate(run_dirs, start=1):
        sweep_args_path = run_dir / "configs" / "args"

        try:
            sweep_args = load_sweep_args(run_dir)
            collision_reward, offramp_reward = extract_reward_values(
                sweep_args, sweep_args_path
            )
            train_config = load_run_config(run_dir)
            training_seed = extract_training_seed(train_config)
            model_path = find_model(run_dir)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            print(f"[{run_idx}/{len(run_dirs)}] Skipping {run_dir.name}: {exc}")
            continue

        jobs.append(
            {
                "run_name": run_dir.name,
                "training_seed": training_seed,
                "collision_reward": collision_reward,
                "offramp_reward": offramp_reward,
                "model_path": str(model_path),
                "evaluation_seeds": evaluation_seeds,
                "eval_options": eval_options,
            }
        )

    if not jobs:
        raise RuntimeError("No valid models were found for evaluation.")

    print(
        f"Evaluating {len(jobs)} models with {args.num_workers} parallel "
        f"worker(s) on device='{args.device}'."
    )
    if args.render and args.num_workers > 1:
        print(
            "Warning: rendering from multiple worker processes may be slow or "
            "may not work reliably. Prefer --num-workers 1 when rendering."
        )

    results = []

    if args.num_workers == 1:
        progress = tqdm(jobs, desc="Models evaluated", unit="model")
        for job in progress:
            try:
                rows = evaluate_job(job)
            except Exception as exc:
                print(f"\nFailed to evaluate {job['run_name']}: {exc}")
                continue
            results.extend(rows)
            mean_crash = np.mean([row["crashrate"] for row in rows])
            mean_merge = np.mean([row["mergerate"] for row in rows])
            progress.set_postfix(
                crash=f"{mean_crash:.3f}",
                merge=f"{mean_merge:.3f}",
            )
    else:
        # 'spawn' is safer than Linux 'fork' when PyTorch/CUDA has been imported.
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=mp_context,
        ) as executor:
            future_to_job = {
                executor.submit(evaluate_job, job): job for job in jobs
            }

            with tqdm(
                total=len(future_to_job),
                desc="Models evaluated",
                unit="model",
            ) as progress:
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        rows = future.result()
                    except Exception as exc:
                        print(f"\nFailed to evaluate {job['run_name']}: {exc}")
                    else:
                        results.extend(rows)
                        mean_crash = np.mean([row["crashrate"] for row in rows])
                        mean_merge = np.mean([row["mergerate"] for row in rows])
                        progress.set_postfix(
                            crash=f"{mean_crash:.3f}",
                            merge=f"{mean_merge:.3f}",
                        )
                    finally:
                        progress.update(1)

    if not results:
        raise RuntimeError("No models were successfully evaluated.")

    # Sort into a convenient order for later pivoting/heatmap generation.
    results.sort(
        key=lambda row: (
            float(row["collision_reward"]),
            float(row["offramp_reward"]),
            row["run_name"],
            int(row["evaluation_seed"]),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "training_seed",
        "collision_reward",
        "offramp_reward",
        "evaluation_seed",
        "crashrate",
        "mergerate",
        "ego_speed",
        "network_speed",
        "other_crashrate",
        "ttc_current_front",
        "ttc_target_front",
        "ttc_target_rear",
        "target_rear_induced_braking",
        "ttc_current_front_below_1_5_pct",
        "ttc_target_front_below_1_5_pct",
        "ttc_target_rear_below_1_5_pct",
        "ttc_any_below_1_5_pct",
        "evaluation_episodes",
        "evaluation_seconds",
        "model_path",
    ]

    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(
        f"Wrote {len(results)} evaluation rows "
        f"({len(results) // args.num_eval_seeds} models x "
        f"{args.num_eval_seeds} seeds) to: {args.output.resolve()}"
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
