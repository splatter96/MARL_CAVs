#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Full sweep grid. Keeping these explicit ensures missing runs still appear
# as blank cells in the 5x5 heatmaps.
COLLISION_REWARDS = [0, 50, 100, 200, 400]
OFFRAMP_REWARDS = [0, 25, 50, 100, 200]

#sns.set_theme()
#sns.set_context("paper")
sns.set(font_scale=1.2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create heatmaps showing mean ± standard deviation for performance "
            "and safety metrics over multiple evaluation seeds."
        )
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="collision_offramp_evaluation.csv",
        help="Evaluation CSV containing one row per model/evaluation seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="heatmaps_mean_std_safety",
        help="Directory in which the generated figures are saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure interactively in addition to saving it.",
    )
    return parser.parse_args()


def make_stat_grids(df: pd.DataFrame, value_column: str):
    """
    Calculate mean and sample standard deviation for one metric for every
    collision/off-ramp reward combination.

    Rows:    off-ramp reward
    Columns: collision reward

    Missing reward combinations remain NaN and are left blank in the heatmap.
    """
    grouped = df.groupby(
        ["offramp_reward", "collision_reward"],
        observed=False,
    )[value_column]

    mean_grid = grouped.mean().unstack("collision_reward")
    std_grid = grouped.std(ddof=1).unstack("collision_reward")
    count_grid = grouped.count().unstack("collision_reward")

    mean_grid = mean_grid.reindex(
        index=OFFRAMP_REWARDS,
        columns=COLLISION_REWARDS,
    )
    std_grid = std_grid.reindex(
        index=OFFRAMP_REWARDS,
        columns=COLLISION_REWARDS,
    )
    count_grid = count_grid.reindex(
        index=OFFRAMP_REWARDS,
        columns=COLLISION_REWARDS,
    )

    return mean_grid, std_grid, count_grid


def make_annotations(
    mean_grid: pd.DataFrame,
    std_grid: pd.DataFrame,
    decimals: int,
) -> pd.DataFrame:
    """Create strings of the form 'mean\n± std' for heatmap annotations."""
    annotations = pd.DataFrame(
        "",
        index=mean_grid.index,
        columns=mean_grid.columns,
        dtype=object,
    )

    for row in mean_grid.index:
        for col in mean_grid.columns:
            mean = mean_grid.loc[row, col]
            std = std_grid.loc[row, col]

            if pd.isna(mean):
                # Missing reward combination: keep cell blank.
                continue

            # If only one evaluation exists, pandas cannot compute a sample
            # standard deviation. Show NaN explicitly instead of silently
            # treating it as zero.
            if pd.isna(std):
                annotations.loc[row, col] = f"{mean:.{decimals}f}\n± n/a"
            else:
                annotations.loc[row, col] = (
                    f"{mean:.{decimals}f}\n± {std:.{decimals}f}"
                )

    return annotations


def plot_heatmap(
    mean_grid: pd.DataFrame,
    std_grid: pd.DataFrame,
    title: str,
    colorbar_label: str,
    decimals: int,
    output_file: Path,
    show: bool = False,
):
    # Color encodes the mean. Text annotation shows mean ± std.
    missing_mask = mean_grid.isna()
    annotations = make_annotations(mean_grid, std_grid, decimals)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    sns.heatmap(
        mean_grid,
        mask=missing_mask,
        annot=annotations,
        fmt="",
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": colorbar_label},
        ax=ax,
    )

    ax.set_xlabel("Collision penalty")
    ax.set_ylabel("Off-ramp penalty")
    ax.set_title(title)

    # Preserve the complete 5x5 grid even when combinations are missing.
    ax.set_xlim(0, len(COLLISION_REWARDS))
    #ax.set_ylim(len(OFFRAMP_REWARDS), 0)
    ax.set_ylim(0, len(OFFRAMP_REWARDS))

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    fig.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def print_metric_summary(name, mean_grid, std_grid, count_grid):
    print(f"{name} mean:")
    print(mean_grid)
    print()
    print(f"{name} standard deviation:")
    print(std_grid)
    print()
    print(f"{name} number of evaluations per cell:")
    print(count_grid)
    print()


def main():
    args = parse_args()

    csv_file = Path(args.csv_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)

    required_columns = {
        "collision_reward",
        "offramp_reward",
        "evaluation_seed",
        "crashrate",
        "mergerate",
        "ego_speed",
        "ttc_current_front_below_1_5_pct",
        "ttc_target_front_below_1_5_pct",
        "ttc_target_rear_below_1_5_pct",
        "ttc_any_below_1_5_pct",
        "target_rear_induced_braking",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    metrics = [
        # Existing performance metrics.
        {
            "column": "crashrate",
            "title": "Collision Rate",
            "colorbar": "Mean collision rate",
            "decimals": 3,
            "filename": "collision_rate_heatmap_mean_std.png",
        },
        {
            "column": "mergerate",
            "title": "Merge Rate",
            "colorbar": "Mean merge rate",
            "decimals": 3,
            "filename": "merge_rate_heatmap_mean_std.png",
        },
        {
            "column": "ego_speed",
            "title": "Average Ego-Vehicle Speed",
            "colorbar": "Mean ego speed",
            "decimals": 2,
            "filename": "ego_speed_heatmap_mean_std.png",
        },

        # TTC safety metrics. The evaluator already stores these values as the
        # percentage of all evaluation steps for which TTC < 1.5 s. Therefore
        # the heatmaps show the threshold-violation percentage rather than the
        # mean finite TTC value.
        {
            "column": "ttc_current_front_below_1_5_pct",
            "title": "Current-Lane Front TTC < 1.5 s",
            "colorbar": "Mean TTC violation rate [%]",
            "decimals": 2,
            "filename": "ttc_current_front_violation_heatmap_mean_std.png",
        },
        {
            "column": "ttc_target_front_below_1_5_pct",
            "title": "Target-Lane Front TTC < 1.5 s",
            "colorbar": "Mean TTC violation rate [%]",
            "decimals": 2,
            "filename": "ttc_target_front_violation_heatmap_mean_std.png",
        },
        {
            "column": "ttc_target_rear_below_1_5_pct",
            "title": "Target-Lane Rear TTC < 1.5 s",
            "colorbar": "Mean TTC violation rate [%]",
            "decimals": 2,
            "filename": "ttc_target_rear_violation_heatmap_mean_std.png",
        },
        {
            "column": "ttc_any_below_1_5_pct",
            "title": "Any Weaving TTC < 1.5 s",
            "colorbar": "Mean TTC violation rate [%]",
            "decimals": 2,
            "filename": "ttc_any_violation_heatmap_mean_std.png",
        },

        # Interaction metric for the vehicle behind the ego on the target lane.
        {
            "column": "target_rear_induced_braking",
            "title": "Induced Braking of Target-Lane Rear Vehicle",
            "colorbar": "Mean induced braking [m/s²]",
            "decimals": 2,
            "filename": "target_rear_induced_braking_heatmap_mean_std.png",
        },
    ]

    for metric in metrics:
        mean_grid, std_grid, count_grid = make_stat_grids(
            df, metric["column"]
        )

        print_metric_summary(
            metric["title"],
            mean_grid,
            std_grid,
            count_grid,
        )

        plot_heatmap(
            mean_grid=mean_grid,
            std_grid=std_grid,
            title=metric["title"],
            colorbar_label=metric["colorbar"],
            decimals=metric["decimals"],
            output_file=output_dir / metric["filename"],
            show=args.show,
        )

    print(f"Saved heatmaps to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
