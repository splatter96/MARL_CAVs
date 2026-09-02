import glob
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
from matplotlib import ticker

def fmt_two_digits(x, pos):
    x /= 1e6
    return f"{x:.1f}"

def parse_args():
    parser = argparse.ArgumentParser(
        description=("Plot two different runs against each other")
    )
    parser.add_argument("run1", type=str, help="folder for experiment1")
    parser.add_argument("run2", type=str, help="folder for experiment2")
    parser.add_argument("metric", type=str, help="the metric to use for plotting")
    parser.add_argument(
        "--smooth",
        type=int,
        help="ammount of samples to smooth together using moving average",
        default=0,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    sns.set_theme()
    sns.set_context("paper")
    sns.set(font_scale=1.1)

    run1_list = []
    for f in glob.iglob(args.run1 + "**/evaluation.csv", recursive=True):
        print(f)
        if "evaluation" in f:
            df = pd.read_csv(f)
            if args.smooth > 0:
                # do averaging (moving average)
                df[args.metric] = df[args.metric].rolling(args.smooth).mean()

            run1_list.append(df)

    curr_runs = pd.concat(run1_list)
    # remove duplicates
    curr_runs = curr_runs.reset_index()

    sns.lineplot(data=curr_runs, x="steps", y=args.metric, label="curriculum learning")

    run2_list = []
    for f in glob.iglob(args.run2 + "**/evaluation.csv", recursive=True):
        if "evaluation" in f:
            df = pd.read_csv(f)
            if args.smooth > 0:
                # do averaging (moving average)
                df[args.metric] = df[args.metric].rolling(args.smooth).mean()
            run2_list.append(df)

    direct_runs = pd.concat(run2_list)

    # remove duplicates
    direct_runs = direct_runs.reset_index()

    ax = sns.lineplot(
        data=direct_runs, x="steps", y=args.metric, label="direct learning"
    )
    # ax.set(xlabel=f"Training steps [1e6]", ylabel="Return")#, title="Return over time")
    ax.set(xlabel="Training steps [1e6]", ylabel="Average vehicle speed [m/s]")#, title="Average vehicle speed over time",)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_two_digits))

    # mark points where curriculum changed env
    # ax.axvline(x=3e5)
    # ax.axvline(x=6e5)
    ax.axvline(x=5e5, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(x=10e5, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8)

    #ax.figure.savefig("reward_paper_resubmission.png", dpi=300, bbox_inches="tight")
    ax.figure.savefig("ego_speed_paper_resubmission.png", dpi=300, bbox_inches="tight")

    # matplotlib.pyplot.show()
