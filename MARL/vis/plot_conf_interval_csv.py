import os
import glob
import matplotlib
import argparse
import seaborn as sns
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description=('Plot two different runs against each other'))
    parser.add_argument('run1', type=str, help="folder for experiment1")
    parser.add_argument('run2', type=str, help="folder for experiment2")
    parser.add_argument('metric', type=str, help="the metric to use for plotting")
    parser.add_argument('--smooth', type=int, help="ammount of samples to smooth together using moving average", default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    sns.set_theme()

    run1_list = []
    for f in glob.iglob(args.run1 + "**/evaluation.csv", recursive=True):
        if 'evaluation' in f:
            df = pd.read_csv(f)
            if args.smooth > 0:
                #do averaging (moving average)
                df[args.metric] = df[args.metric].rolling(args.smooth).mean()

            run1_list.append(df)

    curr_runs = pd.concat(run1_list)
    # remove duplicates
    curr_runs = curr_runs.reset_index()

    sns.lineplot(data=curr_runs, x="steps", y=args.metric, label='curriculum learning')

    run2_list = []
    for f in glob.iglob(args.run2 + "**/evaluation.csv", recursive=True):
        if 'evaluation' in f:
            df = pd.read_csv(f)
            if args.smooth > 0:
                #do averaging (moving average)
                df[args.metric] = df[args.metric].rolling(args.smooth).mean()
            run2_list.append(df)

    direct_runs = pd.concat(run2_list)

    # remove duplicates
    direct_runs = direct_runs.reset_index()

    ax = sns.lineplot(data=direct_runs, x="steps", y=args.metric, label='direct learning')
    ax.set(xlabel='time', ylabel=f"{args.metric}", title=f"{args.metric} over time")

    # mark points where curriculum changed env
    ax.axvline(x=3e5)
    ax.axvline(x=6e5)

    matplotlib.pyplot.show()
