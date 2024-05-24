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
    sns.set_context("paper")

    run1_list = []
    for f in glob.iglob(args.run1 + "**/evaluation.csv", recursive=True):
        print(f)
        if 'evaluation' in f:
            df = pd.read_csv(f)
            if args.smooth > 0:
                #do averaging (moving average)
                df[args.metric] = df[args.metric].rolling(args.smooth).mean()

            # df = df.iloc[::10]
            # sns.lineplot(data=df, x="steps", y=args.metric, label='curriculum learning')
            run1_list.append(df)
            # break

    curr_runs = pd.concat(run1_list)
    # remove duplicates
    curr_runs = curr_runs.reset_index()

    # curr_runs = curr_runs.iloc[::10]

    sns.lineplot(data=curr_runs, x="steps", y=args.metric, label='curriculum learning')

    run2_list = []
    for f in glob.iglob(args.run2 + "**/evaluation.csv", recursive=True):
        if 'evaluation' in f:
            df = pd.read_csv(f)
            if args.smooth > 0:
                #do averaging (moving average)
                df[args.metric] = df[args.metric].rolling(args.smooth).mean()
            run2_list.append(df)
            # jreak

    direct_runs = pd.concat(run2_list)

    # remove duplicates
    direct_runs = direct_runs.reset_index()
    # direct_runs = direct_runs.iloc[::10]

    ax = sns.lineplot(data=direct_runs, x="steps", y=args.metric, label='direct learning')
    # ax.set(xlabel='time steps', ylabel="return", title=f"return over time")
    ax.set(xlabel='time steps', ylabel="average vehicle speed [m/s]", title=f"average vehicle speed over time")

    # mark points where curriculum changed env
    ax.axvline(x=3e5)
    ax.axvline(x=6e5)

    ax.figure.savefig("out.png")

    matplotlib.pyplot.show()
