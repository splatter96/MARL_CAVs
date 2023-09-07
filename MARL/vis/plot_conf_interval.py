import os
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
    for f in os.listdir(args.run1):
        if args.metric in f:
            df = pd.read_csv(f"{args.run1}/{f}")
            if args.smooth > 0:
                #do averaging (moving average)
                df['value'] = df['value'].ewm(span=args.smooth).mean()
            run1_list.append(df)
    curr_runs = pd.concat(run1_list)

    sns.lineplot(data=curr_runs, x="index", y="value", label='curriculum learning')

    run2_list = []
    for f in os.listdir(args.run2):
        if args.metric in f:
            df = pd.read_csv(f"{args.run2}/{f}")
            if args.smooth > 0:
                #do averaging (moving average)
                df['value'] = df['value'].ewm(span=args.smooth).mean()
            run2_list.append(df)
    direct_runs = pd.concat(run2_list)

    ax = sns.lineplot(data=direct_runs, x="index", y="value", label='direct learning')

    ax.set(xlabel='time', ylabel=f"{args.metric}", title=f"{args.metric} over time")

    matplotlib.pyplot.show()
