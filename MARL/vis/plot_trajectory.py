import os
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description=('Plot two different runs against each other'))
    parser.add_argument('runs', type=str, help="folder for experiment1")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    sns.set_theme()

    runs_list = []
    for f in os.listdir(args.runs):
        data = np.load(f"{args.runs}/{f}")
        df = pd.DataFrame(data, columns=["x", "y"])
        ax = sns.lineplot(data=df, x="x", y="y")

    # add information
    ax.axvline(x=150)
    ax.axvline(x=230)
    ax.axvline(x=310)
    ax.axvline(x=460)

    ax.axhline(y=0, linestyle='dashed', color='red')
    ax.axhline(y=4, linestyle='dashed', color='red')

    ax.text(375, 14, 'Main')
    ax.text(250, 14, 'Parallel')
    ax.text(175, 14, 'Diagonal')
    ax.text(120, 13, 'Straight')

    ax.invert_yaxis()

    matplotlib.pyplot.show()
