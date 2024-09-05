import os
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description=('Plot two different runs against each other'))
    parser.add_argument('runs', type=str, help="folder for experiment")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    sns.set_theme()

    # plot_until = 15 #17 #21 #29 #near miss
    plot_until = 19 #15 #17  #crash

    runs_list = []
    for f in glob.iglob(f"{args.runs}/pos_1.npy", recursive=True):
    # for f in glob.iglob(f"{args.runs}/pos_3.npy", recursive=True):
    # for f in glob.iglob(f"{args.runs}/pos_4.npy", recursive=True):
        data = np.load(f"{f}")
        df = pd.DataFrame(data[:plot_until, :], columns=["x", "y"])
        ax = sns.lineplot(data=df, x="x", y="y", marker="D", markevery=[-1])

    for f in glob.iglob(f"{args.runs}/other_pos_1.npy", recursive=True):
    # for f in glob.iglob(f"{args.runs}/other_pos_3.npy", recursive=True):
    # for f in glob.iglob(f"{args.runs}/other_pos_4.npy", recursive=True):
        data = np.load(f"{f}")
        for i in range(data.shape[1]):
            if i in [5,6]: #for run 1, crash
            # if i in [9, 3]: #for run 3, near miss
            # if i in [4,6]: #for run 4, nice merge
                df = pd.DataFrame(data[:plot_until,i,:], columns=["x", "y"])
                ax = sns.lineplot(data=df, x="x", y="y", marker="o", markevery=[-1])

    ax.invert_yaxis()
    ax.grid(False)

    import matplotlib.image as mpimg
    map_img = mpimg.imread('road.png')

    ax.imshow(map_img,
              aspect = ax.get_aspect(),
              extent = [150,460,16,-2],
              zorder = 0) #put the map under the plot

    # ax.set_xlim([225,300])

    # matplotlib.pyplot.show()
    plt.tight_layout()
    # plt.savefig("trajectory_crash.png", dpi=600, bbox_inches="tight")
    # plt.savefig("trajectory_near_miss.png", dpi=600, bbox_inches="tight")
    # plt.savefig("trajectory_nice_merge.png", dpi=600, bbox_inches="tight")

    plt.savefig("trajectory_crash_2.png", dpi=600, bbox_inches="tight")
