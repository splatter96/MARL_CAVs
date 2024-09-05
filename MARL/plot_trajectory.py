import os
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

def parse_args():
    parser = argparse.ArgumentParser(description=('Plot two different runs against each other'))
    parser.add_argument('runs', type=str, help="folder for experiment")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    sns.set_theme()

    # vehicle shape
    height = 2.0
    width = 5.0

    runs = {
            1: {'plot_until': [15, 17, 18], 'ids': [5,6], 'name': "crash"}, #crash
            3: {'plot_until': [15, 17, 20, 30], 'ids': [9, 3], 'name': "near_miss"}, #near miss
            4: {'plot_until': [23, 25, 29, 33], 'ids': [4,6], 'name': "nice_merge"} #nice merge
          }

    run_number = 4 # 1=crash, 3=near miss, 4=nice merge

    for plot_number in range(len(runs[run_number]['plot_until'])):
        print(plot_number)

        for f in glob.iglob(f"{args.runs}/pos_{run_number}.npy", recursive=True):
            data = np.load(f"{f}")
            df = pd.DataFrame(data[:runs[run_number]['plot_until'][plot_number], :], columns=["x", "y", "yaw"])
            ax = sns.lineplot(data=df, x="x", y="y")
            r = Rectangle(
                    xy=(data[runs[run_number]['plot_until'][plot_number]-1,  0]-width/2, data[runs[run_number]['plot_until'][plot_number]-1,  1]-height/2),
                    width=width, height=height, linewidth=1, 
                    color=ax.get_lines()[0].get_color(), fill=True, angle=np.rad2deg(df.tail(1)["yaw"]), rotation_point='center')
            ax.add_patch(r)

        for f in glob.iglob(f"{args.runs}/other_pos_{run_number}.npy", recursive=True):
            data = np.load(f"{f}")
            j=0
            for i in range(data.shape[1]):
                if i in runs[run_number]['ids']:
                    df = pd.DataFrame(data[:runs[run_number]['plot_until'][plot_number],i,:], columns=["x", "y", "yaw"])
                    ax = sns.lineplot(data=df, x="x", y="y")
                    r = Rectangle(xy=(data[runs[run_number]['plot_until'][plot_number]-1, i, 0]-width/2, data[runs[run_number]['plot_until'][plot_number]-1, i, 1]-height/2), width=width, height=height, linewidth=1, color=ax.get_lines()[j+1].get_color(), fill=True)
                    ax.add_patch(r)
                    j+=1

        ax.invert_yaxis()
        ax.grid(False)

        import matplotlib.image as mpimg
        map_img = mpimg.imread('road.png')

        ax.imshow(map_img,
                  aspect = ax.get_aspect(),
                  extent = [150,460,16,-2],
                  zorder = 0) #put the map under the plot

        # ax.set_xlim([225,300])
        ax.set_xlim([150,300])

        plt.gca().set_aspect('equal', adjustable='box')

        # matplotlib.pyplot.show()

        plt.tight_layout()
        plt.savefig(f"trajectory_{runs[run_number]['name']}_{plot_number}.png", dpi=600, bbox_inches="tight")

        plt.close()
