import os
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = np.load("action_before_crash_926.npy", allow_pickle=True)
# data = np.load("action_without_crash_112.npy", allow_pickle=True)

print(data)

sns.set_theme()
sns.set_context("paper")
#sns.set(font_scale=2)

actions_map = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

df = pd.DataFrame(data)

df = df.max(axis=1)
print(df)
# exit(0)

#df[["x", "y"]] = pd.DataFrame(df.pos.tolist())
#df = df.drop('pos', axis=1)
#df['action'] = df['action'].astype(str).astype(int)

#df['action'] = df['action'].map(actions_map)

#print(df.query("action=='LANE_LEFT'"))

#ax = sns.scatterplot(df, x="x", y="y", hue="action", palette="tab10", s=200)
# ax = sns.scatterplot(df.query("action == 'LANE_LEFT'"), x="x", y="y", hue="action", palette="Set2", s=200)

#df.plot(kind="bar", stacked=True)
ax = df.plot(kind="line", stacked=True)

# dis = sns.displot(df.query("action == 'LANE_LEFT'"), x="x", y="y", kind="hist", binwidth=(1, 1),  cbar=True)
# ax = dis.axes[0,0]
# ax.set_aspect(10)

# ax.grid(False)

# import matplotlib.image as mpimg
# map_img = mpimg.imread('road.png')

# ax.imshow(map_img,
          # aspect = ax.get_aspect(),
          # extent = [150,460,16,-2],
          # zorder = 0) #put the map under the heatmap

# # road segments
#ax.axvline(x=150)
#ax.axvline(x=230)
#ax.axvline(x=310)
#ax.axvline(x=460)

#ax.set_ylim(15, -2)
# ax.set_xlim(150, 450)

# straight roads
#ax.axhline(y=0, linestyle='dashed', color='red')
#ax.axhline(y=4, linestyle='dashed', color='red')
#ax.axhline(y=14, linestyle='dashed', color='red')

ax.figure.savefig("out.png")
# plt.tight_layout()
# plt.show()
# plt.savefig("action_certainty_with_crash.png", dpi=600, bbox_inches="tight")
