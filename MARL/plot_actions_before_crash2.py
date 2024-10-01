import glob
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

# import plotly.express as px

crashes = np.zeros(2)
for f in glob.iglob("crash_location*.npy", recursive=True):
    d = np.load(f)
    crashes = np.vstack([crashes, d])

run_list = []
# for f in glob.iglob("action_before_crash_pos*.npy", recursive=True):
for f in glob.iglob("action_before_crash*.npy", recursive=True):
# for f in glob.iglob("action_without_crash_pos*.npy", recursive=True):
    df = pd.DataFrame(data=np.load(f), columns=["x", "y", "certainty"])
    run_list.append(df)

runs = pd.concat(run_list)
runs = runs.reset_index() # remove duplicates

fig, ax = plt.subplots(figsize=(6, 6))

d = stats.binned_statistic_2d(runs["x"], runs["y"], runs["certainty"], statistic="mean", bins=100)
extent = [d[1][0], d[1][-1], d[2][0], d[2][-1]]
# imshow_mean = ax.imshow(d[0].T, cmap='magma', vmin=0.5, vmax=1.0, extent=extent, interpolation='nearest', zorder=1, origin='lower')

# cbar = fig.colorbar(imshow_mean, orientation='horizontal', fraction=0.075, pad=0.04)
# cbar = fig.colorbar(imshow_mean, fraction=0.027, pad=0.04)

#plot the crash locations
# ax.scatter(crashes[:,0], crashes[:,1], alpha=0.6, s=1, marker="x")
ax.scatter(crashes[:,0], crashes[:,1], s=8, marker="x", color='red')


# fig2 = px.density_heatmap(runs, x="x", y="y", z="certainty", nbinsx=100, nbinsy=100, range_color=[0.5, 1.0],  histfunc="avg")
# fig2.update_yaxes(autorange="reversed")
# fig2.show()


ax.set_aspect(10)
ax.grid(False)

ax.set_xlim([150,350])

import matplotlib.image as mpimg
map_img = mpimg.imread('road.png')

ax.imshow(map_img,
          aspect = ax.get_aspect(),
          extent = [150,460,16,-2],
          zorder = 0) #put the map under the heatmap

# plt.title("Action certainty of trajectories with collisions over position in the road network")
plt.title("Positions of collisions")

plt.tight_layout()
# plt.show()
# plt.savefig("action_certainty_position_no_crash.png", dpi=600, bbox_inches="tight")
# plt.savefig("action_certainty_position_with_crash.png", dpi=600, bbox_inches="tight")
plt.savefig("action_certainty_crash_positions.png", dpi=600, bbox_inches="tight")
