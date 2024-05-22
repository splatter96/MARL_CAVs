import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = np.load("crash_pos.npy")
df = pd.DataFrame(data, columns=["x", "y"])

sns.set_theme()
sns.set_context("paper")
# sns.set(font_scale=0.5)

ax = sns.kdeplot(df, x="x", y="y", fill=True, bw_adjust=0.4, thresh=0.1, cbar=True, cbar_kws={"location": "bottom", "label": "probability density for crash"})
ax.set_aspect(10)
ax.grid(False)

cbar = plt.gcf().get_axes()[1]._colorbar
cbar.ax.tick_params(rotation=-45)

import matplotlib.image as mpimg
map_img = mpimg.imread('road.png')

ax.imshow(map_img,
          aspect = ax.get_aspect(),
          extent = [150,460,16,-2],
          zorder = 0) #put the map under the heatmap

plt.savefig("crash_map.png", dpi=600, bbox_inches="tight")

