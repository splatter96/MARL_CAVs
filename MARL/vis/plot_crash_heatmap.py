import os
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = np.load("crash_pos.npy")
df = pd.DataFrame(data, columns=["x", "y"])

sns.set_theme()
sns.set(font_scale=2)

dis = sns.displot(df, x="x", y="y", kind="kde", fill=True, bw_adjust=0.5)
ax = dis.axes[0,0]
# ax.set_aspect(4)

import matplotlib.image as mpimg
map_img = mpimg.imread('road.png')

ax.imshow(map_img,
          aspect = ax.get_aspect(),
          extent = [150,460,16,-2],
          zorder = 0) #put the map under the heatmap

# road segments
ax.axvline(x=150)
ax.axvline(x=230)
ax.axvline(x=310)
ax.axvline(x=460)

ax.set_ylim(15, -2)
ax.set_xlim(150, 450)

# straight roads
ax.axhline(y=0, linestyle='dashed', color='red')
ax.axhline(y=4, linestyle='dashed', color='red')
ax.axhline(y=14, linestyle='dashed', color='red')

plt.tight_layout()
plt.show()
