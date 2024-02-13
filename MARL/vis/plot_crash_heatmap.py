import os
import matplotlib
import argparse
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = np.load("crash_pos.npy")

sns.set_theme()
sns.set(font_scale=2)

# x_low = 210
x_low = 0
x_high = 500
y_low = 0
y_high = 15

x_res = 1
y_res = 1

heat_map = np.empty(shape=(int((x_high-x_low)/x_res), int((y_high-y_low)/y_res)))

for x in range(x_low, x_high, x_res):
    for y in range(y_low, y_high, x_res):
        for pos in data:
            if x <= pos[0] <= x+x_res and y <= pos[1] <= y+y_res:
                heat_map[x-x_low][y-y_low] += 1


plt.subplots(figsize=(25,15))
ax = sns.heatmap(heat_map.transpose(), alpha=0.5, square=False, cbar_kws={"orientation": "horizontal"}, cmap='viridis')
ax.set_aspect(4)

# TODO get proper image for the road only
import matplotlib.image as mpimg
map_img = mpimg.imread('road.png')

ax.imshow(map_img,
          aspect = ax.get_aspect(),
          # extent = ax.get_xlim() + ax.get_ylim(),
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
