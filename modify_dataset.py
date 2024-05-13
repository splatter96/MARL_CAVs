import math
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scipy import interpolate

import warnings
warnings.filterwarnings('ignore')

sns.set_theme()

data = pd.read_csv("drone_dataset.csv")

num_vehicles = data['vehicle_id'].nunique()

v0 = data[data['vehicle_id'] == 500]
# v0 = data[data['vehicle_id'] == 78]

print(v0.head())

fx = interpolate.interp1d(v0["x[m]"], v0["y[m]"], fill_value="extrapolate")
fy = interpolate.interp1d(v0["y[m]"], v0["x[m]"], fill_value="extrapolate")
x_min = v0["x[m]"].min()

# v0["y"] = v0["y[m]"].transform(lambda y: y - y.min())
# v0["x"] = v0["x[m]"].transform(lambda x: x - x.min())

# x_max = v0["x"].max()
# y_max = v0["y"].max()
# d = -y_max / x_max

# x_min = v0["x[m]"].min()
# y_min = v0["y[m]"].min()

# for i in range(num_vehicles):
# for i in range(500, 501):
# for i in range(78, 79):
for i in range(0, 500):
    # print(i)
    v = data[data['vehicle_id'] == i]

    # ax = sns.lineplot(data=v, x="x[m]", y="y[m]")

    # v["y"] = v["y[m]"].transform(lambda y: y - y_min)
    # v["x"] = v["x[m]"].transform(lambda x: x - x_min)
    # v["y"] = v["y[m]"]
    # v["y_normalized"] = v.apply(lambda r: r["y"] - r["x"] * d, axis=1)
    # v["x_normalized"] = v.apply(lambda r: r["x"] + r["y"] * d, axis=1)

    # v["x"] = v["x[m]"]
    # v["y_normalized"] = v.apply(lambda r: r["y[m]"] - np.interp(r["x[m]"], v0["x[m]"], v0["y[m]"]), axis=1)
    v["y_normalized"] = v.apply(lambda r: r["y[m]"] - fx(r["x[m]"]), axis=1)
    v["x_normalized"] = v.apply(lambda r: r["x[m]"] - x_min, axis=1)
    # v["x_normalized"] = v.apply(lambda r: r["x[m]"] - np.interp(r["y[m]"], v0["y[m]"], v0["x[m]"]), axis=1)

    # v["x"] = v["x_normalized"].transform(lambda x: x - x.min())

    # Check if it is the lane we are looking for
    # if v["y_normalized"].between(-8, 5).all() and  v["Time [s]"].between(0, 17).all():
    if  v["Time [s]"].between(0, 17).all():
    # ax = sns.lineplot(data=v, x="x[m]", y="y_normalized")
        ax = sns.lineplot(data=v, x="x_normalized", y="y_normalized")

        # traj = v[["Time [s]", "x_normalized", "y_normalized", "Speed [km/h]"]].to_numpy()
        # np.save(f"traj{i}.npy", traj)

plt.show()

