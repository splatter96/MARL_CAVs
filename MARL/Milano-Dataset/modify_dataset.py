import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scipy import interpolate

import warnings
warnings.filterwarnings('ignore')

sns.set_theme()

# data = pd.read_csv("drone_dataset.csv")
data = pd.read_csv("./AllD_T1.csv")

num_vehicles = data['Vehicle_ID'].nunique()

ids_to_interpoalte = np.load("ids.npy")
v_interp = data[data['Vehicle_ID'].isin(ids_to_interpoalte)].sort_values("x [m]")

z = np.polyfit(v_interp["x [m]"], v_interp["y [m]"], 1)
# fx = np.poly1d(z)

a = z[0] # anstieg
alpha = -np.arctan(a)
x_min = v_interp["x [m]"].min()
x_center = (v_interp["x [m]"].min() + v_interp["x [m]"].max())/2
y_center = (v_interp["y [m]"].min() + v_interp["y [m]"].max())/2

for i in range(10, 1000):
    v = data[data['Vehicle_ID'] == i]

    # v["y_normalized"] = v.apply(lambda r: r["y [m]"] - fx(r["x [m]"]), axis=1)

    v["y"] = v.apply(lambda r: r["y [m]"] - y_center, axis=1)
    v["x"] = v.apply(lambda r: r["x [m]"] - x_center, axis=1)

    v["x_normalized"] = v.apply(lambda r: (r["x"] * np.cos(alpha)) - (r["y"] * np.sin(alpha)) + (x_center-x_min), axis=1)
    v["y_normalized"] = v.apply(lambda r: 2-((r["x"] * np.sin(alpha)) + (r["y"] * np.cos(alpha))), axis=1)
    v["angle_normalized"] = v.apply(lambda r: r["Angle [rad]"] + alpha, axis=1)


    # Check if it is the lane we are looking for
    if v["y_normalized"].between(-2, 5).all() and  v["Time [s]"].between(0, 50).all():
        ax = sns.lineplot(data=v, x="x_normalized", y="y_normalized")
        traj = v[["Time [s]", "x_normalized", "y_normalized", "Speed [km/h]", "angle_normalized"]].to_numpy()
        np.save(f"traj{i}.npy", traj)

plt.show()

