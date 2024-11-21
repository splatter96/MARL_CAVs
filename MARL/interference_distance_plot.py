import numpy as np
from matplotlib import pyplot as plt
from scipy import special
import csv
import scipy.stats as stats

rho_c = 30  # [dBsm] Radar Cross Section
T = 10  # [dB] SRI threshold
chi = 1 / 100  # spectrum collision probability
rho = 1 / 10  # [vehicle/m] vehicle density
f = 76.5e9  # [Hz] center frequency
c = 3e8  # [m/s] speed of light
P0 = 10  # [dBm] transmit power to antenna
a = 2  # path loss exponent
Gt = 45  # [dBi] max Antenna gain

maximum_range = 500

# Rician Channel model parameters
# see https://comyx.readthedocs.io/latest/_modules/comyx/fading/rician.html#Rician
K = 10  # Rician factor, i.e., ratio between the power of direct path and the power of scattered paths
sigma = np.sqrt(
    1 / 20.999
)  # The scale parameter, which is the standard deviation of the distribution.

omega = (2 * K + 2) * sigma**2
nu = np.sqrt((K / (1 + K)) * omega)
gamma1 = Gt**2 * (c / (2 * np.pi * f) ** 2)
gamma2 = rho_c / (4 * np.pi)


def get_channel(size):
    return stats.rice.rvs(nu / sigma, scale=sigma, size=size)


def interference(dist):
    return gamma1 * P0 * dist**-a


def signal(dist):
    print(get_channel(dist.shape).mean())  # should be 1.0
    return gamma1 * gamma2 * P0 * get_channel(dist.shape) * dist ** (-2 * a)


def detection(S, I):
    return (S / I) > T


int_distances = []
target_distances = []
with open("interference_distance.csv", "r") as f:
    reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        int_distances.append(row[0])
        target_distances.append(row[1:])

int_distances = np.array(int_distances)
target_distances = np.array(target_distances)

signals = signal(target_distances)
interferences = interference(int_distances)

detections = []
possible_detections = []  # Tuples of (interferer_distance, target_distance)
for i in range(interferences.shape[0]):
    for j in range(signals[i].shape[0]):
        if target_distances[i][j] == maximum_range:  # skip the "empty" observations
            continue
        det = detection(signals[i][j], interferences[i])
        detections.append(det)
        if det:
            possible_detections.append((int_distances[i], target_distances[i][j]))

detections = np.array(detections)
# print(detections)
print(f"Detection possible: {detections.sum()}")
print(f"Total testpoints: {detections.shape[0]}")
print(f"Detection percentage: {detections.sum()/detections.shape[0]}")

# print(possible_detections[150])
