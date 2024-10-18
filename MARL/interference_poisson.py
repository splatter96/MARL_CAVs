import numpy as np
from matplotlib import pyplot as plt
from scipy import special

rho_c = 30  # [dBsm] Radar Cross Section
T = 10  # [dB] SRI threshold
chi = 1 / 100  # spectrum collision probability
rho = 1 / 10  # [vehicle/m] vehicle density
f = 76.5e9  # [Hz] center frequency
c = 3e8  # [m/s] speed of light
P0 = 10  # [dBm] transmit power to antenna
a = 2  # path loss exponent
Gt = 45  # [dBi] max Antenna gain


gamma1 = Gt**2 * (c / (2 * np.pi * f) ** 2)
gamma2 = rho_c / (4 * np.pi)


def interference(dist):
    return gamma1 * P0 * dist**-a


def signal(dist):
    return gamma1 * gamma2 * P0 * dist ** (-2 * a)


def detection(S, I):
    return (S / I) > T


# static interferer at 200m distance
static_int = interference(40)

# Plotting
r = np.linspace(0, 30, 300)
i = np.linspace(0, 450, 900)
R, I = np.meshgrid(r, i)
# plt.plot(r, signal(r), label="Signal")
# plt.plot(r, interference(r), label="Interference")

# plt.plot(r, detection(signal(r), static_int), label="Detection")
# plt.legend(loc="upper left")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(R, I, detection(signal(R), interference(I)), cmap="viridis")
ax.set_xlabel("Target distance")
ax.set_ylabel("Interferer distance")

plt.show()
