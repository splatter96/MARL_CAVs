import numpy as np
from matplotlib import pyplot as plt
from scipy import special

rho_c = 30  # [dBsm] Radar Cross Section
T = 10  # [dB] SRI threshold
chi = 1 / 100  # spectrum collision probability
rho = 1 / 10  # [vehicle/m] vehicle density


def detection_probability(r: int):
    gamma2 = rho_c / (4 * np.pi)
    return special.erfc(np.sqrt((np.pi * T) / (4 * gamma2)) * chi * rho * r**2)


# Plotting
x = np.linspace(0, 150)
plt.plot(x, detection_probability(x))
plt.show()
