import os
import numpy as np
import matplotlib.pyplot as plt
import turboflow as tf
from matplotlib import cm

from loss_coefficient_perfect_gas import convert_kinetic_energy_to_stagnation_pressure_loss

tf.set_plot_options()

# Create the folder to save figures
out_dir = "figures"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def compute_stagnation_pressure_drop_fraction(Ma, gamma, Y):
    exp = (gamma - 1) / gamma
    p2_p02 = (1 + (gamma - 1) / 2 * Ma ** 2) ** (-1 / exp)
    delta_p0_p01 = 1 - 1 / (1 + Y*(1 - p2_p02))
    return delta_p0_p01


# Plot plot the kinetic energy loss coefficient for fixed stagnation pressure loss coefficient
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_xlabel(r"$\text{Ma}$ - Mach number")
ax.set_ylabel(r"$\Delta p_0 / p_{01}$ - Fraction of stagnation pressure drop (\%)")
gamma = 1.4
Ma = np.linspace(0.01, 2, 101)
Y_array = np.arange(0, 0.35, 0.05)
colormap = cm.Blues(np.linspace(0.5, 1.0, len(Y_array)))
for i, Y in enumerate(Y_array):
    delta_p0_p01 = compute_stagnation_pressure_drop_fraction(Ma, gamma, Y)
    ax.plot(Ma, delta_p0_p01*100, linestyle='-', color=colormap[i], label=f"$Y={Y:0.2f}$")
ax.legend(loc="upper left", ncol=1, fontsize=10)
plt.tight_layout(pad=1)

# Plot plot the kinetic energy loss coefficient for fixed stagnation pressure loss coefficient
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_xlabel(r"$\text{Ma}$ - Mach number")
ax.set_ylabel(r"$\Delta p_0 / p_{01}$ - Fraction of stagnation pressure drop (\%)")
gamma = 1.4
Ma = np.linspace(0.01, 2, 101)
dphi2_array = np.arange(0, 0.35, 0.05)
colormap = cm.Reds(np.linspace(0.5, 1.0, len(dphi2_array)))
for i, dphi2 in enumerate(dphi2_array):
    Y = convert_kinetic_energy_to_stagnation_pressure_loss(Ma, gamma, dphi2)
    delta_p0_p01 = compute_stagnation_pressure_drop_fraction(Ma, gamma, Y)
    ax.plot(Ma, delta_p0_p01*100, linestyle='-', color=colormap[i], label=f"$\Delta \phi^2={dphi2:0.2f}$")
ax.legend(loc="upper left", ncol=1, fontsize=10)
plt.tight_layout(pad=1)

# Show figures
plt.show()