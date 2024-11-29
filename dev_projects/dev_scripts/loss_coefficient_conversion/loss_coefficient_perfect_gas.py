import os
import numpy as np
import matplotlib.pyplot as plt
import turboflow as tf
from matplotlib import cm

tf.set_plot_options()

# Create the folder to save figures
out_dir = "figures"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Plot plot the kinetic energy loss coefficient for fixed stagnation pressure loss coefficient
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_xlabel(r"$\text{Ma}$ - Mach number")
ax.set_ylabel(r"$\Delta \phi^2$ - Kinetic energy loss coefficient")
gamma = 1.4
Ma = np.linspace(0.01, 1.50, 101)
Y_array = np.arange(0, 0.35, 0.05)
colormap = cm.Reds(np.linspace(0.5, 1.0, len(Y_array)))
for i, Y in enumerate(Y_array):
    delta_phi2 = tf.convert_stagnation_pressure_to_kinetic_energy_loss(Ma, gamma, Y)
    ax.plot(Ma, delta_phi2, linestyle='-', color=colormap[i], label=f"$Y={Y:0.2f}$")
ax.legend(loc="upper right", ncol=1, fontsize=10)
plt.tight_layout(pad=1)
tf.savefig_in_formats(fig, os.path.join(out_dir, "stagnation_pressure_to_kinetic_energy_loss_coefficient"))

# Plot plot the stagnation pressure loss coefficient for fixed kinetic energy loss coefficient
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_xlabel(r"$\text{Ma}$ - Mach number")
ax.set_ylabel(r"$Y$ - Stagnation pressure loss coefficient")
gamma = 1.4
Ma = np.linspace(0.01, 1.50, 101)
dphi2_array = np.arange(0, 0.35, 0.05)
colormap = cm.Reds(np.linspace(0.5, 1.0, len(dphi2_array)))
for i, dphi2 in enumerate(dphi2_array):
    Y = tf.convert_kinetic_energy_to_stagnation_pressure_loss(Ma, gamma, dphi2)
    ax.plot(Ma, Y, linestyle='-', color=colormap[i], label=rf"$\Delta \phi^2={dphi2:0.2f}$")
ax.legend(loc="upper left", ncol=1, fontsize=10)
plt.tight_layout(pad=1)
tf.savefig_in_formats(fig, os.path.join(out_dir, "kinetic_energy_to_stagnation_pressure_loss_coefficient"))

# Plot plot the enthalpy energy loss coefficient for fixed stagnation pressure loss coefficient
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_xlabel(r"$\text{Ma}$ - Mach number")
ax.set_ylabel(r"$\zeta$ - Enthalpy loss coefficient")
gamma = 1.4
Ma = np.linspace(0.01, 1.50, 101)
Y_array = np.arange(0, 0.35, 0.05)
colormap = cm.Blues(np.linspace(0.5, 1.0, len(Y_array)))
for i, Y in enumerate(Y_array):
    zeta = tf.convert_stagnation_pressure_to_enthalpy_loss(Ma, gamma, Y)
    ax.plot(Ma, zeta, linestyle='-', color=colormap[i], label=f"$Y={Y:0.2f}$")
ax.legend(loc="upper right", ncol=1, fontsize=10)
plt.tight_layout(pad=1)
tf.savefig_in_formats(fig, os.path.join(out_dir, "stagnation_pressure_to_enthalpy_loss_coefficient"))

# Plot plot the stagnation pressure loss coefficient for fixed enthalpy loss coefficients
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.gca()
ax.set_xlabel(r"$\text{Ma}$ - Mach number")
ax.set_ylabel(r"$Y$ - Stagnation pressure loss coefficient")
gamma = 1.4
Ma = np.linspace(0.01, 1.50, 101)
zeta_array = np.arange(0, 0.35, 0.05)
colormap = cm.Blues(np.linspace(0.5, 1.0, len(zeta_array)))
for i, zeta in enumerate(zeta_array):
    Y = tf.convert_enthalpy_to_stagnation_pressure_loss(Ma, gamma, zeta)
    ax.plot(Ma, Y, linestyle='-', color=colormap[i], label=rf"$\zeta={zeta:0.2f}$")
ax.legend(loc="upper left", ncol=1, fontsize=10)
plt.tight_layout(pad=1)
tf.savefig_in_formats(fig, os.path.join(out_dir, "enthalpy_to_stagnation_pressure_loss_coefficient"))

# Show figures
plt.show()