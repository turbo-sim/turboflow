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

if __name__ == "__main__":

    # Plot the critical Mach number against the kinetic energy loss coefficient
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.gca()
    ax.set_xlabel(r"$\Delta \phi^2$ - Kinetic energy loss coefficient")
    ax.set_ylabel(r"$\text{Ma}_\text{crit}$ - Critical Mach number")
    gamma_array = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    colormap = cm.Blues(np.linspace(0.5, 1.0, len(gamma_array)))
    for i, gamma in enumerate(gamma_array):
        eta = np.logspace(np.log10(1e-9), np.log10(1.00), 100)
        Ma_crit = tf.get_mach_crit(gamma, eta)
        phi_crit = tf.get_flow_capacity(Ma_crit, gamma, eta)
        ax.plot(eta, Ma_crit, linestyle='-', color=colormap[i], label=f"$\gamma={gamma}$")
    ax.legend(loc="lower right", ncol=1, fontsize=11)
    plt.tight_layout(pad=1)
    tf.savefig_in_formats(fig, os.path.join(out_dir, "mach_crit_vs_loss_coefficient"))

    # Plot the dimensionless mass function against the Mach number
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.gca()
    ax.set_xlabel(r"$\text{Ma}$ - Mach number")
    ax.set_ylabel(r"$\Phi = \left(\frac{\dot{m} \sqrt{R T_{0}}}{p_{0}  A}\right)$ - Dimensionless mass flow function")
    ax.set_xscale("linear")
    ax.set_yscale("linear")

    gamma = 1.4
    eta = np.logspace(np.log10(1e-4), np.log10(0.99), 100)
    Ma_crit = tf.get_mach_crit(gamma, eta)
    phi_crit = tf.get_flow_capacity(Ma_crit, gamma, eta)
    ax.plot(Ma_crit, phi_crit, linestyle='--', color="black")

    eta_array = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    colormap = cm.Blues(np.linspace(0.5, 1.0, len(eta_array)))
    for i, eta in enumerate(eta_array):
        Ma_max = np.sqrt(2/(gamma-1)*eta/(1-eta + 1e-6))
        Ma = np.linspace(0.00, np.minimum(Ma_max, 2), 101)
        phi = tf.get_flow_capacity(Ma, gamma, eta)
        Ma_crit = tf.get_mach_crit(gamma, eta)
        phi_crit = tf.get_flow_capacity(Ma_crit, gamma, eta)
        ax.plot(Ma, phi, color=colormap[i], label=f"$\Delta \phi^2 = {1-eta:0.1f}$")
        ax.plot(Ma_crit, phi_crit, marker="o", markerfacecolor="w", color=colormap[i])
    ax.legend(loc="lower right", ncol=2, fontsize=11)
    plt.tight_layout(pad=1)
    tf.savefig_in_formats(fig, os.path.join(out_dir, "mass_flow_vs_mach"))

    # Show figures
    plt.show()