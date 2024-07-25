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


def convert_stagnation_pressure_to_kinetic_energy_loss(Ma, gamma, Y):
    """
    Convert stagnation pressure loss coefficient ($Y$) to kinetic energy loss coefficient ($\Delta \phi^2$).

    This function calculates the kinetic energy loss coefficient ($\Delta \phi^2$) using the 
    given Mach number ($\text{Ma}$), specific heat capacity ratio ($\gamma$), and the stagnation
    pressure loss coefficient ($Y$).
    
    .. math::
        \Delta \phi^2=\frac{\left(\frac{p_2}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}} \left(\left[1+Y \left(1- \frac{p_2}{p_{02}}\right)\right]^{\frac{\gamma-1}{\gamma}}-1\right)}{\left[1+Y \left(1-\frac{p_2}{p_{02}}\right)\right]^{\frac{\gamma-1}{\gamma}}-\left(\frac{p_2}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}}}

    where the static-to-total pressure ratio is given by:

    .. math::
        \left(\frac{p_2}{p_{02}}\right)=\left(\frac{T_2}{T_{02}}\right)^{\frac{\gamma}{\gamma-1}}=\left[1+\left(\frac{\gamma-1}{2}\right) \text{Ma}^2\right]^{-\frac{\gamma}{\gamma-1}}

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    Y : float
        A parameter related to the specific energy.

    Returns
    -------
    float
        The computed value of the kinetic energy loss coefficient (Δφ²).
    """
    exp = (gamma - 1) / gamma
    p2_p02 = (1 + (gamma - 1) / 2 * Ma ** 2) ** (-1 / exp)
    p01_p02 = 1 + Y * (1 - p2_p02)
    delta_phi2 = p2_p02 ** exp * (p01_p02 ** exp - 1) / (p01_p02 ** exp - p2_p02 ** exp)
    return delta_phi2


def convert_kinetic_energy_to_stagnation_pressure_loss(Ma, gamma, delta_phi2):
    """
    Convert kinetic energy loss coefficient ($\Delta \phi^2$) to stagnation pressure loss coefficient ($Y$).

    This function calculates the stagnation pressure loss coefficient ($Y$) using the given 
    Mach number ($\text{Ma}$), specific heat capacity ratio ($\gamma$), and kinetic energy loss 
    coefficient ($\Delta \phi^2$).

    .. math::
        Y=\frac{\left[\left(\frac{1}{1-\Delta \phi^2}\right) \left(1-\Delta \phi^2 \left(\frac{p_2}{p_{02}}\right)^{-\frac{\gamma-1}{\gamma}}\right)\right]^{-\left(\frac{\gamma}{\gamma-1}\right)}-1}{1- \left(\frac{p_2}{p_{02}}\right)}

    where the static-to-total pressure ratio is given by:

    .. math::
        \left(\frac{p_2}{p_{02}}\right)=\left(\frac{T_2}{T_{02}}\right)^{\frac{\gamma}{\gamma-1}}=\left[1+\left(\frac{\gamma-1}{2}\right) \text{Ma}^2\right]^{-\frac{\gamma}{\gamma-1}}

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    delta_phi2 : float
        The kinetic energy loss coefficient ($\Delta \phi^2$).

    Returns
    -------
    float
        The computed value of the stagnation pressure loss ($Y$).
    """
    exp = (gamma - 1) / gamma
    p2_p02 = (1 + (gamma - 1) / 2 * Ma ** 2) ** (-1 / exp)
    Y = 1 / (1 - p2_p02) * (((1 - delta_phi2 * p2_p02 ** (-exp)) / (1 - delta_phi2)) ** (-1 / exp) - 1)
    return Y

def convert_enthalpy_to_stagnation_pressure_loss(Ma, gamma, zeta):
    """
    Convert enthalpy loss coefficient ($\zeta$) to stagnation pressure loss coefficient ($Y$).

    This function calculates the stagnation pressure loss coefficient ($Y$) using the 
    given Mach number ($\text{Ma}$), specific heat capacity ratio ($\gamma$), and enthalpy loss 
    coefficient ($\zeta$).

    The enthalpy loss coefficient ($\zeta$) is first converted to the kinetic energy loss 
    coefficient ($\Delta \phi^2$) using the formula:

    .. math::
        \Delta \phi^2 = \frac{\zeta}{1 + \zeta}

    The resulting $\Delta \phi^2$ is then used in the function 
    `convert_kinetic_energy_to_stagnation_pressure_loss()` to compute $Y$.

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    zeta : float
        Enthalpy loss coefficient ($\zeta$).

    Returns
    -------
    float
        The computed value of the stagnation pressure loss coefficient ($Y$).
    """
    delta_phi2 = zeta / (1 + zeta)
    Y = convert_kinetic_energy_to_stagnation_pressure_loss(Ma, gamma, delta_phi2)
    return Y

def convert_stagnation_pressure_to_enthalpy_loss(Ma, gamma, Y):
    """
    Convert stagnation pressure loss coefficient ($Y$) to enthalpy loss coefficient ($\zeta$).

    This function calculates the enthalpy loss coefficient ($\zeta$) using the given Mach number 
    ($\text{Ma}$), specific heat capacity ratio ($\gamma$), and stagnation pressure loss 
    coefficient ($Y$).

    The stagnation pressure loss coefficient ($Y$) is first converted to the kinetic 
    energy loss coefficient ($\Delta \phi^2$) using the function 
    `convert_stagnation_pressure_to_kinetic_energy_loss()`. The resulting $\Delta \phi^2$ is then 
    used to compute $\zeta$ using the formula:

    .. math::
        \zeta = \frac{\Delta \phi^2}{1 - \Delta \phi^2}

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    Y : float
        Stagnation pressure loss coefficient ($Y$).

    Returns
    -------
    float
        The computed value of the enthalpy loss coefficient ($\zeta$).
    """
    delta_phi2 = convert_stagnation_pressure_to_kinetic_energy_loss(Ma, gamma, Y)
    zeta = delta_phi2 / (1 - delta_phi2)
    return zeta

if __name__ == "__main__":

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
        delta_phi2 = convert_stagnation_pressure_to_kinetic_energy_loss(Ma, gamma, Y)
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
        Y = convert_kinetic_energy_to_stagnation_pressure_loss(Ma, gamma, dphi2)
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
        zeta = convert_stagnation_pressure_to_enthalpy_loss(Ma, gamma, Y)
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
        Y = convert_enthalpy_to_stagnation_pressure_loss(Ma, gamma, zeta)
        ax.plot(Ma, Y, linestyle='-', color=colormap[i], label=rf"$\zeta={zeta:0.2f}$")
    ax.legend(loc="upper left", ncol=1, fontsize=10)
    plt.tight_layout(pad=1)
    tf.savefig_in_formats(fig, os.path.join(out_dir, "enthalpy_to_stagnation_pressure_loss_coefficient"))

    # Show figures
    plt.show()