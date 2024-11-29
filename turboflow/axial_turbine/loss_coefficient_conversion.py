
import jax.numpy as jnp

from .. import math

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


def convert_kinetic_energy_to_stagnation_pressure_loss(Ma, gamma, delta_phi2, limit_output=True):
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
    Ma_lim = jnp.sqrt(2/(gamma - 1) * (1-delta_phi2) / (delta_phi2 + 1e-6))
    Ma = math.smooth_minimum(Ma, 0.9*Ma_lim)
    exp = (gamma - 1) / gamma
    p2_p02 = (1 + (gamma - 1) / 2 * Ma ** 2) ** (-1 / exp)
    Y = 1 / (1 - p2_p02) * (((1 - delta_phi2 * p2_p02 ** (-exp)) / (1 - delta_phi2)) ** (-1 / exp) - 1)
    if limit_output:
        Y = math.smooth_minimum(Y, 1.0- + 0.*Y, method="logsumexp")
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