# demo_ideal_gas_hs.py

import jax.numpy as jnp
from jax import grad
import turboflow as tf

# # Constants
R = 287.0  # Specific gas constant for air (J/(kg*K))
gamma = 1.41  # Specific heat ratio for air
T_ref = 288.15  # Reference temperature (K)
P_ref = 101306.33  # Reference pressure (Pa)
s_ref = 1659.28  # Reference entropy (J/(kg*K))
myu_ref = 1.789e-5  # Reference dynamic viscosity (Kg/(m*s))
S_myu = 110.56  # Sutherland's constant for viscosity (K)
k_ref = 0.0241  # Reference thermal conductivity (W/(m*K))
S_k = 194  # Sutherland's constant for thermal conductivity (K)

# def specific_heat(R, gamma):
#     """Calculate specific heats at constant pressure and volume."""
#     Cp = (gamma * R) / (gamma - 1)  # J/(kg*K)
#     Cv = Cp / gamma  # J/(kg*K)
#     return Cp, Cv

# def temperature(h, s):
#     """Calculate temperature from enthalpy."""
#     Cp, Cv = specific_heat(R, gamma)
#     return h / Cp  # Temperature in Kelvin

# def pressure(h, s):
#     """Calculate pressure using enthalpy and entropy."""
#     T = temperature(h, s)
#     Cp, Cv = specific_heat(R, gamma)
#     return P_ref * jnp.exp(((Cp * jnp.log(T / T_ref)) - (s - s_ref)) / R)

# def density(h, s):
#     """Calculate density using enthalpy and entropy."""
#     T = temperature(h, s)
#     P = pressure(h, s)
#     return P / (R * T)  # Density in kg/m^3

# def viscosity(h, s):
#     """Calculate dynamic viscosity using Sutherland's formula."""
#     T = temperature(h, s)
#     return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

# def thermal_conductivity(h, s):
#     """Calculate thermal conductivity using Sutherland's formula."""
#     T = temperature(h, s)
#     return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

# def speed_of_sound(h, s):
#     T = temperature(h, s)
#     return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

def calculate_properties_hs(h, s):
    """Calculate all thermodynamic properties given enthalpy and Pressure."""

    def specific_heat(R, gamma):
        """Calculate specific heats at constant pressure and volume."""
        Cp = (gamma * R) / (gamma - 1)  # J/(kg*K)
        Cv = Cp / gamma  # J/(kg*K)
        return Cp, Cv

    def temperature(h, s):
        """Calculate temperature from enthalpy."""
        Cp, Cv = specific_heat(R, gamma)
        return h / Cp  # Temperature in Kelvin

    def pressure(h, s):
        """Calculate pressure using enthalpy and entropy."""
        T = temperature(h, s)
        Cp, Cv = specific_heat(R, gamma)
        return P_ref * jnp.exp(((Cp * jnp.log(T / T_ref)) - (s - s_ref)) / R)

    def density(h, s):
        """Calculate density using enthalpy and entropy."""
        T = temperature(h, s)
        P = pressure(h, s)
        return P / (R * T)  # Density in kg/m^3

    def viscosity(h, s):
        """Calculate dynamic viscosity using Sutherland's formula."""
        T = temperature(h, s)
        return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

    def thermal_conductivity(h, s):
        """Calculate thermal conductivity using Sutherland's formula."""
        T = temperature(h, s)
        return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

    def speed_of_sound(h, s):
        T = temperature(h, s)
        return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

    return {"T": temperature(h, s),
            "P": pressure(h, s),
            "d": density(h, s), 
            "h": h,
            "s": s,
            "mu": viscosity(h, s), 
            "k": thermal_conductivity(h, s), 
            "a": speed_of_sound(h, s)}

# Example input: enthalpy (h) and entropy (s)
# h, s = 400980.0, 1991.94

# properties = calculate_properties_hs(h, s)

# print(properties)

# def calculate_properties(h, s):
#     """Calculate all thermodynamic properties given enthalpy and entropy."""
#     T = temperature(h, s)
#     P = pressure(h, s)
#     rho = density(h, s)
#     myu = viscosity(h, s)
#     k = thermal_conductivity(h, s)
    
    # return T, P, rho, myu, k



############################################################ For partial derivatives verification

# def compute_partial_derivatives(h, s):
#     """Compute partial derivatives of properties."""
#     dP_dh = grad(pressure, argnums=0)  # Partial derivative with respect to h at constant s
#     dP_ds = grad(pressure, argnums=1)  # Partial derivative with respect to s at constant h

#     dT_dh = grad(temperature, argnums=0)  # Partial derivative with respect to h at constant s
#     dT_ds = grad(temperature, argnums=1)  # Partial derivative with respect to s at constant h
    
#     drho_dh = grad(density, argnums=0)  # Partial derivative with respect to h at constant s
#     drho_ds = grad(density, argnums=1)  # Partial derivative with respect to s at constant h
    
#     return {
#         "dP_dh_s": dP_dh(h, s),
#         "dP_ds_h": dP_ds(h, s),
#         "dT_dh_s": dT_dh(h, s),
#         "dT_ds_h": dT_ds(h, s),
#         "drho_dh_s": drho_dh(h, s),
#         "drho_ds_h": drho_ds(h, s)
#     }



# # Calculate thermodynamic properties
# # Calculate thermodynamic properties
# # properties = calculate_properties(h, s)

# # T = properties["T"]
# # P = properties["P"]
# # rho = properties["d"]
# # h = properties["h"]
# # s = properties["s"]
# # mu = properties["mu"]
# # k = properties["k"]
# # a = properties["a"]



# # Print the results
# print(f"Temperature: {T}\nPressure: {P}\nDensity: {rho}\nEnthalpy: {h}\nEntropy: {s}\nViscosity: {mu}\nThermal Conductivity: {k}")

# # Compute partial derivatives
# partial_derivatives = compute_partial_derivatives(h, s)

# # Print partial derivatives
# for key, value in partial_derivatives.items():
#     print(f"{key}: {value}")


# ## Verifying the results from automatic differentiation

# dP_dT_s = partial_derivatives["dP_dh_s"] / partial_derivatives["dT_dh_s"]  # Example usage
# dP_drho_h = partial_derivatives["dP_ds_h"] / partial_derivatives["drho_ds_h"]  # Example usage

# # Define the ideal gas equation of state: p = rho * R * T
# def ideal_gas_equation(rho, T):
#     return rho * R * T

# # With Inputs T and rho
# dP_drho = grad(ideal_gas_equation, argnums=0)  # Partial derivative with respect to rho
# dP_dT = grad(ideal_gas_equation, argnums=1)    # Partial derivative with respect to T

# dP_dT_rho = dP_dT(rho, T)
# dP_drho_T = dP_drho(rho, T)

# # if round(dP_dT_s - ((gamma/(gamma - 1))*dP_dT_rho)) == 0.0 and round(dP_drho_h - dP_drho_T) == 0.0 :
# #     print("The partial derivatives calculated from AD method from different forms of equation of state are matching")
# # else:
# #     print("The partial derivatives calculated from AD method from different forms of equation of state are not matching")

# if jnp.isclose(dP_dT_s, (gamma / (gamma - 1)) * dP_dT_rho) and jnp.isclose(dP_drho_h, dP_drho_T):
#     print("The partial derivatives calculated from AD method from different forms of equation of state are matching")
# else:
#     print("The partial derivatives calculated from AD method from different forms of equation of state are not matching")

# # Compare with Turboflow's finite difference approximation

# def wrapped_ideal_gas(rho_T):
#     """Wrapper for ideal gas equation for finite difference approximation."""
#     rho, T = rho_T
#     return ideal_gas_equation(rho, T)

# gradient_approx = tf.approx_gradient(wrapped_ideal_gas, x=jnp.asarray([rho, T]), method="3-point", abs_step=1e-3)

# print("Partial derivatives comparison:")

# print("Partial derivative of pressure wrt rho (AD):", dP_drho_T)
# print("Partial derivative of pressure wrt rho (FD):", gradient_approx[0])
# print("Partial derivative of pressure wrt T (AD):", dP_dT_rho)
# print("Partial derivative of pressure wrt T (FD):", gradient_approx[1])
