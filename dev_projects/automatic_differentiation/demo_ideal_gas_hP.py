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

# def temperature(h, P):
#     """Calculate temperature from enthalpy."""
#     Cp, Cv = specific_heat(R, gamma)
#     return h / Cp  # Temperature in Kelvin

# def density(h, P):
#     """Calculate density using enthalpy and pressure."""
#     T = temperature(h, P)
#     return P / (R * T)  # Density in kg/m^3

# def entropy(h, P):
#     Cp, Cv = specific_heat(R, gamma)
#     T = temperature(h, P)
#     return s_ref + (Cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

# def viscosity(h, P):
#     """Calculate dynamic viscosity using Sutherland's formula."""
#     T = temperature(h, P)
#     return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

# def thermal_conductivity(h, P):
#     """Calculate thermal conductivity using Sutherland's formula."""
#     T = temperature(h, P)
#     return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

# def speed_of_sound(h, P):
#     T = temperature(h, P)
#     return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

def calculate_properties_hP(h, P):
    """Calculate all thermodynamic properties given enthalpy and Pressure."""

    def specific_heat(R, gamma):
        """Calculate specific heats at constant pressure and volume."""
        Cp = (gamma * R) / (gamma - 1)  # J/(kg*K)
        Cv = Cp / gamma  # J/(kg*K)
        return Cp, Cv

    def temperature(h, P):
        """Calculate temperature from enthalpy."""
        Cp, Cv = specific_heat(R, gamma)
        return h / Cp  # Temperature in Kelvin

    def density(h, P):
        """Calculate density using enthalpy and pressure."""
        T = temperature(h, P)
        return P / (R * T)  # Density in kg/m^3

    def entropy(h, P):
        Cp, Cv = specific_heat(R, gamma)
        T = temperature(h, P)
        return s_ref + (Cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

    def viscosity(h, P):
        """Calculate dynamic viscosity using Sutherland's formula."""
        T = temperature(h, P)
        return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

    def thermal_conductivity(h, P):
        """Calculate thermal conductivity using Sutherland's formula."""
        T = temperature(h, P)
        return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

    def speed_of_sound(h, P):
        T = temperature(h, P)
        return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

    return {"T": temperature(h, P),
            "P": P,
            "d": density(h, P), 
            "h": h,
            "s": entropy(h, P),
            "mu": viscosity(h, P), 
            "k": thermal_conductivity(h, P), 
            "a": speed_of_sound(h, P)}


# Example input: enthalpy (h) and Pressure (P)
# h, P = 400980.0, 103587.1484375

# # Calculate thermodynamic properties
# properties = calculate_properties_hP(h, P)

# tf.print_dict(properties)

# T = properties["T"]
# P = properties["P"]
# rho = properties["d"]
# h = properties["h"]
# s = properties["s"]
# mu = properties["mu"]
# k = properties["k"]
# a = properties["a"]



# # Print the results
# print(f"Temperature: {T}\nPressure: {P}\nDensity: {rho}\nEnthalpy: {h}\nEntropy: {s}\nViscosity: {mu}\nThermal Conductivity: {k}")

############################################################ For partial derivatives verification

# # Compute partial derivatives

# def compute_partial_derivatives(h, P):
#     """Compute partial derivatives of properties."""
#     dT_dh = grad(temperature, argnums=0)  # Partial derivative with respect to h at constant P
#     dT_dP = grad(temperature, argnums=1)  # Partial derivative with respect to P at constant h
    
#     drho_dh = grad(density, argnums=0)  # Partial derivative with respect to h at constant P
#     drho_dP = grad(density, argnums=1)  # Partial derivative with respect to P at constant h
    
#     return {
#         "dT_dh_P": dT_dh(h, P),
#         "dT_dP_h": dT_dP(h, P),
#         "drho_dh_P": drho_dh(h, P),
#         "drho_dP_h": drho_dP(h, P)
#     }

# partial_derivatives = compute_partial_derivatives(h, P)

# # Print partial derivatives
# for key, value in partial_derivatives.items():
#     print(f"{key}: {value}")


# ## Verifying the results from automatic differentiation

# dT_drho_P = partial_derivatives["dT_dh_P"] / partial_derivatives["drho_dh_P"]  # Example usage
# drho_dP_T = partial_derivatives["drho_dP_h"]

# # Define the ideal gas equation of state: p = rho * R * T
# def ideal_gas_equation(rho, T):
#     return rho * R * T

# # With Inputs T and rho
# dP_drho = grad(ideal_gas_equation, argnums=0)  # Partial derivative with respect to rho
# dP_dT = grad(ideal_gas_equation, argnums=1)    # Partial derivative with respect to T

# dP_dT_rho = dP_dT(rho, T)
# dP_drho_T = dP_drho(rho, T)

# # if  round(dP_dT_rho*dT_drho_P*drho_dP_T) == -1.0:
# #     print("The partial derivatives calculated from AD method from different forms of equation of state are matching")
# # else:
# #     print("The partial derivatives calculated from AD method from different forms of equation of state are not matching")

# if jnp.isclose(dP_dT_rho * dT_drho_P * drho_dP_T, -1.0):
#     print("The partial derivatives calculated from AD method from different forms of equation of state are matching")
# else:
#     print("The partial derivatives calculated from AD method from different forms of equation of state are not matching")


# # Compare with Turboflow's finite difference approximation (Cannot be done for Pressure as input)

# # def wrapped_ideal_gas(rho_T):
# #     """Wrapper for ideal gas equation for finite difference approximation."""
# #     rho, T = rho_T
# #     return ideal_gas_equation(rho, T)

# # gradient_approx = tf.approx_gradient(wrapped_ideal_gas, x=jnp.asarray([rho, T]), method="3-point", abs_step=1e-3)

# # print("Partial derivatives comparison:")

# # print("Partial derivative of pressure wrt rho (AD):", dP_drho_T)
# # print("Partial derivative of pressure wrt rho (FD):", gradient_approx[0])
# # print("Partial derivative of pressure wrt T (AD):", dP_dT_rho)
# # print("Partial derivative of pressure wrt T (DF):", gradient_approx[1])
