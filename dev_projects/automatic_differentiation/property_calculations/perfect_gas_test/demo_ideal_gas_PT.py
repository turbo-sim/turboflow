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

# def enthalpy(P, T):
#     """Calculate enthalpy from temperature."""
#     Cp, Cv = specific_heat(R, gamma)
#     return Cp*T  # enthalpy in Joules Kg.m^2/s^2

# def density(P, T):
#     """Calculate density using pressure and temperature."""
#     return P / (R * T)  # Density in kg/m^3

# def entropy(P, T):
#     Cp, Cv = specific_heat(R, gamma)
#     return s_ref + (Cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

# def viscosity(P, T):
#     """Calculate dynamic viscosity using Sutherland's formula."""
#     return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

# def thermal_conductivity(P, T):
#     """Calculate thermal conductivity using Sutherland's formula."""
#     return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

# def speed_of_sound(P,T):
#     return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

def calculate_properties_PT(P, T):
    """Calculate all thermodynamic properties given pressure and temperature."""

    def specific_heat(R, gamma):
        """Calculate specific heats at constant pressure and volume."""
        Cp = (gamma * R) / (gamma - 1)  # J/(kg*K)
        Cv = Cp / gamma  # J/(kg*K)
        return Cp, Cv

    def enthalpy(P, T):
        """Calculate enthalpy from temperature."""
        Cp, Cv = specific_heat(R, gamma)
        return Cp*T  # enthalpy in Joules Kg.m^2/s^2

    def density(P, T):
        """Calculate density using pressure and temperature."""
        return P / (R * T)  # Density in kg/m^3

    def entropy(P, T):
        Cp, Cv = specific_heat(R, gamma)
        return s_ref + (Cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

    def viscosity(P, T):
        """Calculate dynamic viscosity using Sutherland's formula."""
        return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

    def thermal_conductivity(P, T):
        """Calculate thermal conductivity using Sutherland's formula."""
        return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

    def speed_of_sound(P,T):
        return jnp.sqrt(gamma*R*T) # Speed of sound in m/s
    
    if T <= 0:
        raise ValueError("Temperature must be greater than 0 K.")
    if P <= 0:
        raise ValueError("Pressure must be greater than 0 Pa.")

    properties = {
        "T": T,
        "P": P,
        "d": density(P, T), 
        "h": enthalpy(P, T),
        "s": entropy(P, T),
        "mu": viscosity(P, T), 
        "k": thermal_conductivity(P, T), 
        "a": speed_of_sound(P, T)
    }
    return properties



# Example input: Pressure (P) and Temperature (T)
# P, T = 103587.1484375, 406.261

# # Calculate thermodynamic properties
# # Calculate thermodynamic properties
# properties = calculate_properties_PT(P, T)

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

# def compute_partial_derivatives(P, T):
#     """Compute partial derivatives of properties."""
    
#     drho_dP_func = grad(density, argnums=0)  # Partial derivative with respect to P at constant T
#     drho_dT_func = grad(density, argnums=1)  # Partial derivative with respect to T at constant P
    
#     return {
#         "drho_dP_T": drho_dP_func(P, T),
#         "drho_dT_P": drho_dT_func(P, T)
#     }


# # Compute partial derivatives
# partial_derivatives = compute_partial_derivatives(P, T)

# # Print partial derivatives
# for key, value in partial_derivatives.items():
#     print(f"{key}: {value}")


# ## Verifying the results from automatic differentiation

# drho_dP_T = partial_derivatives["drho_dP_T"]
# dT_drho_P = 1 / partial_derivatives["drho_dT_P"]  # Example usage

# # Define the ideal gas equation of state: p = rho * R * T
# def ideal_gas_equation(rho, T):
#     return rho * R * T

# # With Inputs T and rho
# dP_drho_func = grad(ideal_gas_equation, argnums=0)  # Partial derivative with respect to rho
# dP_dT_func = grad(ideal_gas_equation, argnums=1)    # Partial derivative with respect to T

# dP_dT_rho = dP_dT_func(rho, T)
# dP_drho_T = dP_drho_func(rho, T)

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
