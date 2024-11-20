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

# def temperature(rho, h):
#     """Calculate temperature from ethalpy."""
#     Cp, Cv = specific_heat(R, gamma)
#     return h / Cp  # enthalpy in Joules Kg.m^2/s^2

# def pressure(rho, h):
#     """Calculate pressure using enthalpy and entropy."""
#     T = temperature(rho, h)
#     return rho*R*T

# def entropy(rho, h):
#     Cp, Cv = specific_heat(R, gamma)
#     T = temperature(rho, h)
#     P = pressure(rho, h)
#     return s_ref + (Cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

# def viscosity(rho, h):
#     """Calculate dynamic viscosity using Sutherland's formula."""
#     T = temperature(rho, h)
#     return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

# def thermal_conductivity(rho, h):
#     """Calculate thermal conductivity using Sutherland's formula."""
#     T = temperature(rho, h)
#     return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

# def speed_of_sound(rho, h):
#     T = temperature(rho, h)
#     return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

def calculate_properties_rhoh(rho, h):
    """Calculate all thermodynamic properties given enthalpy and Pressure."""

    def specific_heat(R, gamma):
        """Calculate specific heats at constant pressure and volume."""
        Cp = (gamma * R) / (gamma - 1)  # J/(kg*K)
        Cv = Cp / gamma  # J/(kg*K)
        return Cp, Cv

    def temperature(rho, h):
        """Calculate temperature from ethalpy."""
        Cp, Cv = specific_heat(R, gamma)
        return h / Cp  # enthalpy in Joules Kg.m^2/s^2

    def pressure(rho, h):
        """Calculate pressure using enthalpy and entropy."""
        T = temperature(rho, h)
        return rho*R*T

    def entropy(rho, h):
        Cp, Cv = specific_heat(R, gamma)
        T = temperature(rho, h)
        P = pressure(rho, h)
        return s_ref + (Cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

    def viscosity(rho, h):
        """Calculate dynamic viscosity using Sutherland's formula."""
        T = temperature(rho, h)
        return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

    def thermal_conductivity(rho, h):
        """Calculate thermal conductivity using Sutherland's formula."""
        T = temperature(rho, h)
        return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

    def speed_of_sound(rho, h):
        T = temperature(rho, h)
        return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

    return {"T": temperature(rho, h),
            "P": pressure(rho, h),
            "d": rho, 
            "h": h,
            "s": entropy(rho, h),
            "mu": viscosity(rho, h), 
            "k": thermal_conductivity(rho, h), 
            "a": speed_of_sound(rho, h)}



# Example input: Pressure (P) and Temperature (T)
# rho, h = 0.8884, 400980.0


# # Calculate thermodynamic properties
# # Calculate thermodynamic properties
# properties = calculate_properties_rhoh(rho, h)

# tf.print_dict(properties)

# T = properties["T"]
# P = properties["P"]
# rho = properties["d"]
# h = properties["h"]
# s = properties["s"]
# mu = properties["mu"]
# k = properties["k"]
# a = properties["a"]



# Print the results
# print(f"Temperature: {T}\nPressure: {P}\nDensity: {rho}\nEnthalpy: {h}\nEntropy: {s}\nViscosity: {mu}\nThermal Conductivity: {k}")

############################################################ For partial derivatives verification
# def compute_partial_derivatives(rho, h):
#     """Compute partial derivatives of properties."""
    
#     dP_drho_func = grad(pressure, argnums=0)  # Partial derivative with respect to P at constant rho
#     dP_dh_func = grad(pressure, argnums=1)  # Partial derivative with respect to T at constant h
    
#     return {
#         "dP_drho_h": dP_drho_func(rho, h),
#         "dP_dh_rho": dP_dh_func(rho, h)
#     }


# def compute_partial_derivatives(rho, h):
#     """Compute partial derivatives of properties."""
    
#     dP_drho_func = grad(pressure, argnums=0)  # Partial derivative with respect to P at constant rho
#     dP_dh_func = grad(pressure, argnums=1)  # Partial derivative with respect to T at constant h
    
#     return {
#         "dP_drho_h": dP_drho_func(rho, h),
#         "dP_dh_rho": dP_dh_func(rho, h)
#     }


# # Compute partial derivatives
# partial_derivatives = compute_partial_derivatives(rho, h)

# # Print partial derivatives
# for key, value in partial_derivatives.items():
#     print(f"{key}: {value}")


# ## Verifying the results from automatic differentiation

# dP_drho_h = partial_derivatives["dP_drho_h"]
# dP_dh_rho = 1 / partial_derivatives["dP_dh_rho"]  # Example usage

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

# if jnp.isclose(dP_drho_h, dP_drho_T):
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
