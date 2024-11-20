import jax.numpy as jnp
from jax import grad
import turboflow as tf

# Constants (common for all cases)
R = 287.0  # Specific gas constant for air (J/(kg*K))
gamma = 1.41  # Specific heat ratio for air
T_ref = 288.15  # Reference temperature (K)
P_ref = 101306.33  # Reference pressure (Pa)
s_ref = 1659.28  # Reference entropy (J/(kg*K))
myu_ref = 1.789e-5  # Reference dynamic viscosity (Kg/(m*s))
S_myu = 110.56  # Sutherland's constant for viscosity (K)
k_ref = 0.0241  # Reference thermal conductivity (W/(m*K))
S_k = 194  # Sutherland's constant for thermal conductivity (K)

def specific_heat(R, gamma):
    """Calculate specific heats at constant pressure and volume."""
    Cp = (gamma * R) / (gamma - 1)  # J/(kg*K)
    Cv = Cp / gamma  # J/(kg*K)
    return Cp, Cv

def temperature(h, prop, is_entropy, is_pressure):
    """Calculate temperature from enthalpy, pressure, or entropy."""
    Cp, Cv = specific_heat(R, gamma)
    
    if is_pressure:
        # Calculate temperature using pressure and entropy
        return T_ref * jnp.exp(((prop - s_ref) + (R * jnp.log(h / P_ref))) / Cp)
    else:
        return h / Cp  # From enthalpy

def density(h, prop, is_entropy, is_pressure):
    """Calculate density using enthalpy and either entropy or pressure."""
    T = temperature(h, prop, is_entropy, is_pressure)
    P = prop if is_pressure else pressure(h, prop)
    return P / (R * T)  # Density in kg/m^3

def pressure(h, s):
    """Calculate pressure using enthalpy and entropy."""
    T = temperature(h, s, True, False)
    Cp, _ = specific_heat(R, gamma)
    return P_ref * jnp.exp(((Cp * jnp.log(T / T_ref)) - (s - s_ref)) / R)

def entropy(h, P):
    """Calculate entropy from enthalpy and pressure."""
    Cp, _ = specific_heat(R, gamma)
    T = temperature(h, P, False, True)
    return s_ref + (Cp * jnp.log(T / T_ref)) - (R * jnp.log(P / P_ref))

def enthalpy(P, T):
    """Calculate enthalpy from temperature."""
    Cp, _ = specific_heat(R, gamma)
    return Cp * T  # enthalpy in Joules Kg.m^2/s^2

def viscosity(h, prop, is_entropy, is_pressure):
    """Calculate dynamic viscosity using Sutherland's formula."""
    T = temperature(h, prop, is_entropy, is_pressure)
    return myu_ref * ((T / T_ref) ** (3 / 2)) * ((T_ref + S_myu) / (T + S_myu))

def thermal_conductivity(h, prop, is_entropy, is_pressure):
    """Calculate thermal conductivity using Sutherland's formula."""
    T = temperature(h, prop, is_entropy, is_pressure)
    return k_ref * ((T / 273) ** (3 / 2)) * ((273 + S_k) / (T + S_k))

def speed_of_sound(h, prop, is_entropy, is_pressure):
    """Calculate speed of sound."""
    T = temperature(h, prop, is_entropy, is_pressure)
    return jnp.sqrt(gamma * R * T)  # Speed of sound in m/s

def calculate_properties_hs(h, s):
    """Calculate all thermodynamic properties given enthalpy and entropy."""
    return {
        "T": temperature(h, s, True, False),
        "P": pressure(h, s),
        "d": density(h, s, True, False), 
        "h": h,
        "s": s,
        "mu": viscosity(h, s, True, False), 
        "k": thermal_conductivity(h, s, True, False), 
        "a": speed_of_sound(h, s, True, False)
    }

def calculate_properties_hP(h, P):
    """Calculate all thermodynamic properties given enthalpy and pressure."""
    return {
        "T": temperature(h, P, False, True),
        "P": P,
        "d": density(h, P, False, True), 
        "h": h,
        "s": entropy(h, P),
        "mu": viscosity(h, P, False, True), 
        "k": thermal_conductivity(h, P, False, True), 
        "a": speed_of_sound(h, P, False, True)
    }

def calculate_properties_Ps(P, s):
    """Calculate all thermodynamic properties given pressure and entropy."""
    return {
        "T": temperature(P, s, True, True),
        "P": P,
        "d": density(P, s, True, True), 
        "h": enthalpy(P, s),
        "s": s,
        "mu": viscosity(P, s, True, True), 
        "k": thermal_conductivity(P, s, True, True), 
        "a": speed_of_sound(P, s, True, True)
    }

def calculate_properties_PT(P, T):
    """Calculate all thermodynamic properties given pressure and temperature."""
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

def calculate_properties_rhoh(rho, h):
    """Calculate all thermodynamic properties given density and enthalpy."""
    return {
        "T": temperature(h, rho),
        "P": pressure(rho, h),
        "d": rho, 
        "h": h,
        "s": entropy(rho, h),
        "mu": viscosity(rho, h), 
        "k": thermal_conductivity(rho, h), 
        "a": speed_of_sound(rho, h)
    }


### Print Results for Comparsion ###

# h, P = 400980.0, 103587.1484375

# # Calculate thermodynamic properties
# properties = calculate_properties_hP(h, P)

# tf.print_dict(properties)