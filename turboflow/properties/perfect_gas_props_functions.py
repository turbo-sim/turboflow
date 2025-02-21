# This script contains perfect gas equations for different input cases 
# This is used to calculate the thermodynamic properties and is imported in perfect_gas_props script

import jax.numpy as jnp
from jax import grad


# Constants
R = 287.0  # Specific gas constant for air (J/(kg*K))
gamma = 1.41  # Specific heat ratio for air
T_ref = 288.15  # Reference temperature (K)
P_ref = 101306.33  # Reference pressure (Pa)
s_ref = 1659.28  # Reference entropy (J/(kg*K))
myu_ref = 1.789e-5  # Reference dynamic viscosity (Kg/(m*s))
S_myu = 110.56  # Sutherland's constant for viscosity (K)
k_ref = 0.0241  # Reference thermal conductivity (W/(m*K))
S_k = 194  # Sutherland's constant for thermal conductivity (K)

PROPERTY_ALIAS = {
    # "P": "p",
    # "rho": "rhomass",
    # "d": "rhomass",
    # "u": "umass",
    # "h": "hmass",
    # "s": "smass",
    # "cv": "cvmass",
    # "cp": "cpmass",
    "speed_sound": "a",
    # "Z": "compressibility_factor",
    # "mu": "viscosity",
    # "k": "conductivity",
} 

def specific_heat(R, gamma):
        """Calculate specific heats at constant pressure and volume."""
        cp = (gamma * R) / (gamma - 1)  # J/(kg*K)
        cv = cp / gamma  # J/(kg*K)
        return cp, cv

def calculate_properties_hs(h, s):
    """Calculate all thermodynamic properties given enthalpy and Pressure."""

    def temperature(h, s):
        """Calculate temperature from enthalpy."""
        cp, cv = specific_heat(R, gamma)
        return jnp.max(jnp.asarray([1.0, h / cp])) # Temperature in Kelvin
        # return h / cp

    def pressure(h, s):
        """Calculate pressure using enthalpy and entropy."""
        T = temperature(h, s)
        cp, cv = specific_heat(R, gamma)
        return P_ref * jnp.exp(((cp * jnp.log(T / T_ref)) - (s - s_ref)) / R)

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

    properties = {"T": temperature(h, s),
            "p": pressure(h, s),
            "d": density(h, s), 
            "h": h,
            "s": s,
            "mu": viscosity(h, s), 
            "k": thermal_conductivity(h, s), 
            "a": speed_of_sound(h, s),
            "gamma": gamma}
    
    # # Add properties as aliases
    # for key, value in PROPERTY_ALIAS.items():
    #     properties[key] = properties[value]

    # Add properties as aliases
    for alias, original in PROPERTY_ALIAS.items():
        properties[alias] = properties[original]

    return properties

def calculate_properties_PT(P, T):
    """Calculate all thermodynamic properties given pressure and temperature."""

    if T <= 0:
        raise ValueError("Temperature must be greater than 0 K.")
    if P <= 0:
        raise ValueError("Pressure must be greater than 0 Pa.")
    
    def enthalpy(P, T):
        """Calculate enthalpy from temperature."""
        cp, cv = specific_heat(R, gamma)
        return cp*T  # enthalpy in Joules Kg.m^2/s^2

    def density(P, T):
        """Calculate density using pressure and temperature."""
        return P / (R * T)  # Density in kg/m^3

    def entropy(P, T):
        cp, cv = specific_heat(R, gamma)
        return s_ref + (cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

    def viscosity(P, T):
        """Calculate dynamic viscosity using Sutherland's formula."""
        return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

    def thermal_conductivity(P, T):
        """Calculate thermal conductivity using Sutherland's formula."""
        return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

    def speed_of_sound(P,T):
        return jnp.sqrt(gamma*R*T) # Speed of sound in m/s
    
    properties = {"T": T,
            "p": P,
            "d": density(P, T), 
            "h": enthalpy(P, T),
            "s": entropy(P, T),
            "mu": viscosity(P, T), 
            "k": thermal_conductivity(P, T), 
            "a": speed_of_sound(P, T),
            "gamma": gamma}
    
    # # Add properties as aliases
    # for key, value in PROPERTY_ALIAS.items():
    #     properties[key] = properties[value]

    # Add properties as aliases
    for alias, original in PROPERTY_ALIAS.items():
        properties[alias] = properties[original]

    return properties

def calculate_properties_hP(h, P):
    """Calculate all thermodynamic properties given enthalpy and Pressure."""

    def temperature(h, P):
        """Calculate temperature from enthalpy."""
        cp, cv = specific_heat(R, gamma)
        return jnp.max(jnp.asarray([1.0, h / cp])) 
        # return h / cp # Temperature in Kelvin

    def density(h, P):
        """Calculate density using enthalpy and pressure."""
        T = temperature(h, P)
        return P / (R * T)  # Density in kg/m^3

    def entropy(h, P):
        cp, cv = specific_heat(R, gamma)
        T = temperature(h, P)
        return s_ref + (cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

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

    properties = {"T": temperature(h, P),
            "p": P,
            "d": density(h, P), 
            "h": h,
            "s": entropy(h, P),
            "mu": viscosity(h, P), 
            "k": thermal_conductivity(h, P), 
            "a": speed_of_sound(h, P),
            "gamma": gamma}

    # # Add properties as aliases
    # for key, value in PROPERTY_ALIAS.items():
    #     properties[key] = properties[value]

    # Add properties as aliases
    for alias, original in PROPERTY_ALIAS.items():
        properties[alias] = properties[original]

    return properties

def calculate_properties_Ps(P, s):
    """Calculate all thermodynamic properties given enthalpy and Pressure."""

    def temperature(P, s):
        """Calculate temperature from enthalpy."""
        cp, cv = specific_heat(R, gamma)
        return jnp.max(jnp.asarray([1.0, T_ref * jnp.exp(((s - s_ref) + (R * jnp.log(P / P_ref))) / cp)]))  # Temperature in Kelvin
        # return T_ref * jnp.exp(((s - s_ref) + (R * jnp.log(P / P_ref))) / cp)
    
    def enthalpy(P, s):
        """Calculate enthalpy from temperature."""
        cp, cv = specific_heat(R, gamma)
        T = temperature(P, s)
        return cp * T  # enthalpy in Joules Kg.m^2/s^2

    def density(P, s):
        """Calculate density using enthalpy and entropy."""
        T = temperature(P, s)
        return P / (R * T)  # Density in kg/m^3

    def viscosity(P, s):
        """Calculate dynamic viscosity using Sutherland's formula."""
        T = temperature(P, s)
        return myu_ref * ((T / T_ref) ** (3/2)) * ((T_ref + S_myu) / (T + S_myu))

    def thermal_conductivity(P, s):
        """Calculate thermal conductivity using Sutherland's formula."""
        T = temperature(P, s)
        return k_ref * ((T / 273) ** (3/2)) * ((273 + S_k) / (T + S_k))

    def speed_of_sound(P, s):
        T = temperature(P, s)
        return jnp.sqrt(gamma*R*T) # Speed of sound in m/s

    properties = {"T": temperature(P, s),
            "p": P,
            "d": density(P, s), 
            "h": enthalpy(P, s),
            "s": s,
            "mu": viscosity(P, s), 
            "k": thermal_conductivity(P, s), 
            "a": speed_of_sound(P, s),
            "gamma": gamma}

    # # Add properties as aliases
    # for key, value in PROPERTY_ALIAS.items():
    #     properties[key] = properties[value]

    # Add properties as aliases
    for alias, original in PROPERTY_ALIAS.items():
        properties[alias] = properties[original]

    return properties

def calculate_properties_rhoh(rho, h):
    """Calculate all thermodynamic properties given enthalpy and Pressure."""

    def temperature(rho, h):
        """Calculate temperature from ethalpy."""
        cp, cv = specific_heat(R, gamma)
        return jnp.max(jnp.asarray([1.0, h / cp]))
        # return h / cp # enthalpy in Joules Kg.m^2/s^2

    def pressure(rho, h):
        """Calculate pressure using enthalpy and entropy."""
        T = temperature(rho, h)
        return rho*R*T

    def entropy(rho, h):
        cp, cv = specific_heat(R, gamma)
        T = temperature(rho, h)
        P = pressure(rho, h)
        return s_ref + (cp*jnp.log(T/T_ref)) - (R*jnp.log(P/P_ref))

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

    properties = {"T": temperature(rho, h),
            "p": pressure(rho, h),
            "d": rho, 
            "h": h,
            "s": entropy(rho, h),
            "mu": viscosity(rho, h), 
            "k": thermal_conductivity(rho, h), 
            "a": speed_of_sound(rho, h),
            "gamma": gamma}

    # # Add properties as aliases
    # for key, value in PROPERTY_ALIAS.items():
    #     properties[key] = properties[value]

    # Add properties as aliases
    for alias, original in PROPERTY_ALIAS.items():
        properties[alias] = properties[original]

    return properties

