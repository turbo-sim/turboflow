
import jax.numpy as jnp
# from jax import grad
import turboflow as tf

from dev_projects.automatic_differentiation.demo_ideal_gas_hs import calculate_properties_hs
from dev_projects.automatic_differentiation.demo_ideal_gas_hP import calculate_properties_hP
from dev_projects.automatic_differentiation.demo_ideal_gas_Ps import calculate_properties_Ps
from dev_projects.automatic_differentiation.demo_ideal_gas_PT import calculate_properties_PT
from dev_projects.automatic_differentiation.demo_ideal_gas_rhoh import calculate_properties_rhoh


# Fluid Constants (Should be changed for different fluids) 
# Given values are for air

R = 287.0  # Specific gas constant for air (J/(kg*K))
gamma = 1.41  # Specific heat ratio for air
T_ref = 288.15  # Reference temperature (K)
P_ref = 101306.33  # Reference pressure (Pa)
s_ref = 1659.28  # Reference entropy (J/(kg*K))
myu_ref = 1.789e-5  # Reference dynamic viscosity (Kg/(m*s))
S_myu = 110.56  # Sutherland's constant for viscosity (K)
k_ref = 0.0241  # Reference thermal conductivity (W/(m*K))
S_k = 194  # Sutherland's constant for thermal conductivity (K)


def perfect_gas_props(input_state, prop1, prop2):

    if input_state == "HmassSmass_INPUTS":

        h = prop1
        s = prop2

        properties = calculate_properties_hs(h, s)
    
    if input_state == "cp.PSmass_INPUTS":

        P = prop1
        s = prop2

        properties = calculate_properties_Ps(P, s)
    
    if input_state == "cp.PT_INPUTS":

        P = prop1
        T = prop2

        properties = calculate_properties_PT(P, T)
    
    if input_state == "cp.HmassP_INPUTS":

        h = prop1
        P = prop2

        properties = calculate_properties_hP(h, P)
    
    if input_state == "CP.DmassHmass_INPUTS":

        rho = prop1
        h = prop2

        properties = calculate_properties_rhoh(rho, h)
    
    return properties

# rho, h = 0.8884, 400980.0

# properties = perfect_gas_props("CP.DmassHmass_INPUTS", rho, h)

# print(properties)
