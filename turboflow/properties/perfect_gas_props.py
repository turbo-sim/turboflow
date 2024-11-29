# This script should be used in turboflow for thermodynamic property calculations 
import jax.numpy as jnp

from . import perfect_gas_props_func1
from ..utilities import print_dict

# Import property calculation functions into a dictionary for easy access
property_calculators = {
    "HmassSmass_INPUTS": perfect_gas_props_func1.calculate_properties_hs,
    "PSmass_INPUTS": perfect_gas_props_func1.calculate_properties_Ps,
    "PT_INPUTS": perfect_gas_props_func1.calculate_properties_PT,
    "HmassP_INPUTS": perfect_gas_props_func1.calculate_properties_hP,
    "DmassHmass_INPUTS": perfect_gas_props_func1.calculate_properties_rhoh
}

# Fluid Constants (Change for different fluids; values are for air)
# fluid_constants = {
#     "R": 287.0,          # Specific gas constant for air (J/(kg*K))
#     "gamma": 1.41,      # Specific heat ratio for air
#     "T_ref": 288.15,    # Reference temperature (K)
#     "P_ref": 101306.33, # Reference pressure (Pa)
#     "s_ref": 1659.28,   # Reference entropy (J/(kg*K))
#     "myu_ref": 1.789e-5, # Reference dynamic viscosity (Kg/(m*s))
#     "S_myu": 110.56,    # Sutherland's constant for viscosity (K)
#     "k_ref": 0.0241,    # Reference thermal conductivity (W/(m*K))
#     "S_k": 194          # Sutherland's constant for thermal conductivity (K)
# }

def perfect_gas_props(input_state, prop1, prop2):

    # print("You are in in-house perfect gas equations script!!")
    """Calculate properties based on the specified input state."""
    
    # Retrieve the appropriate calculation function
    calculate_properties = property_calculators.get(input_state)
    
    if calculate_properties is None:
        raise ValueError(f"Unknown input state: {input_state}")

    # Call the corresponding property calculation function 
    properties = calculate_properties(prop1, prop2)

    # Check for NaN or complex values in the properties
    
    if any(jnp.isnan(value) or isinstance(value, complex) for value in properties.values()):
        # Raise an error with detailed information
            raise ValueError(
                f"For input state '{input_state}' with inputs prop1={prop1:0.2f}, prop2={prop2:0.2f}, some properties are NaN or complex:"
                f"{print_dict(properties, return_output=True)}"
            )
    
    return properties
