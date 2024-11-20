# This script should be used in turboflow for thermodynamic property calculations 

from turboflow.properties import perfect_gas_props_func1

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
    
    return properties

# rho, h = 0.8884, 400980.0

# properties = perfect_gas_props("DmassHmass_INPUTS", rho, h)

# print(properties)
