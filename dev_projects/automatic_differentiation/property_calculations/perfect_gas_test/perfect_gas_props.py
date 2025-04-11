# This script should be used in turboflow for thermodynamic property calculations 

import perfect_gas_props_functions
from turboflow.utilities import print_dict
import jax
import jax.numpy as jnp
from turboflow.pysolver_view import approx_gradient
from jax import grad, jvp, jacobian
from functools import partial

jax.config.update("jax_enable_x64", True)

# Import property calculation functions into a dictionary for easy access
# property_calculators = {
#     "HmassSmass_INPUTS": perfect_gas_props_functions.calculate_properties_hs,
#     "PSmass_INPUTS": perfect_gas_props_functions.calculate_properties_Ps,
#     "PT_INPUTS": perfect_gas_props_functions.calculate_properties_PT,
#     "HmassP_INPUTS": perfect_gas_props_functions.calculate_properties_hP,
#     "DmassHmass_INPUTS": perfect_gas_props_functions.calculate_properties_rhoh
# }

property_calculators = {
    "HmassSmass_INPUTS": perfect_gas_props_functions.calculate_properties_hs,   # "HmassSmass_INPUTS"
    "PSmass_INPUTS": perfect_gas_props_functions.calculate_properties_Ps,   # "PSmass_INPUTS"
    "PT_INPUTS": perfect_gas_props_functions.calculate_properties_PT,   # "PT_INPUTS"
    "HmassP_INPUTS": perfect_gas_props_functions.calculate_properties_hP,   # "HmassP_INPUTS"
    "DmassHmass_INPUTS": perfect_gas_props_functions.calculate_properties_rhoh  # "DmassHmass_INPUTS"
}

# # Create a lookup table to convert strings to integers
# input_state_map = {
#     "HmassSmass_INPUTS": 0.0,
#     "PSmass_INPUTS": 1.0,
#     "PT_INPUTS": 2.0,
#     "HmassP_INPUTS": 3.0,
#     "DmassHmass_INPUTS": 4.0
# }

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


######### WORKING CODE START #########
# Define the custom JVP for the perfect_gas_props function
def perfect_gas_props(fluid_name, input_state, prop1, prop2):

    # print("You are in in-house perfect gas equations script!!")
    """Calculate properties based on the specified input state."""
    
    # if isinstance(input_state, str):
    #     input_state = input_state_map[input_state]

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
                f"For input state '{input_state}' with inputs prop1={prop1:0.2f}, prop2={prop2:0.2f}, some properties are NaN or complex:\n"
                f"{print_dict(properties, return_output=True)}"
            )
    
    return properties

@partial(jax.custom_jvp,nondiff_argnums=(0, 1))
def perfect_gas_props_custom_jvp(fluid_name, input_state, prop1, prop2):
    # input_state = input_state_map[input_state]
    return perfect_gas_props(fluid_name, input_state, prop1, prop2)

# Define the forward-mode JVP function for perfect_gas_props
# Define the JVP function
@perfect_gas_props_custom_jvp.defjvp
def perfect_gas_props_custom_jvp_jvp(fluid_name, input_state, primals, tangents):
    # input_state, prop1, prop2 = primals
    # _, prop1_dot, prop2_dot = tangents  # Directional derivatives

    prop1, prop2 = primals
    prop1_dot, prop2_dot = tangents  # Directional derivatives
    
    # input_state_dot = 0.0  # We don't calculate partial derivatives with respect to input state. This is hard-coded.
    
    # Small step for finite difference
    delta = 1e-3/(prop1**2+prop2**2)**0.5

    # Compute finite difference approximations for partial derivatives
    # properties_dprop = perfect_gas_props(input_state, prop1 + delta*prop1_dot, prop2+delta*prop2_dot)
    # properties_base = perfect_gas_props(input_state, prop1, prop2)
    # properties_dprop1 = perfect_gas_props(input_state, prop1 + delta, prop2)
    # properties_dprop2 = perfect_gas_props(input_state, prop1, prop2 + delta)

    properties_base = perfect_gas_props(fluid_name, input_state, prop1, prop2)
    properties_dprop1 = perfect_gas_props(fluid_name, input_state, prop1 + delta, prop2)
    properties_dprop2 = perfect_gas_props(fluid_name, input_state, prop1, prop2 + delta)


    # Compute partial derivatives
    df_dprop1 = {
        key: (properties_dprop1[key] - properties_base[key]) / delta
        for key in properties_base
    }
    df_dprop2 = {
        key: (properties_dprop2[key] - properties_base[key]) / delta
        for key in properties_base
    }

    # Compute JVP (directional derivative)
    jvp = {
        key: df_dprop1[key] * prop1_dot + df_dprop2[key] * prop2_dot
        for key in properties_base
    }
    
    return properties_base, jvp

######### WORKING CODE END #########

fluid_name =  'test'
input_state_string = "PT_INPUTS"  # Input state as a string
prop1 = 100000.0  # Example property 1 (e.g., enthalpy in J/kg)
prop2 = 300.0   # Example property 2 (e.g., entropy in J/(kg*K))

grad_func = jax.jacfwd(perfect_gas_props_custom_jvp, argnums = (2,3))
gradients = grad_func(fluid_name, input_state_string, prop1, prop2)
print(gradients)

###### BELOW CODES ARE FOR TESTING #######

# def test_custom_jvp():
#     # Example inputs
#     input_state_string = "HmassSmass_INPUTS"  # Input state as a string
#     prop1 = 400980.0  # Example property 1 (e.g., enthalpy in J/kg)
#     prop2 = 1991.94   # Example property 2 (e.g., entropy in J/(kg*K))

#     # Convert string to numeric input state
#     input_state = input_state_map[input_state_string]

#     # Compute JVP using JAX's `jvp()` function
#     primals = (prop1, prop2)

#     # Compute JVP w.r.t. prop1 (tangent (1.0, 0.0)) and prop2 (tangent (0.0, 1.0))
#     properties_base, jvp_custom_prop1 = jvp(lambda p1, p2: perfect_gas_props_custom_jvp(input_state, p1, p2), primals, (1.0, 0.0))
#     _, jvp_custom_prop2 = jvp(lambda p1, p2: perfect_gas_props_custom_jvp(input_state, p1, p2), primals, (0.0, 1.0))

#     print("Custom JVP (Directional Derivatives):")
#     print("∂properties/∂prop1:", jvp_custom_prop1)
#     print("∂properties/∂prop2:", jvp_custom_prop2)

#     # Compute Jacobian using JAX automatic differentiation
#     properties_func = partial(perfect_gas_props_custom_jvp, input_state)  # Fix: Bind input_state
#     jacobian_prop1 = jacobian(properties_func, argnums=0)(prop1, prop2)  # ∂properties/∂prop1
#     jacobian_prop2 = jacobian(properties_func, argnums=1)(prop1, prop2)  # ∂properties/∂prop2

#     print("\nJAX Automatic Differentiation (Jacobian):")
#     print("∂properties/∂prop1:", jacobian_prop1)
#     print("∂properties/∂prop2:", jacobian_prop2)

#     # Compare JVP with JAX Jacobian
#     for key in properties_base.keys():
#         jvp_val_prop1 = jvp_custom_prop1[key]
#         jvp_val_prop2 = jvp_custom_prop2[key]
#         jacobian_val_prop1 = jacobian_prop1[key]
#         jacobian_val_prop2 = jacobian_prop2[key]

#         print(f"Property: {key}, Custom JVP ∂/∂prop1: {jvp_val_prop1}, JAX Jacobian ∂/∂prop1: {jacobian_val_prop1}")
#         print(f"Property: {key}, Custom JVP ∂/∂prop2: {jvp_val_prop2}, JAX Jacobian ∂/∂prop2: {jacobian_val_prop2}")

#         assert jnp.allclose(jvp_val_prop1, jacobian_val_prop1, atol=1e-5), f"JVP mismatch for {key} in ∂/∂prop1"
#         assert jnp.allclose(jvp_val_prop2, jacobian_val_prop2, atol=1e-5), f"JVP mismatch for {key} in ∂/∂prop2"

#     print("\nCustom JVP matches JAX automatic differentiation!")

# # Run the test
# test_custom_jvp()    

# # Now, test the custom JVP with forward-mode differentiation

# input_state_string = "HmassP_INPUTS"  # Example input state
# prop1_value = jnp.array(400980.0)  # Example value for prop1
# prop2_value = jnp.array(103587.1484375)    # Example value for prop2

# # Convert input_state into a JAX-compatible constant (by wrapping it as a string)
# # You do not need to change input_state for differentiation since it's just a key for function lookup
# input_state = input_state_map[input_state_string]

# # List of input variables and their corresponding names
# input_vars = [("h", prop1_value), ("p", prop2_value)]  
# values = (input_state, prop1_value, prop2_value)

# # Dictionary to store partial derivatives
# partial_derivatives = {}

# # Loop over each input variable
# for i, (var_name, var_value) in enumerate(input_vars):
#     # Create tangent vector with 1.0 for the current variable, 0.0 for others
#     tangents = [0.0] * len(input_vars)
#     tangents[i] = 1.0  # Set current variable's tangent to 1.0
    
#     # Compute JVP to get partial derivative with respect to the current variable
#     properties_values, derivs = jax.jvp(perfect_gas_props_custom_jvp, values, (0.0, *tuple(tangents)))
    
#     # Find the other variable name dynamically
#     other_var_name = input_vars[1 - i][0]  # Picks the other variable from input_vars list
    
#     # Store only the partial derivatives for the current variable
#     for key in derivs:
#         partial_derivatives[f"d{key}_d{var_name}_{other_var_name}"] = derivs[key]

# print_dict(partial_derivatives)

# T = properties_values["T"]
# P = properties_values["p"]
# rho = properties_values["d"]
# h = properties_values["h"]
# s = properties_values["s"]
# mu = properties_values["mu"]
# k = properties_values["k"]
# a = properties_values["a"]

# # Define the ideal gas equation of state: p = rho * R * T
# def ideal_gas_equation(rho, T):
#     return rho * 287 * T

# ## Verifying the derivatives from JVP for hs case

# dP_dT_s = partial_derivatives["dp_dh_s"] / partial_derivatives["dT_dh_s"]  # Example usage
# dP_drho_h = partial_derivatives["dp_ds_h"] / partial_derivatives["dd_ds_h"]  # Example usage

# # With Inputs T and rho
# dP_drho = grad(ideal_gas_equation, argnums=0)  # Partial derivative with respect to rho
# dP_dT = grad(ideal_gas_equation, argnums=1)    # Partial derivative with respect to T

# dP_dT_rho = dP_dT(rho, T)
# dP_drho_T = dP_drho(rho, T)

# if jnp.isclose(dP_dT_s, (1.41 / (1.41 - 1)) * dP_dT_rho) and jnp.isclose(dP_drho_h, dP_drho_T):
#     print("The partial derivatives calculated from customjvp and from different forms of equation of state are matching")
# else:
#     print("The partial derivatives calculated from customjvp and from different forms of equation of state are not matching")

# ## Verifying the derivatives from JVP for hP case

# dT_drho_P = partial_derivatives["dT_dh_p"] / partial_derivatives["dd_dh_p"]  # Example usage
# drho_dP_T = partial_derivatives["dd_dp_h"]

# # With Inputs T and rho
# dP_drho = grad(ideal_gas_equation, argnums=0)  # Partial derivative with respect to rho
# dP_dT = grad(ideal_gas_equation, argnums=1)    # Partial derivative with respect to T

# dP_dT_rho = dP_dT(rho, T)
# dP_drho_T = dP_drho(rho, T)

# if jnp.isclose(dP_dT_rho * dT_drho_P * drho_dP_T, -1.0):
#     print("The partial derivatives calculated from customjvp and from different forms of equation of state are matching")
# else:
#     print("The partial derivatives calculated from customjvp and from different forms of equation of state are not matching")
