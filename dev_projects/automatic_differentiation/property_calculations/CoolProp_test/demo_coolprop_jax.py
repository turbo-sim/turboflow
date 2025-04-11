###############     TESTING CODE FOR CUSTOMJVP ON COOLPROP START ###############

import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import copy

import jax
from functools import partial


Fluid = tf.Fluid

# @jax.custom_vjp # Comment when not using custom_vjp
# def get_props(fluid_name, input_state, prop1, prop2):
#     # print(f"get_props called with: input_state={input_state}, fluid_name={fluid_name}, prop1={prop1}, prop2={prop2}")

#     # fluid_name = {v: k for k, v in FLUID_MAP.items()}[fluid_id]

#     fluid = Fluid(fluid_name)  # Making a Fluid class object # Is it traced by JAX??

#     prop_dict = fluid.get_props(input_state, prop1, prop2).to_dict()  # Calling a method of the Fluid Class for the fluid object # Returns a dictionary  

#     return prop_dict

#### CUSTOM_JVP TRIAL START #####

@partial(jax.custom_jvp, nondiff_argnums= (0, 1))
def get_props_custom_jvp(fluid_name, input_state, prop1, prop2):
    # input_state = input_state_map[input_state]
    fluid = Fluid(fluid_name)
    return fluid.get_props(input_state, prop1, prop2).to_dict()

# Define the forward-mode JVP function for perfect_gas_props
# Define the JVP function
@get_props_custom_jvp.defjvp
def get_props_custom_jvp_jvp(fluid_name, input_state, primals, tangents):

    print ("Custom JVP is being used....")

    prop1, prop2 = primals
    prop1_dot, prop2_dot = tangents  # Directional derivatives
    
    # Small step for finite difference
    delta = 1e-3/(1e-6 + (prop1**2 + prop2**2)**0.5)

    fluid = Fluid(fluid_name)

    # Compute finite difference approximations for partial derivatives

    properties_base = fluid.get_props( input_state, prop1, prop2).to_dict()
    properties_dprop1 = fluid.get_props( input_state, prop1 + delta, prop2).to_dict()
    properties_dprop2 = fluid.get_props( input_state, prop1, prop2 + delta).to_dict()

    # Compute partial derivatives
    df_dprop1 = {
    key: 0 if (
        properties_base[key] is None or
        properties_dprop1[key] is None or
        isinstance(properties_base[key], str) or
        isinstance(properties_dprop1[key], str) or
        np.isnan(properties_base[key]) or
        np.isnan(properties_dprop1[key])
    )
    else (properties_dprop1[key] - properties_base[key]) / delta
    for key in properties_base
    }



    # df_dprop1 = {}


    # print(properties_base.keys())

    # for key in properties_base.keys():
    #     pass
    #     print(f"\nKey: {key}")
    #     base_val = properties_base[key]
    #     dprop1_val = properties_dprop1[key]
        
    #     print(f"\nKey: {key}")
    #     print(f"Base value: {base_val}")
    #     print(f"dProp1 value: {dprop1_val}")
        
    #     if (
    #         base_val is None or
    #         dprop1_val is None or
    #         isinstance(base_val, str) or
    #         isinstance(dprop1_val, str) or
    #         np.isnan(base_val) or
    #         np.isnan(dprop1_val)
    #     ):
    #         df_dprop1[key] = 0
    #         print("Condition met for 0. Assigned 0.")
    #     else:
    #         df_dprop1[key] = (dprop1_val - base_val) / delta
    #         print(f"Computed derivative: {df_dprop1[key]}")


    df_dprop2 = {
    key: 0 if (
        properties_base[key] is None or
        properties_dprop2[key] is None or
        isinstance(properties_base[key], str) or
        isinstance(properties_dprop2[key], str) or
        np.isnan(properties_base[key]) or
        np.isnan(properties_dprop2[key])
    )
    else (properties_dprop2[key] - properties_base[key]) / delta
    for key in properties_base
    }

   # df_dprop2 = {
   #     key: (properties_dprop2[key] - properties_base[key]) / delta
   #     for key in properties_base
   # }

    # Compute JVP (directional derivative)
    jvp = {
        key: df_dprop1[key] * prop1_dot + df_dprop2[key] * prop2_dot
        for key in properties_base
    }

    print("jvp", jvp)
    
    return properties_base, jvp

###### CUSTOM JVP TRIAL END #####

###### CUSTOM VJP TRIAL START ######

# def fwd(input_state, prop1, prop2):
#     return get_props(input_state, prop1, prop2), (input_state, prop1, prop2) 

# def bwd(res, g):
#     input_state, prop1, prop2 = res

#     delta = 1e-3/(1e-6 + (prop1**2 + prop2**2)**0.5)

#     properties_base = get_props(input_state, prop1, prop2)
#     properties_dprop1 = get_props(input_state, prop1 + delta, prop2)
#     properties_dprop2 = get_props(input_state, prop1, prop2 + delta)

#     df_dprop1 = {
#         key: (properties_dprop1[key] - properties_base[key]) / delta
#         for key in properties_base
#     }

#     df_dprop2 = {
#         key: (properties_dprop2[key] - properties_base[key]) / delta
#         for key in properties_base
#     }

#     return None, g * df_dprop1, g * df_dprop2

# get_props.defvjp(fwd, bwd)



###### CUSTOM VJP TRIAL END ######


############### TESTING CODE START ################

fluid_name = "CO2"
input_state = CP.HmassSmass_INPUTS
# fluid_id = 0.0
prop1 = 400980.0 
prop2 = 1991.94

jax.config.update("jax_traceback_filtering", "off")


# Encode fluid_name to an integer
# fluid_id = FLUID_MAP[fluid_name]

# fluid_name = {v: k for k, v in FLUID_MAP.items()}[fluid_id]

# fluid = Fluid(fluid_name)  # Making a Fluid class object # Is it traced by JAX??

# prop_dict = get_props(input_state, fluid_name, prop1, prop2)
# print(prop_dict)

# grad_func = jax.jacrev(get_props, argnums=(1, 2))



# grad_func = jax.jacfwd(get_props_custom_jvp, argnums=(2, 3))
# gradients = grad_func(fluid_name, input_state, prop1, prop2)
# print(gradients)


# Fix the nondiff args into a new function
get_props_wrapped = lambda prop1, prop2: get_props_custom_jvp("CO2", CP.HmassSmass_INPUTS, prop1, prop2)

# Then differentiate only w.r.t. the numeric args
grad_func = jax.jacfwd(get_props_wrapped, argnums=(0, 1))
gradients = grad_func(prop1, prop2)


###############     TESTING CODE  END     ###############