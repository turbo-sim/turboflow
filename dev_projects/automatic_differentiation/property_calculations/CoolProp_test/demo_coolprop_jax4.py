###############     TESTING CODE FOR CUSTOMJVP ON COOLPROP START ###############

import turboflow as tf
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import copy

import jax
import jax.numpy as jnp
from functools import partial

Fluid = tf.Fluid

# Define dictionary with dynamically generated fields
PHASE_INDEX = {attr: getattr(CP, attr) for attr in dir(CP) if attr.startswith("iphase")}
INPUT_PAIRS = {attr: getattr(CP, attr) for attr in dir(CP) if attr.endswith("_INPUTS")}
PHASE_INDEX = sorted(PHASE_INDEX.items(), key=lambda x: x[1])
INPUT_PAIRS = sorted(INPUT_PAIRS.items(), key=lambda x: x[1])

INPUT_TYPE_MAP = {v: k for k, v in INPUT_PAIRS}

# Convert each input key to a tuple of FluidState variable names
# Capitalized names that should not be lowercased
preserve_case = {'T', 'Q'}

def extract_vars(name):
    base = name.replace("_INPUTS", "")
    parts = []
    current = base[0]
    for c in base[1:]:
        if c.isupper():
            parts.append(current)
            current = c
        else:
            current += c
    parts.append(current)
    return tuple(p if p in preserve_case else p.lower() for p in parts)

INPUT_PAIR_MAP = {k: extract_vars(v) for k, v in INPUT_TYPE_MAP.items()}

#####  CUSTOM_JVP TRIAL START  #####

fluid = Fluid("CO2")

@partial(jax.custom_jvp, nondiff_argnums= (0, 1))
def get_props_custom_jvp(fluid, input_state, prop1, prop2):
    # input_state = input_state_map[input_state]

    # fluid = tf.Fluid(fluid_name)
    properties_base = fluid.get_props(input_state, prop1, prop2).to_dict()
    properties_base = {
        key: 0. if (
            properties_base[key] is None or
            isinstance(properties_base[key], str) or
            np.isnan(properties_base[key])
        )
        else properties_base[key]
        for key in properties_base
    }

    return properties_base

# Define the forward-mode JVP function for perfect_gas_props
# Define the JVP function
@get_props_custom_jvp.defjvp
def get_props_custom_jvp_jvp(fluid, input_state, primals, tangents):

    # print ("Custom JVP is being used....")

    prop1, prop2 = primals
    prop1_dot, prop2_dot = tangents  # Directional derivatives

    prop1_name, prop2_name = INPUT_PAIR_MAP[input_state]
    
    # Small step for finite difference

    delta_prop1 = 1e-5 * fluid.reference_state[prop1_name]
    delta_prop2 = 1e-5 * fluid.reference_state[prop2_name]

    alpha = jnp.sqrt((prop1_dot/delta_prop1)**2 + (prop2_dot/delta_prop2)**2)

    # prop1_dot = float(prop1_dot)
    # prop2_dot = float(prop2_dot)
    # alpha = float(alpha)

    # prop1_dot = float(jax.lax.stop_gradient(prop1_dot))
    # prop2_dot = float(jax.lax.stop_gradient(prop2_dot))
    # alpha = float(jax.lax.stop_gradient(alpha))

    # prop1_dot = float(prop1_dot.item())
    # prop2_dot = float(prop2_dot.item())
    # alpha = float(alpha.item())

    # print(tf.INPUT_PAIR_MAP)

    # fluid = tf.Fluid(fluid_name)

    # Compute finite difference approximations for partial derivatives
    properties_base = fluid.get_props(input_state, prop1, prop2).to_dict()
    # properties_dprop1 = fluid.get_props( input_state, prop1 + delta_prop1, prop2).to_dict()
    # properties_dprop2 = fluid.get_props( input_state, prop1, prop2 + delta_prop2).to_dict()
    properties_dprop_full = fluid.get_props(input_state, prop1 + (prop1_dot/alpha), prop2 + (prop2_dot/alpha)).to_dict()

    # # Compute partial derivatives
    # df_dprop1 = {
    # key: 0 if (
    #     properties_base[key] is None or
    #     properties_dprop1[key] is None or
    #     isinstance(properties_base[key], str) or
    #     isinstance(properties_dprop1[key], str) or
    #     np.isnan(properties_base[key]) or
    #     np.isnan(properties_dprop1[key])
    # )
    # else (properties_dprop1[key] - properties_base[key]) / delta_prop1
    # for key in properties_base
    # }

    # df_dprop2 = {
    # key: 0 if (
    #     properties_base[key] is None or
    #     properties_dprop2[key] is None or
    #     isinstance(properties_base[key], str) or
    #     isinstance(properties_dprop2[key], str) or
    #     np.isnan(properties_base[key]) or
    #     np.isnan(properties_dprop2[key])
    # )
    # else (properties_dprop2[key] - properties_base[key]) / delta_prop2
    # for key in properties_base
    # }

    properties_base = {
        key: 0. if (
            properties_base[key] is None or
            isinstance(properties_base[key], str) or
            isinstance(properties_base[key], bool) or
            np.isnan(properties_base[key])
        )
        else properties_base[key]
        for key in properties_base
    }

    properties_dprop_full = {
        key: 0. if (
            properties_dprop_full[key] is None or
            isinstance(properties_dprop_full[key], str) or
            isinstance(properties_dprop_full[key], bool) or
            np.isnan(properties_dprop_full[key])
        )
        else properties_dprop_full[key]
        for key in properties_dprop_full
    }

    


    # Compute JVP (directional derivative)
    # jvp = {
    #     key: df_dprop1[key] * prop1_dot + df_dprop2[key] * prop2_dot
    #     for key in properties_base
    # }

    jvp = {
        key: alpha*(properties_dprop_full[key]
                    - properties_base[key])
        for key in properties_base
    }

    # print("jvp", jvp)
    
    return properties_base, jvp

###### CUSTOM JVP TRIAL END #####

############### TESTING CODE START ################

# fluid_name = "CO2"


input_state = CP.HmassSmass_INPUTS
prop1 = 400980.0 
prop2 = 1991.94

stag_in = fluid.compute_reference_state(input_state, prop1, prop2)

jax.config.update("jax_traceback_filtering", "off")

grad_func = jax.jacrev(get_props_custom_jvp, argnums=(2, 3))
gradients = grad_func(fluid, input_state, prop1, prop2)
print(gradients)

###############     TESTING CODE  END     ###############