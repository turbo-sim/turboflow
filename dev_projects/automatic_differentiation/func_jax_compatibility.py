# %%
# All imports in this section

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad
import turboflow as tf
import pandas as pd
import os

from turboflow import math
from turboflow import utilities as utils
from turboflow.axial_turbine import loss_model as lm
from turboflow.axial_turbine import deviation_model as dm
from turboflow.axial_turbine import choking_criterion as cm


# %%
# All inputs 

# CONFIG_FILE = os.path.abspath("one-stage_config.yaml")
# config = tf.load_config(CONFIG_FILE, print_summary=False)

variables = {
    "v_in": 50.,
    "w_out_1": 150.,
    "s_out_1": 5000.,
    "beta_out_1": 10.,
    "v_crit_in": 200.,
    "beta_crit_throat_1": 15.,
    "w_crit_throat_1":400.,
    "s_crit_throat_1": 8000.,
    "w_out_2": 190.,
    "s_out_2": 6000.,
    "beta_out_2": 12.,
    "beta_crit_throat_2": 15.,
    "w_crit_throat_2":400.,
    "s_crit_throat_2": 8000.,
}

boundary_conditions = {
    "h0_in" : 100000.,
    "s_in" : 2000.,
    "omega" : 3000.,
    "alpha_in" : 10.,
    "p0_in": 500000.,
    "p_out": 100000.
}

geometry = {
    "radius_mean_in" : 5.,
    "radius_mean_out" : 3.,
    "radius_mean_throat" : 2.,
    "radius_tip_out" : 4.,
    "radius_hub_out" : 2.,
    "chord" : 10e-3,
    "A_in" : 20.,
    "A_out" : 10.,
    "A_throat" : 6.,
    "opening": 3e-3,
    "gauging_angle": 5.,
    "leading_edge_angle": 5.,
    "number_of_cascades": 1,
    "number_of_stages": 0,
    "interspace_area_ratio": 1
}

# geometry = {
#     "radius_mean_in" : [5., 6.],
#     "radius_mean_out" : [3., 4.],
#     "radius_mean_throat" : [2., 2.5],
#     "radius_tip_out" : [4., 3.],
#     "radius_hub_out" : [2., 1.5],
#     "chord" : [10e-3, 12e-3],
#     "A_in" : [20., 22.],
#     "A_out" : [10., 11.],
#     "A_throat" : [6., 7.],
#     "opening": [3e-3, 3.5e-3],
#     "leading_edge_angle": [5., 6.],
#     "number_of_cascades": 2,
#     "number_of_stages": 1,
#     "interspace_area_ratio": 1
# }

model_options = {
    "loss_model": {"model": "isentropic", "loss_coefficient" : "kinetic_energy"},
    "deviation_model": "aungier",
    "blockage_model": "flat_plate_turbulent",
    "choking_criterion": "critical_mach_number"
}

reference_values = {
    "mass_flow_ref" : 100.,
    "v0": 100.,
    "s_range": 5000.,
    "s_min": 500.,
    "angle_range": 20.,
    "angle_min": 2.,
    "h_out_s": 50000.,
    "d_out_s": 0.555
}

fluid = True

h0_exit = 10000.
v_m_exit = 10.
v_t_exit = 10.
rho_exit = 0.88
radius_exit = 5.
area_exit = 20.
blockage_exit = 0.
radius_inlet = 8. 
area_inlet = 50

cascade_inlet_input = {
    "h0" : 100000.,
    "s" : 2000.,
    "v" : 100.,
    "alpha" : 15.}

angular_speed = 3000.

cascade_exit_input = {
    "w" : jnp.array(149.43, dtype=float),
    "beta": jnp.array(-10.63, dtype=float),
    "s": jnp.array(2000., dtype=float),
    "rothalpy": jnp.array(-288224.0, dtype=float)
}

cascade_throat_input = {
    "w" : 149.43,
    "beta": -10.63,
    "s": 2000.0,
    "rothalpy": -288224.0
}

loss_model = {"model": "isentropic",
              "loss_coefficient" : "kinetic_energy"}

blockage = "flat_plate_turbulent"

choking_input = {
    "v_crit_in": 200.,
    "w_crit_throat": 400.,
    "s_crit_throat": 8000.,
    "beta_crit_throat": 15.
}

### For functions in flow_model 
# %%

# For evaluate_velocity_triangle_in function

U = 1.
v = jnp.sqrt(2)
alpha = 45.

triangle = tf.evaluate_velocity_triangle_in(U, v, alpha)

# tf.print_dict(triangle)

triangle_alpha = jax.jacfwd(tf.evaluate_velocity_triangle_in, argnums = 2)

der_triangle_alpha = triangle_alpha(U, v, alpha)

tf.print_dict(der_triangle_alpha)

# %%
# For evaluate_velocity_triangle_out function

U = 1.
w = jnp.sqrt(2)
beta = 45.0

vel_out = tf.evaluate_velocity_triangle_out(U, w, beta)

# tf.print_dict(vel_out)

der_vel_out_func = jax.jacfwd(tf.evaluate_velocity_triangle_out, argnums=1)

der_vel_out_beta = der_vel_out_func(U, w, beta)

tf.print_dict(der_vel_out_beta)

# %%
# For evaluate_cascade_interspace


# fluid =

properties = tf.evaluate_cascade_interspace(h0_exit, v_m_exit, v_t_exit, rho_exit, radius_exit, area_exit, blockage_exit, radius_inlet, area_inlet, True)
# print(properties)

der_properties_func = jax.jacfwd(tf.evaluate_cascade_interspace, argnums=1)

der_properties_vmexit = der_properties_func(h0_exit, v_m_exit, v_t_exit, rho_exit, radius_exit, area_exit, blockage_exit, radius_inlet, area_inlet, True)

print(der_properties_vmexit)

# %%
# For evaluate_cascade_inlet



# geometry = {
#     "radius_mean_in" : 5.,
#     "radius_mean_out" : 3.,
#     "radius_mean_throat" : 2.,
#     "chord" : 10e-3,
#     "A_in" : 20.,
#     "A_out" : 10.,
#     "A_throat" : 6.,
#     "opening": 3e-3,
#     "leading_edge_angle": 5.,
#     "number_of_cascades": 1,
#     "number_of_stages": 0,
#     "interspace_area_ratio": 1
# }

# geometry = {
#     "radius_mean_in" : [5., 6.],
#     "radius_mean_out" : [3., 4.],
#     "radius_mean_throat" : [2., 2.5],
#     "chord" : [10e-3, 12e-3],
#     "A_in" : [20., 22.],
#     "A_out" : [10., 11.],
#     "A_throat" : [6., 7.],
#     "opening": [3e-3, 3.5e-3],
#     "leading_edge_angle": [5., 6.],
#     "number_of_cascades": 2,
#     "number_of_stages": 1,
#     "interspace_area_ratio": 1
# }

geometry_cascade = {
            key: values[0]
            for key, values in geometry.items()
            if key not in ["number_of_cascades", "number_of_stages", "interspace_area_ratio"]
        }

plane = tf.evaluate_cascade_inlet(cascade_inlet_input, True, geometry, angular_speed)
tf.print_dict(plane)

# der_plane_func = jax.jacfwd(tf.evaluate_cascade_inlet, argnums=0)

# der_plane_cascade_inlet = der_plane_func(cascade_inlet_input, True, geometry, angular_speed)

# tf.print_dict(der_plane_cascade_inlet)

# %%
# For evaluate_cascade_exit

# cascade_exit_input = {
#     "w" : jnp.array(149.43, dtype=float),
#     "beta": jnp.array(-10.63, dtype=float),
#     "s": jnp.array(2000., dtype=float),
#     "rothalpy": jnp.array(-288224.0, dtype=float)
# }

# geometry = {
#     "radius_mean_out" : 3.,
#     "chord" : 10e-3,
#     "A_out" : 10.,
#     "opening": 3e-3,
#     "leading_edge_angle": 5.
# }

inlet_plane = tf.evaluate_cascade_inlet(cascade_inlet_input, True, geometry, angular_speed)

# loss_model = {"model": "isentropic",
#               "loss_coefficient" : "kinetic_energy"}

# blockage = "flat_plate_turbulent"

exit_plane = tf.evaluate_cascade_exit(cascade_exit_input, True, geometry, inlet_plane, angular_speed, blockage, loss_model)

print(exit_plane)

# der_exit_plane_func = jax.jacfwd(tf.evaluate_cascade_exit, argnums=0)

# der_exit_plane_cascade_exit = der_exit_plane_func(cascade_exit_input, True, geometry, inlet_plane, angular_speed, blockage, loss_model)

# print(der_exit_plane_cascade_exit)

# %%
# For evaluate_cascade_throat 

# cascade_throat_input = {
#     "w" : 149.43,
#     "beta": -10.63,
#     "s": 2000.0,
#     "rothalpy": -288224.0
# }

# geometry2 = {
#     "radius_mean_throat" : 2.,
#     "chord" : 10e-3,
#     "A_throat" : 6.,
#     "opening": 3e-3,
#     "leading_edge_angle": 5.
# }

throat_plane = tf.evaluate_cascade_throat(cascade_throat_input, True, geometry, inlet_plane, angular_speed, "flat_plate_turbulent", loss_model)

# tf.print_dict(throat_plane)

der_throat_plane_func = jax.jacfwd(tf.evaluate_cascade_throat, argnums=0)

der_throat_plane_cascade_throat = der_throat_plane_func(cascade_throat_input, True, geometry, inlet_plane, angular_speed, "flat_plate_turbulent", loss_model)

print(der_throat_plane_cascade_throat)

# %%
# For evaluate_cascade

# model_options = {
#     "loss_model": {"model": "isentropic", "loss_coefficient" : "kinetic_energy"},
#     "deviation_model": "aungier",
#     "blockage_model": "flat_plate_turbulent",
#     "choking_criterion": "critical_mach_number"
# }

# reference_values = {
#     "mass_flow_ref" : 100.,
#     "v0": 100.,
#     "s_range": 5000.,
#     "s_min": 500.,
#     "angle_range": 20.,
#     "angle_min": 2.
# }

# choking_input = {
#     "w_crit_throat": 400.,
#     "s_crit_throat": 8000.,
#     "beta_crit_throat": 15.
# }

# boundary_conditions = {
#     "h0_in" : 100000.,
#     "s_in" : 2000.,
#     "omega" : 3000.,
#     "alpha_in" : 10.
# }

fluid = True
# geometry_cascade = {
#             key: values[0]
#             for key, values in geometry.items()
#             if key not in ["number_of_cascades", "number_of_stages", "interspace_area_ratio"]
#         }

# residuals = tf.evaluate_cascade(cascade_inlet_input, cascade_exit_input, choking_input, True, geometry, angular_speed, results, model_options, reference_values)



residuals, inlet_plane, exit_plane, cascade_data = tf.evaluate_cascade(cascade_inlet_input, cascade_exit_input, choking_input, fluid, geometry_cascade, angular_speed, model_options, reference_values)

# tf.print_dict(residuals)

der_residuals_func = jax.jacfwd(tf.evaluate_cascade, argnums=5)

der_residuals_angular_speed = der_residuals_func(cascade_inlet_input, cascade_exit_input, choking_input, fluid, geometry_cascade, angular_speed, model_options, reference_values)

# tf.print_dict(der_residuals_angular_speed[1])

# autograd = jnp.asarray([v for k, v in der_residuals_angular_speed[1].items()])
for key in der_residuals_angular_speed[1]:
    autograd = der_residuals_angular_speed[1][key]

    def myfunc(x):

        residuals, inlet_plane, exit_plane, cascade_data = tf.evaluate_cascade(cascade_inlet_input, cascade_exit_input, choking_input, fluid, geometry_cascade, x, model_options, reference_values)

        # return jnp.asarray([v for k, v in inlet_plane.items()])
        return inlet_plane[key]

    eps = 1e-4
    grad = (myfunc(angular_speed + eps/2) - myfunc(angular_speed - eps/2))/eps

    print(key, autograd, grad)

# for x,y in zip(autograd, grad):
#     print(x, y)



# %%
# For evaluate_axial_turbine
# CONFIG_FILE = os.path.abspath("one-stage_config.yaml")
# config = tf.load_config(CONFIG_FILE, print_summary=False)

# print(config)



results = tf.evaluate_axial_turbine(
    variables,
    boundary_conditions,
    geometry,
    fluid,
    model_options,
    reference_values,
)

# tf.print_dict(results)

der_results_func = jax.jacfwd(tf.evaluate_axial_turbine, argnums=1)

der_results_variables = der_results_func(variables,
    boundary_conditions,
    geometry,
    fluid,
    model_options,
    reference_values)



der_results_variables.pop("geometry")
# tf.print_dict(der_results_variables)

for key in der_results_variables:
    autograd = der_results_variables[key]


    def myfunc(x):

        results = tf.evaluate_axial_turbine(variables, x, geometry, fluid, model_options, reference_values)
        results.pop("geometry")

        # return jnp.asarray([v for k, v in inlet_plane.items()])
        return results[key]

    eps = 1e-4
    # variables_eps_plus = {key: value + (eps/2) for key, value in variables.items()}
    # variables_eps_minus = {key: value - (eps/2) for key, value in variables.items()}
    # dict_eps_plus = myfunc(variables_eps_plus)
    # dict_eps_minus = myfunc(variables_eps_minus)

    for key1 in der_results_variables[key]:
        for key2 in der_results_variables[key][key1]:

            if key2 in boundary_conditions:
                variables_eps_plus = boundary_conditions.copy()
                variables_eps_plus[key2] = boundary_conditions[key2] + eps/2

                variables_eps_minus = boundary_conditions.copy()
                variables_eps_minus[key2] = boundary_conditions[key2] - eps/2
                
            dict_eps_plus = myfunc(variables_eps_plus)
            dict_eps_minus = myfunc(variables_eps_minus)

            grad = (dict_eps_plus[key1] - dict_eps_minus[key1])/eps
           
            print(key1, key2, autograd[key1][key2], grad)


# %%
# For compute_critical_values

critical_state = {}

x_crit = jnp.array(
        [
            choking_input["v_crit_in"],
            choking_input["w_crit_throat"],
            choking_input["s_crit_throat"],
        ]
    )



output = tf.compute_critical_values(
    x_crit,
    inlet_plane,
    fluid,
    geometry,
    angular_speed,
    critical_state,
    model_options,
    reference_values,
)

der_output_func = jax.jacfwd(tf.compute_critical_values, argnums=0)

der_output_x_crit = der_output_func(
    x_crit,
    inlet_plane,
    fluid,
    geometry,
    angular_speed,
    critical_state,
    model_options,
    reference_values,
)

print(der_output_x_crit)

# %%
# for critical_mass_flow_rate

residuals_critical, critical_state =  tf.critical_mass_flow_rate(choking_input, inlet_plane, exit_plane,fluid,geometry, angular_speed, model_options, reference_values,)

der_critical_residual = jax.jacrev(tf.critical_mass_flow_rate, argnums=0)
der_critical_residual_choking_input = der_critical_residual(choking_input, inlet_plane, exit_plane,fluid,geometry, angular_speed, model_options, reference_values,)

print(der_critical_residual_choking_input)
# %%
