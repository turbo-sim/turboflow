
import turboflow as tf
import numpy as np
import CoolProp as cp


operating_point = {
    "fluid_name": "air",
    "T0_in": 288.15,
    "p0_in": 101325,
    "mass_flow_rate" : 0.3,
    "omega": 15000*2*np.pi/60,
    "alpha_in": 0,
}

impeller = {
  "radius_hub_in": 22.5e-3,
  "radius_tip_in": 47.65e-3,
  "radius_out" : 103.7e-3,
  "width_out" : 7.5e-3,
  "leading_edge_angle" : 45,
  "trailing_edge_angle" : 30,
  "number_of_blades" : 12, 
}

vaneless_diffuser = {
    "wall_divergence" : 0,
    "radius_out" : 185e-3,
    "radius_in" : 103.7e-3,
    "width_in" : 7.5e-3,
}

geometry = {
    "impeller" : impeller,
    "vaneless_diffuser" : vaneless_diffuser,
    }

initial_guess = {"v_in_impeller" : 0.001,
                 "w_out_impeller" : 0.2,
                 "beta_out_impeller" : -(30+90)/180,
                 "s_out_impeller" : 0.00,
                 "v_out_vaneless_diffuser" : 0.1,
                 "s_out_vaneless_diffuser" : 0.00,
                 "alpha_out_vaneless_diffuser" : (30+90)/180,
                 }

loss_model = {
    "impeller" :{
        "model" : "isentropic",
        "loss_coefficient" : "static_enthalpy_loss"
    },
    "vaneless_diffuser" :{
        "model" : "custom",
        "loss_coefficient" : "static_enthalpy_loss"
    },
}

# loss_model = {
#     "loss_model" : {
#         "model" : "oh",
#         "loss_coefficient" : "static_enthalpy_loss",
#     }
# }


simulation_options = {
    "slip_model" : "wiesner", 
    "loss_model" : loss_model,
    "Cf" : 0.002,
    "q_w" : 0.00,
    "vaneless_diffuser_model" : "algebraic",  
}

solver_options = {
    "method": "lm",  
    "tolerance": 1e-6,  
    "max_iterations": 100,  
    "derivative_method": "2-point",  
    "derivative_abs_step": 1e-6,  
    "plot_convergence": False,
}


solver, results = tf.centrifugal_compressor.compute_single_operation_point(
    operating_point,
    initial_guess,
    geometry,
    simulation_options,
    solver_options,
    export_results = False,
)

print(results["overall"]["PR_tt"])


