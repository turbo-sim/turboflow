import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml


CONFIG_FILE = "kofskey1972_1stage.yaml"
cascades_data = ml.read_configuration_file(CONFIG_FILE)

Case = 3





def validate_geometry_config(geometry_config):
    required_keys = {
        'n_cascades', 's', 'c', 'b', 'H', 't_max', 'o', 'We', 'le', 'te',
        'xi', 'theta_in', 'theta_out', 't_cl', 'radius', 'r_in', 'r_out',
        'r_ht_in', 'A_in', 'A_out'
    }
    
    # Check for required fields
    missing_keys = required_keys - geometry_config.keys()
    if missing_keys:
        return f"Missing geometry configuration keys: {missing_keys}", False
    
    # Check for extra fields
    extra_keys = geometry_config.keys() - required_keys
    if extra_keys:
        return f"Extra geometry configuration keys: {extra_keys}", False

    # If all checks pass
    return "Geometry configuration is valid.", True

# Example usage:
# Assuming 'config' is your dictionary containing the YAML data
geometry_validation_result, is_valid = validate_geometry_config(cascades_data["geometry"])
if is_valid:
    print(geometry_validation_result)
else:
    print(geometry_validation_result)






# if Case == 0:
#     # Solve using nonlinear equation solver
#     print(cascades_data)
#     operating_point = cascades_data["operation_points"][0]
#     solver = ml.compute_operating_point(operating_point, cascades_data)
#     solver.plot_convergence_history()

# elif Case == 1:
#     # Solve using optimization algorithm
#     cascade_problem = ml.CascadesOptimizationProblem(cascades_data)
#     solver = ml.solver.OptimizationSolver(
#         cascade_problem, cascade_problem.x0, display=True, plot=False
#     )
#     solver = solver.solve(method="trust-constr")


# elif Case == 2:
#     # Compute performance map according to config file
#     operation_points = cascades_data["operation_points"]
#     ml.compute_performance(operation_points, cascades_data)


# elif Case == 3:
#     # Compute performance map according to config file
#     operation_points = cascades_data["performance_map"]
#     omega_frac = np.asarray([0.5, 0.7, 0.9, 1.0])
#     operation_points["omega"] = operation_points["omega"]*omega_frac
#     ml.compute_performance(operation_points, cascades_data)



# elif Case == 4:
#     filename = "Full_Dataset_Kofskey1972_1stage.xlsx"
#     exp_data = pd.read_excel(
#         filename,
#         sheet_name=[
#             "Mass flow rate",
#             "Torque",
#             "Total-to-static efficiency",
#             "Beta_out",
#         ],
#     )

#     p0_in = cascades_data["BC"]["p0_in"]
#     omega_des = cascades_data["BC"]["omega"]

#     p_out = []
#     omega = []

#     filenames = [
#         "mass_flow_rate.xlsx",
#         "torque.xlsx",
#         "total-to-static_efficiency.xlsx",
#         "beta_out.xlsx",
#     ]
#     cascades_data_org = cascades_data.copy()
#     i = 0
#     for key in exp_data.keys():
#         p_out = p0_in / exp_data[key]["PR"]
#         omega = exp_data[key]["omega"] / 100 * omega_des

#         if len(p_out) != len(omega):
#             raise Exception("PR and omega have different dimensions")

#         N = len(p_out)
#         boundary_conditions = {
#             key: val * np.ones(N)
#             for key, val in cascades_data["BC"].items()
#             if key != "fluid_name"
#         }
#         boundary_conditions["fluid_name"] = N * [cascades_data["BC"]["fluid_name"]]
#         boundary_conditions["p_out"] = p_out.values
#         boundary_conditions["omega"] = omega.values

#         ml.calculate.performance_map(
#             boundary_conditions, cascades_data, filename=filenames[i]
#         )

#         cascades_data = cascades_data_org.copy()

#         i += 1



# Show plots
# plt.show()

    # DONE add option to give operation points as list of lists to define several speed lines
    # DONE add option to define range of values for all the parameters of the operating point, including T0_in, p0_in and alpha_in
    # DONE all variables should be ranged to create a nested list of lists
    # DONE the variables should work if they are scalars as well
    # DONE implemented closest-point strategy for initial guess of performance map
    # DONE implement two norm of relative deviation as metric

    # TODO update plotting so the different lines are plotted separately
    # TODO seggregate solver from initial guess in the single point evaluation
    
