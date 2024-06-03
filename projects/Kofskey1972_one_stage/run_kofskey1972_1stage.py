import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turboflow as tf

# Define running option
CASE = 10

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1972_1stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=True)

x0 = {
    "w_out_1": 244.42314206706558,
    "s_out_1": 3787.6640308614674,
    "beta_out_1": 64.43917617780212,
    #   'v*_in_1': 84.77882078058266,
    "w*_throat_1": 309.3325769414344,
    "s*_throat_1": 3789.968019275075,
    "w_out_2": 240.72352688455877,
    "s_out_2": 3797.034102749199,
    "beta_out_2": -61.497567096277514,
    #   'v*_in_2': 250.04121537545407,
    "w*_throat_2": 292.4919351280035,
    "s*_throat_2": 3800.736468314427,
    "v_in": 81.88511236557443,
}

x0 = {
    "w_out_1": 267.78187967067004,
    "s_out_1": 3788.588319333078,
    "beta_out_1": 65.88272377225763,
    #   'beta*_throat_1': 65.8827237722576,
    "w*_throat_1": 253.29023745799333,
    "w_out_2": 253.1073841704115,
    "s_out_2": 3799.9882108125557,
    "beta_out_2": -61.15576777799674,
    #    'beta*_throat_2': -61.15576777799674,
    "w*_throat_2": 229.04793360175023,
    "v_in": 79.91497102905909,
}

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    print(type(operation_points))
    solvers = tf.compute_performance(
        operation_points,
        config,
        initial_guess=None,
        export_results=False,
        stop_on_failure=True,
    )

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    omega_frac = np.asarray(1.00)
    operation_points["omega"] = operation_points["omega"]*omega_frac
    solvers = tf.compute_performance(operation_points, config, export_results=True)


elif CASE == 3:

    # Load experimental dataset
    sheets = ["Mass flow rate", "Torque", "Total-to-static efficiency", "alpha_out"]
    data = pd.read_excel(
        "./experimental_data_kofskey1972_1stage_raw.xlsx", sheet_name=sheets
    )

    pressure_ratio_exp = []
    speed_frac_exp = []
    for sheet in sheets:
        pressure_ratio_exp += list(data[sheet]["PR"].values)
        speed_frac_exp += list(data[sheet]["omega"].values / 100)

    pressure_ratio_exp = np.array(pressure_ratio_exp)
    speed_frac_exp = np.array(speed_frac_exp)

    # Generate operating points with same conditions as dataset
    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):
        if not speed_frac in [
            0.3,
            0.5,
        ]:  # 30 and 50% desing speed not included in validation plot
            current_point = copy.deepcopy(design_point)
            current_point["p_out"] = design_point["p0_in"] / PR
            current_point["omega"] = design_point["omega"] * speed_frac
            operation_points.append(current_point)

    # Compute performance at experimental operating points
    tf.compute_performance(operation_points, config)

elif CASE == 4:
    operation_points = config["operation_points"]

    # solvers = ml.compute_performance(operation_points, config, initial_guess = None, export_results=None, stop_on_failure=True)

    solvers = tf.compute_optimal_turbine(config, export_results=True)
    fig, ax = tf.plot_functions.plot_axial_radial_plane(solvers.problem.geometry)

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
# TODO improve geometry processing
# TODO merge optimization and root finding problems for performance analysis
