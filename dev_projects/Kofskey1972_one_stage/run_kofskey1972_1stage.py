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
config = tf.load_config(CONFIG_FILE, print_summary=False)

# x0 = {'w_out_1': 267.8886994459008, 's_out_1': 3788.592107578678, 'beta_out_1': 65.8827237722576, 'beta_crit_throat_1': 65.8827237722576, 'w_crit_throat_1': 343.9780120412917, 's_crit_throat_1': 3793.347935647536, 'w_out_2': 253.11612518792936, 's_out_2': 3799.910848529705, 'beta_out_2': -61.15576777799674, 'beta_crit_throat_2': -61.15576777799674, 'w_crit_throat_2': 253.11612518658842, 's_crit_throat_2': 3799.910848529209, 'v_in': 79.92441876174165}
x0 = {'w_out_1': 269.6435093424158, 's_out_1': 3788.641302695732, 'beta_out_1': 65.65842671865852, 'w_crit_throat_1': 279.71286825479615, 's_crit_throat_1': 3789.0143623778667, 'w_out_2': 252.31227942324745, 's_out_2': 3799.953046296435, 'beta_out_2': -60.71759067762759, 'w_crit_throat_2': 269.1791334389948, 's_crit_throat_2': 3801.2010069338785, 'v_in': 80.82061185428421}
# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        # initial_guess=None,
        export_results=False,
        stop_on_failure=True,
    )

    print(solvers[0].problem.results["overall"]["mass_flow_rate"])
    print(solvers[0].problem.results["overall"]["efficiency_ts"])
    print(solvers[0].problem.vars_real)

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, 
                                     config, 
                                    #  initial_guess = None, 
                                     export_results=False)

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
    solvers = tf.compute_optimal_turbine(config, export_results=True)
    # fig, ax = tf.plot_functions.plot_axial_radial_plane(solvers.problem.geometry)

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
