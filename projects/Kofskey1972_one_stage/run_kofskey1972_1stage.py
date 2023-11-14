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

# Define running option
CASE = 1

# Load configuration file
CONFIG_FILE = "kofskey1972_1stage.yaml"
config = ml.read_configuration_file(CONFIG_FILE)

# ml.print_dict(config)

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = ml.compute_performance(operation_points, config, initial_guess = None, export_results=False)
    # solvers[0].print_convergence_history(savefile=True)

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_map"]
    # omega_frac = np.asarray([0.5, 0.7, 0.9, 1.0])
    omega_frac = np.asarray(1.00)
    operation_points["omega"] = operation_points["omega"]*omega_frac
    ml.compute_performance(operation_points, config)

elif CASE == 3:
    
    # Load experimental dataset
    data = pd.read_excel("./experimental_data_kofskey1972_1stage_interpolated.xlsx")
    pressure_ratio_exp = data["pressure_ratio_ts"].values
    speed_frac_exp = data["speed_percent"].values/100

    # Generate operating points with same conditions as dataset
    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):
        current_point = copy.deepcopy(design_point)
        current_point['p_out'] = design_point["p0_in"]/PR
        current_point['omega'] = design_point["omega"]*speed_frac
        operation_points.append(current_point)

    # Compute performance at experimental operating points   
    ml.compute_performance(operation_points, config)








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
    
