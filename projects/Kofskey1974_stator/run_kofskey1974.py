# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

@author: laboan
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import meanline_axial as ml

# Define running option
CASE = 2

# Load configuration file
CONFIG_FILE = "kofskey1974_stator.yaml"
config = ml.read_configuration_file(CONFIG_FILE)

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = ml.compute_performance(operation_points, config, initial_guess = None, export_results=False, stop_on_failure=True)
    print(solvers[0].problem.results["overall"]["mass_flow_rate"])   
    print(solvers[0].problem.results["cascade"]["Ma_crit_throat"])
    print(solvers[0].problem.results["cascade"]["Ma_crit_out"])
    print(solvers[0].problem.results["cascade"]["mass_flow_crit_throat"])
    print(solvers[0].problem.results["cascade"]["mass_flow_crit_error"])
    print(solvers[0].problem.results["cascade"]["Y_crit_throat"])
    print(solvers[0].problem.results["cascade"]["Y_crit_out"])
    print(solvers[0].problem.results["cascade"]["beta_crit_out"])
    print(solvers[0].problem.results["cascade"]["d_crit_throat"])
    print(solvers[0].problem.results["cascade"]["d_crit_out"])    
    # solvers[0].print_convergence_history(savefile=True)

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_map"]
    # omega_frac = np.asarray([0.5, 0.7, 0.9, 1.0])
    omega_frac = np.asarray(1.00)
    operation_points["omega"] = operation_points["omega"]*omega_frac
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

