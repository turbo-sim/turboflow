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
import turboflow as tf

# Define running option
CASE = 1

# Load configuration file
CONFIG_FILE = "kofskey1974_stator.yaml"
config = tf.load_config(CONFIG_FILE)

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
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
    operation_points["omega"] = operation_points["omega"] * omega_frac
    tf.compute_performance(operation_points, config)


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
