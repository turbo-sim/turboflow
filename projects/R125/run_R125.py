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
CASE = 4

# Load configuration file
CONFIG_FILE = os.path.abspath("R125.yaml")
config = tf.load_config(CONFIG_FILE)

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        initial_guess=None,
        export_results=None,
        stop_on_failure=True,
    )


elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    omega_frac = np.asarray(1.00)
    operation_points["omega"] = operation_points["omega"] * omega_frac
    solvers = tf.compute_performance(operation_points, config, initial_guess=x0)

elif CASE == 3:
    solvers = tf.compute_optimal_turbine(config)
