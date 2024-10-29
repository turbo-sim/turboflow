import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turboflow as tf

# Define mode 
# MODE = "design_optimzation"
MODE = "performance_analysis"

# Load configuration file
CONFIG_FILE = os.path.abspath("one-stage_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if MODE == "performance_analysis":
    
    # Compute performance at operation point(s) according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        export_results=False,
        stop_on_failure=True,
    )

elif MODE == "performance_map":

    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, export_results=True)


elif MODE == "design_optimzation":

    # Compute optimal turbine
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=True)
    fig, ax = tf.plot_functions.plot_axial_radial_plane(solver.problem.geometry)
    fig, ax = tf.plot_functions.plot_velocity_triangles(solver.problem.results["plane"])
