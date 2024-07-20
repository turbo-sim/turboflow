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
CASE = 3

# Load configuration file
CONFIG_FILE = os.path.abspath("design_case.yaml")
config = tf.load_config(CONFIG_FILE, print_summary = False)

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

    export_results = False
    save_figs = False

    solver = tf.compute_optimal_turbine(config, export_results=export_results, out_filename='design_case')
    print(solver.problem.results["geometry"]["flaring_angle"])
    print(solver.problem.c_eq)
    fig1, ax1 = tf.plot_velocity_triangles(solver.problem.results["plane"])
    fig2, ax2 = tf.plot_axial_radial_plane(solver.problem.geometry)

    if save_figs:

        formats = [".png",".eps"]
        folder = 'figures'
        filename1 = 'velocity_triangles'
        filename2 = 'geometry'
        tf.savefig_in_formats(fig1, os.path.join(folder, filename1), formats = formats)
        tf.savefig_in_formats(fig2, os.path.join(folder, filename2), formats = formats)

