# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import os
import turboflow as tf
import copy

import jax
import jax.numpy as jnp

import dill

# Define mode
# MODE = "performance_analysis"
MODE = "design_optimization"

# Load configuration file
CONFIG_FILE = os.path.abspath("two_stage_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if MODE == "performance_analysis":

    # Compute performance map according to config file
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
    solvers = tf.compute_performance(operation_points, config, initial_guess=None)

elif MODE == "design_optimization":

    # Compute optimal turbine

    operation_points = config["operation_points"]
    # fitness_func, x, x_keys, grad_jax, grad_FD = tf.fitness_gradient(config)

    solver = tf.compute_optimal_turbine(config, export_results=True)

    file_path = os.path.join('output', f"2stage_trial_kofs_FD_1e-3_tol1e-8_{config['design_optimization']['solver_options']['method']}.pkl")
    # Open a file in write-binary mode
    with open(file_path, 'wb') as file:
        # Serialize the object and write it to the file
        dill.dump(solver, file)


    # solver.plot_convergence_history()
    # solver.print_convergence_history()
    # fig, ax = tf.plot_functions.plot_axial_radial_plane(solver.problem.geometry)
    # fig, ax = tf.plot_functions.plot_velocity_triangle(solver.problem.results["planes"])

