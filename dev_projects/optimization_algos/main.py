import os
import sys
import copy
import numpy as np
import pandas as pd
import datetime
import yaml
import matplotlib.pyplot as plt
import turboflow as tf

# Define running option
CASE = 2

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1972_1stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if CASE == 1:
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=False)
    # solver.problem.optimization_process.export_optimization_process(config)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tf.save_to_pickle(solver, "pickle_sga")

elif CASE == 2:
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=False)
    # solver.problem.optimization_process.export_optimization_process(config)
    # tf.save_to_pickle(solver, filename = f"pickle_2stage_{config['design_optimization']['solver_options']['method']}", out_dir = "output")

elif CASE == 3:

    filename_slsqp = "pickle_file_test_slsqp.pkl"

    performance_map = config["performance_analysis"]["performance_map"]
    solver_options = config["performance_analysis"]["solver_options"]
    ig = config["performance_analysis"]["initial_guess"]

    config = tf.build_config(filename_slsqp, performance_map, solver_options, ig)
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        export_results=True,
        stop_on_failure=True,
    )


