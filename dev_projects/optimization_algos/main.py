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
CONFIG_FILE = os.path.abspath("kofskey_constrained.yaml")
# CONFIG_FILE = os.path.abspath("kofskey1972_1stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if CASE == 0:
    problem = tf.CascadesOptimizationProblem(config)
    problem.fitness(problem.initial_guesses[0])
    tf.save_to_pickle(problem, "initial_guess")

elif CASE == 1:
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=False)
    # solver.problem.optimization_process.export_optimization_process(config)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tf.save_to_pickle(solver, "pickle_sga")

elif CASE == 2:
    operation_points = config["operation_points"]
    solver = tf.compute_optimal_turbine(config, export_results=False)
    tf.save_to_pickle(solver, filename = f"real_case_90p", out_dir = "output")

elif CASE == 3:

    operation_points = config["operation_points"]
    solvers = tf.compute_performance(operation_points, config, export_results=False)
    print(solvers[0].problem.results["cascade"]["centrifugal_stress"])

