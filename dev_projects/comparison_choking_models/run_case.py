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
# CONFIG_FILE = os.path.abspath("evaluate_cascade_critical_config.yaml")
CONFIG_FILE = os.path.abspath("evaluate_cascade_throat_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary = False)

x0 = {'w_out_1': 268.66481026324476, 
      's_out_1': 3788.608432483167, 
      'beta_out_1': 65.6898407185912, 
      'beta_crit_throat_1': 65.7701171445284, 
      'w_crit_throat_1': 276.83838095153004, 
      's_crit_throat_1': 3788.9055971964676, 
      'w_out_2': 252.89405313488805, 
      's_out_2': 3799.9125775849143, 
      'beta_out_2': -60.86345273381287, 
      'beta_crit_throat_2': -61.02291126941842, 
      'w_crit_throat_2': 263.17180199642655, 
      's_crit_throat_2': 3800.6679673588987, 
      'v_in': 80.63175359268403}

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        initial_guess=x0,
        export_results=True,
        stop_on_failure=True,
    )
    # print(solvers[0].problem.vars_real)

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, export_results=True, out_filename = 'evaluate_cascade_throat_results')