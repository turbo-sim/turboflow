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

# Define mode
MODE = "performance_analysis"

# Load configuration file
CONFIG_FILE = os.path.abspath("two-stage_config.yaml")
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

