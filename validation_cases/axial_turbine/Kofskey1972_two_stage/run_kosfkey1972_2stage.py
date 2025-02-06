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

# Define running option
CASE = "peformance_map"

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1972_2stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if CASE == "peformance_map":

    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, 
                                     config, 
                                     out_filename="performance_map",
                                     export_results=True)

elif CASE == "experimental_points":

    # Load experimental dataset
    data = pd.read_excel("experimental_data/experimental_data_kofskey1972_2stage_raw.xlsx")
    data = data[data["speed_percent"].isin([110, 100, 90, 70])] 

    ## Generate operating points with same conditions as dataset
    pressure_ratio_exp = data["pressure_ratio_ts"]
    speed_frac_exp = data["speed_percent"].values / 100
    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):
        current_point = copy.deepcopy(design_point)
        current_point["p_out"] = design_point["p0_in"] / PR
        current_point["omega"] = design_point["omega"] * speed_frac
        operation_points.append(current_point)

    # Compute performance at experimental operating points
    tf.compute_performance(operation_points, 
                            config, 
                            out_filename="experimental_points",
                            export_results=True)