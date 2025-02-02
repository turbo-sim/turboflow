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

# Define running case
CASE = "experimental_points"

# Load configuration file
CONFIG_FILE = os.path.abspath("Kofskey1974.yaml")
config = tf.load_config(CONFIG_FILE)

if CASE == "performance_map":
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    tf.compute_performance(operation_points, 
                            config, 
                            out_filename="performance_map",
                            export_results=True)

elif CASE == "experimental_points":

    # Load experimental dataset
    data = pd.read_excel(
        "experimental_data/experimental_data_kofskey1974_raw.xlsx", sheet_name="Interpolated pr_ts"
    )
    data = data[~data["pressure_ratio_ts_interp"].isna()]
    pressure_ratio_exp = data[data["speed_percent"].isin([105, 100, 90, 70])][
        "pressure_ratio_ts_interp"
    ].values
    speed_frac_exp = (
        data[data["speed_percent"].isin([105, 100, 90, 70])]["speed_percent"].values
        / 100
    )

    # Generate operating points with same conditions as dataset
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

