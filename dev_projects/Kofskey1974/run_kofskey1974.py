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
CASE = 2

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1974.yaml")
config = tf.load_config(CONFIG_FILE)


x0 = {'w_out_1': 302.34152618268644, 's_out_1': 3907.4582642372093, 'beta_out_1': 68.2353652893051, 'w_crit_throat_1': 305.3268020186889, 's_crit_throat_1': 3907.5663389138763, 'w_out_2': 202.5739869228336, 's_out_2': 3914.239812962878, 'beta_out_2': -53.39832785308828, 'w_crit_throat_2': 348.2994050296265, 's_crit_throat_2': 3931.0657896251896, 'v_in': 75.3926819503091}


# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        # initial_guess=x0,
        export_results=False,
        stop_on_failure=True,
    )

    print(solvers[0].problem.results["overall"]["mass_flow_rate"])
    print(solvers[0].problem.results["overall"]["efficiency_ts"])
    print(solvers[0].problem.vars_real)

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    tf.compute_performance(operation_points, config, initial_guess=None)

elif CASE == 3:

    # Load experimental dataset
    data = pd.read_excel(
        "./experimental_data_kofskey1974_raw.xlsx", sheet_name="Interpolated pr_ts"
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
    tf.compute_performance(operation_points, config, initial_guess=x0)
