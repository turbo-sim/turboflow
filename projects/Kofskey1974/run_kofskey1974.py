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

# ml.print_dict(config)

# x0 = {'w_out_1': 292.75732924292544, 
#       's_out_1': 3906.8301664035716, 
#       'beta_out_1': 66.56929660739203, 
#       'v*_in_1': 81.05328920719423, 
#       'w*_throat_1': 316.9752020941576, 
#       's*_throat_1': 3907.6621528324263, 
#       'w*_out_1': 314.7785100385781, 
#       'beta*_out_1': 66.68318694304452, 
#       's*_out_1': 3907.58720497623, 
#       'w_out_2': 216.39698251484728, 
#       's_out_2': 3910.52550561793, 
#       'beta_out_2': -53.532811170977446, 
#       'v*_in_2': 309.3855796783943, 
#       'w*_throat_2': 294.37748650567227, 
#       's*_throat_2': 3915.6362223186775, 
#       'w*_out_2': 285.3733617222365,
#       'beta*_out_2': -55.22476996918425, 
#       's*_out_2': 3915.1452936545506, 
#       'v_in': 80.85998313469258}

x0 = {'w_out_1': 293.49185257504075, 
      's_out_1': 3906.855371623767, 
      'beta_out_1': 66.57475889317183, 
      'v*_in_1': 81.05328919923672, 
      'w*_throat_1': 316.97520064515345, 
      's*_throat_1': 3907.6621527854477, 
    #   'w*_out_1': 316.9745088244802, 
    #   'beta*_out_1': 66.68474646782406, 
    #   's*_out_1': 3907.662129216508, 
      'w_out_2': 215.6943694051554, 
      's_out_2': 3910.5547628080762, 
      'beta_out_2': -53.38524787065875, 
      'v*_in_2': 309.451331757237,
     'w*_throat_2': 294.36179468837173, 
     's*_throat_2': 3915.6702302392605, 
    #  'w*_out_2': 294.3615408522683, 
    #  'beta*_out_2': -55.26921939501282, 
    #  's*_out_2': 3915.670216330829, 
     'v_in': 80.87654970689081}



# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(operation_points, config, initial_guess = x0, export_results=False, stop_on_failure=True)

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    omega_frac = np.asarray(1.00)
    operation_points["omega"] = operation_points["omega"]*omega_frac
    tf.compute_performance(operation_points, config, initial_guess = x0)

elif CASE == 3:
    
    # Load experimental dataset
    data = pd.read_excel("./experimental_data_kofskey1974_raw.xlsx", sheet_name = "Interpolated pr_ts")
    data = data[~data["pressure_ratio_ts_interp"].isna()]
    pressure_ratio_exp = data[data["speed_percent"].isin([105, 100, 90, 70])]["pressure_ratio_ts_interp"].values
    speed_frac_exp = data[data["speed_percent"].isin([105, 100, 90, 70])]["speed_percent"].values/100

    # Generate operating points with same conditions as dataset
    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):
        current_point = copy.deepcopy(design_point)
        current_point['p_out'] = design_point["p0_in"]/PR
        current_point['omega'] = design_point["omega"]*speed_frac
        operation_points.append(current_point)

    # Compute performance at experimental operating points   
    tf.compute_performance(operation_points, config, initial_guess = x0)