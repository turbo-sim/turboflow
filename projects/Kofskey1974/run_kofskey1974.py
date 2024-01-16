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
import meanline_axial as ml

# Define running option
CASE = 2

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1974.yaml")
config = ml.read_configuration_file(CONFIG_FILE)

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
      'w*_out_1': 316.9745088244802, 
      'beta*_out_1': 66.68474646782406, 
      's*_out_1': 3907.662129216508, 
      'w_out_2': 215.6943694051554, 
      's_out_2': 3910.5547628080762, 
      'beta_out_2': -53.38524787065875, 
      'v*_in_2': 309.451331757237,
     'w*_throat_2': 294.36179468837173, 
     's*_throat_2': 3915.6702302392605, 
     'w*_out_2': 294.3615408522683, 
     'beta*_out_2': -55.26921939501282, 
     's*_out_2': 3915.670216330829, 
     'v_in': 80.87654970689081}

# x0 = {'w_out_1': 293.74009056378264,
#       's_out_1': 3906.8595523426993, 
#       'beta_out_1': 66.5354196414861,
#       'v*_in_1': 81.05328920752547, 
#       'w*_throat_1': 316.975202887817, 
#       's*_throat_1': 3907.662152859119, 
#       'w*_out_1': 303.10229997861916, 
#       'beta*_out_1': 66.6285046603439, 
#       's*_out_1': 3907.184868529635, 
#       'w_out_2': 215.5400808725567, 
#       's_out_2': 3910.563356657865, 
#       'beta_out_2': -53.28033612758164, 
#       'v*_in_2': 308.97636394689323, 
#       'w*_throat_2': 294.4734855029855, 
#       's*_throat_2': 3915.6080863123752, 
#       'w*_out_2': 285.2554053047456, 
#       'beta*_out_2': -55.22268793295492, 
#       's*_out_2': 3915.10870534079, 
#       'v_in': 81.0249099321003}


# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = ml.compute_performance(operation_points, config, initial_guess = x0, export_results=False, stop_on_failure=True)
    print(solvers[0].problem.vars_real)
    # solvers[0].print_convergence_history(savefile=True)

elif CASE == 2:
    # Compute performance map according to config file
    operation_points = config["performance_map"]
    # omega_frac = np.asarray([0.5, 0.7, 0.9, 1.0])
    omega_frac = np.asarray(1.00)
    operation_points["omega"] = operation_points["omega"]*omega_frac
    ml.compute_performance(operation_points, config, initial_guess = x0)

elif CASE == 3:
    
    # Load experimental dataset
    data = pd.read_excel("./experimental_data_kofskey1974_interpolated.xlsx")
    pressure_ratio_exp = data["pressure_ratio_ts"].values
    speed_frac_exp = data["speed_percent"].values/100

    # Generate operating points with same conditions as dataset
    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):
        if not speed_frac in [0.3, 0.5]:
            current_point = copy.deepcopy(design_point)
            current_point['p_out'] = design_point["p0_in"]/PR
            current_point['omega'] = design_point["omega"]*speed_frac
            operation_points.append(current_point)

    # Compute performance at experimental operating points   
    ml.compute_performance(operation_points, config, initial_guess = x0)
