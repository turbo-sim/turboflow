# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import os
import meanline_axial as ml
import copy

# Define running option
CASE = 2
# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1972_2stage.yaml")
config = ml.read_configuration_file(CONFIG_FILE)

# Design point
x0 = {'w_out_1': 257.7509701606976, 's_out_1': 3830.142368506702, 'beta_out_1': 65.39002538567198, 'v*_in_1': 82.21064268443507, 'w*_throat_1': 310.97345105076045, 's*_throat_1': 3832.0729200938576,  
       'w_out_2': 244.99718072612208, 's_out_2': 3840.23142350161, 'beta_out_2': -61.38474399864242, 'v*_in_2': 262.15572755747144, 'w*_throat_2': 292.74957824239794, 's*_throat_2': 3843.8641284812047,  
       'w_out_3': 263.20397030823335, 's_out_3': 3846.382869681079, 'beta_out_3': 57.75171890938927, 'v*_in_3': 127.7526209617367, 'w*_throat_3': 285.50286313062793, 's*_throat_3': 3847.2637534242563, 
       'w_out_4': 243.9619014983766, 's_out_4': 3858.7766702991757, 'beta_out_4': -49.104357106378295, 'v*_in_4': 264.3932281060922, 'w*_throat_4': 269.72544872415375, 's*_throat_4': 3860.6155137814058, 'v_in': 80.8117628222287}

# Design angular speed, 2.2 pressure ratio
x0 = {'w_out_1': 238.02665198763506, 's_out_1': 3829.4595776741667, 'beta_out_1': 65.1195544434803, 'v*_in_1': 82.21064631807076, 'w*_throat_1': 310.9734513429997, 's*_throat_1': 3832.072920095844, 
      'w_out_2': 203.87858281638486, 's_out_2': 3836.2808254899523, 'beta_out_2': -60.55021086208545, 'v*_in_2': 259.31397475310484, 'w*_throat_2': 293.15155087968225, 's*_throat_2': 3842.987800276261,
      'w_out_3': 175.93452435595785, 's_out_3': 3839.5438989099625, 'beta_out_3': 55.803499783280444, 'v*_in_3': 118.55238676156732, 'w*_throat_3': 289.64926475184683, 's*_throat_3': 3844.234328110235, 
      'w_out_4': 129.5548909794999, 's_out_4': 3841.634128134699, 'beta_out_4': -45.863824120632536, 'v*_in_4': 256.71307578484897, 'w*_throat_4': 277.29174380999876, 's*_throat_4': 3851.9995946217514,  'v_in': 78.98337352389863}


if CASE == 0:

    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = ml.compute_performance(operation_points, config, initial_guess = x0, export_results=False, stop_on_failure=True)
    print(solvers[0].problem.results["cascade"]["Ma_crit_throat"])
    # print(solvers[0].problem.results["cascade"]["Ma_crit_out"])
    # print(solvers[0].problem.results["cascade"]["mass_flow_crit_error"])
    print(solvers[0].problem.geometry["gauging_angle"])
    # print(solvers[0].problem.vars_real)
    

elif CASE == 1:
    # Compute performance map according to config file
    operation_points = config["performance_map"]
    # omega_frac = np.asarray(1.00)
    # operation_points["omega"] = operation_points["omega"]*omega_frac
    solvers = ml.compute_performance(operation_points, config, initial_guess = x0)

elif CASE == 2:

    # Load experimental dataset
    data = pd.read_excel("./experimental_data_kofskey1972_2stage_raw.xlsx")
    data = data[data["speed_percent"].isin([110, 100, 90, 70])]
    # data = data[data["speed_percent"].isin([0])]

    print(data)

    pressure_ratio_exp = data["pressure_ratio_ts"]
    speed_frac_exp = data["speed_percent"].values/100

    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):            
        current_point = copy.deepcopy(design_point)
        current_point['p_out'] = design_point["p0_in"]/PR
        current_point['omega'] = design_point["omega"]*speed_frac
        operation_points.append(current_point)

    # Compute performance at experimental operating points   
    ml.compute_performance(operation_points, config, initial_guess = x0)