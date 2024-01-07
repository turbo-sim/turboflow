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
# borg
x0 = {'w_out_1': 292.2133961596657,
 's_out_1': 3906.8039609258676,
 'beta_out_1': 66.49296711887172,
 'v*_in_1': 81.05380363064901,
 'w*_throat_1': 316.9752175774574,
 's*_throat_1': 3907.6621322123183,
 'w*_out_1': 316.97518449944226,
 's*_out_1': 3907.662131086297,
 'w_out_2': 217.06420172836815,
 's_out_2': 3910.4988757144947,
 'beta_out_2': -53.546585557697206,
 'v*_in_2': 305.7188299530535,
 'w*_throat_2': 295.10961925560974,
 's*_throat_2': 3915.5101224463083,
 'w*_out_2': 271.00681404721223,
 's*_out_2': 3913.9374577082435,
 'v_in': 81.09772718643478}

x0 = {'w_out_1': 284.8403181437687, 
    's_out_1': 3906.5773009012155, 
    'beta_out_1': 66.68464777693927, 
    'v*_in_1': 81.07515491827422, 
    'w*_throat_1': 317.03028180818956, 
    's*_throat_1': 3907.5919369911985, 
    'w*_out_1': 317.0551163891715, 
    's*_out_1': 3907.664777632712, 
    'w_out_2': 223.4468224020445, 
    's_out_2': 3910.2596797662636, 
    'beta_out_2': -55.26890292859983, 
    'v*_in_2': 310.79033839298455,
    'w*_throat_2': 294.04042785685164, 
    's*_throat_2': 3915.589076558821, 
    'w*_out_2': 294.0442056342112, 
    's*_out_2': 3915.6008946851607, 
    'v_in': 80.01025976504931}

# aungier
# x0 = {'w_out_1': 238.167621204032,
#  's_out_1': 3905.0229545279544,
#  'beta_out_1': 66.05563007893358,
#  'v*_in_1': 81.05380362621658,
#  'w*_throat_1': 316.97521783193247,
#  's*_throat_1': 3907.662132235715,
#  'w*_out_1': 316.975217940696,
#  's*_out_1': 3907.662132224468,
#  'w_out_2': 163.32100154565825,
#  's_out_2': 3906.0412308684913,
#  'beta_out_2': -52.68248324558293,
#  'v*_in_2': 303.2007636100022,
#  'w*_throat_2': 295.6814568166116,
#  's*_throat_2': 3913.1517166576623,
#  'w*_out_2': 295.9556431535509,
#  's*_out_2': 3913.1655686142976,
#  'v_in': 76.84796077198823}

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = ml.compute_performance(operation_points, config, initial_guess = x0, export_results=False, stop_on_failure=True)

    # print(solvers[0].problem.results["overall"]["mass_flow_rate"])
    # print(solvers[0].problem.results["overall"]["PR_ts"])
    # print(solvers[0].problem.results["stage"]["reaction"])
    print(solvers[0].problem.results["plane"]["beta"][3:])
    print(solvers[0].problem.results["plane"]["s"][3:])    
    print(solvers[0].problem.results["plane"]["h0"][3:])
    print(solvers[0].problem.results["plane"]["d"][3:]) 
    print(solvers[0].problem.results["plane"]["v"][3:])
    print(solvers[0].problem.results["plane"]["h"][3:])    
    print(solvers[0].problem.results["plane"]["w"][3:])
    print(solvers[0].problem.results["plane"]["h0_rel"][3:])   
    print(solvers[0].problem.results["plane"]["w_m"][3:])   
    print(solvers[0].problem.results["plane"]["Ma_rel"][3:])   
    print(solvers[0].problem.results["cascade"]["Ma_crit_out"])
    print(solvers[0].problem.results["overall"]["mass_flow_rate"])
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
    data = pd.read_excel("./experimental_data_kofskey1972_1stage_interpolated.xlsx")
    pressure_ratio_exp = data["pressure_ratio_ts"].values
    speed_frac_exp = data["speed_percent"].values/100

    # Generate operating points with same conditions as dataset
    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):
        current_point = copy.deepcopy(design_point)
        current_point['p_out'] = design_point["p0_in"]/PR
        current_point['omega'] = design_point["omega"]*speed_frac
        operation_points.append(current_point)

    # Compute performance at experimental operating points   
    ml.compute_performance(operation_points, config)
