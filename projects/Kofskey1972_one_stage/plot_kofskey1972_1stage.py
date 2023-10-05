# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:27:39 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml

cascades_data = ml.get_cascades_data("kofskey1972_1stage.yaml")
design_point = cascades_data["BC"]
filename = 'Performance_data_2023-10-04_14-12-39.xlsx'


performance_data = ml.plot_functions.load_data(filename)

# Plot mass flow rate at different angular speed
subset = ["omega", 0.3*design_point["omega"], 0.5*design_point["omega"], 0.7*design_point["omega"], 0.9*design_point["omega"]]
fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'm', subset, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]")

# Plot total-to-static efficiency at different angular speed
fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'eta_ts', subset, xlabel = "Total-to-static pressure ratio", ylabel = "Total-to-static efficiency [%]")

# Plot mass flow rate at different pressure ratios
subset = ["pr_ts", 3.5]
fig, ax = ml.plot_functions.plot_subsets(performance_data, 'omega', 'm', subset, xlabel = "Angular speed [rad/s]", ylabel = "Mass flow rate [kg/s]")

# Plot mach at all planes at design angular speed
subset = ["omega", design_point["omega"]]
column_names = ["Marel_0", "Marel_1", "Marel_2", "Marel_3", "Marel_4", "Marel_5"]
fig, ax = ml.plot_functions.plot_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Total-to-static pressure ratio", ylabel = "Mach")

# Plot stacked losses at stator on subset
subset = ["omega", design_point["omega"]]
column_names = ["Y_p_2", "Y_te_2", "Y_inc_2", "Y_s_2"]
fig, ax = ml.plot_functions.stack_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Total-to-static pressure ratio", ylabel = "Losses", title = "Rotor losses")

# Plot stacked losses at rotor on subset
subset = ["omega", design_point["omega"]]
column_names = ["Y_p_5", "Y_te_5", "Y_inc_5", "Y_s_5", "Y_cl_5"]
fig, ax = ml.plot_functions.stack_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Total-to-static pressure ratio", ylabel = "Losses", title = "Rotor losses")



