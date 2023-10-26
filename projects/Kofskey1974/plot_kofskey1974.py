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

cascades_data = ml.get_cascades_data("kofskey1974.yaml")
design_point = cascades_data["BC"]

Case = 0

if Case == 0:

    filename = 'Performance_data_2023-10-26_11-41-12.xlsx'
    performance_data = ml.plot_functions.load_data(filename)
    
    # Plot mass flow rate at different angular speed
    subset = ["omega"] + list(np.array([0.7, 0.9, 1, 1.1])*design_point["omega"])
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'm', subset, xlabel = "Pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    # Plot total-to-static efficiency at different angular speed
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'eta_ts', subset, xlabel = "Pressure ratio", ylabel = "Total-to-static efficiency [%]", close_fig = False)
    
    # Plot mach at all planes at design angular speed
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'pr_ts', 'alpha_6', subset, xlabel = "Pressure ratio", ylabel = "Mach", close_fig = False)

    
    # Plot mass flow rate at different pressure ratios
    subset = ["pr_ts", 3.5]
    fig, ax = ml.plot_functions.plot_subsets(performance_data, 'omega', 'm', subset, xlabel = "Angular speed [rad/s]", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    
    # # Plot stacked losses at stator on subset
    # subset = ["omega", design_point["omega"]]
    # column_names = ["Y_p_2", "Y_te_2", "Y_inc_2", "Y_s_2"]
    # fig, ax = ml.plot_functions.stack_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Pressure ratio", ylabel = "Losses", title = "Rotor losses")
    
    # # Plot stacked losses at rotor on subset
    # subset = ["omega", design_point["omega"]]
    # column_names = ["Y_p_5", "Y_te_5", "Y_inc_5", "Y_s_5", "Y_cl_5"]
    # fig, ax = ml.plot_functions.stack_lines_on_subset(performance_data, 'pr_ts', column_names, subset, xlabel = "Pressure ratio", ylabel = "Losses", title = "Rotor losses")
    
elif Case == 1:
    
    filename = 'Performance_data_2023-10-26_11-21-57.xlsx'
    performance_data = ml.plot_functions.load_data(filename)
    
    lines = ["m_crit_2","m"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)

    lines = ["d_crit_2","d_5"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    
    lines = ["w_crit_2","w_5"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)

    lines = ["Ma_crit_2","Marel_5", "Marel_6"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)

    lines = ["beta_5", "beta_6"]
    fig1, ax1 = ml.plot_functions.plot_lines(performance_data, 'pr_ts', lines, xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)


