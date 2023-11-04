# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:12:16 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml

cascades_data = ml.read_configuration_file("Kofskey1972_2stage.yaml")
design_point = cascades_data["BC"]

Case = 1

if Case == 1:
    
    filename = 'Performance_data_2023-10-26_09-23-01.xlsx'
    performance_data = ml.plot_functions.load_data(filename)
    
    fig1, ax1 = ml.plot_functions.plot_line(performance_data, 'pr_ts', 'm', xlabel = "Total-to-static pressure ratio", ylabel = "Mass flow rate [kg/s]", close_fig = False)
    