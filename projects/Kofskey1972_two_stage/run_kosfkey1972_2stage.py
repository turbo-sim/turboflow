# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

@author: laboan
"""

import numpy as np
import pandas as pd

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml


filename = "Kofskey1972_2stage.yaml"
cascades_data = ml.read_configuration_file(filename)

Case = 4

if Case == 0:

    # Solve using nonlinear equation solver
    BC = cascades_data["BC"]
    sol = ml.calculate.compute_operating_point(BC,cascades_data, method = 'hybr', R = 0.3, eta_ts = 0.8, eta_tt = 0.9, Ma_crit = 0.95)
    

elif Case == 4:
    p_min = 2.4
    p_max = 5.0
    N = int((p_max-p_min)*10)+1
    pressure_ratio = np.linspace(p_min,p_max, N)
    p_out = cascades_data["BC"]["p0_in"]/pressure_ratio
    boundary_conditions = {key : val*np.ones(N) for key, val in cascades_data["BC"].items() if key != 'fluid_name'}
    boundary_conditions["fluid_name"] = N*[cascades_data["BC"]["fluid_name"]]
    boundary_conditions["p_out"] = p_out
    ml.calculate.performance_map(boundary_conditions, cascades_data)
