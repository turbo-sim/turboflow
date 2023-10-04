# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

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

filename = "kofskey1972_1stage.yaml"
cascades_data = ml.get_cascades_data(filename)

Case = 0

if Case == 0:

    # Solve using nonlinear equation solver
    BC = cascades_data["BC"]
    sol = ml.calculate.performance(BC,cascades_data, R = 0.3, eta_ts = 0.8, eta_tt = 0.9, Ma_crit = 0.95)
    
elif Case == 1:   
    # Solve using optimization algorithm
    cascade_problem = ml.CascadesOptimizationProblem(cascades_data)
    solver = ml.solver.OptimizationSolver(cascade_problem, cascade_problem.x0, display=True, plot=False)
    sol = solver.solve(method="trust-constr")

elif Case == 2:
    # Calculate a dataset corresponding to a dataset
    p_min = 1.6
    p_max = 4.5 
    speed_min = 0.3
    speed_max = 1.1
    
    N_pressure = int((p_max-p_min)*10)+1
    N_speed = int((speed_max-speed_min)*10)+1
    N = N_pressure*N_speed
    pressure_ratio = np.linspace(p_min,p_max, N_pressure)
    speed = np.linspace(speed_min,speed_max,N_speed)
    p_out = cascades_data["BC"]["p0_in"]/pressure_ratio
    p_out = result = np.concatenate([p_out if i % 2 == 0 else np.flip(p_out) for i in range(N_speed)])
    angular_speed = np.sort(np.repeat(speed*cascades_data["BC"]["omega"], N_pressure))
    boundary_conditions = {key : val*np.ones(N) for key, val in cascades_data["BC"].items() if key != 'fluid_name'}
    boundary_conditions["fluid_name"] = N*[cascades_data["BC"]["fluid_name"]]
    boundary_conditions["p_out"] = p_out
    boundary_conditions["omega"] = angular_speed
    
    ml.calculate.performance_map(boundary_conditions, cascades_data)
    
  
    