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

if __name__ == '__main__':
    
    # Solve using nonlinear equation solver
    # BC = cascades_data["BC"]
    # sol = ml.calculate.performance(BC,cascades_data, R = 0.3, eta_ts = 0.8, eta_tt = 0.9, Ma_crit = 0.95)
        
    # with pd.ExcelWriter('output.xlsx', engine='openpyxl', mode='a') as writer:
        # Add the new DataFrame to a new sheet
        # cascades_data["plane"].to_excel(writer, sheet_name='isentropic_new', index=False)

    
    # Solve using optimization algorithm
    # cascade_problem = ml.CascadesOptimizationProblem(cascades_data)
    # solver = ml.solver.OptimizationSolver(cascade_problem, cascade_problem.x0, display=True, plot=False)
    # sol = solver.solve(method="trust-constr")
    
    N = 10
    boundary_conditions = {key : val*np.ones(N) for key, val in cascades_data["BC"].items() if key != 'fluid_name'}
    boundary_conditions["fluid_name"] = N*[cascades_data["BC"]["fluid_name"]]
    pressure_ratio = np.linspace(2,2.5, N)
    p_out = cascades_data["BC"]["p0_in"]/pressure_ratio
    boundary_conditions["p_out"] = p_out
    
    # ml.calculate.performance_map(boundary_conditions, cascades_data)
    filename = 'Performance_data_2023-10-03_15-34-38.xlsx'
    
    x = 'p_out'
    y = ["Marel_0", "Marel_1", "Marel_2", "Marel_3", "Marel_4", "Marel_5"]
    xlabel = 'Exit static pressure'
    ylabel = 'Mach'
    x, lines = ml.calculate.plot_function(filename, x, y, xlabel = xlabel, ylabel = ylabel)    
    