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
    cascade_problem = ml.CascadesNonlinearSystemProblem(cascades_data)
    solver = ml.solver.NonlinearSystemSolver(cascade_problem, cascade_problem.x0)
    solution = solver.solve(method='hybr')
    
    # Solve using optimization algorithm
    # cascade_problem = ml.CascadesOptimizationProblem(cascades_data)
    # solver = ml.solver.OptimizationSolver(cascade_problem, cascade_problem.x0, display=True, plot=False)
    # sol = solver.solve(method="trust-constr")
    
    # maps = ["m", "eta_ts"]
    # performance_map = ml.PerformanceMap()
    # performance_map.plot_omega_line(cascades_data, maps, N = 20, pr_limits = [1.8,3])
    # omega_list = np.array([0.9, 1])*cascades_data["BC"]["omega"]
    # figs = performance_map.plot_perfomance_map(cascades_data,maps,  omega_list, pr_limits = [1.5, 4], N = 40, method = 'hybr', ig = {'R' : 0.45, 'eta' : 0.9, 
                                                                                            # 'Ma_crit' : 0.92})
    # figs[0].savefig(f"Mass_flow_rate.png")