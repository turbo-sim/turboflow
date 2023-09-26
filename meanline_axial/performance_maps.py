# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:33:20 2023

@author: laboan
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from . import cascade_series as cs
from .non_linear_problem import CascadesNonlinearSystemProblem
from .solver import NonlinearSystemSolver


# Script with functions to generate performance maps

# def compute_performance_map(data_structure):
#     # Define pressure ratio
#     N = 40 # Number of points
#     PR_min = 1.5 # Mnimum pressure ratio
#     PR_max = 4 # Maximum pressure ratio
#     PR = np.linspace(1.5,4,N) 
    
#     # Define cases of percentages of design speed e.g. [0.7,0.9,1,1.1]
#     omega_org = data_structure["overall"]["omega"]
#     omega = np.array([0.7,0.9,1,1.1])
    
#     # Guess of degree of reaction (to get validation plots going from PR = 1.5, one value for each omega)
#     R_guess = np.array([0.2,0.4,0.4,0.4]) #Kacker and Okapuu
#     # R_guess = np.array([0.3,0.45,0.45,0.45]) #Benner

#     # Define array used for plotting
#     Ma2         = np.zeros(N)
#     Ma3         = np.zeros(N)
#     Ma5         = np.zeros(N)
#     Ma6         = np.zeros(N)
#     m           = np.zeros(N)
#     eta         = np.zeros(N)
#     Y_p_stator  = np.zeros(N)
#     Y_s_stator  = np.zeros(N)
#     Y_cl_stator = np.zeros(N)
#     Y_p_rotor   = np.zeros(N)
#     Y_s_rotor   = np.zeros(N)
#     Y_cl_rotor  = np.zeros(N)

#     # If true, uses solution of previous point as initial guess for current point
#     use_previous = True
#     x_real = None
        
#     # Checks and calculates number of stages
#     cs.number_stages(data_structure)
    
#     # Define figures for efficieny and mass flow rate (one curve for each omega)
#     fig1, ax1 = plt.subplots()
#     ax1.set_xlabel("Total-to-static pressure ratio")
#     ax1.set_ylabel("Mass flow rate")
#     fig2, ax2 = plt.subplots()
#     ax2.set_xlabel("Total-to-static pressure ratio")
#     ax2.set_ylabel("efficiency")
        
#     # Calculate mass flow rate vs pressure ratio curve for each omega
#     for j in range(len(omega)):
#         data_structure["overall"]["omega"] = omega[j]*omega_org
    
#         # Calculate mass flow rate vs pressure ratio for current omega 
#         for i in range(N):
#             data_structure["BC"]["p_out"] = data_structure["BC"]["p0_in"]/PR[i]
#             print("\n")
#             print(f"Pressure ratio: {PR[i]}")
#             print(f"Percent of design speed: {omega[j]}")
            
#             # Update fixed parameters for current case (h_out_s, v0 etc)
#             cs.update_fixed_params(data_structure)
            
#             # Initial guess
#             if use_previous and i > 0 and PR[i]!= 3:
#                 x0 = x_real
#             else:
#                 x0 = cs.generate_initial_guess(data_structure, R = R_guess[j])

            
#             x_scaled = cs.scale_x0(x0, data_structure) # Scale initial guess
#             sol, conv_hist = cs.cascade_series_analysis(data_structure, x_scaled) # Solve problem
#             x_real = cs.rescale_x0(sol.x, data_structure) # Calculate real solution
            
#             print(f"max residual: {np.max(abs(sol.fun))}")

#             # Store varibles for plotting
#             m[i] = data_structure["overall"]["m"]
#             Ma2[i] = data_structure["plane"]["Marel"].values[1]
#             Ma3[i] = data_structure["plane"]["Marel"].values[2]
#             Ma5[i] = data_structure["plane"]["Marel"].values[4]
#             Ma6[i] = data_structure["plane"]["Marel"].values[5]
#             eta[i] = data_structure["overall"]["eta_ts"]
#             Y_p_stator[i]  = data_structure["cascade"]["eta_drop_p"].values[0]
#             Y_s_stator[i]  = data_structure["cascade"]["eta_drop_s"].values[0]
#             Y_cl_stator[i] = data_structure["cascade"]["eta_drop_cl"].values[0]
#             Y_p_rotor[i]   = data_structure["cascade"]["eta_drop_p"].values[1]
#             Y_s_rotor[i]   = data_structure["cascade"]["eta_drop_s"].values[1]
#             Y_cl_rotor[i]  = data_structure["cascade"]["eta_drop_cl"].values[1]

# # Make function to compute performance map, and function to plot performance map
# # Make function to compute constant angular speed line and call it within the compute_performance_map
    
#     return x_real

    
class PerformanceMap:
    
    def __init__(self):
        
        self.m = None
        self.eta = None
    
    def get_omega_line(self, cascades_data, pr_limits = [2, 4], N = 40, method = 'hybr', ig = {'R' : 0.4, 'eta' : 0.9, 
                                                                                               'Ma_crit' : 0.92}):
        
        # Define pressure ratio array from given inputs
        pr = np.linspace(pr_limits[0], pr_limits[1],N)
        
        # Define arrays for storing omega_line
        m           = np.zeros(N)
        eta         = np.zeros(N)
        # Ma2         = np.zeros(N)
        # Ma3         = np.zeros(N)
        # Ma5         = np.zeros(N)
        # Ma6         = np.zeros(N)
        # Y_p_stator  = np.zeros(N)
        # Y_s_stator  = np.zeros(N)
        # Y_cl_stator = np.zeros(N)
        # Y_p_rotor   = np.zeros(N)
        # Y_s_rotor   = np.zeros(N)
        # Y_cl_rotor  = np.zeros(N)
        
        # If true, uses solution of previous point as initial guess for current point
        use_previous = True
        x0 = None
        
        for i in range(N):
            
            cascades_data["BC"]["p_out"] = cascades_data["BC"]["p0_in"]/pr[i]
            cascades_problem = CascadesNonlinearSystemProblem(cascades_data, x0, ig)
            solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
            solution = solver.solve(method=method)         
            if use_previous == True:
                x0 = cs.convert_scaled_x0(solution.x, cascades_data)
            
            # Store varibles for plotting
            m[i] = cascades_data["overall"]["m"]
            eta[i] = cascades_data["overall"]["eta_ts"]

            # Ma2[i] = cascades_data["plane"]["Marel"].values[1]
            # Ma3[i] = cascades_data["plane"]["Marel"].values[2]
            # Ma5[i] = cascades_data["plane"]["Marel"].values[4]
            # Ma6[i] = cascades_data["plane"]["Marel"].values[5]
            # Y_p_stator[i]  = cascades_data["cascade"]["eta_drop_p"].values[0]
            # Y_s_stator[i]  = cascades_data["cascade"]["eta_drop_s"].values[0]
            # Y_cl_stator[i] = cascades_data["cascade"]["eta_drop_cl"].values[0]
            # Y_p_rotor[i]   = cascades_data["cascade"]["eta_drop_p"].values[1]
            # Y_s_rotor[i]   = cascades_data["cascade"]["eta_drop_s"].values[1]
            # Y_cl_rotor[i]  = cascades_data["cascade"]["eta_drop_cl"].values[1]
            
        self.m = m
        self.pr = pr
        self.eta = eta
    
    def get_performance_map(self):
        # 
        pass
    
    def plot_omega_line(self, cascades_data, pr_limits = [2, 4], N = 40, method = 'hybr', ig = {'R' : 0.4, 'eta' : 0.9, 
                                                                                               'Ma_crit' : 0.92}):
        
        self.get_omega_line(cascades_data, pr_limits = pr_limits, N = N, method = method, ig = ig)
        
        fig, ax = plt.subplots()
        ax.plot(self.pr, self.m)
        plt.show
    
    def plot_perfomance_map(self):
        pass
    
        
        
        
