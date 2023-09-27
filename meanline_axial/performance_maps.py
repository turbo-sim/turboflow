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
from .plot_options import set_plot_options

set_plot_options()
    
class PerformanceMap:
    
    def __init__(self):
        
        self.m = None
        self.eta = None
    
    def get_omega_line(self, cascades_data, pr_limits, N, method, ig):
        
        # Define pressure ratio array from given inputs
        pr = np.linspace(pr_limits[0], pr_limits[1],N)
        
        # Define arrays for storing omega_line
        m             = np.zeros(N)
        eta_ts       = np.zeros(N)
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
        use_previous = False
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
            eta_ts[i] = cascades_data["overall"]["eta_ts"]

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
        self.eta_ts = eta_ts
    
    def get_performance_map(self, cascades_data, omega_list, pr_limits, N, method, ig):
        
        # Ensure there is one value in ig for eahc omega
        for key, value in ig.items():
            if isinstance(value, (int, float)):  # Check if value is a scalar
                ig[key] = np.ones(len(omega_list))*ig[key]
            elif len(value) != len(omega_list):
                ig[key] = np.ones(len(omega_list))*ig[key][0]
        
        # Calculate omega line for each omega in a list 

        for i in range(len(omega_list)):
            cascades_data["BC"]["omega"] = omega_list[i]
            
            ig_omega = {key: value[i] for key, value in ig.items()}
            self.get_omega_line(cascades_data, pr_limits, N, method, ig_omega)
            
            if i == 0:
                m_array = self.m
                eta_ts_array = self.eta_ts
            else:
                m_array = np.vstack((m_array, self.m))
                eta_ts_array = np.vstack((eta_ts_array, self.eta_ts))
                
        self.m = m_array
        self.eta_ts = eta_ts_array    
    
    def plot_omega_line(self, cascades_data, maps, pr_limits = [2, 4], N = 40, method = 'hybr', ig = {'R' : 0.4, 'eta' : 0.9, 
                                                                                                          'Ma_crit' : 0.92}):
        
        self.get_omega_line(cascades_data, pr_limits = pr_limits, N = N, method = method, ig = ig)

        figs = {}
        if any(element == 'm' for element in maps):
            fig1, ax1 = plt.subplots()
            ax1.plot(self.pr, self.m)
            ax1.set_xlabel('Total-to-static pressure ratio')
            ax1.set_ylabel('Mass flow rate')
            figs["m"] = [fig1, ax1]
            plt.close(fig1)
            
        if any(element == 'eta_ts' for element in maps):
            fig2, ax2 = plt.subplots()
            ax2.plot(self.pr, self.eta_ts)
            ax2.set_xlabel('Total-to-static pressure ratio')
            ax2.set_ylabel('Total to static efficiency')
            figs["eta_ts"] = [fig2, ax2]
            plt.close(fig2)

        
        return figs
    
    def plot_perfomance_map(self, cascades_data, maps, omega_list, pr_limits = [2, 4], N = 40, method = 'hybr', ig = {'R' : 0.4, 'eta' : 0.9, 
                                                                                                          'Ma_crit' : 0.92}):
        self.get_performance_map(cascades_data, omega_list, pr_limits, N, method, ig)
        figs = {}
        # Plot omega lines for each omega
        if any(element == 'm' for element in maps):
            fig1, ax1 = plt.subplots()
            for i in range(len(omega_list)):
                ax1.plot(self.pr, self.m[i], label = str(int(omega_list[i])))
            ax1.set_xlabel('Total-to-static pressure ratio')
            ax1.set_ylabel('Mass flow rate')
            figs["m"] = [fig1, ax1]
            ax1.legend()
            plt.close(fig1)

        if any(element == 'eta_ts' for element in maps):
            fig2, ax2 = plt.subplots()
            for i in range(len(omega_list)):
                ax2.plot(self.pr, self.eta_ts[i], label = str(int(omega_list[i])))
            ax2.set_xlabel('Total-to-static pressure ratio')
            ax2.set_ylabel('Total to static efficiency')
            ax2.legend()
            figs["eta_ts"] = [fig2, ax2]
            plt.close(fig2)

        
        return figs
        
