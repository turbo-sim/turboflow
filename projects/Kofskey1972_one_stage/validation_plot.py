# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:26:19 2023

@author: laboan
"""

import numpy as np
import matplotlib.pyplot as plt
from load_validation_data import mass_flow_rate, efficiency_ts

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml

filename = "kofskey1972_1stage.yaml"
cascades_data = ml.get_cascades_data(filename)

omega_list = omega_list = [int(val) for val in list(mass_flow_rate.keys())]


if __name__ == '__main__':
    
    Case = 2
        
    performance_map = ml.PerformanceMap()
    
    if Case == 0:
        # Calculate angular speed line
        ig = {'R' : 0.2, 'eta_ts' : 0.6, 'eta_tt' : 1, "Ma_crit" : 1}
        cascades_data["BC"]["omega"] *= omega_list[4]/100
        figs = performance_map.plot_omega_line(cascades_data, 'all', pr_limits = [1.6, 4.5], N = 30, method = 'hybr', ig = ig)
        
    elif Case == 1:
        # Calculate angular speed line for a range of speeda 
        ig = {'R' : 0.3, 'eta_ts' : [0.8, 0.8, 0.8, 0.8, 0.8], 'eta_tt' : [1, 1, 1, 1, 1], 'Ma_crit' : 1} # Benner
        omega_list =  omega_list[1:]
        # ig = {'R' : 0.3, 'eta' : [0.8, 0.8, 0.6], 'Ma_crit' : 0.92} # Benner
        omega_input = np.array(omega_list)/100*cascades_data["BC"]["omega"]
        figs = performance_map.plot_perfomance_map(cascades_data,'all', omega_input, pr_limits = [1.6, 4.5], N = 20, method = 'hybr', ig = ig)
        
        compare = False
        
        if compare == True:
            fig_m = figs["m"]
            fig1 = fig_m[0]
            ax1 = fig_m[1]   
            
            p_st = 101353
            T_st = 288.15
            delta = cascades_data["BC"]["p0_in"]/p_st
            theta = (cascades_data["BC"]["T0_in"]/T_st)**2
            
            for i in range(len(omega_list)):
                PR = mass_flow_rate[str(omega_list[i])]["PR"]
                m = mass_flow_rate[str(omega_list[i])]["m"]*delta/np.sqrt(theta)
                
                ax1.scatter(PR, m, label = str(omega_list[i]))
            
            ax1.set_xlabel('Total-to-static pressure ratio')
            ax1.set_ylabel('Mass flow rate') 
            labels = [str(value) for value in omega_list]*2
            ax1.legend(labels, ncols = 2, title = 'Model   Measured', bbox_to_anchor = (1,0.8))    
            plt.close(fig1)
            
            fig2 = figs["eta_ts"][0]
            ax2 = figs["eta_ts"][1]
    
            for i in range(len(omega_list)):
                PR = efficiency_ts[str(omega_list[i])]["PR"]
                eta_ts = efficiency_ts[str(omega_list[i])]["eta_ts"]
                ax2.scatter(PR, eta_ts, label = str(omega_list[i]))
                
            ax2.legend(labels, ncols = 2, title = 'Model   Measured', bbox_to_anchor = (1,0.8))    
            plt.close(fig2)
            
    elif Case == 2:
    # Calculate pressure ratio line
        omega_limits = np.array([1,0.3])*cascades_data["BC"]["omega"]
        cascades_data["BC"]["p_out"] = cascades_data["BC"]["p0_in"]/3.
        ig = {'R' : 0.6, 'eta_ts' : 0.8, 'eta_tt' : 0.9, "Ma_crit" : 0.9}
        figs = performance_map.plot_pr_line(cascades_data, 'all', omega_limits = omega_limits, N = 20, method = 'hybr', ig = ig)
    
