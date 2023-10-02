# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:54:48 2023

@author: laboan
"""

import numpy as np
import matplotlib.pyplot as plt
# from load_validation_data import mass_flow_rate, efficiency_ts

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml
from load_validation_data import mass_flow_rate, efficiency_ts

filename = "kofskey1974_1stage.yaml"
cascades_data = ml.get_cascades_data(filename)

omega_list = omega_list = [int(val) for val in list(mass_flow_rate.keys())]
omega_list = omega_list[0:4]


if __name__ == '__main__':
    
    maps = ["m", "eta_ts"]
    # ig = {'R' : [0.1, 0.1, 0.4, 0.2, 0.2, 0.2], 'Ma_crit' : 0.9, 'eta' : [0.8, 0.8, 0.9, 0.8, 0.9, 0.9]} # Old omega
    # ig = {'R' : [0.1, 0.1, 0.1, 0, 0.4], 'Ma_crit' : 0.9, 'eta' : [0.8, 0.8, 0.8, 0.7, 0.7]}
    ig = {'R' : [0.5, 0.5, 0.4, 0.4], 'Ma_crit' : 0.9, 'eta' : [0.8, 0.8, 0.8, 0.6]}
    # ig = {'R' : 0.1, 'Ma_crit' : 0.9, 'eta' : 0.6}
    
    performance_map = ml.PerformanceMap()
    # cascades_data["BC"]["omega"] *= omega_list[4]/100
    # figs = performance_map.plot_omega_line(cascades_data, maps, pr_limits = [1.5, 3.5], N = 20, method = 'hybr', ig = ig)
    omega_input = np.array(omega_list)/100*cascades_data["BC"]["omega"]
    figs = performance_map.plot_perfomance_map(cascades_data, maps, omega_input, pr_limits = [1.5, 3.5], N = 20, method = 'hybr', ig = ig)
    
    fig_m = figs["m"]
    fig1 = fig_m[0]
    ax1 = fig_m[1]   
    
    p_st = 101353
    T_st = 288.15
    delta = cascades_data["BC"]["p0_in"]/p_st
    theta = (cascades_data["BC"]["T0_in"]/T_st)**2
    
    for i in range(len(omega_list)):
        PR = mass_flow_rate[str(omega_list[i])]["PRts"]
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
        PR = efficiency_ts[str(omega_list[i])]["PRts"]
        eta_ts = efficiency_ts[str(omega_list[i])]["eta_ts"]
        ax2.scatter(PR, eta_ts, label = str(omega_list[i]))
        
    ax2.legend(labels, ncols = 2, title = 'Model   Measured', bbox_to_anchor = (1,0.8))    
    plt.close(fig2)
    
    # fig.savefig("Mass_flow_rate.png", bbox_inches = 'tight')