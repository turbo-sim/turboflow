# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:33:20 2023

@author: laboan
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import Cascade_series as CS
from main_cascade_series import data_structure

# Script with functions to generate performance maps

def m_vs_PR(data_structure):
    N = 50
    PR = np.linspace(1.5,5,N)
    
    Ma2         = np.zeros(N)
    Ma3         = np.zeros(N)
    Ma5         = np.zeros(N)
    Ma6         = np.zeros(N)
    m           = np.zeros(N)
    eta         = np.zeros(N)
    Y_p_stator  = np.zeros(N)
    Y_s_stator  = np.zeros(N)
    Y_cl_stator = np.zeros(N)
    Y_p_rotor   = np.zeros(N)
    Y_s_rotor   = np.zeros(N)
    Y_cl_rotor  = np.zeros(N)

        
    use_previous = True
        
    CS.number_stages(data_structure)
    
    for i in range(N):
        data_structure["BC"]["p_out"] = data_structure["BC"]["p0_in"]/PR[i]
        starttime = time.time()
        
        CS.update_fixed_params(data_structure)
        
        # Solve the set of equations for the currrent cascade
        if use_previous and i > 0:
            x0 = x_real
        else:
            x0 = CS.generate_initial_guess(data_structure, R = 0.4)
        
        x_scaled = CS.scale_x0(x0, data_structure)
        sol, conv_hist = CS.cascade_series_analysis(data_structure, x_scaled)
        x_real = CS.rescale_x0(sol.x, data_structure)
        endtime = time.time()-starttime
        print("\n")
        print(f"Pressure ratio: {PR[i]}")
        print(f"Number of evaluations: {sol.nfev}")
        print(f"max residual: {np.max(abs(sol.fun))}")
        print(f"Time: {endtime}")
        print("\n")
        m[i] = data_structure["overall"]["m"]
        Ma2[i] = data_structure["plane"]["Marel"].values[1]
        Ma3[i] = data_structure["plane"]["Marel"].values[2]
        Ma5[i] = data_structure["plane"]["Marel"].values[4]
        Ma6[i] = data_structure["plane"]["Marel"].values[5]
        eta[i] = data_structure["overall"]["eta_ts"]
        Y_p_stator[i]  = data_structure["cascade"]["eta_drop_p"].values[0]
        Y_s_stator[i]  = data_structure["cascade"]["eta_drop_s"].values[0]
        Y_cl_stator[i] = data_structure["cascade"]["eta_drop_cl"].values[0]
        Y_p_rotor[i]   = data_structure["cascade"]["eta_drop_p"].values[1]
        Y_s_rotor[i]   = data_structure["cascade"]["eta_drop_s"].values[1]
        Y_cl_rotor[i]  = data_structure["cascade"]["eta_drop_cl"].values[1]
    
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("PR")
    ax1.set_ylabel("mass flow")
    ax1.plot(PR, m)
    
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("PR")
    ax1.set_ylabel("efficiency")
    ax1.plot(PR, eta)
    
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("PR")
    ax1.set_ylabel("mach")
    ax1.plot(PR, Ma2, label="Ma2")
    ax1.plot(PR, Ma3, label="Ma3")
    ax1.plot(PR, Ma5, label="Ma5")
    ax1.plot(PR, Ma6, label="Ma6")
    ax1.legend()
    
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("PR")
    ax1.set_ylabel(r"$\eta_{drop}$")
    labels = ["$p_{stator}$", "$s_{stator}$", "$cl_{stator}$", "$p_{rotor}$", "$s_{rotor}$", "$cl_{rotor}$"]
    ax1.stackplot(PR, Y_p_stator, Y_s_stator, Y_cl_stator, Y_p_rotor, Y_s_rotor, Y_cl_rotor, labels = labels)
    ax1.legend(loc = "upper right", bbox_to_anchor = (1.2,1.2))
    
    plt.show()
    
    return

if __name__ == '__main__':
    
    m_vs_PR(data_structure)