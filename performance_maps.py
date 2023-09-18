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
    N = 20
    PR = np.linspace(2,5,N)
    R = np.zeros(N)
    
    # PR = [3.1578947368421053]
    # PR = [3.15]
    # N = len(PR)
    
    Ma2 = np.zeros(N)
    Ma3 = np.zeros(N)
    Ma5 = np.zeros(N)
    Ma6 = np.zeros(N)
    
    m = np.zeros(N)
    eta = np.zeros(N)
    
    sol_vec = np.zeros((N, 17))
    err_vec = np.zeros((N, 17))
    x0_vec  = np.zeros((N, 17))
    # 
    x0 = None
    
    # sol = None
    use_previous = False
    
    for i in range(N):
        data_structure["BC"]["p_out"] = data_structure["BC"]["p0_in"]/PR[i]
        starttime = time.time()
        
        # # Solve the set of equations for the currrent cascade
        # if use_previous and i > 0:
        #     x0 = sol.x
        # else:
        x0 = CS.generate_initial_guess(data_structure, R = 0.4)
            
        sol, conv_hist = CS.cascade_series_analysis(data_structure, x0)
        endtime = time.time()-starttime
        print("\n")
        print(f"Pressure ratio: {PR[i]}")
        print(f"Number of evaluations: {sol.nfev}")
        print(f"max residual: {np.max(abs(sol.fun))}")
        print(f"Time: {endtime}")
        # x0 = sol.x
        # R[i] = data_structure["overall"]["R"]
        m[i] = data_structure["overall"]["m"]
        Ma2[i] = data_structure["plane"]["Marel"].values[1]
        Ma3[i] = data_structure["plane"]["Marel"].values[2]
        Ma5[i] = data_structure["plane"]["Marel"].values[4]
        Ma6[i] = data_structure["plane"]["Marel"].values[5]
        eta[i] = data_structure["overall"]["eta_ts"]
        
        sol_vec[i] = abs(x0)
        x0_vec[i]  = abs(data_structure["x0"])
        err_vec[i] = x0-data_structure["x0"]
    
    keys = ["v1", "v2", "v3", "s2", "s3", "a3", "w5", "w6", "s5", "s6", "b6",
            "v1*", "v2*", "s2*", "w4*", "w5*", "s5*"]
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, x0_vec[:, 0:6], label = keys[0:6])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('x0')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('x0')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, x0_vec[:, 6:11], label = keys[6:11])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('x0')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('x0')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, x0_vec[:, 11:], label = keys[11:])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('x0')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('x0')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, sol_vec[:, 0:6], label = keys[0:6])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('Solution')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('sol.x')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, sol_vec[:, 6:11], label = keys[6:11])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('Solution')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('sol.x')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, sol_vec[:, 11:], label = keys[11:])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('Solution')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('sol.x')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, err_vec[:, 0:6], label = keys[0:6])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('Deviation between x0 and solution')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('sol.x-x0')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, err_vec[:, 6:11], label = keys[6:11])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('Deviation between x0 and solution')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('sol.x-x0')
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(PR, err_vec[:, 11:], label = keys[11:])
    # ax1.legend(bbox_to_anchor = (1,1))
    # ax1.set_title('Deviation between x0 and solution')
    # ax1.set_xlabel('Pressure ratio')
    # ax1.set_ylabel('sol.x-x0')
    
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("PR")
    ax1.set_ylabel("mass flow")
    ax1.plot(PR, m)
        
    # fig1, ax1 = plt.subplots()
    # ax1.set_xlabel("PR")
    # ax1.set_ylabel("reaction")
    # ax1.plot(PR, R)
    
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
    
    plt.show()
    
    return

if __name__ == '__main__':
    
    m_vs_PR(data_structure)