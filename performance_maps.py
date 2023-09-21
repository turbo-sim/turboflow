# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:33:20 2023

@author: laboan
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import Cascade_series as CS
from examples import data_structure
from load_validation_data import *

# Script with functions to generate performance maps

def m_vs_PR(data_structure):
    # Define pressure ratio
    N = 40 # Number of points
    PR_min = 1.5 # Mnimum pressure ratio
    PR_max = 4 # Maximum pressure ratio
    PR = np.linspace(1.5,4,N) 
    
    # Define cases of percentages of design speed e.g. [0.7,0.9,1,1.1]
    omega_org = data_structure["overall"]["omega"]
    omega = np.array([0.7,0.9,1,1.1])
    
    # Guess of degree of reaction (to get validation plots going from PR = 1.5, one value for each omega)
    R_guess = np.array([0.2,0.4,0.4,0.4]) #Kacker and Okapuu
    # R_guess = np.array([0.3,0.45,0.45,0.45]) #Benner

    # Define array used for plotting
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

    # If true, uses solution of previous point as initial guess for current point
    use_previous = True
    x_real = None
        
    # Checks and calculates number of stages
    CS.number_stages(data_structure)
    
    # Define figures for efficieny and mass flow rate (one curve for each omega)
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Total-to-static pressure ratio")
    ax1.set_ylabel("Mass flow rate")
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("Total-to-static pressure ratio")
    ax2.set_ylabel("efficiency")
        
    # Calculate mass flow rate vs pressure ratio curve for each omega
    for j in range(len(omega)):
        data_structure["overall"]["omega"] = omega[j]*omega_org
    
        # Calculate mass flow rate vs pressure ratio for current omega 
        for i in range(N):
            data_structure["BC"]["p_out"] = data_structure["BC"]["p0_in"]/PR[i]
            print("\n")
            print(f"Pressure ratio: {PR[i]}")
            print(f"Percent of design speed: {omega[j]}")
            
            # Update fixed parameters for current case (h_out_s, v0 etc)
            CS.update_fixed_params(data_structure)
            
            # Initial guess
            if use_previous and i > 0 and PR[i]!= 3:
                x0 = x_real
            else:
                x0 = CS.generate_initial_guess(data_structure, R = R_guess[j])

            
            x_scaled = CS.scale_x0(x0, data_structure) # Scale initial guess
            sol, conv_hist = CS.cascade_series_analysis(data_structure, x_scaled) # Solve problem
            x_real = CS.rescale_x0(sol.x, data_structure) # Calculate real solution
            
            print(f"max residual: {np.max(abs(sol.fun))}")

            # Store varibles for plotting
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
    
        # Plot the mass flow rate and efficiency curve for current omega
        ax1.plot(PR, m, label = str(int(omega[j]*100)))
        ax2.plot(PR, eta, label = str(int(omega[j]*100)))
        
        # Plot Mach and Loss distribution for current omega (one plot for each omega)
        fig3, ax3 = plt.subplots()
        ax3.set_xlabel("PR")
        ax3.set_ylabel("mach")
        ax3.plot(PR, Ma2, label="Ma2")
        ax3.plot(PR, Ma3, label="Ma3")
        ax3.plot(PR, Ma5, label="Ma5")
        ax3.plot(PR, Ma6, label="Ma6")
        ax3.legend()
        ax3.set_title('Percent of design speed: ' +str(int(omega[j]*100)))
        

        fig4, ax4 = plt.subplots()
        ax4.set_xlabel("PR")
        ax4.set_ylabel(r"$\eta_{drop}$")
        labels = ["$p_{stator}$", "$s_{stator}$", "$cl_{stator}$", "$p_{rotor}$", "$s_{rotor}$", "$cl_{rotor}$"]
        ax4.stackplot(PR, Y_p_stator, Y_s_stator, Y_cl_stator, Y_p_rotor, Y_s_rotor, Y_cl_rotor, labels = labels)
        ax4.legend(loc = "upper right", bbox_to_anchor = (1.2,1.2))
        ax4.set_title('Percent of design speed: ' +str(int(omega[j]*100)))
    
    # Plot experimental data from Kofskey et al. (1972)
    ax1.plot(x4,y4,'D',label = '70')
    ax1.plot(x3,y3,'s',label = '90')
    ax1.plot(x2,y2,'v',label = '100')
    ax1.plot(x1,y1,'o',label = '110')

    ax1.legend(title = 'Percent of design speed', ncols = 2)
    ax2.legend()
    plt.show()
    
    return x_real

if __name__ == '__main__':
    
    x_real = m_vs_PR(data_structure)
