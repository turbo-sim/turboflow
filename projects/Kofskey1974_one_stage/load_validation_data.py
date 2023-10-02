# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:06:45 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp as CP

import sys
import os

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml
ml.set_plot_options()

data = pd.read_excel('Data_Kofskey_1974.xlsx')

data_sorted = data.sort_values(by = 'PRtt')

omega = data["omega"].unique()

# Interpolate Total-to-static PR from PRtt vs PRts plot
data_interp = data_sorted.copy()
for i in range(len(omega)):
    data_omega = data_interp[data_interp["omega"] == omega[i]]    
    data_PR_conversion = data_omega[data_omega["PRts"]>0]  
    PRtt_measured = data_PR_conversion["PRtt"]
    PRts_measured = data_PR_conversion["PRts"]
    
    PRtt = data_omega[data_omega["PRts"].isna()]["PRtt"]
    PRts = np.interp(PRtt, PRtt_measured, PRts_measured)
    
    index_PRts = data_omega.loc[data_omega["PRts"].isna()].index
    data_interp.loc[index_PRts, "PRts"] = PRts
    
# Remove part of datafram with only PRtt and PRts
data_interp = data_interp.loc[data_sorted.loc[data_sorted["PRts"].isna()].index]

# Get Mass flow rate
mass_flow_rate = {}
for i in range(len(omega)):
    data_omega = data_interp[data_interp["omega"] == omega[i]]
    data_plot = data_omega[data_omega["m"] > 0]
    PRtt = data_plot["PRtt"]
    PRts = data_plot["PRts"]
    m = data_plot["m"]
    mass_flow_rate[str(omega[i])] = {'PRtt' : PRtt,
                                      'PRts' : PRts,
                                  'm' : m}
    
# Get Torque
torque = {}
for i in range(len(omega)):
    data_omega = data_interp[data_interp["omega"] == omega[i]]
    data_plot = data_omega[data_omega["Torque"] > 0]
    PRtt = data_plot["PRtt"]
    PRts = data_plot["PRts"]
    tau = data_plot["Torque"]
    torque[str(omega[i])] = {'PRtt' : PRtt,
                              'PRts' : PRts,
                                  'tau' : tau}
    
# Get efficiency (Calculated from angular speed, mass flow rate, torque and isentropic specific work)
omega_real = 1963
fluid_name = 'air'
fluid = CP.AbstractState('HEOS', fluid_name)
p0_in = 10.82e4            # Inlet total pressure
T0_in = 310.0              # Inlet total temperature
theta = (T0_in/288.15)**2
fluid.update(CP.PT_INPUTS, p0_in, T0_in)
h0_in = fluid.hmass()
s_in = fluid.smass()

efficiency_ts = {}
interp_values = {}
for i in range(len(omega)):
    data_omega = data_interp[data_interp["omega"] == omega[i]]
        
    m = data_omega[data_omega["m"]>0]["m"]
    tau = data_omega[data_omega["Torque"]>0]["Torque"]
    PRtt_m = data_omega[data_omega["m"]>0]["PRtt"]
    PRtt_tau = data_omega[data_omega["Torque"]>0]["PRtt"]

    tau_nan = np.interp(PRtt_m, PRtt_tau, tau)
    m_nan = np.interp(PRtt_tau, PRtt_m, m)
    index_m = data_omega.loc[data_omega["Torque"]>0]["m"].index
    index_tau = data_omega.loc[data_omega["m"]>0]["Torque"].index
    
    data_omega_interp = data_omega.copy()
    data_omega_interp.loc[index_tau, "Torque"] = tau_nan
    data_omega_interp.loc[index_m, "m"] = m_nan
    
    
    PRtt = data_omega_interp["PRtt"] 
    PRts = data_omega_interp["PRts"]    
    m = data_omega_interp["m"]
    tau = data_omega_interp["Torque"]

    h_out_s = np.zeros(len(PRtt))
    for j in range(len(PRtt)):
        fluid.update(CP.PSmass_INPUTS, p0_in/PRts.values[j], s_in)
        h_out_s[j] = fluid.hmass()
    
    W = tau*omega[i]*omega_real/100
    W_is = m*(h0_in-h_out_s)*np.sqrt(theta)
    eta_ts = W/W_is*100
    
    efficiency_ts[str(omega[i])] = {'PRtt' : PRtt,
                                    'PRts' : PRts,
                                    'eta_ts' : eta_ts}
    
    interp_values[str(omega[i])] = {'PRtt' : PRtt,
                                    'PRts'  :PRts,
                                    'tau' : tau,
                                    'm' : m}
        
if __name__ == '__main__':
    
    # Plot interpolation of P static points
    fig1, ax1 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega)))
    for i in range(len(omega)):
        data_omega = data_sorted[data_sorted["omega"] == omega[i]]
        data_omega_interp = data_interp[data_interp["omega"] == omega[i]]
        ax1.scatter(data_omega_interp["PRts"], data_omega_interp["PRtt"], color = colors[i], marker = 'x', label = str(int(omega[i])))
        ax1.scatter(data_omega[data_omega["PRts"]>0]["PRts"], data_omega[data_omega["PRts"]>0]["PRtt"], color = colors[i], label = str(int(omega[i])))
    ax1.legend(ncols = 2)
    ax1.set_title("Interpolated and measured relaation between TT and TS pressure ratio")
    
    
    # Plot mass flow rate
    fig2, ax2 = plt.subplots()
    for i in range(len(omega)):
        PR = mass_flow_rate[str(omega[i])]["PRts"]
        m = mass_flow_rate[str(omega[i])]["m"]
        
        ax2.scatter(PR, m, label = str(omega[i]))
    
    ax2.set_xlabel('Total-to-static pressure ratio')
    ax2.set_ylabel('Mass flow rate') 
    ax2.legend()
    
    
    # Plot Torque
    fig3, ax3 = plt.subplots()
    for i in range(len(omega)):
        PR = torque[str(omega[i])]["PRts"]
        tau = torque[str(omega[i])]["tau"]
        
        ax3.scatter(PR, tau, label = str(omega[i]))
    
    ax3.set_xlabel('Total-to-static pressure ratio')
    ax3.set_ylabel('Torque') 
    ax3.legend()
    
    
    # Plot efficiency
    fig4, ax4 = plt.subplots()
    for i in range(len(omega)):
        PR = efficiency_ts[str(omega[i])]["PRts"]
        eta_ts = efficiency_ts[str(omega[i])]["eta_ts"]
        
        ax4.scatter(PR, eta_ts, label = str(omega[i]))
    
    ax4.set_xlabel('Total-to-static pressure ratio')
    ax4.set_ylabel('Total to static efficiency')
    ax4.legend()
    
    # Plot interpolated data with real data
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega)))
    for i in range(len(omega)):
        PR = mass_flow_rate[str(omega[i])]["PRts"]
        m = mass_flow_rate[str(omega[i])]["m"]
        PR_interp = interp_values[str(omega[i])]["PRts"]
        m_interp = interp_values[str(omega[i])]["m"]
        
        ax5.scatter(PR, m, color = colors[i], label = str(int(omega[i])))
        ax5.scatter(PR_interp, m_interp, color = colors[i], marker = 'x', label = str(int(omega[i])))
        
        PR = torque[str(omega[i])]["PRts"]
        tau = torque[str(omega[i])]["tau"]
        tau_interp = interp_values[str(omega[i])]["tau"]
        
        ax6.scatter(PR, tau, color = colors[i], label = str(int(omega[i])))
        ax6.scatter(PR_interp, tau_interp, color = colors[i], marker = 'x', label = str(int(omega[i])))
        
    ax5.set_xlabel('Total-to-static pressure ratio')
    ax5.set_ylabel('Mass flow rate') 
    ax5.set_title('Measured and interpolated values')
    ax5.legend(ncols = 2, title = "Measured  Interpolated")
        
    ax6.set_xlabel('Total-to-static pressure ratio')
    ax6.set_ylabel('Torque') 
    ax6.set_title('Measured and interpolated values')
    ax6.legend(ncols = 2, title = "Measured  Interpolated")

    # Close figures
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
