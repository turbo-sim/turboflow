# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:42:49 2023

@author: laboan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp as CP

data = pd.read_excel('Data_Kofskey1972_1stage.xlsx')

data_sorted = data.sort_values(by = 'PR')
omega = data["omega"].unique()


# Get Mass flow rate
mass_flow_rate = {}
for i in range(len(omega)):
    data_omega = data_sorted[data_sorted["omega"] == omega[i]]
    data_plot = data_omega[data_omega["m"] > 0]
    PR = data_plot["PR"]
    m = data_plot["m"]
    mass_flow_rate[str(omega[i])] = {'PR' : PR,
                                  'm' : m}
    
# Get Torque
torque = {}
for i in range(len(omega)):
    data_omega = data_sorted[data_sorted["omega"] == omega[i]]
    data_plot = data_omega[data_omega["Torque"] > 0]
    PR = data_plot["PR"]
    tau = data_plot["Torque"]
    torque[str(omega[i])] = {'PR' : PR,
                                  'tau' : tau}
    
# Get efficiency (Calculated from angular speed, mass flow rate, torque and isentropic specific work)
omega_real = 1627
fluid_name = 'air'
fluid = CP.AbstractState('HEOS', fluid_name)
p0_in = 13.8e4            # Inlet total pressure
T0_in = 295.6             # Inlet total temperature
theta = (T0_in/288.15)**2
fluid.update(CP.PT_INPUTS, p0_in, T0_in)
h0_in = fluid.hmass()
s_in = fluid.smass()

efficiency_ts = {}
interp_values = {}
for i in range(len(omega)):
    data_omega = data_sorted[data_sorted["omega"] == omega[i]]
    
    m = data_omega[data_omega["m"]>0]["m"]
    tau = data_omega[data_omega["Torque"]>0]["Torque"]
    PR_m = data_omega[data_omega["m"]>0]["PR"]
    PR_tau = data_omega[data_omega["Torque"]>0]["PR"]

    tau_nan = np.interp(PR_m, PR_tau, tau)
    m_nan = np.interp(PR_tau, PR_m, m)
    index_m = data_omega.loc[data_omega["Torque"]>0]["m"].index
    index_tau = data_omega.loc[data_omega["m"]>0]["Torque"].index
    
    data_omega_interp = data_omega.copy()
    data_omega_interp.loc[index_tau, "Torque"] = tau_nan
    data_omega_interp.loc[index_m, "m"] = m_nan
    
    PR = data_omega_interp["PR"]    
    m = data_omega_interp["m"]
    tau = data_omega_interp["Torque"]

    h_out_s = np.zeros(len(PR))
    for j in range(len(PR)):
        fluid.update(CP.PSmass_INPUTS, p0_in/PR.values[j], s_in)
        h_out_s[j] = fluid.hmass()
    
    W = tau*omega[i]*omega_real/100
    W_is = m*(h0_in-h_out_s)*np.sqrt(theta)
    eta_ts = W/W_is*100
    
    efficiency_ts[str(omega[i])] = {'PR' : PR,
                                    'eta_ts' : eta_ts}
    
    interp_values[str(omega[i])] = {'PR' : PR,
                                    'tau' : tau,
                                    'm' : m}
        
if __name__ == '__main__':
    
    # Plot mass flow rate
    fig1, ax1 = plt.subplots()
    for i in range(len(omega)):
        PR = mass_flow_rate[str(omega[i])]["PR"]
        m = mass_flow_rate[str(omega[i])]["m"]
        
        ax1.scatter(PR, m, label = str(omega[i]))
    
    ax1.set_xlabel('Total-to-static pressure ratio')
    ax1.set_ylabel('Mass flow rate') 
    ax1.legend()
    
    
    # Plot Torque
    fig2, ax2 = plt.subplots()
    for i in range(len(omega)):
        PR = torque[str(omega[i])]["PR"]
        tau = torque[str(omega[i])]["tau"]
        
        ax2.scatter(PR, tau, label = str(omega[i]))
    
    ax2.set_xlabel('Total-to-static pressure ratio')
    ax2.set_ylabel('Torque') 
    ax2.legend()
    
    
    # Plot efficiency
    fig3, ax3 = plt.subplots()
    for i in range(len(omega)):
        PR = efficiency_ts[str(omega[i])]["PR"]
        eta_ts = efficiency_ts[str(omega[i])]["eta_ts"]
        
        ax3.scatter(PR, eta_ts, label = str(omega[i]))
    
    ax3.set_xlabel('Total-to-static pressure ratio')
    ax3.set_ylabel('Total to static efficiency')
    ax3.legend()
    
    # Plot interpolated data with real data
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega)))
    for i in range(len(omega)):
        PR = mass_flow_rate[str(omega[i])]["PR"]
        m = mass_flow_rate[str(omega[i])]["m"]
        PR_interp = interp_values[str(omega[i])]["PR"]
        m_interp = interp_values[str(omega[i])]["m"]
        
        ax4.scatter(PR, m, color = colors[i], label = str(int(omega[i])))
        ax4.scatter(PR_interp, m_interp, color = colors[i], marker = 'x', label = str(int(omega[i])))
        
        PR = torque[str(omega[i])]["PR"]
        tau = torque[str(omega[i])]["tau"]
        tau_interp = interp_values[str(omega[i])]["tau"]
        
        ax5.scatter(PR, tau, color = colors[i], label = str(int(omega[i])))
        ax5.scatter(PR_interp, tau_interp, color = colors[i], marker = 'x', label = str(int(omega[i])))
        
    ax4.set_xlabel('Total-to-static pressure ratio')
    ax4.set_ylabel('Mass flow rate') 
    ax4.set_title('Measured and interpolated values')
    ax4.legend(ncols = 2, title = "Measured  Interpolated")
        
    ax5.set_xlabel('Total-to-static pressure ratio')
    ax5.set_ylabel('Torque') 
    ax5.set_title('Measured and interpolated values')
    ax5.legend(ncols = 2, title = "Measured  Interpolated")

    # Close figures
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)

        
        
        
    
    