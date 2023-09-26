# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:42:49 2023

@author: laboan
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp as CP

data = pd.read_excel('Data_Kofskey1972_1stage.xlsx')

data_sorted = data.sort_values(by = 'PR')
data_sorted_110 = data_sorted[data_sorted["omega"] == 110]
data_sorted_110 = data_sorted_110.round({'PR' : 2})

omega = data["omega"].unique()
omega_real = 1627
fluid_name = 'air'
fluid = CP.AbstractState('HEOS', fluid_name)
p0_in = 13.8e4            # Inlet total pressure
T0_in = 295.6             # Inlet total temperature
theta = (T0_in/288.15)**2
fluid.update(CP.PT_INPUTS, p0_in, T0_in)
h0_in = fluid.hmass()
s_in = fluid.smass()

# Get Mass flow rate
for i in range(len(omega)):
    data_omega = data_sorted[data_sorted["omega"] == omega[i]]
    data_plot = data_omega[data_omega["m"] > 0]
    PR = data_plot["PR"]
    m = data_plot["m"]

# Get Torque
for i in range(len(omega)):
    data_omega = data_sorted[data_sorted["omega"] == omega[i]]
    data_plot = data_omega[data_omega["Torque"] > 0]
    PR = data_plot["PR"]
    tau = data_plot["Torque"]
    
# Get efficiency
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
        fluid.update(CP.PSmass_INPUTS, p0_in/PR[j], s_in)
        h_out_s[i] = fluid.hmass()
    
    W = tau*omega[i]*omega_real/100
    W_is = m*(h0_in-h_out_s)*np.sqrt(theta)
    eta = W/W_is*100
        