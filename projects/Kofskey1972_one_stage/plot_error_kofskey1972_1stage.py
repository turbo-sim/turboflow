# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:10:28 2023

@author: laboan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load experimental data
filename = 'Full_Dataset_Kofskey1972_1stage.xlsx'
validation_data = pd.read_excel(filename, sheet_name = ['Mass flow rate', 'Torque', 'Total-to-static efficiency', 'Beta_out'])
m_exp = validation_data["Mass flow rate"]["m"] 
tau_exp = validation_data["Torque"]["Torque"]
eta_ts_exp = validation_data["Total-to-static efficiency"]["Efficiency_ts"]
alpha_exp = validation_data["Beta_out"]["Beta"]*np.pi/180

# Load simulation data
folder = 'Case 4/'
filenames = ['mass_flow_rate.xlsx', 'torque.xlsx', 'total-to-static_efficiency.xlsx', 'beta_out.xlsx']
model_data = pd.read_excel(folder+filenames[0], sheet_name = ['BC', 'plane', 'cascade', 'stage', 'overall'])
m_model = model_data["overall"]["m"]

model_data = pd.read_excel(folder + filenames[1], sheet_name = ['BC', 'plane', 'cascade', 'stage', 'overall'])
tau_model = model_data["overall"]["torque"]

model_data = pd.read_excel(folder + filenames[2], sheet_name = ['BC', 'plane', 'cascade', 'stage', 'overall'])
eta_ts_model = model_data["overall"]["eta_ts"]

model_data = pd.read_excel(folder + filenames[3], sheet_name = ['BC', 'plane', 'cascade', 'stage', 'overall'])
alpha_model = model_data["plane"]["alpha_6"]


# Mass flow rate
fig1, ax1 = plt.subplots()
ax1.plot(m_exp, m_model, 'o')
ax1.set_xlim([2.55,2.85])
xlims = ax1.get_xlim()
ax1.set_ylim(xlims)
N = 10
error = 0.05
ax1.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--', label=f'Error Range (±{int(error*100)}%)')
ax1.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax1.set_title("Mass flow rate")
ax1.set_xlabel('Experimental data')
ax1.set_ylabel('Model')
ax1.legend()

# Torque
fig2, ax2 = plt.subplots()
ax2.plot(tau_exp, tau_model, 'o')
xlims = ax2.get_xlim()
ax2.set_ylim(xlims)
N = 10
error = 0.05
ax2.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--', label=f'Error Range (±{int(error*100)}%)')
ax2.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax2.set_title("Torque")
ax2.set_xlabel('Experimental data')
ax2.set_ylabel('Model')
ax2.legend()

# Efficiency
fig3, ax3 = plt.subplots()
ax3.plot(eta_ts_exp, eta_ts_model, 'o')
xlims = ax3.get_xlim()
ax3.set_ylim(xlims)
N = 10
error = 0.05
ax3.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--', label=f'Error Range (±{int(error*100)}%)')
ax3.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax3.set_title("Total-to-static efficiency")
ax3.set_xlabel('Experimental data')
ax3.set_ylabel('Model')
ax3.legend()

# Efficiency
fig4, ax4 = plt.subplots()
ax4.plot(alpha_exp, alpha_model, 'o')
xlims = ax4.get_xlim()
ax4.set_ylim(xlims)
N = 10
error = 0.05
ax4.plot(np.linspace(xlims[0], xlims[1], N),(1-error)*np.linspace(xlims[0], xlims[1], N) , 'r--', label=f'Error Range (±{int(error*100)}(%)')
ax4.plot(np.linspace(xlims[0], xlims[1], N),(1+error)*np.linspace(xlims[0], xlims[1], N) , 'r--')
ax4.set_title("Exit relative flow angle")
ax4.set_xlabel('Experimental data')
ax4.set_ylabel('Model')
ax4.legend()

# Calculate error
error_m = (m_exp-m_model)/m_exp*100
error_tau = (tau_exp-tau_model)/tau_exp*100
error_eta_ts = (eta_ts_exp-eta_ts_model)/eta_ts_exp*100
error_alpha = (alpha_exp-alpha_model)/alpha_exp*100

# fig1.savefig(folder + "Error_mass_flow_rate.png", bbox_inches = 'tight')
# fig2.savefig(folder + "Error_torque.png", bbox_inches = 'tight')
# fig3.savefig(folder + "Error_total-to-static_efficiency.png", bbox_inches = 'tight')
# fig4.savefig(folder + "Error_alpha_exit.png", bbox_inches = 'tight')


